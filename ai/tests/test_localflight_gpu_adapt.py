"""Regression checks for partial adaptation, budgets and promotion evidence."""

import json

import numpy as np
import pytest
from safetensors.torch import load_file
from test_localflight_adapt import flight
from test_localflight_calibration import make_flight

from localflight.adapt import fit
from localflight.adapt_review import _event_detection, review
from localflight.adapt_support import parameter_names
from localflight.catalog import DEFAULT_BUNDLE, digest
from localflight.registry import register, select_model, validate
from localflight.runtime import Runtime
from localflight.tune import run, select_reviewed


@pytest.mark.parametrize(
    "model_id", ["forecast_transformer_h64_l4", "run_health_transformer_h64_l2"]
)
def test_partial_fit_changes_only_allowed_layers_and_cpu_restores(
    tmp_path, model_id
):
    train = flight(tmp_path, "train", "adaptation")
    dev = flight(tmp_path, "dev", "development")
    model = Runtime(model_id).model
    allowed = parameter_names(model, "staged_last_blocks")
    fitted = tmp_path / "fitted"
    result = fit(
        model_id,
        [train],
        [dev],
        fitted,
        bundle=DEFAULT_BUNDLE,
        epochs=3,
        strategy="staged_last_blocks",
    )
    last = load_file(str(fitted / "last.safetensors"))
    original = model.state_dict()
    changed = {
        k
        for k in last
        if not np.array_equal(last[k].numpy(), original[k].numpy())
    }
    assert changed <= allowed and any(
        not k.startswith("head.") for k in changed
    )
    assert result["history"][-1]["phase"] == "staged_last_blocks"
    assert (
        validate(fitted, DEFAULT_BUNDLE)["model_sha256"]
        == result["model_sha256"]
    )


def test_budget_without_epoch_never_becomes_selectable(tmp_path):
    train = flight(tmp_path, "train", "adaptation")
    fitted = tmp_path / "fitted"
    result = fit(
        "forecast_lstm_h64_l2",
        [train],
        [],
        fitted,
        bundle=DEFAULT_BUNDLE,
        time_limit_seconds=1e-9,
    )
    assert result["status"] == "budget_exhausted_without_checkpoint"
    with pytest.raises(ValueError, match="complete"):
        register(tmp_path / "registry", fitted, bundle=DEFAULT_BUNDLE)


def test_prediction_only_comparison_cannot_promote_new_candidate(tmp_path):
    from localflight.analysis import compare

    train = flight(tmp_path, "train", "adaptation")
    evaluation = flight(tmp_path, "test", "evaluation")
    fitted = tmp_path / "fitted"
    fit(
        "forecast_lstm_h64_l2",
        [train],
        [],
        fitted,
        bundle=DEFAULT_BUNDLE,
        epochs=1,
    )
    saved = register(tmp_path / "registry", fitted, bundle=DEFAULT_BUNDLE)
    comparison = compare(fitted, evaluation, bundle=DEFAULT_BUNDLE)
    with pytest.raises(ValueError, match="퇴행"):
        select_model(
            tmp_path / "registry",
            saved["id"],
            comparison,
            bundle=DEFAULT_BUNDLE,
        )
    assert not (tmp_path / "registry/model-selection.json").exists()


def test_event_gate_excludes_pre_fault_alarm_and_unrelated_signal():
    result = dict(
        model_id="forecast_fixture",
        scoring=dict(time_us=[10, 20, 30], scores=[10, 10, 10]),
        metrics=dict(
            series=[dict(residual=[[0, 0, 10, 0], [0, 0, 10, 0], [0, 0, 0, 3]])]
        ),
    )
    event = dict(start_us=15, end_us=25, target="pack_voltage")
    measured = _event_detection(result, {"threshold": 2}, event, 1)
    assert measured["covered"] and not measured["detected"]
    event["end_us"] = 35
    assert (
        _event_detection(result, {"threshold": 2}, event, 1)["first_alarm_us"]
        == 30
    )


def test_review_blocks_missing_labels_and_accepts_unchanged_detector(tmp_path):
    train = flight(tmp_path, "train", "adaptation")
    calibrations = [
        make_flight(tmp_path, f"cal{i}", "calibration") for i in range(2)
    ]
    normal = flight(tmp_path, "normal", "evaluation")
    abnormal = flight(tmp_path, "abnormal", "evaluation", "abnormal")
    meta = json.loads(abnormal.read_text())
    archive = abnormal.parent / "frames.npz"
    with np.load(archive) as z:
        values, times = z["values"].copy(), z["time_us"].copy()
    values[150:, 17] -= 8
    values[150:, 4] += 10
    np.savez(archive, values=values, time_us=times)
    meta["file"]["sha256"] = digest(archive)
    abnormal.write_text(json.dumps(meta))
    fitted = tmp_path / "fitted"
    fit(
        "forecast_lstm_h64_l2",
        [train],
        [],
        fitted,
        bundle=DEFAULT_BUNDLE,
        epochs=1,
        learning_rate=1e-12,
    )
    missing = review(
        fitted,
        calibrations,
        [normal],
        tmp_path / "missing",
        bundle=DEFAULT_BUNDLE,
    )
    assert missing["promotion_check"]["status"] == "pending"
    checked = review(
        fitted,
        calibrations,
        [normal, abnormal],
        tmp_path / "checked",
        bundle=DEFAULT_BUNDLE,
        events={
            meta["source_sha256"]: [dict(start_us=15000000, end_us=24900000)]
        },
    )
    assert checked["promotion_check"]["status"] == "passed"
    root = tmp_path / "registry"
    saved = register(root, fitted, bundle=DEFAULT_BUNDLE)
    selected = select_reviewed(
        root, saved["id"], checked, bundle=DEFAULT_BUNDLE
    )
    assert selected["calibration"]["enabled"]
    assert selected["model"]["selected"] == saved["id"]


def test_pair_resume_preserves_completed_model_across_multiple_resumes(
    tmp_path,
):
    train = flight(tmp_path, "train", "adaptation")
    dev = flight(tmp_path, "dev", "development")
    request = dict(
        models=["forecast_transformer_h64_l4", "run_health_transformer_h64_l2"],
        adaptation=[str(train)],
        development=[str(dev)],
        epochs=3,
        budget_seconds=60,
        output=str(tmp_path / "first"),
    )
    first = run(request, bundle=DEFAULT_BUNDLE)
    assert all(
        r["review_status"] == "requires_calibration_and_labelled_evaluation"
        for r in first["tuned_models"]
    )
    second = run(
        {
            **request,
            "resume": request["output"],
            "output": str(tmp_path / "second"),
        },
        bundle=DEFAULT_BUNDLE,
    )
    third = run(
        {
            **request,
            "resume": second["output"],
            "output": str(tmp_path / "third"),
        },
        bundle=DEFAULT_BUNDLE,
    )
    assert [r["adaptation"]["model_sha256"] for r in third["tuned_models"]] == [
        r["adaptation"]["model_sha256"] for r in first["tuned_models"]
    ]
    assert not third["automatically_activated"]
