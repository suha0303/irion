"""Check bounded adaptation, resume and independent flight comparison."""

import hashlib
import json

import numpy as np
import pytest
from safetensors.torch import load_file

from capstone_ai.mapping import FEATURES
from localflight.adapt import fit
from localflight.analysis import compare
from localflight.catalog import DEFAULT_BUNDLE, digest
from localflight.prepared import REVISION, load, windows
from localflight.runtime import Runtime


def flight(root, name, role, health="confirmed_normal"):
    folder = root / name
    folder.mkdir()
    values = np.zeros((250, 19), dtype="f8")
    values[:, :4] = 0.4
    values[:, 14] = -9.81
    values[:, 17] = 16 - np.arange(250) * 0.0001
    values[:, 18] = 8
    np.savez(
        folder / "frames.npz",
        values=values,
        time_us=np.arange(250, dtype="i8") * 100000,
    )
    profile = dict(
        schema="drone.real-aircraft/1",
        id="fixture-quad",
        name="Synthetic fixture",
        autopilot="px4",
        firmware="fixture",
        rotor_count=4,
        series_cells=4,
        capacity_ah=12,
        battery_id="fixture-battery",
        motor_channels=dict(
            front_right=1, front_left=2, rear_right=3, rear_left=4
        ),
        output_min=0,
        output_max=1000,
        mapping_evidence="synthetic test fixture",
    )

    m = dict(
        schema="drone.real-common19/1",
        feature_order=FEATURES,
        preprocessing_revision=REVISION,
        units="physical_pack_V_A",
        origin="synthetic_unit_fixture",
        role=role,
        health=health,
        health_evidence="synthetic construction, not a real flight",
        profile=profile,
        mapping_verified=True,
        mapping_evidence="synthetic",
        source_sha256=hashlib.sha256(name.encode()).hexdigest(),
        file=dict(path="frames.npz", sha256=digest(folder / "frames.npz")),
    )
    path = folder / "manifest.json"
    path.write_text(json.dumps(m), encoding="utf-8")
    return path


def test_unknown_health_and_evaluation_cannot_fit(tmp_path):
    unknown = flight(tmp_path, "unknown", "adaptation", "unknown")
    with pytest.raises(ValueError, match="Confirmed normal"):
        load(unknown, normal=True)
    evaluation = flight(tmp_path, "evaluation", "evaluation")
    with pytest.raises(ValueError, match="Evaluation"):
        fit(
            "forecast_lstm_h64_l2",
            [evaluation],
            [],
            tmp_path / "bad",
            bundle=DEFAULT_BUNDLE,
        )


@pytest.mark.parametrize(
    "model_id",
    [
        "forecast_lstm_h64_l2",
        "run_health_lstm_h64_l2",
        "run_health_transformer_h64_l2",
    ],
)
def test_head_only_fit_restore_and_independent_comparison(tmp_path, model_id):
    train = flight(tmp_path, "train", "adaptation")
    dev = flight(tmp_path, "dev", "development")
    evaluation = flight(tmp_path, "test", "evaluation")
    base = Runtime(model_id).model.state_dict()
    fit(
        model_id,
        [train],
        [dev],
        tmp_path / "full",
        bundle=DEFAULT_BUNDLE,
        epochs=2,
    )
    fit(
        model_id,
        [train],
        [dev],
        tmp_path / "part",
        bundle=DEFAULT_BUNDLE,
        epochs=2,
        stop_after_epoch=1,
    )
    fit(
        model_id,
        [train],
        [dev],
        tmp_path / "resume",
        bundle=DEFAULT_BUNDLE,
        epochs=2,
        resume=tmp_path / "part",
    )
    full = load_file(str(tmp_path / "full/model.safetensors"))
    resumed = load_file(str(tmp_path / "resume/model.safetensors"))
    for key in full:
        np.testing.assert_array_equal(full[key].numpy(), resumed[key].numpy())
        if not key.startswith("head."):
            np.testing.assert_array_equal(full[key].numpy(), base[key].numpy())
    assert any(
        not np.array_equal(full[k].numpy(), base[k].numpy())
        for k in full
        if k.startswith("head.")
    )
    r = compare(tmp_path / "full", evaluation, bundle=DEFAULT_BUNDLE)
    assert r["independent"] and not r["detection_performance_verified"]
    assert r["adapted"]["health_verdict"] is None
    if model_id.startswith("forecast_"):
        assert len(r["adapted"]["metrics"]["rmse"]) == 4
    else:
        assert len(r["adapted"]["metrics"]["segments"]) == 1
        assert not r["adapted"]["metrics"]["whole_flight_verdict"]
    m = json.loads(evaluation.read_text(encoding="utf-8"))
    m["source_sha256"] = json.loads(train.read_text(encoding="utf-8"))[
        "source_sha256"
    ]
    evaluation.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="already used"):
        compare(tmp_path / "full", evaluation, bundle=DEFAULT_BUNDLE)


def test_windows_are_causal_and_scaler_is_frozen(tmp_path):
    path = flight(tmp_path, "input", "adaptation")
    m, v, t = load(path)
    scaler = Runtime("forecast_lstm_h64_l2").scaler
    before = {k: a.copy() for k, a in scaler.items()}
    x, y, _, origins = windows(m, v, t, scaler)
    v2 = v.copy()
    v2[t > origins[0], 17] += 1
    x2, y2, _, _ = windows(m, v2, t, scaler)
    np.testing.assert_array_equal(x[0], x2[0])
    assert not np.array_equal(y[0], y2[0])
    for key in scaler:
        np.testing.assert_array_equal(scaler[key], before[key])
