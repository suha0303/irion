"""Finite all-model sensitivity gate: order, evidence, packing and isolation."""

import hashlib
import json
import shutil
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from execution import engine
from execution.common import read, sha, write
from learning import sensitivity


def arrays(run, count=3, *, block=0):
    return dict(
        X=np.zeros((count, 100, 19)),
        C=np.zeros((count, 6)),
        Y=np.ones((count, 10, 4)),
        run_id=np.array([run] * count),
        segment=np.zeros(count, dtype=np.int64),
        origin_us=np.arange(count, dtype=np.int64) * 1_000_000,
        source_indices=np.arange(count, dtype=np.int64),
        selection_index=np.full(count, block, dtype=np.int64),
        selection_evidence=[dict(run_id=run, selected_start=0, selected_stop=count)],
    )


@pytest.fixture
def gate(tmp_path, monkeypatch):
    root, out = tmp_path / "upstream", tmp_path / "compute"
    dataset = root / "dataset"
    names = ["first", "second", "third"]
    rows = [
        dict(run_id=role, role=role, paired_case_id=role)
        for role in ("normal_validation", "normal_calibration", "normal_dev_eval")
    ]
    write(dataset / "manifest.json", dict(runs=rows))
    np.savez(dataset / "scaler.npz", y_scale=np.ones(4))
    for name in names:
        folder = out / "models" / name
        write(folder / "integrity.json", {})
        shutil.copy2(dataset / "scaler.npz", folder / "scaler.npz")
    samples = {role: arrays(role) for role in ("scale", "calibration", "check")}
    plan = {
        "selections": {
            role: [dict(run_id=role, paired_case_id=role, slot=role)] for role in samples
        }
    }
    policy = {"batch_size": 256, "frozen": "same-all-models"}
    events, assessment_calls = [], []
    failure_models = set()
    fatal_models = set()

    def predictor(folder, device):
        name = folder.name

        def predict(x, c):
            events.append(("predict", name, len(x)))
            if name in fatal_models:
                raise RuntimeError("fixture forward failed")
            return np.zeros((len(x), 10, 4))

        return SimpleNamespace(
            name=name,
            metadata={"name": "lstm", "config": {"batch_size": 256}},
            scaler={"y_scale": np.ones(4)},
            predict_scaled=predict,
        )

    def normalized(dataset, role, scaler):
        events.append(("load", role))
        assert role in {"normal_validation", "normal_calibration"}
        return arrays(role)

    def select(values, selections):
        key = selections[0]["slot"]
        assert str(values["run_id"][0]) == (
            "normal_calibration" if key == "calibration" else "normal_validation"
        )
        return {
            k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in samples[key].items()
        }

    def fit(*args):
        events.append(("diagnostic_fit",))
        return dict(
            high=[1.0, 2.0],
            reference_error_scale_scaled=np.ones((10, 4)).tolist(),
            independent_false_alarm_target_verified=False,
        )

    def assess(reference, first, repeat, actual, identities, fitted, physical, supplied):
        name = next(event[1] for event in reversed(events) if event[0] == "predict")
        events.append(("assessment", name))
        assessment_calls.append((name, supplied.copy(), fitted.copy()))
        assert fitted["purpose"] == "diagnostic_only_never_reuse_for_full_calibration"
        assert physical.tolist() == [1.0, 1.0, 1.0, 4.0]
        return (
            dict(status="FAILED_STOPPED" if name in failure_models else "PASSED"),
            dict(reference=reference, batch=first, repeated=repeat, actual=actual),
        )

    monkeypatch.setattr(sensitivity, "validate_policy", lambda p: p)
    monkeypatch.setattr(sensitivity, "validate_plan", lambda p, path: p)
    monkeypatch.setattr(sensitivity, "Predictor", predictor)
    monkeypatch.setattr(sensitivity, "normalized_role", normalized)
    monkeypatch.setattr(sensitivity, "select_arrays", select)
    monkeypatch.setattr(sensitivity, "fit_calibration", fit)
    monkeypatch.setattr(sensitivity, "assess_sensitivity", assess)
    return SimpleNamespace(
        root=root,
        out=out,
        names=names,
        plan=plan,
        policy=policy,
        events=events,
        samples=samples,
        assessment_calls=assessment_calls,
        failure_models=failure_models,
        fatal_models=fatal_models,
        predictor=predictor,
        normalized=normalized,
        progress=SimpleNamespace(check=lambda: None, update=lambda **kw: None),
        backend=lambda device: nullcontext(),
    )


def run(gate, **changes):
    return sensitivity.run_sensitivity_gate(
        gate.root,
        gate.out,
        gate.progress,
        plan=gate.plan,
        policy=gate.policy,
        model_names=changes.get("model_names", gate.names),
        predict_device="cpu",
        backend_factory=gate.backend,
    )


def test_common_inputs_policy_and_separate_diagnostic_fit(gate):
    summary = run(gate)
    assert summary["status"] == "PASSED"
    assert summary["diagnostic_calibration_reused_for_full"] is False
    assert summary["full_calibration_started"] is False
    assert not (gate.out / "calibration").exists()
    assert [x for x in gate.events if x[0] == "load"] == [
        ("load", "normal_validation"),
        ("load", "normal_calibration"),
    ]
    assert sum(x[0] == "diagnostic_fit" for x in gate.events) == len(gate.names)
    assert [x[1] for x in gate.assessment_calls] == [gate.policy] * len(gate.names)
    signatures = []
    for name in gate.names:
        folder = gate.out / "sensitivity" / name
        report = read(folder / "report.json")
        assert report["evidence"]["sha256"] == sha(folder / "evidence.npz")
        assert report["diagnostic_calibration_file_sha256"] == sha(
            folder / "diagnostic-calibration.json"
        )
        assert report["batch_packing"]["batch"] == report["batch_packing"]["repeat"] == [3]
        signatures.append(report["sample_signatures"])
    assert signatures == [summary["sample_signatures"]] * len(gate.names)


def test_actual_assessor_keeps_semantic_and_file_calibration_hashes(gate, monkeypatch):
    from capstone_ai.detection_protocol import fit_calibration
    from learning.sensitivity_metrics import assess_sensitivity

    gate.policy = dict(
        schema="drone-ai.sensitivity-policy/1",
        batch_size=256,
        require_same_batch_exact=True,
        max_scaled_abs=1e-4,
        physical_max=[1e-5, 1e-5, 1e-5, 1e-4],
        max_score_delta_over_high=0.01,
        max_ewma_delta_over_high=0.01,
        max_changed_alarm_states=0,
    )
    monkeypatch.setattr(sensitivity, "assess_sensitivity", assess_sensitivity)
    monkeypatch.setattr(sensitivity, "fit_calibration", fit_calibration)
    run(gate)
    for name in gate.names:
        folder = gate.out / "sensitivity" / name
        report = read(folder / "report.json")
        fitted = read(folder / "diagnostic-calibration.json")
        canonical = hashlib.sha256(
            json.dumps(fitted, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        assert report["diagnostic_calibration_sha256"] == canonical
        assert report["diagnostic_calibration_file_sha256"] == sha(
            folder / "diagnostic-calibration.json"
        )


def test_numerical_failure_checks_remaining_models_then_stops(gate):
    gate.failure_models.update(gate.names[:2])
    with pytest.raises(ValueError, match="failed before full calibration"):
        run(gate)
    summary = read(gate.out / "sensitivity/summary.json")
    assert summary["status"] == "FAILED_STOPPED"
    assert summary["failures"] == gate.names[:2]
    assert summary["models_completed"] == 3
    assert [x[1] for x in gate.events if x[0] == "assessment"] == gate.names
    assert not (gate.out / "calibration").exists()


def test_fatal_runtime_failure_preserves_prior_evidence_and_does_not_continue(gate):
    gate.fatal_models.add("second")
    with pytest.raises(RuntimeError, match="fixture forward"):
        run(gate)
    summary = read(gate.out / "sensitivity/summary.json")
    assert summary["status"] == "FAILED_STOPPED"
    assert summary["models_completed"] == 1
    assert "RuntimeError" in summary["error"]
    assert read(gate.out / "sensitivity/first/report.json")["status"] == "PASSED"
    assert not (gate.out / "sensitivity/third").exists()
    assert not (gate.out / "calibration").exists()


def test_existing_evidence_never_overwritten(gate):
    write(gate.out / "sensitivity/summary.json", {"status": "previous failure"})
    previous = (gate.out / "sensitivity/summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="no automatic retry"):
        run(gate)
    assert (gate.out / "sensitivity/summary.json").read_bytes() == previous
    assert gate.events == []


def test_changed_scaler_stops_before_predictions(gate):
    np.savez(gate.out / "models/second/scaler.npz", y_scale=np.ones(4) * 2)
    with pytest.raises(ValueError, match="same train-only scaler"):
        run(gate)
    assert gate.events == []
    assert not (gate.out / "sensitivity").exists()


@pytest.mark.parametrize("names", [[], ["first", "first"], ["../first"], ["."], [""], [1]])
def test_invalid_model_identifiers_rejected_before_work(gate, names):
    with pytest.raises(ValueError, match="model list|model identifier"):
        run(gate, model_names=names)
    assert gate.events == []
    assert not (gate.out / "sensitivity").exists()


def test_internal_neural_batch_mismatch_is_a_fatal_configuration_error(gate, monkeypatch):
    original = gate.predictor

    def bad(folder, device):
        p = original(folder, device)
        p.metadata["config"]["batch_size"] = 128
        return p

    monkeypatch.setattr(sensitivity, "Predictor", bad)
    with pytest.raises(ValueError, match="Neural batch"):
        run(gate)
    report = read(gate.out / "sensitivity/summary.json")
    assert report["status"] == "FAILED_STOPPED"
    assert not any(x[0] == "predict" for x in gate.events)


def test_predict_blocks_preserves_run_boundaries_and_tail():
    first, second = arrays("a", 257, block=0), arrays("b", 96, block=1)
    packed = {
        key: np.concatenate([first[key], second[key]]) for key in ("X", "C", "selection_index")
    }
    packed["X"][:257, 0, 0] = 10
    packed["X"][257:, 0, 0] = 20
    seen = []

    def predict(x, c):
        seen.append((len(x), np.unique(x[:, 0, 0]).tolist()))
        return np.broadcast_to(x[:, :1, :1], (len(x), 10, 4)).copy()

    result, sizes = sensitivity.predict_blocks(
        SimpleNamespace(predict_scaled=predict), packed, 256, SimpleNamespace(check=lambda: None)
    )
    assert sizes == [256, 1, 96]
    assert seen == [(256, [10]), (1, [10]), (96, [20])]
    assert np.all(result[:257] == 10) and np.all(result[257:] == 20)


def test_predict_blocks_rejects_interleaved_block_indices():
    sample = arrays("a", 4)
    sample["selection_index"] = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="not contiguous"):
        sensitivity.predict_blocks(
            SimpleNamespace(), sample, 256, SimpleNamespace(check=lambda: None)
        )


def test_nonfinite_prediction_retains_exact_failure_array(tmp_path):
    sample = arrays("bad", 3)
    failure_path = tmp_path / "failure.npz"
    predictor = SimpleNamespace(predict_scaled=lambda x, c: np.full((len(x), 10, 4), np.nan))
    with pytest.raises(ValueError, match="Nonfinite/malformed"):
        sensitivity.predict_blocks(
            predictor, sample, 256, SimpleNamespace(check=lambda: None), failure_path
        )
    report = read(failure_path.with_suffix(".json"))
    assert report["status"] == "FAILED_STOPPED"
    assert report["evidence_sha256"] == sha(failure_path)
    with np.load(failure_path, allow_pickle=False) as evidence:
        assert np.isnan(evidence["prediction"]).all()
        np.testing.assert_array_equal(evidence["source_indices"], np.arange(3))


def test_array_signature_binds_inputs_targets_and_identities():
    sample = arrays("a")
    original = sensitivity.array_signature(sample)
    for key in ("X", "C", "Y", "segment", "origin_us", "source_indices", "selection_index"):
        changed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sample.items()}
        changed[key].flat[0] += 1
        assert sensitivity.array_signature(changed) != original
    sample["run_id"] = np.array(["a"] * 3, dtype=object)
    with pytest.raises(ValueError, match="Object arrays"):
        sensitivity.array_signature(sample)


def test_all_models_finish_gate_before_any_full_fit_or_development(gate, monkeypatch):
    monkeypatch.setattr(engine, "Predictor", gate.predictor)
    monkeypatch.setattr(engine, "normalized_role", gate.normalized)

    def full_fit(*args):
        gate.events.append(("full_fit",))
        return {"purpose": "fresh full calibration"}

    monkeypatch.setattr(engine, "fit_calibration", full_fit)
    monkeypatch.setattr(engine, "seal_development", lambda *a, **kw: gate.events.append(("dev",)))
    monkeypatch.setattr(
        engine, "preflight_models", lambda *a, **kw: pytest.fail("Legacy gate rerun")
    )
    engine.calibrate(
        gate.root,
        gate.out,
        gate.progress,
        model_names=gate.names,
        backend_factory=gate.backend,
        prediction_batch_size=256,
        verify_prediction_batch=True,
        sensitivity_plan=gate.plan,
        sensitivity_policy=gate.policy,
    )
    assessments = [i for i, item in enumerate(gate.events) if item[0] == "assessment"]
    full = [i for i, item in enumerate(gate.events) if item[0] in {"full_fit", "dev"}]
    assert len(assessments) == len(gate.names)
    assert max(assessments) < min(full)
    assert sum(x[0] == "full_fit" for x in gate.events) == len(gate.names)
    for name in gate.names:
        assert (
            read(gate.out / "calibration" / name / "calibration.json")["purpose"]
            == "fresh full calibration"
        )


def test_failed_all_model_gate_blocks_all_full_work(gate, monkeypatch):
    gate.failure_models.add("third")
    monkeypatch.setattr(
        engine, "fit_calibration", lambda *a: pytest.fail("Full calibration started")
    )
    monkeypatch.setattr(
        engine, "seal_development", lambda *a, **kw: pytest.fail("Development started")
    )
    with pytest.raises(ValueError, match="failed before full calibration"):
        engine.calibrate(
            gate.root,
            gate.out,
            gate.progress,
            model_names=gate.names,
            backend_factory=gate.backend,
            prediction_batch_size=256,
            verify_prediction_batch=True,
            sensitivity_plan=gate.plan,
            sensitivity_policy=gate.policy,
        )
    assert not (gate.out / "calibration").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"sensitivity_plan": None},
        {"sensitivity_policy": None},
        {"numerical_policy": {"legacy": True}},
        {"prediction_batch_size": 128},
        {"verify_prediction_batch": False},
        {"reuse_calibration": {"first": {"files": {}}}},
    ],
)
def test_sensitivity_cannot_mix_criteria_or_reuse_calibration(gate, monkeypatch, change):
    monkeypatch.setattr(
        sensitivity,
        "run_sensitivity_gate",
        lambda *a, **kw: pytest.fail("Invalid configuration ran gate"),
    )
    settings = dict(
        prediction_batch_size=256,
        verify_prediction_batch=True,
        sensitivity_plan=gate.plan,
        sensitivity_policy=gate.policy,
    )
    settings.update(change)
    with pytest.raises(ValueError, match="Sensitivity requires"):
        engine.calibrate(gate.root, gate.out, gate.progress, **settings)
    assert gate.events == []
