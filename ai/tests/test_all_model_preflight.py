"""No model may calibrate or replay before every model passes common preflight."""

import shutil
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from execution import engine
from execution.common import read, write
from learning import replay_validation


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root, out = tmp_path / "upstream", tmp_path / "compute"
    names = ("first", "second", "third")
    dataset = root / "dataset"
    write(dataset / "manifest.json", {"runs": [
        {"run_id": "val", "role": "normal_validation", "paired_case_id": "v"},
        {"run_id": "cal", "role": "normal_calibration", "paired_case_id": "c"},
        {"run_id": "dev", "role": "normal_dev_eval", "paired_case_id": "d"},
    ]})
    np.savez(dataset / "scaler.npz", y_scale=np.ones(4))
    for name in names:
        folder = out / "models" / name
        write(folder / "integrity.json", {})
        shutil.copyfile(dataset / "scaler.npz", folder / "scaler.npz")
    events = []

    def predictor(folder, device):
        return SimpleNamespace(name=folder.name, scaler={"y_scale": np.ones(4)},
                               predict_scaled=lambda x, c: np.zeros((len(x), 10, 4)))

    def normalized(dataset, role, scaler):
        events.append(("load", role))
        assert role in {"normal_validation", "normal_calibration"}
        return dict(X=np.zeros((2, 100, 19)), C=np.zeros((2, 6)),
                    Y=np.zeros((2, 10, 4)), run_id=np.array(["val", "val"]),
                    segment=np.ones(2), origin_us=np.array([1, 2]))

    def fit(*args):
        events.append(("fit",))
        return {"status": "fixture"}

    monkeypatch.setattr(engine, "Predictor", predictor)
    monkeypatch.setattr(engine, "normalized_role", normalized)
    monkeypatch.setattr(engine, "fit_calibration", fit)
    monkeypatch.setattr(engine, "seal_development",
                        lambda *args, **kw: events.append(("dev", args[1].name)))
    return SimpleNamespace(root=root, out=out, names=names, events=events,
                           progress=SimpleNamespace(check=lambda: None, update=lambda **k: None),
                           backend=lambda device: nullcontext())


def save_report(path, status="PASSED", **extra):
    report = dict(status=status, input_sha256="same-frozen-sample",
                  sample_sha256="same-normal-targets-and-identities", **extra)
    write(path, report)
    return report


def test_all_checks_finish_before_first_full_calibration_or_replay(prepared, monkeypatch):
    p = prepared

    def check(predictor, normal, output, **kwargs):
        assert kwargs["batch_size"] == 256
        p.events.append(("gate", predictor.name))
        return save_report(output)

    monkeypatch.setattr(replay_validation, "check_prediction_batch", check)
    engine.calibrate(p.root, p.out, p.progress, model_names=p.names,
                     backend_factory=p.backend, prediction_batch_size=256,
                     verify_prediction_batch=True)
    gates = [index for index, event in enumerate(p.events) if event[0] == "gate"]
    full_work = [index for index, event in enumerate(p.events)
                 if event[0] in {"fit", "dev"} or event == ("load", "normal_calibration")]
    assert len(gates) == len(p.names)
    assert max(gates) < min(full_work)
    assert [event[1] for event in p.events if event[0] == "dev"] == list(p.names)
    report = read(p.out / "batch-checks/preflight.json")
    assert report["status"] == "PASSED"
    assert report["full_calibration_started"] is False
    assert report["development_inference_started"] is False


def test_numerical_failure_collects_other_models_and_prevents_all_full_work(prepared, monkeypatch):
    p = prepared

    def check(predictor, normal, output, **kwargs):
        p.events.append(("gate", predictor.name))
        if predictor.name in p.names[:2]:
            save_report(output, "FAILED_STOPPED")
            raise ValueError("Numerical check failed")
        return save_report(output)

    monkeypatch.setattr(replay_validation, "check_prediction_batch", check)
    with pytest.raises(ValueError, match="failed before calibration: first, second"):
        engine.calibrate(p.root, p.out, p.progress, model_names=p.names,
                         backend_factory=p.backend, prediction_batch_size=256,
                         verify_prediction_batch=True)
    assert [event[1] for event in p.events if event[0] == "gate"] == list(p.names)
    assert all(event[0] not in {"fit", "dev"} for event in p.events)
    assert ("load", "normal_calibration") not in p.events
    assert not (p.out / "calibration").exists()
    report = read(p.out / "batch-checks/preflight.json")
    assert report["failures"] == ["first", "second"]
    assert report["status"] == "FAILED_STOPPED"
    assert report["models_completed"] == 3


@pytest.mark.parametrize("diagnostic_only", [True, False])
def test_rich_diagnostic_uses_one_policy_and_no_calibration(prepared, monkeypatch, diagnostic_only):
    p = prepared
    observed_policies = []
    policy = dict(schema="drone-ai.numeric-policy/1", batch_size=256, windows=257,
                  rtol=1e-5, atol=1e-6, require_same_batch_exact=True)

    def diagnose(predictor, normal, output, **kwargs):
        p.events.append(("gate", predictor.name))
        assert kwargs["batch_size"] == 256
        return save_report(output, "DIAGNOSTIC_COMPLETE")

    def assess(report_path, supplied_policy, output):
        observed_policies.append(supplied_policy.copy())
        return save_report(output)

    monkeypatch.setattr(replay_validation, "diagnose_prediction_batch", diagnose)
    monkeypatch.setattr(replay_validation, "assess_diagnostic", assess)
    summary = engine.preflight_models(
        p.root, p.out, p.progress, model_names=p.names, backend_factory=p.backend,
        diagnostic_only=diagnostic_only, numerical_policy=None if diagnostic_only else policy,
    )
    assert [event[1] for event in p.events if event[0] == "gate"] == list(p.names)
    assert observed_policies == ([] if diagnostic_only else [policy] * len(p.names))
    assert summary["status"] == ("DIAGNOSTIC_COMPLETE" if diagnostic_only else "PASSED")
    assert summary["calibration_changed"] is summary["event_truth_used"] is False
    assert not (p.out / "calibration").exists()


def test_changed_scaler_stops_before_any_model_forward(prepared, monkeypatch):
    p = prepared
    np.savez(p.out / "models/second/scaler.npz", y_scale=np.ones(4) * 2)
    with pytest.raises(ValueError, match="same frozen train scaler: second"):
        engine.preflight_models(p.root, p.out, p.progress, model_names=p.names,
                                backend_factory=p.backend)
    assert p.events == []


def test_runtime_error_stops_immediately_and_records_failure(prepared, monkeypatch):
    p = prepared

    def broken(*args, **kwargs):
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(replay_validation, "check_prediction_batch", broken)
    with pytest.raises(RuntimeError, match="device unavailable"):
        engine.preflight_models(p.root, p.out, p.progress, model_names=p.names,
                                backend_factory=p.backend)
    assert p.events == [("load", "normal_validation")]
    assert read(p.out / "batch-checks/preflight.json")["status"] == "FAILED_STOPPED"
    assert not (p.out / "calibration").exists()


def test_existing_preflight_evidence_is_never_overwritten(prepared):
    p = prepared
    output = p.out / "batch-checks/first.json"
    save_report(output, "FAILED_STOPPED", previous_attempt=True)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="new attempt"):
        engine.preflight_models(p.root, p.out, p.progress, model_names=p.names,
                                backend_factory=p.backend)
    assert output.read_bytes() == original
    assert p.events == []


def test_neural_internal_batch_cannot_silently_reduce_common_batch(prepared, monkeypatch):
    p = prepared
    monkeypatch.setattr(engine, "Predictor", lambda *args, **kwargs: SimpleNamespace(
        scaler={"y_scale": np.ones(4)},
        metadata={"name": "transformer", "config": {"batch_size": 128}},
    ))
    with pytest.raises(ValueError, match="Neural predictor batch differs"):
        engine.preflight_models(p.root, p.out, p.progress, model_names=p.names,
                                backend_factory=p.backend)
    assert read(p.out / "batch-checks/preflight.json")["status"] == "FAILED_STOPPED"
    assert not (p.out / "calibration").exists()


def test_rich_preflight_rejects_different_normal_targets_or_identity(prepared, monkeypatch):
    p = prepared

    def diagnose(predictor, normal, output, **kwargs):
        report = save_report(output, "DIAGNOSTIC_COMPLETE")
        if predictor.name == "second":
            report["sample_sha256"] = "different-run-membership"
            write(output, report)
        return report

    monkeypatch.setattr(replay_validation, "diagnose_prediction_batch", diagnose)
    with pytest.raises(ValueError, match="targets or run identities differ"):
        engine.preflight_models(p.root, p.out, p.progress, model_names=p.names,
                                backend_factory=p.backend, diagnostic_only=True)
    assert read(p.out / "batch-checks/preflight.json")["status"] == "FAILED_STOPPED"
    assert not (p.out / "calibration").exists()


def test_actual_rich_diagnostic_and_policy_complete_before_any_full_work(prepared, monkeypatch):
    p = prepared
    normal = dict(X=np.zeros((300, 100, 19)), C=np.zeros((300, 6)),
                  Y=np.zeros((300, 10, 4)), run_id=np.full(300, "val"),
                  segment=np.ones(300, dtype=np.int64), origin_us=np.arange(300))

    def normalized(dataset, role, scaler):
        assert role == "normal_validation"
        return normal

    monkeypatch.setattr(engine, "normalized_role", normalized)
    policy = dict(schema="drone-ai.numeric-policy/1", batch_size=256, windows=257,
                  rtol=1e-5, atol=1e-6, require_same_batch_exact=True)
    summary = engine.preflight_models(p.root, p.out, p.progress, model_names=p.names,
                                      backend_factory=p.backend, numerical_policy=policy)
    assert summary["status"] == "PASSED"
    assert len(summary["models"]) == len(p.names)
    assert len(summary["common_sample_sha256"]) == 64
    for name in p.names:
        assert read(p.out / "batch-checks" / (name + ".decision.json"))["status"] == "PASSED"
        assert (p.out / "batch-checks" / (name + ".npz")).is_file()
    assert not (p.out / "calibration").exists()


def test_policy_failure_for_first_model_still_checks_remaining_models(prepared, monkeypatch):
    p = prepared
    policy = dict(schema="drone-ai.numeric-policy/1", batch_size=256, windows=257,
                  rtol=1e-5, atol=1e-6, require_same_batch_exact=True)
    checked = []

    def diagnose(predictor, normal, output, **kwargs):
        return save_report(output, "DIAGNOSTIC_COMPLETE")

    def assess(report_path, policy, output):
        checked.append(report_path.stem)
        if report_path.stem == "first":
            save_report(output, "FAILED_STOPPED")
            raise ValueError("Policy limit exceeded")
        return save_report(output)

    monkeypatch.setattr(replay_validation, "diagnose_prediction_batch", diagnose)
    monkeypatch.setattr(replay_validation, "assess_diagnostic", assess)
    with pytest.raises(ValueError, match="failed before calibration: first"):
        engine.calibrate(p.root, p.out, p.progress, model_names=p.names,
                         backend_factory=p.backend, numerical_policy=policy,
                         prediction_batch_size=256, verify_prediction_batch=True)
    assert checked == list(p.names)
    assert read(p.out / "batch-checks/preflight.json")["failures"] == ["first"]
    assert not (p.out / "calibration").exists()
