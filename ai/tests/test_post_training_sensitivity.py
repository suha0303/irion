"""Frozen sensitivity authorization and worker plumbing; no server/GPU calls."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_post_training import fresh_config

from execution.common import read, sha, write
from learning import post_training as post


def policy():
    return dict(
        schema="drone-ai.sensitivity-policy/1",
        batch_size=256,
        require_same_batch_exact=True,
        max_scaled_abs=1e-4,
        physical_max=[1e-5, 1e-5, 1e-5, 1e-4],
        max_score_delta_over_high=0.01,
        max_ewma_delta_over_high=0.01,
        max_changed_alarm_states=0,
    )


def config():
    return dict(fresh_config(), sensitivity_policy=policy(), sensitivity_plan={"frozen": "plan"})


def test_sensitivity_is_one_fresh_full_comparison():
    post.config_check(config())


@pytest.mark.parametrize(
    "change",
    [
        {
            "numerical_diagnostic_only": True,
            "calibration_authorized": False,
            "evaluation_authorized": False,
        },
        {"all_model_calibration_fresh": False},
        {"prediction_batch_size": 1},
        {"sensitivity_plan": None},
        {"sensitivity_policy": None},
        {"sensitivity_policy": {**policy(), "model_overrides": {}}},
        {"sensitivity_policy": {**policy(), "batch_size": 128}},
        {"sensitivity_policy": {**policy(), "require_same_batch_exact": False}},
        {"reuse_calibration": {"persistence": {"files": {}}}},
        {"verify_prediction_batch": False},
        {"models": ["persistence"]},
        {"production_training_authorized": True},
        {"final_test_used": True},
    ],
)
def test_config_rejects_changed_scope_or_mixed_conditions(change):
    cfg = config()
    cfg.update(change)
    with pytest.raises(ValueError):
        post.config_check(cfg)


@pytest.mark.parametrize(
    "changes",
    [
        {"fresh_calibration": False},
        {"fresh_calibration": True, "diagnostic_only": True},
        {"fresh_calibration": True, "numerical_policy": "/unused/old-policy.json"},
    ],
)
def test_prepare_rejects_conflicting_mode_before_io(changes):
    with pytest.raises(ValueError, match="sensitivity-policy"):
        post.prepare(
            SimpleNamespace(
                approve_post_training=True, sensitivity_policy="/unused/sensitivity.json", **changes
            )
        )


@pytest.mark.parametrize("tampered", ["sensitivity_policy", "sensitivity_plan"])
def test_file_and_config_cannot_diverge_even_with_new_inventory(tmp_path, monkeypatch, tampered):
    cfg = config()
    root = tmp_path / "attempt"
    source = tmp_path / "source"
    source.mkdir()
    for key in ("sensitivity_policy", "sensitivity_plan"):
        path = root / "input" / (key.replace("_", "-") + ".json")
        write(path, cfg[key])
        cfg[key + "_sha256"] = sha(path)
    cfg.update(source_training=str(source), source_training_files={}, model_files={})
    write(root / "input/config.json", cfg)
    write(root / "server-profile.json", {})
    write(
        root / "authorization.json",
        dict(
            config_sha256=sha(root / "input/config.json"),
            execution_scope=post.SCOPE,
        ),
    )

    def inventory():
        names = ("config.json", "sensitivity-policy.json", "sensitivity-plan.json")
        write(root / "input/inventory.json", {name: sha(root / "input" / name) for name in names})
        write(
            root / "prepared.json",
            dict(
                input_inventory_sha256=sha(root / "input/inventory.json"),
                server_profile_sha256=sha(root / "server-profile.json"),
                authorization_sha256=sha(root / "authorization.json"),
            ),
        )

    monkeypatch.setattr(post, "runtime_profile", lambda *a, **kw: {})
    monkeypatch.setattr(post, "profile_check", lambda *a: None)
    inventory()
    post.checked(root)
    path = root / "input" / (tampered.replace("_", "-") + ".json")
    write(path, {**cfg[tampered], "changed": True})
    inventory()
    with pytest.raises(ValueError, match="Frozen common sensitivity inputs changed"):
        post.checked(root)


@pytest.mark.parametrize("failed", [False, True])
def test_worker_passes_one_frozen_gate_and_seals_only_after_success(tmp_path, monkeypatch, failed):
    import execution.engine
    import learning.telemetry
    import learning.training
    import pipeline.runtime

    cfg = config()
    cfg.update(
        upstream=str(tmp_path / "upstream"),
        source_training_seal_sha256="sealed-training",
        evaluation_input_files={
            "dataset/normal_validation.npz": "v",
            "dataset/normal_calibration.npz": "c",
            "dataset/scaler.npz": "s",
            "dataset/manifest.json": "m",
            "dataset/canonical/development/samples.jsonl": "d",
        },
    )
    write(tmp_path / "input/config.json", cfg)
    (tmp_path / "compute").mkdir()
    monkeypatch.setattr(post, "checked", lambda root: (root, cfg))
    monkeypatch.setattr(post.signal, "signal", lambda *a: None)
    monkeypatch.setattr(post, "memory_snapshot", lambda: {"available_bytes": 32 * 1024**3})
    monkeypatch.setattr(pipeline.runtime, "check", lambda *a, **kw: {"approved": True})
    monkeypatch.setattr(learning.training, "project_lock", lambda *a: nullcontext())
    monkeypatch.setattr(
        learning.telemetry,
        "Telemetry",
        lambda out: SimpleNamespace(
            start=lambda: None,
            stop=lambda: None,
        ),
    )
    checked_files, calls = [], []
    monkeypatch.setattr(post, "verify", lambda root, files: checked_files.append(files))
    monkeypatch.setattr(
        execution.engine,
        "preflight_models",
        lambda *a, **kw: pytest.fail("Diagnostic-only phase was incorrectly called"),
    )

    def calibrate(root, out, progress, **kw):
        calls.append(kw)
        write(
            out / "sensitivity/summary.json", {"status": "FAILED_STOPPED" if failed else "PASSED"}
        )
        if failed:
            raise ValueError("sensitivity gate failed")

    monkeypatch.setattr(execution.engine, "calibrate", calibrate)
    if failed:
        with pytest.raises(ValueError, match="sensitivity gate failed"):
            post.calibrate_worker(tmp_path)
        assert not (tmp_path / "compute/result-seal.json").exists()
        assert read(tmp_path / "compute/progress.json")["status"] == "FAILED_STOPPED"
    else:
        post.calibrate_worker(tmp_path)
        seal = read(tmp_path / "compute/result-seal.json")
        assert seal["all_model_sensitivity_passed"] is True
        assert seal["diagnostic_calibration_reused"] is False
        assert seal["normal_calibration_reused"] == []
        assert seal["training_repeated"] is seal["truth_join"] is False
    assert len(calls) == 1
    assert calls[0]["sensitivity_plan"] == cfg["sensitivity_plan"]
    assert calls[0]["sensitivity_policy"] == cfg["sensitivity_policy"]
    assert calls[0]["model_names"] == cfg["models"]
    assert calls[0]["prediction_batch_size"] == 256
    assert calls[0]["reuse_calibration"] == {}
    assert calls[0]["numerical_policy"] is None
    assert checked_files == [cfg["evaluation_input_files"]]
