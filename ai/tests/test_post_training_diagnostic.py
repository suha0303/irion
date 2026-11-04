"""Diagnostic-only authorization and release boundaries; no server or GPU work."""

import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_post_training import config, fresh_config

from execution.common import read, sha, write
from learning import operator
from learning import post_training as post


def diagnostic_config():
    return dict(
        fresh_config(),
        numerical_diagnostic_only=True,
        calibration_authorized=False,
        evaluation_authorized=False,
    )


def policy():
    return dict(
        schema="drone-ai.numeric-policy/1",
        batch_size=256,
        windows=257,
        rtol=1e-5,
        atol=1e-5,
        require_same_batch_exact=True,
        max_absolute_difference=1e-4,
    )


def test_diagnostic_mode_is_explicit_and_normal_only():
    post.config_check(diagnostic_config())


@pytest.mark.parametrize(
    "change",
    [
        {"numerical_diagnostic_only": "true"},
        {"calibration_authorized": True},
        {"evaluation_authorized": True},
        {"all_model_calibration_fresh": False},
        {"prediction_batch_size": 1},
        {"numerical_policy": policy()},
    ],
)
def test_diagnostic_scope_rejects_fitting_or_policy_selection(change):
    cfg = diagnostic_config()
    cfg.update(change)
    with pytest.raises(ValueError):
        post.config_check(cfg)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"diagnostic_only": True, "fresh_calibration": False}, "fresh-calibration"),
        (
            {
                "diagnostic_only": True,
                "fresh_calibration": True,
                "numerical_policy": "/unused/policy.json",
            },
            "without selecting a policy",
        ),
        (
            {"numerical_policy": "/unused/policy.json", "fresh_calibration": False},
            "fresh-calibration",
        ),
    ],
)
def test_prepare_rejects_conflicting_modes_before_io(tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        post.prepare(SimpleNamespace(approve_post_training=True, **options))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("phase", ["fit", "calibrate", "evaluate"])
def test_diagnostic_cannot_submit_other_phases(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(post, "checked", lambda root: (root, diagnostic_config()))
    with pytest.raises(ValueError):
        post.submit(tmp_path, phase, diagnostic_config())
    assert not list(tmp_path.glob("*-submitted.claim"))


def test_submit_uses_frozen_mode_not_caller_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "checked", lambda root: (root, diagnostic_config()))
    monkeypatch.setattr(post, "SUBMIT_LOCK", tmp_path / "submit.lock")
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    monkeypatch.setattr(post, "python_for", lambda *a, **kw: "/approved/python")
    submitted = []
    monkeypatch.setattr(operator, "command", lambda argv: submitted.append(argv) or "123")
    with pytest.raises(ValueError):
        post.submit(tmp_path, "calibrate", config())
    assert not submitted
    assert not (tmp_path / "calibrate-submitted.claim").exists()


@pytest.mark.parametrize("frozen_diagnostic,requested_diagnostic", [(True, False), (False, True)])
def test_wrong_worker_phase_stops_before_claim(
    tmp_path, monkeypatch, frozen_diagnostic, requested_diagnostic
):
    cfg = diagnostic_config() if frozen_diagnostic else fresh_config()
    monkeypatch.setattr(post, "checked", lambda root: (root, cfg))
    with pytest.raises(ValueError, match="frozen authorization"):
        post.calibrate_worker(tmp_path, diagnostic_only=requested_diagnostic)
    assert not list(tmp_path.rglob("*-started.claim"))


def test_diagnostic_evaluator_stops_before_runtime_or_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "checked", lambda root: (root, diagnostic_config()))
    with pytest.raises(ValueError, match="cannot evaluate"):
        post.evaluate_worker(tmp_path)
    assert not (tmp_path / "truth").exists()
    assert not (tmp_path / "evaluation-runtime").exists()


def test_diagnostic_worker_only_checks_normal_validation_and_seals(tmp_path, monkeypatch):
    import execution.engine
    import learning.telemetry
    import learning.training
    import pipeline.runtime

    cfg = diagnostic_config()
    upstream = tmp_path / "upstream"
    cfg.update(
        upstream=str(upstream),
        evaluation_input_files={
            "dataset/normal_validation.npz": "validation-hash",
            "dataset/scaler.npz": "scaler-hash",
            "dataset/manifest.json": "manifest-hash",
            "dataset/normal_calibration.npz": "calibration-hash",
            "dataset/canonical/anomaly-development/samples.jsonl": "development-hash",
        },
    )
    write(tmp_path / "input/config.json", cfg)
    (tmp_path / "compute").mkdir()
    monkeypatch.setattr(post, "checked", lambda root: (root, cfg))
    monkeypatch.setattr(post.signal, "signal", lambda *a: None)
    monkeypatch.setattr(post, "memory_snapshot", lambda: {"available_bytes": 32 * 1024**3})
    monkeypatch.setattr(pipeline.runtime, "check", lambda *a, **kw: {"approved": True})
    monkeypatch.setattr(learning.training, "project_lock", lambda *a: nullcontext())
    checks, preflights = [], []
    monkeypatch.setattr(post, "verify", lambda root, files: checks.append((root, files)))
    monkeypatch.setattr(
        learning.telemetry,
        "Telemetry",
        lambda out: SimpleNamespace(
            start=lambda: None,
            stop=lambda: None,
        ),
    )
    monkeypatch.setattr(
        execution.engine,
        "calibrate",
        lambda *a, **kw: pytest.fail(
            "Diagnostic must not fit calibration or replay development data"
        ),
    )

    def preflight(root, out, progress, **kwargs):
        preflights.append(kwargs)
        write(out / "batch-checks/preflight.json", {"diagnostic_only": True})

    monkeypatch.setattr(execution.engine, "preflight_models", preflight)
    post.calibrate_worker(tmp_path, diagnostic_only=True)
    assert len(preflights) == 1
    assert preflights[0]["diagnostic_only"] is True
    assert preflights[0]["prediction_batch_size"] == 256
    assert preflights[0]["model_names"] == cfg["models"]
    assert checks == [
        (
            upstream,
            {
                "dataset/normal_validation.npz": "validation-hash",
                "dataset/scaler.npz": "scaler-hash",
                "dataset/manifest.json": "manifest-hash",
            },
        )
    ]
    seal = read(tmp_path / "compute/diagnostic-seal.json")
    assert seal["status"] == "DIAGNOSTIC_COMPLETE_PENDING_RELEASE"
    assert seal["calibration_started"] is False
    assert seal["truth_join"] is False
    assert not (tmp_path / "compute/calibration").exists()
    assert not (tmp_path / "compute/result-seal.json").exists()


def controller_fixture(tmp_path, monkeypatch):
    cfg = diagnostic_config()
    write(tmp_path / "input/config.json", cfg)
    write(tmp_path / "compute/batch-checks/preflight.json", {"models": cfg["models"]})
    write(
        tmp_path / "compute/diagnostic-seal.json",
        dict(
            status="DIAGNOSTIC_COMPLETE_PENDING_RELEASE",
            models=cfg["models"],
            config_sha256=sha(tmp_path / "input/config.json"),
            files={
                "batch-checks/preflight.json": sha(tmp_path / "compute/batch-checks/preflight.json")
            },
            training_repeated=False,
            calibration_started=False,
            inference_started=False,
            truth_join=False,
        ),
    )
    monkeypatch.setattr(post, "checked", lambda root: (root, cfg))
    monkeypatch.setattr(post, "python_for", lambda *a, **kw: sys.executable)
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    phases = []

    def submit(root, phase, frozen):
        assert frozen == cfg
        assert not (root / "completion.json").exists()
        phases.append(phase)
        return dict(job_id="123", phase=phase, uid=os.getuid())

    monkeypatch.setattr(post, "submit", submit)
    return cfg, phases


def test_diagnostic_control_releases_one_job_and_never_calibrates(tmp_path, monkeypatch):
    cfg, phases = controller_fixture(tmp_path, monkeypatch)

    def release(root, record, state, budget):
        assert phases == ["diagnose"]
        assert not (root / "completion.json").exists()
        return dict(job_id="123", released=True, slurm_state="COMPLETED", exit_code="0:0")

    monkeypatch.setattr(operator, "wait_release", release)
    monkeypatch.setattr(operator, "report", lambda *a: pytest.fail("Evaluation report forbidden"))
    post.control(tmp_path)
    assert phases == ["diagnose"]
    completed = read(tmp_path / "completion.json")
    assert completed["status"] == "DIAGNOSTIC_READY_STOPPED"
    assert completed["diagnostic_release"]["released"] is True
    assert completed["calibration_started"] is False
    assert completed["inference_started"] is False
    assert not (tmp_path / "evaluation-release.json").exists()
    assert not (tmp_path / "truth").exists()


def test_failed_diagnostic_release_never_marks_completion(tmp_path, monkeypatch):
    _, phases = controller_fixture(tmp_path, monkeypatch)
    failed = dict(job_id="123", released=True, slurm_state="FAILED", exit_code="1:0")
    monkeypatch.setattr(operator, "wait_release", lambda *a: failed)
    monkeypatch.setattr(operator, "snapshot", lambda *a: failed)
    with pytest.raises(RuntimeError):
        post.control(tmp_path)
    assert phases == ["diagnose"]
    assert read(tmp_path / "controller.json")["status"] == "FAILED_STOPPED"
    assert not (tmp_path / "completion.json").exists()


def test_unacknowledged_diagnostic_submission_is_unconfirmed(tmp_path, monkeypatch):
    controller_fixture(tmp_path, monkeypatch)

    def unknown(root, phase, cfg):
        write(root / "diagnose-submission-intent.json", {"argv": ["sbatch"]})
        raise TimeoutError("unknown submission")

    monkeypatch.setattr(post, "submit", unknown)
    with pytest.raises(TimeoutError):
        post.control(tmp_path)
    assert (
        read(tmp_path / "controller.json")["status"] == "CONTROLLER_STOPPED_JOB_STATE_UNCONFIRMED"
    )
    assert not (tmp_path / "completion.json").exists()


def test_diagnostic_seal_rejects_changed_arrays_before_completion(tmp_path, monkeypatch):
    cfg, _ = controller_fixture(tmp_path, monkeypatch)
    write(tmp_path / "compute/batch-checks/preflight.json", {"tampered": True})
    with pytest.raises(ValueError, match="Input hash changed"):
        post.diagnostic_seal_check(tmp_path, cfg)


def test_common_policy_requires_every_calibration_fresh():
    cfg = fresh_config()
    cfg["numerical_policy"] = policy()
    post.config_check(cfg)
    del cfg["all_model_calibration_fresh"]
    with pytest.raises(ValueError, match="fresh calibration for every model"):
        post.config_check(cfg)


def test_policy_copy_cannot_diverge_from_frozen_config(tmp_path, monkeypatch):
    cfg = fresh_config()
    source = tmp_path / "source"
    source.mkdir()
    root = tmp_path / "attempt"
    write(root / "input/numerical-policy.json", policy())
    cfg.update(
        numerical_policy=policy(),
        numerical_policy_sha256=sha(root / "input/numerical-policy.json"),
        source_training=str(source),
        source_training_files={},
        model_files={},
    )
    write(root / "input/config.json", cfg)
    write(root / "server-profile.json", {})
    write(
        root / "authorization.json",
        dict(
            config_sha256=sha(root / "input/config.json"),
            execution_scope=post.SCOPE,
        ),
    )

    def bind_inventory():
        write(
            root / "input/inventory.json",
            {name: sha(root / "input" / name) for name in ("config.json", "numerical-policy.json")},
        )
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
    bind_inventory()
    post.checked(root)
    # Even a separately refreshed inventory cannot detach the copied policy from
    # the already authorized config and its independently recorded policy hash.
    write(root / "input/numerical-policy.json", {**policy(), "atol": 0.1})
    bind_inventory()
    with pytest.raises(ValueError, match="Frozen common numerical policy changed"):
        post.checked(root)
