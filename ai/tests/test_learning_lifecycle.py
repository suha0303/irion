"""CPU fixtures and simulated scheduler responses. No remote submission or production fit."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from execution.common import read, sha, write
from learning import lifecycle, operator, telemetry


def server_profile(root, monkeypatch):
    profile = root / "server-profile.json"
    write(
        profile,
        dict(
            schema="drone-ai.profile/1",
            name="fixture",
            project_root=str(root),
            python=sys.executable,
            runtime_profile="torch21",
            dataset_root=str(root / "dataset"),
            output_root=str(root / "results"),
            ssh=None,
            slurm=dict(
                partition="p02", node="DIS01", nodes=1, ntasks=1, gpus=1, cpus=4, memory_gib=16
            ),
        ),
    )
    write(root / "prepared.json", dict(server_profile_sha256=sha(profile)))
    monkeypatch.setenv("DRONE_AI_PROFILE", str(profile))
    return profile


def test_gpu_telemetry_maps_worker_not_physical_index():
    rows = [["10", "GPU-other"], ["20", "GPU-owned"]]
    uuid = telemetry.own_gpu_uuid(rows, 20)
    result = telemetry.parse_gpu(
        [
            ["GPU-other", "99", "99", "40000", "48000", "250", "80"],
            ["GPU-owned", "61", "20", "1024", "48000", "150", "50"],
        ],
        uuid,
    )
    assert result["utilization_pct"] == 61 and result["used_mib"] == 1024
    with pytest.raises(ValueError):
        telemetry.own_gpu_uuid(rows, 30)
    with pytest.raises(ValueError):
        telemetry.own_gpu_uuid(rows + [["20", "GPU-other"]], 20)
    with pytest.raises(ValueError):
        telemetry.parse_gpu([], uuid)


def test_purged_job_and_failed_accounting_are_not_completed():
    result = operator.parse_snapshot("123", "", "123|me|COMPLETED|0:0", "me")
    operator.require_success(result)
    assert result["released"]
    pending = operator.parse_snapshot("123", "123|x500-next-fit|PENDING|Resources", "", "me")
    with pytest.raises(RuntimeError):
        operator.require_success(pending)
    unknown = operator.parse_snapshot("123", "", "", "me")
    assert unknown["status"] == "UNKNOWN_ACCOUNTING" and not unknown["released"]
    with pytest.raises(RuntimeError):
        operator.require_success(operator.parse_snapshot("123", "", "123|me|FAILED|1:0", "me"))
    with pytest.raises(ValueError):
        operator.parse_snapshot("123", "", "123|other|COMPLETED|0:0", "me")


def test_exact_single_submission_and_scope(tmp_path, monkeypatch):
    (tmp_path / "input").mkdir()
    write(tmp_path / "input/inventory.json", {})
    server_profile(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "LOCK", tmp_path / "project.lock")
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    commands = []

    def command(argv):
        commands.append(argv)
        return "123"

    monkeypatch.setattr(operator, "command", command)
    result = operator.submit(tmp_path, "fit", {})
    assert result["job_id"] == "123" and result["uid"] == os.getuid()
    assert all(
        x in commands[0]
        for x in [
            "--partition=p02",
            "--nodelist=DIS01",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            "--no-requeue",
        ]
    )
    with pytest.raises(FileExistsError):
        operator.submit(tmp_path, "fit", {})
    assert len(commands) == 1
    write(tmp_path / "release.json", dict(released=False, slurm_state="RUNNING"))
    with pytest.raises(RuntimeError):
        operator.submit(tmp_path, "evaluate", {})
    assert len(commands) == 1 and not (tmp_path / "evaluate-submitted.claim").exists()


def test_controller_stops_after_failed_fit_without_evaluation(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    server_profile(tmp_path, monkeypatch)
    write(tmp_path / "input/config.json", dict(revision="fixture"))
    phases = []

    def submit(root, phase, cfg):
        phases.append(phase)
        return dict(job_id="123", phase=phase, uid=os.getuid())

    failure = dict(released=True, slurm_state="FAILED", exit_code="1:0", remaining_own_queue="")
    monkeypatch.setattr(operator, "submit", submit)
    monkeypatch.setattr(operator, "wait_release", lambda *args: failure)
    monkeypatch.setattr(operator, "snapshot", lambda *args: failure)
    monkeypatch.setattr(
        operator, "wait_phase_files", lambda *a: pytest.fail("No visibility wait for failed jobs")
    )
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    with pytest.raises(RuntimeError):
        operator.control(tmp_path)
    assert phases == ["fit"] and read(tmp_path / "controller.json")["status"] == "FAILED_STOPPED"
    with pytest.raises(FileExistsError):
        operator.control(tmp_path)
    assert phases == ["fit"]


def test_memory_gate_failure_never_runs_resume_or_fit(tmp_path, monkeypatch):
    def failed(*a, **k):
        raise MemoryError("fixture insufficient memory")

    monkeypatch.setattr(lifecycle, "measure", failed)
    monkeypatch.setattr(
        lifecycle.subprocess, "run", lambda *a, **k: pytest.fail("no child after memory failure")
    )
    cfg = dict(gate_windows=512, hidden=128, layers=2, heads=4)
    arrays = {"X": np.zeros((512, 100, 19)), "C": np.zeros((512, 6)), "Y": np.zeros((512, 10, 4))}
    progress = SimpleNamespace(update=lambda **k: None, check=lambda: None)
    with pytest.raises(MemoryError):
        lifecycle.gates(tmp_path, cfg, tmp_path, arrays, arrays, progress)
    assert not (tmp_path / "gates/passed.json").exists()


def test_display_preserves_unknown_and_terminal_state(tmp_path):
    assert "미확인" in operator.render(tmp_path)
    write(
        tmp_path / "controller.json",
        dict(status="REPORT_READY_STOPPED", stage="stopped", job_id="123"),
    )
    write(
        tmp_path / "compute/gpu-live.json",
        dict(
            status="SAMPLED",
            timestamp=1,
            utilization_pct=80,
            used_mib=1024,
            total_mib=48000,
            power_w=100,
            temperature_c=50,
        ),
    )
    shown = operator.render(tmp_path)
    assert "REPORT_READY_STOPPED" in shown and "마지막 기록" in shown and "실시간:" not in shown


def visibility_clock(monkeypatch, after_sleep=None):
    now, sleeps = [0.0], []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
        if after_sleep:
            after_sleep(len(sleeps))

    monkeypatch.setattr(operator.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(operator.time, "sleep", sleep)
    return sleeps


def test_released_fit_waits_for_seal_and_members_without_hash_retry(tmp_path, monkeypatch):
    def publish(index):
        if index == 1:
            write(
                tmp_path / "compute/result-seal.json",
                dict(status="TRAIN_CALIBRATION_INFERENCE_COMPLETE", files={"weights.bin": "hash"}),
            )
        elif index == 2:
            (tmp_path / "compute/weights.bin").write_bytes(b"weights")

    sleeps = visibility_clock(monkeypatch, publish)
    monkeypatch.setattr(
        operator, "verify", lambda *a: pytest.fail("No repeated hashes while waiting")
    )
    state = dict(status="RUNNING", stage="fit")
    operator.wait_phase_files(tmp_path, "fit", state, {})
    assert sleeps == [5, 5]
    assert state["allocation_released"] and state["file_visibility"] == "VISIBLE"
    assert state["file_visibility_waited_s"] == 10
    assert "file_visibility_last_error" not in state
    assert read(tmp_path / "controller.json")["file_visibility_phase"] == "fit"


def test_released_evaluation_waits_for_all_report_metadata(tmp_path, monkeypatch):
    def publish(index):
        if index == 1:
            write(tmp_path / "evaluation/report.json", dict(status="REPORT_READY_STOPPED"))
        elif index == 2:
            write(tmp_path / "evaluation/metrics.json", {})
        elif index == 3:
            write(tmp_path / "compute/models/model/training.json", {})

    sleeps = visibility_clock(monkeypatch, publish)
    state = {}
    operator.wait_phase_files(tmp_path, "evaluate", state, dict(models=["model"]))
    assert sleeps == [5, 5, 5]
    assert state["file_visibility"] == "VISIBLE" and state["allocation_released"]


def test_released_artifact_wait_is_bounded_and_never_resubmits(tmp_path, monkeypatch):
    sleeps = visibility_clock(monkeypatch)
    monkeypatch.setattr(operator, "submit", lambda *a: pytest.fail("No job resubmission"))
    state = {}
    with pytest.raises(FileNotFoundError, match="120s; no job retry"):
        operator.wait_phase_files(tmp_path, "fit", state, {})
    assert len(sleeps) == 24 and sum(sleeps) == 120
    assert state["file_visibility"] == "TIMED_OUT" and state["allocation_released"]


@pytest.mark.parametrize("invalid", ["{invalid-json", '{"status":"FAILED_STOPPED","files":{}}'])
def test_released_artifact_invalid_json_or_status_never_waits(tmp_path, monkeypatch, invalid):
    path = tmp_path / "compute/result-seal.json"
    path.parent.mkdir()
    path.write_text(invalid)
    sleeps = visibility_clock(monkeypatch)
    state = {}
    with pytest.raises(ValueError):
        operator.wait_phase_files(tmp_path, "fit", state, {})
    assert not sleeps and state["allocation_released"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_controller_keeps_single_hash_verification_after_visibility(tmp_path, monkeypatch, corrupt):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    server_profile(tmp_path, monkeypatch)
    write(tmp_path / "input/config.json", dict(revision="fixture", models=["model"]))
    write(tmp_path / "compute/models/model/training.json", {})
    artifact = tmp_path / "compute/model.bin"
    artifact.write_bytes(b"model")
    write(
        tmp_path / "compute/result-seal.json",
        dict(
            status="TRAIN_CALIBRATION_INFERENCE_COMPLETE",
            files={"model.bin": "0" * 64 if corrupt else sha(artifact)},
        ),
    )
    write(tmp_path / "evaluation/metrics.json", {})
    write(
        tmp_path / "evaluation/report.json",
        dict(
            status="REPORT_READY_STOPPED", metrics_sha256=sha(tmp_path / "evaluation/metrics.json")
        ),
    )
    phases, verified = [], []

    def submit(root, phase, cfg):
        phases.append(phase)
        return dict(job_id=str(123 + len(phases)), phase=phase, uid=os.getuid())

    success = dict(released=True, slurm_state="COMPLETED", exit_code="0:0", remaining_own_queue="")
    original_verify = operator.verify

    def verify(root, inventory):
        verified.append(root)
        original_verify(root, inventory)

    monkeypatch.setattr(operator, "submit", submit)
    monkeypatch.setattr(operator, "verify", verify)
    monkeypatch.setattr(operator, "wait_release", lambda *a: success)
    monkeypatch.setattr(operator, "snapshot", lambda *a: success)
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    monkeypatch.setattr(
        operator, "report", lambda root, cfg: (root / "report.txt").write_text("report")
    )
    sleeps = visibility_clock(monkeypatch)
    if corrupt:
        with pytest.raises(ValueError, match="Input hash changed"):
            operator.control(tmp_path)
    else:
        operator.control(tmp_path)
    assert phases == ["fit", "evaluate"] and verified == [tmp_path / "compute"]
    assert not sleeps
    expected = "FAILED_STOPPED" if corrupt else "REPORT_READY_STOPPED"
    assert read(tmp_path / "controller.json")["status"] == expected


def test_released_artifact_permission_failure_never_waits(tmp_path, monkeypatch):
    sleeps = visibility_clock(monkeypatch)

    def denied(*args):
        raise PermissionError("Permission denied")

    monkeypatch.setattr(operator, "phase_files_visible", denied)
    with pytest.raises(PermissionError):
        operator.wait_phase_files(tmp_path, "fit", {}, {})
    assert not sleeps
