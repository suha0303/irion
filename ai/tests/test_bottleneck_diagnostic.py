"""Only synthetic CPU work; no server submission or GPU operations."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from execution.common import read, sha, write
from learning.bottleneck import profiled_step
from learning.diagnostic import same
from learning.models import StudySpec, make_model
from learning.optimization import make_optimizer, policy_for
from learning.training import backend, safe_step


@pytest.mark.parametrize("name", ["lstm", "transformer"])
@pytest.mark.parametrize("kind", ["sgd", "adam"])
def test_profiled_work_matches_production_step(name, kind):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    spec = StudySpec(name, 8)
    original, annotated = make_model(spec), make_model(spec)
    annotated.load_state_dict(original.state_dict())
    policy = policy_for(kind)
    left = make_optimizer(original.parameters(), 0.001, policy)
    right = make_optimizer(annotated.parameters(), 0.001, policy)
    inputs = (torch.randn(2, 100, 19), torch.randn(2, 6), torch.randn(2, 10, 4))
    with backend("cpu"):
        value = safe_step(original, left, *inputs)
        measured = profiled_step(annotated, right, *inputs)
    assert same(value, measured)
    assert same(original.state_dict(), annotated.state_dict())
    assert same(left.state_dict(), right.state_dict())


@pytest.fixture(name="controller")
def fixture_controller(tmp_path, monkeypatch):
    script = (
        Path(__file__).resolve().parents[1]
        / "reports/bottleneck-20261101/attempt-01/operator/control.py"
    )
    spec = importlib.util.spec_from_file_location(
        "diagnostic_controller", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    write(
        tmp_path / "input/config.json",
        dict(
            python=sys.executable,
            source=str(tmp_path / "source"),
            wall_hours=4,
            queue_hours=24,
        ),
    )
    write(
        tmp_path / "authorization.json",
        dict(
            config_sha256=sha(tmp_path / "input/config.json"),
            gpu_diagnostic_authorized=True,
            main_training_authorized=False,
        ),
    )
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(module, "SUBMIT_LOCK", tmp_path / "submit.lock")
    monkeypatch.setattr(module, "GPU_LOCK", tmp_path / "gpu.lock")
    return module


def test_controller_stops_before_duplicate_submission(
    controller, tmp_path, monkeypatch
):
    monkeypatch.setattr(controller.operator, "own_queue", lambda: "other-job")
    monkeypatch.setattr(
        controller.operator, "command", lambda *a: pytest.fail("No submission")
    )
    with pytest.raises(RuntimeError, match="pending/running"):
        controller.control(tmp_path)
    assert not (tmp_path / "benchmark-submitted.claim").exists()


def test_controller_never_retries_unknown_submission(
    controller, tmp_path, monkeypatch
):
    monkeypatch.setattr(controller.operator, "own_queue", lambda: "")
    commands = []
    monkeypatch.setattr(
        controller.operator,
        "command",
        lambda args: commands.append(args) or "UNKNOWN",
    )
    with pytest.raises(RuntimeError, match="Unknown submission"):
        controller.control(tmp_path)
    assert len(commands) == 1
    assert (
        read(tmp_path / "controller.json")["status"]
        == "CONTROLLER_STOPPED_JOB_STATE_UNCONFIRMED"
    )
    with pytest.raises(FileExistsError):
        controller.control(tmp_path)
    assert len(commands) == 1


def test_controller_records_completion_only_after_release(
    controller, tmp_path, monkeypatch
):
    monkeypatch.setattr(controller.operator, "own_queue", lambda: "")
    commands = []
    monkeypatch.setattr(
        controller.operator,
        "command",
        lambda args: commands.append(args) or "123",
    )
    monkeypatch.setattr(
        controller.operator,
        "wait_release",
        lambda *a: dict(
            job_id="123",
            released=True,
            slurm_state="COMPLETED",
            exit_code="0:0",
            remaining_own_queue="",
        ),
    )
    write(
        tmp_path / "compute/result.json",
        dict(
            status="DIAGNOSTIC_COMPLETE_PENDING_RELEASE",
            all16_measured=True,
            main_training_started=False,
        ),
    )
    controller.control(tmp_path)
    assert (
        read(tmp_path / "completion.json")["status"]
        == "DIAGNOSTIC_READY_STOPPED"
    )
    for flag in (
        "--partition=p02",
        "--nodelist=DIS01",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:1",
        "--no-requeue",
    ):
        assert flag in commands[0]
