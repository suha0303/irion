"""Portable console policy and submission checks, without SSH/GPU/Slurm work."""

import copy
import errno
import getpass
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from console import cli, control, environment, profile


@pytest.fixture
def machine(tmp_path):
    return {
        "schema": profile.SCHEMA, "name": "test", "project_root": str(profile.ROOT),
        "python": sys.executable, "runtime_profile": "torch21",
        "dataset_root": str(tmp_path / "upstream"), "output_root": str(tmp_path / "results"),
        "ssh": {"host": "gpu.example.org", "user": "tester", "port": 2222},
        "slurm": {"partition": "p02", "node": "DIS01", "nodes": 1,
                  "ntasks": 1, "gpus": 1, "cpus": 4, "memory_gib": 16},
    }


def test_profile_roundtrip_private_no_overwrite(tmp_path, machine):
    path = tmp_path / "profile.json"
    assert profile.write(path, machine) == path
    assert profile.load(path) == machine
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        profile.write(path, machine)


@pytest.mark.parametrize("change", [
    {"password": "forbidden"}, {"python": "python3"}, {"runtime_profile": "latest"},
    {"ssh": {"host": "-oProxyCommand=bad", "user": "tester", "port": 22}},
    {"ssh": {"host": "host", "user": "user", "port": True}},
    {"ssh": {"host": "host", "user": "user", "port": 22, "key": "forbidden"}},
])
def test_profile_rejects_secrets_unsafe_or_unspecified_runtime(machine, change):
    machine.update(change)
    with pytest.raises((ValueError, TypeError)):
        profile.validate(machine)


def test_profile_symlink_rejection_and_explicit_python_link(tmp_path, machine):
    source = tmp_path / "source"
    source.mkdir()
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    machine["dataset_root"] = str(link)
    with pytest.raises(ValueError, match="symlink"):
        profile.validate(machine)
    machine["dataset_root"] = str(source)
    interpreter = tmp_path / "python"
    interpreter.symlink_to(sys.executable)
    machine["python"] = str(interpreter)
    assert profile.validate(machine) is machine


def test_ssh_has_no_credentials_or_hostkey_bypass(machine):
    command = profile.ssh_command(machine)
    assert command == ["ssh", "-o", "StrictHostKeyChecking=yes", "-p", "2222",
                       "tester@gpu.example.org"]


def test_slurm_command_keeps_gpu_visibility_and_one_allocation(tmp_path, machine):
    argv = control.slurm_command(machine, tmp_path)
    for token in ("--partition=p02", "--nodelist=DIS01", "--nodes=1", "--ntasks=1",
                  "--gres=gpu:1", "--no-requeue"):
        assert token in argv
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in argv[-1]
    assert "/compute" in argv[-1]
    alternate = copy.deepcopy(machine)
    alternate["slurm"]["node"] = "DIS05"
    with pytest.raises(ValueError, match="DIS01"):
        control.slurm_command(alternate, tmp_path)


def test_initialize_is_exclusive_and_preserves_profile_binding(tmp_path, machine):
    path = profile.write(tmp_path / "machine.json", machine)
    root = Path(machine["output_root"]) / "smoke01"
    control.initialize(path, root)
    assert profile.load(root / "server-profile.json") == machine
    record = control.read(root / "request.json")
    assert record["profile_sha256"] == profile.digest(root / "server-profile.json")
    assert record["training_authorized"] is False
    assert record["retries"] == 0
    with pytest.raises(FileExistsError):
        control.initialize(path, root)


def test_controller_stops_before_submission_for_existing_job(tmp_path, machine, monkeypatch):
    path = profile.write(tmp_path / "machine.json", machine)
    root = Path(machine["output_root"]) / "busy"
    control.initialize(path, root)
    monkeypatch.setattr(control.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(control, "own_queue", lambda: ["33|RUNNING|tester|p02|DIS01"])
    monkeypatch.setattr(control, "command", lambda args: pytest.fail("Must not submit"))
    with pytest.raises(RuntimeError, match="pending/running"):
        control.controller(root)
    assert control.read(root / "control.json")["status"] == "FAILED_STOPPED"
    with pytest.raises(FileExistsError):
        control.controller(root)


def test_controller_success_requires_accounting_and_released_allocation(tmp_path, machine, monkeypatch):
    path = profile.write(tmp_path / "machine.json", machine)
    root = Path(machine["output_root"]) / "complete"
    control.initialize(path, root)
    (root / "compute").mkdir()
    control.write(root / "compute/result.json", {
        "status": "SMOKE_COMPUTE_FINISHED_PENDING_RELEASE", "training_eligible": False
    })
    monkeypatch.setattr(control.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(control, "own_queue", lambda: [])
    monkeypatch.setattr(control, "job_description", lambda job, owner: {})
    monkeypatch.setattr(control, "accounting", lambda job, owner: {
        "state": "COMPLETED", "exit_code": "0:0"
    })
    commands = []

    def fake(args):
        commands.append(args)
        return "123" if args[0] == "sbatch" else ""

    monkeypatch.setattr(control, "command", fake)
    control.controller(root)
    state = control.read(root / "control.json")
    assert state["status"] == "SMOKE_READY_STOPPED"
    assert state["allocation_released"] is True
    assert state["submitted_jobs"] == ["123"]
    assert sum(args[0] == "sbatch" for args in commands) == 1


def test_environment_uses_metadata_without_cuda_or_install(machine, monkeypatch):
    calls = []

    def version(name):
        calls.append(name)
        return profile.RUNTIME_PROFILES["torch21"].get(name, "1.0")

    monkeypatch.setattr(environment.importlib.metadata, "version", version)
    result = environment.diagnose(machine)
    assert set(calls) == {"torch", "numpy", "safetensors", "PyYAML", "jsonschema", "pyulog"}
    assert result["gpu_operations"] == 0
    assert result["submitted_jobs"] == []
    assert result["installation_performed"] is False
    assert result["training_started"] is False


def test_bootstrap_help_does_not_require_ai_environment():
    result = subprocess.run([str(profile.ROOT / "ai/drone-ai"), "--help"],
                            text=True, capture_output=True, check=True)
    assert "configure" in result.stdout and "smoke" in result.stdout


def test_configure_noninteractive_and_status_without_profile(tmp_path, capsys):
    destination = tmp_path / "profile.json"
    assert cli.main([
        "configure", "--non-interactive", "--profile", str(destination), "--name", "test",
        "--python", sys.executable, "--dataset-root", str(tmp_path / "dataset"),
        "--output-root", str(tmp_path / "output")
    ]) == 0
    assert profile.load(destination)["ssh"] is None
    root = tmp_path / "results"
    root.mkdir()
    control.write(root / "control.json", {"status": "FAILED_STOPPED"})
    assert cli.main(["status", "--root", str(root), "--json"]) == 0
    assert "FAILED_STOPPED" in capsys.readouterr().out


def test_unknown_submission_never_reports_stopped_or_retries(tmp_path, machine, monkeypatch):
    path = profile.write(tmp_path / "machine.json", machine)
    root = Path(machine["output_root"]) / "unknown"
    control.initialize(path, root)
    monkeypatch.setattr(control.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(control, "own_queue", lambda: [])
    attempts = []

    def fake(args):
        attempts.append(args)
        raise subprocess.TimeoutExpired(args, 30)

    monkeypatch.setattr(control, "command", fake)
    with pytest.raises(subprocess.TimeoutExpired):
        control.controller(root)
    state = control.read(root / "control.json")
    assert state["status"] == "SMOKE_ERROR_REQUIRES_REVIEW"
    assert state["submission_attempted"] is True
    assert "allocation_released" not in state
    assert len(attempts) == 1


def test_changed_submission_profile_is_rejected_before_any_commands(tmp_path, machine, monkeypatch):
    path = profile.write(tmp_path / "machine.json", machine)
    root = Path(machine["output_root"]) / "changed"
    control.initialize(path, root)
    profile.write(root / "server-profile.json", {**machine, "name": "changed"}, replace=True)
    monkeypatch.setattr(control, "command", lambda args: pytest.fail("Must not execute"))
    with pytest.raises(ValueError, match="snapshot changed"):
        control.controller(root)
    assert not (root / "controller.claim").exists()


def test_failed_accounting_is_preserved_and_confirmed_released(tmp_path, machine, monkeypatch):
    path = profile.write(tmp_path / "machine.json", machine)
    root = Path(machine["output_root"]) / "failed"
    control.initialize(path, root)
    monkeypatch.setattr(control.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(control, "own_queue", lambda: [])
    monkeypatch.setattr(control, "job_description", lambda job, owner: {})
    monkeypatch.setattr(control, "accounting", lambda job, owner: {
        "state": "FAILED", "exit_code": "1:0"
    })
    monkeypatch.setattr(control, "command", lambda args: "456" if args[0] == "sbatch" else "")
    with pytest.raises(RuntimeError, match="GPU smoke failed"):
        control.controller(root)
    state = control.read(root / "control.json")
    assert state["status"] == "FAILED_STOPPED"
    assert state["allocation_released"] is True
    assert state["submitted_jobs"] == ["456"]
    assert state["exit_code"] == "1:0"


def test_accounting_refuses_another_owner(monkeypatch):
    monkeypatch.setattr(control, "command", lambda args: "999|someone_else|COMPLETED|0:0|")
    with pytest.raises(ValueError, match="owner mismatch"):
        control.accounting("999", "tester")


def test_missing_optional_parser_packages_do_not_block_npz_smoke(tmp_path, machine, monkeypatch):
    dataset = Path(machine["dataset_root"])
    (dataset / "dataset").mkdir(parents=True)
    for name in ("dataset/normal_train.npz", "dataset/scaler.npz", "config.json", "inventory.json"):
        (dataset / name).touch()

    def version(name):
        if name not in profile.RUNTIME_PROFILES["torch21"]:
            raise importlib.metadata.PackageNotFoundError(name)
        return profile.RUNTIME_PROFILES["torch21"][name]

    monkeypatch.setattr(environment.importlib.metadata, "version", version)
    monkeypatch.setattr(environment.shutil, "which", lambda command: "/usr/bin/" + command)
    result = environment.diagnose(machine)
    assert result["status"] == "READY_FOR_GPU_SMOKE"
    assert all(value is None for value in result[
        "optional_raw_parser_packages_not_used_by_prepared_npz_path"
    ].values())


def test_current_flat_preprocessing_output_presence_supports_functionality_gate(tmp_path, machine, monkeypatch):
    dataset = Path(machine["dataset_root"])
    (dataset / "dataset").mkdir(parents=True)
    for name in ("dataset/normal_train.npz", "dataset/scaler.npz", "handoff.json"):
        (dataset / name).touch()
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name:
                        profile.RUNTIME_PROFILES["torch21"].get(name, "1.0"))
    monkeypatch.setattr(environment.shutil, "which", lambda command: "/usr/bin/" + command)
    result = environment.diagnose(machine)
    assert result["status"] == "READY_FOR_GPU_SMOKE"
    presence = next(c for c in result["checks"] if c["name"] == "dataset")
    assert presence["detail"]["format"] == "current_prepared_flat"
    assert "do not authorize" in presence["detail"]["main_training_admission"]


def test_train_prepare_only_delegates_explicit_command_and_profile(tmp_path, machine, monkeypatch):
    path = profile.write(tmp_path / "machine.json", machine)
    calls = []

    def fake(settings, module, arguments, **kwargs):
        calls.append((module, arguments))
        return 0

    monkeypatch.setattr(cli, "selected_python", fake)
    assert cli.main(["train", "--profile", str(path), "prepare", "--root", str(tmp_path / "prep")]) == 0
    assert calls == [("learning.prepare_compare", [
        "prepare", "--root", str(tmp_path / "prep"), "--profile", str(path)
    ])]


def test_explicit_selected_profile_overrides_inherited_environment(tmp_path, machine, monkeypatch):
    inherited = tmp_path / "profile-a.json"
    selected = profile.write(tmp_path / "profile-b.json", machine)
    monkeypatch.setenv("DRONE_AI_PROFILE", str(inherited))
    calls = []

    def fake(argv, **kwargs):
        calls.append(kwargs["env"]["DRONE_AI_PROFILE"])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake)
    assert cli.main(["train", "--profile", str(selected), "status", "--root", str(tmp_path)]) == 0
    assert calls == [str(selected)]
    assert "DRONE_AI_PROFILE" not in cli.environment(machine)


def test_status_reports_compute_success_as_pending_until_controller_release(tmp_path):
    (tmp_path / "compute").mkdir()
    (tmp_path / "compute/result.json").write_text(json.dumps({
        "status": "SMOKE_COMPUTE_FINISHED_PENDING_RELEASE", "training_eligible": False
    }))
    value = cli.snapshot(tmp_path)
    assert "control.json" not in value["records"]
    assert value["records"]["compute/result.json"]["status"] != "SMOKE_READY_STOPPED"


def test_unified_watch_renders_real_main_fit_model_epoch_and_gpu(tmp_path, capsys):
    (tmp_path / "input").mkdir()
    (tmp_path / "compute").mkdir()
    control.write(tmp_path / "controller.json", {"status": "RUNNING", "stage": "fit", "job_id": "77"})
    control.write(tmp_path / "input/config.json", {
        "models": list(range(10)), "seed": 0, "batch_size": 256,
    })
    control.write(tmp_path / "compute/progress.json", {
        "stage": "fit", "model": "lstm_h128_l2_s0", "epoch": 3, "epochs": 60,
        "batch": 20, "batches": 100, "done": 20, "total": 100,
        "models_completed": ["persistence"], "train_loss": 0.02,
    })
    control.write(tmp_path / "compute/gpu-live.json", {
        "status": "SAMPLED", "utilization_pct": 80, "used_mib": 2000, "total_mib": 48000,
        "timestamp": cli.time.time(),
    })
    cli.display(cli.snapshot(tmp_path))
    result = capsys.readouterr().out
    for expected in ("RUNNING", "77", "10종", "lstm_h128_l2_s0", "epoch 3/60", "batch 20/100",
                     "80%", "2000/48000", "20.0%", "False"):
        assert expected in result


def test_unified_watch_uses_evaluator_progress_and_confirms_both_releases(tmp_path, capsys):
    control.write(tmp_path / "controller.json", {"status": "RUNNING", "stage": "evaluate"})
    (tmp_path / "compute").mkdir()
    control.write(tmp_path / "compute/progress.json", {"stage": "stale_fit", "done": 10, "total": 10})
    control.write(tmp_path / "evaluation-progress.json", {"stage": "independent_evaluation", "done": 3, "total": 10})
    cli.display(cli.snapshot(tmp_path))
    result = capsys.readouterr().out
    assert "independent_evaluation" in result and "30.0%" in result
    assert "stale_fit" not in result and "반납 확인: False" in result
    control.write(tmp_path / "completion.json", {
        "status": "REPORT_READY_STOPPED", "fit_release": {"released": True},
        "evaluation_release": {"released": True},
    })
    cli.display(cli.snapshot(tmp_path))
    assert "반납 확인: True" in capsys.readouterr().out


def allocated_job(job_id="19348", *, state="RUNNING", nodes="1"):
    return {
        "JobId": job_id, "JobState": state,
        "UserId": f"{getpass.getuser()}({os.getuid()})", "Partition": "p02",
        "NumNodes": nodes, "NumTasks": "1", "ReqNodeList": "DIS01",
        "NodeList": "(null)" if state == "PENDING" else "DIS01",
        "TRES": "cpu=4,mem=16G,node=1,gres/gpu=1",
    }


@pytest.mark.parametrize("nodes,state,accepted", [
    ("1-1", "PENDING", True), ("1", "PENDING", True), ("1", "RUNNING", True),
    ("1-2", "PENDING", False), ("1-1", "RUNNING", False), ("0-1", "PENDING", False),
])
def test_submission_one_node_pending_range_and_running_count(nodes, state, accepted, monkeypatch):
    fields = allocated_job(state=state, nodes=nodes)
    monkeypatch.setattr(control, "command", lambda args: " ".join(
        key + "=" + value for key, value in fields.items()
    ))
    if accepted:
        assert control.job_description("19348", getpass.getuser()) == fields
    else:
        with pytest.raises(ValueError, match="NumNodes") as error:
            control.job_description("19348", getpass.getuser())
        assert nodes in str(error.value)


@pytest.mark.parametrize("field,value", [
    ("NumTasks", "2"), ("ReqNodeList", "DIS05"), ("TRES", "cpu=4,gres/gpu=2"),
])
def test_pending_node_range_does_not_relax_other_resource_checks(field, value):
    fields = allocated_job(state="PENDING", nodes="1-1")
    fields[field] = value
    with pytest.raises(ValueError) as error:
        control.validate_job_description(fields, "19348", getpass.getuser())
    assert field in str(error.value) and value in str(error.value)


def finished_smoke(root, machine, tmp_path):
    selected = profile.write(tmp_path / "profile.json", machine)
    control.initialize(selected, root)
    control.write(root / "control.json", {
        "status": "SMOKE_ERROR_REQUIRES_REVIEW", "owner": getpass.getuser(),
        "job_id": "19348", "submitted_jobs": ["19348"], "submission_attempted": True,
        "reason": "Original pending resource mismatch",
    })
    compute = root / "compute"
    compute.mkdir()
    (compute / "inputs.npz").write_bytes(b"bounded input fixture")
    control.write(compute / "smoke.json", {
        "device": "cuda", "profile_sha256": profile.digest(root / "server-profile.json"),
        "inputs_sha256": control.file_sha(compute / "inputs.npz"),
    })
    (compute / "verified-job-id.txt").write_text("19348\n")
    control.write(compute / "environment.json", {"job": allocated_job()})
    check_names = {
        "model", "optimizer", "rng", "best", "best_epoch", "best_state", "next_epoch",
        "config", "binding", "environment", "loss_history", "safetensors_export",
        "completed_two_epochs", "optimizer_executed_four_steps", "finite_losses",
    }
    proofs = {}
    for name in ("lstm", "transformer"):
        for phase in ("full", "resume"):
            folder = compute / f"{name}-{phase}"
            folder.mkdir()
            (folder / "last.pt").write_bytes(b"unchanged checkpoint fixture")
        proofs[name] = {
            "checks": {key: True for key in check_names}, "optimizer_steps_executed": 8,
            "full_checkpoint_sha256": control.file_sha(compute / f"{name}-full/last.pt"),
            "resumed_checkpoint_sha256": control.file_sha(compute / f"{name}-resume/last.pt"),
        }
    control.write(compute / "restore-proof.json", proofs)
    control.write(compute / "result.json", {
        "status": "SMOKE_COMPUTE_FINISHED_PENDING_RELEASE", "passed": True,
        "production_training_authorized": False, "training_eligible": False,
        "final_test_used": False, "truth_used": False, "automatic_retry": False,
        "llm_calls": 0, "optimizer_steps_total": 16, "proofs": proofs,
    })
    return root


def test_reconcile_existing_job_only_preserves_error_and_drives_watch(tmp_path, machine, monkeypatch, capsys):
    root = finished_smoke(Path(machine["output_root"]) / "complete", machine, tmp_path)
    before = (root / "control.json").read_bytes()
    commands = []

    def fake(args):
        commands.append(args[0])
        if args[0] == "sacct":
            return f"19348|{getpass.getuser()}|COMPLETED|0:0|"
        assert args[0] == "squeue"
        return ""

    monkeypatch.setattr(control, "command", fake)
    assert cli.main(["reconcile", "--root", str(root)]) == 0
    assert (root / "control.json").read_bytes() == before
    assert commands == ["sacct", "squeue"]
    result = control.read(root / "reconciliation.json")
    assert result["new_jobs_submitted"] == [] and result["allocation_released"] is True
    cli.display(cli.snapshot(root))
    assert "기존 종료 작업의 검산 결과" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        control.reconcile(root)
    control.write(root / "control.json", {"status": "changed"})
    assert cli.snapshot(root)["records"]["reconciliation.json"]["binding_valid"] is False


@pytest.mark.parametrize("failure", ["active_queue", "failed_accounting", "proof_changed", "profile_changed"])
def test_reconcile_never_starts_or_accepts_unproven_work(tmp_path, machine, monkeypatch, failure):
    root = finished_smoke(Path(machine["output_root"]) / "reject", machine, tmp_path)
    before = (root / "control.json").read_bytes()
    monkeypatch.setattr(control, "accounting", lambda job, owner: {
        "state": "FAILED" if failure == "failed_accounting" else "COMPLETED", "exit_code": "0:0"
    })
    monkeypatch.setattr(control, "own_queue", lambda: ["active"] if failure == "active_queue" else [])
    monkeypatch.setattr(control, "command", lambda args: pytest.fail("No submit/restart permitted"))
    if failure == "proof_changed":
        (root / "compute/lstm-resume/last.pt").write_bytes(b"changed")
    if failure == "profile_changed":
        profile.write(root / "server-profile.json", {**machine, "name": "changed"}, replace=True)
    with pytest.raises(ValueError):
        control.reconcile(root)
    assert (root / "control.json").read_bytes() == before
    assert not (root / "reconciliation.json").exists()


@pytest.mark.parametrize("visibility", ["eventual", "timeout", "malformed", "permission"])
def test_result_visibility_wait_never_resubmits_or_retries_gpu(tmp_path, machine, monkeypatch, visibility):
    selected = profile.write(tmp_path / "machine.json", machine)
    root = Path(machine["output_root"]) / "visibility"
    control.initialize(selected, root)
    (root / "compute").mkdir()
    control.write(root / "compute/result.json", {
        "status": "SMOKE_COMPUTE_FINISHED_PENDING_RELEASE", "training_eligible": False
    })
    monkeypatch.setattr(control.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(control, "own_queue", lambda: [])
    monkeypatch.setattr(control, "job_description", lambda job, owner: {})
    monkeypatch.setattr(control, "accounting", lambda job, owner: {
        "state": "COMPLETED", "exit_code": "0:0"
    })
    submitted = []

    def command(args):
        submitted.append(args[0])
        return "789" if args[0] == "sbatch" else ""

    monkeypatch.setattr(control, "command", command)
    clock, sleeps = {"now": 0}, []
    monkeypatch.setattr(control.time, "monotonic", lambda: clock["now"])

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(control.time, "sleep", sleep)
    original_read = control.read
    reads = []

    def simulated_read(path):
        if Path(path) == root / "compute/result.json":
            reads.append(path)
            if visibility == "timeout" or visibility == "eventual" and len(reads) < 3:
                raise FileNotFoundError(errno.ENOENT, "NFS result not yet visible", str(path))
            if visibility == "malformed":
                raise json.JSONDecodeError("Malformed result", "{", 1)
            if visibility == "permission":
                raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return original_read(path)

    monkeypatch.setattr(control, "read", simulated_read)
    original_write, waiting_states = control.write, []

    def record_write(path, value):
        if Path(path) == root / "control.json" and value.get("status") == "SMOKE_WAITING_RESULT_VISIBILITY":
            waiting_states.append(dict(value))
        return original_write(path, value)

    monkeypatch.setattr(control, "write", record_write)
    if visibility == "eventual":
        control.controller(root)
    else:
        expected = {"timeout": TimeoutError, "malformed": json.JSONDecodeError,
                    "permission": PermissionError}[visibility]
        with pytest.raises(expected):
            control.controller(root)
    state = original_read(root / "control.json")
    assert submitted.count("sbatch") == 1
    assert "scancel" not in submitted
    assert state["submitted_jobs"] == ["789"] and state["allocation_released"] is True
    if visibility == "eventual":
        assert sleeps == [5, 5] and len(reads) == 3
        assert len(waiting_states) == 2 and all(s["allocation_released"] is True for s in waiting_states)
        assert state["status"] == "SMOKE_READY_STOPPED"
        assert state["result_visibility_limit_s"] == 120
        assert "NFS result not yet visible" in state["result_visibility_last_error"]
    elif visibility == "timeout":
        assert sum(sleeps) == 120 and len(reads) == 25
        assert waiting_states[-1]["result_visibility_waited_s"] == 120
        assert state["status"] == "FAILED_STOPPED"
        assert "invisible for 120s" in state["reason"]
        assert state["result_visibility_waited_s"] == 120
    else:
        assert not sleeps and len(reads) == 1
        assert not waiting_states
        assert state["status"] == "FAILED_STOPPED"
        assert "result_visibility_last_error" not in state
