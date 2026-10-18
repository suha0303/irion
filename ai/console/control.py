"""Finite tmux/Slurm smoke controller. No LLM calls or automatic retries."""

import argparse
import contextlib
import errno
import fcntl
import getpass
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .profile import ROOT, absolute_path, digest, load
from .profile import write as write_profile

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def command(args):
    return subprocess.run(args, text=True, capture_output=True, check=True, timeout=30).stdout.strip()


def approved_resources(profile):
    resources = profile["slurm"]
    if resources["partition"] != "p02" or resources["node"].upper() != "DIS01":
        raise ValueError("Current authorization permits p02 / DIS01 only")
    return resources


def worker_command(profile, root):
    return [profile["python"], "-B", "-m", "console.smoke", "--profile",
            str(root / "server-profile.json"), "--root", str(root / "compute")]


def slurm_command(profile, root):
    resources = approved_resources(profile)
    prefix = ":".join(str(ROOT / part) for part in ("ai", "ai/src", "Data/generation"))
    wrapped = "export PYTHONPATH=" + shlex.quote(prefix)
    wrapped += "; export DRONE_AI_PROFILE=" + shlex.quote(str(root / "server-profile.json"))
    wrapped += "; export CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1"
    wrapped += "; exec " + shlex.join(
        worker_command(profile, root)
    )
    return ["sbatch", "--parsable", "--partition=p02", "--nodelist=DIS01", "--nodes=1",
            "--ntasks=1", "--gres=gpu:1", "--cpus-per-task=" + str(resources["cpus"]),
            "--mem=" + str(resources["memory_gib"]) + "G", "--time=00:20:00", "--no-requeue",
            "--job-name=drone-ai-benchmark", "--output=" + str(root / "slurm.log"),
            "--error=" + str(root / "slurm.log"), "--chdir=" + str(ROOT), "--wrap", wrapped]


def initialize(profile_file, root):
    profile = load(profile_file)
    approved_resources(profile)
    if absolute_path(profile["project_root"]) != ROOT:
        raise ValueError("Run on the configured server clone, or use --remote")
    root = absolute_path(root)
    output = absolute_path(profile["output_root"])
    dataset = absolute_path(profile["dataset_root"])
    if root == output or not root.is_relative_to(output):
        raise ValueError("Use a new attempt directory below the configured output_root")
    if root.is_relative_to(dataset) or dataset.is_relative_to(root):
        raise ValueError("Smoke output and immutable dataset must be separate")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    snapshot = write_profile(root / "server-profile.json", profile)
    write(root / "request.json", {
        "status": "SMOKE_REQUESTED_NOT_SUBMITTED", "created": time.time(),
        "profile_sha256": digest(snapshot), "source_profile_sha256": digest(profile_file),
        "training_authorized": False, "final_test_used": False, "retries": 0,
        "llm_calls": 0, "python": profile["python"], "main_training_started": False,
        "scope": "Finite correctness test only; short optimizer/save/restore steps are allowed",
    })
    return root, profile


def start(profile_file, root):
    root, profile = initialize(profile_file, root)
    session = "drone-ai-smoke-" + root.name
    # tmux owns the controller; closing the launcher, SSH, or watch screen does not stop it.
    body = "export PYTHONPATH=" + shlex.quote(os.environ.get("PYTHONPATH", str(ROOT / "ai")))
    body += "; exec " + shlex.join([
        sys.executable, "-B", "-m", "console.control", "--root", str(root)
    ]) + " >> " + shlex.quote(str(root / "controller.log")) + " 2>&1"
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, body], check=True)
    except BaseException as exc:
        write(root / "control.json", {
            "status": "FAILED_STOPPED", "reason": f"{type(exc).__name__}: {exc}",
            "job_id": None, "submitted_jobs": [],
            "main_training_started": False,
        })
        raise
    return {"status": "SMOKE_CONTROLLER_STARTED", "tmux_session": session,
            "root": str(root), "profile": profile["name"], "main_training_started": False}


def own_queue():
    text = command(["squeue", "--me", "--noheader", "--format=%i|%T|%u|%P|%N"])
    return [line for line in text.splitlines() if line.strip()]


def accounting(job_id, owner):
    text = command(["sacct", "-n", "-P", "-j", job_id,
                    "--format=JobIDRaw,User,State,ExitCode"])
    for line in text.splitlines():
        parts = line.split("|")
        if parts[0] == job_id:
            if len(parts) < 4 or parts[1] != owner:
                raise ValueError("Accounting job owner mismatch")
            return {"state": parts[2].split()[0].rstrip("+"), "exit_code": parts[3]}
    return None


def job_description(job_id, owner):
    text = command(["scontrol", "show", "job", "-o", job_id])
    fields = dict(token.split("=", 1) for token in shlex.split(text) if "=" in token)
    return validate_job_description(fields, job_id, owner)


def validate_job_description(fields, job_id, owner):
    relevant = {key: fields.get(key) for key in (
        "JobId", "JobState", "UserId", "Partition", "NumNodes", "NumTasks",
        "ReqNodeList", "NodeList", "ReqTRES", "TRES", "AllocTRES", "Gres",
    )}

    def fail(reason):
        raise ValueError(reason + ": " + json.dumps(relevant, sort_keys=True))

    if fields.get("JobId") != job_id or not fields.get("UserId", "").startswith(owner + "("):
        fail("Submitted job ID / owner mismatch")
    node_count = fields.get("NumNodes")
    one_node = node_count == "1" or (
        fields.get("JobState") == "PENDING" and node_count == "1-1"
    )
    if fields.get("Partition") != "p02" or not one_node or fields.get("NumTasks") != "1":
        fail("Submitted job resources differ from authorized resources")
    if fields.get("ReqNodeList", "").upper() != "DIS01":
        fail("Requested node differs from DIS01")
    if fields.get("JobState") == "RUNNING" and fields.get("NodeList", "").upper() != "DIS01":
        fail("Allocated node differs from DIS01")
    requested = fields.get("ReqTRES") or fields.get("TRES", "")
    resources = dict(item.split("=", 1) for item in requested.split(",") if "=" in item)
    typed = [value for key, value in resources.items() if key.startswith("gres/gpu:")]
    if resources.get("gres/gpu") != "1" or (typed and (len(typed) != 1 or typed[0] != "1")):
        fail("Requested GPU count differs from one")
    return fields


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def released_result(path, update, *, visibility_seconds=120, interval_seconds=5):
    """After confirmed Slurm release, tolerate only transient ENOENT visibility."""
    started = time.monotonic()
    deadline = started + visibility_seconds
    while True:
        try:
            return read(path)
        except FileNotFoundError as exc:
            if exc.errno != errno.ENOENT:
                raise
            elapsed = time.monotonic() - started
            update(
                status="SMOKE_WAITING_RESULT_VISIBILITY", allocation_released=True,
                slurm_state="COMPLETED", exit_code="0:0",
                result_visibility_path=str(path), result_visibility_waited_s=elapsed,
                result_visibility_limit_s=visibility_seconds,
                result_visibility_last_error=f"{type(exc).__name__}: {exc}",
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Released job result stayed invisible for {visibility_seconds}s: {path}; "
                    f"last error: {exc}"
                ) from exc
            time.sleep(min(interval_seconds, remaining))


def reconcile(root):
    """Inspect one existing terminated job, preserving its original controller evidence."""
    root = absolute_path(root)
    if (root / "reconciliation.json").exists():
        raise FileExistsError("A reconciliation already exists; never rewrite historical evidence")
    profile_file = root / "server-profile.json"
    settings = load(profile_file)
    approved_resources(settings)
    request = read(root / "request.json")
    if digest(profile_file) != request.get("profile_sha256") or (
        request.get("training_authorized") is not False
        or request.get("final_test_used") is not False
        or request.get("main_training_started") is not False
    ):
        raise ValueError("Original smoke request/profile binding or scope changed")
    original = read(root / "control.json")
    owner, job_id = original.get("owner"), original.get("job_id")
    if owner != getpass.getuser() or not isinstance(job_id, str) or not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("Existing numeric job ID and current account owner are required")
    if original.get("submitted_jobs") != [job_id] or original.get("submission_attempted") is not True:
        raise ValueError("Original single submitted-job receipt is required")
    record = accounting(job_id, owner)
    if record != {"state": "COMPLETED", "exit_code": "0:0"}:
        raise ValueError("Existing job is not accounting-confirmed COMPLETED/0:0: " + str(record))
    compute = root / "compute"
    result = read(compute / "result.json")
    expected = {
        "status": "SMOKE_COMPUTE_FINISHED_PENDING_RELEASE", "passed": True,
        "production_training_authorized": False, "training_eligible": False,
        "final_test_used": False, "truth_used": False, "automatic_retry": False,
        "llm_calls": 0, "optimizer_steps_total": 16,
    }
    if any(type(result.get(key)) is not type(value) or result[key] != value
           for key, value in expected.items()):
        raise ValueError("Successful bounded GPU smoke result is required")
    cfg = read(compute / "smoke.json")
    if cfg.get("profile_sha256") != digest(profile_file) or cfg.get("device") != "cuda" or (
        cfg.get("inputs_sha256") != file_sha(compute / "inputs.npz")
    ):
        raise ValueError("Smoke input/profile binding changed")
    if (compute / "verified-job-id.txt").read_text().strip() != job_id:
        raise ValueError("Worker allocation evidence differs from the submitted job")
    worker_environment = read(compute / "environment.json")
    evidence = worker_environment.get("job", {})
    validate_job_description(evidence, job_id, owner)
    if evidence.get("JobState") != "RUNNING" or evidence.get("UserId", "").split("(")[-1].rstrip(")") != str(os.getuid()):
        raise ValueError("Worker did not verify this account's running allocation")
    proofs = read(compute / "restore-proof.json")
    if result.get("proofs") != proofs or set(proofs) != {"lstm", "transformer"}:
        raise ValueError("Saved restore proof differs from the result")
    expected_checks = {
        "model", "optimizer", "rng", "best", "best_epoch", "best_state", "next_epoch",
        "config", "binding", "environment", "loss_history", "safetensors_export",
        "completed_two_epochs", "optimizer_executed_four_steps", "finite_losses",
    }
    for model, proof in proofs.items():
        checks = proof.get("checks", {})
        if set(checks) != expected_checks or any(value is not True for value in checks.values()) or (
            proof.get("optimizer_steps_executed") != 8
        ):
            raise ValueError("Incomplete/failed save/restore proof for " + model)
        for phase, field in (("full", "full_checkpoint_sha256"), ("resume", "resumed_checkpoint_sha256")):
            if file_sha(compute / f"{model}-{phase}/last.pt") != proof.get(field):
                raise ValueError("Smoke checkpoint changed: " + model + " / " + phase)
    if own_queue():
        raise ValueError("Account still has pending/running jobs; release is not confirmed")
    reconciliation = {
        "status": "SMOKE_READY_STOPPED", "reconciled": time.time(), "owner": owner,
        "job_id": job_id, "submitted_jobs": [job_id], "slurm_state": "COMPLETED",
        "exit_code": "0:0", "allocation_released": True, "main_training_started": False,
        "new_jobs_submitted": [], "automatic_retry": False, "llm_calls": 0,
        "original_control_sha256": file_sha(root / "control.json"),
        "request_sha256": file_sha(root / "request.json"),
        "profile_sha256": digest(profile_file), "result_sha256": file_sha(compute / "result.json"),
        "restore_proof_sha256": file_sha(compute / "restore-proof.json"),
        "scope": "Read-only Slurm reconciliation of one existing smoke; no GPU operation or retry",
    }
    # Exclusive write preserves both original control.json and any previous reconciliation.
    with (root / "reconciliation.json").open("x") as stream:
        json.dump(reconciliation, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return reconciliation


def controller(root):
    root = absolute_path(root)
    profile_file = root / "server-profile.json"
    request = read(root / "request.json")
    if digest(profile_file) != request["profile_sha256"]:
        raise ValueError("Submitted profile snapshot changed")
    profile = load(profile_file)
    if Path(profile["project_root"]) != ROOT:
        raise ValueError("Controller imported from a different clone")
    descriptor = os.open(root / "controller.claim", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()) + "\n")
    state = {"status": "SMOKE_PREFLIGHT", "pid": os.getpid(), "owner": getpass.getuser(),
             "started": time.time(), "job_id": None, "submitted_jobs": [],
             "final_test_used": False, "llm_calls": 0, "retries": 0,
             "submission_attempted": False, "main_training_started": False,
             "scope": "Short GPU optimizer/save/restore correctness test; no main training"}

    def update(**fields):
        state.update(fields, updated=time.time())
        write(root / "control.json", state)

    update()
    lock = None
    try:
        lock = (Path.home() / ".drone-ai-submit.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if own_queue():
            raise RuntimeError("Account has pending/running jobs; no duplicate GPU submission")
        # Recheck immutable request immediately before the single sbatch call.
        if digest(profile_file) != request["profile_sha256"]:
            raise ValueError("Profile changed before submission")
        update(submission_attempted=True, status="SMOKE_SUBMISSION_REQUESTED")
        result = command(slurm_command(profile, root))
        job_id = result.split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+", job_id):
            raise ValueError("Unrecognized sbatch job ID; no retry")
        update(status="SMOKE_SUBMITTED", job_id=job_id, submitted_jobs=[job_id])
        job_description(job_id, state["owner"])
        deadline = time.monotonic() + 2 * 3600
        # This finite program waits for its single job; no LLM API, callback or auto-retry exists.
        while True:
            record = accounting(job_id, state["owner"])
            queue = command(["squeue", "-h", "-j", job_id, "-o", "%i|%T|%u|%P|%N"])
            if queue:
                parts = queue.splitlines()[0].split("|")
                if len(parts) != 5 or parts[0] != job_id or parts[2] != state["owner"]:
                    raise ValueError("Active job identity changed")
                update(status="SMOKE_RUNNING" if parts[1] == "RUNNING" else "SMOKE_QUEUED",
                       slurm_state=parts[1], node=parts[4])
            elif record and record["state"] in TERMINAL:
                if record != {"state": "COMPLETED", "exit_code": "0:0"}:
                    raise RuntimeError("GPU smoke failed: " + str(record))
                result = released_result(root / "compute/result.json", update)
                if result.get("status") != "SMOKE_COMPUTE_FINISHED_PENDING_RELEASE" or (
                    result.get("training_eligible") is not False
                ):
                    raise ValueError("Smoke computation result is absent or unsuccessful")
                update(status="SMOKE_READY_STOPPED", slurm_state="COMPLETED",
                       exit_code="0:0", allocation_released=True, stopped=time.time())
                break
            if time.monotonic() > deadline:
                job_description(job_id, state["owner"])
                command(["scancel", job_id])
                raise TimeoutError("Finite smoke wait expired; own confirmed job cancellation requested")
            time.sleep(5)
    except BaseException as exc:
        update(status="SMOKE_ERROR_REQUIRES_REVIEW" if state["submission_attempted"] else "FAILED_STOPPED",
               reason=f"{type(exc).__name__}: {exc}", stopped=time.time())
        # Never label a submitted allocation stopped without evidence from Slurm.
        if state["job_id"]:
            with contextlib.suppress(Exception):
                record = accounting(state["job_id"], state["owner"])
                queued = command(["squeue", "-h", "-j", state["job_id"], "-o", "%i"])
                if record and record["state"] in TERMINAL and not queued:
                    update(status="FAILED_STOPPED", allocation_released=True,
                           slurm_state=record["state"], exit_code=record["exit_code"])
        raise
    finally:
        if lock is not None:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    controller(args.root)


if __name__ == "__main__":
    main()
