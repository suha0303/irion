"""Login-node submission/status only; CUDA code runs exclusively inside the checked job."""

import argparse
import fcntl
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import (
    GPU_LOCK,
    Progress,
    config_check,
    memory_available,
    python_for,
    read,
    sha,
    verify,
    write,
)

REMOTE_LOCK = GPU_LOCK


def command(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.PIPE, timeout=30).strip()


def verify_input(root):
    root = Path(root).resolve()
    config_check(read(root / "config.json"))
    verify(root, read(root / "inventory.json"))
    return dict(status="INPUT_VERIFIED_NO_TRAINING", inventory_sha256=sha(root / "inventory.json"))


def submit(root):
    root = Path(root).resolve()
    job = root.parent
    cfg = read(root / "config.json")
    config_check(cfg)
    verify(root, read(root / "inventory.json"))
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Submit from login node only; do not nest allocations")
    if Path(sys.executable).resolve() != Path(python_for(root, required=True)).resolve():
        raise ValueError("Approved interpreter required; no environment installation")
    with REMOTE_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = command(["squeue", "-h", "-u", str(os.getuid()), "-o", "%A %j %T"])
        if queue:
            raise ValueError("Existing own pending/running job; not submitted: " + queue)
        with (job / "submitted.claim").open("x") as f:
            f.write("One submission only; unknown outcome is never retried\n")
        out = job / "output"
        out.mkdir()
        inventory_sha = sha(root / "inventory.json")
        script = job / "job.sh"
        script.write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            + "export PYTHONPATH="
            + shlex.quote(str(root / "ai") + ":" + str(root / "ai/src"))
            + "\n"
            + "export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
            + "export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n"
            + "exec "
            + shlex.join(
                [
                    python_for(root, required=True),
                    "-B",
                    "-m",
                    "execution.remote",
                    "worker",
                    "--root",
                    str(root),
                    "--inventory-sha",
                    inventory_sha,
                ]
            )
            + "\n"
        )
        argv = [
            "sbatch",
            "--parsable",
            "--partition=p02",
            "--nodelist=DIS01",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            "--mem=16G",
            "--time=12:00:00",
            "--no-requeue",
            "--job-name=drone-ai-train",
            "--output=" + str(job / "slurm.log"),
            "--error=" + str(job / "slurm.log"),
            "--chdir=" + str(root),
            str(script),
        ]
        write(
            job / "submission-intent.json",
            dict(argv=argv, inventory_sha256=inventory_sha, automatic_retry=False),
        )
        response = command(argv)
        identifier = response.split(";")[0]
        if not identifier.isdigit():
            raise ValueError("Unknown submission response; do not retry: " + response)
        record = dict(
            job_id=identifier,
            uid=os.getuid(),
            submitted=time.time(),
            root=str(root),
            inventory_sha256=inventory_sha,
        )
        write(job / "submission.json", record)
        print(__import__("json").dumps(record), flush=True)


def snapshot(root):
    root = Path(root).resolve()
    job = root.parent
    submitted = read(job / "submission.json")
    identifier = submitted["job_id"]
    if not str(identifier).isdigit() or submitted["uid"] != os.getuid():
        raise ValueError("Invalid job identity/owner")
    try:
        queue = command(
            ["squeue", "-h", "-j", identifier, "-u", str(os.getuid()), "-o", "%A|%T|%R"]
        )
    except subprocess.CalledProcessError as exc:
        # Slurm can purge a finished job from squeue before its accounting disappears.
        # Other query failures must not be interpreted as release.
        if "invalid job id" not in (exc.stderr or "").lower():
            raise
        queue = ""
    state = (
        read(job / "output/progress.json")
        if (job / "output/progress.json").exists()
        else dict(status="PENDING", stage="allocation")
    )
    result = dict(job_id=identifier, queue=queue, progress=state, released=False)
    if not queue:
        accounting = command(
            ["sacct", "-n", "-P", "-j", identifier, "--format=JobIDRaw,User,State,ExitCode"]
        )
        entries = [
            line.split("|") for line in accounting.splitlines() if line.split("|")[0] == identifier
        ]
        if len(entries) == 1 and entries[0][2].split()[0] in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
            "REVOKED",
        }:
            fields = entries[0]
            if fields[1] != __import__("pwd").getpwuid(os.getuid()).pw_name:
                raise ValueError("Accounting owner mismatch")
            result.update(released=True, slurm_state=fields[2], exit_code=fields[3])
            result["remaining_own_job_ids"] = command(
                ["squeue", "-h", "-u", str(os.getuid()), "-o", "%A"]
            ).split()
        else:
            result["status"] = "UNKNOWN_ACCOUNTING"
    return result


def cancel(root):
    root = Path(root).resolve()
    s = read(root.parent / "submission.json")
    identifier = s["job_id"]
    if not identifier.isdigit() or s["uid"] != os.getuid():
        raise ValueError("Owned exact job required")
    raw = command(["scontrol", "show", "job", "-o", identifier])
    record = dict(re.findall(r"(?:^|\s)(\w+)=(\S+)", raw))
    if (
        record.get("JobId") != identifier
        or record.get("JobName") != "drone-ai-train"
        or record.get("Partition") != "p02"
        or record.get("UserId", "").split("(")[-1].rstrip(")") != str(os.getuid())
    ):
        raise ValueError("Refuse cancellation: identity mismatch")
    subprocess.run(["scancel", identifier], check=True, timeout=30)
    print(
        "Cancellation requested for owned job "
        + identifier
        + "; final release requires status check."
    )


def worker(root, inventory_sha):
    root = Path(root).resolve()
    out = root.parent / "output"
    progress = Progress(out / "progress.json")

    def interrupted(number, _frame):
        raise InterruptedError("Job interrupted by signal " + str(number))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if sha(root / "inventory.json") != inventory_sha:
            raise ValueError("Changed input inventory")
        verify(root, read(root / "inventory.json"))
        cfg = read(root / "config.json")
        config_check(cfg)
        with REMOTE_LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from pipeline.runtime import check

            progress.update(stage="resource_admission")
            # Existing DIS01/owner/TRES/Python/Torch/CUDA/GPU checks; no CUDA_VISIBLE_DEVICES writes.
            environment = check(root / "ai", out, "train", job_prefix="drone-ai-")
            available = memory_available()
            required = cfg["memory"]["estimated_host_peak_bytes"]
            if available < required + 1024**3:
                raise MemoryError(
                    f"RAM admission: available{available}, conservative peak{required}, reserve1GiB"
                )
            write(
                out / "host-memory-admission.json",
                dict(available_bytes=available, estimate=cfg["memory"], job=environment["job"]),
            )
            from .engine import bundles, calibrate

            bundles(root, out, cfg, progress)
            calibrate(root, out, progress)
            progress.update(stage="seal_outputs", done=0, total=1)
            if sha(root / "inventory.json") != inventory_sha:
                raise ValueError("Input inventory changed during execution")
            verify(root, read(root / "inventory.json"))
            files = {
                str(p.relative_to(out)): sha(p)
                for p in out.rglob("*")
                if p.is_file() and p.name != "progress.json"
            }
            write(
                out / "result-seal.json",
                dict(
                    files=files,
                    config_sha256=sha(root / "config.json"),
                    models=cfg["models"],
                    seed=17,
                    final_test_used=False,
                    status="TRAIN_CALIBRATION_INFERENCE_COMPLETE",
                    truth_join=False,
                    next_round_authorized=False,
                ),
            )
            progress.update(
                status="COMPUTE_COMPLETE_PENDING_RELEASE",
                stage="stopped",
                done=1,
                total=1,
                phase_eta_s=None,
            )
    except BaseException as exc:
        progress.update(
            status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}", automatic_retry=False
        )
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["submit", "snapshot", "cancel", "worker", "verify"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--inventory-sha")
    a = p.parse_args()
    if a.action == "verify":
        print(__import__("json").dumps(verify_input(a.root)))
    elif a.action == "submit":
        submit(a.root)
    elif a.action == "worker":
        worker(a.root, a.inventory_sha)
    elif a.action == "cancel":
        cancel(a.root)
    else:
        print(__import__("json").dumps(snapshot(a.root)))


if __name__ == "__main__":
    main()
