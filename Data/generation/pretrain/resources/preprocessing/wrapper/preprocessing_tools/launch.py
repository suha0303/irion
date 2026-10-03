"""One explicit stage, one durable claim. No next-stage submission or retry."""

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from preprocessing_core.control import read, sha, write

from .control import LEDGER, STORAGE, admission, claim

PYTHON = "/abr/coss43/miniconda3/envs/mamba/bin/python"


def command(root, out, approval, lock, stage, python):
    if stage not in {"prepare", "main", "calibrate"}:
        raise ValueError("unsupported stage; no automatic gate")
    worker = [
        python,
        "-B",
        "-m",
        "preprocessing_tools.stage",
        "--root",
        str(root),
        "--out",
        str(out),
        "--approval",
        str(approval),
        "--lock",
        str(lock),
        "--stage",
        stage,
    ]
    if stage == "main":
        if python != PYTHON:
            raise ValueError("approved mamba interpreter required")
        return [
            "srun",
            "--partition=p02",
            "--nodelist=DIS01",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            "--immediate=900",
            "--kill-on-bad-exit=1",
            "--job-name=x500-r3-long15-main",
            "--time=180",
            *worker,
        ]
    return worker


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "out", "approval", "lock"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--stage", choices=("prepare", "main", "calibrate"), required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("do not nest allocations or run local stages on compute nodes")
    gpu = args.stage == "main"
    if gpu and Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError("approved mamba login interpreter required")
    if not gpu and sys.version_info[:2] != (3, 10):
        raise ValueError("existing local Python3.10 AI environment required")
    lock, _, _ = admission(root, args.approval, args.lock, args.stage, out)
    if shutil.disk_usage(out.parent).free < 50_000_000_000:
        raise ValueError("free disk below 50GB; no fallback storage")
    binding = {
        "package_sha256": sha(root / "PACKAGE.json"),
        "approval_sha256": sha(args.approval),
        "lock_sha256": sha(args.lock),
        "frozen_manifest_sha256": lock["frozen_manifest"]["sha256"],
    }
    mutex = Path("/abr/coss43/.x500-ai-gpu.lock") if gpu else STORAGE / "metadata/ai-stage.lock"
    ledger = LEDGER if gpu else STORAGE / "metadata/ai-claims"
    with mutex.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = ["squeue", "-h", "-u", str(os.getuid()), "-o", "%A"]
        if gpu and subprocess.check_output(queue, text=True).strip():
            raise ValueError("own queue not empty; no additional GPU submission")
        claim(out, ledger, args.stage, binding)
        argv = command(
            root,
            out,
            args.approval.resolve(),
            args.lock.resolve(),
            args.stage,
            PYTHON if gpu else sys.executable,
        )
        write(
            out / "submission.json",
            {
                "argv": argv,
                **binding,
                "retries": 0,
                "stage_minutes": 180 if gpu else 120,
                "queue_minutes": 15 if gpu else 0,
            },
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(str(root / p) for p in ("wrapper", "ai", "ai/src"))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Existing fit already fixes Torch threads; bound CPU ridge BLAS as well.
        env["OPENBLAS_NUM_THREADS"] = "2"
        env["OMP_NUM_THREADS"] = "2"
        started = time.monotonic()
        with (out / "worker.log").open("x") as log:
            try:
                result = subprocess.run(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=None if gpu else 7200,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        write(
            out / "exit.json",
            {"exit_code": code, "wall_including_queue_s": time.monotonic() - started},
        )
        released, job_id = not gpu, None
        if gpu:
            id_path = out / "verified-job-id.txt"
            job_id = id_path.read_text().strip() if id_path.exists() else None
            for _ in range(10):
                active = subprocess.check_output(queue, text=True).split()
                write(out / "own-queue-after.json", {"job_id": job_id, "active_ids": active})
                if job_id and job_id.isdigit() and job_id not in active:
                    released = True
                    break
                if job_id is None:
                    break
                time.sleep(1)
        success = read(out / "artifact.json") if (out / "artifact.json").exists() else {}
        passed = code == 0 and success.get("passed") is True and released
        record = {
            "passed": passed,
            "released": released,
            "job_id": job_id,
            "artifact_sha256": sha(out / "artifact.json") if success else None,
            "output": str(out),
            "retry": False,
            "next_stage_authorized": False,
        }
        write(out / "release.json", record)
        write(ledger / (args.stage + ".done.json"), record)
        if not passed:
            raise RuntimeError("stage failed or release unverified; preserve output and STOP")
        print("Stage complete; no automatic next stage.", flush=True)


if __name__ == "__main__":
    main()
