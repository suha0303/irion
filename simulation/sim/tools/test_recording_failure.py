"""Fail actual SITL writers with OS per-process file-size limits; never fill a disk."""

import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim/src"))
from capstone_sim.recovery import recover  # noqa: E402

out = ROOT / "artifacts/remediation-20261014"
results = []
for index, target in enumerate(["producer", "native"]):
    before = set((ROOT / "runs").iterdir())
    log = (out / f"{target}-write-failure.log").open("w")
    p = subprocess.Popen(
        [
            str(ROOT / "capstone-sim"),
            "run",
            "--scenario",
            "hover_normal",
            "--seed",
            str(211 + index),
            "--duration",
            "90",
            "--instance",
            "4",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    run = None
    try:
        deadline = time.monotonic() + 90
        state = {}
        while time.monotonic() < deadline and p.poll() is None:
            for candidate in set((ROOT / "runs").iterdir()) - before:
                path = candidate / "evaluation/scenario.json"
                if path.exists() and json.loads(path.read_text())["seed"] == 211 + index:
                    run = candidate
            if run and (run / "run.json").exists():
                state = json.loads((run / "run.json").read_text())
                if state["status"] == "mission":
                    break
            time.sleep(0.1)
        if not run or state.get("status") != "mission":
            raise RuntimeError("did not reach mission before write-error injection")
        pid = p.pid if target == "producer" else state["owned_process_groups"]["px4"]
        file = run / (
            "observations/streams.jsonl" if target == "producer" else "native_logs/capture.jsonl"
        )
        limit = file.stat().st_size + 65536
        resource.prlimit(pid, resource.RLIMIT_FSIZE, (limit, limit))
        code = p.wait(timeout=45)
        final = json.loads((run / "run.json").read_text())
        alive = []
        for name, pgid in final["owned_process_groups"].items():
            try:
                os.killpg(pgid, 0)
                alive.append(name)
            except ProcessLookupError:
                pass
        recovered = recover(run, out / f"recovered-{target}-{run.name}")
        passed = (
            code == 4
            and final["status"] == "failed"
            and not alive
            and recovered["recovered_events"] > 0
        )
        if target == "native":
            passed = passed and 74 in final["process_exit_codes"]
        results.append(
            dict(
                target=target,
                run=str(run),
                exit_code=code,
                limit_bytes=limit,
                final=final,
                remaining_groups=alive,
                recovery=recovered,
                passed=passed,
            )
        )
        print(target, code, final["termination_reason"], passed, flush=True)
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            p.wait(timeout=40)
        log.close()
    (out / "recording-failure.json").write_text(json.dumps(results, indent=2) + "\n")
code = subprocess.call(
    [
        str(ROOT / "capstone-sim"),
        "run",
        "--scenario",
        "hover_normal",
        "--seed",
        "213",
        "--duration",
        "15",
        "--instance",
        "4",
    ],
    stdout=(out / "failure-restart.json").open("w"),
)
(out / "recording-failure-summary.json").write_text(
    json.dumps(
        dict(
            passed=all(r["passed"] for r in results) and code == 0,
            restart_exit_code=code,
            tests=results,
        ),
        indent=2,
    )
    + "\n"
)
raise SystemExit(0 if all(r["passed"] for r in results) and code == 0 else 4)
