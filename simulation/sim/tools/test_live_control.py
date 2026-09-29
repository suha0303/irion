"""Actual Gazebo RPC integration, separate from synthetic queue tests."""

import json
import subprocess
import time

from capstone_sim.control import control
from capstone_sim.recording import dump, verify
from capstone_sim.runner import ROOT

before = set((ROOT / "runs").iterdir())
log = (ROOT / "artifacts/live-control-run.stdout.json").open("w")
p = subprocess.Popen(
    [
        str(ROOT / "capstone-sim"),
        "run",
        "--scenario",
        "hover_normal",
        "--seed",
        "61",
        "--duration",
        "90",
        "--instance",
        "2",
    ],
    stdout=log,
)
run = None
results = []
try:
    until = time.monotonic() + 100
    while time.monotonic() < until:
        for d in set((ROOT / "runs").iterdir()) - before:
            config = d / "evaluation/scenario.json"
            if config.exists() and json.loads(config.read_text())["seed"] == 61:
                run = d
        if run and json.loads((run / "run.json").read_text())["status"] == "airborne":
            break
        time.sleep(0.2)
    assert run is not None
    # Same-time requests apply FIFO at physics boundary, identical duplicate is a no-op.
    args = (run, "apply_parameter_change", 100, "battery_r0", 0.05, 50000000)
    results.append(control(*args))
    results.append(control(*args))
    results.append(control(run, "apply_parameter_change", 100, "battery_r0", 0.06, 50000000))
    results.append(control(run, "apply_parameter_change", 101, "motor_friction", 0.00002, 50000000))
    while time.monotonic() < until:
        receipt = control(run, "status", 100)
        if "data: 50000000" in receipt["stdout"] or (
            "data: -1" not in receipt["stdout"] and receipt["stdout"].strip()
        ):
            results.append(receipt)
            break
        time.sleep(0.5)
    exit_code = p.wait(timeout=200)
    log.close()
    events = [json.loads(s) for s in (run / "evaluation/events.jsonl").read_text().splitlines()]
    actual = {e["event_id"]: e for e in events}
    passed = (
        exit_code == 0
        and len(events) == 2
        and "data: 0" in results[0]["stdout"]
        and "data: 1" in results[1]["stdout"]
        and "data: -1" in results[2]["stdout"]
        and actual["request-100"]["applied_time_us"] >= 50000000
        and actual["request-101"]["profile_revision"] == 2
    )
    dump(
        ROOT / "artifacts/live-control.json",
        dict(
            passed=passed,
            exit_code=exit_code,
            run=str(run),
            responses=results,
            events=events,
            contract=verify(run),
        ),
    )
    raise SystemExit(0 if passed else 4)
finally:
    if p.poll() is None:
        p.send_signal(2)
        p.wait(timeout=30)
