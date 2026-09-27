"""Kill only the process group recorded by an owned SITL run; verify recovery."""

import json
import os
import signal
import subprocess
import time

from capstone_sim.recording import dump, verify
from capstone_sim.recovery import recover
from capstone_sim.runner import ROOT, run_flight

before = set((ROOT / "runs").iterdir())
with (ROOT / "artifacts/process-kill-run.stdout.json").open("w") as log:
    p = subprocess.Popen(
        [str(ROOT / "capstone-sim"), "run", "--seed", "71", "--duration", "90", "--instance", "2"],
        stdout=log,
    )
    run = None
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        for d in set((ROOT / "runs").iterdir()) - before:
            conf = d / "evaluation/scenario.json"
            if conf.exists() and json.loads(conf.read_text())["seed"] == 71:
                run = d
        if run:
            state = json.loads((run / "run.json").read_text())
            if state["status"] == "airborne":
                break
        time.sleep(0.2)
    assert run and state["status"] == "airborne"
    groups = state["owned_process_groups"]
    os.killpg(groups["gazebo"], signal.SIGKILL)
    exit_code = p.wait(timeout=30)
    state = json.loads((run / "run.json").read_text())
    alive = []
    for name, pid in groups.items():
        try:
            os.killpg(pid, 0)
            alive.append(name)
        except ProcessLookupError:
            pass
    recovery = recover(run, ROOT / "artifacts" / ("recovered-" + run.name))
    restart = run_flight("hover_normal", 71, 10, 2, run.name)
    report = dict(
        origin="px4_sitl_fault_injection",
        killed_run=str(run),
        exit_code=exit_code,
        status=state,
        alive_groups=alive,
        recovery=recovery,
        restart=restart,
        restart_contract=verify(restart["run"]),
    )
    report["passed"] = (
        exit_code == 4
        and state["status"] == "failed"
        and not alive
        and restart["status"] == "completed"
        and report["restart_contract"]["status"] == "passed"
    )
    dump(ROOT / "artifacts/process-recovery.json", report)
    raise SystemExit(0 if report["passed"] else 4)
