import json
import subprocess
import time

from capstone_sim.recording import dump, verify
from capstone_sim.runner import ROOT

with (ROOT / "artifacts/final-smoke.stdout.json").open("w") as log:
    cmd = [
        str(ROOT / "capstone-sim"),
        "run",
        "--scenario",
        "hover_normal",
        "--seed",
        "83",
        "--duration",
        "90",
        "--instance",
        "2",
    ]
    p = subprocess.Popen(cmd, stdout=log)
    time.sleep(2)
    conflict = subprocess.run(
        [str(ROOT / "capstone-sim"), "run", "--duration", "10", "--instance", "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    code = p.wait(timeout=240)
run = json.loads((ROOT / "artifacts/final-smoke.stdout.json").read_text())
result = dict(
    command=cmd,
    exit_code=code,
    run=run,
    conflict_exit_code=conflict.returncode,
    conflict_stdout=conflict.stdout,
    contract=verify(run["run"]),
)
result["passed"] = (
    code == 0 and conflict.returncode == 3 and result["contract"]["status"] == "passed"
)
dump(ROOT / "artifacts/instance-lease.json", result)
raise SystemExit(0 if result["passed"] else 4)
