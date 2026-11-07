"""Run the producer through its own launcher; Data never imports its flight runtime."""

import json
import os
import subprocess
from pathlib import Path

from .common import ROOT


def command(case, parent=None, paired_options=(), instance=0, timing=False):
    if type(instance) is not int or not 0 <= instance < 16:
        raise ValueError("Native collection instance must be in0..15")
    args = [
        str(ROOT / "simulation/capstone-sim"),
        "run",
        "--configuration",
        "collection",
        "--scenario",
        case["scenario"],
        "--seed",
        str(case["seed"]),
        "--duration",
        str(case["duration"]),
        "--instance",
        str(instance),
        "--real-time-factor",
        "2",
        "--operation-variation",
        json.dumps(case["operation_variation"], sort_keys=True),
    ]
    if parent is not None:
        args += ["--parent-attempt-id", parent]
    if case.get("battery_stage") is not None:
        args += ["--battery-stage", case["battery_stage"]]
    if case.get("fault_after_mission_s") is not None:
        args += ["--fault-after-mission-s", str(case["fault_after_mission_s"])]
    if timing:
        args.append("--timing")
    if not isinstance(paired_options, (list, tuple)) or any(
        not isinstance(v, str) for v in paired_options
    ):
        raise ValueError(
            "Reviewed native CLI argv required, not guessed Python kwargs"
        )
    return args + [
        token.format(paired_case_id=case["paired_case_id"])
        for token in paired_options
    ]


def environment_check():
    """Real interpreter/import/CLI boundary, with no run dispatch or flight."""
    probe = """import json,sys,pymavlink
from capstone_sim.runner import run_flight
print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,version=sys.version,
    pymavlink_path=pymavlink.__file__,run_flight_imported=callable(run_flight),flight_started=False)))"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "simulation/sim/src"))
    args = [str(ROOT / "simulation/.venv/bin/python"), "-B", "-c", probe]
    result = subprocess.run(
        args, env=env, capture_output=True, text=True, timeout=30, check=True
    )
    data = json.loads(result.stdout)
    if (
        Path(data["prefix"]) != ROOT / "simulation/.venv"
        or not data["run_flight_imported"]
    ):
        raise ValueError("Verified Sim interpreter/runtime required")
    cli = subprocess.run(
        [str(ROOT / "simulation/capstone-sim"), "run", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    if any(
        flag not in cli.stdout
        for flag in (
            "--operation-variation",
            "--parent-attempt-id",
            "--fault-after-mission-s",
        )
    ):
        raise ValueError("Actual producer CLI is missing required arguments")
    return dict(
        status="PASSED_NO_FLIGHT",
        probe=data,
        probe_command=args,
        cli_command=[str(ROOT / "simulation/capstone-sim"), "run", "--help"],
        cli_exit=cli.returncode,
    )


def read_result(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(
            "Producer stdout exceeds bounded JSON result size; preserve for review"
        )
    value = json.loads(path.read_text())
    if value.get("status") not in {"completed", "failed"} or not isinstance(
        value.get("run"), str
    ):
        raise ValueError("Producer did not return an actual run/status result")
    raw = Path(value["run"]).absolute()
    if raw.parent != ROOT / "simulation/runs":
        raise ValueError("Producer returned a run outside the native raw root")
    return value
