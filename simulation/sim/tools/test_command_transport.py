"""Actual Gazebo protocol rejection, frozen clock, coast timeout and RPC exit codes."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim/src"))
from capstone_sim.control import control  # noqa: E402
from capstone_sim.runner import prepare_world  # noqa: E402

out = ROOT / "artifacts/remediation-20261014"
folder = out / "protocol-world"
(folder / "reference").mkdir(parents=True, exist_ok=True)
(folder / "run.json").write_text(
    json.dumps(dict(run_id=folder.name, status="running", origin="gazebo_protocol_fixture"))
)
world = prepare_world(folder, 0)
env = os.environ.copy()
env.update(
    GZ_PARTITION="x500_" + folder.name,
    GZ_IP="127.0.0.1",
    GZ_HOMEDIR=str(folder / "gz"),
    GZ_SIM_RESOURCE_PATH=str(ROOT / "vendor/PX4-Autopilot/Tools/simulation/gz/models"),
    X500_TRUTH_PATH=str(folder / "truth.jsonl"),
)
log = (folder / "gz.log").open("w")
p = subprocess.Popen(
    ["gz", "sim", "-s", "-r", str(world)],
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)


def read_rows():
    path = folder / "truth.jsonl"
    return (
        [json.loads(x) for x in path.read_text().splitlines() if x.endswith("}")]
        if path.exists()
        else []
    )


def pause(value):
    r = subprocess.run(
        [
            "gz",
            "service",
            "-s",
            "/world/x500_world/control",
            "--reqtype",
            "gz.msgs.WorldControl",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            f"pause: {str(value).lower()}",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode or "true" not in r.stdout:
        raise RuntimeError(r.stdout + r.stderr)
    time.sleep(0.15)


def publish(seq, sample, order="FR,RL,FL,RR", semantic="post_mapping_count_0_1000"):
    payload = "header { stamp { sec: %d nsec: %d } " % (
        sample // 1000000,
        (sample % 1000000) * 1000,
    )
    for key, value in [
        ("schema", "x500.esc.request/1"),
        ("order", order),
        ("semantic", semantic),
        ("sequence", str(seq)),
    ]:
        payload += 'data { key: "' + key + '" value: "' + value + '" } '
    payload += "} velocity: 300 velocity: 300 velocity: 300 velocity: 300"
    r = subprocess.run(
        [
            "gz",
            "topic",
            "-t",
            "/x500_0/command/motor_speed",
            "-m",
            "gz.msgs.Actuators",
            "-p",
            payload,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if r.returncode:
        raise RuntimeError(r.stderr)


try:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        rows = read_rows()
        if rows and rows[-1]["time_us"] > 2000000:
            break
        time.sleep(0.1)
    pause(True)
    rows = read_rows()
    frozen = rows[-1]
    time.sleep(0.3)
    still = read_rows()[-1]
    now = frozen["time_us"]
    publish(1, now)
    publish(1, now)
    publish(2, now, order="FR,FL,RR,RL")
    publish(2, now, semantic="rpm")
    publish(2, now + 1000000)
    rpc = []
    for args in [(1, "battery_r0", 0.05), (1, "battery_r0", 0.05), (1, "battery_r0", 0.06)]:
        rpc.append(control(folder, "apply_parameter_change", args[0], args[1], args[2], 0))
    cli = subprocess.run(
        [
            str(ROOT / "capstone-sim"),
            "control",
            "--run",
            str(folder),
            "--operation",
            "apply_parameter_change",
            "--request-id",
            "1",
            "--target",
            "battery_r0",
            "--value",
            ".06",
        ],
        capture_output=True,
        text=True,
    )
    pause(False)
    time.sleep(1.0)
    receipt = control(folder, "status", 1)
    rows = read_rows()
    trace = [
        json.loads(x) for x in (folder / "truth.jsonl.command-trace.jsonl").read_text().splitlines()
    ]
    reasons = [x.get("reason") for x in trace]
    states = [x.get("state") for x in trace]
    active = [x for x in rows if any(x["command"])]
    golden = all(
        x["command_sample_us"] <= x["time_us"] + 20000
        and x["force_state_time_us"] == x["time_us"] - 1000
        for x in active
    )
    checks = dict(
        pause_frozen=frozen == still,
        replay_rejected="sequence_or_time_regression" in reasons,
        order_and_semantic_rejected=reasons.count("protocol_mismatch") >= 2,
        future_rejected="stale_or_future_sample" in reasons,
        timeout_coast="active" in states and "timeout_coast" in states,
        actual_command=bool(active) and all(abs(x["command"][0] - 0.3) < 1e-12 for x in active),
        golden_step_timestamps=golden,
        rpc_queued=rpc[0]["status"] == "queued",
        rpc_duplicate=rpc[1]["status"] == "duplicate",
        rpc_rejected=rpc[2]["status"] == "rejected",
        cli_rejected_exit4=cli.returncode == 4,
        rpc_applied=receipt["status"] == "applied" and receipt["revision"] == 1,
    )
    report = dict(
        origin="actual_gazebo_engine_fixture",
        passed=all(checks.values()),
        checks=checks,
        trace=trace,
        rpc=rpc,
        cli_exit_code=cli.returncode,
        receipt=receipt,
    )
    (out / "command-transport.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
finally:
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=15)
    log.close()
raise SystemExit(0 if report["passed"] else 4)
