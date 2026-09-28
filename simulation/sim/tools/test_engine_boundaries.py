"""Actual Gazebo engine rotation, command protocol, pause and timeout tests; no PX4."""

import json
import os
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim/src"))
from capstone_sim.profile import load_profile  # noqa: E402
from capstone_sim.runner import prepare_world  # noqa: E402

out = ROOT / "artifacts/remediation-20261014"
profile = load_profile()
positions = np.array(
    [[r["position_m"][0], -r["position_m"][1], -r["position_m"][2]] for r in profile["rotors"]]
)
m = profile["geometry"]["rotor_mass_kg_each"]
mb = profile["geometry"]["body_mass_kg"]
j = profile["propulsion"]["rotor_inertia_kg_m2"]
com = positions.sum(axis=0) * m / (mb + 4 * m)
inertia = np.diag(profile["geometry"]["body_inertia_diagonal_kg_m2"])
for pos in positions:
    inertia += np.diag([j / 2, j / 2, j]) + m * (np.eye(3) * pos.dot(pos) - np.outer(pos, pos))
inertia -= (mb + 4 * m) * (np.eye(3) * com.dot(com) - np.outer(com, com))
report = dict(
    origin="actual_gazebo_engine_fixture",
    mass_kg=mb + 4 * m,
    com_flu_m=com.tolist(),
    locked_inertia_com=inertia.tolist(),
    rotor_spin_inertia=j,
    rotors_fixed=True,
    tests=[],
)
for axis in range(3):
    folder = out / f"engine-axis-{axis}"
    (folder / "reference").mkdir(parents=True, exist_ok=True)
    world = prepare_world(folder, 0)
    tree = ET.parse(world)
    w = tree.getroot().find("world")
    w.find("gravity").text = "0 0 0"
    model = w.find("model[@name='x500_0']")
    model.find("pose").text = "0 0 10 0 0 0"
    tree.write(world, encoding="unicode")
    env = os.environ.copy()
    env.update(
        GZ_PARTITION=f"x500-remediation-axis{axis}",
        GZ_IP="127.0.0.1",
        GZ_HOMEDIR=str(folder / "gz"),
        GZ_SIM_RESOURCE_PATH=str(ROOT / "vendor/PX4-Autopilot/Tools/simulation/gz/models"),
        X500_TRUTH_PATH=str(folder / "truth.jsonl"),
        X500_ROTATION_TEST=str(axis),
    )
    with (folder / "gz.log").open("w") as log:
        p = subprocess.Popen(
            ["gz", "sim", "-s", "-r", "--iterations", "5001", str(world)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = p.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=10)
            raise
    rows = [json.loads(x) for x in (folder / "truth.jsonl").read_text().splitlines()]
    t = np.array([r["time_us"] / 1e6 for r in rows])
    omega = np.array([r["body_rate_flu"][axis] for r in rows])
    mom = np.array([r["moment_flu_nm"][axis] for r in rows])
    # Compare measured engine acceleration with the separately assembled mass tensor.
    k = np.where((t >= 1) & (t <= 4))[0]
    lo, hi = k[0], k[-1]
    observed = inertia[axis, axis] * (omega[hi] - omega[lo])
    impulse = np.trapezoid(mom[lo : hi + 1], t[lo : hi + 1])
    error = abs(observed - impulse) / max(abs(impulse), 1e-12)
    result = dict(
        axis=axis,
        exit_code=code,
        angular_momentum_change=observed,
        torque_impulse=impulse,
        relative_error=error,
        criterion=0.015,
    )
    if axis == 2:
        h = np.array(
            [sum(s * j * v for s, v in zip([1, 1, -1, -1], r["omega_rad_s"])) for r in rows]
        )
        external = np.array(
            [
                -sum(sign * q for sign, q in zip([1, 1, -1, -1], r["shaft_torque_nm"]))
                - 8.06428e-5
                * r["body_rate_flu"][2]
                * sum(
                    speed * (pos[0] ** 2 + pos[1] ** 2)
                    for speed, pos in zip(r["omega_rad_s"], positions)
                )
                for r in rows
            ]
        )
        total = observed + h[hi] - h[lo]
        external_impulse = np.trapezoid(external[lo : hi + 1], t[lo : hi + 1])
        result.update(
            total_angular_momentum_change=total,
            external_propeller_and_lateral_drag_impulse=external_impulse,
            total_relative_error=abs(total - external_impulse) / abs(external_impulse),
        )
    result["passed"] = bool(
        code == 0 and error < 0.015 and result.get("total_relative_error", 0) < 0.015
    )
    report["tests"].append(result)
    print(result, flush=True)
report["passed"] = all(r["passed"] for r in report["tests"])
(out / "engine-rotation.json").write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if report["passed"] else 4)
