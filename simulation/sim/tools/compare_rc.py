"""Independent SciPy DOP853 1RC/thermal comparison to C++ midpoint core."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=root / "artifacts/rc-bench.csv")
parser.add_argument("--output", type=Path, default=root / "artifacts/rc-comparison.json")
args = parser.parse_args()
p = json.loads((root / "profiles/collection/battery.json").read_text())
x = np.genfromtxt(args.input, names=True, delimiter=",")
state = [0.9, 0.0, 298.15]
ys = []
for start, end, current in [(0, 30, 10), (30, 60, 20), (60, 120, 0)]:
    t = x["t"][(x["t"] > start) & (x["t"] <= end)]

    def rhs(time, y):
        return [
            -current / (3600 * p["actual_capacity_ah"]),
            (current - y[1] / p["r1_ohm"]) / p["c1_f"],
            (current**2 * p["r0_ohm"] + y[1] ** 2 / p["r1_ohm"]
             - p["thermal"]["cooling_w_k"] * (y[2] - 298.15))
            / p["thermal"]["heat_capacity_j_k"],
        ]

    result = solve_ivp(rhs, (start, end), state, method="DOP853", t_eval=t, rtol=1e-11, atol=1e-12)
    assert result.success
    state = result.y[:, -1]
    ys.append(result.y.T)
y = np.concatenate(ys)
voltage = (
    p["ns"] * np.interp(y[:, 0], p["ocv_soc"], p["ocv_cell_v"])
    - p["r0_ohm"] * x["current"]
    - y[:, 1]
)
errors = {
    "soc": float(np.max(np.abs(y[:, 0] - x["soc"]))),
    "polarization_v": float(np.max(np.abs(y[:, 1] - x["vp"]))),
    "temperature_k": float(np.max(np.abs(y[:, 2] - x["temperature"]))),
    "terminal_v": float(np.max(np.abs(voltage - x["voltage"]))),
    "energy_j": float(np.max(np.abs(x["energy_residual"]))),
}
limits = dict(soc=1e-8, polarization_v=1e-7, temperature_k=1e-7, terminal_v=1e-6, energy_j=1e-4)
report = dict(
    origin="independent_numerical_comparison",
    solver="SciPy DOP853 vs C++ RK2",
    profile="CMU VAH01 measured 4S4P baseline; 10A 30s,20A30s,rest60s",
    max_errors=errors,
    limits=limits,
    passed=all(errors[k] < limits[k] for k in errors),
    hardware_validated=False,
)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 4)
