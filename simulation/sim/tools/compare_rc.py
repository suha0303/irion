"""Independent SciPy DOP853 1RC/thermal comparison to C++ midpoint core."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

root = Path(__file__).resolve().parents[2]
x = np.genfromtxt(root / "artifacts/rc-bench.csv", names=True, delimiter=",")
state = [0.9, 0.0, 298.15]
ys = []
for start, end, current in [(0, 30, 10), (30, 60, 20), (60, 120, 0)]:
    t = x["t"][(x["t"] > start) & (x["t"] <= end)]

    def rhs(time, y):
        return [
            -current / (3600 * 6),
            (current - y[1] / 0.015) / 2000,
            (current**2 * 0.04 + y[1] ** 2 / 0.015 - 1.5 * (y[2] - 298.15)) / 650,
        ]

    result = solve_ivp(rhs, (start, end), state, method="DOP853", t_eval=t, rtol=1e-11, atol=1e-12)
    assert result.success
    state = result.y[:, -1]
    ys.append(result.y.T)
y = np.concatenate(ys)
voltage = (
    4 * np.interp(y[:, 0], [0, 0.1, 0.3, 0.6, 0.9, 1], [3.2, 3.5, 3.65, 3.8, 4.05, 4.2])
    - 0.04 * x["current"]
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
    profile="prior 4S/6Ah matched OCV/R0/R1/C1/thermal; 10A 30s,20A30s,rest60s",
    max_errors=errors,
    limits=limits,
    passed=all(errors[k] < limits[k] for k in errors),
    hardware_validated=False,
)
(root / "artifacts/rc-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 4)
