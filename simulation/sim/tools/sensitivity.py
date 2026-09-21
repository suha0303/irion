import io
import json
import subprocess
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
rows = []
for soc, duty, inertia in [
    (0.9, 0.55, 1),
    (0.4, 0.55, 1),
    (0.9, 0.4, 1),
    (0.9, 0.7, 1),
    (0.9, 0.55, 0.5),
    (0.9, 0.55, 2),
]:
    p = subprocess.run(
        [str(root / "build/powertrain_bench"), "2", str(soc), str(duty), "6", str(inertia)],
        capture_output=True,
        text=True,
        check=True,
    )
    x = np.genfromtxt(io.StringIO(p.stdout), delimiter=",", names=True)
    name = f"bench-soc{soc}-duty{duty}-J{inertia}.csv"
    (root / "artifacts" / name).write_text(p.stdout)
    rows.append(
        dict(
            soc=soc,
            duty=duty,
            inertia_multiplier=inertia,
            source=name,
            exit_code=p.returncode,
            omega_at_100ms=float(x["omega_rad_s"][0]),
            final_voltage_v=float(x["voltage_v"][-1]),
            final_current_a=float(x["current_a"][-1]),
            final_thrust_n=float(x["thrust_total_n"][-1]),
            motor_energy_residual_j=float(x["energy_residual_j"][-1]),
        )
    )
report = dict(
    origin="physics_bench",
    profile="analytic_prior",
    not_hardware_validation=True,
    rows=rows,
    passed=rows[1]["final_voltage_v"] < rows[0]["final_voltage_v"]
    and rows[1]["final_thrust_n"] < rows[0]["final_thrust_n"]
    and rows[3]["final_current_a"] > rows[2]["final_current_a"]
    and rows[5]["omega_at_100ms"] < rows[4]["omega_at_100ms"],
)
(root / "artifacts/sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 4)
