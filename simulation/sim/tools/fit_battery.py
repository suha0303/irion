"""CMU physical parameter identification; no AI features or training split."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "data/public/VAH07.csv"
meta = json.loads((ROOT / "data/public/metadata.json").read_text())
entry = next(f for f in meta["files"] if f["name"] == path.name)
assert path.stat().st_size == entry["size"]
assert hashlib.md5(path.read_bytes()).hexdigest() == entry["computed_md5"]
x = np.genfromtxt(path, delimiter=",", names=True)
assert np.isfinite(np.column_stack([x[n] for n in x.dtype.names])).all()
assert np.diff(x["time_s"]).min() >= 0
# First low-rate RPT: cumulative discharge is retained by the source across steps 3..7.
rpt = x[792:19182]
capacity = rpt["QDischarge_mA_h"][-1] / 1000
soc = 1 - rpt["QDischarge_mA_h"] / 1000 / capacity
knots = np.linspace(0, 1, 31)
lowrate = np.interp(knots, soc[::-1], rpt["Ecell_V"][::-1])
# Mission slices are defined by Ns transitions, never by unreliable cycleNumber.
idx = np.where(np.r_[True, np.diff(x["Ns"]) != 0])[0]
missions = []
for j, a in enumerate(idx):
    if a < 19300 or x["Ns"][a] != 4:
        continue
    if j + 4 >= len(idx):
        break
    b = idx[j + 4]
    s = x[a - 1 : b].copy()
    if s["Ns"][-1] != 7:
        continue
    missions.append(s)
    if len(missions) == 4:
        break
assert len(missions) == 4


def simulate(theta, s):
    r0, r1, tau, q, offset = theta
    ocv = lowrate + 0.6 * (r0 + r1) + offset
    t = s["time_s"]
    i = -s["I_mA"] / 1000
    v = s["Ecell_V"]
    z = np.interp(v[0], ocv, knots)
    vp = 0
    pred = []
    for k in range(len(t)):
        if k:
            dt = t[k] - t[k - 1]
            current = 0.5 * (i[k - 1] + i[k])
            z -= current * dt / (3600 * q)
            a = np.exp(-dt / tau)
            vp = a * vp + r1 * (1 - a) * current
        pred.append(np.interp(z, knots, ocv) - r0 * i[k] - vp)
    return np.asarray(pred)


# Bounds precede fitting; Q constrained by independently measured RPT, other bounds physical priors.
fit = least_squares(
    lambda p: np.concatenate([simulate(p, s) - s["Ecell_V"] for s in missions[:2]]),
    [0.015, 0.008, 30, capacity, 0],
    bounds=(
        [0.001, 0.0001, 1, capacity * 0.95, -0.06],
        [0.08, 0.08, 300, capacity * 1.05, 0.06],
    ),
    x_scale="jac",
)
r0, r1, tau, q, offset = fit.x
sv = np.linalg.svd(fit.jac, compute_uv=False)
report = {
    "origin": "public_measurement",
    "source": entry,
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "license": meta["license"],
    "cell": "VAH07 Sony-Murata VTC6",
    "protocol": "first low-rate RPT OCV proxy; first two mission+rest fit; next two holdout",
    "source_units": {"time_s": "s", "Ecell_V": "V", "I_mA": "mA charging positive"},
    "conversion": "I_discharge_A=-I_mA/1000",
    "duplicate_timestamps": int(np.sum(np.diff(x["time_s"]) == 0)),
    "largest_gap_s": float(np.diff(x["time_s"]).max()),
    "measured_rpt_capacity_ah": capacity,
    "cell_parameters": {
        "r0_ohm": r0,
        "r1_ohm": r1,
        "tau_s": tau,
        "capacity_ah": q,
        "ocv_offset_v": offset,
    },
    "jacobian_condition": float(sv[0] / sv[-1]),
    "optimizer_success": bool(fit.success),
    "fit_rmse_cell_v": [
        float(np.sqrt(np.mean((simulate(fit.x, s) - s["Ecell_V"]) ** 2))) for s in missions[:2]
    ],
    "holdout_rmse_cell_v": [
        float(np.sqrt(np.mean((simulate(fit.x, s) - s["Ecell_V"]) ** 2))) for s in missions[2:]
    ],
    "acceptance_cell_rmse_v": 0.15,
    "limitations": [
        "low-rate terminal voltage is an OCV proxy, not fully rested OCV",
        "same cell and adjacent early missions; no independent-cell or real X500 validation",
        "coarse 10/30-second samples do not resolve instantaneous R0; weak identifiability must be retained",
        "thermal parameters and temperature dependence remain engineering assumptions",
    ],
}
report["comparison_status"] = "passed" if max(report["holdout_rmse_cell_v"]) < 0.15 else "failed"
(ROOT / "artifacts/battery-fit.json").write_text(json.dumps(report, indent=2) + "\n")
pack = {
    "profile_id": "cmu_vah07_4s2p",
    "ns": 4,
    "np": 2,
    "capacity_ah": float(2 * q),
    "r0_ohm": float(2 * r0),
    "r1_ohm": float(2 * r1),
    "c1_f": float(0.5 * tau / r1),
    "ocv_soc": knots.tolist(),
    "ocv_cell_v": (lowrate + 0.6 * (r0 + r1) + offset).tolist(),
    "provenance": "artifacts/battery-fit.json",
}
(ROOT / "profiles/cmu_vah07_4s2p.json").write_text(json.dumps(pack, indent=2) + "\n")
(ROOT / "profiles/battery_runtime.txt").write_text(
    " ".join(str(pack[k]) for k in ["r0_ohm", "r1_ohm", "c1_f", "capacity_ah"])
    + "\n"
    + " ".join(str(v) for v in pack["ocv_cell_v"])
    + "\n"
)
print(json.dumps(report, indent=2))
