import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
x = np.genfromtxt(ROOT / "data/public/holybro_air2216_t1045.csv", delimiter=",", names=True)
w = x["rpm"] * 2 * np.pi / 60
t = x["thrust_g"] * 0.00980665
q = x["torque_nm"]
power = 16 * x["current_a"]
train = np.arange(len(x)) % 2 == 0
hold = ~train
kt = float(np.dot(w[train] ** 2, t[train]) / np.dot(w[train] ** 2, w[train] ** 2))
kq = float(np.dot(w[train] ** 2, q[train]) / np.dot(w[train] ** 2, w[train] ** 2))
ke = 60 / (2 * np.pi * 920)
resistance = 0.115


# Fit aggregate friction and ESC loss against manufacturer power, preserving SI torque/power identities.
def prediction(p):
    b, eta = p
    c = ke * 0.8 - b * (10 - resistance * 0.8) / ke
    im = (kq * w * w + b * w + c) / ke
    return (ke * w + resistance * im) * im / eta


fit = least_squares(
    lambda p: (prediction(p)[train] - power[train]) / power[train],
    [2e-6, 0.9],
    bounds=([0, 0.7], [ke * 0.8 / ((10 - resistance * 0.8) / ke), 1]),
)
b, eta = fit.x
c = ke * 0.8 - b * (10 - resistance * 0.8) / ke
report = dict(
    origin="manufacturer_table_comparison",
    source="https://cdn.shopify.com/s/files/1/0604/5905/7341/files/X500MotorSpec.png?v=1678791632",
    transcription="manual checked table, grams force converted with standard gravity 9.80665m/s²; RPM to rad/s",
    fit_rows=np.where(train)[0].tolist(),
    holdout_rows=np.where(hold)[0].tolist(),
    parameters=dict(
        kt_prop=kt,
        kq_prop=kq,
        friction=float(b),
        coulomb=float(c),
        esc_efficiency=float(eta),
        resistance=0.115,
    ),
    holdout_thrust_max_relative=float(np.max(np.abs(kt * w[hold] ** 2 / t[hold] - 1))),
    holdout_torque_max_absolute_nm=float(np.max(np.abs(kq * w[hold] ** 2 - q[hold]))),
    holdout_power_max_relative=float(np.max(np.abs(prediction(fit.x)[hold] / power[hold] - 1))),
    criterion=dict(thrust_relative=0.15, torque_absolute_nm=0.015, power_relative=0.25),
    limits=[
        "manufacturer aggregate bench reference, no independent lab uncertainty",
        "torque rounded to 0.01Nm; inferred friction/ESC parameters are weakly identified",
        "manufacturer throttle percentage is NOT our post-mapping effective duty",
        "no measured dynamic motor transient; rotor inertia and cooling remain assumptions",
    ],
)
report["passed"] = bool(
    report["holdout_thrust_max_relative"] < 0.15
    and report["holdout_torque_max_absolute_nm"] < 0.015
    and report["holdout_power_max_relative"] < 0.25
)
u = np.genfromtxt(ROOT / "data/public/uiuc_apcsf_10x4.7_static.txt", names=True)
report["uiuc_comparison"] = dict(
    different_propeller=True,
    model="APC SF 10x4.7, not Holybro T1045",
    kt_range=(u["CT"] * 1.225 * 0.254**4 / (2 * np.pi) ** 2).tolist(),
    kq_range=(u["CP"] / (2 * np.pi) * 1.225 * 0.254**5 / (2 * np.pi) ** 2).tolist(),
    source="https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_static_kt0835.txt",
    license="copyright UIUC; public download, no explicit redistribution license identified",
    role="context comparison only, not target prop validation",
)
(ROOT / "artifacts/propulsion-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
(ROOT / "profiles/propulsion_runtime.txt").write_text(
    " ".join(
        str(report["parameters"][k])
        for k in ["kt_prop", "kq_prop", "friction", "coulomb", "esc_efficiency", "resistance"]
    )
    + "\n"
)
print(json.dumps({k: v for k, v in report.items() if k != "uiuc_comparison"}, indent=2))
