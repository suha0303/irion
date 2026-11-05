"""VAH01-only RPT/mission calibration and sealed temporal validation.

Capacity and pulse response come from measured RPTs. Only the first four
complete missions per RPT enter fitting; development and final windows are
kept separate. This module never starts a simulator or an AI training job.
"""

import csv
import fcntl
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .battery_aging import FLIGHT_POLICY

ROOT = Path(__file__).resolve().parents[3]
SOURCE_SHA256 = "596ab1e1e80eeac3509f789609f751527c2073a4310deec8274cbfaf4fef39c4"
IMPEDANCE_SHA256 = "f92a2e6b141d732863fbf5ba09e8867c073f15df2d35c80c83887051a1d31b25"
CATALOG = ROOT / "profiles/battery-aging/cmu-vah01.json"
PROTOCOL: dict[str, Any] = {
    "revision": "cmu-vah01-baseline/1",
    "source_cell": "VAH01",
    "selection": "user-selected single baseline cell; first canonical baseline, no listed source issues",
    "fit_mission_indices": [0, 1, 2, 3],
    "development_mission_indices": [4, 5, 6, 7],
    "holdout_mission_indices": [8, 9, 10, 11],
    "parameter_order": ["r0_ohm", "r1_ohm", "tau_s", "ocv_offset_v"],
    "bounds_lower": [0.001, 0.0001, 0.1, -0.08],
    "bounds_upper": [0.08, 0.08, 300.0, 0.08],
    "capacity": "fixed to measured complete RPT discharge Ah",
    "rpt_constraint": "measured current-interruption voltage response at 20% and 60% DOD",
    "ocv_correction": "0.6A steady low-rate IR correction plus fitted 4*SOC*(1-SOC) interior correction; endpoints unshifted",
    "acceptance": {"rmse_cell_v": 0.1, "absolute_bias_cell_v": 0.05,
                   "p95_absolute_error_cell_v": 0.2},
    "acceptance_basis": "predeclared engineering response criteria; no hardware or detectability guarantee",
}


def source_data():
    source = ROOT / "data/public/VAH01.csv"
    impedance = ROOT / "data/public/VAH01_impedance.csv"
    for path, expected in ((source, SOURCE_SHA256), (impedance, IMPEDANCE_SHA256)):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("CMU input cannot use symlinks")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("CMU baseline input integrity mismatch: " + path.name)
    x = np.genfromtxt(source, delimiter=",", names=True)
    if x.dtype.names is None or not np.isfinite(np.column_stack([x[n] for n in x.dtype.names])).all():
        raise ValueError("Nonfinite source record")
    if np.any(np.diff(x["time_s"]) < 0):
        raise ValueError("Source time regression")
    with impedance.open() as stream:
        dcir = list(csv.DictReader(stream))
    return x, dcir


def windows(x):
    starts = np.flatnonzero(np.r_[True, np.diff(x["Ns"]) != 0])
    ends = np.r_[starts[1:], len(x)]
    rpts, missions, excluded = [], [], []
    for j, a in enumerate(starts):
        if (j + 4 < len(starts)
                and list(x["Ns"][starts[j:j + 5]]) == [3, 4, 5, 6, 7]
                and ends[j] - a > 1000
                and -1000 < np.mean(x["I_mA"][a:ends[j]]) < -200):
            b = int(ends[j + 4])
            rpt = x[a:b]
            if (rpt["Ecell_V"][0] < 4.1 or not 2.45 <= rpt["Ecell_V"][-1] <= 2.55
                    or np.any(np.diff(rpt["QDischarge_mA_h"]) < -1e-6)):
                excluded.append({"rows": [int(a), b], "reason": "incomplete RPT or charge reset"})
            else:
                rpts.append((int(a), b))
        if (a > 0 and j + 3 < len(starts)
                and list(x["Ns"][starts[j:j + 4]]) == [4, 5, 6, 7]
                and np.mean(x["I_mA"][a:ends[j]]) < -1000
                and np.max(np.abs(x["I_mA"][starts[j + 3]:ends[j + 3]])) < 10):
            b = int(ends[j + 3])
            if np.min(x["Ecell_V"][a:b]) < 2.4 or np.max(x["Ecell_V"][a:b]) > 4.25:
                excluded.append({"rows": [int(a), b], "reason": "mission voltage outside source protocol"})
            else:
                missions.append((int(a - 1), b))
    return rpts, missions, excluded


def pulse_samples(rpt):
    starts = np.flatnonzero(np.r_[True, np.diff(rpt["Ns"]) != 0])
    ends = np.r_[starts[1:], len(rpt)]
    pulses = []
    for number in (1, 3):
        a, b = starts[number], ends[number]
        before = a - 1
        delta_i = (-rpt["I_mA"][before] + np.mean(rpt["I_mA"][a:b])) / 1000
        if delta_i <= .3:
            raise ValueError("Invalid low-rate RPT current interruption")
        t = rpt["time_s"][a:b] - rpt["time_s"][before]
        effective_r = (rpt["Ecell_V"][a:b] - rpt["Ecell_V"][before]) / delta_i
        if len(t) < 20 or not 0 < t[0] <= 2 or t[-1] < 29:
            raise ValueError("Incomplete RPT relaxation pulse")
        pulses.append((t, effective_r))
    return pulses


def ocv_curve(theta, lowrate):
    r0, r1, _, offset = theta
    z = np.linspace(0, 1, 31)
    return np.maximum.accumulate(lowrate + .6 * (r0 + r1) + offset * 4 * z * (1 - z))


def replay(theta, mission, capacity_ah, lowrate):
    r0, r1, tau, _ = theta
    curve = ocv_curve(theta, lowrate)
    knots = np.linspace(0, 1, 31)
    current = -mission["I_mA"] / 1000
    soc = np.interp(mission["Ecell_V"][0], curve, knots)
    vp = 0.0
    predicted = []
    for k, time in enumerate(mission["time_s"]):
        if k:
            dt = time - mission["time_s"][k - 1]
            mean_i = .5 * (current[k - 1] + current[k])
            soc -= mean_i * dt / (3600 * capacity_ah)
            decay = np.exp(-dt / tau)
            vp = decay * vp + r1 * (1 - decay) * mean_i
        predicted.append(np.interp(soc, knots, curve) - r0 * current[k] - vp)
    return np.asarray(predicted)


def metrics(error):
    return {"rmse_cell_v": float(np.sqrt(np.mean(error ** 2))),
            "bias_cell_v": float(np.mean(error)),
            "p95_absolute_error_cell_v": float(np.quantile(np.abs(error), .95)),
            "max_absolute_error_cell_v": float(np.max(np.abs(error)))}


def accepted(result):
    c = PROTOCOL["acceptance"]
    return (result["rmse_cell_v"] <= c["rmse_cell_v"]
            and abs(result["bias_cell_v"]) <= c["absolute_bias_cell_v"]
            and result["p95_absolute_error_cell_v"] <= c["p95_absolute_error_cell_v"])


def fit_snapshots(x, dcir, *, pulse_weight=1.0):
    rpts, missions, excluded = windows(x)
    stages = []
    for number, (a, b) in enumerate(rpts):
        stop = rpts[number + 1][0] if number + 1 < len(rpts) else len(x)
        available = [(c, d) for c, d in missions if c >= b and d <= stop]
        if len(available) < 12:
            excluded.append({"rpt_index": number, "rows": [a, b],
                             "reason": "fewer than twelve complete post-RPT missions",
                             "complete_missions": len(available)})
            continue
        rpt = x[a:b]
        capacity = float(rpt["QDischarge_mA_h"][-1] / 1000)
        z = 1 - rpt["QDischarge_mA_h"] / (1000 * capacity)
        lowrate = np.interp(np.linspace(0, 1, 31), z[::-1], rpt["Ecell_V"][::-1])
        pulses = pulse_samples(rpt)

        def pulse_error(p):
            r0, r1, tau = p[:3]
            return np.concatenate([r0 + r1 * (1 - np.exp(-t / tau)) - r
                                   for t, r in pulses])

        prior = least_squares(pulse_error, [.02, .02, 10],
                              bounds=([.001, .0001, .1], [.08, .08, 300]), x_scale="jac")
        samples = [x[c:d] for c, d in available[:8]]

        def residual(p):
            mission_error = np.concatenate([replay(p, s, capacity, lowrate) - s["Ecell_V"]
                                            for s in samples[:4]])
            # Normalize each source group so sample frequency does not silently
            # decide its weight. 30mV mission / 5mOhm pulse scales are declared priors.
            return np.r_[mission_error / (.03 * np.sqrt(len(mission_error))),
                         pulse_weight * pulse_error(p) / (.005 * np.sqrt(sum(len(t) for t, _ in pulses)))]

        fit = least_squares(residual, [*prior.x, 0],
                            bounds=(PROTOCOL["bounds_lower"], PROTOCOL["bounds_upper"]), x_scale="jac")
        r0, r1, tau, offset = map(float, fit.x)
        development = [metrics(replay(fit.x, s, capacity, lowrate) - s["Ecell_V"])
                       for s in samples[4:8]]
        stage = {
            "id": f"rpt_{number:02d}", "rpt_index": number,
            "rpt_rows_start_inclusive_end_exclusive": [a, b],
            "complete_missions_before_rpt": sum(d <= a for c, d in missions),
            "first_fit_delay_after_rpt_s": float(x["time_s"][available[0][0]] - x["time_s"][b - 1]),
            "rpt_temperature_range_c": [float(rpt["Temperature__C"].min()), float(rpt["Temperature__C"].max())],
            "mission_rows": {"fit": [list(w) for w in available[:4]],
                             "development": [list(w) for w in available[4:8]],
                             "holdout": [list(w) for w in available[8:12]]},
            "cell": {"capacity_ah": capacity, "r0_ohm": r0, "r1_ohm": r1,
                     "c1_f": tau / r1, "tau_s": tau, "ocv_offset_v": offset,
                     "ocv_cell_v": ocv_curve(fit.x, lowrate).tolist()},
            "theta": fit.x.tolist(), "lowrate_ocv_proxy_v": lowrate.tolist(),
            "optimizer_success": bool(fit.success), "active_bounds": fit.active_mask.tolist(),
            "jacobian_condition": float(np.linalg.cond(fit.jac)),
            "pulse_fit_ohm_rmse": float(np.sqrt(np.mean(pulse_error(fit.x) ** 2))),
            "published_dcir_ohm": dcir[number] if number < len(dcir) else None,
            "development": development,
            "development_status": "passed" if fit.success and all(accepted(m) for m in development) else "failed",
            "holdout_status": "not_run",
        }
        stages.append(stage)
    initial = stages[0]["cell"]["capacity_ah"]
    for s in stages:
        s["measured_capacity_fraction"] = s["cell"]["capacity_ah"] / initial
    return {
        "schema": "drone.battery-aging/2", "revision": PROTOCOL["revision"],
        "flight_policy": dict(FLIGHT_POLICY),
        "source": {"cell": "VAH01 VTC6 baseline", "file": "VAH01.csv", "sha256": SOURCE_SHA256,
                   "impedance_sha256": IMPEDANCE_SHA256, "license": "CC-BY-4.0",
                   "dataset_doi": "10.1184/R1/14226830.v3", "paper_doi": "10.1038/s41597-023-02180-5"},
        "protocol": dict(PROTOCOL, pulse_weight=pulse_weight),
        "development_status": "passed" if all(s["development_status"] == "passed" for s in stages) else "failed",
        "holdout_status": "not_run", "excluded": excluded, "stages": stages,
        "limitations": [
            "One user-selected baseline cell only; temporal holdout is not independent-cell validation",
            "RPT capacity defines accessible charge; no SOH or RUL estimator",
            "Low-rate voltage is an OCV proxy; one RC cannot uniquely separate every electrochemical mechanism",
            "Pulse and mission parameters are effective approximations; SOC and temperature dependence unresolved",
            "Identical balanced cell-to-pack scaling, fixed interconnect and lumped thermal assumptions",
            "Prescribed linear stage transitions are fault injections, not measured aging kinetics",
            "No real X500 hardware or assembled-pack validation",
        ],
    }


def finalize(catalog, x):
    if catalog["development_status"] != "passed":
        raise ValueError("Development criteria failed; final holdout must remain unopened")
    for stage in catalog["stages"]:
        results = []
        for a, b in stage["mission_rows"]["holdout"]:
            predicted = replay(stage["theta"], x[a:b], stage["cell"]["capacity_ah"],
                               np.asarray(stage["lowrate_ocv_proxy_v"]))
            results.append(metrics(predicted - x["Ecell_V"][a:b]))
        stage["holdout"] = results
        stage["holdout_status"] = "passed" if all(accepted(m) for m in results) else "failed"
    catalog["holdout_status"] = "passed" if all(s["holdout_status"] == "passed" for s in catalog["stages"]) else "failed"
    return catalog


def fit_to_file(*, final=False):
    with ExitStack() as stack:
        paths = [ROOT / ".cache/setup.lock", ROOT.parent / "Data/operations/collection.lock"]
        paths += [ROOT / ".cache/instances" / f"{i}.lock" for i in range(32)]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("CMU fit blocked by active setup/collection/run lock") from exc
        x, dcir = source_data()
        if final:
            catalog = json.loads(CATALOG.read_text())
            catalog = finalize(catalog, x)
        else:
            catalog = fit_snapshots(x, dcir)
        CATALOG.parent.mkdir(parents=True, exist_ok=True)
        if CATALOG.exists():
            history = ROOT / "artifacts/baseline-20261107/catalog-history"
            history.mkdir(parents=True, exist_ok=True)
            saved = history / (hashlib.sha256(CATALOG.read_bytes()).hexdigest() + ".json")
            if not saved.exists():
                saved.write_bytes(CATALOG.read_bytes())
        temporary = CATALOG.with_suffix(".json.writing")
        if temporary.exists() or temporary.is_symlink() or CATALOG.is_symlink():
            raise ValueError("Preserved or unsafe fit output")
        temporary.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
        temporary.replace(CATALOG)
    status = catalog["holdout_status"] if final else catalog["development_status"]
    return {"status": status, "catalog": str(CATALOG), "stages": len(catalog["stages"]),
            "validation": "temporal_holdout" if final else "development", "flight_started": False}
