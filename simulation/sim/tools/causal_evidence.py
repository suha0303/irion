"""Offline physics/producer verification, never AI preprocessing."""

import json
from pathlib import Path

import numpy as np
from capstone_sim.recording import dump
from pyulog import ULog

ROOT = Path(__file__).resolve().parents[2]
cp = json.loads((ROOT / "artifacts/campaign-v2.json").read_text())
final = json.loads((ROOT / "artifacts/campaign-final.json").read_text())


def load(row):
    run = Path(row["run"])
    truth = [json.loads(s) for s in (run / "evaluation/truth.jsonl").read_text().splitlines()]
    return run, truth


def stats(truth, a, b):
    rows = [r for r in truth if a <= r["time_us"] / 1e6 <= b]
    out = {
        k: float(np.mean([r[k] for r in rows]))
        for k in ["voltage_v", "current_a", "altitude_m", "soc"]
    }
    out.update(
        command=float(np.mean([r["command"] for r in rows])),
        thrust_n=float(np.mean([sum(r["thrust_n"]) for r in rows])),
    )
    return out


comparison = {}
for key, i in [
    ("healthy", 0),
    ("r0_existing", 1),
    ("low_soc_healthy", 6),
    ("propulsion_existing", 7),
    ("payload_healthy", 10),
]:
    run, truth = load(cp[str(i)])
    comparison[key] = dict(run=str(run), means_30_90s=stats(truth, 30, 90))
capacity = {}
for key, i in [("healthy", 14), ("reduced", 15)]:
    run, truth = load(cp[str(i)])
    first, last = truth[0], truth[-1]
    capacity[key] = dict(
        run=str(run),
        initial_soc=first["soc"],
        final_soc=last["soc"],
        capacity_ah=first["capacity_ah"],
        soc_used=first["soc"] - last["soc"],
        charge_ah=(first["soc"] - last["soc"]) * first["capacity_ah"],
        r0_ohm=first["r0_ohm"],
        mean_final_60s=stats(truth, 350, 410),
    )
changes = []
for i in [1, 2, 3, 8]:
    if str(i) not in final:
        continue
    run, truth = load(final[str(i)])
    events = [json.loads(s) for s in (run / "evaluation/events.jsonl").read_text().splitlines()]
    changes.append(
        dict(
            run=str(run),
            before=stats(truth, 40, 55),
            after=stats(truth, 85, 100),
            events=events,
            bus_residual_v=max(abs(r["solver_residual_v"]) for r in truth),
            motor_energy_relative=max(
                abs(r["energy_residual_j"]) / r["bus_energy_j"]
                for r in truth
                if r["bus_energy_j"] > 10
            ),
            battery_energy_residual_j=max(abs(r["battery_energy_residual_j"]) for r in truth),
        )
    )
run, truth = load(final["0"])
events = [json.loads(s) for s in (run / "observations/streams.jsonl").read_text().splitlines()]
tt = np.array([r["time_us"] for r in truth])
# Capture latency and topic gaps retain original timestamps; no resampled data is exported.
topics = {}
for topic in sorted({e["topic"] for e in events}):
    es = [e for e in events if e["topic"] == topic]
    ts = np.array([e["sample_time_us"] for e in es])
    delay = np.array([e["available_time_us"] - e["sample_time_us"] for e in es])
    topics[topic] = dict(
        count=len(es),
        median_period_us=float(np.median(np.diff(ts))),
        max_period_us=int(np.max(np.diff(ts))),
        p99_capture_delay_us=float(np.quantile(delay, 0.99)),
        max_capture_delay_us=int(delay.max()),
        null_fields=sum(
            v is None or isinstance(v, list) and None in v for e in es for v in e["fields"].values()
        ),
    )
bs = [
    e
    for e in events
    if e["topic"] == "battery_status" and 30000000 <= e["sample_time_us"] <= 90000000
]
idx = np.searchsorted(tt, [e["sample_time_us"] for e in bs]).clip(0, len(truth) - 1)
battery_error = {
    k: float(
        np.sqrt(
            np.mean([(e["fields"][k] - truth[j][k]) ** 2 for e, j in zip(bs, idx, strict=True)])
        )
    )
    for k in ["voltage_v", "current_a"]
}
request = [e for e in events if e["topic"] == "actuator_outputs" and e["sample_time_us"] > 10000000]
ix = np.searchsorted(tt, [e["sample_time_us"] for e in request]).clip(0, len(truth) - 1)
command_rms = float(
    np.sqrt(
        np.mean(
            [
                (np.array(e["fields"]["output"]) / 1000 - np.array(truth[j]["command"])) ** 2
                for e, j in zip(request, ix, strict=True)
            ]
        )
    )
)
steady_acc = [
    e["fields"]["xyz"]
    for e in events
    if e["topic"] == "vehicle_acceleration" and 30000000 <= e["sample_time_us"] <= 90000000
]
mean_acc = np.mean(steady_acc, axis=0).tolist()
ulogs = list((run / "px4").rglob("*.ulg"))
u = ULog(str(ulogs[0]), message_name_filter_list=[])
params = {
    key: u.initial_parameters.get(key)
    for key in [
        "SIM_BAT_ENABLE",
        "BAT1_N_CELLS",
        "BAT1_CAPACITY",
        "BAT1_V_EMPTY",
        "BAT1_V_CHARGED",
        "CA_ROTOR0_PX",
        "CA_ROTOR0_PY",
    ]
}
healthy = comparison["healthy"]["means_30_90s"]
r0 = comparison["r0_existing"]["means_30_90s"]
prop = comparison["propulsion_existing"]["means_30_90s"]
checks = dict(
    r0_load_voltage_drop=r0["voltage_v"] < healthy["voltage_v"] - 0.15,
    r0_controller_compensation=r0["command"] > healthy["command"],
    propulsion_increased_loss=prop["current_a"] > healthy["current_a"],
    hover_force_balance=abs(healthy["thrust_n"] - (2 + 4 * 0.016076923) * 9.80665) < 1,
    pack_sensor_voltage=battery_error["voltage_v"] < 0.05,
    pack_sensor_current=battery_error["current_a"] < 0.2,
    postmapping_command_match=command_rms < 0.03,
    specific_force_hover=abs(mean_acc[2] + 9.80665) < 0.2,
    capacity_isolated=capacity["healthy"]["r0_ohm"] == capacity["reduced"]["r0_ohm"],
    capacity_effect=capacity["reduced"]["soc_used"] > capacity["healthy"]["soc_used"] * 1.3,
    sufficient_capacity_exposure=capacity["healthy"]["charge_ah"] > 1,
    fake_battery_disabled=params["SIM_BAT_ENABLE"] == 0,
    actual_vertical_target=all(
        e["fields"]["vz"] is not None
        for e in events
        if e["topic"] == "vehicle_local_position_setpoint"
        and 30000000 <= e["sample_time_us"] <= 90000000
    ),
    all_fault_forms=len(changes) == 4 and all(c["events"] for c in changes),
)
report = dict(
    origin="px4_sitl",
    passed=all(checks.values()),
    checks=checks,
    paired=comparison,
    capacity=capacity,
    changes=changes,
    sensor_comparison_run=str(run),
    battery_sensor_rmse=battery_error,
    command_rms=command_rms,
    hover_specific_force_frd=mean_acc,
    topics=topics,
    ulog_initial_parameters=params,
    C9="deferred_ai_integration",
)
dump(ROOT / "artifacts/causal-evidence.json", report)
print(
    json.dumps(
        dict(
            passed=report["passed"],
            checks=checks,
            battery_sensor_rmse=battery_error,
            command_rms=command_rms,
        ),
        indent=2,
    )
)
raise SystemExit(0 if report["passed"] else 4)
