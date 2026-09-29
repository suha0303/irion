import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "artifacts/remediation-20261014"
checkpoint = json.loads((out / "sensor-campaign-v2.json").read_text())
results = []
for index, row in checkpoint.items():
    run = Path(row["run"])
    scenario = json.loads((run / "evaluation/scenario.json").read_text())
    cfg = json.loads((run / "reference/sensor_profile.json").read_text())["battery"]
    sensor = [
        json.loads(x)
        for x in (run / "evaluation/truth.jsonl.sensor-trace.jsonl").read_text().splitlines()
    ]
    raw = [json.loads(x) for x in (run / "observations/streams.jsonl").read_text().splitlines()]
    battery = [x for x in raw if x["topic"] == "battery_status" and x["sample_time_us"] > 20e6]
    dv = np.array([x["voltage_v"] - x["physical_voltage_v"] for x in sensor])
    di = np.array([x["current_a"] - x["physical_current_a"] for x in sensor])
    delay = np.array([x["publish_s"] - x["sample_s"] for x in sensor])
    available = np.array([(x["available_time_us"] - x["sample_time_us"]) / 1e6 for x in battery])
    gaps = np.diff([x["sample_time_us"] / 1e6 for x in battery])
    counts = [x for x in sensor if cfg["drop_begin_s"] <= x["sample_s"] < cfg["drop_end_s"]]
    checks = dict(
        flight=row["status"] == "completed",
        producer=row["contract"]["status"] == "passed",
        voltage_bias=abs(float(dv.mean()) - cfg["voltage_bias_v"]) < 0.003,
        current_bias=abs(float(di.mean()) - cfg["current_bias_a"]) < 0.006,
        voltage_noise=abs(float(dv.std()) - 0.01) < 0.002,
        current_noise=abs(float(di.std()) - 0.02) < 0.004,
        publish_delay=abs(float(np.median(delay)) - cfg["delay_s"]) < 0.002,
        raw_delay=float(np.median(available)) >= cfg["delay_s"] - 0.002,
        metadata=json.loads((run / "observations/metadata.json").read_text())["semantic_bindings"][
            "pack_voltage_current"
        ]["sensor_profile"]
        == cfg,
    )
    if scenario["name"] == "sensor_dropout_quality":
        checks.update(
            no_samples_in_dropout=len(counts) == 0,
            raw_gap=float(gaps.max()) >= 2.0,
            quality_not_healthy=scenario["health_eligibility"] == "uncertain",
        )
    acceleration = [
        x["fields"]["xyz"]
        for x in raw
        if x["topic"] == "vehicle_acceleration" and 30e6 < x["sample_time_us"] < 80e6
    ]
    avg = np.mean(acceleration, axis=0)
    checks["hover_specific_force"] = bool(abs(avg[2] + 9.8) < 0.3 and np.linalg.norm(avg[:2]) < 0.3)
    results.append(
        dict(
            run=str(run),
            scenario=scenario["name"],
            checks=checks,
            passed=all(checks.values()),
            voltage_mean_error_v=float(dv.mean()),
            voltage_noise_std_v=float(dv.std()),
            current_mean_error_a=float(di.mean()),
            current_noise_std_a=float(di.std()),
            raw_median_capture_age_s=float(np.median(available)),
            largest_sample_gap_s=float(gaps.max()),
            mean_hover_specific_force_frd=avg.tolist(),
        )
    )
report = dict(
    origin="actual_PX4_SITL_sensor_path", passed=all(x["passed"] for x in results), tests=results
)
(out / "sensor-validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 4)
