"""Offline evaluator: joins stored forecasts and separate evaluation records."""

import json
import time
from pathlib import Path

import numpy as np

from .inference import Alarm, PendingForecasts
from .source import dump, read_json, sha
from .training import Predictor, normalized


def scores_for(prediction, actual, error_scale):
    z = (actual - prediction) ** 2 / error_scale
    return np.stack([z[..., :3].mean(axis=(1, 2)), z[..., 3].mean(axis=1)], axis=1)


def episodes(scores, ids, origins, high):
    n = 0
    alarm = None
    last = None
    for s, r, t in zip(scores, ids, origins):
        if r != last:
            alarm = Alarm(high)
            last = r
        assert alarm is not None
        n += alarm.update(s, int(t) + 1000000)["episode_start"]
    return n


def flight_bootstrap(episode_counts, exposure_hours):
    if len(episode_counts) < 2:
        return None
    rng = np.random.default_rng(17)
    draws = rng.integers(0, len(episode_counts), (1000, len(episode_counts)))
    rates = np.asarray(episode_counts)[draws].sum(1) / np.asarray(exposure_hours)[draws].sum(1)
    return np.quantile(rates, [0.025, 0.975]).tolist()


def match_event(event, decisions, observation_start):
    kind, onset = event.get("onset_kind", "unknown"), event.get("physical_onset_time_us")
    end = event.get("end_time_us")
    valid = [d for d in decisions if d["scores"] is not None]
    first_valid = min((d["decision_time_us"] for d in valid), default=None)
    previous = [d for d in decisions if onset is not None and d["decision_time_us"] < onset]
    pre_active = bool(previous and previous[-1]["alarm_active"])
    eligible = [
        d
        for d in valid
        if d["state"] == "anomalous"
        and (onset is None or d["decision_time_us"] >= onset)
        and (end is None or d["decision_time_us"] <= end)
        and (not pre_active or d["episode_start"])
    ]
    detected = min((d["decision_time_us"] for d in eligible), default=None)
    start = event.get("observation_start_time_us", observation_start)
    if start is None:
        start = observation_start
    return {
        "event_id": event["event_id"],
        "onset_kind": kind,
        "physical_onset_time_us": onset,
        "detection_time_us": detected,
        "alarm_active_before_onset": pre_active,
        "first_valid_decision_us": first_valid,
        "physical_onset_delay_s": (detected - onset) / 1e6
        if onset is not None and detected is not None
        else None,
        "recognition_from_first_valid_s": (detected - first_valid) / 1e6
        if kind == "pre_existing" and detected is not None and first_valid is not None
        else None,
        "recognition_from_observation_start_s": (detected - start) / 1e6
        if kind == "pre_existing" and detected is not None
        else None,
    }


def calibrate(folder, dataset):
    folder = Path(folder)
    p = Predictor(folder)
    val = normalized(dataset, "validation", p.scaler)
    actual = p.physical(val["Y"])
    pred = p.physical(p.predict_scaled(val["X"], val["C"]))
    error_scale = np.maximum(np.quantile((pred - actual) ** 2, 0.95, axis=0), 1e-8)
    cal = normalized(dataset, "calibration", p.scaler)
    prediction = p.physical(p.predict_scaled(cal["X"], cal["C"]))
    scores = scores_for(prediction, p.physical(cal["Y"]), error_scale)
    hours = len(scores) / 3600
    candidates = []
    for q in [0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999]:
        high = np.maximum(np.quantile(scores, q, axis=0), 1e-8)
        n = episodes(scores, cal["run_id"], cal["origin_us"], high)
        candidates.append(
            {"q": q, "high": high.tolist(), "episodes": n, "false_alarms_per_hour": n / hours}
        )
    chosen = next(
        (c for c in candidates if c["false_alarms_per_hour"] <= 1.0),
        min(candidates, key=lambda x: x["false_alarms_per_hour"]),
    )
    per_flight = []
    for run in dict.fromkeys(cal["run_id"].tolist()):
        idx = cal["run_id"] == run
        per_flight.append(
            {
                "run_id": run,
                "hours": int(idx.sum()) / 3600,
                "episodes": episodes(
                    scores[idx], cal["run_id"][idx], cal["origin_us"][idx], chosen["high"]
                ),
            }
        )
    result = {
        "status": "provisional" if chosen["false_alarms_per_hour"] <= 1 else "target_unmet",
        "error_scale": error_scale.tolist(),
        "high": chosen["high"],
        "chosen": chosen,
        "candidates": candidates,
        "exposure_hours": hours,
        "zero_event_poisson_95_upper_per_hour": 2.996 / hours if chosen["episodes"] == 0 else None,
        "target_false_alarms_per_hour": 1.0,
        "required_zero_alarm_hours_for_poisson_upper_1": 2.996,
        "calibration_run_ids": sorted(set(cal["run_id"].tolist())),
        "flight_bootstrap_95_interval": flight_bootstrap(
            [r["episodes"] for r in per_flight], [r["hours"] for r in per_flight]
        ),
        "per_flight": per_flight,
        "calibration_dataset_sha256": sha(Path(dataset) / "calibration.npz"),
    }
    dump(folder / "calibration.json", result)
    integrity = read_json(folder / "integrity.json")
    integrity["calibration.json"] = sha(folder / "calibration.json")
    dump(folder / "integrity.json", integrity)
    return result


def evaluate(folder, dataset, out):
    folder, dataset, out = Path(folder), Path(dataset), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = Predictor(folder)
    cal = read_json(folder / "calibration.json")
    test = normalized(dataset, "test", p.scaler)
    manifest = read_json(dataset / "manifest.json")
    rows = {r["run_id"]: r for r in manifest["runs"]}
    prediction_path = out / "forecasts.jsonl"
    decision_path = out / "decisions.jsonl"
    records = []
    latencies = []
    with prediction_path.open("w") as fw, decision_path.open("w") as dw:
        for run_id in dict.fromkeys(test["run_id"].tolist()):
            indices = np.flatnonzero(test["run_id"] == run_id)
            store = PendingForecasts(cal["error_scale"], cal["high"])
            for idx in indices:
                origin = int(test["origin_us"][idx])
                segment = int(test["segment"][idx])
                started = time.perf_counter()
                forecast = p.physical(
                    p.predict_scaled(test["X"][idx : idx + 1], test["C"][idx : idx + 1])
                )[0]
                elapsed = time.perf_counter() - started
                latencies.append(elapsed)
                # Offline logical issuance, not a live latency claim.
                key = store.issue(run_id, "px4-boot", segment, origin, origin, forecast)
                fw.write(
                    json.dumps(
                        {
                            **store.forecasts[key],
                            "execution": "offline_replay",
                            "compute_seconds": elapsed,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                fw.flush()
                actual = p.physical(test["Y"][idx : idx + 1])[0]
                for j, target in enumerate(actual):
                    t = origin + (j + 1) * 100000
                    store.observe(run_id, "px4-boot", segment, t, t, target.tolist())
                    store.advance(t)
                decision = store.decisions[key]
                dw.write(json.dumps(decision, allow_nan=False) + "\n")
            decisions = list(store.decisions.values())
            n = sum(d["episode_start"] for d in decisions)
            events = [
                json.loads(s)
                for s in (Path(rows[run_id]["path"]) / "evaluation/events.jsonl")
                .read_text()
                .splitlines()
                if s.strip()
            ]
            canonical = read_json(dataset / "canonical" / run_id / "summary.json")
            with (dataset / "canonical" / run_id / "frames.jsonl").open() as frame_file:
                observation_start = json.loads(next(frame_file))["time_us"]
            event_results = [match_event(event, decisions, observation_start) for event in events]
            residual = np.array(
                [d["signed_residual"] for d in decisions if d["signed_residual"] is not None]
            )
            canonical = read_json(dataset / "canonical" / run_id / "summary.json")
            exposure = len(decisions) / 3600
            records.append(
                {
                    "run_id": run_id,
                    "health_eligibility": rows[run_id]["health_eligibility"],
                    "paired_case_id": rows[run_id]["paired_case_id"],
                    "scenario": rows[run_id].get("scenario", "not_declared"),
                    "decisions": len(decisions),
                    "valid_exposure_hours": exposure,
                    "alarm_episodes": n,
                    "false_alarms_per_hour": n / exposure
                    if rows[run_id]["health_eligibility"] == "confirmed_healthy"
                    else None,
                    "mse_physical": (residual**2).mean(axis=(0, 1)).tolist(),
                    "events": event_results,
                    "window_target_coverage": len(decisions) / max(1, canonical["windows"]),
                    "preprocessing_exclusions": canonical["exclusions"],
                }
            )
    normal = [r for r in records if r["health_eligibility"] == "confirmed_healthy"]
    hours = sum(r["valid_exposure_hours"] for r in normal)
    alarms = sum(r["alarm_episodes"] for r in normal)
    rng = np.random.default_rng(17)
    bootstrap = []
    for _ in range(1000):
        sample = [normal[i] for i in rng.integers(0, len(normal), len(normal))]
        bootstrap.append(
            sum(r["alarm_episodes"] for r in sample)
            / sum(r["valid_exposure_hours"] for r in sample)
        )
    result = {
        "purpose": "independent_run_local_small_training_evaluation_not_final_research",
        "model": p.name,
        "independent_test_groups": len({r["paired_case_id"] for r in records}),
        "independent_abnormal_groups": len(
            {r["paired_case_id"] for r in records if r["health_eligibility"] == "abnormal"}
        ),
        "dataset_manifest_sha256": sha(dataset / "manifest.json"),
        "bundle_integrity_sha256": sha(folder / "integrity.json"),
        "calibration_status": cal["status"],
        "normal_test_hours": hours,
        "normal_alarm_episodes": alarms,
        "false_alarms_per_hour": alarms / hours,
        "flight_bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "zero_alarm_poisson_95_upper_per_hour": 2.996 / hours if alarms == 0 else None,
        "run_results": records,
        "compute_seconds_p50_p95": np.quantile(latencies, [0.5, 0.95]).tolist(),
        "live_forecast_timing": "not_executed; timestamps are offline logical replay clock",
        "forecast_sha256": sha(prediction_path),
        "decisions_sha256": sha(decision_path),
        "limitations": [
            "Short normal exposure; thresholds provisional.",
            "No SOH/RUL or healthy certification.",
            "Constant degraded trajectories can have small persistence error.",
            "Missing targets excluded from dataset; coverage and exclusions retained.",
        ],
    }
    dump(out / "evaluation.json", result)
    return result
