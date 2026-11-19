"""Score every sealed final flight with a previously frozen new19 bundle."""

from collections import Counter
from pathlib import Path

import numpy as np

from .calibration import flight_decisions, load_calibration
from .data import STEP_US, digest, load_split, read, write
from .inference import replay_forecasts
from .scoring import FlightReference, RunReference, run_components


def evaluate(bundle, runtime, manifest, calibration, output, *, batch_size):
    bundle, output = Path(bundle), Path(output)
    params = load_calibration(bundle, calibration)
    manifest_hash = digest(manifest)
    if read(calibration)["manifest_sha256"] != manifest_hash:
        raise ValueError(
            "Final manifest differs from frozen calibration roster"
        )
    binding = dict(
        manifest_sha256=manifest_hash,
        bundle_sha256=digest(bundle / "bundle.json"),
        weights_sha256=digest(bundle / "model.safetensors"),
        calibration_sha256=digest(calibration),
        cadence_us=STEP_US,
        semantics="SITL stored MAVLink replay; static pre-existing degradation",
    )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "final-plan.json", binding)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir()
    roles = load_split(manifest, ("normal_test", "anomaly_test"))
    inflight = runtime.model.variant.task == "inflight"
    if inflight:
        reference = FlightReference(
            np.array(params["center"]),
            np.array(params["precision"]),
            params["voltage_scale"],
        )
    else:
        reference = RunReference(np.array(params["scale"]), params["threshold"])
    records = []
    for role, runs in roles.items():
        for i, run in enumerate(runs, 1):
            seconds = len(run.time_us) * STEP_US / 1_000_000
            record = dict(
                run_id=run.run_id, role=role, operational_seconds=seconds
            )
            path = prediction_dir / f"{role}-{i:04d}.npz"
            if inflight:
                predictions, count, residuals = replay_forecasts(
                    runtime, run, batch_size
                )
                decisions = list(
                    flight_decisions(
                        run,
                        residuals,
                        reference,
                        params["selected"]["threshold"],
                    )
                )
                valid = np.array(
                    [r["status"] == "available" for r in decisions]
                )
                active = np.array([r["alarm"] is True for r in decisions])
                warned = np.flatnonzero(active)
                scored = np.flatnonzero(valid)
                decision = bool(active.any()) if len(scored) else None
                delay = (
                    float(
                        (run.time_us[warned[0]] - run.time_us[scored[0]]) / 1e6
                    )
                    if len(warned)
                    else None
                )
                record.update(
                    alarm=decision,
                    status="available" if len(scored) else "unavailable",
                    episodes=sum(r["episode_start"] for r in decisions),
                    scored_seconds=float(valid.sum() * STEP_US / 1e6),
                    first_score_to_alarm_s=delay,
                )
                np.savez_compressed(
                    path,
                    time_us=run.time_us,
                    observed=run.observed,
                    predictions=predictions,
                    residuals=residuals,
                    contributors=count,
                    scored=valid,
                    alarm=active,
                    domains=np.asarray(
                        [r.get("domains", (False, False)) for r in decisions],
                        dtype=bool,
                    ),
                )
            else:
                predicted = runtime.voltage(run)
                components = (
                    None
                    if predicted is None
                    else run_components(run, predicted)
                )
                record.update(reference.decide(components))
                if predicted is not None:
                    np.savez_compressed(
                        path,
                        time_us=run.time_us,
                        predicted_cell_v=predicted,
                        observed_cell_v=run.values[:, 17] / 4,
                        observed=run.observed[:, 3],
                    )
            record["predictions_sha256"] = (
                digest(path) if path.is_file() else None
            )
            records.append(record)
        write(output / "completed-runs.json", records)
    if (
        digest(calibration) != binding["calibration_sha256"]
        or digest(bundle / "model.safetensors") != binding["weights_sha256"]
    ):
        raise ValueError(
            "Frozen weights or calibration changed during final evaluation"
        )
    summary = {}
    for role in roles:
        group = [r for r in records if r["role"] == role]
        count = Counter(r["status"] for r in group)
        n_alarm = sum(r["alarm"] is True for r in group)
        summary[role] = dict(
            runs=len(group),
            available=count["available"],
            unavailable=count["unavailable"],
            alarms=n_alarm,
            alarm_run_rate=n_alarm / len(group),
        )
        if role == "anomaly_test":
            summary[role]["not_detected_including_unavailable"] = (
                len(group) - n_alarm
            )
        if inflight:
            scored_hours = sum(r["scored_seconds"] for r in group) / 3600
            exposure_hours = sum(r["operational_seconds"] for r in group) / 3600
            episodes = sum(r["episodes"] for r in group)
            summary[role].update(
                episodes=episodes,
                scored_hours=scored_hours,
                coverage=scored_hours / exposure_hours,
                episodes_per_scored_hour=episodes / scored_hours
                if scored_hours
                else None,
            )
    result = dict(
        status="final_complete",
        locked=binding,
        summary=summary,
        total_runs=len(records),
    )
    write(output / "result.json", result)
    return result
