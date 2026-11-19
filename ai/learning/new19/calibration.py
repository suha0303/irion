"""Fit new alarm parameters from normal validation and calibration only."""

from pathlib import Path

import numpy as np

from .data import STEP_US, digest, load_split, read, write
from .inference import replay_forecasts
from .scoring import FlightAlarm, FlightReference, RunReference, run_components


def flight_decisions(run, residuals, reference, thresholds):
    return decisions_from_scores(
        run, reference.score_series(residuals), thresholds
    )


def decisions_from_scores(run, scores, thresholds):
    alarm = FlightAlarm(thresholds)
    previous = None
    active = False
    for t, segment, r in zip(run.time_us, run.segment, scores, strict=True):
        if segment != previous:
            alarm = FlightAlarm(thresholds)
            active = False
            previous = segment
        score = r if np.isfinite(r).all() else None
        result = alarm.update(int(t), score)
        result.update(
            time_us=int(t),
            scores=None if score is None else score.tolist(),
            episode_start=result["alarm"] is True and not active,
        )
        if result["alarm"] is not None:
            active = result["alarm"]
        yield result


def calibrate(bundle, runtime, manifest, output, *, batch_size=64):
    output = Path(output)
    manifest = Path(manifest)
    fit = read(Path(bundle) / "fit-plan.json")
    if digest(manifest) != fit.get("manifest_sha256"):
        raise ValueError("Calibration must use the same sealed role manifest")
    roles = load_split(manifest, ("normal_validation", "normal_calibration"))
    val, cal = (
        roles[role] for role in ("normal_validation", "normal_calibration")
    )
    if runtime.model.variant.task == "whole-run":

        def components(runs):
            for run in runs:
                prediction = runtime.voltage(run)
                row = (
                    None
                    if prediction is None
                    else run_components(run, prediction)
                )
                if row is None:
                    raise ValueError(
                        f"Unscorable normal calibration or validation run: {run.run_id}"
                    )
                yield row[0]

        reference = RunReference.fit(
            list(components(val)), list(components(cal))
        )
        parameters = dict(
            scale=reference.scale.tolist(),
            threshold=reference.threshold,
            alpha=0.05,
        )
    else:
        validation_residuals = np.concatenate(
            [replay_forecasts(runtime, run, batch_size)[2] for run in val]
        )
        reference = FlightReference.fit(
            validation_residuals[np.isfinite(validation_residuals).all(1)]
        )
        raw, scores = [], []
        for run in cal:
            residuals = replay_forecasts(runtime, run, batch_size)[2]
            run_scores = reference.score_series(residuals)
            matched = run_scores[np.isfinite(run_scores).all(1)]
            if not len(matched):
                raise ValueError(
                    f"No valid calibration evidence in {run.run_id}"
                )
            raw.append((run, run_scores))
            scores.append(matched)
        targets = np.concatenate(scores)
        hours = len(targets) * STEP_US / 3_600_000_000
        candidates = []
        for q in (0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999):
            threshold = np.maximum(np.quantile(targets, q, axis=0), 1e-10)
            episodes = sum(
                decision["episode_start"]
                for run, scores in raw
                for decision in decisions_from_scores(run, scores, threshold)
            )
            candidates.append(
                dict(
                    quantile=q,
                    threshold=threshold.tolist(),
                    episodes_per_hour=episodes / hours,
                )
            )
            if episodes / hours <= 1:
                break
        selected = min(
            candidates,
            key=lambda row: (
                row["episodes_per_hour"] > 1,
                row["quantile"]
                if row["episodes_per_hour"] <= 1
                else row["episodes_per_hour"],
            ),
        )
        parameters = dict(
            center=reference.center.tolist(),
            precision=reference.precision.tolist(),
            voltage_scale=reference.voltage_scale,
            selected=selected,
            candidates=candidates,
            scored_hours=hours,
            timing="offline_10hz",
        )
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema="drone-ai.new19.calibration/1",
        bundle_sha256=digest(Path(bundle) / "bundle.json"),
        parameters=parameters,
        manifest_sha256=digest(manifest),
        validation_runs=len(val),
        calibration_runs=len(cal),
    )
    write(output / "calibration.json", result)
    return result


def load_calibration(bundle, path):
    result = read(path)
    if result.get("schema") != "drone-ai.new19.calibration/1" or result.get(
        "bundle_sha256"
    ) != digest(Path(bundle) / "bundle.json"):
        raise ValueError("Calibration belongs to a different model bundle")
    return result["parameters"]
