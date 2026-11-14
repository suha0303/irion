"""Compare frozen candidates on disjoint normal and labelled anomaly flights.

This is a regression gate, not an estimate of real-aircraft sensitivity. Known
failure cases remain regression data and never become a fresh final holdout.
"""

from pathlib import Path
from time import perf_counter

import numpy as np

from . import calibration
from .adapt import write
from .analysis import analyze
from .catalog import Catalog, read_json
from .prepared import distinct_flights, load

POLICY = "normal-and-anomaly-regression/1"


def _normal_error(result):
    if result["model_id"].startswith("forecast_"):
        return np.asarray(result["metrics"]["rmse"], dtype=float)
    return np.array(
        [
            np.hypot(
                s["components"]["short_cell_v"], s["components"]["long_cell_v"]
            )
            for s in result["metrics"]["segments"]
        ]
    )


def _event_detection(result, record, event, voltage_scale):
    """Count only alarms inside the declared event, optionally voltage-only."""
    start, end = event["start_us"], event["end_us"]
    target = event.get("target", "any")
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end
        or target not in {"any", "pack_voltage"}
    ):
        raise ValueError(
            "Anomaly event needs integer start/end and known target"
        )
    times = np.asarray(result["scoring"]["time_us"])
    scores = np.asarray(result["scoring"]["scores"])
    if target == "pack_voltage" and result["model_id"].startswith("forecast_"):
        scores = np.concatenate(
            [
                np.abs(np.asarray(s["residual"])[:, 3]) / voltage_scale
                for s in result["metrics"]["series"]
            ]
        )
    covered = (times >= start) & (times <= end)
    alarms = times[covered & (scores > record["threshold"])]
    return dict(
        covered=bool(covered.any()),
        detected=bool(len(alarms)),
        first_alarm_us=int(alarms[0]) if len(alarms) else None,
        delay_seconds=float((alarms[0] - start) / 1e6) if len(alarms) else None,
        target=target,
    )


def review(
    adapted,
    calibrations,
    evaluations,
    output,
    *,
    bundle,
    events=None,
    deadline=None,
):
    """Freeze normal-only thresholds, then check held-out regression flights.

    Args:
        adapted: Completed candidate folder; never changed by this operation.
        calibrations: At least two separate confirmed-normal calibration files.
        evaluations: Separate evaluation manifests with explicit health labels.
        output: New report directory, also containing candidate calibrations.
        bundle: Immutable base model delivery.
        events: Source SHA to labelled anomaly intervals; absent labels keep
            promotion pending. These labels are evaluator-only information.
    """
    events = events or {}

    def check_budget():
        if deadline is not None and perf_counter() >= deadline:
            raise TimeoutError(
                "Review time budget reached; candidate remains inactive"
            )

    check_budget()
    result = read_json(Path(adapted) / "result.json")
    model_id = result["base_model_id"]
    if not 1 <= len(evaluations) <= 20:
        raise ValueError("Select 1..20 separate evaluation flights")
    metadata = [load(p)[0] for p in [*calibrations, *evaluations]]
    distinct_flights(metadata)
    used = set(result["binding"]["source_sha256"])
    if used & {m["source_sha256"] for m in metadata}:
        raise ValueError("Review flights overlap adaptation/development")
    if any(
        m["profile"] != result["binding"]["profile"]
        or m["origin"] != result["binding"]["origin"]
        for m in metadata
    ):
        raise ValueError("Review profile/origin differs from fitted model")
    evaluated = metadata[len(calibrations) :]
    if any(m["role"] != "evaluation" for m in evaluated):
        raise ValueError("Review requires evaluation-role flights")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, candidate in [("original", None), ("adapted", adapted)]:
        check_budget()
        records[name] = calibration.fit(
            out, model_id, calibrations, bundle=bundle, adapted=candidate
        )
    entry = Catalog(bundle).entry(model_id)
    with np.load(entry["files"]["scaler.npz"], allow_pickle=False) as scaler:
        voltage_scale = (
            float(scaler["y_scale"][3]) * metadata[0]["profile"]["series_cells"]
        )
    rows, reasons, missing = [], [], []
    normal_count = anomaly_count = 0
    first_original = first_adapted = None
    for manifest, meta in zip(evaluations, evaluated):
        check_budget()
        pair = {
            "original": analyze(model_id, manifest, bundle=bundle),
            "adapted": analyze(
                model_id, manifest, bundle=bundle, adapted=adapted
            ),
        }
        if first_original is None:
            first_original, first_adapted = pair["original"], pair["adapted"]
        decisions = {}
        for name, analysis in pair.items():
            if analysis["calibration_binding"] != records[name]["binding"]:
                raise ValueError(
                    "Evaluation clock/input policy differs from calibration"
                )
            decisions[name] = calibration.decide(
                records[name],
                analysis["scoring"]["scores"],
                source_sha256=meta["source_sha256"],
            )
        row = dict(
            source_sha256=meta["source_sha256"],
            health=meta["health"],
            decisions=decisions,
        )
        if meta["health"] == "confirmed_normal" and meta.get("health_evidence"):
            normal_count += 1
            before, after = (
                _normal_error(pair[n]) for n in ("original", "adapted")
            )
            row["error_before"], row["error_after"] = (
                before.tolist(),
                after.tolist(),
            )
            # Every predicted quantity must retain its normal-data accuracy.
            # A tiny FP32 roundoff allowance is not a health threshold change.
            if np.any(after > before + np.maximum(1e-7, np.abs(before) * 1e-6)):
                reasons.append("normal_prediction_regression")
            if (
                decisions["adapted"]["exceedances"]
                > decisions["original"]["exceedances"]
            ):
                reasons.append("normal_false_alarm_regression")
        elif meta["health"] == "abnormal" and meta.get("health_evidence"):
            anomaly_count += 1
            annotated = events.get(meta["source_sha256"], [])
            if not annotated:
                missing.append("anomaly_event_timing_or_target_missing")
            row["events"] = []
            for event in annotated:
                detections = {
                    n: _event_detection(
                        pair[n], records[n], event, voltage_scale
                    )
                    for n in pair
                }
                row["events"].append(dict(event=event, **detections))
                before, after = detections["original"], detections["adapted"]
                if not before["covered"] or not after["covered"]:
                    missing.append("anomaly_event_not_scored")
                elif not after["detected"]:
                    reasons.append(
                        "lost_anomaly_detection"
                        if before["detected"]
                        else "known_anomaly_still_missed"
                    )
                elif (
                    before["detected"]
                    and after["first_alarm_us"] > before["first_alarm_us"]
                ):
                    reasons.append("anomaly_detection_delayed")
        else:
            missing.append("evaluation_health_unconfirmed")
        rows.append(row)
    if not normal_count:
        missing.append("normal_evaluation_missing")
    if not anomaly_count:
        missing.append("abnormal_evaluation_missing")
    check = dict(
        policy=POLICY,
        model_sha256=result["model_sha256"],
        status="rejected" if reasons else "pending" if missing else "passed",
        reasons=sorted(set(reasons)),
        missing=sorted(set(missing)),
        normal_flights=normal_count,
        abnormal_flights=anomaly_count,
        calibrations=records,
        flights=rows,
        scope="Known-case regression only; not independent final sensitivity",
    )
    report = dict(
        original=first_original,
        adapted=first_adapted,
        independent=True,
        detection_performance_verified=False,
        promotion_check=check,
    )
    check_budget()
    write(out / "review.json", report)
    return report
