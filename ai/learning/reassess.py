"""Explicit development re-assessment of immutable forecasts, without model execution.

Changes to matching windows, awareness/new-episode criteria, or fixed threshold
multipliers are exploratory development results. They do not re-fit calibration
or certify independent false-alarm performance. The original result stays intact.
"""

import argparse
import copy
import math
from pathlib import Path

import numpy as np

from capstone_ai.detection import DomainAlarms
from capstone_ai.detection_protocol import evaluation_metrics, event_result, point_metrics
from capstone_ai.detection_replay import records
from execution.common import child, read, sha, verify, write
from execution.evaluate import (
    ResidualSummary,
    default_protocol_manifest,
    forecast_errors,
    protocol_hash,
    summarize_model,
)

DEFAULT_CRITERIA = {
    "event_success": "awareness",
    "max_detection_delay_s": None,
    "threshold_multiplier": [1.0, 1.0],
}


def checked_criteria(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_CRITERIA):
        raise ValueError(
            "Unknown evaluation criterion; only explicit supported fields are accepted"
        )
    result = {**DEFAULT_CRITERIA, **value}
    if result["event_success"] not in ("awareness", "new_episode"):
        raise ValueError("event_success must be awareness or new_episode")
    limit = result["max_detection_delay_s"]
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, (int, float))
        or not math.isfinite(limit)
        or limit < 0
    ):
        raise ValueError("max_detection_delay_s must be null or finite nonnegative seconds")
    factors = np.asarray(result["threshold_multiplier"], dtype=float)
    if factors.shape != (2,) or not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("threshold_multiplier requires positive propulsion,battery multipliers")
    result["threshold_multiplier"] = factors.tolist()
    return result


def transformed_decisions(decisions, high, criteria):
    """Reuse original raw scores/timestamps; never read labels to set thresholds."""
    if criteria["threshold_multiplier"] == [1.0, 1.0]:
        return decisions
    threshold = np.asarray(high) * criteria["threshold_multiplier"]
    previous, detector = None, None
    result = []
    for decision in decisions:
        scope = (decision["run_id"], decision["clock_id"], decision["segment_id"])
        if scope != previous:
            detector, previous = DomainAlarms(threshold), scope
        update = detector.update(decision["scores"], decision["origin_time_us"])
        if decision["scores"] is None:
            update["state"] = decision["state"]
        result.append({**decision, **update})
    return result


def matched_event(event, decisions, observation_start, criteria):
    truth = dict(event)
    limit = criteria["max_detection_delay_s"]
    if limit is not None:
        start = truth.get("observation_start_time_us") or observation_start
        if truth.get("onset_kind") != "pre_existing":
            onset = truth.get("applied_time_us", truth.get("physical_onset_time_us"))
            if onset is not None:
                start = max(start, onset)
        end = start + int(round(limit * 1_000_000))
        truth["end_time_us"] = (
            min(truth["end_time_us"], end) if truth.get("end_time_us") is not None else end
        )
    result = event_result(truth, decisions, observation_start)
    if criteria["event_success"] == "new_episode":
        detected_at = result["new_episode_time_us"]
        result["awareness_detected"] = result["detected"]
        result["detected"] = detected_at is not None
        onset = result["physical_onset_time_us"]
        result["onset_delay_s"] = (
            (detected_at - onset) / 1e6 if detected_at is not None and onset is not None else None
        )
        boundary = truth.get("degradation_boundary_time_us")
        boundary_at = next(
            (
                d["decision_time_us"]
                for d in decisions
                if d["scores"] is not None
                and d["episode_start"]
                and boundary is not None
                and d["decision_time_us"] >= boundary
                and (onset is None or d["decision_time_us"] >= onset)
                and (
                    truth.get("end_time_us") is None
                    or d["decision_time_us"] <= truth["end_time_us"]
                )
            ),
            None,
        )
        result["boundary_delay_s"] = (
            (boundary_at - boundary) / 1e6 if boundary_at is not None else None
        )
        if result["onset_kind"] == "pre_existing":
            start = truth.get("observation_start_time_us") or observation_start
            first = result["first_evaluable_decision_time_us"]
            result["recognition_from_observation_s"] = (
                (detected_at - start) / 1e6 if detected_at is not None else None
            )
            result["recognition_from_first_valid_s"] = (
                (detected_at - first) / 1e6
                if detected_at is not None and first is not None
                else None
            )
    result["pre_event_30_opportunities_met"] = (
        None
        if result["onset_kind"] == "pre_existing"
        else result["pre_onset_valid_decisions"] >= 30
    )
    return result


def reassess(evaluation, output, criteria, *, inference_root=None):
    criteria = checked_criteria(criteria)
    evaluation, output = Path(evaluation).resolve(), Path(output).resolve()
    if output.exists() or output == evaluation or evaluation in output.parents:
        raise ValueError("Use a fresh output outside the immutable base evaluation directory")
    report = read(evaluation / "report.json")
    evidence = read(evaluation / "evidence.json")
    protocol = read(evaluation / "protocol.json")
    if (
        sha(evaluation / "evidence.json") != report.get("evidence_sha256")
        or sha(evaluation / "metrics.json") != report["metrics_sha256"]
        or protocol_hash(protocol) != report.get("protocol_sha256")
        or evidence.get("protocol_sha256") != report.get("protocol_sha256")
        or evidence.get("schema") != "drone.reusable-evaluation/1"
        or evidence.get("final_test_used") is not False
    ):
        raise ValueError("Base evaluation evidence/protocol seal mismatch")
    inference = Path(inference_root or evidence["inference_root"]).resolve()
    if output == inference or inference in output.parents:
        raise ValueError("Use a fresh output outside the immutable inference directory")
    if sha(inference / "result-seal.json") != evidence["inference_seal_sha256"]:
        raise ValueError("Inference result seal changed")
    seal = read(inference / "result-seal.json")
    names = evidence["models"]
    if (
        not names
        or len(set(names)) != len(names)
        or seal["models"] != names
        or seal.get("final_test_used") is not False
        or seal.get("truth_join") is not False
        or seal.get("status") != "TRAIN_CALIBRATION_INFERENCE_COMPLETE"
    ):
        raise ValueError("Expected frozen development inference without truth access")
    rows = evidence["runs"]
    if (
        not rows
        or len({r["run_id"] for r in rows}) != len(rows)
        or any(r["role"] not in ("normal_dev_eval", "anomaly_dev") for r in rows)
    ):
        raise ValueError("Only unique development rows may be reassessed")
    output.mkdir(parents=True, exist_ok=False)
    revised_protocol = {
        **copy.deepcopy(protocol),
        **criteria,
        "base_protocol_sha256": protocol_hash(protocol),
        "base_implementation_sha256": protocol["implementation_sha256"],
        "implementation_sha256": default_protocol_manifest()["implementation_sha256"],
        "reassessment_implementation_sha256": sha(__file__),
        "threshold_fit": "unchanged normal-only calibration; fixed multiplier only",
        "point_metrics": "original event truth intervals, no matching-window truncation",
    }
    write(output / "protocol.json", revised_protocol)
    per_model = {name: [] for name in names}
    all_errors = {name: ResidualSummary() for name in names}
    common_errors = {name: ResidualSummary() for name in names}
    common_count = 0
    calibration = {}
    for name in names:
        key = f"calibration/{name}/calibration.json"
        verify(inference, {key: seal["files"][key]})
        calibration[name] = read(child(inference, key))
    for row in rows:
        truth_path = child(evaluation, row["truth_file"])
        if sha(truth_path) != row["truth_sha256"]:
            raise ValueError("Sealed development truth changed")
        truth = records(truth_path)
        if row["role"] == "normal_dev_eval" and truth:
            raise ValueError("Normal development control has anomaly events")
        first = row["observation_start_us"]
        keyed_models = {}
        for name in names:
            prefix = f"calibration/{name}/{row['run_id']}"
            for file in ("summary.json", "seal.json", "decisions.jsonl", "forecasts.jsonl"):
                key = prefix + "/" + file
                verify(inference, {key: seal["files"][key]})
            replay = child(inference, prefix)
            verify(replay, read(replay / "seal.json")["files"])
            summary = read(replay / "summary.json")
            ds = transformed_decisions(
                records(replay / "decisions.jsonl"), calibration[name]["high"], criteria
            )
            events = [matched_event(event, ds, first, criteria) for event in truth]
            if summary["operational_seconds"] > 0:
                metrics = evaluation_metrics(
                    events, ds, summary["operational_seconds"], summary.get("operational_spans_us")
                )
            else:
                if ds:
                    raise ValueError("Decisions without operational exposure")
                metrics = dict(
                    operational_seconds=0,
                    scored_seconds=0,
                    coverage=None,
                    episodes={k: [] for k in ("union", "battery", "propulsion")},
                    reason="no_observed_airborne_exposure",
                )
            per_model[name].append(
                dict(
                    run_id=row["run_id"],
                    role=row["role"],
                    scenario=row["scenario"],
                    group=row.get("paired_case_id") or row["group_id"],
                    events=events,
                    metrics=metrics,
                    point_metrics=point_metrics(truth, ds, first),
                    forecast_seal=sha(replay / "seal.json"),
                    truth_sha256=sha(truth_path),
                    forecast_errors=forecast_errors(ds),
                    replay_summary=summary,
                )
            )
            keyed = {}
            seen = set()
            for decision in ds:
                key = (decision["clock_id"], decision["segment_id"], decision["origin_time_us"])
                if key in seen:
                    raise ValueError("Duplicate forecast origin")
                seen.add(key)
                if decision["scores"] is not None:
                    keyed[key] = decision["signed_residual"]
            keyed_models[name] = keyed
            all_errors[name].add(list(keyed.values()))
        common = set.intersection(*(set(v) for v in keyed_models.values()))
        common_count += len(common)
        for name in names:
            common_errors[name].add([keyed_models[name][key] for key in sorted(common)])
    metrics = dict(
        evaluation_mode="development_reassessment",
        final_test_used=False,
        seed=report["seed"],
        common_valid_origins=common_count,
        point_adjustment=False,
        protocol_sha256=protocol_hash(revised_protocol),
        models={
            name: summarize_model(
                per_model[name], all_errors[name].result(), common_errors[name].result(), len(rows)
            )
            for name in names
        },
        limitations=[
            "simulation development data, not an independent final test",
            "criteria selected after development exposure are exploratory",
            "no model inference, training, or normal threshold fitting executed",
        ],
    )
    write(output / "metrics.json", metrics)
    write(
        output / "report.json",
        dict(
            status="REASSESSMENT_COMPLETE_STOPPED",
            evaluation_mode="development_reassessment",
            final_test_used=False,
            models=names,
            seed=report["seed"],
            metrics_sha256=sha(output / "metrics.json"),
            protocol_sha256=protocol_hash(revised_protocol),
            base_report=str(evaluation / "report.json"),
            base_report_sha256=sha(evaluation / "report.json"),
            base_evidence_sha256=sha(evaluation / "evidence.json"),
            inference_seal_sha256=evidence["inference_seal_sha256"],
            training_repeated=False,
            inference_repeated=False,
            calibration_repeated=False,
            next_round_authorized=False,
            retries=0,
            llm_calls=0,
        ),
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--criteria",
        required=True,
        type=Path,
        help="JSON: event_success, max_detection_delay_s, threshold_multiplier",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        help="Optional relocated inference directory with the same sealed contents",
    )
    args = parser.parse_args(argv)
    reassess(args.evaluation, args.output, read(args.criteria), inference_root=args.inference_root)
    print(args.output / "report.json")


if __name__ == "__main__":
    main()
