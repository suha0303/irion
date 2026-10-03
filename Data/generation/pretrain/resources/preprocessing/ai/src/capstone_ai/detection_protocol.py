"""Normal-only calibration and sealed-record evaluation, independent of model fitting."""

import math
from collections import Counter
from typing import Any

import numpy as np

from .detection import DOMAINS, DomainAlarms

QUANTILES = [.95, .975, .99, .995, .999, .9995, .9999]


def choose_threshold(candidates):
    ordered = sorted(candidates, key=lambda row: row["q"])
    return next((c for c in ordered if c["false_alarms_per_hour"] <= 1),
                min(ordered, key=lambda c: c["false_alarms_per_hour"]))


def score_sequence(scores, run_ids, segments, origins, high):
    alarm, scope, previous = None, None, None
    decisions = []
    for score, run, segment, origin in zip(scores, run_ids, segments, origins):
        current = (str(run), int(segment))
        if current != scope:
            alarm, scope, previous = DomainAlarms(high), current, None
        if previous is not None and origin - previous < 1000000:
            raise ValueError("duplicate or overlapping calibration blocks")
        previous = origin
        assert alarm is not None
        decisions.append({"run_id": str(run), "segment_id": int(segment),
                          "origin_time_us": int(origin), "decision_time_us": int(origin) + 1000000,
                          "scores": np.asarray(score).tolist(), **alarm.update(score, int(origin))})
    return decisions


def grouped_rate_interval(records, group_key="group", seed=17):
    totals: dict[str, list[float]] = {}
    for row in records:
        counts, hours = totals.setdefault(row[group_key], [0, 0.0])
        totals[row[group_key]] = [counts + row["episodes"], hours + row["hours"]]
    if not totals or any(h <= 0 for _, h in totals.values()):
        return {"groups": len(totals), "interval": None, "reason": "no_valid_exposure"}
    if len(totals) < 2:
        return {"groups": len(totals), "interval": None, "reason": "insufficient_independent_groups"}
    values = np.array(list(totals.values()))
    draws = np.random.default_rng(seed).integers(0, len(values), (1000, len(values)))
    samples = values[draws].sum(axis=1)
    rates = samples[:, 0] / samples[:, 1]
    return {"groups": len(totals), "interval": np.quantile(rates, [.025, .975]).tolist(),
            "reason": "zero_event_bootstrap_is_degenerate_not_a_safety_bound"
            if np.all(values[:, 0] == 0) else None}


def fit_calibration(validation_residual_scaled, calibration_residual_scaled, run_ids, segments,
                    origins, physical_target_scale, groups):
    val, cal = [np.asarray(v, dtype=float) for v in
                (validation_residual_scaled, calibration_residual_scaled)]
    if any(a.ndim != 3 or a.shape[1:] != (10, 4) or len(a) == 0 or not np.isfinite(a).all()
           for a in [val, cal]):
        raise ValueError("finite nonempty normal validation/calibration residuals required")
    if not (len(cal) == len(run_ids) == len(segments) == len(origins)):
        raise ValueError("calibration identity length mismatch")
    scale = np.maximum(np.quantile(val**2, .95, axis=0), 1e-8)
    contribution = cal**2 / scale
    scores = np.stack([contribution[:, :, :3].mean((1, 2)), contribution[:, :, 3].mean(1)], axis=1)
    hours = len(cal) / 3600
    candidates = []
    for q in QUANTILES:
        high = np.maximum(np.quantile(scores, q, axis=0), 1e-8)
        sequence = score_sequence(scores, run_ids, segments, origins, high)
        count = sum(d["episode_start"] for d in sequence)
        candidates.append({"q": q, "high": high.tolist(), "union_episodes": count,
                           "false_alarms_per_hour": count / hours,
                           "domain_episodes": {name: sum(d["domains"][name]["episode_start"]
                                                         for d in sequence) for name in DOMAINS}})
    chosen = choose_threshold(candidates)
    decisions = score_sequence(scores, run_ids, segments, origins, chosen["high"])
    per_run = []
    for run in dict.fromkeys(str(r) for r in run_ids):
        part = [d for d in decisions if d["run_id"] == run]
        per_run.append({"run_id": run, "group": groups[run], "hours": len(part) / 3600,
                        "episodes": sum(d["episode_start"] for d in part)})
    physical = np.asarray(physical_target_scale, dtype=float)
    if physical.shape != (4,) or not np.isfinite(physical).all() or np.any(physical <= 0):
        raise ValueError("positive physical target transform required")
    zero_upper = -math.log(.05) / hours if chosen["union_episodes"] == 0 else None
    return {
        "revision": "normal-postprocess-v7", "status": "target_unmet"
        if chosen["false_alarms_per_hour"] > 1 else "provisional",
        "reference_error_scale_scaled": scale.tolist(),
        "error_scale_physical": (scale * physical[None, :]**2).tolist(),
        "scale_fit": "normal validation only, q95 per lead/target, scaled floor1e-8",
        "high": chosen["high"], "chosen": chosen, "candidates": candidates,
        "valid_exposure_hours": hours, "per_run": per_run,
        "group_bootstrap": grouped_rate_interval(per_run),
        "poisson_zero_event_95_upper_per_hour": zero_upper,
        "poisson_assumption": "independent constant-rate episodes; reference, not a guarantee",
        "additional_zero_event_hours_for_reference_upper_1": max(0, -math.log(.05) - hours),
        "independent_false_alarm_target_verified": False,
        "reason": "threshold-fit data cannot certify independent false-alarm performance",
    }


def event_result(event, decisions, observation_start):
    ordered = sorted(decisions, key=lambda d: (d["decision_time_us"], d["origin_time_us"]))
    kind = event.get("onset_kind", "unknown")
    onset = event.get("physical_onset_time_us")
    applied = event.get("applied_time_us", onset)
    onset = None if kind == "pre_existing" else applied
    start = event.get("observation_start_time_us") or observation_start
    if onset is not None:
        start = max(start, onset)
    end = event.get("end_time_us")
    opportunity = [d for d in ordered if d["scores"] is not None and d["decision_time_us"] >= start
                   and (end is None or d["decision_time_us"] <= end)]
    before = [d for d in ordered if onset is not None and d["decision_time_us"] < onset]
    pre_active = bool(before and before[-1]["alarm_active"])
    aware = next((d["decision_time_us"] for d in opportunity if d["alarm_active"]), None)
    new = next((d["decision_time_us"] for d in opportunity if d["episode_start"]), None)
    first = opportunity[0]["decision_time_us"] if opportunity else None
    boundary = event.get("degradation_boundary_time_us")
    boundary_detection = next((d["decision_time_us"] for d in opportunity
                               if boundary is not None and d["decision_time_us"] >= boundary
                               and d["alarm_active"]), None)
    return {"event_id": event["event_id"], "onset_kind": kind, "physical_onset_time_us": onset,
            "preexisting_alarm": pre_active, "awareness_time_us": aware, "new_episode_time_us": new,
            "first_evaluable_decision_time_us": first, "valid_opportunities": len(opportunity),
            "pre_onset_valid_decisions": sum(d["scores"] is not None for d in before),
            "onset_to_first_evaluable_s": (first - onset) / 1e6
            if first is not None and onset is not None else None,
            "insufficient_data": not opportunity, "detected": aware is not None,
            "new_episode_detected": new is not None,
            "onset_delay_s": (aware - onset) / 1e6 if onset is not None and aware is not None else None,
            "recognition_from_observation_s": (aware - start) / 1e6
            if kind == "pre_existing" and aware is not None else None,
            "recognition_from_first_valid_s": (aware - first) / 1e6
            if kind == "pre_existing" and aware is not None and first is not None else None,
            "boundary_delay_s": (boundary_detection - boundary) / 1e6
            if boundary_detection is not None else None,
            "unavailable_reasons": dict(Counter(reason for d in ordered for reason in
                                                d.get("unscored_reasons", []))) if not opportunity else {}}


def episode_intervals(decisions):
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in ["union", *DOMAINS]}
    open_episodes = {}
    previous, scope = None, None
    for d in decisions:
        current = (d["run_id"], d["clock_id"], d["segment_id"])
        if scope is not None and current != scope:
            for name, start in open_episodes.items():
                result[name].append({**start, "end_us": previous, "censored": True,
                                     "end_reason": "segment_reset"})
            open_episodes.clear()
        scope = current
        for name in result:
            state = d if name == "union" else d["domains"][name]
            if state["episode_start"]:
                open_episodes[name] = {"start_us": d["decision_time_us"], "scope": list(scope)}
            if name in open_episodes and not state["alarm_active"]:
                result[name].append({**open_episodes.pop(name), "end_us": d["decision_time_us"],
                                     "censored": False, "end_reason": "hysteresis_exit"})
        previous = d["decision_time_us"]
    for name, start in open_episodes.items():
        result[name].append({**start, "end_us": previous, "censored": True, "end_reason": "run_end"})
    return result


def evaluation_metrics(events, decisions, operational_seconds, operational_spans=None):
    if operational_seconds <= 0:
        raise ValueError("positive operational exposure required")
    valid = [d for d in decisions if d["scores"] is not None]
    identities = [(d["run_id"], d["clock_id"], d["segment_id"], d["origin_time_us"]) for d in decisions]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate decision exposure")
    last: dict[tuple[str, str, int], int] = {}
    for run, clock, segment, origin in identities:
        scope = (run, clock, segment)
        if scope in last and origin - last[scope] < 1000000:
            raise ValueError("overlapping or unordered decision exposure")
        last[scope] = origin
    eligible = sum(not e["insufficient_data"] for e in events)
    scored = float(len(valid))
    if operational_spans is not None:
        scored = sum(max(0, min(d["origin_time_us"] + 1000000, operational_spans[d["clock_id"]][1])
                         - max(d["origin_time_us"], operational_spans[d["clock_id"]][0])) / 1e6
                     for d in valid if d["clock_id"] in operational_spans)
    if scored > operational_seconds + 1e-9:
        raise ValueError("valid coverage exceeds operational exposure")
    return {"events_total": len(events), "events_detected": sum(e["detected"] for e in events),
            "events_with_opportunity": eligible,
            "events_insufficient_data": len(events) - eligible,
            "recall_all": sum(e["detected"] for e in events) / len(events) if events else None,
            "recall_conditional": sum(e["detected"] for e in events) / eligible if eligible else None,
            "new_episode_recall_all": sum(e["new_episode_detected"] for e in events) / len(events)
            if events else None,
            "operational_seconds": operational_seconds, "scored_seconds": scored,
            "valid_decisions": len(valid), "coverage": scored / operational_seconds,
            "unscored_forecasts": len(decisions) - len(valid),
            "unscored_reasons": dict(Counter(r for d in decisions for r in d["unscored_reasons"])),
            "episodes": episode_intervals(decisions)}


def point_metrics(events, decisions, observation_start):
    counts = dict.fromkeys(["tp", "fp", "fn", "tn", "unavailable", "unlabeled"], 0)
    intervals = []
    for event in events:
        start = event.get("observation_start_time_us", observation_start) if event.get("onset_kind") == "pre_existing" \
            else event.get("applied_time_us", event.get("physical_onset_time_us"))
        if start is None:
            return {**counts, "unlabeled": len(decisions), "precision": None, "recall": None,
                    "f1": None, "reason": "unknown event onset; point truth undefined"}
        intervals.append((start, event.get("end_time_us")))
    for decision in decisions:
        if decision["scores"] is None:
            counts["unavailable"] += 1
            continue
        t = decision["decision_time_us"]
        actual = any(start <= t and (end is None or t <= end) for start, end in intervals)
        predicted = decision["alarm_active"]
        counts["tp" if actual and predicted else "fn" if actual else "fp" if predicted else "tn"] += 1
    tp, fp, fn = [counts[n] for n in ["tp", "fp", "fn"]]
    return {**counts, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "point_adjustment": False, "time_axis": "actual decision time"}
