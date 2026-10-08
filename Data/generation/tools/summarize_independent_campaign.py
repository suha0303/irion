"""Keep every planned case and failed attempt visible beside frozen v7 metrics."""

import argparse
import json
import math
from pathlib import Path


def account(lock, attempts, comparison):
    cases = {c["case_id"]: c for c in lock["cases"]}
    by_case = {key: [] for key in cases}
    seen = set()
    for attempt in attempts:
        case = attempt["case_id"]
        if case not in cases:
            raise ValueError("unplanned case in evaluation campaign")
        if attempt["run_id"] in seen:
            raise ValueError("duplicate attempt identity")
        seen.add(attempt["run_id"])
        by_case[case].append(attempt)
    if any(len(rows) > 2 for rows in by_case.values()):
        raise ValueError("locked maximum one infrastructure retry exceeded")
    all_rows = {a["run_id"]: a for a in attempts}
    models = {}
    for model, result in comparison.items():
        evaluated = {r["run_id"]: r for r in result["per_run"]}
        if len(evaluated) != len(result["per_run"]) or set(evaluated) - seen:
            raise ValueError("duplicate or unregistered evaluated run")
        slots, detected, missed, unavailable = 0, 0, 0, 0
        ledger = []
        for case, rows in by_case.items():
            abnormal = cases[case]["scenario"] != "hover_normal"
            if abnormal:
                slots += max(1, len(rows))
            if not rows:
                unavailable += int(abnormal)
                ledger.append({"case_id": case, "run_id": None, "status": "not_collected",
                               "planned_abnormal": abnormal})
            for attempt in rows:
                row = evaluated.get(attempt["run_id"])
                entry = {**attempt, "planned_abnormal": abnormal}
                if row is None:
                    unavailable += int(abnormal)
                    entry["evaluation_status"] = "unavailable_retained"
                else:
                    events = row["events"]
                    if len(events) != int(abnormal):
                        raise ValueError("truth event count differs from locked one-event case")
                    if abnormal:
                        event = events[0]
                        detected += int(event["detected"])
                        missed += int(not event["detected"] and not event["insufficient_data"])
                        unavailable += int(event["insufficient_data"])
                        entry["event"] = event
                    entry["evaluation_status"] = "evaluated"
                    entry["valid_seconds"] = row["metrics"]["scored_seconds"]
                    entry["operational_seconds"] = row["metrics"]["operational_seconds"]
                ledger.append(entry)
        normal = [r for r in evaluated.values()
                  if cases[all_rows[r["run_id"]]["case_id"]]["scenario"] == "hover_normal"]
        seconds = sum(r["metrics"]["scored_seconds"] for r in normal)
        alarms = sum(len(r["metrics"]["episodes"]["union"]) for r in normal)
        events = [e for r in evaluated.values() for e in r["events"]]
        inflight = [e for e in events if e["onset_kind"] in {"in_flight_step", "in_flight_ramp"}]
        if detected + missed + unavailable != slots:
            raise ValueError("intent-to-evaluate denominator mismatch")
        models[model] = {
            "planned_cases": len(cases), "actual_attempts": len(attempts),
            "unevaluated_attempts": sorted(seen - set(evaluated)),
            "uncollected_cases": [c for c, rows in by_case.items() if not rows],
            "intended_abnormal_attempt_slots": slots, "detected_attempts": detected,
            "missed_with_opportunity": missed, "unavailable_abnormal_slots": unavailable,
            "detection_rate_all_attempt_slots": detected / slots if slots else None,
            "normal_scored_seconds": seconds, "normal_union_episodes": alarms,
            "normal_poisson_zero_event_95_upper_per_hour": -math.log(.05) * 3600 / seconds
            if seconds > 0 and alarms == 0 else None,
            "poisson_assumption": "independent constant-rate episodes; reference only",
            "normal_exposure_scope": "three hover paired groups, not diverse or long-duration qualification",
            "new_episode_detections": sum(e["new_episode_detected"] for e in events),
            "preexisting_alarm_events": sum(e["preexisting_alarm"] for e in events),
            "inflight_pre_onset_minimum_decisions": min((e["pre_onset_valid_decisions"] for e in inflight),
                                                        default=None),
            "inflight_pre_onset_requirement_failures": [e["event_id"] for e in inflight
                                                        if e["pre_onset_valid_decisions"] < 30],
            "ledger": ledger,
        }
    return {"models": models, "test_tuning": False,
            "missing_or_failed_attempts_are_not_healthy": True,
            "performance_pass_claim": False,
            "interpretation": "frozen-model independent pilot; report misses, no recall acceptance threshold invented"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = account(json.loads(args.lock.read_text()), json.loads(args.attempts.read_text())["attempts"],
                     json.loads(args.comparison.read_text()))
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
