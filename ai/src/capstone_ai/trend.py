"""Compare explicitly continuous, unchanged assets; never infer SOH or RUL."""

from .source import dump, read_json


def comparison_gate(previous, current):
    reasons = []
    if current.get("replacement_event") or any(
        previous.get(k) != current.get(k)
        for k in ["vehicle_instance_id", "battery_instance_id", "propulsion_assembly_ids"]
    ):
        reasons.append("asset_replacement_or_identity_change")
    continuity = [previous.get("continuity"), current.get("continuity")]
    if "independent_reset" in continuity:
        reasons.append("independent_reset")
    elif any(x != "same_asset_future" for x in continuity):
        reasons.append("insufficient_continuity")
    a, b = previous.get("comparable_operation", {}), current.get("comparable_operation", {})
    required = {"mission", "ambient_temperature_k"}
    if (
        not required.issubset(a)
        or not required.issubset(b)
        or any(v is None for v in [*a.values(), *b.values()])
    ):
        reasons.append("insufficient_comparable_conditions")
    elif a != b:
        reasons.append("operation_conditions_changed")
    if not previous.get("bundle_sha256") or not current.get("bundle_sha256"):
        reasons.append("bundle_revision_unavailable")
    elif previous["bundle_sha256"] != current["bundle_sha256"]:
        reasons.append("bundle_revision_changed")
    end, start = previous.get("end_time_utc"), current.get("start_time_utc")
    if end is None or start is None:
        reasons.append("chronology_unavailable")
    else:
        from datetime import datetime

        if datetime.fromisoformat(end) >= datetime.fromisoformat(start):
            reasons.append("nonchronological_runs")
    return {
        "previous_run": previous["run_id"],
        "current_run": current["run_id"],
        "comparable": not reasons,
        "reasons": reasons,
    }


def analyze_registry(path, out):
    registry = read_json(path)
    records = registry["records"]
    comparisons = [comparison_gate(a, b) for a, b in zip(records, records[1:])]
    result = {
        "origin": registry["origin"],
        "consumer": "capstone_ai.trend",
        "comparisons": comparisons,
        "claim": "comparability only; no continuous aging, health certification, SOH or RUL inferred",
        "aggregate_trend": None
        if not comparisons or not all(x["comparable"] for x in comparisons)
        else "eligible_for_scored_history",
    }
    dump(out, result)
    return result
