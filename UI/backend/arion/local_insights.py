"""Expose actual local results without impersonating an external provider."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=128)
def _json(path, modified_ns, size):
    del modified_ns, size  # The immutable cache key detects replacements.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read(path):
    stat = path.stat()
    return _json(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _alarms(path, modified_ns, size):
    del modified_ns, size
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            # A live writer may not have finished its last line yet.
            if not line.endswith("\n"):
                break
            rows.append(json.loads(line))
    return dict(count=len(rows), rows=rows[-200:])


def insights(root, run_id, state):
    """Bind saved analyses to this exact recording and pinned model variants."""
    folder = root / "live" / run_id
    history = folder / "alarms.jsonl"
    output = dict(history_available=history.exists(), alarm_count=0, alarms=[])
    if history.exists():
        stat = history.stat()
        alarms = _alarms(str(history), stat.st_mtime_ns, stat.st_size)
        output.update(alarm_count=alarms["count"], alarms=alarms["rows"])
    result = state.get("result")
    if not result:
        return output
    analyses = [row["analysis"] for row in result.get("analyses", [])]
    analyses.extend(
        row["analysis"] for row in result.get("forecast_analyses", [])
    )
    output.update(
        report=dict(analyses=analyses, run_summary=result.get("run_summary")),
        report_source="flight_completion",
    )
    # Older recordings only contain a post-flight model. Reuse a completed
    # analysis of this exact file when both pinned variants match; no inference
    # or historical live decisions are fabricated by this read endpoint.
    if result.get("forecast_analyses"):
        return output
    models = {
        state.get("forecast_model"): state.get("adapted_sha256"),
        state.get("post_model"): state.get("post_adapted_sha256"),
    }
    models.pop(None, None)
    baseline = analyses[0] if analyses else {}
    input_keys = (
        "origin",
        "preprocessing",
        "live_input",
        "policy",
        "clock",
        "fast_max_age_ms",
        "battery_max_age_ms",
    )
    baseline_input = baseline.get("calibration_binding", {})
    jobs = sorted(
        (root / "jobs").glob("*.json"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    for path in jobs:
        job = read(path)
        if job.get("status") != "complete" or job.get("action") != "analyze":
            continue
        candidates = (job.get("result") or {}).get("analyses", [])
        if not candidates:
            continue
        if {a.get("model_id") for a in candidates} != set(models):
            continue
        if any(
            a.get("source_sha256") != result.get("source_sha256")
            or a.get("adapted_sha256") != models[a["model_id"]]
            or a.get("profile_version") != baseline.get("profile_version")
            or a.get("calibration_binding", {}).get("clock")
            != "receipt_monotonic"
            or any(
                a.get("calibration_binding", {}).get(key)
                != baseline_input.get(key)
                for key in input_keys
            )
            for a in candidates
        ):
            continue
        output.update(
            report=job["result"],
            report_source="saved_log_analysis",
            report_job=job["id"],
        )
        break
    return output
