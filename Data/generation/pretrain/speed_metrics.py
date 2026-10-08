"""Fixed mission-relative intervals for speed information comparison, protocol v2."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .common import ROOT, read, runs_root

PROTOCOL = "information-v2"
INTERVALS = dict(anchor="mission", telemetry_seconds=[20, 110], window_origin_seconds=[60, 110])


def records(path, expected):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for body in stream:
            digest.update(body)
            yield json.loads(body)
    if digest.hexdigest() != expected:
        raise ValueError("Changed speed evidence: " + str(path))


def topic_metrics(path, expected, run_id, topics, start, end):
    stats = {
        name: dict(
            count=0,
            first=None,
            last=None,
            max_gap_us=0,
            invalid=0,
            duration_us=end - start,
            covered=False,
            max_availability_delay_us=0,
        )
        for name in topics
    }
    previous, before, after = {}, {}, {}
    for event in records(path, expected):
        if event["run_id"] != run_id or event["clock_id"] != "px4-boot":
            raise ValueError("Raw comparison identity/clock mismatch")
        name, t = event["topic"], event["sample_time_us"]
        if name not in stats:
            raise ValueError("Unexpected raw topic")
        s = stats[name]
        prior = previous.get(name)
        if prior is not None and t < prior:
            raise ValueError("Raw sample clock reversed")
        if prior is not None and t >= start and prior < end:
            s["max_gap_us"] = max(s["max_gap_us"], t - prior)
        if t <= start:
            before[name] = t
        if t >= end and name not in after:
            after[name] = t
        if start <= t < end:
            s["count"] += 1
            s["first"] = t if s["first"] is None else s["first"]
            s["last"] = t
            s["invalid"] += sum(v != "valid" for v in event["field_quality"].values())
            s["max_availability_delay_us"] = max(
                s["max_availability_delay_us"], event["available_time_us"] - t
            )
        previous[name] = t
    for name, s in stats.items():
        s["covered"] = name in before and name in after and s["count"] > 0
    return stats


def metrics(canonical, record):
    from .common import package_imports

    package_imports()
    from capstone_ai.mapping import TARGETS

    canonical = Path(canonical)
    rid = record["run_id"]
    raw = runs_root(ROOT) / rid
    inventory = record["export"]["members"]
    timeline = raw / "evaluation/execution.json"
    if (
        hashlib.sha256(timeline.read_bytes()).hexdigest()
        != inventory["evaluation/execution.json"]["sha256"]
    ):
        raise ValueError("Changed flight timeline")
    states = read(timeline)
    missions = [p for p in states if p["state"] == "mission"]
    landings = [p for p in states if p["state"] == "landing"]
    if len(missions) != 1 or len(landings) != 1:
        raise ValueError("Unique mission and landing anchors required")
    anchor = round(missions[0]["sim_time"] * 1e6)
    landing = round(landings[0]["sim_time"] * 1e6)
    start, end = anchor + 20_000_000, anchor + 110_000_000
    if landing <= end + 1_000_000:
        raise ValueError("Insufficient steady mission including forecast horizon")
    summary = read(canonical / "summary.json")
    if summary["run_id"] != rid or not summary["closed"]:
        raise ValueError("Closed matching canonical run required")
    if (
        summary["input_hashes"]["observations/streams.jsonl"]
        != inventory["observations/streams.jsonl"]["sha256"]
    ):
        raise ValueError("Canonical/raw input binding mismatch")
    frames = list(records(canonical / "frames.jsonl", summary["output_hashes"]["frames"]))
    names = summary["feature_order"]
    if any(f["run_id"] != rid or f["clock_id"] != "px4-boot" for f in frames):
        raise ValueError("Canonical identity/clock mismatch")
    if any(b["time_us"] - a["time_us"] != 100_000 for a, b in zip(frames, frames[1:])):
        raise ValueError("Canonical10Hz information grid changed")
    valid = [
        f
        for f in frames
        if f["operating_state"] == "airborne" and all(f["quality"][n] == "valid" for n in names)
    ]
    selected = [f for f in frames if start <= f["time_us"] < end]
    expected_frames = (end - start) // 100_000
    if len(selected) != expected_frames or any(
        f["operating_state"] != "airborne" for f in selected
    ):
        raise ValueError("Incomplete/non-airborne fixed comparison interval")
    time = np.asarray([(f["time_us"] - anchor) / 1e6 for f in valid])
    grid = np.arange(20.0, 110.01, 0.1)
    if len(time) < 2 or time[0] > grid[0] or time[-1] < grid[-1]:
        raise ValueError("Insufficient matched numerical comparison interval")
    values = np.asarray([[f["values"][n] for n in names] for f in valid])
    aligned = np.stack([np.interp(grid, time, values[:, i]) for i in range(len(names))], axis=1)
    lookup = {(f["clock_id"], f["segment_id"], f["time_us"]): f for f in frames}
    windows, eligible = 0, 0
    for w in records(canonical / "windows.jsonl", summary["output_hashes"]["windows"]):
        if not anchor + 60_000_000 <= w["origin_us"] < end:
            continue
        eligible += 1
        if w["context_start_us"] < anchor or min(w["history_times_us"]) < anchor:
            raise ValueError("Window context/history crosses mission boundary")
        targets = [lookup.get((w["clock_id"], w["segment_id"], t)) for t in w["target_times_us"]]
        if targets and all(
            f is not None and all(f["quality"][n] == "valid" for n in TARGETS) for f in targets
        ):
            windows += 1
    topics = topic_metrics(
        raw / "observations/streams.jsonl",
        inventory["observations/streams.jsonl"]["sha256"],
        rid,
        record["assessment"]["topic_quality"],
        start,
        end,
    )
    return dict(
        protocol=PROTOCOL,
        comparison_intervals=INTERVALS,
        anchor_us=anchor,
        absolute_interval_us=[start, end],
        features=names,
        config=summary["config"],
        aligned=aligned.tolist(),
        valid_airborne_fraction=sum(
            all(f["quality"][n] == "valid" for n in names) for f in selected
        )
        / expected_frames,
        windows=windows,
        eligible_windows=eligible,
        expected_window_origins=50,
        topic_quality=topics,
        effective_rtf=record["assessment"].get("effective_rtf"),
        raw_run=rid,
        canonical=str(canonical),
        whole_run_diagnostic=dict(windows=summary["windows"], frames=summary["frames"]),
    )
