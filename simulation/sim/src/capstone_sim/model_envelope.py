"""Preserve whole-run limits while qualifying only the recorded mission interval."""

import json
import math
from pathlib import Path
from typing import Any

POLICY = "airborne_through_landing_inclusive_v1"


def check(truth_path, timeline):
    def boundary(state):
        value = next((event.get("sim_time") for event in timeline if event["state"] == state), None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            return round(value * 1e6)
        return None

    start, end = boundary("airborne"), boundary("landing")
    complete = start is not None and end is not None and end > start
    segments: dict[str, dict[str, Any]] = {
        name: dict(samples=0, outside_samples=0, undefined_flow_samples=0,
                   first_outside_time_us=None, max_inplane_advance_ratio=None)
        for name in ("before_mission", "mission", "after_mission", "unscoped")}
    with Path(truth_path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Incomplete model-envelope truth record")
            entry = json.loads(line)
            stamp = entry["time_us"]
            if type(stamp) is not int or stamp < 0:
                raise ValueError("Invalid model-envelope truth timestamp")
            name = ("unscoped" if start is None or end is None or end <= start
                    else "before_mission" if stamp < start
                    else "after_mission" if stamp > end else "mission")
            segment = segments[name]
            segment["samples"] += 1
            invalid = entry.get("model_valid") is not True
            segment["outside_samples"] += invalid
            if invalid and segment["first_outside_time_us"] is None:
                segment["first_outside_time_us"] = stamp
            segment["undefined_flow_samples"] += entry.get("inplane_advance_ratio_defined") is False
            advance = entry.get("max_inplane_advance_ratio")
            if type(advance) in (int, float) and math.isfinite(advance):
                previous = segment["max_inplane_advance_ratio"]
                segment["max_inplane_advance_ratio"] = advance if previous is None else max(previous, advance)
    mission = segments["mission"]
    return dict(policy=POLICY, mission_start_us=start, mission_end_us=end,
                mission_interval_complete=complete, segments=segments,
                samples=sum(s["samples"] for s in segments.values()),
                outside_samples=sum(s["outside_samples"] for s in segments.values()),
                normal_eligibility_passed=bool(complete and mission["samples"]
                                             and not mission["outside_samples"]))
