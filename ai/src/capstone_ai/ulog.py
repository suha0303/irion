"""PX4 v1.16 ULog fields mapped to the same observation roles, without truth."""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pyulog import ULog

from .mapping import BINDINGS, mapped
from .source import dump, read_json, sha

SAMPLE_TOPICS = {
    "vehicle_angular_velocity",
    "vehicle_attitude",
    "vehicle_acceleration",
    "vehicle_local_position",
}
ARRAYS = {
    "actuator_outputs": ("output", 4),
    "vehicle_angular_velocity": ("xyz", 3),
    "vehicle_acceleration": ("xyz", 3),
    "vehicle_attitude": ("q", 4),
}


def decode(path, metadata):
    ulog = ULog(str(path))
    events = []
    fields_by_topic = {
        topic: metadata["semantic_bindings"][role]["fields"]
        for role, (topic, _) in BINDINGS.items()
    }
    for data in ulog.data_list:
        topic = data.name
        if topic not in fields_by_topic:
            continue
        timestamp_field = "timestamp_sample" if topic in SAMPLE_TOPICS else "timestamp"
        for i, sample in enumerate(data.data[timestamp_field]):
            fields: dict[str, Any] = {}
            quality = {}
            value: Any
            for field in fields_by_topic[topic]:
                if topic in ARRAYS and field == ARRAYS[topic][0]:
                    value = [float(data.data[f"{field}[{j}]"][i]) for j in range(ARRAYS[topic][1])]
                    good = all(math.isfinite(v) for v in value)
                    value = [v if math.isfinite(v) else None for v in value]
                else:
                    value = float(data.data[field][i])
                    good = math.isfinite(value)
                    value = value if good else None
                fields[field] = value
                quality[field] = "valid" if good else "unavailable"
            events.append(
                {
                    "schema": "drone.raw.event/2.0",
                    "source": "px4.uorb",
                    "topic": topic,
                    "instance": int(data.multi_id),
                    "seq": i,
                    "sample_time_us": int(sample),
                    "available_time_us": int(sample),
                    "clock_id": metadata["clock"]["id"],
                    "run_id": metadata["run_id"],
                    "fields": fields,
                    "field_quality": quality,
                }
            )
    events.sort(
        key=lambda e: (e["available_time_us"], e["source"], e["topic"], e["instance"], e["seq"])
    )
    return events


def parity(run, out):
    run = Path(run)
    metadata = read_json(run / "observations/metadata.json")
    info = read_json(run / "run.json")
    path = run / info["native_logs"][0]
    native = decode(path, metadata)
    index = {(e["topic"], e["instance"], e["sample_time_us"]): e for e in native}
    matched: Counter[str] = Counter()
    unmatched: Counter[str] = Counter()
    max_error = 0.0
    errors = []
    # Raw JSON uses nine significant digits. Tolerance fixed before parity execution.
    atol, rtol = 1e-6, 1e-6
    for line in (run / "observations/streams.jsonl").read_text().splitlines():
        raw = json.loads(line)
        key = (raw["topic"], raw["instance"], raw["sample_time_us"])
        event = index.get(key)
        if event is None:
            unmatched[raw["topic"]] += 1
            continue
        a, b = mapped(raw, metadata), mapped(event, metadata)
        matched[raw["topic"]] += 1
        for name, (value, quality) in a.items():
            other, oq = b[name]
            if quality != oq:
                errors.append(
                    {
                        "topic": raw["topic"],
                        "field": name,
                        "sample_time_us": raw["sample_time_us"],
                        "reason": "quality",
                    }
                )
            elif value is not None and other is not None:
                max_error = max(max_error, abs(value - other))
                if not np.isclose(value, other, atol=atol, rtol=rtol):
                    errors.append(
                        {
                            "topic": raw["topic"],
                            "field": name,
                            "sample_time_us": raw["sample_time_us"],
                            "raw": value,
                            "native": other,
                        }
                    )
    required = {topic for topic, _ in BINDINGS.values()}
    missing = sorted(required - set(matched))
    result = {
        "passed": not errors and not missing,
        "run_id": info["run_id"],
        "native_sha256": sha(path),
        "native_events": len(native),
        "matched": dict(matched),
        "unmatched_native_cadence_or_log_start": dict(unmatched),
        "missing_roles": missing,
        "max_absolute_error": max_error,
        "atol": atol,
        "rtol": rtol,
        "errors": errors[:20],
        "availability_policy": "ULog decode uses source_replay sample time; never claims original capture arrival",
        "scope": "same-timestamp ULog/raw mapped values only; ULog and captured stream have different sampling cadence",
    }
    dump(out, result)
    return result
