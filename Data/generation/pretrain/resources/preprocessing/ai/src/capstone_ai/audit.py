import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
from pyulog import ULog

from .mapping import BINDINGS, validate_metadata
from .source import dump, read_json, sha

ROOT = Path(__file__).resolve().parents[3]


def audit(run, out):
    run = Path(run).resolve()
    metadata = read_json(run / "observations/metadata.json")
    config = read_json(ROOT / "ai/configs/preprocessing.json")
    validate_metadata(metadata, config)
    validator = jsonschema.Draft202012Validator(read_json(ROOT / "contracts/raw-event.schema.json"))
    topics: Counter[str] = Counter()
    quality: Counter[str] = Counter()
    previous: dict[tuple, dict[str, Any]] = {}
    seq_gaps: Counter[str] = Counter()
    sample_reversals: Counter[str] = Counter()
    cadence, delay = defaultdict(list), defaultdict(list)
    examples: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    with (run / "observations/streams.jsonl").open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            e = json.loads(line)
            for error in validator.iter_errors(e):
                if len(errors) < 20:
                    errors.append({"line": i + 1, "path": list(error.path), "error": error.message})
            topic = e["topic"]
            key = (e["clock_id"], e["source"], topic, e["instance"])
            topics[topic] += 1
            quality.update(e["field_quality"].values())
            delay[topic].append(e["available_time_us"] - e["sample_time_us"])
            if key in previous:
                p = previous[key]
                cadence[topic].append(e["sample_time_us"] - p["sample_time_us"])
                seq_gaps[topic] += max(0, e["seq"] - p["seq"] - 1)
                sample_reversals[topic] += e["sample_time_us"] < p["sample_time_us"]
            previous[key] = e
            examples.setdefault(topic, e)

    def stats(values):
        return {
            "min": int(min(values)),
            "median": float(np.median(values)),
            "max": int(max(values)),
        }

    run_meta = read_json(run / "run.json")
    native: dict[str, Any] = {}
    for name in run_meta["native_logs"]:
        path = run / name
        if not path.exists():
            native[name] = {"status": "missing"}
            continue
        ulog = ULog(str(path))
        native[name] = {
            "sha256": sha(path),
            "start_us": ulog.start_timestamp,
            "end_us": ulog.last_timestamp,
            "roles": {
                role: [
                    {
                        "topic": d.name,
                        "instance": d.multi_id,
                        "samples": len(d.data["timestamp"]),
                        "fields": list(d.data),
                    }
                    for d in ulog.data_list
                    if d.name == topic
                ]
                for role, (topic, _) in BINDINGS.items()
            },
        }
    diag = [
        json.loads(s)
        for s in (run / "observations/diagnostics.jsonl").read_text().splitlines()
        if s.strip()
    ]
    skipped: dict[str, int] = {}
    for row in diag:
        if "skipped_total" in row:
            skipped[row["topic"]] = max(skipped.get(row["topic"], 0), row["skipped_total"])
    result = {
        "passed": not errors,
        "run": str(run),
        "run_id": run_meta["run_id"],
        "origin": run_meta["origin"],
        "metadata": metadata,
        "topics": dict(topics),
        "quality": dict(quality),
        "schema_errors": errors,
        "sample_period_us": {k: stats(v) for k, v in cadence.items()},
        "availability_delay_us": {k: stats(v) for k, v in delay.items()},
        "seq_gaps": dict(seq_gaps),
        "sample_reversals": dict(sample_reversals),
        "examples": examples,
        "diagnostic_rows": len(diag),
        "native_skipped_max_not_sum": skipped,
        "native": native,
        "input_hashes": {
            str(p.relative_to(run)): sha(p)
            for base in ["observations", "reference", "native_logs"]
            for p in (run / base).rglob("*")
            if p.is_file()
        },
        "producer_lock_sha256": sha(run / "reference/source_lock.json"),
        "notes": [
            "ULog availability cannot recover capture receiver timestamps; native audit is separate from C9.",
            "Nominal energy denominator uses explicit configured 3.7 V/cell engineering reference; producer does not declare nominal cell voltage.",
        ],
    }
    dump(out, result)
    return {
        "passed": result["passed"],
        "report": str(out),
        "events": sum(topics.values()),
        "native_logs": len(native),
    }
