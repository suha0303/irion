"""Causal canonical replay: seal predictions before the separate truth evaluator."""

import json
import os
import time
from pathlib import Path

import numpy as np

from .dataset import feature_transform
from .detection import DetectionLedger
from .detection_protocol import evaluation_metrics, point_metrics
from .mapping import TARGETS
from .source import dump, read_json, sha
from .training import Predictor


def records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def replay(canonical, bundle, calibration, out, *, predictor=None):
    canonical, out = Path(canonical), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    predictor = Predictor(bundle) if predictor is None else predictor
    if Path(predictor.folder).resolve() != Path(bundle).resolve():
        raise ValueError("Replay predictor bundle mismatch")
    config = read_json(canonical / "summary.json")["config"]
    if config != read_json(Path(bundle) / "preprocessing.json"):
        raise ValueError("preprocessing contract mismatch")
    windows = {(w["clock_id"], w["segment_id"], w["origin_us"]): w
               for w in records(canonical / "windows.jsonl")}
    frames = records(canonical / "frames.jsonl")
    ledger = DetectionLedger(calibration["error_scale_physical"], calibration["high"])
    written, previous, last_time = set(), None, None
    latencies = []
    with (out / "forecasts.jsonl").open("w") as fw, (out / "decisions.jsonl").open("w") as dw:
        def emit():
            for key, decision in ledger.decisions.items():
                if key not in written:
                    dw.write(json.dumps(decision, allow_nan=False) + "\n")
                    written.add(key)

        for frame in frames:
            t = frame["time_us"]
            scope = (frame["clock_id"], frame["segment_id"])
            if previous is not None and previous != scope:
                ledger.advance(last_time, closed=True)
                emit()
            previous, last_time = scope, t
            target = [frame["values"][n] for n in TARGETS] \
                if all(frame["quality"][n] == "valid" for n in TARGETS) else None
            ledger.observe(frame["run_id"], *scope, t, t, target)
            ledger.advance(t)
            emit()
            w = windows.get((*scope, t))
            if w is None:
                continue
            scale = predictor.scaler
            x = (feature_transform([w["X"]], config) - scale["x_mean"]) / scale["x_scale"]
            c = (np.asarray([w["C"]]) - scale["c_mean"]) / scale["c_scale"]
            start = time.perf_counter()
            prediction = predictor.physical(predictor.predict_scaled(x, c))[0]
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            key = ledger.issue(frame["run_id"], *scope, t, t, prediction)
            fw.write(json.dumps({**ledger.forecasts[key], "execution": "offline_logical_replay",
                                 "compute_seconds": elapsed}, allow_nan=False) + "\n")
            fw.flush()
            os.fsync(fw.fileno())
        if last_time is not None:
            ledger.advance(last_time, closed=True)
        emit()
    decisions = list(ledger.decisions.values())
    spans: dict[str, list[int]] = {}
    for f in frames:
        if f["operating_state"] == "airborne":
            spans.setdefault(f["clock_id"], []).append(f["time_us"])
    operational = sum(max(ts) - min(ts) + 100000 for ts in spans.values()) / 1e6
    intervals = {clock: [min(ts), max(ts) + 100000] for clock, ts in spans.items()}
    metrics = evaluation_metrics([], decisions, operational, intervals)
    summary = read_json(canonical / "summary.json")
    metrics.update({"forecast_count": len(ledger.forecasts), "decisions": len(decisions),
                    "operational_spans_us": intervals,
                    "exposure_definition": "first-to-last observed airborne envelope per clock, includes intervening gaps",
                    "preprocessing_exclusions": summary["exclusions"],
                    "warmup_frames": sum(f["warmup"] for f in frames),
                    "logical_replay_only": True, "live_deadline_verification": "not_executed",
                    "compute_seconds_p50_p95_p99": np.quantile(latencies, [.5, .95, .99]).tolist()
                    if latencies else None})
    dump(out / "summary.json", metrics)
    dump(out / "seal.json", {"files": {p.name: sha(p) for p in out.glob("*.jsonl")},
                             "canonical": summary["output_hashes"],
                             "bundle_integrity_sha256": sha(Path(bundle) / "integrity.json"),
                             "execution": "sealed_before_evaluation_truth_join"})
    return metrics


def lineage_keys(row):
    keys = {"run:" + row["run_id"]}
    keys.update("run:" + value for value in row.get("previous_attempts", []))
    for field in ["parent_attempt_id", "paired_case_id", "raw_sha256", "raw_flight_id", "continuation_id"]:
        if row.get(field):
            keys.add(("run:" if field == "parent_attempt_id" else field + ":") + row[field])
    return keys


def require_new_holdout(rows, prior_rows):
    prior = set().union(*(lineage_keys(r) for r in prior_rows))
    seen = set()
    if not rows:
        raise ValueError("no independent evaluation rows")
    for row in rows:
        if row["run_id"] in seen or lineage_keys(row) & prior:
            raise ValueError("previously exposed raw/paired/retry run cannot be new holdout")
        seen.add(row["run_id"])


def evaluate_sealed(replay_folder, events_path, observation_start):
    from .detection_protocol import event_result

    folder = Path(replay_folder)
    seal = read_json(folder / "seal.json")
    for name, value in seal["files"].items():
        if sha(folder / name) != value:
            raise ValueError("sealed predictions/decisions changed before truth join")
    decisions = records(folder / "decisions.jsonl")
    truth = records(events_path)
    events = [event_result(e, decisions, observation_start) for e in truth]
    summary = read_json(folder / "summary.json")
    return {"events": events, "metrics": evaluation_metrics(events, decisions, summary["operational_seconds"],
                                                             summary.get("operational_spans_us")),
            "point_metrics": point_metrics(truth, decisions, observation_start),
            "forecast_seal": sha(folder / "seal.json"), "truth_sha256": sha(events_path),
            "matching_rule": "per-event awareness and new-episode recall; simultaneous events may share one alarm; no cause attribution"}
