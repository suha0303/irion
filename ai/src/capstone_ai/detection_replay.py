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


def prediction_stream(ordered_windows, predictor, config, batch_size, timings, check=None):
    """Batch independent past-only inputs; the replay still issues in logical time.

    Batching does not measure online single-request latency. A chunk may contain
    later origins, but its predictions cannot enter the ledger before those origins.
    No target frames, detector state, or evaluator truth enter this calculation.
    """
    scale = predictor.scaler
    for offset in range(0, len(ordered_windows), batch_size):
        if check is not None:
            check()
        chunk = ordered_windows[offset : offset + batch_size]
        x = (feature_transform([w["X"] for w in chunk], config) - scale["x_mean"]) / scale["x_scale"]
        c = (np.asarray([w["C"] for w in chunk]) - scale["c_mean"]) / scale["c_scale"]
        started = time.perf_counter()
        predictions = predictor.physical(predictor.predict_scaled(x, c))
        elapsed = time.perf_counter() - started
        if np.shape(predictions) != (len(chunk), 10, 4):
            raise ValueError("malformed batched forecast")
        timings.append(elapsed)
        for prediction in predictions:
            yield prediction, elapsed / len(chunk), len(chunk)


def prepare_replay(canonical, bundle, *, frames=None):
    """Read one immutable run; inference and CPU replay can consume it separately."""
    canonical, bundle = Path(canonical), Path(bundle)
    config = read_json(canonical / "summary.json")["config"]
    if config != read_json(bundle / "preprocessing.json"):
        raise ValueError("preprocessing contract mismatch")
    windows = {(w["clock_id"], w["segment_id"], w["origin_us"]): w
               for w in records(canonical / "windows.jsonl")}
    frames = records(canonical / "frames.jsonl") if frames is None else frames
    ordered = [windows[key] for f in frames
               if (key := (f["clock_id"], f["segment_id"], f["time_us"])) in windows]
    return dict(config=config, windows=windows, frames=frames, ordered_windows=ordered,
                summary=read_json(canonical / "summary.json"))


def replay(canonical, bundle, calibration, out, *, predictor=None, prediction_batch_size=1):
    if type(prediction_batch_size) is not int or not 1 <= prediction_batch_size <= 4096:
        raise ValueError("prediction_batch_size must be an integer within1..4096")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    predictor = Predictor(bundle) if predictor is None else predictor
    if Path(predictor.folder).resolve() != Path(bundle).resolve():
        raise ValueError("Replay predictor bundle mismatch")
    prepared = prepare_replay(canonical, bundle)
    batch_seconds = []
    predictions = prediction_stream(prepared["ordered_windows"], predictor, prepared["config"],
                                    prediction_batch_size, batch_seconds)
    return replay_prepared(prepared, bundle, calibration, out, predictions, batch_seconds,
                           prediction_batch_size)


def replay_prepared(prepared, bundle, calibration, out, predictions, batch_seconds,
                    prediction_batch_size):
    """CPU-only causal ledger/writer; consumes past-only forecasts in frame order."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    windows, frames = prepared["windows"], prepared["frames"]
    ledger = DetectionLedger(calibration["error_scale_physical"], calibration["high"])
    previous, last_time = None, None
    latencies = []
    predictions = iter(predictions)
    with (out / "forecasts.jsonl").open("w") as fw, (out / "decisions.jsonl").open("w") as dw:
        def emit(completed):
            for decision in completed:
                dw.write(json.dumps(decision, allow_nan=False) + "\n")

        for frame in frames:
            t = frame["time_us"]
            scope = (frame["clock_id"], frame["segment_id"])
            if previous is not None and previous != scope:
                emit(ledger.advance(last_time, closed=True))
            previous, last_time = scope, t
            target = [frame["values"][n] for n in TARGETS] \
                if all(frame["quality"][n] == "valid" for n in TARGETS) else None
            ledger.observe(frame["run_id"], *scope, t, t, target)
            emit(ledger.advance(t))
            w = windows.get((*scope, t))
            if w is None:
                continue
            prediction, elapsed, batch_count = next(predictions)
            latencies.append(elapsed)
            key = ledger.issue(frame["run_id"], *scope, t, t, prediction)
            fw.write(json.dumps({**ledger.forecasts[key], "execution": "offline_logical_replay",
                                 "compute_seconds": elapsed,
                                 "compute_seconds_scope": "single_forecast_forward" if prediction_batch_size == 1
                                 else "batch_forward_amortized_per_forecast",
                                 "prediction_batch_size": batch_count}, allow_nan=False) + "\n")
        if last_time is not None:
            emit(ledger.advance(last_time, closed=True))
        # An interrupted run remains unsealed. Only complete files are made
        # durable and subsequently admitted as reusable per-run evidence.
        for stream in (fw, dw):
            stream.flush()
            os.fsync(stream.fileno())
    decisions = list(ledger.decisions.values())
    spans: dict[str, list[int]] = {}
    for f in frames:
        if f["operating_state"] == "airborne":
            spans.setdefault(f["clock_id"], []).append(f["time_us"])
    operational = sum(max(ts) - min(ts) + 100000 for ts in spans.values()) / 1e6
    intervals = {clock: [min(ts), max(ts) + 100000] for clock, ts in spans.items()}
    metrics = evaluation_metrics([], decisions, operational, intervals)
    summary = prepared["summary"]
    metrics.update({"forecast_count": len(ledger.forecasts), "decisions": len(decisions),
                    "operational_spans_us": intervals,
                    "exposure_definition": "first-to-last observed airborne envelope per clock, includes intervening gaps",
                    "preprocessing_exclusions": summary["exclusions"],
                    "warmup_frames": sum(f["warmup"] for f in frames),
                    "logical_replay_only": True, "live_deadline_verification": "not_executed",
                    "prediction_batch_size_requested": prediction_batch_size,
                    "prediction_batches": len(batch_seconds),
                    "durability": "both JSONL files synced before run seal; interrupted run is not reusable",
                    "compute_seconds_p50_p95_p99": np.quantile(latencies, [.5, .95, .99]).tolist()
                    if latencies and prediction_batch_size == 1 else None,
                    "amortized_compute_seconds_p50_p95_p99": np.quantile(latencies, [.5, .95, .99]).tolist()
                    if latencies else None,
                    "batch_compute_seconds_p50_p95_p99": np.quantile(batch_seconds, [.5, .95, .99]).tolist()
                    if batch_seconds else None})
    dump(out / "summary.json", metrics)
    with (out / "summary.json").open("rb") as stream:
        os.fsync(stream.fileno())
    dump(out / "seal.json", {"files": {p.name: sha(p) for p in
                                       [out / "forecasts.jsonl", out / "decisions.jsonl", out / "summary.json"]},
                             "canonical": summary["output_hashes"],
                             "bundle_integrity_sha256": sha(Path(bundle) / "integrity.json"),
                             "execution": "sealed_before_evaluation_truth_join"})
    with (out / "seal.json").open("rb") as stream:
        os.fsync(stream.fileno())
    directory = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
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
