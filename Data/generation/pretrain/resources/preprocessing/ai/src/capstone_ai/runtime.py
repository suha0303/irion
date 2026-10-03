"""Run a saved model over causal canonical frames; truth is not read here."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import feature_transform
from .inference import PendingForecasts, save_pending
from .mapping import TARGETS
from .source import dump, read_json, sha
from .training import Predictor


def replay_forecasts(canonical, bundle, out):
    canonical, bundle, out = Path(canonical), Path(bundle), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    predictor = Predictor(bundle)
    cal = read_json(bundle / "calibration.json")
    summary = read_json(canonical / "summary.json")
    if summary["config"] != read_json(bundle / "preprocessing.json"):
        raise ValueError("bundle preprocessing mismatch")
    windows = {}
    for line in (canonical / "windows.jsonl").read_text().splitlines():
        w = json.loads(line)
        windows[(w["clock_id"], w["segment_id"], w["origin_us"])] = w
    ledger = PendingForecasts(cal["error_scale"], cal["high"])
    written = set()
    last_time = None
    with (
        (out / "forecasts.jsonl").open("w") as forecasts,
        (out / "decisions.jsonl").open("w") as decisions,
    ):

        def emit():
            for k, d in ledger.decisions.items():
                if k not in written:
                    decisions.write(json.dumps(d, allow_nan=False) + "\n")
                    decisions.flush()
                    written.add(k)

        previous_segment = None
        for line in (canonical / "frames.jsonl").read_text().splitlines():
            frame = json.loads(line)
            t = frame["time_us"]
            last_time = t
            segment = (frame["clock_id"], frame["segment_id"])
            if previous_segment is not None and previous_segment != segment:
                ledger.advance(t, closed=True)
                emit()
                # Pending forecasts remain immutable, alarm state cannot span a new segment.
                ledger.alarm = type(ledger.alarm)(cal["high"])
            previous_segment = segment
            valid = all(frame["quality"][n] == "valid" for n in TARGETS)
            actual = [frame["values"][n] for n in TARGETS] if valid else None
            ledger.observe(frame["run_id"], frame["clock_id"], frame["segment_id"], t, t, actual)
            ledger.advance(t)
            emit()
            w = windows.get((*segment, t))
            if w is None:
                continue
            scale = predictor.scaler
            x = (feature_transform([w["X"]]) - scale["x_mean"]) / scale["x_scale"]
            c = (np.array([w["C"]]) - scale["c_mean"]) / scale["c_scale"]
            start = time.perf_counter()
            prediction = predictor.physical(predictor.predict_scaled(x, c))[0]
            elapsed = time.perf_counter() - start
            key = ledger.issue(
                frame["run_id"], frame["clock_id"], frame["segment_id"], t, t, prediction
            )
            forecasts.write(
                json.dumps(
                    {
                        **ledger.forecasts[key],
                        "execution": "offline_logical_replay",
                        "compute_seconds": elapsed,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            forecasts.flush()
        if last_time is not None:
            ledger.advance(last_time, closed=True)
            emit()
    save_pending(ledger, out / "pending.json")
    result = {
        "forecasts": len(ledger.forecasts),
        "decisions": len(ledger.decisions),
        "valid_scores": sum(d["scores"] is not None for d in ledger.decisions.values()),
        "missing_or_late": sum(d["scores"] is None for d in ledger.decisions.values()),
        "execution": "offline_logical_replay; live inference clock synchronization not verified",
        "forecast_hash": ledger.forecast_hash(),
        "files": {p.name: sha(p) for p in out.glob("*.jsonl")},
    }
    dump(out / "summary.json", result)
    return result
