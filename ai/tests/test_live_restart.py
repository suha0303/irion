"""Synthetic adapter exercise, explicitly not evidence of a real live producer."""

import copy
import json

import numpy as np
import pytest
from test_preprocessing import CONFIG, META, events, run

from capstone_ai import live_inference as live
from capstone_ai.source import dump, read_json


@pytest.mark.parametrize("stop_counts", [[5], [2, 2, 1]])
def test_resume_does_not_close_pending_at_replayed_segment_boundary(
    tmp_path, monkeypatch, stop_counts
):
    source = tmp_path / "run"
    (source / "observations").mkdir(parents=True)
    dump(source / "run.json", {"status": "mission"})
    dump(source / "observations/metadata.json", META)
    (source / "observations/streams.jsonl").touch()
    bundle = tmp_path / "bundle"
    dump(bundle / "preprocessing.json", CONFIG)
    dump(bundle / "calibration.json", {"error_scale": np.ones((10, 4)).tolist(), "high": [1, 1]})
    dump(bundle / "integrity.json", {})
    # A real preprocessor fixture with a historical gap before context warmup.
    engine = run([e for e in events(71) if not 0 < e["available_time_us"] < 2000000])
    assert len({f["segment_id"] for f in engine.frames}) > 1
    clock = {"time": 0, "nonce": 0}

    class Tail:
        def __init__(self, path, consumer):
            self.engine, self.index, self.closed, self.last_mark = consumer, 0, False, None

        def poll(self):
            frame = copy.deepcopy(engine.frames[self.index])
            self.engine.frames.append(frame)
            self.engine.windows.extend(
                copy.deepcopy(w) for w in engine.windows if w["origin_us"] == frame["time_us"]
            )
            clock["time"] = frame["time_us"]
            self.index += 1
            self.closed = self.index == len(engine.frames)
            return True

        def checkpoint(self):
            return {"fixture": True}

        def verify_checkpoint(self, checkpoint):
            assert checkpoint == {"fixture": True}

    class Probe:
        def __init__(self, *args):
            self.count, self.errors = 0, 0

        def query(self):
            clock["nonce"] += 1
            self.count += 1
            return {
                "nonce": clock["nonce"],
                "sim_time_us": clock["time"] + 1000,
                "host_send_ns": 10**30,
            }

        def close(self):
            pass

    class Predictor:
        def __init__(self, *args):
            self.scaler = {"x_mean": 0, "x_scale": 1, "c_mean": 0, "c_scale": 1}

        def predict_scaled(self, x, c):
            return np.zeros((1, 10, 4))

        def physical(self, prediction):
            return prediction

    monkeypatch.setattr(live, "DirectoryTail", Tail)
    monkeypatch.setattr(live, "ClockProbe", Probe)
    monkeypatch.setattr(live, "Predictor", Predictor)
    monkeypatch.setattr(live, "output", lambda *args: None)
    monkeypatch.setattr(
        live,
        "deadline_bounds",
        lambda before, after, done, deadline: {
            "status": "on_time",
            "upper_us": deadline - 1,
        },
    )
    out = tmp_path / "resumed"
    for index, count in enumerate(stop_counts):
        first = live.live_forecast(source, bundle, out, 0, stop_after=count, resume=index > 0)
        assert first["paused_for_restart"]
    assert first["forecasts"] == 5
    pending = read_json(out / "pending.json")
    pending_id = next(k for k in pending["forecasts"] if k not in pending["decisions"])
    prior_alarm = pending["alarm"]
    assert prior_alarm["active"]  # Recovery must preserve an already active alarm too.
    result = live.live_forecast(source, bundle, out, 0, resume=True)
    assert result["closed"]
    resumed = [json.loads(s) for s in (out / "decisions.jsonl").read_text().splitlines()]
    recovered = next(d for d in resumed if d["forecast_id"] == pending_id)
    assert recovered["coverage"] == 1 and recovered["scores"] is not None
    assert recovered["alarm_active"] and not recovered["episode_start"]
    full = tmp_path / "full"
    live.live_forecast(source, bundle, full, 0)
    uninterrupted = [json.loads(s) for s in (full / "decisions.jsonl").read_text().splitlines()]
    fields = ["forecast_id", "coverage", "scores", "state", "alarm_active", "episode_start", "ewma"]
    assert [{k: d.get(k) for k in fields} for d in resumed] == [
        {k: d.get(k) for k in fields} for d in uninterrupted
    ]
