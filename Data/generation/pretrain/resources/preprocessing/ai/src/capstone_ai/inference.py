"""No evaluation imports. Forecasts are immutable and targets are matched by time."""

import copy
import hashlib
import json

import numpy as np


class Alarm:
    def __init__(self, high):
        self.high = np.array(high)
        self.ewma = None
        self.active = False
        self.enter = self.exit = 0
        self.last = None

    def update(self, scores, time_us):
        if scores is None:
            self.ewma = None
            self.enter = self.exit = 0
            self.last = None
            return {
                "state": "insufficient_data",
                "alarm_active": self.active,
                "episode_start": False,
            }
        if self.last is not None and time_us - self.last != 1000000:
            self.ewma = None
            self.enter = self.exit = 0
        self.last = time_us
        scores = np.array(scores)
        self.ewma = scores if self.ewma is None else 0.3 * scores + 0.7 * self.ewma
        self.enter = self.enter + 1 if np.any(self.ewma > self.high) else 0
        self.exit = self.exit + 1 if np.all(self.ewma < 0.7 * self.high) else 0
        start = not self.active and self.enter >= 3
        if start:
            self.active = True
        if self.active and self.exit >= 10:
            self.active = False
        return {
            "state": "anomalous" if self.active else "ready",
            "alarm_active": self.active,
            "episode_start": start,
            "ewma": self.ewma.tolist(),
        }


class PendingForecasts:
    def __init__(self, error_scale, threshold):
        self.error_scale = np.maximum(error_scale, 1e-8)
        self.alarm = Alarm(threshold)
        self.forecasts = {}
        self.actuals = {}
        self.decisions = {}

    def issue(self, run_id, clock, segment, origin, issued, prediction, timing_status=None):
        key = f"{run_id}:{clock}:{segment}:{origin}"
        forecast = {
            "schema": "drone.forecast/3.0",
            "forecast_id": key,
            "run_id": run_id,
            "clock_id": clock,
            "segment_id": segment,
            "origin_time_us": origin,
            "issued_time_us": issued,
            "target_times_us": [origin + (j + 1) * 100000 for j in range(10)],
            "prediction": np.asarray(prediction).tolist(),
            "status": timing_status
            or (
                "forecast_timing_unverified"
                if issued is None
                else "forecast_late"
                if issued >= origin + 100000
                else "awaiting_targets"
            ),
        }
        if key in self.forecasts:
            if self.forecasts[key] != forecast:
                raise ValueError("immutable forecast replacement rejected")
            return key
        self.forecasts[key] = forecast
        return key

    def observe(self, run_id, clock, segment, time_us, available_us, values):
        key = (run_id, clock, segment, time_us)
        actual = {"available_us": available_us, "values": values}
        if key in self.actuals and self.actuals[key] != actual:
            raise ValueError("conflicting target duplicate")
        self.actuals[key] = actual

    def advance(self, decision_time_us, closed=False):
        for key, f in self.forecasts.items():
            if key in self.decisions:
                continue
            times = f["target_times_us"]
            actual = [
                self.actuals.get((f["run_id"], f["clock_id"], f["segment_id"], t)) for t in times
            ]
            available = all(
                a is not None and a["available_us"] <= decision_time_us and t <= decision_time_us
                for a, t in zip(actual, times)
            )
            if not available and not closed:
                continue
            valid = (
                available
                and all(a is not None and a["values"] is not None for a in actual)
                and f["status"] == "awaiting_targets"
            )
            scores, residual = None, None
            if valid:
                residual = np.array([a["values"] for a in actual if a is not None]) - np.array(
                    f["prediction"]
                )
                normalized = residual**2 / self.error_scale
                scores = [float(normalized[:, :3].mean()), float(normalized[:, 3].mean())]
            # Arrival jitter must not break consecutive origin blocks in a live stream.
            status = self.alarm.update(scores, f["origin_time_us"] + 1000000)
            if f["status"] in {"forecast_late", "forecast_timing_unverified"}:
                status["state"] = f["status"]
            self.decisions[key] = {
                "schema": "drone.decision/3.0",
                "forecast_id": key,
                "origin_time_us": f["origin_time_us"],
                "decision_time_us": decision_time_us,
                "last_target_available_us": max(
                    (a["available_us"] for a in actual if a), default=None
                ),
                "coverage": sum(a is not None and a["values"] is not None for a in actual) / 10,
                "scores": scores,
                "signed_residual": residual.tolist() if residual is not None else None,
                **status,
            }

    def snapshot(self):
        return copy.deepcopy(self)

    def forecast_hash(self):
        return hashlib.sha256(
            json.dumps(self.forecasts, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


def save_pending(store, path):
    from .source import dump

    dump(
        path,
        {
            "error_scale": store.error_scale.tolist(),
            "threshold": store.alarm.high.tolist(),
            "forecasts": store.forecasts,
            "decisions": store.decisions,
            "actuals": [[list(k), v] for k, v in store.actuals.items()],
            "alarm": {
                "ewma": store.alarm.ewma.tolist() if store.alarm.ewma is not None else None,
                "active": store.alarm.active,
                "enter": store.alarm.enter,
                "exit": store.alarm.exit,
                "last": store.alarm.last,
            },
        },
    )


def load_pending(path):
    from .source import read_json

    state = read_json(path)
    store = PendingForecasts(state["error_scale"], state["threshold"])
    store.forecasts = state["forecasts"]
    store.decisions = state["decisions"]
    store.actuals = {tuple(k): v for k, v in state["actuals"]}
    for key, value in state["alarm"].items():
        setattr(store.alarm, key, np.array(value) if key == "ewma" and value is not None else value)
    return store
