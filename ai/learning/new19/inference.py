"""Independent new19 bundle inference and immutable 10 Hz forecast matching."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .data import (
    STEP_US,
    TARGET_INDICES,
    digest,
    forecast_origins,
    physical_features,
    read,
    whole_admitted,
)
from .models import make_model


class Runtime:
    def __init__(self, folder, *, device="cpu"):
        folder = Path(folder)
        bundle = read(folder / "bundle.json")
        weights = folder / "model.safetensors"
        if bundle.get("schema") != "drone-ai.new19.bundle/1" or digest(
            weights
        ) != bundle.get("model_sha256"):
            raise ValueError("New19 bundle revision or model hash mismatch")
        self.device = torch.device(device)
        self.model = make_model(bundle["variant"]).to(self.device)
        self.model.load_state_dict(load_file(str(weights), device="cpu"))
        self.model.eval()
        self.scaler = bundle["scaler"]
        for stem, width in (("", 19), ("delta_", 4)):
            if stem and self.model.variant.task != "inflight":
                break
            for key in ("mean", "scale"):
                values = np.asarray(self.scaler[stem + key], dtype=float)
                if (
                    values.shape != (width,)
                    or not np.isfinite(values).all()
                    or (key == "scale" and np.any(values <= 0))
                ):
                    raise ValueError("Invalid bundle scaler")

    def normalized(self, physical):
        return torch.from_numpy(
            ((physical - self.scaler["mean"]) / self.scaler["scale"]).astype(
                "f4"
            )
        ).to(self.device)

    @torch.inference_mode()
    def forecast(self, values):
        if self.model.variant.task != "inflight":
            raise ValueError("Expected inflight bundle")
        physical = physical_features(values)
        delta = self.model(self.normalized(physical)).cpu().numpy()
        delta = delta * self.scaler["delta_scale"] + self.scaler["delta_mean"]
        absolute = physical[:, -1:, TARGET_INDICES] + delta
        absolute[..., 3] *= 4
        if not np.isfinite(absolute).all():
            raise ValueError("Nonfinite model forecast")
        return absolute

    @torch.inference_mode()
    def voltage(self, run):
        if self.model.variant.task != "whole-run":
            raise ValueError("Expected whole-run bundle")
        if not whole_admitted(run):
            return None
        values = self.normalized(physical_features(run.values))[None]
        result = (
            self.model(
                values, torch.tensor([len(run.values)], device=self.device)
            )[0, :, 0]
            .cpu()
            .numpy()
        )
        return result * self.scaler["scale"][17] + self.scaler["mean"][17]


def replay_forecasts(runtime, run, batch_size=64):
    """Replay the 10 Hz logical cadence. It cannot attest wall-clock deadlines."""
    origins = forecast_origins(run)
    total = np.zeros((len(run.values), 4))
    count = np.zeros(len(run.values), dtype=np.int64)
    breaks = np.r_[0, np.cumsum(run.segment[1:] != run.segment[:-1])]
    for start in range(0, len(origins), batch_size):
        ids = origins[start : start + batch_size]
        forecasts = runtime.forecast(
            np.stack([run.values[i - 99 : i + 1] for i in ids])
        )
        for h in range(1, 11):
            target = ids + h
            inside = target < len(total)
            inside[inside] &= breaks[target[inside]] == breaks[ids[inside]]
            total[target[inside]] += forecasts[inside, h - 1]
            count[target[inside]] += 1
    matched = (count > 0) & run.observed.all(axis=1)
    predictions = np.full(total.shape, np.nan)
    predicted = count > 0
    predictions[predicted] = total[predicted] / count[predicted, None]
    residuals = run.values[:, TARGET_INDICES] - predictions
    residuals[~matched] = np.nan
    return predictions, count, residuals


@dataclass(frozen=True)
class Forecast:
    origin_us: int
    issued_us: int
    values: tuple


class FlightSession:
    """One flight's prediction/target lifecycle on a single mapped clock."""

    def __init__(self, runtime, clock_us=None, *, hold_us=0):
        self.runtime = runtime
        self.clock_us = clock_us or (lambda: time.monotonic_ns() // 1000)
        self.hold_us = hold_us
        self.history = deque(maxlen=100)
        self.pending = defaultdict(list)
        self.previous = None
        self.segment = None

    def step(
        self, origin_us, segment, row, observed, *, last_input_available_us
    ):
        if self.previous is not None and origin_us <= self.previous:
            raise ValueError("Frame times must be strictly increasing")
        if last_input_available_us > origin_us + self.hold_us:
            raise ValueError(
                "Late observations must not alter committed frame inputs"
            )
        values = np.asarray(row, dtype="f4")
        observed = np.asarray(observed)
        if (
            values.shape != (19,)
            or observed.shape != (4,)
            or observed.dtype != np.bool_
        ):
            raise ValueError("Expected common19 and four target quality flags")
        if segment != self.segment or (
            self.previous is not None and origin_us - self.previous != STEP_US
        ):
            self.history.clear()
            self.pending.clear()
        self.previous, self.segment = origin_us, segment
        matches = self.pending.pop(origin_us, ())
        residual = None
        actual = values[list(TARGET_INDICES)]
        if (
            matches
            and observed.all()
            and np.isfinite(actual).all()
            and segment >= 0
        ):
            residual = actual - np.mean(matches, axis=0)
        if segment < 0 or not np.isfinite(values).all():
            self.history.clear()
            return None, residual, "input_unavailable"
        self.history.append(values.copy())
        if len(self.history) != 100 or not observed.all():
            return None, residual, "history_or_anchor_unavailable"
        forecast = self.runtime.forecast(np.stack(self.history)[None])[0]
        issued = self.clock_us()
        if issued < origin_us + self.hold_us:
            raise ValueError("Issue clock precedes the frame publication time")
        if not self.hold_us and issued >= origin_us + STEP_US:
            return None, residual, "deadline_missed"
        issued_forecast = Forecast(
            origin_us, issued, tuple(tuple(v) for v in forecast.tolist())
        )
        for lead, value in enumerate(issued_forecast.values, 1):
            self.pending[origin_us + lead * STEP_US].append(value)
        return issued_forecast, residual, "published"
