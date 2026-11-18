"""Residual scores and timestamp-based alarms for the new models."""

from dataclasses import dataclass
from math import ceil, exp, log

import numpy as np

from .data import STEP_US, whole_admitted


@dataclass(frozen=True)
class FlightReference:
    center: np.ndarray
    precision: np.ndarray
    voltage_scale: float

    @classmethod
    def fit(cls, validation):
        r = np.array(validation, dtype=np.float64)
        if (
            r.ndim != 2
            or r.shape[1] != 4
            or len(r) < 5
            or not np.isfinite(r).all()
        ):
            raise ValueError(
                "Independent normal validation needs finite [N,4] residuals"
            )
        covariance = np.cov(r[:, :3], rowvar=False)
        regularized = 0.95 * covariance + 0.05 * np.diag(np.diag(covariance))
        regularized += np.eye(3) * max(
            1e-12, 1e-6 * float(np.trace(covariance))
        )
        return cls(
            r[:, :3].mean(0),
            np.linalg.inv(regularized),
            max(1e-10, float(np.mean(r[:, 3] ** 2))),
        )

    def score(self, residual):
        r = np.asarray(residual, dtype=np.float64)
        if r.shape != (4,) or not np.isfinite(r).all():
            raise ValueError("Expected four observed residuals")
        p = r[:3] - self.center
        battery = r[3] ** 2 * (1.0 if r[3] < 0 else 0.2) / self.voltage_scale
        return np.array((np.sqrt(max(0.0, p @ self.precision @ p)), battery))

    def score_series(self, residuals):
        residuals = np.asarray(residuals, dtype=np.float64)
        result = np.full((len(residuals), 2), np.nan)
        valid = np.isfinite(residuals).all(axis=1)
        r = residuals[valid]
        p = r[:, :3] - self.center
        distance = np.einsum("ij,ij->i", p @ self.precision, p)
        volts = r[:, 3]
        battery = volts**2 * np.where(volts < 0, 1.0, 0.2) / self.voltage_scale
        result[valid] = np.column_stack(
            (np.sqrt(np.maximum(distance, 0)), battery)
        )
        return result


class FlightAlarm:
    def __init__(self, thresholds):
        self.high = np.asarray(thresholds, dtype=np.float64)
        if (
            self.high.shape != (2,)
            or not np.isfinite(self.high).all()
            or np.any(self.high <= 0)
        ):
            raise ValueError("Expected two finite positive alarm thresholds")
        self.active = np.zeros(2, dtype=bool)
        self.reset()

    def reset(self):
        self.previous = None
        self.ema = None
        self.above_us = np.zeros(2, dtype=np.int64)
        self.below_us = np.zeros(2, dtype=np.int64)

    def update(self, time_us, score):
        if score is None:
            self.reset()
            return {"status": "unavailable", "alarm": None}
        score = np.asarray(score, dtype=np.float64)
        if (
            score.shape != (2,)
            or not np.isfinite(score).all()
            or np.any(score < 0)
        ):
            raise ValueError("Expected two finite residual scores")
        if self.previous is not None and time_us <= self.previous:
            raise ValueError(
                "Only one decision per increasing target time is allowed"
            )
        if self.previous is not None and time_us - self.previous != STEP_US:
            self.reset()
        alpha = 1 - exp(log(0.7) * STEP_US / 1_000_000)
        self.ema = (
            score
            if self.ema is None
            else alpha * score + (1 - alpha) * self.ema
        )
        self.previous = time_us
        self.above_us = np.where(
            self.ema > self.high, self.above_us + STEP_US, 0
        )
        self.below_us = np.where(
            self.ema < 0.7 * self.high, self.below_us + STEP_US, 0
        )
        self.active = (self.active | (self.above_us >= 3_000_000)) & (
            self.below_us < 10_000_000
        )
        return {
            "status": "available",
            "alarm": bool(self.active.any()),
            "domains": self.active.tolist(),
            "ema": self.ema.tolist(),
        }


def run_components(run, predicted_cell_v):
    predicted = np.asarray(predicted_cell_v)
    if (
        not whole_admitted(run)
        or predicted.shape != (len(run.values),)
        or not np.isfinite(predicted).all()
    ):
        return None
    residual = run.values[:, 17] / 4 - predicted
    short, long, count = 0.0, 0.0, 0
    for start in range(100, len(residual), 100):
        v = residual[start : start + 100][run.observed[start : start + 100, 3]]
        if len(v):
            center = v.mean()
            short += float(((v - center) ** 2).sum())
            long += float(center**2 * len(v))
            count += len(v)
    return np.sqrt(np.array((short, long)) / count), count / (
        len(residual) - 100
    )


@dataclass(frozen=True)
class RunReference:
    scale: np.ndarray
    threshold: float

    @classmethod
    def fit(cls, validation, calibration, alpha=0.05):
        v, c = (
            np.asarray(role, dtype=np.float64)
            for role in (validation, calibration)
        )
        if (
            any(
                a.ndim != 2
                or a.shape[1] != 2
                or not len(a)
                or not np.isfinite(a).all()
                or np.any(a < 0)
                for a in (v, c)
            )
            or not 0 < alpha < 1
        ):
            raise ValueError(
                "Two nonnegative residual components per independent run required"
            )
        scale = np.maximum(np.sqrt(np.mean(v**2, axis=0)), 1e-8)
        rank = ceil((len(c) + 1) * (1 - alpha))
        if rank > len(c):
            raise ValueError("Insufficient independent calibration runs")
        return cls(
            scale, float(np.sort(np.linalg.norm(c / scale, axis=1))[rank - 1])
        )

    def decide(self, record):
        if record is None:
            return {"status": "unavailable", "score": None, "alarm": None}
        components, fraction = record
        score = float(np.linalg.norm(components / self.scale))
        return {
            "status": "available",
            "short_cell_v": float(components[0]),
            "long_cell_v": float(components[1]),
            "coverage": fraction,
            "score": score,
            "alarm": score > self.threshold,
        }
