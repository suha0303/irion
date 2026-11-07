"""Run residual components and normal-only threshold calibration, separate from forward."""

import math
from dataclasses import dataclass

import numpy as np

from .models import BLOCK


def components(predicted, observed, *, cell_voltage_scale):
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    if predicted.ndim != 1 or predicted.shape != observed.shape:
        raise ValueError(
            "One unpadded predicted/observed voltage trajectory required"
        )
    if not math.isfinite(cell_voltage_scale) or cell_voltage_scale <= 0:
        raise ValueError("Positive frozen train voltage scale required")
    if len(predicted) <= BLOCK:
        return dict(status="unavailable", reason="no_post_initial_observations")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        return dict(
            status="unavailable", reason="missing_voltage_or_prediction"
        )
    residual = (observed - predicted) * cell_voltage_scale
    short, long, samples = 0.0, 0.0, 0
    for start in range(BLOCK, len(residual), BLOCK):
        block = residual[start : start + BLOCK]
        mean = block.mean()
        short += float(np.square(block - mean).sum())
        long += len(block) * float(mean**2)
        samples += len(block)
    return dict(
        status="available",
        samples=samples,
        short_cell_v=math.sqrt(short / samples),
        long_cell_v=math.sqrt(long / samples),
    )


def values(record):
    if record.get("status") != "available":
        raise ValueError(
            "Unavailable runs cannot be healthy zero-error observations"
        )
    result = np.array([record["short_cell_v"], record["long_cell_v"]])
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Invalid residual components")
    return result


@dataclass(frozen=True)
class Calibration:
    scales: tuple
    threshold: float
    alpha: float

    @classmethod
    def fit(cls, validation, calibration, *, alpha=0.05):
        # Records include a Run for role/lineage checks; its metadata never enters forward.
        from .data import validate_splits

        validate_splits([r for r, _ in [*validation, *calibration]])
        if not validation or not calibration or not 0 < alpha < 1:
            raise ValueError(
                "Independent normal validation and calibration required"
            )
        if any(r.role != "normal_validation" for r, _ in validation) or any(
            r.role != "normal_calibration" for r, _ in calibration
        ):
            raise ValueError(
                "Threshold/scales may not use train, anomaly or test runs"
            )
        # One run per independent calibration lineage group: no duplicate pseudo-counts.
        groups = set()
        for run, _ in calibration:
            if groups.intersection(run.lineage):
                raise ValueError(
                    "Calibration requires independent lineage groups"
                )
            groups.update(run.lineage)
        scales = np.maximum(
            np.sqrt(np.mean([values(c) ** 2 for _, c in validation], axis=0)),
            1e-8,
        )
        scores = sorted(
            float(np.max(values(c) / scales)) for _, c in calibration
        )
        rank = math.ceil((len(scores) + 1) * (1 - alpha))
        if rank > len(scores):
            raise ValueError(
                "Insufficient independent normal calibration runs for finite threshold"
            )
        return cls(tuple(scales), scores[rank - 1], alpha)

    def decide(self, record):
        if record.get("status") != "available":
            return dict(
                status="unavailable", alarm=None, reason=record.get("reason")
            )
        score = float(np.max(values(record) / np.array(self.scales)))
        return dict(
            status="available", score=score, alarm=score > self.threshold
        )
