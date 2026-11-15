"""Aircraft normal references with the frozen final-evaluation score kernels.

Reference data, threshold data and evaluation data are separate flights. Few
normal flights permit empirical calibration, not a false-alarm guarantee.
"""

import math

import numpy as np

from capstone_ai import detection_protocol
from capstone_ai.detection import DomainAlarms
from learning.run_health19.scoring import Calibration, values

POLICY = "aircraft-final-protocol/1"


def validate_parameters(task, parameters):
    """Reject malformed domain thresholds before selecting a saved version."""
    if task == "forecast":
        DomainAlarms(parameters["high"])
        forecast_scores(dict(residual_scaled=np.zeros((1, 10, 4))), parameters)
    elif task == "run_health":
        scales = np.asarray(parameters["scales"])
        if (
            scales.shape != (2,)
            or not np.isfinite(scales).all()
            or np.any(scales <= 0)
        ):
            raise ValueError("Two positive health reference scales required")
        threshold = parameters["threshold"]
        if (
            not math.isfinite(threshold)
            or threshold <= 0
            or not 0 < parameters["alpha"] < 1
        ):
            raise ValueError("Invalid health threshold or alpha")
    else:
        raise ValueError("Unknown detector task")


class LiveDetector:
    """Keep the frozen temporal alarm state across arrived forecast blocks."""

    def __init__(self, record):
        self.parameters = record["parameters"]
        self.calibration_id = record["id"]
        self.state = DomainAlarms(self.parameters["high"])
        self.last = None
        self.reset = None

    def update(self, residual, origin, reset):
        if self.reset != reset:
            self.state = DomainAlarms(self.parameters["high"])
        if (
            self.last is not None
            and self.reset == reset
            and origin - self.last < 1000000
        ):
            raise ValueError("duplicate or overlapping live forecast blocks")
        score = forecast_scores(
            dict(residual_scaled=np.asarray(residual)[None]), self.parameters
        )[0]
        row = self.state.update(score, origin)
        self.last, self.reset = origin, reset
        ratios = [
            row["domains"][name]["ewma"][0] / self.parameters["high"][i]
            for i, name in enumerate(("propulsion", "battery"))
        ]
        return dict(
            status="suspect" if row["alarm_active"] else "normal",
            threshold=1.0,
            scores=[float(max(ratios))],
            alarms=[row["alarm_active"]],
            observations=1,
            exceedances=int(row["alarm_active"]),
            calibration_id=self.calibration_id,
            detector_policy=POLICY,
            domain_scores=score.tolist(),
            domain_thresholds=self.parameters["high"],
            active_domains=[
                name
                for name, value in row["domains"].items()
                if value["alarm_active"]
            ],
            episode_start=row["episode_start"],
            real_aircraft_validated=False,
        )


def forecast_scores(evidence, parameters):
    """Use all ten leads and independent propulsion/battery domains."""
    residual = np.asarray(evidence["residual_scaled"], dtype="f8")
    scale = np.asarray(parameters["reference_error_scale_scaled"], dtype="f8")
    if (
        residual.ndim != 3
        or residual.shape[1:] != (10, 4)
        or scale.shape != (10, 4)
        or np.any(scale <= 0)
        or not np.isfinite(residual).all()
        or not np.isfinite(scale).all()
    ):
        raise ValueError(
            "Finite ten-lead residuals and positive reference required"
        )
    contribution = residual**2 / scale
    return np.stack(
        [contribution[:, :, :3].mean((1, 2)), contribution[:, :, 3].mean(1)],
        axis=1,
    )


def fit_parameters(task, references, calibrations, physical_scale):
    """Fit normal scales and thresholds without reading evaluation labels."""
    if not references or not calibrations:
        raise ValueError(
            "Separate normal reference and calibration flights required"
        )
    if task == "forecast":

        def residuals(rows):
            return np.concatenate(
                [r["scoring"]["residual_scaled"] for r in rows]
            )

        ids = np.concatenate(
            [
                np.repeat(r["source_sha256"], len(r["scoring"]["scores"]))
                for r in calibrations
            ]
        )
        segments = np.concatenate(
            [r["scoring"]["segment_id"] for r in calibrations]
        )
        origins = np.concatenate(
            [r["scoring"]["origin_us"] for r in calibrations]
        )
        return detection_protocol.fit_calibration(
            residuals(references),
            residuals(calibrations),
            ids,
            segments,
            origins,
            physical_scale,
            {r["source_sha256"]: r["source_sha256"] for r in calibrations},
        )
    # The frozen health score uses independently scaled short/long residuals.
    # Equal flight weighting prevents many segments from becoming fake runs.
    per_reference = [
        np.mean([values(c) ** 2 for c in r["scoring"]["components"]], axis=0)
        for r in references
    ]
    scales = np.maximum(np.sqrt(np.mean(per_reference, axis=0)), 1e-8)
    per_flight = [
        max(
            float(np.max(values(c) / scales))
            for c in r["scoring"]["components"]
        )
        for r in calibrations
    ]
    rank = math.ceil((len(per_flight) + 1) * 0.95)
    finite_rank = rank <= len(per_flight)
    threshold = sorted(per_flight)[rank - 1] if finite_rank else max(per_flight)
    return dict(
        scales=scales.tolist(),
        threshold=max(threshold, 1e-8),
        alpha=0.05,
        method="finite_sample_rank"
        if finite_rank
        else "few_flight_empirical_max",
        finite_sample_rank_available=finite_rank,
        independent_false_alarm_target_verified=False,
        normal_calibration_flights=len(per_flight),
    )


def decide(task, evidence, parameters):
    """Return temporal alarms using the same kernels as the frozen evaluator."""
    if task == "forecast":
        raw = forecast_scores(evidence, parameters)
        rows = detection_protocol.score_sequence(
            raw,
            np.repeat("local-flight", len(raw)),
            evidence["segment_id"],
            evidence["origin_us"],
            parameters["high"],
        )
        alarms = [bool(r["alarm_active"]) for r in rows]
        ratios = [
            max(
                np.asarray(
                    [
                        r["domains"][n]["ewma"][0]
                        for n in ("propulsion", "battery")
                    ]
                )
                / parameters["high"]
            )
            for r in rows
        ]
        details = dict(
            domain_scores=raw.tolist(),
            domain_thresholds=parameters["high"],
            domain_alarms={
                n: [bool(r["domains"][n]["alarm_active"]) for r in rows]
                for n in ("propulsion", "battery")
            },
            domain_episode_starts={
                n: [bool(r["domains"][n]["episode_start"]) for r in rows]
                for n in ("propulsion", "battery")
            },
            episode_starts=[bool(r["episode_start"]) for r in rows],
            episodes=sum(r["episode_start"] for r in rows),
        )
    else:
        cal = Calibration(
            tuple(parameters["scales"]),
            parameters["threshold"],
            parameters["alpha"],
        )
        rows = [cal.decide(c) for c in evidence["components"]]
        alarms = [bool(r["alarm"]) for r in rows]
        ratios = [r["score"] / cal.threshold for r in rows]
        details = dict(
            component_scales=list(cal.scales), raw_threshold=cal.threshold
        )
    return dict(
        status="suspect" if any(alarms) else "normal",
        scores=list(map(float, ratios)),
        alarms=alarms,
        exceedances=sum(alarms),
        observations=len(alarms),
        threshold=1.0,
        score_unit="ratio_to_domain_threshold_after_frozen_postprocessing",
        detector_policy=POLICY,
        real_aircraft_validated=False,
        false_alarm_guarantee=False,
        **details,
    )
