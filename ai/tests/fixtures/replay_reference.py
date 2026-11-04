"""Frozen origin-ordered ledger before the replay efficiency change.

Independent regression oracle; never imported by production code.
"""
import numpy as np

from capstone_ai.detection import DomainAlarms, residual_evidence
from capstone_ai.inference import PendingForecasts


class DetectionLedger(PendingForecasts):
    """Origin-ordered decisions. A missing earlier block cannot be overtaken."""

    def __init__(self, error_scale, threshold):
        scale = np.asarray(error_scale, dtype=float)
        if scale.shape != (10, 4) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("finite positive 10x4 error scale required")
        super().__init__(scale, threshold)
        self.error_scale = scale  # Do not apply a second floor in physical units.
        self.domain_alarm = DomainAlarms(threshold)
        self.scope = None
        self.last_origin = None
        self.target_reasons = {}

    def issue(self, run_id, clock, segment, origin, issued, prediction, timing_status=None):
        values = np.asarray(prediction, dtype=float)
        if values.shape != (10, 4) or not np.isfinite(values).all():
            raise ValueError("nonfinite or malformed forecast")
        if issued is not None and issued < origin:
            raise ValueError("issuance precedes input origin")
        if timing_status == "awaiting_targets" and (issued is None or issued >= origin + 100000):
            raise ValueError("timing status cannot override the strict deadline")
        scope = (run_id, clock, segment)
        if scope != self.scope:
            if len(self.forecasts) != len(self.decisions):
                raise ValueError("close pending forecasts before a segment reset")
            self.domain_alarm = DomainAlarms(self.alarm.high)
            self.scope, self.last_origin = scope, None
        if self.last_origin is not None and origin < self.last_origin:
            raise ValueError("forecast origin regressed")
        key = super().issue(run_id, clock, segment, origin, issued, values, timing_status)
        self.last_origin = origin
        return key

    def observe(self, run_id, clock, segment, time_us, available_us, values):
        if available_us < time_us:
            raise ValueError("target available before sample")
        reason = None
        if values is None:
            reason = "invalid_target_quality"
        else:
            a = np.asarray(values, dtype=float)
            if a.shape != (4,) or not np.isfinite(a).all():
                values, reason = None, "nonfinite_or_malformed_target"
            else:
                values = a.tolist()
        super().observe(run_id, clock, segment, time_us, available_us, values)
        self.target_reasons[(run_id, clock, segment, time_us)] = reason

    def advance(self, decision_time_us, closed=False):
        for key, forecast in self.forecasts.items():
            if key in self.decisions:
                continue
            if decision_time_us < max(forecast["origin_time_us"], forecast["issued_time_us"] or 0):
                raise ValueError("decision precedes forecast issuance")
            times = forecast["target_times_us"]
            keys = [(forecast["run_id"], forecast["clock_id"], forecast["segment_id"], t) for t in times]
            actual = [self.actuals.get(k) for k in keys]
            arrived = [a is not None and a["available_us"] <= decision_time_us and t <= decision_time_us
                       for a, t in zip(actual, times)]
            if not all(arrived) and not closed:
                break
            valid = [arrived[i] and a is not None and a["values"] is not None
                     for i, a in enumerate(actual)]
            reasons = []
            if not all(arrived):
                reasons.append("targets_missing_or_not_yet_available")
            reasons.extend(sorted({r for k in keys if (r := self.target_reasons.get(k)) is not None}))
            if forecast["status"] != "awaiting_targets":
                reasons.append(forecast["status"])
            evidence = {"scores": None, "signed_residual": None, "absolute_error": None,
                        "squared_error": None, "normalized_contribution": None}
            if all(valid) and not reasons:
                evidence = residual_evidence(forecast["prediction"],
                                             [a["values"] for a in actual if a is not None], self.error_scale)
            alarm = self.domain_alarm.update(evidence["scores"], forecast["origin_time_us"])
            if forecast["status"] != "awaiting_targets":
                alarm["state"] = forecast["status"]
            self.decisions[key] = {
                "schema": "drone.decision/3.0", "detector_revision": "normal-postprocess-v7",
                "forecast_id": key, "run_id": forecast["run_id"], "clock_id": forecast["clock_id"],
                "segment_id": forecast["segment_id"], "origin_time_us": forecast["origin_time_us"],
                "decision_time_us": decision_time_us,
                "target_times_us": times, "matched_target_count": sum(valid), "coverage": sum(valid) / 10,
                "last_target_available_us": max((a["available_us"] for a, ok in zip(actual, arrived)
                                                 if ok and a is not None), default=None),
                "unscored_reasons": reasons, "diagnosis_status": "not_established",
                "health_estimate": {"soh": None, "rul": None}, **evidence, **alarm,
            }
