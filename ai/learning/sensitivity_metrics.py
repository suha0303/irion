"""Frozen-prediction batch sensitivity on continuous normal-only blocks.

The caller supplies a separately fitted diagnostic calibration. This module never
fits a threshold, loads a model, reads development truth, or changes a criterion.
"""

import hashlib
import json

import numpy as np

from capstone_ai.detection import DOMAINS
from capstone_ai.detection_protocol import score_sequence
from capstone_ai.mapping import TARGETS

SCHEMA = "drone-ai.sensitivity-policy/1"


def validate_policy(policy):
    required = {
        "schema",
        "batch_size",
        "require_same_batch_exact",
        "max_scaled_abs",
        "physical_max",
        "max_score_delta_over_high",
        "max_ewma_delta_over_high",
        "max_changed_alarm_states",
    }
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("One complete common sensitivity policy is required; extra keys forbidden")
    if (
        policy["schema"] != SCHEMA
        or type(policy["batch_size"]) is not int
        or policy["batch_size"] != 256
        or policy["require_same_batch_exact"] is not True
        or type(policy["max_changed_alarm_states"]) is not int
        or policy["max_changed_alarm_states"] != 0
    ):
        raise ValueError("Sensitivity requires batch256, exact repeats and zero alarm changes")
    for key in ("max_scaled_abs", "max_score_delta_over_high", "max_ewma_delta_over_high"):
        value = policy[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("Sensitivity budgets must be finite nonnegative numbers")
    limits = policy["physical_max"]
    if (
        not isinstance(limits, list)
        or len(limits) != 4
        or any(
            type(value) not in (int, float) or not np.isfinite(value) or value < 0
            for value in limits
        )
    ):
        raise ValueError("Four common finite physical limits are required")


def _canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _number_array(value, shape, label, *, finite=False, positive=False):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "iuf":
        raise ValueError(f"Invalid {label} shape or numeric type")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"Nonfinite {label}")
    if positive and not (value > 0).all():
        raise ValueError(f"Positive {label} required")
    return value.copy()


def _identities(values, count):
    if not isinstance(values, dict) or set(values) != {"run_id", "segment", "origin_us"}:
        raise ValueError("Explicit run/segment/origin identities required")
    run = np.asarray(values["run_id"])
    if run.shape != (count,) or run.dtype.kind not in "US" or np.any(run.astype(str) == ""):
        raise ValueError("Nonempty string run identities required")
    result = {"run_id": run.astype(str)}
    for key in ("segment", "origin_us"):
        value = np.asarray(values[key])
        if (
            value.shape != (count,)
            or value.dtype.kind not in "iu"
            or (value < 0).any()
            or (value > np.iinfo(np.int64).max).any()
        ):
            raise ValueError("Nonnegative integer segment and origin identities required")
        result[key] = value.astype(np.int64)
    seen, previous_scope, previous_origin = set(), None, None
    for run_id, segment, origin in zip(result["run_id"], result["segment"], result["origin_us"]):
        scope = (str(run_id), int(segment))
        if scope != previous_scope:
            if scope in seen:
                raise ValueError("A run/segment block must not reappear")
            seen.add(scope)
        elif int(origin) - previous_origin != 1_000_000:
            raise ValueError("Continuous blocks require exactly one-second ordered origins")
        previous_scope, previous_origin = scope, int(origin)
    return result


def _trajectory(scores, identities, high):
    decisions = score_sequence(
        scores, identities["run_id"], identities["segment"], identities["origin_us"], high
    )
    count = len(decisions)
    result = {"scores": scores, "ewma": np.empty((count, 2), dtype=np.float64)}
    for key in ("active", "start", "end"):
        result["union_" + key] = np.zeros(count, dtype=np.bool_)
        result["domain_" + key] = np.zeros((count, 2), dtype=np.bool_)
    previous_scope, previous_active = None, np.zeros(2, dtype=np.bool_)
    for index, decision in enumerate(decisions):
        scope = (decision["run_id"], decision["segment_id"])
        if scope != previous_scope:
            previous_active = np.zeros(2, dtype=np.bool_)
        result["union_active"][index] = decision["alarm_active"]
        result["union_start"][index] = decision["episode_start"]
        result["union_end"][index] = decision["episode_end"]
        for domain_index, name in enumerate(DOMAINS):
            domain = decision["domains"][name]
            active = domain["alarm_active"]
            result["ewma"][index, domain_index] = domain["ewma"][0]
            result["domain_active"][index, domain_index] = active
            result["domain_start"][index, domain_index] = domain["episode_start"]
            result["domain_end"][index, domain_index] = previous_active[domain_index] and not active
        previous_scope = scope
        previous_active = result["domain_active"][index].copy()
    return result


def assess_sensitivity(
    reference_batch1,
    batch256,
    repeat256,
    actual_scaled,
    identities,
    diagnostic_calibration,
    physical_scale4,
    policy,
):
    """Return ``(report, evidence_arrays)`` without modifying supplied inputs.

    Numerical rejection returns FAILED_STOPPED so the caller can preserve exact
    arrays. Malformed data/calibration/identity raises ValueError. Physical scales
    must already map the four scaled targets to rad/s, rad/s, rad/s and pack volts.
    Every comparison uses the same fixed diagnostic scale and high thresholds.
    """
    validate_policy(policy)
    policy = json.loads(json.dumps(policy, allow_nan=False))
    actual = np.asarray(actual_scaled)
    if actual.ndim != 3 or actual.shape[1:] != (10, 4) or len(actual) == 0:
        raise ValueError("Nonempty N x10 x4 normal targets required")
    shape = actual.shape
    actual = _number_array(actual, shape, "normal target", finite=True)
    reference = _number_array(reference_batch1, shape, "reference prediction")
    batch = _number_array(batch256, shape, "batch prediction")
    repeat = _number_array(repeat256, shape, "repeat prediction")
    ids = _identities(identities, len(actual))
    physical = _number_array(physical_scale4, (4,), "physical scale", finite=True, positive=True)
    if not isinstance(diagnostic_calibration, dict):
        raise ValueError("One fixed diagnostic calibration required")
    scale = _number_array(
        diagnostic_calibration.get("reference_error_scale_scaled"),
        (10, 4),
        "diagnostic residual scale",
        finite=True,
        positive=True,
    )
    high = _number_array(
        diagnostic_calibration.get("high"), (2,), "diagnostic threshold", finite=True, positive=True
    )
    # Bind the complete calibration, including fitting provenance where supplied.
    calibration_sha = _canonical_hash(diagnostic_calibration)
    finite = all(np.isfinite(value).all() for value in (reference, batch, repeat))
    repeated = batch.dtype == repeat.dtype and batch.tobytes() == repeat.tobytes()
    constraints = {"finite_predictions": bool(finite), "same_batch_exact": bool(repeated)}
    report = {
        "schema": "drone-ai.sensitivity-assessment/1",
        "status": "FAILED_STOPPED",
        "policy": policy,
        "policy_sha256": _canonical_hash(policy),
        "diagnostic_calibration_sha256": calibration_sha,
        "windows": len(actual),
        "target_order": list(TARGETS),
        "domain_order": list(DOMAINS),
        "physical_units": ["rad/s", "rad/s", "rad/s", "V"],
        "physical_target_scale": physical.tolist(),
        "constraints": constraints,
        "weights_changed": False,
        "threshold_changed_between_paths": False,
        "event_truth_used": False,
        "production_calibration_reused": False,
        "interpretation": "normal-block sensitivity only; not event recall, false-alarm certification, "
        "real-time latency validation or equivalence for every possible input",
    }
    arrays = {
        "reference_batch1": reference,
        "batch256": batch,
        "repeat256": repeat,
        "actual_scaled": actual,
        **ids,
        "physical_target_scale": physical,
        "diagnostic_error_scale": scale,
        "diagnostic_high": high,
    }
    if not finite:
        report["reason"] = "Nonfinite predictions; score and alarm comparison not executed"
        return report, arrays
    reference, batch, actual = [value.astype(np.float64) for value in (reference, batch, actual)]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = np.abs(reference - batch)
        physical_delta = delta * physical
        contributions = [(actual - prediction) ** 2 / scale for prediction in (reference, batch)]
        scores = [
            np.stack([value[:, :, :3].mean((1, 2)), value[:, :, 3].mean(1)], axis=1)
            for value in contributions
        ]
    if not all(np.isfinite(value).all() for value in (delta, physical_delta, *scores)):
        constraints["finite_derived_scores"] = False
        report["reason"] = "Derived numeric error or score overflow; alarm comparison not executed"
        return report, arrays
    constraints.update(
        finite_derived_scores=True,
        scaled_absolute=bool(delta.max() <= policy["max_scaled_abs"]),
        physical_absolute=bool(np.all(physical_delta.max(axis=(0, 1)) <= policy["physical_max"])),
    )
    trajectories = [_trajectory(value, ids, high) for value in scores]
    before, after = trajectories
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score_delta = np.abs(before["scores"] - after["scores"]) / high
        ewma_delta = np.abs(before["ewma"] - after["ewma"]) / high
    for prefix, values in zip(("reference", "batch"), trajectories):
        arrays.update({prefix + "_" + key: value for key, value in values.items()})
    if not np.isfinite(score_delta).all() or not np.isfinite(ewma_delta).all():
        constraints["finite_normalized_deltas"] = False
        report["reason"] = "Threshold-normalized score or EWMA delta overflow"
        return report, arrays
    changed = {}
    changed_origin = np.zeros(len(actual), dtype=np.bool_)
    for kind in ("active", "start", "end"):
        union = before["union_" + kind] != after["union_" + kind]
        domains = before["domain_" + kind] != after["domain_" + kind]
        changed["union_" + kind] = int(union.sum())
        changed["domain_" + kind] = dict(zip(DOMAINS, map(int, domains.sum(axis=0))))
        changed_origin |= union | domains.any(axis=1)
    constraints.update(
        finite_normalized_deltas=True,
        score_delta=bool(score_delta.max() <= policy["max_score_delta_over_high"]),
        ewma_delta=bool(ewma_delta.max() <= policy["max_ewma_delta_over_high"]),
        alarm_states=bool(int(changed_origin.sum()) <= policy["max_changed_alarm_states"]),
    )
    report.update(
        status="PASSED" if all(constraints.values()) else "FAILED_STOPPED",
        max_scaled_absolute_difference=float(delta.max()),
        per_target_max_scaled_absolute=delta.max(axis=(0, 1)).tolist(),
        per_target_max_physical=physical_delta.max(axis=(0, 1)).tolist(),
        max_score_delta_over_high=float(score_delta.max()),
        per_domain_max_score_delta_over_high=score_delta.max(axis=0).tolist(),
        max_ewma_delta_over_high=float(ewma_delta.max()),
        per_domain_max_ewma_delta_over_high=ewma_delta.max(axis=0).tolist(),
        changed_alarm_origins=int(changed_origin.sum()),
        changed_alarm_fields=changed,
        reference_union_episodes=int(before["union_start"].sum()),
        batch_union_episodes=int(after["union_start"].sum()),
        reference_domain_episodes=dict(zip(DOMAINS, map(int, before["domain_start"].sum(axis=0)))),
        batch_domain_episodes=dict(zip(DOMAINS, map(int, after["domain_start"].sum(axis=0)))),
        legacy_mixed_failed_elements=int(
            np.count_nonzero(delta > 1e-5 + 1e-5 * np.maximum(np.abs(reference), np.abs(batch)))
        ),
        legacy_mixed_is_admission_criterion=False,
    )
    arrays.update(
        score_delta_over_high=score_delta,
        ewma_delta_over_high=ewma_delta,
        changed_alarm_origin=changed_origin,
    )
    return report, arrays
