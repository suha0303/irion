"""Bounded normal-only inference batch check; no fitting or event truth."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from capstone_ai.mapping import TARGETS
from execution.common import child, read, sha, write

_TARGETS = tuple(TARGETS)
_EXPECTED_OUTPUT = (10, 4)
_POLICY_SCHEMA = "drone-ai.numeric-policy/1"


def _fresh(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Diagnostic evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _array_signature(arrays):
    signature = hashlib.sha256()
    for key, value in arrays.items():
        value = np.asarray(value)
        signature.update(key.encode())
        signature.update(str((value.shape, value.dtype.str)).encode())
        signature.update(np.ascontiguousarray(value).tobytes())
    return signature.hexdigest()


def _difference(reference, actual, physical_scale):
    """Diagnostic statistics only; no acceptance criterion is selected here."""
    reference = reference.astype(np.float64, copy=False)
    actual = actual.astype(np.float64, copy=False)
    absolute = np.abs(reference - actual)
    relative = absolute / np.maximum(np.abs(reference), np.finfo(np.float64).eps)
    physical = absolute * physical_scale[None, None, :]
    old_fail = absolute > 1e-6 + 1e-5 * np.abs(actual)
    at = tuple(int(i) for i in np.unravel_index(np.argmax(absolute), absolute.shape))
    return {
        "max_absolute_difference": float(absolute.max()),
        "p99_absolute_difference": float(np.quantile(absolute, 0.99)),
        "max_relative_difference": float(relative.max()),
        "p99_relative_difference": float(np.quantile(relative, 0.99)),
        "relative_denominator": "max(abs(reference_prediction), float64_epsilon)",
        "reference_exact_zero_count": int(np.count_nonzero(reference == 0)),
        "per_target_max_absolute": absolute.max(axis=(0, 1)).tolist(),
        "per_target_p99_absolute": np.quantile(absolute, 0.99, axis=(0, 1)).tolist(),
        "per_target_max_relative": relative.max(axis=(0, 1)).tolist(),
        "per_target_p99_relative": np.quantile(relative, 0.99, axis=(0, 1)).tolist(),
        "per_lead_target_max_absolute": absolute.max(axis=0).tolist(),
        "per_lead_target_p99_absolute": np.quantile(absolute, 0.99, axis=0).tolist(),
        "per_lead_target_max_relative": relative.max(axis=0).tolist(),
        "per_lead_target_p99_relative": np.quantile(relative, 0.99, axis=0).tolist(),
        "per_target_max_physical": physical.max(axis=(0, 1)).tolist(),
        "per_target_p99_physical": np.quantile(physical, 0.99, axis=(0, 1)).tolist(),
        "per_lead_target_max_physical": physical.max(axis=0).tolist(),
        "per_lead_target_p99_physical": np.quantile(physical, 0.99, axis=0).tolist(),
        "max_location": {"sample": at[0], "lead": at[1], "target": _TARGETS[at[2]]},
        "legacy_check": {
            "rtol": 1e-5, "atol": 1e-6,
            "formula": "abs(reference-actual) <= atol + rtol*abs(actual)",
            "failed_elements": int(old_fail.sum()),
            "per_target_failed_elements": old_fail.sum(axis=(0, 1)).tolist(),
        },
    }


def diagnose_prediction_batch(
    predictor, normal, output, *, batch_size=256, check=lambda: None, save_arrays=True
):
    """Save an untuned, normal-only diagnostic for one full batch and one tail.

    This deliberately does not make an acceptance decision from batch-size
    differences. Only invalid/nonfinite data or outputs stop this diagnostic.
    A separately frozen policy must assess the saved evidence for every model.
    """
    if type(batch_size) is not int or batch_size < 2 or len(normal["X"]) < batch_size + 1:
        raise ValueError("One full inference batch plus tail is required")
    output = _fresh(output)
    arrays_path = output.with_suffix(".npz")
    if save_arrays:
        _fresh(arrays_path)
    selected = np.linspace(0, len(normal["X"]) - 1, batch_size + 1, dtype=np.int64)
    sample = {key: np.asarray(normal[key][selected]) for key in ("X", "C", "Y")}
    identity = {key: np.asarray(normal[key][selected])
                for key in ("run_id", "segment", "origin_us")}
    identity["run_id"] = identity["run_id"].astype(str)
    scale = np.asarray(predictor.scaler["y_scale"], dtype=np.float64).copy()
    if scale.shape != (4,) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Expected four positive finite target scales")
    scale[-1] *= 4  # The trained voltage target is per cell; evidence reports pack volts.
    expected = (len(selected), *_EXPECTED_OUTPUT)
    if sample["Y"].shape != expected or not all(np.isfinite(v).all() for v in sample.values()):
        raise ValueError("Normal diagnostic sample has invalid shape or nonfinite values")

    def predict(size):
        predictions = []
        started = time.perf_counter()
        for start in range(0, len(selected), size):
            check()
            predictions.append(np.asarray(predictor.predict_scaled(
                sample["X"][start:start + size], sample["C"][start:start + size]
            )).copy())
        return np.concatenate(predictions), time.perf_counter() - started

    # Identical input/order/backend settings for both production-size passes.
    reference, single_s = predict(1)
    first, first_s = predict(batch_size)
    repeat, repeat_s = predict(batch_size)
    outputs = {"reference_batch1": reference, "batch_first": first, "batch_repeat": repeat}
    finite = all(value.shape == expected and np.isfinite(value).all()
                 for value in outputs.values())
    evidence = dict(
        **outputs, sample_y=sample["Y"], selected_indices=selected,
        physical_target_scale=scale, **identity,
    )
    arrays = None
    if save_arrays:
        with arrays_path.open("xb") as stream:
            np.savez_compressed(stream, **evidence)
        arrays = {"file": arrays_path.name, "sha256": sha(arrays_path)}
    report = {
        "schema": "drone-ai.numeric-diagnostic/1",
        "status": "DIAGNOSTIC_COMPLETE" if finite else "FAILED_STOPPED",
        "role": "normal_validation", "windows": len(selected),
        "requested_batch_size": batch_size, "tail_windows": 1,
        "selected_indices": selected.tolist(),
        "input_sha256": _array_signature({"X": sample["X"], "C": sample["C"],
                                         "selected": selected}),
        "sample_sha256": _array_signature({**sample, **identity, "selected": selected}),
        "arrays": arrays, "finite": bool(finite),
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "output_dtypes": {key: str(value.dtype) for key, value in outputs.items()},
        "target_order": list(_TARGETS), "physical_units": ["rad/s", "rad/s", "rad/s", "V"],
        "physical_target_scale": scale.tolist(),
        "same_batch_repeated_exactly": bool(finite and np.array_equal(first, repeat)),
        "single_window_seconds": single_s, "batched_seconds": first_s,
        "repeat_batched_seconds": repeat_s,
        "timing_scope": "small resident sample; forward/transfer only, not training or replay throughput",
        "weights_changed": False, "calibration_changed": False, "event_truth_used": False,
        "acceptance_policy_applied": False,
        "interpretation": "sample numerical diagnostic; not a guarantee of identical threshold decisions",
    }
    if finite:
        report["batch1_vs_batch"] = _difference(reference, first, scale)
        report["same_batch_repeat"] = _difference(first, repeat, scale)
        residual = (sample["Y"].astype(np.float64) - first.astype(np.float64)) * scale
        report["per_target_sample_residual_rmse_physical"] = np.sqrt(
            np.mean(residual ** 2, axis=(0, 1))
        ).tolist()
    write(output, report)
    if not finite:
        raise ValueError("Nonfinite or malformed prediction diagnostic; preserve evidence and stop")
    return report


def _validate_policy(policy):
    required = {"schema", "batch_size", "windows", "rtol", "atol", "require_same_batch_exact"}
    optional = {"physical_atol", "max_absolute_difference"}
    if not isinstance(policy, dict) or set(policy) - required - optional or required - set(policy):
        raise ValueError("One explicit common numerical policy is required; unknown keys forbidden")
    if policy["schema"] != _POLICY_SCHEMA:
        raise ValueError("Unsupported numerical policy schema")
    if (type(policy["batch_size"]) is not int or policy["batch_size"] < 2
            or type(policy["windows"]) is not int
            or policy["windows"] != policy["batch_size"] + 1
            or type(policy["require_same_batch_exact"]) is not bool):
        raise ValueError("Invalid common diagnostic batch/window/repeat policy")
    scalar_limits = ["rtol", "atol"]
    if "max_absolute_difference" in policy:
        scalar_limits.append("max_absolute_difference")
    for key in scalar_limits:
        if type(policy[key]) not in (int, float) or not np.isfinite(policy[key]) or policy[key] < 0:
            raise ValueError("Common tolerances must be finite nonnegative numbers")
    if "physical_atol" in policy:
        value = policy["physical_atol"]
        if (not isinstance(value, list) or len(value) != 4
                or any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in value)):
            raise ValueError("Physical tolerance must contain four common finite nonnegative limits")


def assess_diagnostic(report_path, policy, output=None):
    """Apply an explicit common policy to hash-verified saved predictions.

    Callers must freeze and reuse this exact policy across every compared model.
    No model-specific policy keys, fitted tolerances or automatic retries exist.
    """
    _validate_policy(policy)
    report_path = Path(report_path)
    output = _fresh(output) if output is not None else None
    report = read(report_path)
    if (report.get("schema") != "drone-ai.numeric-diagnostic/1"
            or report.get("status") != "DIAGNOSTIC_COMPLETE"
            or report.get("role") != "normal_validation"
            or report.get("windows") != policy["windows"]
            or report.get("requested_batch_size") != policy["batch_size"]):
        raise ValueError("Diagnostic does not match the frozen common policy")
    seal = report.get("arrays")
    if not isinstance(seal, dict):
        raise ValueError("Assessment requires saved exact diagnostic arrays")
    arrays_path = child(report_path.parent, seal["file"])
    if sha(arrays_path) != seal["sha256"]:
        raise ValueError("Diagnostic array hash changed")
    with np.load(arrays_path, allow_pickle=False) as values:
        reference, first, repeat, scale = [values[key] for key in (
            "reference_batch1", "batch_first", "batch_repeat", "physical_target_scale"
        )]
    expected = (policy["windows"], *_EXPECTED_OUTPUT)
    finite = all(v.shape == expected and np.isfinite(v).all() for v in (reference, first, repeat))
    if scale.shape != (4,) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid saved physical scale")
    constraints = {"finite": bool(finite)}
    differences = None
    if finite:
        absolute = np.abs(reference.astype(np.float64) - first.astype(np.float64))
        constraints["same_batch_repeat"] = bool(
            not policy["require_same_batch_exact"] or np.array_equal(first, repeat)
        )
        magnitude = np.maximum(np.abs(reference.astype(np.float64)),
                               np.abs(first.astype(np.float64)))
        constraints["batch1_vs_batch"] = bool(np.all(
            absolute <= policy["atol"] + policy["rtol"] * magnitude
        ))
        if "max_absolute_difference" in policy:
            constraints["normalized_max_absolute"] = bool(
                absolute.max() <= policy["max_absolute_difference"]
            )
        if "physical_atol" in policy:
            constraints["physical_max_absolute"] = bool(np.all(
                absolute * scale <= np.asarray(policy["physical_atol"])
            ))
        differences = _difference(reference, first, scale)
    passed = all(constraints.values())
    result = {
        "schema": "drone-ai.numeric-assessment/1", "status": "PASSED" if passed else "FAILED_STOPPED",
        "policy": policy,
        "policy_sha256": hashlib.sha256(json.dumps(
            policy, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest(),
        "diagnostic_sha256": sha(report_path), "arrays_sha256": seal["sha256"],
        "input_sha256": report["input_sha256"], "constraints": constraints,
        "common_comparison_formula": "abs(reference-batched) <= atol + rtol*max(abs(reference),abs(batched))",
        "batch1_vs_batch": differences,
        "weights_changed": False, "calibration_changed": False, "event_truth_used": False,
        "interpretation": "frozen common sample acceptance; not proof of globally identical threshold decisions",
    }
    if output is not None:
        write(output, result)
    if not passed:
        raise ValueError("Common numerical policy failed; preserve evidence and stop")
    return result


def check_prediction_batch(predictor, normal, output, *, batch_size=256,
                           rtol=1e-5, atol=1e-6, check=lambda: None):
    if type(batch_size) is not int or batch_size < 2 or len(normal["X"]) < batch_size + 1:
        raise ValueError("One full inference batch plus tail is required")
    selected = np.linspace(0, len(normal["X"]) - 1, batch_size + 1, dtype=np.int64)
    x, c = (normal[key][selected] for key in ("X", "C"))
    signature = hashlib.sha256()
    for key, value in (("X", x), ("C", c), ("selected", selected)):
        signature.update(key.encode())
        signature.update(str((value.shape, value.dtype.str)).encode())
        signature.update(np.ascontiguousarray(value).tobytes())
    check()
    started = time.perf_counter()
    reference = []
    for i in range(len(x)):
        check()
        reference.append(predictor.predict_scaled(x[i:i+1], c[i:i+1]))
    reference = np.concatenate(reference)
    single_s = time.perf_counter() - started
    check()
    started = time.perf_counter()
    batched = np.concatenate([
        predictor.predict_scaled(x[i:i+batch_size], c[i:i+batch_size])
        for i in range(0, len(x), batch_size)
    ])
    batch_s = time.perf_counter() - started
    finite = (reference.shape == batched.shape == (len(x), 10, 4)
              and np.isfinite(reference).all() and np.isfinite(batched).all())
    passed = finite and np.allclose(reference, batched, rtol=rtol, atol=atol)
    result = dict(
        status="PASSED" if passed else "FAILED_STOPPED", role="normal_validation",
        windows=len(x), requested_batch_size=batch_size, tail_windows=1,
        input_sha256=signature.hexdigest(), selected_indices=selected.tolist(),
        rtol=rtol, atol=atol, finite=bool(finite),
        max_absolute_difference=float(np.max(np.abs(reference-batched))) if finite else None,
        single_window_seconds=single_s, batched_seconds=batch_s,
        timing_scope="small resident sample, includes forward/transfer; excludes file reading and replay",
        weights_changed=False, calibration_changed=False, event_truth_used=False,
        interpretation="numerical diagnostic, not a guarantee of identical threshold decisions",
    )
    write(output, result)
    if not passed:
        raise ValueError("Inference batch numerical check failed; preserve evidence and stop")
    return result
