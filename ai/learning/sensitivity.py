"""One finite all-model normal-only sensitivity gate before fresh full calibration."""

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from capstone_ai.detection_protocol import fit_calibration
from capstone_ai.training import Predictor
from execution.common import sha, write
from pipeline.pipeline import normalized_role

from .sensitivity_data import select_arrays, validate_plan
from .sensitivity_metrics import assess_sensitivity, validate_policy


def array_signature(arrays):
    digest = hashlib.sha256()
    for key in (
        "X",
        "C",
        "Y",
        "run_id",
        "segment",
        "origin_us",
        "source_indices",
        "selection_index",
    ):
        value = np.asarray(arrays[key])
        if value.dtype.kind == "O":
            raise ValueError("Object arrays are not valid frozen sensitivity inputs")
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def predict_blocks(predictor, arrays, size, progress, failure_path=None):
    """Match replay packing: restart batches at each independently selected block."""
    result, calls = [], []
    indices = arrays["selection_index"]
    for block in dict.fromkeys(indices.tolist()):
        positions = np.flatnonzero(indices == block)
        if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
            raise ValueError("Selected block is not contiguous in packed inputs")
        for start in range(0, len(positions), size):
            progress.check()
            chosen = positions[start : start + size]
            output = np.asarray(predictor.predict_scaled(arrays["X"][chosen], arrays["C"][chosen]))
            if output.shape != (len(chosen), 10, 4) or not np.isfinite(output).all():
                if failure_path is not None:
                    np.savez_compressed(
                        failure_path,
                        prediction=output,
                        source_indices=arrays["source_indices"][chosen],
                    )
                    write(
                        failure_path.with_suffix(".json"),
                        dict(
                            status="FAILED_STOPPED",
                            reason="Nonfinite/malformed prediction",
                            batch_size=len(chosen),
                            block=int(block),
                            block_offset=start,
                            observed_shape=list(output.shape),
                            evidence_sha256=sha(failure_path),
                        ),
                    )
                raise ValueError("Nonfinite/malformed sensitivity prediction")
            result.append(output.copy())
            calls.append(len(chosen))
    return np.concatenate(result), calls


def run_sensitivity_gate(
    root, out, progress, *, plan, policy, model_names, predict_device, backend_factory
):
    """No development data/truth, fitting weights, reuse, retry, or policy adjustment."""
    validate_policy(policy)
    dataset = root / "dataset"
    validate_plan(plan, dataset / "manifest.json")
    names = list(model_names)
    if (
        not names
        or any(
            not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}
            for name in names
        )
        or len(names) != len(set(names))
    ):
        raise ValueError("Unique finite model list required")
    folder = out / "sensitivity"
    if folder.exists() or folder.is_symlink():
        raise FileExistsError("Sensitivity evidence exists; no automatic retry/overwrite")
    scaler_sha = sha(dataset / "scaler.npz")
    for name in names:
        if sha(out / "models" / name / "scaler.npz") != scaler_sha:
            raise ValueError("All models must use the same train-only scaler")
    folder.mkdir()
    summary = dict(
        schema="drone-ai.all-model-sensitivity/1",
        status="RUNNING",
        models=[],
        failures=[],
        model_names=names,
        plan=plan,
        policy=policy,
        common_scaler_sha256=scaler_sha,
        common_policy_sha256=hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        weights_changed=False,
        event_truth_used=False,
        development_data_used=False,
        full_calibration_started=False,
        development_inference_started=False,
        diagnostic_calibration_reused_for_full=False,
        automatic_retry=False,
        limitation="Normal sampled-block numerical sensitivity only; not final detector accuracy or live certification",
    )
    summary_path = folder / "summary.json"
    write(summary_path, summary)
    try:
        progress.update(stage="sensitivity_inputs", done=0, total=len(names))
        with np.load(dataset / "scaler.npz", allow_pickle=False) as values:
            scaler = {key: values[key].copy() for key in values.files}
        validation = normalized_role(dataset, "normal_validation", scaler)
        samples = {
            key: select_arrays(validation, plan["selections"][key]) for key in ("scale", "check")
        }
        del validation
        gc.collect()
        calibration = normalized_role(dataset, "normal_calibration", scaler)
        samples["calibration"] = select_arrays(calibration, plan["selections"]["calibration"])
        del calibration
        gc.collect()
        summary["sample_signatures"] = {
            key: array_signature(value) for key, value in samples.items()
        }
        summary["selection_evidence"] = {
            key: value["selection_evidence"] for key, value in samples.items()
        }
        write(summary_path, summary)
        physical = np.asarray(scaler["y_scale"], dtype=float).copy()
        physical[-1] *= 4
        groups = {
            row["run_id"]: row["paired_case_id"]
            for rows in plan["selections"].values()
            for row in rows
        }
        for index, name in enumerate(names):
            progress.check()
            progress.update(
                stage="sensitivity_diagnostic_calibration",
                model=name,
                done=index,
                total=len(names),
                phase_eta_s=None,
            )
            target = folder / name
            target.mkdir()
            summary["current_model"] = name
            write(summary_path, summary)
            predictor = None
            try:
                predictor = Predictor(out / "models" / name, device=predict_device)
                metadata = predictor.metadata
                if (
                    metadata.get("name") in {"lstm", "transformer"}
                    and metadata.get("config", {}).get("batch_size") != policy["batch_size"]
                ):
                    raise ValueError("Neural batch differs from shared sensitivity policy")
                with backend_factory(predict_device):
                    ps, scale_calls = predict_blocks(
                        predictor, samples["scale"], 256, progress, target / "scale-failure.npz"
                    )
                    pc, calibration_calls = predict_blocks(
                        predictor,
                        samples["calibration"],
                        256,
                        progress,
                        target / "calibration-failure.npz",
                    )
                    vr, cr = samples["scale"]["Y"] - ps, samples["calibration"]["Y"] - pc
                    cal = samples["calibration"]
                    fitted = fit_calibration(
                        vr, cr, cal["run_id"], cal["segment"], cal["origin_us"], physical, groups
                    )
                    fitted.update(
                        purpose="diagnostic_only_never_reuse_for_full_calibration",
                        bundle_integrity_sha256=sha(out / "models" / name / "integrity.json"),
                    )
                    write(target / "diagnostic-calibration.json", fitted)
                    progress.update(
                        stage="sensitivity_compare", model=name, done=index, total=len(names)
                    )
                    ref, reference_calls = predict_blocks(
                        predictor, samples["check"], 1, progress, target / "reference-failure.npz"
                    )
                    first, batch_calls = predict_blocks(
                        predictor, samples["check"], 256, progress, target / "batch-failure.npz"
                    )
                    repeat, repeated_calls = predict_blocks(
                        predictor, samples["check"], 256, progress, target / "repeat-failure.npz"
                    )
                checked = samples["check"]
                identities = {key: checked[key] for key in ("run_id", "segment", "origin_us")}
                report, evidence = assess_sensitivity(
                    ref, first, repeat, checked["Y"], identities, fitted, physical, policy
                )
                evidence.update(diagnostic_scale_residual=vr, diagnostic_calibration_residual=cr)
                for role in ("scale", "calibration", "check"):
                    for key in (
                        "run_id",
                        "segment",
                        "origin_us",
                        "source_indices",
                        "selection_index",
                    ):
                        evidence[f"{role}_{key}"] = samples[role][key]
                path = target / "evidence.npz"
                np.savez_compressed(path, **evidence)
                report.update(
                    model=name,
                    bundle_integrity_sha256=fitted["bundle_integrity_sha256"],
                    diagnostic_calibration_file_sha256=sha(target / "diagnostic-calibration.json"),
                    evidence=dict(file=path.name, sha256=sha(path)),
                    sample_signatures=summary["sample_signatures"],
                    batch_packing=dict(
                        scale=scale_calls,
                        calibration=calibration_calls,
                        reference=reference_calls,
                        batch=batch_calls,
                        repeat=repeated_calls,
                    ),
                )
                write(target / "report.json", report)
                summary["models"].append(
                    dict(
                        model=name,
                        status=report["status"],
                        report_sha256=sha(target / "report.json"),
                    )
                )
                if report["status"] != "PASSED":
                    summary["failures"].append(name)
                summary["models_completed"] = index + 1
                write(summary_path, summary)
            finally:
                del predictor
                gc.collect()
                if predict_device == "cuda":
                    torch.cuda.empty_cache()
        summary["status"] = "FAILED_STOPPED" if summary["failures"] else "PASSED"
        write(summary_path, summary)
        progress.update(stage="sensitivity_complete", done=len(names), total=len(names))
        if summary["failures"]:
            raise ValueError(
                "All-model sensitivity failed before full calibration: "
                + ", ".join(summary["failures"])
            )
        return summary
    except BaseException as exc:
        summary.update(status="FAILED_STOPPED", error=f"{type(exc).__name__}: {exc}")
        write(summary_path, summary)
        raise
