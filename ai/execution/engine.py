"""Reuse existing forecast models/loss; bounded batches and equivalent primal ridge."""

import gc
import hashlib
import json
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from capstone_ai.detection_protocol import fit_calibration
from capstone_ai.mapping import FEATURES, TARGETS
from capstone_ai.models import make_model
from capstone_ai.training import Predictor, flat, loss, seed_all
from pipeline.backend import backend_scope, check_model, trace_batch
from pipeline.fit import batch
from pipeline.pipeline import normalized_role, seal_development

from .common import MODELS, read, sha, verify, write


@contextmanager
def deterministic_backend(device):
    config = dict(device=device, lstm_backend="aten_cuda" if device == "cuda" else "cpu_regression")
    with (
        backend_scope(config),
        torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ),
    ):
        yield


def ridge_fit(train, validation, folder, progress):
    # (X'X + alpha I)^-1 X'Y == X'(XX' + alpha I)^-1 Y, including regularized bias.
    d = train["X"].shape[1] * train["X"].shape[2] + train["C"].shape[1] + 1
    gram, rhs = np.zeros((d, d)), np.zeros((d, 40))
    started = time.monotonic()
    for offset in range(0, len(train["X"]), 512):
        progress.check()
        x = flat({k: train[k][offset : offset + 512] for k in ("X", "C")})
        y = train["Y"][offset : offset + 512].reshape(len(x), 40)
        gram += x.T @ x
        rhs += x.T @ y
        done = min(offset + 512, len(train["X"]))
        progress.update(
            force=False,
            stage="ridge_statistics",
            model="ridge_forecast",
            done=done,
            total=len(train["X"]),
        )
    history, best, chosen, coefficients = [], float("inf"), None, None
    for alpha in (1.0, 10.0, 100.0):
        progress.check()
        progress.update(
            stage="ridge_solve", model="ridge_forecast", alpha=alpha, done=len(history), total=3
        )
        matrix = gram.copy()
        matrix.flat[:: d + 1] += alpha
        candidate = np.linalg.solve(matrix, rhs)
        total = 0.0
        for i in range(0, len(validation["X"]), 512):
            progress.check()
            x = flat({k: validation[k][i : i + 512] for k in ("X", "C")})
            error = ((x @ candidate).reshape(-1, 10, 4) - validation["Y"][i : i + 512]) ** 2
            total += (0.5 * error[..., :3].mean() + 0.5 * error[..., 3].mean()) * len(x)
        score = float(total / len(validation["X"]))
        if not np.isfinite(score) or not np.isfinite(candidate).all():
            raise ValueError("Nonfinite ridge solve")
        history.append(dict(alpha=alpha, validation_loss=score))
        if score < best:
            best, chosen, coefficients = score, alpha, candidate
    np.savez(folder / "ridge.npz", coefficients=coefficients, alpha=chosen)
    return dict(
        history=history,
        solver="primal_float64_chunked_statistics",
        selected_alpha=chosen,
        seconds=time.monotonic() - started,
        device="cpu",
    )


def probe_models(cfg, train, out, progress, device="cuda"):
    evidence = {}
    for name in ("lstm", "transformer"):
        progress.check()
        progress.update(stage="gpu_batch_admission", model=name, done=0, total=1)
        seed_all(cfg["seed"])
        with deterministic_backend(device):
            model = make_model(name, cfg["hidden"]).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5
            )
            if device == "cuda":
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                torch.cuda.reset_peak_memory_stats()
            x, c, y = batch(train, slice(0, cfg["batch_size"]), torch.device(device))
            model.train()
            value = loss(model(x, c), y)
            if not torch.isfinite(value):
                raise ValueError("Nonfinite admission loss")
            value.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise ValueError("Nonfinite admission gradient")
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_reserved()
                if free < 1.5 * peak + 1024**3:
                    raise MemoryError("Insufficient GPU headroom for measured full training batch")
                evidence[name] = dict(
                    batch_size=len(x),
                    free_before_bytes=free,
                    total_bytes=total,
                    peak_reserved_bytes=peak,
                    includes=["transfer", "forward", "backward", "optimizer"],
                    policy="discard probe weights and reset seed before main fit",
                )
            else:
                evidence[name] = dict(test_only_cpu=True)
            del model, optimizer, x, c, y, value
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    write(out / "gpu-memory-admission.json", evidence)
    return evidence


def neural_fit(name, cfg, train, val, folder, progress, device="cuda"):
    started = time.monotonic()
    seed_all(cfg["seed"])
    history, best, best_epoch = [], float("inf"), -1
    steps = (len(train["X"]) + cfg["batch_size"] - 1) // cfg["batch_size"]
    best_state = None
    with deterministic_backend(device):
        model = make_model(name, cfg["hidden"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5
        )
        if name == "lstm":
            check_model(model, torch.device(device))
        for epoch in range(cfg["epochs"]):
            progress.check()
            model.train()
            order = torch.randperm(len(train["X"]))
            total = 0.0
            epoch_started = time.monotonic()
            for index, indices in enumerate(order.split(cfg["batch_size"])):
                progress.check()
                x, c, y = batch(train, indices.numpy(), torch.device(device))
                with trace_batch(
                    folder,
                    "main-train",
                    torch.device(device),
                    True,
                    name == "lstm" and epoch == 0 and index == 0,
                ):
                    optimizer.zero_grad(set_to_none=True)
                    value = loss(model(x, c), y)
                    if not torch.isfinite(value):
                        raise ValueError("Nonfinite training loss")
                    value.backward()
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        raise ValueError("Nonfinite gradient")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total += value.item() * len(indices)
                elapsed = time.monotonic() - epoch_started
                progress.update(
                    force=False,
                    stage="training",
                    model=name,
                    epoch=epoch + 1,
                    epochs=cfg["epochs"],
                    batch=index + 1,
                    batches=steps,
                    done=epoch * steps + index + 1,
                    total=cfg["epochs"] * steps,
                    train_loss=total / min((index + 1) * cfg["batch_size"], len(train["X"])),
                    steps_per_s=(index + 1) / elapsed,
                    phase_eta_s=elapsed / (index + 1) * (steps - index - 1),
                    eta_scope="remaining train batches of this epoch; validation/other models excluded",
                    gpu_reserved_bytes=torch.cuda.memory_reserved() if device == "cuda" else None,
                )
                del x, c, y, value
            train_seconds = time.monotonic() - epoch_started
            model.eval()
            vtotal = 0.0
            vt = time.monotonic()
            with torch.no_grad():
                for i in range(0, len(val["X"]), cfg["batch_size"]):
                    progress.check()
                    x, c, y = batch(val, slice(i, i + cfg["batch_size"]), torch.device(device))
                    value = loss(model(x, c), y).item()
                    vtotal += value * len(x)
                    progress.update(
                        force=False,
                        stage="validation",
                        model=name,
                        done=min(i + len(x), len(val["X"])),
                        total=len(val["X"]),
                        phase_eta_s=None,
                    )
                    del x, c, y
            vloss = vtotal / len(val["X"])
            if not np.isfinite(vloss):
                raise ValueError("Nonfinite validation loss")
            history.append(
                dict(
                    epoch=epoch + 1,
                    train_loss=total / len(train["X"]),
                    validation_loss=vloss,
                    train_seconds=train_seconds,
                    validation_seconds=time.monotonic() - vt,
                )
            )
            improved = vloss < best
            if improved:
                best, best_epoch = vloss, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint = dict(
                schema="x500-gpu104-four-v1",
                model_name=name,
                config=cfg,
                epoch=epoch,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                best=best,
                best_epoch=best_epoch,
                best_state=best_state,
                history=history,
                torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
                automatic_resume=False,
            )
            torch.save(checkpoint, folder / "last.pt")
            if improved:
                torch.save(checkpoint, folder / "best.pt")
            write(folder / "epoch-history.json", history)
            progress.update(
                validation_loss=vloss,
                best_epoch=best_epoch + 1,
                stage="checkpoint",
                phase_eta_s=None,
            )
            print(name, history[-1], flush=True)
            if epoch - best_epoch >= cfg["early_stopping_patience"]:
                break
        model.load_state_dict(best_state)
        model.eval()
        save_file(
            {k: v.contiguous() for k, v in best_state.items()}, str(folder / "model.safetensors")
        )
        restored = make_model(name, cfg["hidden"]).to(device)
        restored.load_state_dict(load_file(str(folder / "model.safetensors")))
        restored.eval()
        with torch.no_grad():
            x, c, _ = batch(val, slice(0, 2), torch.device(device))
            difference = (model(x, c) - restored(x, c)).abs().max().item()
        if not np.isfinite(difference) or difference > 1e-6:
            raise ValueError("Export/reload prediction mismatch")
    return dict(
        history=history,
        selected_epoch=best_epoch + 1,
        save_load_max_abs=difference,
        seconds=time.monotonic() - started,
        device=device,
        precision="float32",
        automatic_resume=False,
    )


def bundles(root, out, cfg, progress, *, device="cuda"):
    dataset = root / "dataset"
    scaler = dict(np.load(dataset / "scaler.npz", allow_pickle=False))
    train, val = [
        normalized_role(dataset, r, scaler) for r in ("normal_train", "normal_validation")
    ]
    probe_models(cfg, train, out, progress, device=device)
    models = out / "models"
    models.mkdir()
    for name in MODELS:
        progress.check()
        folder = models / name
        folder.mkdir()
        progress.update(
            model=name,
            stage="model_start",
            done=0,
            total=1,
            phase_eta_s=None,
            models_completed=list(MODELS[: MODELS.index(name)]),
        )
        if name in ("lstm", "transformer"):
            result = neural_fit(name, cfg, train, val, folder, progress, device=device)
        elif name == "ridge_forecast":
            result = ridge_fit(train, val, folder, progress)
        else:
            result = dict(fit="none", device="cpu")
        for file in ("scaler.npz", "manifest.json"):
            shutil.copyfile(
                dataset / file,
                folder / ("training_manifest.json" if file == "manifest.json" else file),
            )
        write(
            folder / "model.json",
            dict(
                name=name,
                hidden=cfg["hidden"],
                task="forecast",
                target_order=TARGETS,
                feature_order=FEATURES,
                config=cfg,
            ),
        )
        shutil.copyfile(root / "preprocessing.json", folder / "preprocessing.json")
        write(
            folder / "forecast_contract.json",
            dict(
                schema="drone.forecast/3.0",
                history=100,
                horizon=10,
                step_us=100000,
                cadence_us=1000000,
                require_issued_before_first_target=True,
                teacher_forcing=False,
            ),
        )
        write(folder / "training.json", result)
        write(folder / "integrity.json", {p.name: sha(p) for p in folder.iterdir() if p.is_file()})
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    progress.update(models_completed=list(MODELS), stage="models_complete", done=4, total=4)


def preflight_models(
    root,
    out,
    progress,
    *,
    predict_device="cpu",
    backend_factory=deterministic_backend,
    model_names=MODELS,
    prediction_batch_size=256,
    diagnostic_only=False,
    numerical_policy=None,
):
    """Check every model on the same normal validation sample before any full replay.

    Diagnostic mode records observations only. A supplied policy is frozen once and
    applied unchanged to every model. Without a policy the legacy strict numerical
    check is retained. No calibration fit or development input is accessed here.
    """
    from learning import replay_validation

    names = tuple(model_names)
    if (not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name or Path(name).name != name
                   or name in {".", ".."} for name in names)):
        raise ValueError("A unique, nonempty model list is required")
    if type(prediction_batch_size) is not int or prediction_batch_size < 2:
        raise ValueError("Preflight requires a full inference batch plus tail")
    if diagnostic_only and numerical_policy is not None:
        raise ValueError("Diagnostic mode cannot choose or apply acceptance criteria")
    # Detach the policy from the caller and bind its exact common value in evidence.
    policy_json = json.dumps(numerical_policy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    policy = json.loads(policy_json)
    policy_sha = hashlib.sha256(policy_json.encode()).hexdigest() if policy is not None else None
    if policy is not None and policy.get("batch_size") != prediction_batch_size:
        raise ValueError("Common numerical policy batch differs from the requested batch")
    dataset = root / "dataset"
    scaler_sha = sha(dataset / "scaler.npz")
    for name in names:
        if sha(out / "models" / name / "scaler.npz") != scaler_sha:
            raise ValueError("All-model preflight requires the same frozen train scaler: " + name)
    folder = out / "batch-checks"
    summary_path = folder / "preflight.json"
    if (summary_path.exists()
            or any((folder / (name + suffix)).exists()
                   for name in names for suffix in (".json", ".decision.json", ".npz"))):
        raise ValueError("Preflight evidence exists; use a new attempt without overwriting it")
    records, failures = [], []
    common_input_sha, common_sample_sha = None, None
    summary = dict(
        schema="drone-ai.all-model-preflight/1", status="RUNNING",
        mode=("diagnostic_only" if diagnostic_only else
              "common_numerical_policy" if policy is not None else "legacy_strict"),
        model_names=list(names), models=records, failures=failures,
        requested_batch_size=prediction_batch_size, role="normal_validation",
        dataset_manifest_sha256=sha(dataset / "manifest.json"),
        common_scaler_sha256=scaler_sha, numerical_policy=policy,
        numerical_policy_sha256=policy_sha,
        legacy_criteria={"rtol": 1e-5, "atol": 1e-6} if policy is None and not diagnostic_only else None,
        device=predict_device, backend_factory=backend_factory.__name__
        if hasattr(backend_factory, "__name__") else type(backend_factory).__name__,
        weights_changed=False, calibration_changed=False, event_truth_used=False,
        full_calibration_started=False, development_inference_started=False,
    )
    write(summary_path, summary)
    try:
        for index, name in enumerate(names):
            progress.check()
            progress.update(stage="all_model_preflight", model=name, done=index,
                            total=len(names), phase_eta_s=None)
            report_path = folder / (name + ".json")
            decision_path = folder / (name + ".decision.json")
            predictor = Predictor(out / "models" / name, device=predict_device)
            normal = normalized_role(dataset, "normal_validation", predictor.scaler)
            failed = None
            expected_failure_path = report_path
            try:
                metadata = getattr(predictor, "metadata", {})
                if (metadata.get("name") in {"lstm", "transformer"}
                        and metadata.get("config", {}).get("batch_size") != prediction_batch_size):
                    raise ValueError("Neural predictor batch differs from common preflight batch: " + name)
                with backend_factory(predict_device):
                    if diagnostic_only or policy is not None:
                        report = replay_validation.diagnose_prediction_batch(
                            predictor, normal, report_path,
                            batch_size=prediction_batch_size, check=progress.check,
                        )
                    else:
                        report = replay_validation.check_prediction_batch(
                            predictor, normal, report_path,
                            batch_size=prediction_batch_size, check=progress.check,
                        )
                if policy is not None:
                    expected_failure_path = decision_path
                    replay_validation.assess_diagnostic(report_path, policy, output=decision_path)
            except ValueError as exc:
                # A completed numerical failure is evidence for the comparison; keep
                # checking the other models. Unexpected/data errors stop immediately.
                if (not expected_failure_path.is_file()
                        or read(expected_failure_path).get("status") != "FAILED_STOPPED"):
                    raise
                report = read(report_path)
                failed = str(exc)
            finally:
                del normal, predictor
                gc.collect()
                if predict_device == "cuda":
                    torch.cuda.empty_cache()
            sample_sha = report.get("input_sha256")
            if not isinstance(sample_sha, str) or not sample_sha:
                raise ValueError("Preflight report lacks the shared sample binding: " + name)
            if common_input_sha is None:
                common_input_sha = sample_sha
            elif sample_sha != common_input_sha:
                raise ValueError("All-model preflight inputs differ: " + name)
            if diagnostic_only or policy is not None:
                full_sample_sha = report.get("sample_sha256")
                if not isinstance(full_sample_sha, str) or not full_sample_sha:
                    raise ValueError("Preflight report lacks normal target/identity binding: " + name)
                if common_sample_sha is None:
                    common_sample_sha = full_sample_sha
                elif full_sample_sha != common_sample_sha:
                    raise ValueError("All-model preflight targets or run identities differ: " + name)
            record = dict(model=name, report=report_path.name, report_sha256=sha(report_path),
                          bundle_integrity_sha256=sha(out / "models" / name / "integrity.json"),
                          status="FAILED_STOPPED" if failed else report["status"])
            if decision_path.is_file():
                record.update(decision=decision_path.name, decision_sha256=sha(decision_path),
                              status=read(decision_path)["status"], diagnostic_status=report["status"])
            if failed:
                record["reason"] = failed
                failures.append(name)
            records.append(record)
            summary.update(models_completed=len(records), common_input_sha256=common_input_sha,
                           common_sample_sha256=common_sample_sha)
            write(summary_path, summary)
        summary["status"] = ("FAILED_STOPPED" if failures else
                             "DIAGNOSTIC_COMPLETE" if diagnostic_only else "PASSED")
        write(summary_path, summary)
        progress.update(stage="all_model_preflight", done=len(names), total=len(names))
    except Exception as exc:
        summary.update(status="FAILED_STOPPED", error=f"{type(exc).__name__}: {exc}")
        write(summary_path, summary)
        raise
    if failures:
        raise ValueError("All-model numerical preflight failed before calibration: " + ", ".join(failures))
    return summary


def calibrate(
    root,
    out,
    progress,
    *,
    predict_device="cpu",
    backend_factory=deterministic_backend,
    model_names=MODELS,
    preserve_normal_residuals=False,
    reuse_calibration=None,
    prediction_batch_size=1,
    verify_prediction_batch=False,
    numerical_policy=None,
    sensitivity_plan=None,
    sensitivity_policy=None,
    development_executor=None,
):
    model_names = tuple(model_names)
    if development_executor is not None and (prediction_batch_size != 256 or reuse_calibration):
        raise ValueError("Parallel development requires batch256 and fresh calibration")
    sensitivity = sensitivity_plan is not None or sensitivity_policy is not None
    if sensitivity:
        if (sensitivity_plan is None or sensitivity_policy is None or numerical_policy is not None
                or not verify_prediction_batch or prediction_batch_size != 256 or reuse_calibration):
            raise ValueError("Sensitivity requires one shared gate and fresh full calibration")
        from learning.sensitivity import run_sensitivity_gate

        run_sensitivity_gate(root, out, progress, plan=sensitivity_plan, policy=sensitivity_policy,
                             model_names=model_names, predict_device=predict_device,
                             backend_factory=backend_factory)
    if numerical_policy is not None and not verify_prediction_batch:
        raise ValueError("A numerical policy requires all-model preflight")
    if verify_prediction_batch and not sensitivity:
        preflight_models(
            root, out, progress, predict_device=predict_device,
            backend_factory=backend_factory, model_names=model_names,
            prediction_batch_size=prediction_batch_size, numerical_policy=numerical_policy,
        )
    dataset = root / "dataset"
    rows = read(dataset / "manifest.json")["runs"]
    groups = {r["run_id"]: r["paired_case_id"] for r in rows}
    dev = [r for r in rows if r["role"] in ("normal_dev_eval", "anomaly_dev")]
    normal_cache = {}
    common_scaler = sha(dataset / "scaler.npz") if development_executor is not None else None
    for name in model_names:
        progress.check()
        progress.update(stage="calibration", model=name, done=0, total=1, phase_eta_s=None)
        predictor = Predictor(out / "models" / name, device=predict_device)
        if development_executor is not None and sha(out / "models" / name / "scaler.npz") != common_scaler:
            raise ValueError("Shared normal cache requires the same train-only scaler for every model")
        folder = out / "calibration" / name
        if development_executor is not None:
            if "normal_validation" not in normal_cache:
                normal_cache["normal_validation"] = normalized_role(dataset, "normal_validation", predictor.scaler)
            val = normal_cache["normal_validation"]
        else:
            val = normalized_role(dataset, "normal_validation", predictor.scaler)
        reused = (reuse_calibration or {}).get(name)
        if reused:
            if set(reused["files"]) != {"calibration.json", "normal-residuals.npz", "normal-residuals.json"}:
                raise ValueError("Complete normal calibration evidence required")
            verify(folder, reused["files"])
            fitted = read(folder / "calibration.json")
            evidence = read(folder / "normal-residuals.json")
            if (fitted["bundle_integrity_sha256"] != sha(out / "models" / name / "integrity.json")
                    or evidence["dataset_manifest_sha256"] != sha(dataset / "manifest.json")
                    or evidence["truth_used"] is not False
                    or evidence["sha256"] != sha(folder / "normal-residuals.npz")
                    or evidence["calibration_sha256"] != sha(folder / "calibration.json")):
                raise ValueError("Reused normal calibration binding changed")
            progress.update(stage="calibration_reused", model=name, calibration_refitted=False)
        else:
            progress.update(stage="calibration", model=name, done=0, total=1)
            if development_executor is not None:
                if "normal_calibration" not in normal_cache:
                    normal_cache["normal_calibration"] = normalized_role(dataset, "normal_calibration", predictor.scaler)
                cal = normal_cache["normal_calibration"]
            else:
                cal = normalized_role(dataset, "normal_calibration", predictor.scaler)
            # Fixed normal-only calibration. No event truth is present in the remote input package.
            with backend_factory(predict_device):
                vr = val["Y"] - predictor.predict_scaled(val["X"], val["C"])
                cr = cal["Y"] - predictor.predict_scaled(cal["X"], cal["C"])
            physical = predictor.scaler["y_scale"].copy()
            physical[-1] *= 4
            fitted = fit_calibration(
                vr, cr, cal["run_id"], cal["segment"], cal["origin_us"], physical, groups
            )
            fitted["bundle_integrity_sha256"] = sha(out / "models" / name / "integrity.json")
            folder.mkdir(parents=True)
            write(folder / "calibration.json", fitted)
            if preserve_normal_residuals:
                # Exact normal-only inputs permit a later, separately authorized calibration
                # policy without another model forward pass or access to development truth.
                evidence = folder / "normal-residuals.npz"
                np.savez_compressed(
                    evidence,
                    validation_residual_scaled=vr,
                    validation_run_id=val["run_id"],
                    validation_segment=val["segment"],
                    validation_origin_us=val["origin_us"],
                    calibration_residual_scaled=cr,
                    calibration_run_id=cal["run_id"],
                    calibration_segment=cal["segment"],
                    calibration_origin_us=cal["origin_us"],
                    physical_target_scale=physical,
                )
                write(folder / "normal-residuals.json", dict(
                    schema="drone-ai.normal-residual-evidence/1",
                    model=name,
                    file=evidence.name,
                    sha256=sha(evidence),
                    dataset_manifest_sha256=sha(dataset / "manifest.json"),
                    bundle_integrity_sha256=fitted["bundle_integrity_sha256"],
                    roles=["normal_validation", "normal_calibration"],
                    target_order=TARGETS,
                    residual_convention="observed_minus_predicted_in_train_scaler_space",
                    truth_used=False,
                    paired_case_ids={
                        run_id: groups[run_id]
                        for run_id in sorted(set(map(str, val["run_id"])) | set(map(str, cal["run_id"])))
                    },
                    calibration_sha256=sha(folder / "calibration.json"),
                ))
            del cal, vr, cr
        del val
        gc.collect()
        if development_executor is not None:
            with backend_factory(predict_device):
                development_executor.run(dataset, out / "models" / name, fitted, folder,
                                         predictor, dev, prediction_batch_size, name)
        for index, row in enumerate(dev if development_executor is None else ()):
            progress.check()
            progress.update(
                stage="development_inference",
                model=name,
                done=index,
                total=len(dev),
                run_id=row["run_id"],
            )
            with backend_factory(predict_device):
                seal_development(
                    dataset / "canonical" / row["run_id"],
                    out / "models" / name,
                    fitted,
                    folder / row["run_id"],
                    predictor=predictor,
                    prediction_batch_size=prediction_batch_size,
                )
        progress.update(done=len(dev), total=len(dev))
        del predictor
        gc.collect()
