"""Versioned R5/R6: exact-config gates, fresh fit, normal calibration, sealed inference."""

import argparse
import gc
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from capstone_ai.mapping import FEATURES, TARGETS
from capstone_ai.training import Predictor
from execution.common import MODELS, Progress, memory_snapshot, read, sha, verify, write
from execution.engine import calibrate, ridge_fit
from pipeline.pipeline import normalized_role
from pipeline.runtime import check

from . import comparison
from .benchmark import estimate_fit_budget, measure
from .diagnostic import same, stop_signal
from .models import ModelSpec, make_model
from .telemetry import Telemetry
from .training import backend, project_lock, run_fit


def config_check(cfg):
    if comparison.is_comparison(cfg):
        comparison.config_check(cfg)
        return
    fixed = dict(
        revision="ROUND_001/R5_H128_B256_S17",
        models=list(MODELS),
        seed=17,
        hidden=128,
        layers=2,
        heads=4,
        batch_size=256,
        epochs=60,
        early_stopping_patience=10,
        learning_rate=0.001,
        precision="float32",
        amp=False,
        retries=0,
        final_test_used=False,
        gate_epochs=4,
        gate_windows=512,
        production_training_authorized=True,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("Approved R5 configuration mismatch")


def paths(root):
    root = Path(root).resolve()
    cfg = read(root / "input/config.json")
    config_check(cfg)
    return root, cfg, Path(cfg["upstream"]), root / "compute"


def neural_ids(cfg):
    return (
        [v["id"] for v in cfg["variants"]]
        if comparison.is_comparison(cfg)
        else ["lstm", "transformer"]
    )


def model_ids(cfg):
    return cfg.get("models", list(MODELS))


def spec_for(cfg, name):
    if comparison.is_comparison(cfg):
        entry = next(v for v in cfg["variants"] if v["id"] == name)
        if cfg["model_revision"] != "explicit_dropout_v1":
            from .study import spec_for as study_spec

            return study_spec(entry)
        return ModelSpec(entry["name"], entry["hidden"], entry["layers"], entry["heads"])
    return ModelSpec(name, cfg["hidden"], cfg["layers"], cfg["heads"])


def fit_settings(root, cfg, name):
    if cfg.get("model_revision", "explicit_dropout_v1") == "explicit_dropout_v1":
        return dict(learning_rate=cfg["learning_rate"])
    from .study import selected_settings

    return selected_settings(cfg, name, root / "compute")


def gate_child(root, name, phase):
    root, cfg, upstream, out = paths(root)
    if name not in neural_ids(cfg) or phase not in {"full", "partial", "resume"}:
        raise ValueError("Unknown gate child")
    with np.load(out / "gates/inputs.npz", allow_pickle=False) as archive:
        train = {k: archive["train_" + k] for k in ("X", "C", "Y")}
        val = {k: archive["val_" + k] for k in ("X", "C", "Y")}
    run_fit(
        spec_for(cfg, name),
        train,
        val,
        out / "gates" / f"{name}-{phase}",
        binding=cfg["binding"],
        device="cuda",
        gpu_authorized=True,
        seed=cfg.get("seed", 17),
        batch_size=256,
        epochs=4,
        patience=10,
        **fit_settings(root, cfg, name),
        stop_after_epoch=2 if phase == "partial" else None,
        resume=out / "gates" / f"{name}-partial/last.pt" if phase == "resume" else None,
        progress=Progress(out / "gates" / f"{name}-{phase}-progress.json", hours=0.5),
    )


def gates(root, cfg, out, train, val, progress):
    training_only = comparison.execution_scope(cfg) == "training_only"
    folder = out / "gates"
    folder.mkdir()
    if cfg.get("model_revision") == "interlayer_norm_no_dropout":
        from .study import select_rates

        select_rates(cfg, out, train, val, progress)
    n = cfg["gate_windows"]
    np.savez(
        folder / "inputs.npz",
        **{"train_" + k: train[k][:n] for k in ("X", "C", "Y")},
        **{"val_" + k: val[k][:n] for k in ("X", "C", "Y")},
    )
    proofs = {}
    for name in neural_ids(cfg):
        progress.update(
            stage="batch256_memory_gate", model=name, done=0, total=len(neural_ids(cfg))
        )
        proofs[name] = dict(
            memory=measure(
                spec_for(cfg, name),
                train,
                [256],
                folder / (name + "-memory"),
                device="cuda",
                warmup=50 if training_only else 10,
                steps=500 if training_only else 20,
                seed=cfg.get("seed", 17),
                gpu_authorized=True,
                runtime_stage="fit",
                progress=progress,
                **({"validation": val} if training_only else {}),
                **(fit_settings(root, cfg, name)
                   if cfg.get("model_revision") == "interlayer_norm_no_dropout" else {}),
            )[0]
        )
        for phase in ("full", "partial", "resume"):
            progress.check()
            progress.update(stage="batch256_resume_gate", model=name, gate_phase=phase)
            with (folder / f"{name}-{phase}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "learning.lifecycle",
                        "gate-child",
                        "--root",
                        str(root),
                        "--model",
                        name,
                        "--phase",
                        phase,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
        full = torch.load(folder / f"{name}-full/last.pt", map_location="cpu", weights_only=False)
        resumed = torch.load(
            folder / f"{name}-resume/last.pt", map_location="cpu", weights_only=False
        )
        checks = {
            k: same(full[k], resumed[k])
            for k in ("model", "optimizer", "rng", "best", "best_epoch", "best_state")
        }
        checks["loss_history"] = all(
            [x[k] for x in full["history"]] == [x[k] for x in resumed["history"]]
            for k in ("train_loss", "validation_loss")
        )
        checks["export"] = sha(folder / f"{name}-full/model.safetensors") == sha(
            folder / f"{name}-resume/model.safetensors"
        )
        proofs[name]["resume"] = checks
        write(folder / "gate-results.json", proofs)
        if not all(checks.values()):
            raise ValueError(name + " exact-config resume gate failed; no fit or retry")
    if training_only:
        budget = estimate_fit_budget(
            [proofs[name]["memory"] for name in neural_ids(cfg)],
            train_windows=len(train["X"]),
            validation_windows=len(val["X"]),
            epochs=cfg["epochs"],
            elapsed_seconds=time.time() - progress.started,
            wall_hours=cfg["wall_hours"] - 0.1,
        )
        budget["measurement_context"] = dict(
            config_sha256=sha(root / "input/config.json"),
            source_dataset_sha256=cfg.get("source_dataset_sha256"),
            preprocessing_sha256=cfg.get("upstream_files", {}).get("preprocessing.json"),
            model_code_sha256=cfg.get("binding", {}).get("sources"),
            scaler_sha256=cfg.get("binding", {}).get("scaler"),
            training_stride_us=cfg["training_stride_us"],
            history=cfg["history"],
            horizon=cfg["future"],
            sample_step_us=cfg["step_us"],
            seed=cfg["seed"],
            batch_size=cfg["batch_size"],
            precision=cfg["precision"],
            amp=cfg["amp"],
            model_revision=cfg["model_revision"],
            models=cfg["variants"],
        )
        write(out / "training-budget.json", budget)
        progress.update(stage="training_budget_admission", budget_passed=budget["passed"])
        if not budget["passed"]:
            raise TimeoutError(
                "Measured maximum-epoch fit estimate exceeds finite wall budget; no production fit or retry"
            )
    write(
        folder / "passed.json",
        dict(
            status="BATCH256_MEMORY_AND_RESUME_PASSED",
            proofs=proofs,
            discarded_gate_weights=True,
            config=dict(
                variants=cfg.get("variants"),
                hidden=cfg.get("hidden"),
                layers=cfg.get("layers"),
                batch=256,
                seed=cfg.get("seed", 17),
            ),
            scope=(
                "50 warmup and500 full optimizer steps on actual training loader, full validation timing; "
                "512 normal train/validation windows and four resume-gate epochs"
                if training_only
                else "512 actual normal train and validation windows; four gate epochs"
            ),
        ),
    )


def make_bundles(cfg, upstream, out, train, val, progress, *, device="cuda"):
    models = out / "models"
    models.mkdir()
    for index, name in enumerate(model_ids(cfg)):
        started = time.monotonic()
        folder = models / name
        progress.update(
            stage="model_start",
            model=name,
            models_completed=model_ids(cfg)[:index],
            done=0,
            total=1,
            gate_phase=None,
        )
        if name in neural_ids(cfg):
            selection = None
            if cfg.get("model_revision") == "interlayer_norm_no_dropout":
                from .study import validated_selection

                selection = validated_selection(cfg, name, out)
                settings = {key: selection[key] for key in ("learning_rate", "optimization")}
            else:
                settings = fit_settings(out.parent, cfg, name)
            history = run_fit(
                spec_for(cfg, name),
                train,
                val,
                folder,
                binding=cfg["binding"],
                device=device,
                gpu_authorized=device == "cuda",
                seed=cfg.get("seed", 17),
                batch_size=cfg["batch_size"],
                epochs=cfg["epochs"],
                patience=cfg["early_stopping_patience"],
                **settings,
                progress=progress,
                model_id=name,
            )
            result = dict(
                history=history,
                selected_epoch=min(history, key=lambda x: x["validation_loss"])["epoch"],
                seconds=time.monotonic() - started,
                device=device,
                precision="float32",
                initial_weights=f"fresh seed{cfg['seed']}; no gate or previous-round weights",
                automatic_resume=False,
            )
            if selection is not None:
                initial = read(folder / "initialization.json")["state_sha256"]
                if initial != selection["initial_state_sha256"]:
                    raise ValueError("Main fit did not start from the same frozen initial weights")
                result.update(**settings, initial_state_sha256=initial)
        else:
            folder.mkdir()
            result = (
                ridge_fit(train, val, folder, progress)
                if name == "ridge_forecast"
                else dict(fit="none", device="cpu")
            )
        for file in ("scaler.npz", "manifest.json"):
            shutil.copyfile(
                upstream / "dataset" / file,
                folder / ("training_manifest.json" if file == "manifest.json" else file),
            )
        metadata = dict(
            name=spec_for(cfg, name).name if name in neural_ids(cfg) else name,
            hidden=spec_for(cfg, name).hidden if name in neural_ids(cfg) else cfg.get("hidden", 0),
            model_id=name,
            task="forecast",
            target_order=TARGETS,
            feature_order=FEATURES,
            config=cfg,
        )
        if name in neural_ids(cfg):
            metadata["model_revision"] = spec_for(cfg, name).revision
        write(folder / "model.json", metadata)
        shutil.copyfile(upstream / "preprocessing.json", folder / "preprocessing.json")
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
        if name in neural_ids(cfg):
            predictor = Predictor(folder, device=device)
            restored = make_model(spec_for(cfg, name)).to(device)
            best = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            restored.load_state_dict(best["best_state"])
            restored.eval()
            with backend(device), torch.no_grad():
                expected = (
                    restored(
                        torch.as_tensor(val["X"][:2], dtype=torch.float32, device=device),
                        torch.as_tensor(val["C"][:2], dtype=torch.float32, device=device),
                    )
                    .cpu()
                    .numpy()
                )
                actual = predictor.predict_scaled(val["X"][:2], val["C"][:2])
            difference = float(np.max(np.abs(expected - actual)))
            if not np.isfinite(difference) or difference > 1e-6:
                raise ValueError("Best-checkpoint bundle reload mismatch")
            write(out / (name + "-export-check.json"), dict(max_abs=difference, passed=True))
            del predictor, restored, best
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    progress.update(
        models_completed=model_ids(cfg),
        stage="models_complete",
        done=len(model_ids(cfg)),
        total=len(model_ids(cfg)),
    )


def fit_worker(root):
    root, cfg, upstream, out = paths(root)
    out.mkdir(exist_ok=False)
    progress = Progress(out / "progress.json", hours=cfg["wall_hours"] - 0.1)
    telemetry = None
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        verify(root / "input", read(root / "input/inventory.json"))
        verify(upstream, cfg["upstream_files"])
        progress.update(stage="resource_admission", round=cfg["revision"])
        environment = check(root / "input/ai", out, "fit", job_prefix="drone-ai-")
        required = (
            cfg["memory"]["estimated_host_peak_bytes"] + 1024**3
            if comparison.is_comparison(cfg)
            else 9 * 1024**3
        )
        snapshot = memory_snapshot()
        admitted = snapshot["available_bytes"] >= required
        write(
            out / "host-memory-admission.json",
            dict(
                available_bytes=snapshot["available_bytes"],
                required_bytes=required,
                passed=admitted,
                snapshot=snapshot,
                environment=environment,
            ),
        )
        if not admitted:
            raise MemoryError(
                f"Host memory admission failed: need{required} bytes including reserve; "
                f"available{snapshot['available_bytes']} bytes; see host-memory-admission.json"
            )
        telemetry = Telemetry(out)
        telemetry.start()
        scaler = dict(np.load(upstream / "dataset/scaler.npz", allow_pickle=False))
        train, val = [
            (
                comparison.normalized_mmap(
                    upstream / "dataset", role, scaler, out / "normalized" / role, progress
                )
                if comparison.is_comparison(cfg)
                else normalized_role(upstream / "dataset", role, scaler)
            )
            for role in ("normal_train", "normal_validation")
        ]
        gates(root, cfg, out, train, val, progress)
        make_bundles(cfg, upstream, out, train, val, progress)
        del train, val, scaler
        gc.collect()
        if comparison.is_comparison(cfg):
            cache = out / "normalized"
            entries = {
                str(p.relative_to(cache)): p.stat().st_size for p in cache.rglob("*") if p.is_file()
            }
            shutil.rmtree(cache)  # Only this job's disposable normalized mmap cache.
            write(
                out / "normalized-cache-cleanup.json", dict(removed=entries, models_exported=True)
            )
        torch.cuda.empty_cache()
        training_only = comparison.execution_scope(cfg) == "training_only"
        if not training_only:
            with project_lock("cuda"):
                calibrate(
                    upstream,
                    out,
                    progress,
                    predict_device="cuda",
                    backend_factory=backend,
                    model_names=model_ids(cfg),
                )
        telemetry.stop()
        telemetry = None
        progress.update(stage="seal_outputs")
        verify(root / "input", read(root / "input/inventory.json"))
        verify(upstream, cfg["upstream_files"])
        files = {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "progress.json"
        }
        write(
            out / "result-seal.json",
            dict(
                files=files,
                config_sha256=sha(root / "input/config.json"),
                models=model_ids(cfg),
                seed=cfg.get("seed", 17),
                final_test_used=False,
                status="TRAINING_COMPLETE"
                if training_only
                else "TRAIN_CALIBRATION_INFERENCE_COMPLETE",
                execution_scope=comparison.execution_scope(cfg),
                calibration_started=not training_only,
                inference_started=not training_only,
                evaluation_started=False,
                truth_join=False,
                next_round_authorized=False,
            ),
        )
        progress.update(
            status="COMPUTE_COMPLETE_PENDING_RELEASE",
            stage="stopped",
            done=1,
            total=1,
            phase_eta_s=None,
        )
    except BaseException as exc:
        progress.update(
            status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}", automatic_retry=False
        )
        raise
    finally:
        if telemetry is not None:
            telemetry.stop()


def evaluate_worker(root):
    root, cfg, upstream, out = paths(root)
    if comparison.execution_scope(cfg) == "training_only":
        raise ValueError("Evaluation is not authorized by this training-only attempt")
    progress = Progress(
        root / "evaluation-progress.json", hours=cfg.get("evaluation_hours", 2) - 0.1
    )
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        verify(root / "input", read(root / "input/inventory.json"))
        check(root / "input/ai", root, "evaluate", job_prefix="drone-ai-")
        from execution.evaluate import evaluate

        if comparison.is_comparison(cfg):
            result = read(out / "result-seal.json")
            if result.get("seed") != cfg["seed"] or result.get("config_sha256") != sha(
                root / "input/config.json"
            ):
                raise ValueError("Comparison output config/seed binding mismatch before truth join")
        inventory = read(root / "truth/index.json")
        if sha(root / "truth/index.json") != cfg["truth_index_sha256"]:
            raise ValueError("Coordinator truth inventory changed")

        def truth_provider(row, target, budget):
            budget.check()
            if row["role"] not in {"normal_dev_eval", "anomaly_dev"}:
                raise ValueError("Development truth only")
            expected = row["archive"]["members"]["evaluation/events.jsonl"]
            proof = inventory[row["run_id"]]
            source = root / "truth" / (row["run_id"] + ".jsonl")
            if (
                proof["sha256"] != expected["sha256"]
                or source.stat().st_size != expected["bytes"]
                or sha(source) != expected["sha256"]
            ):
                raise ValueError("Development truth binding mismatch")
            shutil.copyfile(source, target)

        eval_cfg = {**cfg, "dataset": str(upstream / "dataset")}
        with project_lock("cuda"):
            evaluate(
                eval_cfg,
                out,
                root / "evaluation",
                progress,
                truth_provider=truth_provider,
                model_names=model_ids(cfg),
                development_counts=cfg.get("development_counts"),
            )
        progress.update(status="EVALUATION_COMPLETE_PENDING_RELEASE", stage="stopped")
    except BaseException as exc:
        progress.update(
            status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}", automatic_retry=False
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fit", "evaluate", "gate-child"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--phase")
    args = parser.parse_args()
    if args.action == "fit":
        fit_worker(args.root)
    elif args.action == "evaluate":
        evaluate_worker(args.root)
    else:
        gate_child(args.root, args.model, args.phase)


if __name__ == "__main__":
    main()
