"""Finite disposable batch benchmark; requires explicit approval for any CUDA operation."""

import gc
import math
import os
import time
from pathlib import Path

import torch

from capstone_ai.training import loss, seed_all
from execution.common import write

from .models import make_model
from .training import Batches, backend, project_lock, safe_step


def _validation_timing(model, arrays, batch_size, device, progress):
    """One separately timed full validation pass through the production loader."""
    prepare_started = time.monotonic()
    batches = Batches(arrays, batch_size, device)
    prepare_seconds = time.monotonic() - prepare_started
    model.eval()
    with torch.no_grad():
        x, c, y = batches.get(torch.arange(min(batch_size, batches.count)))
        if not bool(torch.isfinite(loss(model(x, c), y))):
            raise ValueError("Nonfinite validation warmup")
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        total = torch.zeros((), device=device, dtype=torch.float64)
        finite = torch.ones((), device=device, dtype=torch.bool)
        for offset in range(0, batches.count, batch_size):
            if progress is not None:
                progress.check()
            x, c, y = batches.get(torch.arange(offset, min(offset + batch_size, batches.count)))
            value = loss(model(x, c), y)
            total.add_(value.to(torch.float64) * len(x))
            finite.logical_and_(torch.isfinite(value))
            if progress is not None:
                progress.update(
                    force=False,
                    benchmark_validation_done=min(offset + batch_size, batches.count),
                    benchmark_validation_total=batches.count,
                )
        if not bool(finite):
            raise ValueError("Nonfinite validation loss")
        mean_loss = float(total) / batches.count
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
    return dict(
        windows=batches.count,
        steps=math.ceil(batches.count / batch_size),
        seconds=elapsed,
        windows_per_s=batches.count / elapsed,
        prepare_seconds=prepare_seconds,
        warmup_steps=1,
        loss=mean_loss,
        full_validation=True,
        disposable_weights=True,
        detection_performance_measured=False,
        includes=["loader gather/read", "GPU transfer when CUDA", "forward", "loss aggregation"],
        excludes=["backward", "optimizer", "calibration", "development evaluation"],
    )


def estimate_fit_budget(
    rows, *, train_windows, validation_windows, epochs, elapsed_seconds, wall_hours
):
    """Measured-time projection, with explicit assumptions, before fresh main fits.

    A successful admission is not a prediction guarantee: the enforced wall budget
    still stops the job. Early stopping is never assumed to make an over-budget run fit.
    """
    if not rows or any(
        type(v) is not int or v < 1 for v in (train_windows, validation_windows, epochs)
    ):
        raise ValueError("Nonempty measurements and positive window/epoch counts required")
    if (
        not all(math.isfinite(v) and v >= 0 for v in (elapsed_seconds, wall_hours))
        or not wall_hours
    ):
        raise ValueError("Finite elapsed time and positive wall budget required")
    estimates = []
    for row in rows:
        validation = row.get("validation") or {}
        if (
            row.get("passed") is not True
            or row.get("device") != "cuda"
            or row.get("steps", 0) < 500
            or row.get("warmup_steps", 0) < 50
            or row.get("representative_windows") != train_windows
            or validation.get("windows") != validation_windows
            or validation.get("full_validation") is not True
            or type(row.get("batch")) is not int
            or row["batch"] < 1
        ):
            raise ValueError("Full-data CUDA training/validation timing evidence required")
        if any(
            not math.isfinite(v) or v <= 0 for v in (row["step_seconds"], validation["seconds"])
        ):
            raise ValueError("Finite positive measured timings required")
        train_seconds = epochs * math.ceil(train_windows / row["batch"]) * row["step_seconds"]
        validation_seconds = epochs * validation["seconds"]
        estimates.append(
            dict(
                model_spec=row.get("model_spec"),
                seed=row.get("seed"),
                batch=row["batch"],
                epochs_max=epochs,
                train_seconds=train_seconds,
                validation_seconds=validation_seconds,
                measured_epoch_validation_seconds=validation["seconds"],
            )
        )
    projected = sum(v["train_seconds"] + v["validation_seconds"] for v in estimates)
    factor, fixed_reserve = 1.25, 3600
    total = elapsed_seconds + projected * factor + fixed_reserve
    return dict(
        passed=total <= wall_hours * 3600,
        elapsed_preflight_seconds=elapsed_seconds,
        models=estimates,
        projected_neural_fit_and_validation_seconds=projected,
        projection_safety_factor=factor,
        baseline_checkpoint_and_io_reserve_seconds=fixed_reserve,
        projected_total_with_reserves_seconds=total,
        approved_hard_wall_seconds=wall_hours * 3600,
        assumptions=[
            "Full maximum epoch count; no early-stopping discount",
            "Warm-cache actual loader measurements extrapolated to every training batch",
            "25% timing margin plus one-hour unmeasured baseline/checkpoint/I/O reserve",
            "Already elapsed normalization, memory/resume and timing gates included once",
        ],
        excluded_stages=["calibration", "development inference", "development evaluation"],
        automatic_budget_extension=False,
    )


def _measure(
    spec,
    arrays,
    candidates,
    folder,
    *,
    device="cpu",
    warmup=50,
    steps=500,
    gpu_authorized=False,
    progress=None,
    runtime_stage="benchmark",
    seed=17,
    validation=None,
    learning_rate=0.001,
    optimization=None,
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if device not in {"cpu", "cuda"} or steps < 1 or warmup < 0:
        raise ValueError("Invalid benchmark")
    if (spec.revision == "interlayer_norm_no_dropout") != (optimization is not None):
        raise ValueError("Study benchmark requires the actual optimizer policy")
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(type(b) is not int or b < 1 for b in candidates)
    ):
        raise ValueError("Explicit finite unique batch list required")
    if any(b > len(arrays["X"]) for b in candidates):
        raise ValueError("Cannot claim requested batch larger than representative dataset")
    if device == "cuda":
        if not gpu_authorized or progress is None:
            raise ValueError("Separate explicit GPU benchmark approval/budget required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Fixed CUDA workspace policy required")
        from pipeline.runtime import check

        if runtime_stage not in {"benchmark", "fit"}:
            raise ValueError("Unknown benchmark allocation purpose")
        check(Path(__file__).resolve().parents[1], folder, runtime_stage, job_prefix="drone-ai-")
    results = []
    try:
        for batch_size in candidates:
            if progress is not None:
                progress.check()
            seed_all(seed)
            load_start = time.monotonic()
            batches = Batches(arrays, batch_size, device)
            prepare_seconds = time.monotonic() - load_start
            free_before = 0
            if device == "cuda":
                torch.cuda.empty_cache()
                free_before, _ = torch.cuda.mem_get_info()
            with backend(device):
                model_setup_started = time.monotonic()
                model = make_model(spec).to(device)
                if optimization is not None:
                    from .optimization import WarmupCosine, make_optimizer

                    optimizer = make_optimizer(model.parameters(), learning_rate, optimization)
                    schedule = WarmupCosine(warmup + steps, learning_rate, optimization)
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
                order = torch.randperm(batches.count)
                if device == "cuda":
                    torch.cuda.synchronize()
                model_setup_seconds = time.monotonic() - model_setup_started
                cursor = 0
                model.train()
                measured_samples = 0
                for index in range(warmup + steps):
                    if progress is not None:
                        progress.check()
                    if index == warmup:
                        if device == "cuda":
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                        started = time.monotonic()
                    if cursor == batches.count:
                        order = torch.randperm(batches.count)
                        cursor = 0
                    ids = order[cursor : min(cursor + batch_size, batches.count)]
                    cursor += len(ids)
                    x, c, y = batches.get(ids)
                    if optimization is not None:
                        rate = schedule(index)
                        for group in optimizer.param_groups:
                            group["lr"] = rate
                    safe_step(model, optimizer, x, c, y)
                    if progress is not None:
                        progress.update(
                            force=False, benchmark_step=index + 1, benchmark_steps=warmup + steps
                        )
                    if index >= warmup:
                        measured_samples += len(ids)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                validation_result = (
                    _validation_timing(model, validation, batch_size, device, progress)
                    if validation is not None
                    else None
                )
                peak = torch.cuda.max_memory_reserved() if device == "cuda" else 0
                free_after = torch.cuda.mem_get_info()[0] if device == "cuda" else 0
                passed = (
                    (peak <= 0.85 * free_before and free_after >= 4 * 1024**3)
                    if device == "cuda"
                    else True
                )
                row = dict(
                    batch=batch_size,
                    steps=steps,
                    warmup_steps=warmup,
                    actual_samples=measured_samples,
                    windows_per_s=measured_samples / elapsed,
                    step_seconds=elapsed / steps,
                    measured_seconds=elapsed,
                    prepare_seconds=prepare_seconds,
                    model_setup_seconds=model_setup_seconds,
                    peak_reserved_bytes=peak,
                    free_before_bytes=free_before,
                    free_after_bytes=free_after,
                    passed=passed,
                    device=device,
                    representative_windows=batches.count,
                    input_mode="Actual Batches loader over supplied arrays/mmap; finite-input scan warms cache; cold disk load excluded",
                    includes=[
                        "loader gather/read",
                        "GPU transfer when CUDA",
                        "forward",
                        "backward",
                        "gradient checks/clipping",
                        (optimization["optimizer"] if optimization else "AdamW") + " optimizer",
                    ],
                    excludes=[
                        "archive restore",
                        "normalization",
                        "initial finite-input scan",
                        "checkpoint saving",
                        "calibration",
                        "development evaluation",
                    ],
                    model_spec=spec.to_dict(),
                    precision="float32",
                    amp=False,
                    validation=validation_result,
                    learning_rate=learning_rate,
                    optimization=optimization,
                    discarded_weights=True,
                    seed=seed,
                    production_training=False,
                )
                if not math.isfinite(row["windows_per_s"]):
                    raise ValueError("Invalid timing")
                results.append(row)
                write(folder / "measurements.json", results)
                if not passed:
                    raise MemoryError("Memory admission failed; no smaller-batch retry")
                del model, optimizer, x, c, y
            del batches
            gc.collect()
        write(
            folder / "result.json",
            dict(
                status="BENCHMARK_STOPPED",
                next_training_authorized=False,
                device=device,
                measured_candidates=len(results),
            ),
        )
        return results
    except BaseException as exc:
        write(
            folder / "failure.json",
            dict(status="FAILED_STOPPED", reason=str(exc), automatic_retry=False),
        )
        raise


def measure(spec, arrays, candidates, folder, **kwargs):
    if kwargs.get("device") == "cuda" and not kwargs.get("gpu_authorized"):
        raise ValueError("Separate explicit GPU benchmark approval required")
    with project_lock(kwargs.get("device", "cpu")):
        return _measure(spec, arrays, candidates, folder, **kwargs)
