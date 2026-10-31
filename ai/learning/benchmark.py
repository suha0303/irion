"""Finite disposable batch benchmark; requires explicit approval for any CUDA operation."""

import gc
import math
import os
import time
from pathlib import Path

import torch

from capstone_ai.training import seed_all
from execution.common import write

from .models import make_model
from .training import Batches, backend, project_lock, safe_step


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
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if device not in {"cpu", "cuda"} or steps < 1 or warmup < 0:
        raise ValueError("Invalid benchmark")
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
                model = make_model(spec).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
                order = torch.randperm(batches.count)
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
                    peak_reserved_bytes=peak,
                    free_before_bytes=free_before,
                    free_after_bytes=free_after,
                    passed=passed,
                    device=device,
                    representative_windows=batches.count,
                    input_mode="RAM resident arrays; includes gather, transfer, forward, backward, optimizer; cold disk load excluded",
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
