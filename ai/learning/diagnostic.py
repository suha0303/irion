"""One finite DIS01 diagnostic: fresh-process resume gate, then real-data batch timing."""

import argparse
import gc
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from execution.common import Progress, read, sha, verify, write
from pipeline.pipeline import normalized_role
from pipeline.runtime import check

from .benchmark import measure
from .models import ModelSpec
from .profiles import PROFILES, select_batch, specs
from .training import run_fit


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def candidate_batches(profile, pilot):
    """Conservative admission, not a claim of measured maximum batch size."""
    free, peak = pilot["free_before_bytes"], pilot["peak_reserved_bytes"]
    if free <= 4 * 1024**3 or peak <= 0:
        raise MemoryError("No valid measured memory headroom")
    grid = [128, 256, 512, 1024, 2048, 4096]
    admitted = [b for b in grid if peak * b / 32 * 1.5 <= min(0.85 * free, free - 4 * 1024**3)]
    preferred = [b for b in PROFILES[profile]["batches"] if b in admitted]
    selected = sorted(set(([32] if profile == "reference" else []) + preferred[:1] + admitted[-1:]))
    if not selected:
        raise MemoryError("No candidate admitted from full-step peak; stop")
    return selected, [b for b in grid if b not in admitted]


def stop_signal(signum, frame):
    raise RuntimeError(f"Received signal {signum}; preserve diagnostic evidence and stop")


def gate_child(root, name, phase):
    root = Path(root)
    settings = read(root / "run.json")
    data = dict(np.load(root / "output/gate-input.npz"))
    progress = Progress(root / "output" / f"gate-{name}-{phase}.json", hours=0.15)
    run_fit(
        ModelSpec(name, 32),
        data,
        data,
        root / "output" / f"gate-{name}-{phase}",
        binding=settings["binding"],
        device="cuda",
        gpu_authorized=True,
        runtime_stage="benchmark",
        batch_size=32,
        epochs=4,
        patience=5,
        stop_after_epoch=2 if phase == "partial" else None,
        resume=root / "output" / f"gate-{name}-partial/last.pt" if phase == "resume" else None,
        progress=progress,
    )


def resume_gate(root, arrays, progress):
    gate = {k: arrays[k][:64] for k in ("X", "C", "Y")}
    np.savez(root / "output/gate-input.npz", **gate)
    result = {}
    for name in ["lstm", "transformer"]:
        for phase in ["full", "partial", "resume"]:
            progress.check()
            progress.update(stage="resume_gate", model=name, phase=phase)
            with (root / "output" / f"gate-{name}-{phase}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "learning.diagnostic",
                        "child",
                        "--root",
                        str(root),
                        "--model",
                        name,
                        "--phase",
                        phase,
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=480,
                )
        full = torch.load(
            root / "output" / f"gate-{name}-full/last.pt", map_location="cpu", weights_only=False
        )
        resumed = torch.load(
            root / "output" / f"gate-{name}-resume/last.pt", map_location="cpu", weights_only=False
        )
        checks = {
            key: same(full[key], resumed[key])
            for key in ["model", "optimizer", "rng", "best", "best_epoch", "best_state"]
        }
        checks["loss_history"] = all(
            [x[k] for x in full["history"]] == [x[k] for x in resumed["history"]]
            for k in ("train_loss", "validation_loss")
        )
        checks["export"] = sha(root / "output" / f"gate-{name}-full/model.safetensors") == sha(
            root / "output" / f"gate-{name}-resume/model.safetensors"
        )
        result[name] = checks
        write(root / "output/resume-gate.json", result)
        if not all(checks.values()):
            raise ValueError(f"{name} CUDA resume mismatch; no benchmark or main training")
    return result


def main_run(root):
    root = Path(root).resolve()
    cfg = read(root / "run.json")
    progress = Progress(root / "output/progress.json", hours=cfg["diagnostic_hours"])
    (root / "output").mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        if (
            cfg.get("purpose") != "disposable_diagnostic"
            or cfg.get("production_training_authorized") is not False
        ):
            raise ValueError("Diagnostic-only scope required")
        verify(root, read(root / "inventory.json"))
        upstream = Path(cfg["upstream"])
        if sha(upstream / "inventory.json") != cfg["upstream_inventory_sha256"]:
            raise ValueError("Original frozen input binding changed")
        verify(upstream, cfg["upstream_used_files"])
        check(root / "ai", root / "output", "benchmark", job_prefix="drone-ai-")
        progress.update(stage="load_normal_train")
        started = time.monotonic()
        scaler = dict(np.load(upstream / "dataset/scaler.npz"))
        train = normalized_role(upstream / "dataset", "normal_train", scaler)
        write(
            root / "output/loading.json",
            dict(
                seconds=time.monotonic() - started,
                windows=len(train["X"]),
                mode="normal_train RAM resident; no truth or final test",
            ),
        )
        resume_gate(root, train, progress)
        rows = []
        planned = len(PROFILES) * 2
        for profile in PROFILES:
            for spec in specs(profile):
                label = profile + "-" + spec.name
                progress.update(stage="memory_probe", model=label, done=len(rows), total=planned)
                pilot = measure(
                    spec,
                    train,
                    [32],
                    root / "output" / f"{label}-pilot",
                    device="cuda",
                    warmup=1,
                    steps=1,
                    gpu_authorized=True,
                    progress=progress,
                )[0]
                # Use measured full-step peak to reject risky larger candidates BEFORE executing them.
                selected, rejected = candidate_batches(profile, pilot)
                write(
                    root / "output" / f"{label}-admission.json",
                    dict(
                        pilot=pilot,
                        selected=selected,
                        rejected=rejected,
                        extrapolation_safety_factor=1.5,
                        not_a_measured_maximum=True,
                    ),
                )
                measured = []
                for b in selected:
                    progress.update(
                        stage="batch_benchmark",
                        model=label,
                        batch_size=b,
                        done=len(rows),
                        total=planned,
                    )
                    measured += measure(
                        spec,
                        train,
                        [b],
                        root / "output" / f"{label}-batch{b}",
                        device="cuda",
                        warmup=50,
                        steps=500,
                        gpu_authorized=True,
                        progress=progress,
                    )
                chosen = select_batch(measured)
                row = dict(
                    profile=profile,
                    model=spec.to_dict(),
                    measurements=measured,
                    selected=chosen,
                    estimated20epoch_train_minutes=20
                    * len(train["X"])
                    / chosen["windows_per_s"]
                    / 60,
                    excluded_time=[
                        "validation",
                        "calibration",
                        "development inference",
                        "download",
                        "evaluation",
                    ],
                    detection_performance="not measured by a speed diagnostic",
                )
                rows.append(row)
                write(root / "output/comparison.json", rows)
                (root / "output/report.txt").write_text(
                    "\n".join(
                        f"{r['profile']} {r['model']['name']}: batch{r['selected']['batch']}, "
                        f"{r['selected']['windows_per_s']:.1f} windows/s, "
                        f"peak{r['selected']['peak_reserved_bytes'] / 1024**3:.2f}GiB, "
                        f"20epoch TRAIN ONLY {r['estimated20epoch_train_minutes']:.1f}min"
                        for r in rows
                    )
                    + "\n",
                )
                gc.collect()
                torch.cuda.empty_cache()
        verify(root, read(root / "inventory.json"))
        progress.update(
            status="DIAGNOSTIC_COMPLETE_PENDING_RELEASE",
            stage="stopped",
            done=planned,
            total=planned,
            production_training=False,
            next_training_authorized=False,
        )
    except BaseException as exc:
        progress.update(
            status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}", automatic_retry=False
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "child"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--phase")
    args = parser.parse_args()
    if args.action == "child":
        gate_child(args.root, args.model, args.phase)
    else:
        main_run(args.root)


if __name__ == "__main__":
    main()
