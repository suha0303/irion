"""Run one new19 lifecycle in its own verified Slurm GPU step."""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

from . import calibration, final, training
from .data import digest, load_run, read, write
from .inference import Runtime
from .models import VARIANTS, make_model

MANIFEST_SHA256 = (
    "ea0fcdaba639c5f226270a4190259045f1a11da8a07ec4f3f41a74e8da479e84"
)
PHASE = {"inflight": 256, "whole-run": 8}


def verify_environment(leader, node, *, standalone=False):
    names = {"DIS03": 1} if standalone else {"DIS01": 6, "DIS05": 2}
    if node not in names or socket.gethostname().split(".")[0] != node:
        raise RuntimeError("Expected authorized compute node")
    raw = subprocess.check_output(
        ["scontrol", "show", "job", "-o", leader], text=True
    )
    matches = []
    for line in raw.splitlines():
        row = {
            k: v
            for part in line.split()
            if "=" in part
            for k, v in (part.split("=", 1),)
        }
        if row.get("NodeList") == node:
            matches.append(row)
    if len(matches) != 1:
        raise RuntimeError("Cannot identify the current Slurm component")
    row = matches[0]
    correct_id = (
        row.get("JobId") == leader
        if standalone
        else row.get("HetJobId") == leader
    )
    if (
        not correct_id
        or row.get("Partition") != "p02"
        or row.get("UserId") != "coss43(10171)"
        or row.get("JobState") != "RUNNING"
        or row.get("NumNodes") != "1"
        or row.get("TresPerNode") != f"gres:gpu:{names[node]}"
        or row.get("NumCPUs") != str(names[node] * 4)
        or os.environ.get("SLURM_JOB_ID") not in {leader, row.get("JobId")}
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
        or not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        or len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1
        or sys.version_info[:2] != (3, 10)
        or torch.__version__ != "2.5.1+cu121"
        or torch.version.cuda != "12.1"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("Allocation, owner, visible GPU or runtime mismatch")
    gpu = torch.cuda.get_device_properties(0)
    if "A6000" not in gpu.name or gpu.total_memory < 45 * 1024**3:
        raise RuntimeError("Expected one NVIDIA RTX A6000 with 48 GB")
    return dict(
        node=node,
        gpu=gpu.name,
        memory_bytes=gpu.total_memory,
        leader=leader,
        component=row["JobId"],
        step=os.environ["SLURM_STEP_ID"],
        torch=str(torch.__version__),
        cuda_build=torch.version.cuda,
        python=str(sys.version.split()[0]),
    )


def set_backend():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probe(root, name, output, batch_size):
    probe_data = read(root / "probe.json")
    original = root / "data" / probe_data["path"]
    if digest(original) != probe_data["sha256"]:
        raise ValueError("Production shape probe source changed")
    run = load_run(original, probe_data["run_id"])
    task = VARIANTS[name].task
    train = training.training_batches([run], task)
    model = make_model(name).cuda()
    optimizer = training.adamw(model, 1e-3)
    idx = [i % len(train.items) for i in range(batch_size)]
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(4):
        start = time.perf_counter()
        loss = training.update(model, optimizer, train.batch(idx), 1e-3)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        val = float(
            training.objective(
                model, training.to_device(train.batch(idx), "cuda")
            )
        )
        torch.cuda.synchronize()
    val_seconds = time.perf_counter() - start
    write(
        output / "shape.json",
        dict(
            batch_size=batch_size,
            longest_run_frames=len(run.values),
            loss=loss,
            validation=val,
            train_batch_seconds=times,
            validation_batch_seconds=val_seconds,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        ),
    )
    del model, optimizer, train
    torch.cuda.empty_cache()
    plan = root / "probe-mini" / "manifest.json"
    target = output / "restored"
    opts = dict(
        rate=1e-3,
        epochs=2,
        batch_size=16 if task == "inflight" else 2,
        device="cuda",
    )
    training.fit(plan, name, target, until_epoch=1, **opts)
    result = training.fit(plan, name, target, resume=True, **opts)
    Runtime(target, device="cuda")
    return result


def run(root, name, output, batch_size):
    manifest = root / "data" / "new19-mavlink-1750-20261120.json"
    if digest(manifest) != MANIFEST_SHA256:
        raise ValueError("Sealed 1750-run roster hash mismatch")
    fit_dir, calibration_dir = output / "fit", output / "calibration"
    write(output / "status.json", {"phase": "lr_range"})
    for i, bounds in enumerate(((1e-6, 1e-1), (1e-7, 1e-2), (1e-5, 3e-1)), 1):
        lr = training.lr_range(
            manifest,
            name,
            output / f"range-{i}",
            start_lr=bounds[0],
            end_lr=bounds[1],
            steps=100,
            batch_size=batch_size,
            device="cuda",
        )
        if lr["candidate"] is not None:
            break
    if lr["candidate"] is None:
        raise ValueError(
            "Three bounded LR Range Tests yielded no falling EMA region"
        )
    rate = lr["candidate"]
    write(output / "status.json", {"phase": "fit", "lr": rate})
    fit = training.fit(
        manifest,
        name,
        fit_dir,
        rate=rate,
        epochs=60,
        patience=10,
        batch_size=batch_size,
        device="cuda",
    )
    if fit["status"] not in {"complete", "early_stopped"}:
        raise ValueError("Model fit not complete")
    torch.cuda.empty_cache()
    runtime = Runtime(fit_dir, device="cuda")
    write(
        output / "status.json",
        {"phase": "calibration", "lr": rate, "best_epoch": fit["best_epoch"]},
    )
    calibration.calibrate(
        fit_dir, runtime, manifest, calibration_dir, batch_size=256
    )
    write(
        output / "status.json",
        {
            "phase": "final_evaluation",
            "lr": rate,
            "best_epoch": fit["best_epoch"],
        },
    )
    result = final.evaluate(
        fit_dir,
        runtime,
        manifest,
        calibration_dir / "calibration.json",
        output / "final",
        batch_size=256,
    )
    if result["total_runs"] != 450:
        raise ValueError("Final cohort count mismatch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--leader", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--model", choices=VARIANTS, required=True)
    args = parser.parse_args()
    output = args.root / args.mode / args.model
    evidence = verify_environment(
        args.leader, args.node, standalone=args.standalone
    )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "environment.json", evidence)
    set_backend()
    torch.manual_seed(0)
    mode = probe if args.mode == "probe" else run
    try:
        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.MATH]
        ):
            result = mode(
                args.root, args.model, output, PHASE[VARIANTS[args.model].task]
            )
    except Exception as exc:
        write(
            output / "status.json",
            {"phase": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    write(output / "status.json", {"phase": "complete", "result": result})


if __name__ == "__main__":
    main()
