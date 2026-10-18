"""Fail-closed allocated-node probe; never submits jobs or installs packages."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

RUNTIME_PROFILES = {
    "torch21": {"torch": "2.1.2", "numpy": "1.26.4", "safetensors": "0.4.5"},
    "mamba25": {"torch": "2.5.1+cu121", "numpy": "1.26.4", "safetensors": "0.6.2"},
}


def validate_allocation(job, env, hostname, nodes, uid):
    expected = {"JobId": env.get("SLURM_JOB_ID"), "Partition": "p02",
                "NumNodes": "1", "NumTasks": "1", "JobState": "RUNNING"}
    if not expected["JobId"] or not expected["JobId"].isdigit():
        raise ValueError("no numeric Slurm allocation job ID")
    for key, value in expected.items():
        if job.get(key) != value:
            raise ValueError("Slurm allocation mismatch: " + key)
    if job.get("UserId", "").split("(")[-1].rstrip(")") != str(uid):
        raise ValueError("allocation owner mismatch")
    if [n.upper() for n in nodes] != ["DIS01"] or hostname.split(".")[0].upper() != "DIS01":
        raise ValueError("must execute on the allocated DIS01 node")
    # Slurm 22.05.2 prints the allocated TRES as TRES for a RUNNING job.
    # Prefer modern AllocTRES when present; never use ReqTRES as evidence.
    resource_text = job.get("AllocTRES", job.get("TRES", ""))
    resources = dict(item.split("=", 1) for item in resource_text.split(",") if "=" in item)
    if resources.get("gres/gpu") != "1":
        raise ValueError("running-job AllocTRES/TRES must confirm exactly one GPU")
    typed_gpus = [value for key, value in resources.items() if key.startswith("gres/gpu:")]
    if typed_gpus and (len(typed_gpus) != 1 or typed_gpus[0] != "1"):
        raise ValueError("typed GPU allocation conflicts with the single-GPU limit")
    if not env.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("Slurm CUDA_VISIBLE_DEVICES missing; do not create it manually")
    if env.get("SLURM_JOB_PARTITION") not in {None, "p02"}:
        raise ValueError("Slurm partition environment mismatch")


def validate_torch_environment(python_version, torch_version, available, count, name, memory,
                               profile="torch21", cuda_build=None):
    expected = RUNTIME_PROFILES[profile]
    version_ok = (torch_version == expected["torch"] if profile == "mamba25"
                  else torch_version.split("+")[0] == expected["torch"])
    if tuple(python_version[:2]) != (3, 10) or not version_ok:
        raise ValueError("Python3.10 and the explicitly selected Torch profile required")
    if profile == "mamba25" and cuda_build != "12.1":
        raise ValueError("mamba25 requires the inventoried CUDA12.1 Torch build")
    if not available or count != 1:
        raise ValueError("CUDA unavailable or visible GPU count is not exactly one")
    if "A6000" not in name.upper() or not 45 * 1024**3 <= memory <= 50 * 1024**3:
        raise ValueError("GPU differs from announced A6000 48GB")


def check_package(root):
    spec = importlib.util.spec_from_file_location("frozen_gpu_handoff", root / "run_smoke.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    return wrapper.verify(root)


def batch_backward(root, device, profile="torch21"):
    import numpy as np
    import torch
    from safetensors.torch import load_file

    # Test the unchanged checkpoint under an explicitly approved runtime.
    expected = RUNTIME_PROFILES[profile]
    version = torch.__version__ if profile == "mamba25" else torch.__version__.split("+")[0]
    if version != expected["torch"]:
        raise ValueError("model test runtime differs from selected profile")
    for name, version in {k: v for k, v in expected.items() if k != "torch"}.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("frozen model dependency mismatch: " + name)
    sys.path.insert(0, str(root / "ai/src"))
    from capstone_ai import models

    if Path(models.__file__).resolve() != root / "ai/src/capstone_ai/models.py":
        raise ValueError("unexpected model import path")
    cfg = json.loads((root / "input/config.json").read_text())
    with np.load(root / "input/train.npz", allow_pickle=False) as data, np.load(
            root / "input/scaler.npz", allow_pickle=False) as scaler:
        arrays = [(data[k][:2] - scaler[n + "_mean"]) / scaler[n + "_scale"]
                  for k, n in [("X", "x"), ("C", "c"), ("Y", "y")]]
    x, context, target = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in arrays]
    torch.manual_seed(cfg["seed"])
    model = models.make_model("lstm", hidden=cfg["hidden"])
    model.load_state_dict(load_file(root / "reference/model.safetensors"))
    model = model.to(device).train()
    prediction = model(x, context)
    if tuple(prediction.shape) != (2, 10, 4):
        raise ValueError("unexpected forecast shape")
    error = (prediction - target).square()
    loss = 0.5 * error[..., :3].mean() + 0.5 * error[..., 3].mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite model loss")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None or not torch.isfinite(g).all() for g in grads):
        raise ValueError("missing or nonfinite model gradient")
    grad_l2 = sum(float(g.detach().square().sum().item()) for g in grads)**0.5
    if grad_l2 <= 0:
        raise ValueError("zero model gradient")
    if device == "cuda":
        torch.cuda.synchronize()
    return {"model": "frozen v5 LSTM hidden16", "batch_size": 2, "shape": list(prediction.shape),
            "loss": float(loss.item()), "gradient_l2": grad_l2,
            "prediction_first_lead": prediction[0, 0].detach().cpu().tolist(),
            "optimizer_steps": 0, "checkpoint_modified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-profile", choices=sorted(RUNTIME_PROFILES), required=True)
    args = parser.parse_args()
    root = args.package.resolve()
    out = args.out.resolve()
    if out.is_relative_to(root):
        raise ValueError("output must be outside the immutable package")
    out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    report = {"hostname": socket.gethostname(), "python_executable": sys.executable,
              "runtime_profile": args.runtime_profile,
              "expected_packages": RUNTIME_PROFILES[args.runtime_profile],
              "python": platform.python_version(), "allocation": "not_verified",
              "gpu_operation": "not_executed", "model_batch": "not_executed",
              "environment": {k: os.environ.get(k) for k in ["SLURM_JOB_ID", "SLURM_JOB_PARTITION",
                              "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"]}}

    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        job_id = os.environ.get("SLURM_JOB_ID", "")
        if not job_id.isdigit():
            raise ValueError("no Slurm allocation; login-node execution refused before torch import")
        raw = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True)
        job = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", raw))
        nodes = subprocess.check_output(["scontrol", "show", "hostnames", job.get("NodeList", "")],
                                        text=True).split()
        validate_allocation(job, os.environ, report["hostname"], nodes, os.getuid())
        own_jobs = subprocess.check_output(["squeue", "-h", "-u", str(os.getuid()), "-o", "%A"],
                                           text=True).split()
        if set(own_jobs) != {job_id}:
            raise ValueError("other own jobs or unverifiable queue; no concurrent probe")
        report["allocation"] = {k: job[k] for k in ["JobId", "UserId", "Partition", "NumNodes",
                                                    "NumTasks", "NodeList", "AllocTRES", "TRES"] if k in job}
        (out / "verified-job-id.txt").write_text(job_id + "\n")
        report["runtime_packages"] = {}
        for name in ["torch", "numpy", "safetensors"]:
            try:
                report["runtime_packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report["runtime_packages"][name] = None
        # Read driver metadata only; no physical GPU selection or CUDA env changes.
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                     "--format=csv,noheader"], capture_output=True,
                                    text=True, timeout=10)
            report["driver_inventory"] = {"exit_code": driver.returncode,
                                           "versions": sorted(set(driver.stdout.splitlines())),
                                           "stderr": driver.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            report["driver_inventory"] = {"unavailable": str(error)}
        report["package_path_visible"] = (root / "MANIFEST.json").is_file()
        save()
        import torch

        torch.set_num_threads(1)
        available, count = torch.cuda.is_available(), torch.cuda.device_count()
        properties = torch.cuda.get_device_properties(0) if available and count == 1 else None
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": available, "device_count": count,
                           "device_name": properties.name if properties else None,
                           "memory_bytes": properties.total_memory if properties else None}
        save()
        validate_torch_environment(sys.version_info, torch.__version__, available, count,
                                   properties.name if properties else "", properties.total_memory if properties else 0,
                                   profile=args.runtime_profile, cuda_build=torch.version.cuda)
        x = torch.tensor([1., 2., 3.], device="cuda", requires_grad=True)
        value = x.square().sum()
        value.backward()
        torch.cuda.synchronize()
        if value.item() != 14 or x.grad.tolist() != [2., 4., 6.]:
            raise ValueError("small GPU arithmetic/backward mismatch")
        report["gpu_operation"] = {"passed": True, "value": value.item(), "gradient": x.grad.tolist()}
        save()
        report["package_integrity"] = check_package(root)
        report["model_batch"] = {"passed": True, **batch_backward(root, "cuda", args.runtime_profile)}
        check_package(root)
        if report["environment"]["CUDA_VISIBLE_DEVICES"] != os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("CUDA_VISIBLE_DEVICES changed")
        report["passed"] = True
        save()
        print(json.dumps(report), flush=True)
        return 0
    except Exception as error:
        report.update({"passed": False, "error": str(error)})
        save()
        print(json.dumps(report), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
