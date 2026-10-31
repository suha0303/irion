"""Checked allocation and explicitly selected runtime, before model execution."""

import importlib.metadata
import importlib.util
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

from execution.common import runtime_profile

from .control import write


def check(root, out, stage, job_prefix="drone-ai-"):
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise ValueError("Slurm allocation required before CUDA import")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True)
    job = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", raw))
    if job.get("JobName") != job_prefix + stage:
        raise ValueError("unexpected round job name")
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", job.get("NodeList", "")], text=True
    ).split()
    spec = importlib.util.spec_from_file_location(
        "allocation_resource_checks", Path(root) / "remote/gpu_probe.py"
    )
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.validate_allocation(job, os.environ, socket.gethostname(), nodes, os.getuid())
    own = subprocess.check_output(
        ["squeue", "-h", "-u", str(os.getuid()), "-o", "%A"], text=True
    ).split()
    if set(own) != {job_id}:
        raise ValueError("own queue mismatch; no concurrent job allowed")
    (Path(out) / "verified-job-id.txt").write_text(job_id + "\n")
    profile = runtime_profile(Path(root).parent, required=True)
    selected_runtime = profile["runtime_profile"]
    if selected_runtime not in checks.RUNTIME_PROFILES:
        raise ValueError("Unknown explicitly selected runtime")
    expected = checks.RUNTIME_PROFILES[selected_runtime]
    versions = {n: importlib.metadata.version(n) for n in expected}
    if Path(sys.executable).resolve() != Path(profile["python"]).resolve():
        raise ValueError("Selected profile Python executable required")
    if versions != expected or sys.version_info[:2] != (3, 10):
        raise ValueError("Selected package/Python profile mismatch; installation forbidden")
    import torch

    available, count = torch.cuda.is_available(), torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0) if available and count == 1 else None
    checks.validate_torch_environment(
        sys.version_info,
        torch.__version__,
        available,
        count,
        props.name if props else "",
        props.total_memory if props else 0,
        selected_runtime,
        torch.version.cuda,
    )
    evidence = {
        "job": job,
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "runtime_profile": selected_runtime,
        "versions": versions,
        "cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled_before_fit_scope": torch.backends.cudnn.enabled,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu_count": count,
        "gpu_name": props.name,
        "gpu_memory": props.total_memory,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    evidence["optional_inventory_not_used_by_drone_raw_path"] = {}
    for name in ("jsonschema", "pyulog", "pymavlink", "PyYAML"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        evidence["optional_inventory_not_used_by_drone_raw_path"][name] = version
    write(Path(out) / "environment.json", evidence)
    return evidence
