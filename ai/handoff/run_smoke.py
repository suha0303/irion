"""Verified portable entry point. Uses only stdlib until payload hashes pass."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for name, entry in manifest["files"].items():
        relative = Path(name)
        file = root / relative
        if relative.is_absolute() or ".." in relative.parts or file.is_symlink():
            raise ValueError("invalid manifest path: " + name)
        if not file.resolve().is_relative_to(root) or not file.is_file():
            raise ValueError("missing/escaped file: " + name)
        if file.stat().st_size != entry["bytes"] or digest(file) != entry["sha256"]:
            raise ValueError("hash mismatch: " + name)
    allowed = set(manifest["files"]) | {"MANIFEST.json", "SHA256SUMS"}
    extras = [str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and str(p.relative_to(root)) not in allowed
              and not (p.suffix == ".pyc" and p.parent.name == "__pycache__")]
    if extras:
        raise ValueError("unlisted package files: " + repr(extras))
    sums = {line.split("  ", 1)[1]: line.split("  ", 1)[0]
            for line in (root / "SHA256SUMS").read_text().splitlines()}
    if set(sums) != set(manifest["files"]) | {"MANIFEST.json"}:
        raise ValueError("checksum list membership mismatch")
    if any(digest(root / name) != value for name, value in sums.items()):
        raise ValueError("checksum list mismatch")
    return {"verified_files": len(manifest["files"]),
            "manifest_sha256": digest(root / "MANIFEST.json")}


def environment(root, device):
    policy = json.loads((root / "environment/policy.json").read_text())
    versions = {name: importlib.metadata.version(name) for name in policy["runtime_versions"]}
    if any(versions[name] != expected for name, expected in policy["runtime_versions"].items()):
        raise ValueError("runtime package version mismatch: " + json.dumps(versions))
    if sys.version_info[:2] != (3, 10):
        raise ValueError("Python 3.10 required; found " + platform.python_version())
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("explicit cpu or cuda[:index] device required")
    if importlib.metadata.version("torch").split("+")[0] != policy["torch_release"]:
        raise ValueError("torch " + policy["torch_release"] + " required")
    # Use the extracted sources even if the interpreter has another editable install.
    sys.path.insert(0, str(root / "ai/src"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    from capstone_ai import training
    from capstone_ai.models import device_for

    imported = Path(training.__file__).resolve()
    if imported != root / "ai/src/capstone_ai/training.py":
        raise ValueError("wrong training module: " + str(imported))
    chosen = device_for(device)
    result = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "runtime_versions": versions,
        "requested_device": device, "imported_training": str(imported),
        "cuda_available": torch.cuda.is_available(), "gpu": None,
        "installed_distributions": {d.metadata["Name"]: d.version
                                    for d in importlib.metadata.distributions()},
    }
    if chosen.type == "cuda":
        properties = torch.cuda.get_device_properties(chosen)
        result["gpu"] = {"name": properties.name, "memory_bytes": properties.total_memory,
                         "capability": [properties.major, properties.minor]}
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True)
            result["nvidia_smi"] = {"exit_code": smi.returncode, "stdout": smi.stdout,
                                    "stderr": smi.stderr}
        except FileNotFoundError:
            result["nvidia_smi"] = {"status": "unavailable"}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = None
    try:
        if sys.flags.optimize:
            raise ValueError("optimized Python disables lifecycle assertions; use without -O")
        integrity = verify(root)
        if args.verify_only:
            print(json.dumps(integrity))
            return 0
        if not args.device or args.out is None:
            raise ValueError("--device and --out required")
        destination = args.out.resolve()
        if destination.is_relative_to(root):
            raise ValueError("output must be outside the immutable package")
        destination.mkdir(parents=True, exist_ok=False)
        output = destination
        write_json(output / "preflight.json", {"integrity": integrity, "device": args.device})
        measured = environment(root, args.device)
        write_json(output / "preflight.json", {"integrity": integrity, "environment": measured})
        command = [sys.executable, str(root / "ai/tools/neural_lifecycle_smoke.py"),
                   "--data", str(root / "input"), "--out", str(output / "lifecycle"),
                   "--device", args.device]
        env = {**os.environ, "PYTHONPATH": str(root / "ai/src"), "PYTHONNOUSERSITE": "1",
               "PYTHONDONTWRITEBYTECODE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        env.pop("PYTHONOPTIMIZE", None)
        write_json(output / "command.json", {"argv": command, "cwd": str(root)})
        print("Verified package; starting four-epoch / two-plus-two lifecycle", flush=True)
        with (output / "lifecycle.log").open("w") as log:
            completed = subprocess.run(command, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
        write_json(output / "command.json", {"argv": command, "cwd": str(root),
                                             "exit_code": completed.returncode})
        if completed.returncode:
            raise ValueError("lifecycle failed; see lifecycle.log and lifecycle/report.json")
        report = json.loads((output / "lifecycle/report.json").read_text())
        if not report["passed"] or report["device"] != args.device:
            raise ValueError("lifecycle acceptance failed")
        reference_equal = None
        if args.device == "cpu":
            reference = json.loads((root / "reference/cpu-report.json").read_text())
            reference_equal = (
                reference["history"] == report["history"]
                and digest(output / "lifecycle/full/model.safetensors")
                == digest(root / "reference/model.safetensors")
            )
            if not reference_equal:
                raise ValueError("CPU result differs from frozen v5 reference")
        verify(root)
        write_json(output / "result.json", {"passed": True, "device": args.device,
                   "integrity": integrity, "cpu_reference_exact": reference_equal,
                   "gpu_status": report["gpu_status"], "environment": measured,
                   "result_hashes": {str(p.relative_to(output)): digest(p)
                                     for p in output.rglob("*") if p.is_file()},
                   "next_gate": "inspect GPU success before preliminary/main training"})
        print("PASS; " + str(output / "result.json"), flush=True)
        return 0
    except Exception as error:
        failure = {"passed": False, "error": str(error), "device": args.device,
                   "gpu_status": "not_executed_or_not_accepted"}
        if output:
            write_json(output / "result.json", failure)
        print(json.dumps(failure), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
