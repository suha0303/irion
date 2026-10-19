"""CPU-only environment diagnostics: no CUDA context, installation, or GPU job."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from .profile import ROOT, RUNTIME_PROFILES, absolute_path, load


def diagnose(profile):
    checks = []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("project", Path(profile["project_root"]) == ROOT,
          {"configured": profile["project_root"], "actual": str(ROOT)})
    check("python_executable", Path(sys.executable).resolve() == Path(profile["python"]).resolve(),
          {"configured": profile["python"], "actual": sys.executable})
    check("python_version", sys.version_info[:2] == (3, 10), platform.python_version())
    packages = {}
    required = RUNTIME_PROFILES[profile["runtime_profile"]]
    for name, expected in required.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[name] = actual
        matches = actual is not None and (
            expected is None or actual == expected or (
                name == "torch" and profile["runtime_profile"] == "torch21"
                and actual.split("+")[0] == expected
            )
        )
        check("package:" + name, matches, {"expected": expected, "actual": actual})
    optional = {}
    for name in ("jsonschema", "PyYAML", "pyulog"):
        try:
            optional[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            optional[name] = None
    for command in ("sbatch", "squeue", "scontrol", "sacct", "scancel", "tmux"):
        executable = shutil.which(command)
        check("command:" + command, executable is not None, executable)
    dataset = absolute_path(profile["dataset_root"])
    standard_evidence = (dataset / "evidence/handoff.json").is_file()
    current_flat_evidence = (dataset / "handoff.json").is_file()
    legacy_smoke_snapshot = (dataset / "config.json").is_file() and (
        dataset / "inventory.json"
    ).is_file()
    check("dataset", (dataset / "dataset/normal_train.npz").is_file()
          and (dataset / "dataset/scaler.npz").is_file()
          and (standard_evidence or current_flat_evidence or legacy_smoke_snapshot),
          {"path": str(dataset), "scope": "presence only; smoke verifies the sealed input",
           "format": "prepared_upstream" if standard_evidence else (
               "current_prepared_flat" if current_flat_evidence else "legacy_smoke_snapshot"
           ),
           "main_training_admission": "unchanged; functionality-only inputs do not authorize main training"})
    output = absolute_path(profile["output_root"])
    ancestor = output
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    check("output", ancestor.is_dir() and os.access(ancestor, os.W_OK),
          {"path": str(output), "existing_parent": str(ancestor),
           "free_bytes": shutil.disk_usage(ancestor).free if ancestor.is_dir() else None})
    return {
        "status": "READY_FOR_GPU_SMOKE" if all(c["passed"] for c in checks)
        else "BLOCKED_STOPPED",
        "profile": profile["name"], "checks": checks, "packages": packages,
        "optional_raw_parser_packages_not_used_by_prepared_npz_path": optional,
        "gpu_operations": 0, "submitted_jobs": [], "installation_performed": False,
        "training_started": False, "main_training_started": False,
        "installation_plan": (
            "Select an existing supported Python environment. If dependencies are missing, "
            "review a user-owned environment installation plan first; setup never changes "
            "a shared environment or installs packages automatically."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    result = diagnose(load(args.profile))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "READY_FOR_GPU_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
