"""Build and verify a relocatable Ubuntu 24.04 x86_64 CPU application."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import sysconfig
from pathlib import Path

from .catalog import ROOT, Catalog, digest

LAUNCHER = """#!/bin/sh
set -eu
app_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONHOME="$app_root/runtime"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$app_root/ai:$app_root/ai/src:$app_root/irion-upload/UI/backend"
export ARION_AI_ROOT="$app_root"
export ARION_AI_PYTHON="$app_root/runtime/bin/python3.10"
export ARION_DATA_DIR="${ARION_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/drone-local}"
case "${1:-start}" in
  start|stop|status) exec "$ARION_AI_PYTHON" -m arion.service "${1:-start}" ;;
  doctor) exec "$ARION_AI_PYTHON" -m localflight.distribution doctor ;;
  worker) exec "$ARION_AI_PYTHON" -m localflight.cli worker ;;
  migrate-data) shift; exec "$ARION_AI_PYTHON" -m arion.data_transfer "$@" ;;
  *) echo 'Usage: ./drone-local [start|stop|status|doctor|worker|migrate-data]' >&2; exit 2 ;;
esac
"""


def copy_regular(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError(
            f"Only regular application files may be packaged: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source, destination, *, python_only=False):
    for p in sorted(Path(source).rglob("*")):
        if p.is_dir():
            continue
        relative = p.relative_to(source)
        if (
            "__pycache__" in relative.parts
            or p.suffix in {".pyc", ".pth"}
            or relative.parts[0] == "bin"
        ):
            continue
        if python_only and p.suffix != ".py":
            continue
        copy_regular(p, Path(destination) / relative)


def _metadata_directories(root):
    for path in sorted(Path(root).iterdir()):
        if not path.name.endswith((".dist-info", ".egg-info")):
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError(
                f"Regular package metadata directory required: {path}"
            )
        metadata = importlib.metadata.Distribution.at(path).metadata
        name = metadata["Name"]
        if not name:
            raise ValueError(f"Package metadata has no Name: {path}")
        yield name.lower().replace("_", "-"), path


def prune_overlay_metadata(site: str | Path, overlay: str | Path) -> None:
    """Remove superseded metadata from a copied application site-packages.

    Only metadata directories for packages present in the UI overlay are
    removed. Package code and unrelated distributions remain in place.

    Args:
        site: Copied application site-packages, which this operation may edit.
        overlay: UI package directory, read without modification.

    Raises:
        ValueError: Metadata is linked, is not a directory, or has no Name.
    """
    overlay_names = {
        name: path.name for name, path in _metadata_directories(overlay)
    }
    for name, path in _metadata_directories(site):
        if name in overlay_names and path.name != overlay_names[name]:
            shutil.rmtree(path)


def ui_source(root):
    """Find UI sources in a public clone or the developer upload checkout.

    Args:
        root: Project root containing the AI package.

    Returns:
        The UI source directory. Publication layout takes precedence.

    Raises:
        ValueError: Neither supported layout contains the backend package.
    """
    for candidate in (Path(root) / "UI", Path(root) / "irion-upload/UI"):
        if (candidate / "backend/arion").is_dir():
            return candidate
    raise ValueError(
        "UI backend sources are missing; use a complete project checkout"
    )


def build(output, ui_packages):
    if (
        sys.version_info[:2] != (3, 10)
        or platform.machine() != "x86_64"
        or platform.system() != "Linux"
    ):
        raise ValueError(
            "Build with the existing Python 3.10 Ubuntu x86_64 AI environment"
        )
    output = Path(output).absolute()
    ui_packages = Path(ui_packages).absolute()
    if output.exists():
        raise ValueError(
            (
                "Choose a new output directory; existing packages are never "
                "overwritten"
            )
        )
    # Existing binary runtimes are copied; no installation or shared-env
    # modification.
    base = Path(sys.base_prefix)
    ai_site = Path(sysconfig.get_paths()["purelib"])
    required = (
        sum(
            p.stat().st_size
            for folder in (base, ai_site, ui_packages)
            for p in folder.rglob("*")
            if p.is_file()
        )
        + 512 * 1024**2
    )
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < required * 2:
        raise ValueError(
            "Insufficient free space for staging and application package"
        )
    ui_root = ui_source(ROOT)
    if (
        not (ui_packages / "fastapi").is_dir()
        or not (ui_root / "frontend/dist/index.html").is_file()
    ):
        raise ValueError(
            (
                "Resolve UI dependencies into the isolated build folder and "
                "build the frontend first"
            )
        )
    catalog = Catalog()
    catalog.check_model_sources()
    output.mkdir()
    for directory in (
        "ai/localflight",
        "ai/learning",
        "ai/execution",
        "ai/src",
    ):
        copy_tree(ROOT / directory, output / directory, python_only=True)
    copy_regular(
        ROOT / "ai/localflight/requirements-gpu.lock",
        output / "ai/localflight/requirements-gpu.lock",
    )
    copy_tree(
        ui_root / "backend/arion",
        output / "irion-upload/UI/backend/arion",
    )
    copy_tree(
        ui_root / "frontend/dist",
        output / "irion-upload/UI/frontend/dist",
    )
    for name in (
        "LOCAL_DESKTOP_UBUNTU.md",
        "REAL_AIRCRAFT_CONNECTION_GUIDE.md",
        "LOCAL_REAL_FLIGHT_TIME_HANDLING.md",
        "LOCAL_REAL_FLIGHT_CHECKLIST.md",
    ):
        copy_regular(ROOT / "docs/ai" / name, output / "docs/ai" / name)
    for name in (
        "CODE_REVIEW_M17_M18.md",
        "CODE_REVIEW_M19_M21.md",
        "CODE_REVIEW_M22_M24.md",
        "CODE_REVIEW_M26_M29.md",
        "LOCAL_GPU_G1_G3.md",
    ):
        copy_regular(
            ROOT / "docs/development" / name,
            output / "docs/development" / name,
        )
    frozen = catalog.root.relative_to(ROOT)
    required_model_files = set()
    for model in catalog.manifest["models"]:
        required_model_files.update(
            str(p.relative_to(catalog.root))
            for p in catalog.entry(model)["files"].values()
        )
    required_model_files.update(
        f"code/ai/learning/{name}/models.py"
        for name in ("history19", "run_health19")
    )
    copy_regular(
        catalog.root / "MANIFEST.json", output / frozen / "MANIFEST.json"
    )
    for name in sorted(required_model_files):
        copy_regular(catalog.checked(name), output / frozen / name)
    # Keep the delivered manifest intact; explicitly declare which files this
    # runtime needs.
    (output / "MODEL_CONTENTS.json").write_text(
        json.dumps(
            dict(
                source_manifest_sha256=digest(catalog.root / "MANIFEST.json"),
                shipped_files=sorted(required_model_files),
                evaluation_predictions_included=False,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    copy_regular(base / "bin/python3.10", output / "runtime/bin/python3.10")
    copy_tree(base / "lib/python3.10", output / "runtime/lib/python3.10")
    site = output / "runtime/lib/python3.10/site-packages"
    copy_tree(ai_site, site)
    # Overlay separately resolved UI packages. Exact installed versions are
    # recorded below.
    copy_tree(ui_packages, site)
    # Remove stale duplicate distribution metadata when the overlay upgrades a
    # shared package.
    prune_overlay_metadata(site, ui_packages)
    (output / "drone-local").write_text(LAUNCHER, encoding="utf-8")
    (output / "drone-local").chmod(0o755)
    packages = sorted(
        {
            d.metadata["Name"] + "==" + d.version
            for d in importlib.metadata.distributions(path=[str(site)])
        }
    )
    (output / "DEPENDENCIES.lock").write_text(
        "\n".join(packages) + "\n", encoding="utf-8"
    )
    licenses = output / "licenses"
    licenses.mkdir()
    for label, path in [
        ("python", base / "lib/python3.10/LICENSE.txt"),
        ("ui", ui_root.parent / "LICENSE"),
    ]:
        if path.is_file():
            copy_regular(path, licenses / (label + ".txt"))
    (output / "README.txt").write_text(
        "Ubuntu 24.04 x86_64 CPU local flight analysis\n"
        "./drone-local doctor\n./drone-local start\nOpen http://"
        "127.0.0.1:8765\n./drone-local stop\n"
        "No runtime downloads or GPU server. Data: ${XDG_DATA_HOME:-"
        "~/.local/share}/drone-local\n"
        "ARION_PORT and ARION_DATA_DIR may select a different local "
        "port/data folder.\n"
        "Stop the old app before upgrading. Another installation on the "
        "same port is not accepted as this release.\n"
        "Offline data copy: ./drone-local migrate-data --source /old/data "
        "--destination /new/data (new destination only; original retained).\n"
        "Instructions/checklist: docs/ai/LOCAL_DESKTOP_UBUNTU.md\n"
        "Keep the folder intact; rename/move it freely. Do not copy "
        "raw flight logs into it.\n"
        "Empirical aircraft/model thresholds supported; real-"
        "aircraft detection performance unverified.\n",
        encoding="utf-8",
    )
    hashes = {
        str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    manifest = dict(
        schema="drone.local-app/1",
        platform="Ubuntu 24.04 x86_64",
        release=output.name,
        python=sys.version,
        models=16,
        device="cpu",
        files=hashes,
        bytes=sum((output / p).stat().st_size for p in hashes),
        source_model_manifest_sha256=digest(catalog.root / "MANIFEST.json"),
    )
    (output / "APP_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return dict(
        path=str(output),
        files=len(hashes),
        bytes=manifest["bytes"],
        manifest_sha256=digest(output / "APP_MANIFEST.json"),
    )


def doctor():
    import numpy as np
    import torch

    from .runtime import Runtime

    manifest = json.loads(
        (ROOT / "APP_MANIFEST.json").read_text(encoding="utf-8")
    )
    for relative, expected in manifest["files"].items():
        p = ROOT / relative
        if p.is_symlink() or not p.is_file() or digest(p) != expected:
            raise ValueError("Application integrity mismatch: " + relative)
    if torch.version.cuda is not None:
        raise ValueError("This distribution requires a CPU-only PyTorch build")
    catalog = Catalog()
    rows = catalog.list()
    for item in rows:
        runtime = Runtime(item["id"], catalog=catalog)
        if item["task"] == "forecast":
            runtime.predict_scaled(np.zeros((1, 100, 19), dtype="f4"))
        else:
            runtime.predict_scaled(
                np.zeros((1, 110, 19), dtype="f4"), np.array([110], dtype="i8")
            )
    for module in (
        "fastapi",
        "uvicorn",
        "pymavlink",
        "pyulog",
        "arion.file_io",
        "arion.data_transfer",
        "localflight.cli",
    ):
        importlib.import_module(module)

    return dict(
        status="ok",
        models=len(rows),
        device="cpu",
        torch=str(torch.__version__),
        external_server=False,
        root=str(ROOT),
        python_prefix=sys.prefix,
        data_directory=os.environ.get("ARION_DATA_DIR"),
        real_aircraft_validated=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "doctor"])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ui-packages",
        type=Path,
        default=ROOT / "ai/.local/desktop-build/ui-packages",
    )
    args = parser.parse_args(argv)
    try:
        if args.action == "build" and args.output is None:
            raise ValueError("--output is required")
        result = (
            build(args.output, args.ui_packages)
            if args.action == "build"
            else doctor()
        )
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
