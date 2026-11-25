"""Prepare the pinned native build runtime and assemble a desktop candidate."""

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from .catalog import ROOT, digest
from .desktop_release import ELECTRON_VERSION, PLATFORMS

RUNTIME_RELEASE = "20260929"
RUNTIME_VERSION = "3.10.21"
NATIVE_INPUTS = {
    "linux-x64": (
        "x86_64-unknown-linux-gnu",
        "b6c7d6cff8e70f143b5bbd47c2bb7ecc8353ee70ef30fe4ff79bfbe7179624f7",
        "36b44cd0c3d355ee74557bea2353e4539f96e1a09733a52a052bd093134a6142",
    ),
    "win32-x64": (
        "x86_64-pc-windows-msvc",
        "806de086b36a83ad2863a9d6c2945de105f707fded114e248e8974a2dc64b7eb",
        "20155c8b8aa5b1f575f3757bdf21490e0bac2f19a65a26ce9ba99bcf9f2c69bd",
    ),
    "darwin-arm64": (
        "aarch64-apple-darwin",
        "796f646044a07f05802f96e7d6493903cab8c6e7fe3936962705b3f0a9bba702",
        "00d345b061064de7e6e2c02bec7ffa1adb7ef488fe4b673bd8ad3e059a182245",
    ),
    "darwin-x64": (
        "x86_64-apple-darwin",
        "2f8b3f1f30a841ef091198225b5523793d50d3a993dd63b8423e7e2199951a37",
        "9ce4b0e25ce95e743b2d8f04ebfbfd76027060ff8b47829de22334fbdbdd4cc4",
    ),
}


def download(folder, base, name, expected):
    target = folder / name
    with urlopen(base + "/" + name.replace("+", "%2B"), timeout=60) as response:
        with target.open("xb") as stream:
            shutil.copyfileobj(response, stream)
    if digest(target) != expected:
        raise ValueError(f"Official archive checksum mismatch: {name}")
    return target


def build(model_source, output):
    """Build with a temporary platform-native Python and wheel installation."""
    electron_target, _ = PLATFORMS[(platform.system(), platform.machine())]
    triplet, python_sha, electron_sha = NATIVE_INPUTS[electron_target]
    with TemporaryDirectory(prefix="arion-native-") as temporary:
        folder = Path(temporary)
        runtime_archive = download(
            folder,
            "https://github.com/astral-sh/python-build-standalone/releases/download/"
            + RUNTIME_RELEASE,
            f"cpython-{RUNTIME_VERSION}+{RUNTIME_RELEASE}-{triplet}-install_only.tar.gz",
            python_sha,
        )
        electron = download(
            folder,
            f"https://github.com/electron/electron/releases/download/v{ELECTRON_VERSION}",
            f"electron-v{ELECTRON_VERSION}-{electron_target}.zip",
            electron_sha,
        )
        checksums = folder / "electron-checksums.txt"
        checksums.write_text(
            f"{electron_sha} *{electron.name}\n", encoding="utf-8"
        )
        subprocess.run(
            ["tar", "-xzf", runtime_archive, "-C", folder], check=True
        )
        runtime = folder / "python"
        python = runtime / (
            "python.exe" if os.name == "nt" else "bin/python3.10"
        )
        env = dict(
            os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"
        )
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        install = [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--only-binary=:all:",
        ]
        torch = (
            "torch==2.1.2"
            if platform.system() == "Darwin"
            else "torch==2.1.2+cpu"
        )
        torch_index = (
            []
            if platform.system() == "Darwin"
            else [
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        lock = ROOT / "ai/localflight"
        subprocess.run(
            [
                *install,
                "-r",
                lock / "requirements-desktop-ubuntu.lock",
                "-r",
                lock / "requirements-desktop-native.lock",
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [*install, "--no-deps", *torch_index, torch], check=True, env=env
        )
        env["PYTHONPATH"] = str(ROOT / "ai")
        subprocess.run(
            [
                python,
                "-B",
                "-m",
                "localflight.desktop_release",
                "--model-source",
                model_source,
                "--runtime",
                runtime,
                "--electron-archive",
                electron,
                "--electron-checksums",
                checksums,
                "--output",
                output,
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.model_source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
