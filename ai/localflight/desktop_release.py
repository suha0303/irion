"""Assemble ARION V0.2 desktop release candidates with a bundled CPU runtime."""

import argparse
import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from .catalog import ROOT, digest, read_json
from .distribution import (
    copy_regular,
    copy_tree,
    prune_overlay_metadata,
    stage_v02_models,
    ui_source,
)

ELECTRON_VERSION = "45.5.0"
PLATFORMS = {
    ("Linux", "x86_64"): ("linux-x64", "ubuntu-amd64"),
    ("Windows", "AMD64"): ("win32-x64", "windows-x64"),
    ("Darwin", "arm64"): ("darwin-arm64", "macos-arm64"),
    ("Darwin", "x86_64"): ("darwin-x64", "macos-x64"),
}
LINUX_DESKTOP = """[Desktop Entry]
Type=Application
Name=ARION
Comment=Local flight monitoring and analysis
Exec=/opt/arion/arion
Icon=applications-science
Terminal=false
Categories=Science;Utility;
StartupWMClass=ARION
"""
LINUX_DEPENDS = (
    "libnss3, libnspr4, libdbus-1-3, libexpat1, "
    "libx11-6, libx11-xcb1, libxcb1, libxcomposite1, libxdamage1, "
    "libxext6, libxfixes3, libxrandr2, libxkbcommon0, libgbm1, "
    "libatspi2.0-0t64 | libatspi2.0-0, libatk1.0-0t64 | libatk1.0-0, "
    "libatk-bridge2.0-0t64 | libatk-bridge2.0-0, "
    "libgtk-3-0t64 | libgtk-3-0, libglib2.0-0t64 | libglib2.0-0, "
    "libcups2t64 | libcups2, libasound2t64 | libasound2, libdrm2, "
    "libpango-1.0-0, libcairo2, libstdc++6"
)


def copy_runtime(destination, runtime, ui_packages):
    """Copy a prepared native runtime or reuse the existing Linux CPU build."""
    if runtime is not None:
        python = "python.exe" if os.name == "nt" else "bin/python3.10"
        if not (runtime / python).is_file():
            raise ValueError(f"Missing native Python: {runtime / python}")
        source = runtime.resolve()
        for path in source.rglob("*"):
            if path.is_symlink() and (
                os.path.isabs(os.readlink(path))
                or not path.resolve(strict=True).is_relative_to(source)
            ):
                raise ValueError(
                    f"Runtime contains a nonrelocatable link: {path}"
                )
        shutil.copytree(
            source,
            destination,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        return
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise ValueError("Supply a native Python 3.10 runtime with --runtime")
    if not (ui_packages / "fastapi").is_dir():
        raise ValueError("A build-only UI dependency folder is required")
    base = Path(sys.base_prefix)
    lib = destination / "lib/python3.10"
    copy_regular(base / "bin/python3.10", destination / "bin/python3.10")
    copy_tree(base / "lib/python3.10", lib)
    site = lib / "site-packages"
    copy_tree(sysconfig.get_paths()["purelib"], site)
    copy_tree(ui_packages, site)
    prune_overlay_metadata(site, ui_packages)


def stage_core(core, ui, models, runtime, ui_packages):
    """Copy the app's code, frontend, model assets and native runtime."""
    for name in ("localflight", "learning", "execution", "src"):
        copy_tree(ROOT / "ai" / name, core / "ai" / name, python_only=True)
    for name in ("backend/arion", "frontend/dist"):
        copy_tree(ui / name, core / "irion-upload/UI" / name)
    stage_v02_models(models, core / "models/v02")
    copy_runtime(core / "runtime", runtime, ui_packages)
    license_path = ui.parent / "LICENSE"
    if license_path.is_file():
        copy_regular(license_path, core / "LICENSE")


def _extract_electron(archive, manifest, electron_target, destination):
    expected_name = f"electron-v{ELECTRON_VERSION}-{electron_target}.zip"
    hashes = {
        name.lstrip("*"): value
        for value, name in (
            line.split(maxsplit=1)
            for line in manifest.read_text(encoding="utf-8").splitlines()
        )
    }
    if archive.name != expected_name or digest(archive) != hashes.get(
        expected_name
    ):
        raise ValueError(
            f"Electron asset must match official SHA-256: {expected_name}"
        )
    if sys.platform == "darwin":
        subprocess.run(["ditto", "-x", "-k", archive, destination], check=True)
        return
    with ZipFile(archive) as zipped:
        for item in zipped.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unexpected member path in Electron archive")
            unpacked = Path(zipped.extract(item, destination))
            mode = item.external_attr >> 16 & 0o777
            if mode and os.name != "nt":
                unpacked.chmod(mode)


def _deb(app, version, output, min_libc):
    app.rename(app.parent / "arion")
    app = app.parent / "arion"
    (app / "electron").rename(app / "arion")
    (app / "chrome-sandbox").chmod(0o4755)
    root = app.parents[1]
    control = root / "DEBIAN/control"
    control.parent.mkdir()
    control.write_text(
        "Package: arion\n"
        f"Version: {version.replace('-', '~', 1)}\n"
        "Section: science\nPriority: optional\nArchitecture: amd64\n"
        "Maintainer: ARION Project\n"
        f"Depends: libc6 (>= {min_libc}), {LINUX_DEPENDS}\n"
        "Description: ARION local flight monitoring and analysis\n",
        encoding="utf-8",
    )
    entry = root / "usr/share/applications/arion.desktop"
    entry.parent.mkdir(parents=True)
    entry.write_text(LINUX_DESKTOP, encoding="utf-8")
    subprocess.run(
        [
            "dpkg-deb",
            "--root-owner-group",
            "-Zxz",
            "-z3",
            "--build",
            root,
            output,
        ],
        check=True,
    )


def _mac_app(extracted):
    app = extracted / "Electron.app"
    app.rename(extracted / "ARION.app")
    app = extracted / "ARION.app"
    plist = app / "Contents/Info.plist"
    with plist.open("rb") as stream:
        metadata = plistlib.load(stream)
    metadata.update(
        CFBundleName="ARION",
        CFBundleDisplayName="ARION",
        CFBundleIdentifier="io.github.suha0303.arion",
    )
    with plist.open("wb") as stream:
        plistlib.dump(metadata, stream)
    return app


def build(args):
    """Build on the current operating system and write a release asset."""
    electron_target, target = PLATFORMS[(platform.system(), platform.machine())]
    ui = ui_source(ROOT)
    if not (ui / "frontend/dist/index.html").is_file():
        raise ValueError("Build UI/frontend before assembling the installation")
    version = read_json(ui / "desktop/package.json")["version"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ext = {"linux": "deb", "win32": "exe", "darwin": "dmg"}[sys.platform]
    asset = output / f"ARION-{version}-{target}.{ext}"
    sha_file = asset.with_name(asset.name + ".sha256")
    if asset.exists() or sha_file.exists():
        raise ValueError("The release asset already exists")

    with TemporaryDirectory(dir=output) as temporary:
        temp = Path(temporary)
        extracted = (
            temp / "deb/opt/electron"
            if sys.platform == "linux"
            else temp / "native"
        )
        extracted.mkdir(parents=True)
        _extract_electron(
            args.electron_archive,
            args.electron_checksums,
            electron_target,
            extracted,
        )
        app = _mac_app(extracted) if sys.platform == "darwin" else extracted
        resources = app / (
            "Contents/Resources" if sys.platform == "darwin" else "resources"
        )
        default = resources / "default_app.asar"
        default.unlink(missing_ok=True)
        for name in ("main.cjs", "package.json"):
            copy_regular(ui / "desktop" / name, resources / "app" / name)
        stage_core(
            resources / "arion-core",
            ui,
            args.model_source,
            args.runtime,
            args.ui_packages,
        )
        candidate = temp / asset.name
        if sys.platform == "linux":
            _deb(app, version, candidate, "2.35" if args.runtime else "2.38")
        elif sys.platform == "win32":
            (app / "electron.exe").rename(app / "ARION.exe")
            compiler = shutil.which("iscc") or (
                Path(os.environ["ProgramFiles(x86)"]) / "Inno Setup 6/ISCC.exe"
            )
            subprocess.run(
                [
                    compiler,
                    f"/DAppVersion={version}",
                    f"/DAppSource={app}",
                    f"/DAppOutput={temp}",
                    f"/DAppName={asset.stem}",
                    ui / "desktop/arion.iss",
                ],
                check=True,
            )
        else:
            subprocess.run(
                ["codesign", "--force", "--deep", "--sign", "-", app],
                check=True,
            )
            image_root = temp / "disk-image"
            image_root.mkdir()
            app.rename(image_root / app.name)
            (image_root / "Applications").symlink_to("/Applications")
            subprocess.run(
                [
                    "hdiutil",
                    "create",
                    "-volname",
                    "ARION",
                    "-srcfolder",
                    image_root,
                    "-format",
                    "UDZO",
                    candidate,
                ],
                check=True,
            )
        candidate.rename(asset)
    sha_file.write_text(f"{digest(asset)}  {asset.name}\n", encoding="utf-8")
    return {
        "asset": str(asset),
        "bytes": asset.stat().st_size,
        "sha256": str(sha_file),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model-source",
        "electron-archive",
        "electron-checksums",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument(
        "--ui-packages",
        type=Path,
        default=ROOT / "ai/.local/desktop-build/ui-packages",
    )
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
