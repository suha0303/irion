"""Verify portable source trust separately from a machine's compiled runtime."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

SOURCE_MANIFEST = "config/producer-sources.json"
RUNTIME_MANIFEST = "artifacts/runtime-manifest.json"
PLUGIN = "build/current/libDronePowertrain.so"
TRANSPORT = "runtime/gazebo-transport/lib/libgz-transport13.so.13.6.0"
PX4_BINARY = "vendor/PX4-Autopilot/build/px4_sitl_default/bin/px4"
PROFILE = "profiles/collection/vehicle.yaml"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_file(root, name):
    """Manifests cannot introduce an absolute path or escape the checkout."""
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe producer manifest path: {name}")
    path = Path(root) / name
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Producer manifest path escaped checkout: {name}")
    return path


def checked_files(root, files):
    changed = []
    for name, expected in files.items():
        path = relative_file(root, name)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
            or not path.is_file()
            or sha(path) != expected
        ):
            changed.append(name)
    if changed:
        raise ValueError("X500 frozen inputs changed: " + ", ".join(changed))


def verify_sources(root):
    source = json.loads((Path(root) / SOURCE_MANIFEST).read_text())
    if source.get("schema") != "x500.producer-sources/1" or not source.get("files"):
        raise ValueError("Unsupported or empty portable producer source manifest")
    checked_files(root, source["files"])
    return source


def verify_runtime(root):
    root = Path(root)
    source = verify_sources(root)
    path = root / RUNTIME_MANIFEST
    if not path.is_file():
        raise ValueError("X500 runtime is not prepared; run simulation/sim/tools/bootstrap.sh")
    runtime = json.loads(path.read_text())
    if (
        runtime.get("schema") != "x500.producer-runtime/1"
        or runtime.get("revision") != "X500_CURRENT"
        or runtime.get("source_manifest") != SOURCE_MANIFEST
        or runtime.get("source_manifest_sha256") != sha(root / SOURCE_MANIFEST)
    ):
        raise ValueError("X500 machine runtime does not match the portable source manifest")
    if "fc_parameters" in source:
        from .collection_config import fc_parameters, fc_verified

        policy = source["fc_parameters"]
        if (
            not isinstance(policy, dict)
            or set(policy) != set(fc_parameters())
            or any(type(value) not in (int, float) for value in policy.values())
            or not fc_verified(policy)
            or runtime.get("fc_parameters") != policy
        ):
            raise ValueError("X500 runtime FC policy does not match reviewed source policy")
    files = runtime.get("files", {})
    required = (
        set(source["files"])
        | set(source["vendor_files"])
        | {
            SOURCE_MANIFEST,
            "artifacts/source_lock.json",
            PLUGIN,
            TRANSPORT,
            PX4_BINARY,
        }
    )
    if not required <= files.keys():
        raise ValueError("X500 runtime seal is missing required source or binary inputs")
    for name, digest in (source["files"] | source["vendor_files"]).items():
        if files.get(name) != digest:
            raise ValueError("X500 runtime source digest mismatch: " + name)
    checked_files(root, files)
    if runtime.get("px4_binary_sha256") != files[PX4_BINARY]:
        raise ValueError("X500 PX4 runtime identity mismatch")
    if runtime.get("physics_plugin_sha256") != files[PLUGIN]:
        raise ValueError("X500 physics runtime identity mismatch")
    if runtime.get("x500_profile_sha256") != files[PROFILE]:
        raise ValueError("X500 profile runtime identity mismatch")
    if runtime.get("source_revision_sha256") != files["artifacts/source_lock.json"]:
        raise ValueError("X500 source revision identity mismatch")
    library = root / TRANSPORT
    if (library.parent / "libgz-transport13.so.13").resolve() != library:
        raise ValueError("Pinned private Gazebo transport SONAME missing or replaced")
    if not os.access(root / PX4_BINARY, os.X_OK):
        raise ValueError("PX4 binary is not executable")
    return runtime
