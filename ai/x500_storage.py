"""Per-task storage selection; no implicit fallback when a selected device disappears."""

import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "Data"
LEGACY_ROOT = Path("/media/sa/X31/Data/X500/ROUND_001/R3_LONG15")
ENV_ROOT = "X500_STORAGE_ROOT"
ENV_POLICY = "X500_STORAGE_POLICY"


def absolute(path):
    path = Path(path).expanduser().absolute()
    if ".." in path.parts:
        raise ValueError("Storage path traversal forbidden")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Symlink storage path forbidden: " + str(part))
        if part.exists() and not part.is_dir():
            raise ValueError("Storage directory required: " + str(part))
    return path


def selected_root():
    return absolute(os.environ.get(ENV_ROOT, DEFAULT_ROOT))


def capture(root):
    root = absolute(root)
    # Missing removable mount paths must not become directories on the system disk.
    if root.parts[1:2] == ("media",) and len(root.parts) >= 4:
        mount = Path(*root.parts[:4])
        if not mount.is_mount():
            raise ValueError("Mounted external storage unavailable: " + str(mount))
    elif root.parts[1:2] == ("mnt",) and len(root.parts) >= 3:
        mount = Path(*root.parts[:3])
        if not mount.is_mount():
            raise ValueError("Mounted external storage unavailable: " + str(mount))
    existing = next(p for p in (root, *root.parents) if p.exists())
    mount = next(p for p in (existing, *existing.parents) if p.is_mount())
    return dict(
        schema="x500.storage/1", root=str(root), mount=str(mount), device=existing.stat().st_dev
    )


def guard(path, policy=None):
    path = absolute(path)
    policy = policy or capture(path)
    root, mount = absolute(policy["root"]), absolute(policy["mount"])
    if not path.is_relative_to(root):
        raise ValueError("Path outside selected storage root")
    if not mount.is_mount() or mount.stat().st_dev != policy["device"]:
        raise ValueError("Selected storage filesystem unavailable/changed; no fallback")
    existing = next(p for p in (path, *path.parents) if p.exists())
    if existing.stat().st_dev != policy["device"]:
        raise ValueError("Selected storage device changed; no fallback")
    return path


def task_root(config):
    return absolute(config.get("external_storage", LEGACY_ROOT))


def environment(config):
    root = task_root(config)
    policy = config.get("storage_policy") or capture(root)
    if policy["root"] != str(root):
        raise ValueError("Storage policy/config mismatch")
    guard(root, policy)
    return {ENV_ROOT: str(root), ENV_POLICY: json.dumps(policy)}


def active_policy():
    policy = os.environ.get(ENV_POLICY)
    if policy:
        policy = json.loads(policy)
        if policy["root"] != str(selected_root()):
            raise ValueError("Storage environment mismatch")
        return policy
    return capture(selected_root())
