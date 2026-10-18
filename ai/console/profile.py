"""Untracked machine profiles. Passwords and private keys are never accepted."""

import hashlib
import json
import os
import re
from pathlib import Path

SCHEMA = "drone-ai.profile/1"
ROOT = Path(__file__).resolve().parents[2]
PROFILE_HOME = ROOT / "ai/.local/profiles"
RUNTIME_PROFILES = {
    "torch21": {"torch": "2.1.2", "numpy": "1.26.4", "safetensors": "0.4.5"},
    "mamba25": {"torch": "2.5.1+cu121", "numpy": "1.26.4", "safetensors": "0.6.2"},
}


def absolute_path(value, *, allow_leaf_symlink=False):
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("Path must be a string")
    path = Path(value).expanduser()
    checked = path.parents if allow_leaf_symlink else (path, *path.parents)
    if not path.is_absolute() or ".." in path.parts or any(
        part.is_symlink() for part in checked
    ):
        raise ValueError("An absolute path without symlinks or '..' is required")
    if any(character in str(path) for character in ("\n", "\r", "\x00")):
        raise ValueError("Control characters are forbidden in paths")
    return path


def validate(value):
    if not isinstance(value, dict) or set(value) != {
        "schema", "name", "project_root", "python", "runtime_profile", "dataset_root",
        "output_root", "ssh", "slurm",
    }:
        raise ValueError("Exact machine profile fields required; secrets are not accepted")
    if value["schema"] != SCHEMA or not isinstance(value["name"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value["name"]
    ):
        raise ValueError("Invalid profile schema or name")
    for key in ("project_root", "python", "dataset_root", "output_root"):
        absolute_path(value[key], allow_leaf_symlink=key == "python")
    if value["runtime_profile"] not in RUNTIME_PROFILES:
        raise ValueError("An explicitly supported runtime profile is required")
    ssh = value["ssh"]
    if ssh is not None:
        if not isinstance(ssh, dict) or set(ssh) != {"host", "user", "port"}:
            raise ValueError("SSH accepts host/user/port only; use terminal or SSH agent for auth")
        if not isinstance(ssh["host"], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:-]{0,252}", ssh["host"]
        ):
            raise ValueError("Invalid SSH host")
        if not isinstance(ssh["user"], str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.-]{0,63}", ssh["user"]
        ):
            raise ValueError("Invalid SSH user")
        if type(ssh["port"]) is not int or not 1 <= ssh["port"] <= 65535:
            raise ValueError("SSH port must be within 1..65535")
    slurm = value["slurm"]
    if not isinstance(slurm, dict) or set(slurm) != {
        "partition", "node", "nodes", "ntasks", "gpus", "cpus", "memory_gib"
    }:
        raise ValueError("Explicit single-node Slurm resource fields are required")
    for key in ("partition", "node"):
        if not isinstance(slurm[key], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", slurm[key]
        ):
            raise ValueError("Invalid Slurm " + key)
    for key in ("nodes", "ntasks", "gpus"):
        if type(slurm[key]) is not int or slurm[key] != 1:
            raise ValueError("Only one node/task/GPU is supported")
    for key in ("cpus", "memory_gib"):
        if type(slurm[key]) is not int or not 1 <= slurm[key] <= 1024:
            raise ValueError("Invalid Slurm resource value: " + key)
    return value


def profile_path(value="default"):
    value = str(value)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        return PROFILE_HOME / (value + ".json")
    return absolute_path(value)


def load(path):
    path = profile_path(path)
    absolute_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    return validate(value)


def digest(path):
    return hashlib.sha256(profile_path(path).read_bytes()).hexdigest()


def write(path, value, *, replace=False):
    path = profile_path(path)
    absolute_path(path)
    validate(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if replace else os.O_EXCL)
    descriptor = os.open(path, mode | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return path


def ssh_command(profile):
    ssh = profile["ssh"]
    if ssh is None:
        raise ValueError("This profile has no SSH destination")
    # Do not disable host-key checks or collect passwords. SSH handles terminal authentication.
    return ["ssh", "-o", "StrictHostKeyChecking=yes", "-p", str(ssh["port"]),
            ssh["user"] + "@" + ssh["host"]]
