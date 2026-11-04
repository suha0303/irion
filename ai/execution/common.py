import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

MODELS = ("persistence", "ridge_forecast", "lstm", "transformer")
COUNTS = dict(
    normal_train=63, normal_validation=9, normal_calibration=9, normal_dev_eval=11, anomaly_dev=12
)
ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "Data"
HOME = ROOT / "ai/operations/legacy-comparison"
HANDOFF = STORE / "processed/prepared/handoff.json"
# Retained internal execution helpers use explicit profiles for any remote work.
# The local executable is only an import-safe default for legacy CPU tests.
REMOTE_PYTHON = sys.executable
REMOTE_BASE = str(Path.home() / "drone-ai")
HOST = ""
CONTROL = ROOT / "ai/operations/ssh/control"
GPU_LOCK = Path.home() / ".drone-ai-gpu.lock"
SUBMIT_LOCK = Path.home() / ".drone-ai-submit.lock"


def runtime_profile(root=None, *, required=False):
    """Load a selected profile without guessing a server or changing its environment."""
    selected = os.environ.get("DRONE_AI_PROFILE")
    candidates = []
    if root is not None:
        base = Path(root)
        candidates.extend((base / "server-profile.json", base.parent / "server-profile.json"))
    if selected:
        candidates.append(Path(selected))
    for candidate in candidates:
        if candidate.is_file():
            from console.profile import load

            return load(candidate)
    if required:
        raise ValueError("An explicit server profile is required; set DRONE_AI_PROFILE")
    return None


def python_for(root=None, *, required=False):
    profile = runtime_profile(root, required=required)
    return profile["python"] if profile else sys.executable


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def child(root, name):
    name = Path(name)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError("Unsafe relative input: " + str(name))
    path = Path(root) / name
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink input forbidden: " + str(path))
    return path


def verify(root, inventory):
    for name, expected in inventory.items():
        if sha(child(root, name)) != expected:
            raise ValueError("Input hash changed: " + name)


def config_check(cfg):
    expected = dict(
        models=list(MODELS),
        seed=17,
        hidden=32,
        batch_size=32,
        learning_rate=0.001,
        epochs=20,
        early_stopping_patience=5,
        precision="float32",
        amp=False,
        retries=0,
        final_test_used=False,
        memory_gib=16,
        cpus=4,
        wall_hours=12,
        queue_hours=24,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in expected.items()):
        raise ValueError("Fixed four-model seed17 configuration mismatch")


def _memory_mounts(membership, mountinfo):
    """Resolve memory hierarchies using the process's mount namespace."""
    mounts = []
    for line in mountinfo.splitlines():
        left, right = line.split(" - ", 1)
        before, after = left.split(), right.split()
        if after[0] == "cgroup2" or (after[0] == "cgroup" and "memory" in after[2].split(",")):

            def decode(text):
                # mountinfo escapes whitespace and backslashes.
                return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), text)

            mounts.append((after[0], Path(decode(before[3])), Path(decode(before[4]))))
    for line in membership.splitlines():
        _, controllers, relative = line.split(":", 2)
        version = "cgroup2" if controllers == "" else "cgroup"
        if controllers and "memory" not in controllers.split(","):
            continue
        relative = Path(relative)
        if not relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid memory cgroup membership")
        candidates = [m for m in mounts if m[0] == version and relative.is_relative_to(m[1])]
        if not candidates:
            raise ValueError("Memory cgroup mount is not visible for " + str(relative))
        for _, mount_root, mount in candidates:
            yield version, mount, mount / relative.relative_to(mount_root)


def _cache_credit(folder, version, current):
    """Credit only a conservative estimate of inactive clean file-backed cache.

    Never credit active cache, anonymous memory, tmpfs, dirty/writeback or slab.
    Missing counters mean zero credit, not an assumed zero dirty/shared count.
    """
    result = dict(stats={}, credited_cache_bytes=0, basis="no cache credit")
    try:
        for line in (folder / "memory.stat").read_text().splitlines():
            key, text = line.split()
            value = int(text)
            if value < 0 or key in result["stats"]:
                raise ValueError("Negative or duplicate memory.stat counter")
            result["stats"][key] = value
        if version == "cgroup2":
            names = ("inactive_file", "file", "file_dirty", "file_writeback", "shmem")
            result["stats_scope"] = "hierarchical"
        else:
            hierarchy = (folder / "memory.use_hierarchy").read_text().strip()
            result["use_hierarchy"] = hierarchy
            if hierarchy not in {"0", "1"}:
                raise ValueError("Unknown v1 hierarchy accounting")
            prefix = "total_" if hierarchy == "1" else ""
            names = tuple(
                prefix + name for name in ("inactive_file", "cache", "dirty", "writeback", "shmem")
            )
            result["stats_scope"] = "hierarchical" if prefix else "local"
        inactive, file_bytes, dirty, writeback, shmem = (result["stats"][k] for k in names)
        result["credited_cache_bytes"] = max(
            0, min(current, inactive, file_bytes) - dirty - writeback - shmem
        )
        result["basis"] = "max(0, min(current,inactive_file,file_cache)-dirty-writeback-shmem)"
    except (OSError, ValueError, KeyError) as exc:
        result["stats_error"] = str(exc)
    return result


def memory_snapshot(proc_root=Path("/proc")):
    """One read-only, conservative host/cgroup admission observation.

    File cache charged by integrity reads is potentially reclaimable, not RSS.
    This is an estimate, never a reservation; host/ancestor caps and the caller's
    reserve still apply. The kernel's limits and reclaim policy are unchanged.
    """
    result = dict(
        timestamp=time.time(),
        node_available_bytes=None,
        available_bytes=0,
        cgroups=[],
        errors=[],
        accounting="inactive_clean_file_credit",
    )
    try:
        fields = dict(
            line.split(":", 1) for line in (proc_root / "meminfo").read_text().splitlines()
        )
        available = int(fields["MemAvailable"].split()[0]) * 1024
        if available < 0:
            raise ValueError("Negative node MemAvailable")
        result["node_available_bytes"] = available
        membership = (proc_root / "self/cgroup").read_text()
        mountinfo = (proc_root / "self/mountinfo").read_text()
        result["membership"] = membership
        seen = set()
        for version, mount, folder in _memory_mounts(membership, mountinfo):
            if not folder.is_dir():
                raise ValueError("Memory cgroup directory is not visible: " + str(folder))
            while folder.is_relative_to(mount):
                key = (version, str(folder))
                if key not in seen:
                    seen.add(key)
                    limit_name, current_name = (
                        ("memory.max", "memory.current")
                        if version == "cgroup2"
                        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
                    )
                    limit_path, current_path = folder / limit_name, folder / current_name
                    if limit_path.exists() or current_path.exists():
                        row = dict(path=str(folder), version=version)
                        result["cgroups"].append(row)
                        # v2 root has current/stat but no limit; its bound is node MemAvailable.
                        if not limit_path.exists() and folder == mount and version == "cgroup2":
                            row["limit_bytes"] = None
                            row["current_bytes"] = int(current_path.read_text())
                        else:
                            text = limit_path.read_text().strip()
                            limit = None if text == "max" else int(text)
                            current = int(current_path.read_text())
                            if current < 0 or (limit is not None and limit < 0):
                                raise ValueError("Negative cgroup memory counter")
                            row.update(
                                limit_bytes=limit,
                                current_bytes=current,
                                **_cache_credit(folder, version, current),
                            )
                            if limit is not None:
                                row["raw_headroom_bytes"] = max(0, limit - current)
                                row["available_bytes"] = max(
                                    0, limit - current + row["credited_cache_bytes"]
                                )
                                available = min(available, row["available_bytes"])
                if folder == mount:
                    break
                folder = folder.parent
        result["available_bytes"] = available
    except (OSError, ValueError, KeyError, IndexError) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        # Unknown limit/accounting is never permission to use unrestricted host RAM.
        result["available_bytes"] = 0
    return result


def memory_available():
    return memory_snapshot()["available_bytes"]


class Progress:
    def __init__(self, path, hours=12):
        self.path = Path(path)
        self.started = time.time()
        self.deadline = time.monotonic() + hours * 3600
        self.state = dict(
            status="RUNNING",
            started=self.started,
            pid=os.getpid(),
            llm_calls=0,
            retries=0,
            final_test_used=False,
        )
        self.last = 0

    def update(self, force=True, **fields):
        if "model" in fields and fields["model"] != self.state.get("model"):
            for key in (
                "epoch",
                "epochs",
                "batch",
                "batches",
                "train_loss",
                "validation_loss",
                "best_epoch",
                "steps_per_s",
                "gpu_reserved_bytes",
                "phase_eta_s",
            ):
                self.state.pop(key, None)
        if "stage" in fields and fields["stage"] != self.state.get("stage"):
            self.state.pop("phase_eta_s", None)
        self.state.update(fields)
        if force or time.monotonic() - self.last >= 2:
            self.last = time.monotonic()
            self.state.update(updated=time.time(), elapsed_s=time.time() - self.started)
            write(self.path, self.state)

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Finite wall budget exhausted; no retry")
        snapshot = memory_snapshot()
        if snapshot["available_bytes"] < 1024**3:
            evidence = self.path.with_name(self.path.stem + "-memory-failure.json")
            write(evidence, dict(required_bytes=1024**3, passed=False, snapshot=snapshot))
            self.state["memory_failure"] = str(evidence)
            raise MemoryError("Less than1GiB available job/host RAM; stop without retry")
