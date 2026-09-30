import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from x500_storage import active_policy, guard, selected_root

ROOT = Path(__file__).resolve().parents[3]
SIMULATION = ROOT / "simulation"
RESOURCES = ROOT / "Data/generation/pretrain/resources"
HOME = ROOT / "Data/operations/pretrain"
STORE = selected_root()
PACKAGE = RESOURCES / "preprocessing"
ADMISSION = RESOURCES / "admission"
PYTHON = ROOT / "Data/generation/.venv/bin/python"
PRODUCER_MANIFEST = SIMULATION / "artifacts/runtime-manifest.json"
CHOICES = (1, 2, 4, 8, 16)
MAX_FLIGHT_WORKERS = 16  # PX4 instance/port contract: 0..15; CPU counts are unrestricted.
TASK_WALL_SECONDS = 7 * 24 * 3600
COLLECTION_WAIT_BUDGETS = dict(
    flight_wall_s=14400,
    verify_wall_s=7200,
    archive_wall_s=14400,
    archive_verify_wall_s=14400,
    byte_stall_s=3600,
    campaign_wall_s=TASK_WALL_SECONDS,
)
# local-20261015-100232 measured preprocess peak1.401GB per900s run.
# Audit has no measured peak yet;1.5GB is a conservative planning assumption.
CPU_WORKER_BYTES = dict(
    verify=64_000_000, extract=100_000_000, preprocess=1_500_000_000, audit=1_500_000_000
)


def read(path):
    return json.loads(Path(path).read_text())


def service_unit(task, config=None):
    """Resolve the actual unit without rewriting historical task/ownership records."""
    task = Path(task)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", task.name):
        raise ValueError("Invalid task name for service identity")
    if config is None:
        config_path = task / "config.json"
        config = read(config_path) if config_path.exists() else {}
    legacy = "x500-pretrain-" + task.name + ".service"
    current = "drone-pretrain-" + task.name + ".service"
    expected = config.get("service_unit", legacy)
    record = task / "operator/service.json"
    unit = read(record).get("unit", expected) if record.exists() else expected
    if unit not in {legacy, current} or unit != expected:
        raise ValueError("Task service identity mismatch")
    return unit


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def runtime_identity():
    executable = Path(sys.executable).resolve()
    return dict(executable=str(executable), sha256=sha(executable), version=sys.version)


def verify_runtime(config):
    expected = config.get("python_runtime")
    if expected is not None and expected != runtime_identity():
        raise ValueError("Python runtime changed since task preparation; create a new task")


def now():
    return datetime.now(timezone.utc).isoformat()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collector():
    return load("pretrain_collector", ROOT / "Data/generation/tools/collect_long_runs.py")


def producer_manifest():
    """Verify this machine's build against the checked-in source manifest."""
    sys.path.insert(0, str(SIMULATION / "sim/src"))
    from capstone_sim import collection_config

    collection_config.assert_frozen()
    path = collection_config.producer_lock_path()
    if path != PRODUCER_MANIFEST:
        raise ValueError("Current producer runtime manifest required; run ./Data/generation/drone-data setup --simulation")
    record = read(path)
    if any(
        record.get(key) != expected
        for key, expected in {
            "schema": "x500.producer-runtime/1",
            "revision": "X500_CURRENT",
            "configuration": "collection",
            "profile_id": "vtc6_4s4p",
            "source_manifest": "config/producer-sources.json",
        }.items()
    ):
        raise ValueError("Producer runtime identity mismatch; run ./Data/generation/drone-data doctor")
    if not record.get("files"):
        raise ValueError("Producer runtime inventory is empty")
    return record


def producer_sources():
    """Freeze both reviewed sources and machine-specific binaries for one task."""
    record = producer_manifest()
    files = dict(record["files"])
    files["config/producer-sources.json"] = record["source_manifest_sha256"]
    files[str(PRODUCER_MANIFEST.relative_to(SIMULATION))] = sha(PRODUCER_MANIFEST)
    for name, digest in files.items():
        relative = Path(name)
        path = SIMULATION / relative
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe producer inventory path: " + name)
        if (
            any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file()
            or sha(path) != digest
        ):
            raise ValueError("Producer runtime/source changed: " + name)
    return files


def package_imports():
    """Select the hash-locked preprocessing_core/preprocessing_tools package paths."""
    import sys

    sys.dont_write_bytecode = True  # The sealed package must never gain __pycache__ files.
    for p in reversed((PACKAGE / "wrapper", PACKAGE / "ai", PACKAGE / "ai/src")):
        sys.path.insert(0, str(p))


def verify(files):
    for name, digest in files.items():
        path = Path(name)
        if any(p.is_symlink() for p in (path, *path.parents)) or sha(path) != digest:
            raise ValueError("Source/input changed: " + name)


def external(path):
    return guard(path, active_policy())


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def overlap(records):
    events = [(r["started"], 1) for r in records] + [(r["finished"], -1) for r in records]
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


class Budget:
    """Cheap worker checks; parent checks disk/RAM once per second for the whole batch."""

    def __init__(self, seconds=TASK_WALL_SECONDS):
        self.started = time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic() - self.started > self.seconds:
            raise TimeoutError(f"Finite task budget exhausted: {self.seconds}s")


def admission(workers, peak_per_worker, available):
    required = int(workers * peak_per_worker * 1.25 + 256_000_000)
    return dict(
        passed=required <= 8_000_000_000 and available - required >= 4_000_000_000,
        estimated_bytes=required,
        available_bytes=available,
        basis="1.25 * worker peak RSS * workers +256MB; >=4GB free, <=8GB own RAM",
    )


def runs_root(root=None):
    """The frozen simulator owns temporary raw runs; final datasets use STORE."""
    root = Path(root) if root is not None else ROOT
    return root / "simulation/runs" if (root / "simulation").is_dir() else root / "runs"


def pythonpath():
    return os.pathsep.join((str(ROOT / "Data/generation"), str(ROOT / "ai")))
