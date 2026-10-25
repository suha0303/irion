import hashlib
import json
import os
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


def memory_available():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(fields["MemAvailable"].split()[0]) * 1024
    # Job cgroup limits can be tighter than node MemAvailable (v1 and v2).
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if controllers == "":
            folder = Path("/sys/fs/cgroup") / relative.lstrip("/")
            pairs = [("memory.max", "memory.current")]
        elif "memory" in controllers.split(","):
            folder = Path("/sys/fs/cgroup/memory") / relative.lstrip("/")
            pairs = [("memory.limit_in_bytes", "memory.usage_in_bytes")]
        else:
            continue
        mount = Path("/sys/fs/cgroup") if controllers == "" else Path("/sys/fs/cgroup/memory")
        while folder.is_relative_to(mount):
            for limit, used in pairs:
                if (folder / limit).exists() and (folder / used).exists():
                    text = (folder / limit).read_text().strip()
                    if text != "max":
                        available = min(
                            available, max(0, int(text) - int((folder / used).read_text()))
                        )
            if folder == mount:
                break
            folder = folder.parent
    return available


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
        if memory_available() < 1024**3:
            raise MemoryError("Less than1GiB available job/host RAM; stop without retry")
