import concurrent.futures
import json
import math
import re
import threading
from pathlib import Path, PurePosixPath

import yaml

from .recording import dump, sha, verify
from .runner import ROOT, SCENARIOS, run_flight

CONFIG_KEYS = {"checkpoint", "concurrency", "instance_offset", "cases"}
CASE_KEYS = {"scenario", "seed", "duration", "parent_attempt_id"}
PROTECTED_CHECKPOINTS = {
    "source_lock.json", "runtime-manifest.json", "setup-status.json", "environment.json",
    "battery-fit.json", "propulsion-comparison.json", "mass-inertia-ledger.json",
}


def checkpoint_path(value):
    if not isinstance(value, str):
        raise ValueError("campaign checkpoint requires a relative .json path")
    relative = PurePosixPath(value)
    if (
        not value or relative.is_absolute() or ".." in relative.parts or "\\" in value
        or relative.suffix != ".json" or relative.as_posix() in PROTECTED_CHECKPOINTS
    ):
        raise ValueError("campaign checkpoint must be a separate relative .json below artifacts")
    target = ROOT / "artifacts" / relative
    if any(path.is_symlink() for path in (target, *target.parents)):
        raise ValueError("campaign checkpoint cannot contain symlinks")
    return target


def supported_keys(value, allowed, where):
    unsupported = value.keys() - allowed
    if unsupported:
        raise ValueError(
            f"Unsupported {where} keys: {', '.join(sorted(map(str, unsupported)))}. "
            "Campaigns use the default X500; select another prepared vehicle with "
            "capstone-sim run --vehicle-profile <path>."
        )


def plan(path):
    try:
        cfg = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        raise ValueError("Invalid campaign YAML: " + str(exc)) from exc
    if not isinstance(cfg, dict):
        raise ValueError("campaign requires a YAML mapping")
    supported_keys(cfg, CONFIG_KEYS, "campaign")
    checkpoint_path(cfg.get("checkpoint"))
    concurrency = cfg.get("concurrency", 1)
    instance_offset = cfg.get("instance_offset", 0)
    if type(concurrency) is not int or concurrency not in (1, 2):
        raise ValueError("campaign concurrency must be 1 or 2")
    if type(instance_offset) is not int or not 0 <= instance_offset <= 10 - concurrency:
        raise ValueError("campaign instance range must be within 0..9")
    if not isinstance(cfg.get("cases"), list) or not cfg["cases"]:
        raise ValueError("campaign cases must be a nonempty list")
    for index, case in enumerate(cfg["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"campaign case {index} requires a mapping")
        supported_keys(case, CASE_KEYS, f"campaign case {index}")
        if not isinstance(case.get("scenario"), str) or case["scenario"] not in SCENARIOS:
            raise ValueError(f"campaign case {index} has an unsupported scenario")
        seed = case.get("seed", 17)
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError(f"campaign case {index} seed must be a uint32 integer")
        duration = case.get("duration", 90)
        if (
            isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or not 0 < duration <= 1800
        ):
            raise ValueError(f"campaign case {index} duration must be within (0,1800] seconds")
        parent = case.get("parent_attempt_id")
        if parent is not None and (
            not isinstance(parent, str) or not re.fullmatch(r"[0-9a-f]{32}", parent)
        ):
            raise ValueError(f"campaign case {index} parent_attempt_id must be a run ID")
    seconds = sum(c.get("duration", 90) + 40 for c in cfg["cases"])
    return {
        "cases": len(cfg["cases"]),
        "simulation_seconds": seconds,
        "estimated_bytes": int(seconds * 650000),
        "concurrency": concurrency,
        "vehicle": "x500_reference",
        "config": cfg,
    }


def campaign(path):
    cfg = plan(path)["config"]
    checkpoint = checkpoint_path(cfg["checkpoint"])
    config_record = checkpoint.with_suffix(".config.json")
    digest = sha(path)
    if config_record.exists() and json.loads(config_record.read_text())["sha256"] != digest:
        raise ValueError("campaign config changed; use a new checkpoint name")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    dump(config_record, {"path": str(Path(path).resolve()), "sha256": digest, "config": cfg})
    records = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    concurrency = cfg.get("concurrency", 1)
    instance_offset = cfg.get("instance_offset", 0)
    queue = [
        (i, c)
        for i, c in enumerate(cfg["cases"])
        if records.get(str(i), {}).get("status") != "completed"
        or records.get(str(i), {}).get("contract", {}).get("status") != "passed"
    ]
    # Each worker owns a fixed PX4 instance; each run also has a unique Gazebo partition.
    lock = threading.Lock()

    def worker(instance, cases):
        for i, c in cases:
            old = records.get(str(i), {})
            result = run_flight(
                c["scenario"],
                c.get("seed", 17),
                c.get("duration", 90),
                instance,
                old.get("run_id", c.get("parent_attempt_id")),
            )
            result["run_id"] = Path(result["run"]).name
            result["contract"] = verify(result["run"])
            with lock:
                result["previous_attempts"] = old.get("previous_attempts", []) + (
                    [
                        {
                            "run_id": old["run_id"],
                            "status": old["status"],
                            "reason": old.get("reason"),
                        }
                    ]
                    if old
                    else []
                )
                records[str(i)] = result
                dump(checkpoint, records)
            yield i, result

    def collect(instance, cases):
        return list(worker(instance, cases))

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(collect, instance_offset + n, queue[n::concurrency])
            for n in range(concurrency)
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    return {
        "status": "completed"
        if all(
            r["status"] == "completed" and r["contract"]["status"] == "passed"
            for r in records.values()
        )
        else "partial",
        "completed": sum(r["status"] == "completed" for r in records.values()),
        "total": len(cfg["cases"]),
        "checkpoint": str(checkpoint),
    }
