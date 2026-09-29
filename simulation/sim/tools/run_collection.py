"""Explicit Collection producer entry point; check the new lock before each single run."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim/src"))


def check_lock():
    from capstone_sim import collection_config

    collection_config.assert_frozen()
    path = collection_config.producer_lock_path()
    lock = json.loads(path.read_text())
    errors = []
    for name, expected in lock["files"].items():
        p = ROOT / name
        actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        if actual != expected:
            errors.append(name)
    if errors:
        raise ValueError("Collection frozen inputs changed: " + ", ".join(errors))
    return {"status": "passed", "files": len(lock["files"]),
            "lock_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope": "hash check only; not a SITL or physical acceptance"}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check")
    run = sub.add_parser("run")
    run.add_argument("--scenario", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--duration", type=float, required=True)
    run.add_argument("--instance", type=int, required=True)
    run.add_argument("--fault-at-s", type=float)
    args = parser.parse_args()
    try:
        result = check_lock()
        if args.action == "run":
            from capstone_sim.runner import run_flight

            result = run_flight(args.scenario, seed=args.seed, duration=args.duration,
                                instance=args.instance, fault_at_s=args.fault_at_s,
                                configuration="collection", timing=False, capture_interval_ms=5)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in ("passed", "completed") else 4
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
