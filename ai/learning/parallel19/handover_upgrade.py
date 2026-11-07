"""Replace only a dispatcher; adopt its GPU workers and preserve its evidence."""

import argparse
import copy
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .control import allocations, read, sha, write
from .handover import identity


def preserve_final_selections(old, output, snapshot):
    for name, selected in snapshot.get("frozen_models", {}).items():
        source = old / "output/final-selections" / f"{name}.json"
        if read(source)["selected"] != selected:
            raise ValueError("Prior frozen selection changed")
        dest = output / "final-selections" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != sha(source):
            raise ValueError("Conflicting final selection receipt")
        if not dest.exists():
            shutil.copy2(source, dest)


def run(args):
    root = Path(args.root)
    old = Path(args.previous)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "upgrade-state.json").exists():
        raise ValueError("Upgrade already claimed")
    os.environ["SLURM_JOB_ID"] = str(args.job)
    original = identity(args.pid)
    parts = original[0].decode().split("\x00")
    if "learning.parallel19.refine" not in parts or "--plan" not in parts:
        raise ValueError("Wrong dispatcher")
    prior_plan = Path(parts[parts.index("--plan") + 1])
    if prior_plan.parent != old or prior_plan.name not in {
        "plan-template.json",
        "migration-plan.json",
    }:
        raise ValueError("Wrong dispatcher plan")
    allocations(read(old / "output/plan.json"))

    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    paused = False
    started = False
    child = None
    state = {
        "status": "WAITING_UPGRADE_PACKAGE",
        "dispatcher": args.pid,
        "at": time.time(),
    }
    try:
        deadline = time.monotonic() + 15
        while True:
            os.kill(args.pid, signal.SIGSTOP)
            paused = True
            snapshot = read(old / "output/state.json")
            children = {
                int(x)
                for x in (
                    Path("/proc")
                    / str(args.pid)
                    / "task"
                    / str(args.pid)
                    / "children"
                )
                .read_text()
                .split()
            }
            declared = {r["pid"] for r in snapshot.get("active", [])}
            if (
                children == declared
                and snapshot["status"] == "RUNNING"
                and snapshot["stage"]
                in {
                    "full_fit_lr_search",
                    "full_fit_with_development",
                    "search_development_and_final",
                }
            ):
                break
            os.kill(args.pid, signal.SIGCONT)
            paused = False
            if time.monotonic() > deadline:
                raise RuntimeError("No consistent dispatcher boundary found")
            time.sleep(0.05)
        prior = read(old / "output/plan.json")
        if prior.get("adopted_allocation") == args.job and any(
            r["pid"] not in snapshot.get("adopted_complete", [])
            for r in prior.get("adopted_steps", [])
        ):
            raise ValueError(
                "Prior adopted workers must finish before another handover"
            )
        adopted = []
        for entry in snapshot["active"]:
            cmd, start = identity(entry["pid"])
            parts = cmd.decode().split("\x00")
            if (
                parts[0].rsplit("/", 1)[-1] != "srun"
                or "--request" not in parts
            ):
                raise ValueError("Unknown child")
            adopted.append(
                {
                    **entry,
                    "request": parts[parts.index("--request") + 1],
                    "start": start,
                }
            )
        write(root / "previous-state.json", snapshot)
        write(root / "adopted-workers.json", adopted)
        write(root / "upgrade-state.json", state)
        until = time.monotonic() + 1800
        while not (root / "READY.json").exists():
            if (root / "STOP").exists():
                raise InterruptedError("Upgrade cancelled")
            if time.monotonic() > until:
                raise TimeoutError("Upgrade package preparation timeout")
            time.sleep(2)
        ready = read(root / "READY.json")
        template = Path(ready["plan"])
        if sha(template) != ready["sha256"]:
            raise ValueError("Upgrade template changed")
        plan = read(template)
        plan.update(adopted_steps=adopted, adopted_allocation=args.job)
        preserve_final_selections(old, Path(plan["output"]), snapshot)
        write(root / "migration-plan.json", plan)
        snapshot.update(
            status="MIGRATION_READY",
            active=[],
            adopted_complete=[],
            migration_source=str(old / "output"),
        )
        write(Path(plan["output"]) / "state.json", snapshot)
        state.update(status="RUNNING_EVALUATED_SEARCH")
        write(root / "upgrade-state.json", state)
        env = dict(
            os.environ,
            PYTHONPATH=plan["code"] + "/ai:" + plan["code"] + "/ai/src",
        )
        started = True
        child = subprocess.Popen(
            [
                plan["python"],
                "-B",
                "-m",
                "learning.parallel19.refine",
                "--plan",
                str(root / "migration-plan.json"),
            ],
            env=env,
        )
        state["exit"] = child.wait()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=90)
        if paused and identity(args.pid) == original:
            # Successful adoption owns all remaining work. Retire the previous
            # dispatcher via its normal cleanup only after its successor stops.
            if started:
                os.kill(args.pid, signal.SIGTERM)
            os.kill(args.pid, signal.SIGCONT)
        state.update(
            status="SUCCESSOR_EXITED" if started else "OLD_DISPATCHER_RESUMED",
            updated=time.time(),
        )
        write(root / "upgrade-state.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--previous", required=True)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--job", type=int, required=True)
    run(p.parse_args())
