"""Fresh full-fit LR comparisons; never fine-tune a completed checkpoint."""

import argparse
import datetime
import fcntl
import hashlib
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .control import allocations, completed_result, read, sha, write

LOW, HIGH = 1e-5, 1.0
MIN_IMPROVEMENT = 0.005


def key(rate):
    return format(float(rate), ".12g")


def candidates(item):
    best = item["best"]["lr"]
    factor = item["factor"]
    width = item.get("candidate_width", 2)
    if width > 2:
        # A bounded parallel neighborhood, retaining the original two neighbors
        # before extending and filling geometric midpoints. No final-score input.
        rates = []
        seen = set(item["tested"])
        seen.add(key(best))
        for exponent in (
            -1,
            1,
            2,
            -2,
            0.5,
            -0.5,
            1.5,
            -1.5,
            3,
            -3,
            2.5,
            -2.5,
            0.25,
            -0.25,
            0.75,
            -0.75,
            1.25,
            -1.25,
            1.75,
            -1.75,
        ):
            rate = max(LOW, min(HIGH, best * factor**exponent))
            if key(rate) in seen:
                continue
            rates.append(rate)
            seen.add(key(rate))
            if len(rates) >= width:
                break
        return rates
    return [
        r
        for r in (max(LOW, best / factor), min(HIGH, best * factor))
        if key(r) not in item["tested"]
        and not math.isclose(r, best, rel_tol=1e-10)
    ]


def finish_round(item, results):
    previous = item["best"]["best_validation_loss"]
    for result in results:
        item["tested"][key(result["lr"])] = result
        if (
            result["status"] == "FIT_COMPLETE_STOPPED"
            and result["best_validation_loss"]
            < item["best"]["best_validation_loss"]
        ):
            item["best"] = result
    improvement = (previous - item["best"]["best_validation_loss"]) / max(
        abs(previous), 1e-12
    )
    item["rounds"] += 1
    if improvement >= MIN_IMPROVEMENT:
        item["no_improvement"] = 0
    else:
        item["no_improvement"] += 1
        item["factor"] = math.sqrt(item["factor"])
    if item["no_improvement"] >= 2:
        item["status"] = "LOCAL_NO_MEANINGFUL_IMPROVEMENT"
    elif not candidates(item):
        item["status"] = "LOCAL_CANDIDATES_EXHAUSTED"
    item["last_relative_improvement"] = improvement
    return item


def stopped(root):
    return (root / "STOP").exists() or (root.parent / "STOP").exists()


def worker(args):
    import torch

    from capstone_ai.training import seed_all

    from .fit import (
        Data,
        Diverged,
        backend,
        main_fit,
        model,
        runtime,
        state_hash,
    )

    plan = read(args.plan)
    entry = next(e for e in plan["models"] if e["id"] == args.entry)
    request = read(args.request)
    entry = {**entry, "node": request["node"]}
    out = Path(request["output"])
    out.mkdir(parents=True, exist_ok=False)
    runtime(plan, entry, out)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    seed_all(0)
    initial = state_hash(model(entry))
    if initial != request["initial"]:
        raise ValueError("Fresh initial weights differ from baseline")
    binding = {
        "cache": plan["cache_sha256"],
        "sources": plan["source_sha256"],
        "entry": entry,
        "seed": 0,
        "comparison": "fresh_full_fit_same60epoch_schedule",
        "lr": request["lr"],
    }
    data = Data(Path(plan["inputs"]) / "main", entry)
    try:
        with backend("cuda"):
            main_fit(entry, data, out, request["lr"], binding)
        result = read(out / "result.json")
        if result["initial"] != initial:
            raise ValueError("Initial state changed inside main fit")
        shutil.copy2(
            Path(plan["inputs"]) / "main/scaler.json", out / "scaler.json"
        )
        write(
            out / "integrity.json",
            {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        )
    except Diverged as e:
        # Numerical divergence is LR evidence; input/OOM/runtime errors propagate.
        result = {
            "status": "LR_DIVERGED",
            "lr": request["lr"],
            "reason": str(e),
            "initial": initial,
        }
        write(out / "result.json", result)


def baseline(plan):
    root = Path(plan["baseline"])
    state = read(root / "state.json")
    old = read(root / "plan.json")
    if state["status"] != "TRAINING_COMPLETE_STOPPED":
        receipt = read(Path(plan["handover"]) / "baseline-ready.json")
        if (
            receipt["baseline"] != str(root)
            or receipt["allocation"]["leader"] != plan["allocation"]["leader"]
        ):
            raise ValueError("Baseline handover allocation mismatch")
        for entry in plan["models"]:
            if (
                sha(root / entry["id"] / "main/result.json")
                != receipt["models"][entry["id"]]
            ):
                raise ValueError("Baseline completion changed")
    if (
        old["models"] != plan["models"]
        or old["cache_sha256"] != plan["cache_sha256"]
    ):
        raise ValueError("Baseline data/models changed")
    for name, digest in old["source_sha256"].items():
        if plan["source_sha256"].get(name) != digest:
            raise ValueError(
                "Baseline numerical implementation changed: " + name
            )
    models = {}
    for entry in plan["models"]:
        folder = root / entry["id"] / "main"
        r = completed_result(folder / "result.json")
        integrity = completed_result(folder / "integrity.json")
        for name in (
            "result.json",
            "model.safetensors",
            "history.json",
            "scaler.json",
        ):
            if sha(folder / name) != integrity[name]:
                raise ValueError("Baseline artifact changed")
        if r["status"] != "FIT_COMPLETE_STOPPED":
            raise ValueError("Incomplete baseline model")
        r = {
            **r,
            "folder": str(folder),
            "model_sha256": sha(folder / "model.safetensors"),
        }
        models[entry["id"]] = {
            "best": r,
            "tested": {key(r["lr"]): r},
            "factor": 2.0,
            "rounds": 0,
            "no_improvement": 0,
            "status": "SEARCHING",
        }
    return models


def batch(plan, root, requests, state, phase="search"):
    active = {}
    finished = []
    pending = list(requests)
    try:
        while pending or active:
            if stopped(root):
                raise InterruptedError(
                    "User STOP; preserve checkpoints; no final evaluation"
                )
            if time.time() > state["deadline"]:
                raise TimeoutError(
                    "Scheduler allocation end approaching; not evidence of optimum"
                )
            for node, capacity in [("DIS05", 5), ("DIS06", 4)]:
                used = sum(e["node"] == node for _, e, _, _ in active.values())
                while used < capacity:
                    job = next(iter(pending), None)
                    if job is None:
                        break
                    pending.remove(job)
                    entry = {**job["entry"], "node": node}
                    req = job["request"]
                    request = read(req)
                    request["node"] = node
                    write(req, request)
                    module = (
                        "learning.parallel19.refine"
                        if phase == "search"
                        else "learning.parallel19.final_verify"
                    )
                    argv = [
                        "srun",
                        "--jobid=" + str(plan["allocation"]["leader"]),
                        "--het-group=" + str(0 if node == "DIS05" else 1),
                        "--partition=p02",
                        "--nodelist=" + node,
                        "--nodes=1",
                        "--ntasks=1",
                        "--cpus-per-task=4",
                        "--gres=gpu:1",
                        "--mem=16G",
                        "--exclusive",
                        "--exact",
                        "--kill-on-bad-exit=1",
                        plan["python"],
                        "-B",
                        "-m",
                        module,
                        "--plan",
                        str(root / "plan.json"),
                        "--entry",
                        entry["id"],
                        "--request",
                        str(req),
                    ]
                    if phase == "search":
                        argv.append("--worker")
                    handle = Path(str(req) + ".log").open("xb")
                    p = subprocess.Popen(
                        argv, stdout=handle, stderr=subprocess.STDOUT
                    )
                    active[p.pid] = (p, entry, job, handle)
                    used += 1
            for pid, (p, entry, job, handle) in list(active.items()):
                rc = p.poll()
                if rc is None:
                    continue
                handle.close()
                del active[pid]
                if rc:
                    raise RuntimeError(
                        f"{phase}:{entry['id']}:exit{rc}; no automatic retry"
                    )
                request = read(job["request"])
                result = completed_result(
                    Path(request["output"]) / "result.json"
                )
                if phase == "search" and result["status"] not in {
                    "FIT_COMPLETE_STOPPED",
                    "LR_DIVERGED",
                }:
                    raise ValueError("Invalid trial result")
                finished.append(
                    (entry["id"], {**result, "folder": request["output"]})
                )
            state.update(
                active=[
                    {
                        "model": e["id"],
                        "node": e["node"],
                        "output": read(j["request"])["output"],
                        "pid": p.pid,
                    }
                    for p, e, j, _ in active.values()
                ],
                queued=len(pending),
                updated_at=time.time(),
            )
            write(root / "state.json", state)
            if active:
                time.sleep(2)
    finally:
        for p, _, _, _ in active.values():
            p.terminate()
        for p, _, _, handle in active.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            handle.close()
    return finished


def load_baseline(plan, entry):
    folder = Path(plan["baseline"]) / entry["id"] / "main"
    if not (folder / "integrity.json").exists():
        return None
    r = read(folder / "result.json")
    integrity = read(folder / "integrity.json")
    for name in (
        "result.json",
        "model.safetensors",
        "history.json",
        "scaler.json",
    ):
        if sha(folder / name) != integrity[name]:
            raise ValueError("Baseline artifact changed")
    if r["status"] != "FIT_COMPLETE_STOPPED":
        raise ValueError("Baseline model incomplete")
    r = {
        **r,
        "folder": str(folder),
        "model_sha256": sha(folder / "model.safetensors"),
    }
    return {
        "best": r,
        "tested": {key(r["lr"]): r},
        "factor": 2.0,
        "rounds": 0,
        "no_improvement": 0,
        "status": "SEARCHING",
        "round_results": [],
        "round_open": False,
    }


def baseline_occupancy(plan):
    """Count only the original dispatcher-owned srun processes, not GPU indices."""
    old = read(Path(plan["baseline"]) / "state.json")
    busy = {"DIS05": 0, "DIS06": 0}
    for e in old.get("active", []):
        p = Path("/proc") / str(e["pid"])
        try:
            if p.stat().st_uid != os.getuid():
                raise ValueError("Old worker owner changed")
            state = (
                (p / "stat")
                .read_text(encoding="utf-8")
                .rsplit(")", 1)[1]
                .split()[0]
            )
            cmd = (p / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        if state == "Z":
            continue
        if b"srun" not in cmd or e["model"].encode() not in cmd:
            raise ValueError("Old worker PID identity changed")
        busy[e["node"]] += 1
    return busy


def start_trial(plan, root, entry, request_path, node, phase="search"):
    request = read(request_path)
    request["node"] = node
    write(request_path, request)
    module = (
        "learning.parallel19.refine"
        if phase == "search"
        else "learning.parallel19.final_verify"
    )
    argv = [
        "srun",
        "--jobid=" + str(plan["allocation"]["leader"]),
        "--het-group=" + str(0 if node == "DIS05" else 1),
        "--partition=p02",
        "--nodelist=" + node,
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--gres=gpu:1",
        "--mem=16G",
        "--exclusive",
        "--exact",
        "--kill-on-bad-exit=1",
        plan["python"],
        "-B",
        "-m",
        module,
        "--plan",
        str(root / "plan.json"),
        "--entry",
        entry["id"],
        "--request",
        str(request_path),
    ]
    if phase == "search":
        argv.append("--worker")
    handle = Path(str(request_path) + ".log").open("xb")
    process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT)
    return process, handle


def next_attempt(root, job):
    """Keep interrupted evidence; only incomplete candidates get a fresh attempt."""
    request = read(job["request"])
    old = Path(request["output"])
    if (old / "result.json").exists():
        return
    if not old.exists() and not Path(job["request"] + ".log").exists():
        return
    attempt = job.get("attempt", 1) + 1
    job["attempt"] = attempt
    new = old.parent / (old.name.split("-attempt")[0] + f"-attempt{attempt:02}")
    request["output"] = str(new)
    path = Path(job["request"]).with_name(
        Path(job["request"]).stem.split("-attempt")[0]
        + f"-attempt{attempt:02}.json"
    )
    write(path, request)
    job["request"] = str(path)


def training_records(item):
    rows = [*item.get("tested", {}).values(), *item.get("round_results", [])]
    return {
        r["folder"]: r
        for r in rows
        if r.get("status") == "FIT_COMPLETE_STOPPED"
    }


def development_done(item, development):
    if item["status"] == "WAITING_BASELINE":
        return False
    return all(
        development.get(folder, {}).get("status")
        == "DEVELOPMENT_EVALUATION_COMPLETE"
        for folder in training_records(item)
    )


def enqueue_development(plan, root, models, pending, development):
    for name, item in models.items():
        for folder, result in training_records(item).items():
            if folder in development:
                continue
            source = Path(folder)
            integrity = completed_result(source / "integrity.json")
            for filename in ("model.safetensors", "result.json", "scaler.json"):
                if sha(source / filename) != integrity[filename]:
                    raise ValueError("Candidate artifact changed")
            selected = {
                **result,
                "model_sha256": integrity["model.safetensors"],
            }
            token = hashlib.sha256(
                (folder + selected["model_sha256"]).encode()
            ).hexdigest()[:24]
            selection = root / "development-selections" / f"{name}-{token}.json"
            write(
                selection,
                {
                    "selected": selected,
                    "evaluation_cohort": "development",
                    "final_used": False,
                },
            )
            output = root / "development" / name / token
            request = root / "requests" / f"{name}-development-{token}.json"
            write(
                request,
                {
                    "output": str(output),
                    "selected": selected,
                    "cohort": "development",
                    "selection_manifest": str(selection),
                    "selected_manifest_sha256": sha(selection),
                },
            )
            development[folder] = {
                "status": "PENDING",
                "model": name,
                "lr": selected["lr"],
                "model_sha256": selected["model_sha256"],
                "request": str(request),
            }
            pending.append(
                {
                    "kind": "development",
                    "model": name,
                    "training_folder": folder,
                    "request": str(request),
                    "attempt": 1,
                }
            )


def enqueue_final(root, models, pending, development, state):
    """Freeze each finished search before opening its final cohort; never feed back scores."""
    frozen = state.setdefault("frozen_models", {})
    for name, item in models.items():
        if not item["status"].startswith("LOCAL_") or not development_done(
            item, development
        ):
            continue
        best = item["best"]
        source = Path(best["folder"])
        integrity = completed_result(source / "integrity.json")
        for filename in ("model.safetensors", "result.json", "scaler.json"):
            if sha(source / filename) != integrity[filename]:
                raise ValueError("Selected artifact changed")
        selected = {**best, "model_sha256": integrity["model.safetensors"]}
        if name in frozen:
            if frozen[name] != selected:
                raise ValueError("Per-model final selection changed")
            continue
        selection = root / "final-selections" / f"{name}.json"
        receipt = {
            "model": name,
            "selected": selected,
            "selection": "normal_validation_only",
            "search_status": item["status"],
            "final_test_used_for_selection": False,
        }
        if selection.exists() and read(selection) != receipt:
            raise ValueError("Final selection receipt changed")
        if not selection.exists():
            write(selection, receipt)
        request = root / "requests" / f"{name}-final.json"
        write(
            request,
            {
                "output": str(root / name / "final"),
                "selected": selected,
                "cohort": "final",
                "selection_manifest": str(selection),
                "selected_manifest_sha256": sha(selection),
            },
        )
        state.setdefault("final_requests", {})[name] = {
            "request": str(request),
            "attempt": 1,
        }
        frozen[name] = selected
        pending.append(
            {
                "kind": "final",
                "model": name,
                "request": str(request),
                "attempt": 1,
            }
        )


def accept_final(root, state, job, result):
    chosen = state["frozen_models"][job["model"]]
    if (
        result.get("status") != "FINAL_VERIFICATION_COMPLETE"
        or result.get("evaluation_cohort") != "final"
        or result.get("model_sha256") != chosen["model_sha256"]
        or result.get("final_used_for_selection") is not False
    ):
        raise ValueError("Final evaluation binding mismatch")
    state.setdefault("final_results", {})[job["model"]] = result
    state["final_requests"][job["model"]] = {
        "request": job["request"],
        "attempt": job.get("attempt", 1),
    }
    write(root / "final-results.json", state["final_results"])


def widen_open_rounds(plan, root, models, pending):
    """Add independent LR fits without resubmitting active/completed candidates."""
    widths = plan.get("parallel_lr_candidate_widths", {})
    total = plan.get("parallel_lr_total_candidates")
    if total is not None:
        if type(total) is not int or not 2 <= total <= 9:
            raise ValueError("Invalid total parallel LR width")
        names = [
            name for name in widths if models[name]["status"] == "SEARCHING"
        ]
        if names:
            widths = {
                name: max(2, total // len(names) + (i < total % len(names)))
                for i, name in enumerate(names)
            }
    for name, width in widths.items():
        item = models[name]
        if item["status"] != "SEARCHING":
            continue
        if type(width) is not int or not 2 <= width <= 9:
            raise ValueError("Invalid parallel LR width")
        item["candidate_width"] = width
        marker = f"{item['rounds']}:{width}"
        if (
            not item.get("round_open")
            or item.get("parallel_round_widened") == marker
        ):
            continue
        outstanding = [
            j
            for j in pending
            if j["model"] == name and j.get("kind", "search") == "search"
        ]
        occupied = {key(read(j["request"])["lr"]) for j in outstanding}
        occupied.update(key(r["lr"]) for r in item.get("round_results", []))
        for rate in candidates(item):
            if len(occupied) >= width:
                break
            if key(rate) in occupied:
                continue
            token = key(rate).replace(".", "p")
            number = item["rounds"] + 1
            out = root / name / f"round-{number:02}" / f"parallel-{token}"
            request = (
                root
                / "requests"
                / f"{name}-round{number:02}-parallel-{token}.json"
            )
            if request.exists():
                raise ValueError("Parallel LR request already exists")
            write(
                request,
                {
                    "lr": rate,
                    "initial": item["best"]["initial"],
                    "output": str(out),
                },
            )
            pending.append(
                {"model": name, "request": str(request), "attempt": 1}
            )
            occupied.add(key(rate))
        item["parallel_round_widened"] = marker


def adopted_occupancy(plan, state):
    busy = {"DIS05": 0, "DIS06": 0}
    live = set()
    if plan.get("adopted_allocation") != plan["allocation"]["leader"]:
        return busy, live
    completed = state.setdefault("adopted_complete", [])
    for row in plan.get("adopted_steps", []):
        if row["pid"] in completed:
            continue
        root = Path("/proc") / str(row["pid"])
        try:
            if root.stat().st_uid != os.getuid():
                raise ValueError("Adopted worker owner changed")
            fields = (
                (root / "stat")
                .read_text(encoding="utf-8")
                .rsplit(")", 1)[1]
                .split()
            )
            if fields[19] != row["start"]:
                raise ValueError("Adopted worker PID reused")
            alive = fields[0] != "Z"
            if alive:
                cmd = (root / "cmdline").read_bytes()
                if row["request"].encode() not in cmd:
                    raise ValueError("Adopted command changed")
        except FileNotFoundError:
            alive = False
        if alive:
            busy[row["node"]] += 1
            live.add(row["request"])
        else:
            request = read(row["request"])
            result = completed_result(Path(request["output"]) / "result.json")
            expected = {
                "search": {"FIT_COMPLETE_STOPPED", "LR_DIVERGED"},
                "development": {"DEVELOPMENT_EVALUATION_COMPLETE"},
                "final": {"FINAL_VERIFICATION_COMPLETE"},
            }
            if result["status"] not in expected[row.get("kind", "search")]:
                raise ValueError("Adopted worker failed")
            completed.append(row["pid"])
    return busy, live


def campaign(template):
    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    plan = allocations(read(template))
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=True)
    for group, node in enumerate(("DIS05", "DIS06")):
        os.environ["SLURM_JOB_ID_HET_GROUP_" + str(group)] = str(
            plan["allocation"]["components"][node]
        )
    lock = (root / "controller.lock").open("w", encoding="utf-8")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, digest in plan["source_sha256"].items():
        if sha(Path(plan["code"]) / name) != digest:
            raise ValueError("Code changed")
    input_bindings = [
        (plan["inputs"], plan["cache_sha256"]),
        (plan["final_inputs"], plan["final_cache_sha256"]),
    ]
    required_dev = plan.get("development_required", False)
    if required_dev:
        input_bindings.append(
            (plan["development_inputs"], plan["development_cache_sha256"])
        )
    for location, manifest_sha in input_bindings:
        folder = Path(location)
        if sha(folder / "cache.json") != manifest_sha:
            raise ValueError("Input manifest changed")
        for name, digest in read(folder / "cache.json")["files"].items():
            if sha(folder / name) != digest:
                raise ValueError("Input changed")
    old = read(Path(plan["baseline"]) / "plan.json")
    if (
        old["models"] != plan["models"]
        or old["cache_sha256"] != plan["cache_sha256"]
    ):
        raise ValueError("Different baseline/data")
    for name, digest in old["source_sha256"].items():
        if plan["source_sha256"].get(name) != digest:
            raise ValueError("Baseline numerical code changed")
    ends = [
        x.split("=", 1)[1]
        for x in plan["allocation"]["raw"].split()
        if x.startswith("EndTime=")
    ]
    deadline = (
        min(datetime.datetime.fromisoformat(t).timestamp() for t in ends) - 120
    )
    if (root / "state.json").exists():
        state = read(root / "state.json")
        if state["status"] not in {
            "BUDGET_STOPPED_NOT_CONVERGED",
            "MIGRATION_READY",
        }:
            raise ValueError(
                "Only approved handover or scheduler-budget continuation authorized"
            )
        write(
            root
            / "allocation-history"
            / f"before-{plan['allocation']['leader']}.json",
            state,
        )
        if state["status"] == "BUDGET_STOPPED_NOT_CONVERGED":
            for job in state.get("pending_trials", []):
                next_attempt(root, job)
    else:
        state = {
            "models": {
                e["id"]: {"status": "WAITING_BASELINE"} for e in plan["models"]
            },
            "pending_trials": [],
            "stage": "full_fit_lr_search",
            "final_test_opened": False,
            "created_at": time.time(),
        }
    state.update(
        status="RUNNING",
        deadline=deadline,
        allocation=plan["allocation"],
        active=[],
    )
    state.pop("reason", None)
    write(root / "plan.json", plan)
    write(root / "state.json", state)
    models = state["models"]
    pending = state["pending_trials"]
    development = state.setdefault("development", {})
    widen_open_rounds(plan, root, models, pending)
    streaming = plan.get("final_per_model_when_search_stopped", False)
    if streaming and not required_dev:
        raise ValueError("Streaming final requires development gate")
    if required_dev and state["stage"] != "final_verification":
        state["stage"] = (
            "search_development_and_final"
            if streaming
            else "full_fit_with_development"
        )
    write(root / "development-results.json", development)
    if streaming:
        enqueue_final(root, models, pending, development, state)
    active = {}
    entries = {e["id"]: e for e in plan["models"]}
    try:
        while (
            any(
                x["status"] in {"WAITING_BASELINE", "SEARCHING"}
                for x in models.values()
            )
            or pending
            or active
            or (
                required_dev
                and not all(
                    development_done(m, development) for m in models.values()
                )
            )
        ):
            if stopped(root):
                raise InterruptedError("User STOP; no final evaluation")
            if time.time() > deadline:
                raise TimeoutError(
                    "Scheduler deadline; preserve and reallocate remaining work"
                )
            adopted_busy, adopted_live = adopted_occupancy(plan, state)
            widen_open_rounds(plan, root, models, pending)
            if required_dev:
                enqueue_development(plan, root, models, pending, development)
            for name, item in list(models.items()):
                if item["status"] == "WAITING_BASELINE":
                    found = load_baseline(plan, entries[name])
                    if found:
                        models[name] = item = found
                if (
                    item["status"] == "SEARCHING"
                    and not item.get("round_open")
                    and (
                        not required_dev or development_done(item, development)
                    )
                ):
                    rates = candidates(item)
                    if not rates:
                        item["status"] = "LOCAL_CANDIDATES_EXHAUSTED"
                        continue
                    item["round_open"] = True
                    item["round_results"] = []
                    for i, rate in enumerate(rates):
                        number = item["rounds"] + 1
                        out = (
                            root
                            / name
                            / f"round-{number:02}"
                            / f"candidate-{i + 1}"
                        )
                        request = (
                            root
                            / "requests"
                            / f"{name}-round{number:02}-candidate{i + 1}.json"
                        )
                        write(
                            request,
                            {
                                "lr": rate,
                                "initial": item["best"]["initial"],
                                "output": str(out),
                            },
                        )
                        pending.append(
                            {
                                "model": name,
                                "request": str(request),
                                "attempt": 1,
                            }
                        )
            # Receipts from a prior allocation are collected without retraining.
            for job in list(pending):
                result_path = (
                    Path(read(job["request"])["output"]) / "result.json"
                )
                if (
                    job["request"] in adopted_live
                    or any(j is job for _, _, j, _ in active.values())
                    or not result_path.exists()
                ):
                    continue
                result = read(result_path)
                result = {**result, "folder": str(result_path.parent)}
                if job.get("kind") == "development":
                    if (
                        result["status"] != "DEVELOPMENT_EVALUATION_COMPLETE"
                        or result.get("evaluation_cohort") != "development"
                    ):
                        raise ValueError("Invalid development receipt")
                    record = development[job["training_folder"]]
                    if record["model_sha256"] != result["model_sha256"]:
                        raise ValueError("Development weight binding changed")
                    record.update(status=result["status"], result=result)
                    write(root / "development-results.json", development)
                elif job.get("kind") == "final":
                    accept_final(root, state, job, result)
                else:
                    if result["status"] not in {
                        "FIT_COMPLETE_STOPPED",
                        "LR_DIVERGED",
                    }:
                        raise ValueError("Invalid trial receipt")
                    models[job["model"]]["round_results"].append(result)
                pending.remove(job)
            for name, item in models.items():
                if (
                    item.get("round_open")
                    and not any(j["model"] == name for j in pending)
                    and (
                        not required_dev or development_done(item, development)
                    )
                ):
                    finish_round(item, item["round_results"])
                    item["round_results"] = []
                    item["round_open"] = False
            if streaming:
                enqueue_final(root, models, pending, development, state)
            busy = baseline_occupancy(plan)
            for node, count in adopted_busy.items():
                busy[node] += count
            for node, capacity in [("DIS05", 5), ("DIS06", 4)]:
                used = busy[node] + sum(
                    n == node for _, _, _, n in active.values()
                )
                while used < capacity:
                    ready = [
                        j
                        for j in pending
                        if j["request"] not in adopted_live
                        and not any(a is j for _, _, a, _ in active.values())
                        and (
                            j.get("kind") in {"development", "final"}
                            or not required_dev
                            or development_done(
                                {**models[j["model"]], "round_results": []},
                                development,
                            )
                        )
                    ]
                    if not ready:
                        break
                    job = max(
                        ready,
                        key=lambda j: (
                            {"search": 0, "development": 1, "final": 2}[
                                j.get("kind", "search")
                            ],
                            entries[j["model"]]["compute_seconds"],
                        ),
                    )
                    if job.get("kind") == "final":
                        state["final_test_opened"] = True
                    process, handle = start_trial(
                        plan,
                        root,
                        entries[job["model"]],
                        job["request"],
                        node,
                        phase=job.get("kind", "search"),
                    )
                    active[process.pid] = (process, handle, job, node)
                    used += 1
            for pid, (process, handle, job, node) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                handle.close()
                del active[pid]
                if rc:
                    raise RuntimeError(
                        f"{job['model']} exit{rc}; no error retry"
                    )
                completed_result(
                    Path(read(job["request"])["output"]) / "result.json"
                )
            state.update(
                active=[
                    {
                        "model": j["model"],
                        "kind": j.get("kind", "search"),
                        "node": n,
                        "output": read(j["request"])["output"],
                        "pid": p.pid,
                    }
                    for p, _, j, n in active.values()
                ],
                queued=len(pending) - len(active) - len(adopted_live),
                baseline_busy=busy,
                development_completed=sum(
                    v["status"] == "DEVELOPMENT_EVALUATION_COMPLETE"
                    for v in development.values()
                ),
                development_total=len(development),
                final_completed=len(state.get("final_results", {})),
                final_total=len(state.get("frozen_models", {})),
                updated_at=time.time(),
            )
            write(root / "state.json", state)
            if any(
                x["status"] == "WAITING_BASELINE" for x in models.values()
            ) and not any(busy.values()):
                # A failed original worker must not leave the successor waiting forever.
                since = state.setdefault("missing_baseline_since", time.time())
                if time.time() - since > 60:
                    raise RuntimeError(
                        "Original worker ended without complete baseline receipt"
                    )
            else:
                state.pop("missing_baseline_since", None)
            time.sleep(2)
        frozen = {
            name: {
                **item["best"],
                "model_sha256": sha(
                    Path(item["best"]["folder"]) / "model.safetensors"
                ),
            }
            for name, item in models.items()
        }
        selected = {
            "models": frozen,
            "selection": "normal_validation_only",
            "criteria": {
                "relative_improvement": MIN_IMPROVEMENT,
                "consecutive_rounds": 2,
                "LR_bounds": [LOW, HIGH],
                **(
                    {
                        "parallel_candidate_widths": plan[
                            "parallel_lr_candidate_widths"
                        ]
                    }
                    if plan.get("parallel_lr_candidate_widths")
                    else {}
                ),
            },
            "final_test_used_for_selection": False,
        }
        if not (root / "selected-models.json").exists():
            write(root / "selected-models.json", selected)
        elif read(root / "selected-models.json") != selected:
            raise ValueError("Frozen selected models changed")
        if streaming:
            if state.get("frozen_models") != frozen:
                raise ValueError("Final selection changed after evaluation")
            results = state.get("final_results", {})
            if set(results) != set(frozen) or len(results) != 16:
                raise ValueError("Incomplete streaming final verification")
            write(root / "final-results.json", results)
            state.update(
                status="FINAL_VERIFICATION_COMPLETE_PENDING_RELEASE_CHECK",
                stage="complete",
                active=[],
                queued=0,
            )
            return
        if stopped(root):
            raise InterruptedError("STOP before final")
        state.update(stage="final_verification", final_test_opened=True)
        results = []
        requests = []
        for entry in plan["models"]:
            saved = state.setdefault("final_requests", {}).get(entry["id"])
            req = (
                Path(saved["request"])
                if saved
                else root / "requests" / f"{entry['id']}-final.json"
            )
            if not req.exists():
                write(
                    req,
                    {
                        "output": str(root / entry["id"] / "final"),
                        "selected": frozen[entry["id"]],
                        "selected_manifest_sha256": sha(
                            root / "selected-models.json"
                        ),
                    },
                )
            target = Path(read(req)["output"]) / "result.json"
            if target.exists():
                results.append((entry["id"], read(target)))
                continue
            recovery = saved or {"request": str(req), "attempt": 1}
            next_attempt(root, recovery)
            state["final_requests"][entry["id"]] = recovery
            requests.append({"entry": entry, "request": recovery["request"]})
        write(root / "state.json", state)
        results += batch(plan, root, requests, state, phase="final")
        if len(results) != 16 or any(
            r["status"] != "FINAL_VERIFICATION_COMPLETE" for _, r in results
        ):
            raise RuntimeError("Incomplete final verification")
        write(root / "final-results.json", dict(results))
        state.update(
            status="FINAL_VERIFICATION_COMPLETE_PENDING_RELEASE_CHECK",
            stage="complete",
            active=[],
            queued=0,
        )
    except InterruptedError as e:
        state.update(status="USER_STOPPED", reason=str(e))
        raise
    except TimeoutError as e:
        state.update(status="BUDGET_STOPPED_NOT_CONVERGED", reason=str(e))
        raise
    except BaseException as e:
        state.update(status="FAILED_STOPPED", reason=f"{type(e).__name__}: {e}")
        raise
    finally:
        for process, _, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, handle, _, _ in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            handle.close()
        state.update(active=[], updated_at=time.time())
        write(root / "state.json", state)
        lock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--entry")
    p.add_argument("--request")
    a = p.parse_args()
    if a.worker:
        worker(a)
    else:
        campaign(a.plan)
