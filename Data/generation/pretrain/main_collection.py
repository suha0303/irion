"""User-started LR108/main1674 collection; finite batches, checkpoints, no model work."""

import argparse
import fcntl
import hashlib
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from x500_storage import capture, guard

from . import (
    ai19,
    main_storage,
    native_flight,
    pilot45,
    pilot_bridge,
    pilot_parallel,
    training19,
    training19_main,
)
from .common import (
    HOME,
    PACKAGE,
    ROOT,
    STORE,
    collector,
    identity,
    now,
    producer_sources,
    read,
    runtime_identity,
    service_unit,
    sha,
    verify,
    write,
)
from .retry import shared_failure

AUTH = (
    ROOT
    / "ai/reports/collection-hardening-20261108/attempt-01/authorization.json"
)
AUTH_SHA = "3a1e5d7952af7eb38ec0bf1f5afb5cc97200b3413d79a31f4960d18db470933f"
HANDOFF = (
    ROOT
    / "ai/reports/targeted-recovery-20261108/attempt-04/release/data-handoff.json"
)
REPORT = (
    ROOT / "docs/data/reports/main-collection-connection-20261107/attempt-01"
)
CATALOG = ROOT / "Data/operations/training19/catalog"
NAMESPACE = "main1674-20261107-01"
FRESH_AUTH = (
    ROOT
    / "ai/reports/main19-fresh-start-20261107/attempt-01/authorization.json"
)
FRESH_AUTH_SHA = (
    "e01572a1858fa0d66889d884b85c898a713f7ff9caa5673494e7015845e4b448"
)


def plans(scope):
    if scope not in {"main", "lr108"}:
        raise ValueError("Select main or lr108")
    verify({str(ai19.no_links(AUTH)): AUTH_SHA})
    auth = read(AUTH)
    for ref in auth["frozen_plans"].values():
        verify({str(ai19.no_links(ref["path"])): ref["sha256"]})
    main = read(auth["frozen_plans"]["main"]["path"])
    lr = read(auth["frozen_plans"]["LR"]["path"])
    training19_main.approval()
    training19.validate(lr)
    training19_main.validate(main["cases"], lr)
    return auth, main if scope == "main" else lr


def dependencies(path=HANDOFF):
    value = read(ai19.no_links(path))
    if (
        value.get("schema") != "drone-ai.collection19.data-handoff/1"
        or value.get("status") != "READY"
        or value.get("code_frozen") is not True
    ):
        raise ValueError(
            "Independent production AI19 native/aggregate READY handoff required"
        )
    frozen = value["source_freeze"]
    verify({str(ai19.no_links(frozen["path"])): frozen["sha256"]})
    for name, digest in read(frozen["path"])["files"].items():
        if not name.startswith("ai/") or ".." in Path(name).parts:
            raise ValueError("Actual AI source freeze required")
        verify({str(ai19.no_links(ROOT / name)): digest})
    for name, path in value["paths"].items():
        verify({str(ai19.no_links(path)): value["sha256"][name]})
    if set(value["paths"]) != {
        "expected_producer",
        "paired_policy",
        "native_policy",
        "source_manifest",
        "runtime_manifest",
    }:
        raise ValueError("Exact production path aliases required")
    policy = read(value["paths"]["paired_policy"])
    auth, _ = plans("main")
    if (
        policy.get("scope_task") != "MAIN_COLLECTION_CONNECTION_20261107"
        or policy.get("approved_for_collection") is not True
        or policy.get("purpose") != "main_dataset_preparation"
        or policy.get("authorization_sha256") != AUTH_SHA
        or policy.get("frozen_plans") != auth["frozen_plans"]
        or policy.get("duration_profile") != "standard480"
        or policy.get("native_adapter") != "joint-native-consumption-c2-v2"
        or policy.get("fixture_only") is not False
        or any(
            policy.get(k) is not False
            for k in (
                "approved_for_pilot",
                "approved_for_short_cycle",
                "approved_for_final16",
                "authorizes_automatic_start",
                "authorizes_training",
                "posthoc_adjustment",
            )
        )
    ):
        raise ValueError(
            "Pilot or short READY cannot authorize production collection"
        )
    binding = ai19.producer_binding()
    ai19.ai_modules()
    from learning.admission19 import check_producer
    from learning.collection19 import CRITICAL
    from learning.collection19.contract import validate_policy

    if not set(CRITICAL) <= set(read(frozen["path"])["files"]):
        raise ValueError(
            "Complete production AI critical source freeze required"
        )
    check_producer(value["producer"], binding)
    validate_policy(policy, value["paths"]["native_policy"])
    if (binding["source_manifest_sha256"], binding["runtime_sha256"]) != (
        auth["producer"]["source_sha256"],
        auth["producer"]["runtime_sha256"],
    ):
        raise ValueError(
            "Actual production source/runtime differs from authority"
        )
    return value


def validate(task):
    cfg = read(task / "config.json")
    auth, plan = plans(cfg["scope"])
    manifest = read(task / "manifest.json")
    if manifest["cases"] != plan["cases"] or cfg["plan_sha256"] != sha(
        auth["frozen_plans"]["main" if cfg["scope"] == "main" else "LR"]["path"]
    ):
        raise ValueError(
            "Actual immutable approved cases/roles/seeds/groups required"
        )
    expected = dict(
        mode="main19",
        planned_cases=1674 if cfg["scope"] == "main" else 108,
        flight_workers=16,
        requested_real_time_factor=2,
        cpu_workers_total_max=4,
        unit_attempts_max=3,
        nonflight_when_flights_idle=not cfg.get("continuous_refill", False),
        min_available_ram_gb=1,
        use_available_ram=True,
        use_swap=True,
        training_authorized=False,
        fit_authorized=False,
    )
    if any(
        type(cfg.get(k)) is not type(v) or cfg[k] != v
        for k, v in expected.items()
    ):
        raise ValueError(
            "Explicit16/RTF2/CPU4/three-attempt user-start scope required"
        )
    if type(cfg.get("continuous_refill", False)) is not bool:
        raise ValueError("Explicit boolean continuous_refill required")
    if cfg.get("continuous_refill") and cfg.get(
        "max_outstanding_raw_runs"
    ) not in (None, 64):
        raise ValueError(
            "Only explicit unlimited or historical64 raw queue is supported"
        )
    if cfg.get("fresh_start"):
        verify({str(FRESH_AUTH): FRESH_AUTH_SHA})
        if cfg.get("lineage_namespace") != NAMESPACE + ":" + task.name:
            raise ValueError(
                "A fresh campaign needs its own immutable namespace"
            )
        proof = cfg["fresh_start"]
        if proof.get("path") != str(task / "fresh-start.json"):
            raise ValueError("Task-owned fresh-start evidence required")
        verify({proof["path"]: proof["sha256"]})
    elif cfg.get("lineage_namespace") != NAMESPACE:
        raise ValueError("Original production namespace required")
    return cfg, manifest


def resources(cfg, *, scratch=0):
    root = guard(cfg["external_storage"], cfg["storage_policy"])
    snapshot = collector().memory_snapshot()
    snapshot["disk_free_bytes"] = shutil.disk_usage(root).free
    snapshot["local_raw_free_bytes"] = shutil.disk_usage(
        ROOT / "simulation/runs"
    ).free
    if snapshot["local_raw_free_bytes"] < 50_000_000_000:
        raise pilot45.Blocked(
            "Local raw filesystem reserve unavailable; preserve current data"
        )
    if snapshot["available_bytes"] < 1_000_000_000:
        raise pilot45.Blocked("Available RAM below1GB; checkpoint and stop")
    if snapshot["disk_free_bytes"] < 50_000_000_000 + scratch:
        raise pilot45.Blocked(
            "Disk50GB reserve plus bounded scratch unavailable"
        )
    return snapshot


def preflight(task):
    import subprocess

    state_path = task / "operator/state.json"
    previous_state = read(state_path) if state_path.exists() else {}
    if (
        previous_state.get("data_discarded") is True
        or previous_state.get("status") == "USER_CANCELLED_DISCARDED"
    ):
        raise ValueError(
            "Discarded task cannot start again; create a new collection"
        )
    cfg, manifest = validate(task)
    active = subprocess.run(
        [
            "systemctl",
            "--user",
            "list-units",
            "drone-pretrain-*",
            "--state=running,activating,deactivating",
            "--no-pager",
            "--no-legend",
            "--plain",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    own = "drone-pretrain-" + task.name + ".service"
    in_own = (
        Path("/proc/self/cgroup")
        .read_text(encoding="utf-8")
        .rstrip()
        .endswith("/" + own)
    )
    if any(
        line.split()[0] != own or not in_own
        for line in active.stdout.splitlines()
        if line.strip()
    ):
        raise pilot45.Blocked(
            "Existing collection/service cleanup is active; no overlapping campaign"
        )
    verify(
        {
            str(task / "config.json"): (task / "config.sha256")
            .read_text(encoding="utf-8")
            .strip()
        }
    )
    verify(cfg["source_sha256"])
    dependencies(cfg["dependency_handoff"])
    current = ai19.producer_binding()
    if current != cfg["producer"]:
        raise ValueError("Producer changed; no automatic reseal")
    resources(cfg, scratch=read(task / "budget.json")["batch_scratch_bytes"])
    budget = read(task / "budget.json")
    snapshot = resources(cfg)
    warnings = []
    if (
        snapshot["disk_free_bytes"] < budget["required_initial_free_bytes"]
        and not (task / "operator/claimed").exists()
    ):
        if not cfg.get("fresh_start"):
            raise pilot45.Blocked(
                "Planning upper/reserve exceeds selected storage free space; choose a larger storage root before starting"
            )
        warnings.append(
            f"초기 저장 상한 추정 {budget['required_initial_free_bytes'] / 1e9:.1f}GB > 여유 {snapshot['disk_free_bytes'] / 1e9:.1f}GB. "
            + "실제50GB 여유+현재 묶음 임시공간 보호를 유지하며, 부족하면 보존 후 중지합니다."
        )
    native_flight.environment_check()
    pilot_parallel.released()
    return dict(
        status="PASSED_NO_FLIGHT",
        cases=len(manifest["cases"]),
        resources=resources(cfg),
        warnings=warnings,
    )


def case_digest(case):
    return hashlib.sha256(
        json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def retryable(result, attempts, stopped=False, *, binding=None, attempts_max=3):
    """Known independent failed UUIDs get two retries; shared failures always stop."""
    if stopped:
        return False
    reason = str(result.get("reason", "")).lower()
    raw = Path(result.get("run", ""))
    meta = read(raw / "run.json") if (raw / "run.json").exists() else {}
    failure = json.dumps(meta.get("primary_failure", {})).lower() + " " + reason
    if any(
        token in failure
        for token in (
            "disk",
            "no space",
            "memory",
            "source changed",
            "source_lock",
            "producer revision",
            "contract",
            "integrity",
            "protection",
            "cancel",
            "user_stop",
        )
    ):
        result["main_shared_failure"] = failure
        return False
    if (
        result.get("status") != "failed"
        or meta.get("status") != "failed"
        or not raw.name
    ):
        result["main_shared_failure"] = (
            "Missing or ambiguous native failure identity"
        )
        return False
    if meta.get("run_id") != raw.name or any(
        identity(pid) is not None
        for pid in meta.get("owned_process_groups", {}).values()
    ):
        result["main_shared_failure"] = (
            "Uncertain failed UUID or live native owner"
        )
        return False
    if binding:
        try:
            collector().verify_run_source_snapshot(
                dict(
                    protocol="DRONE_COLLECTION",
                    source_hashes=binding["source_hashes"],
                ),
                read(raw / "reference/source_lock.json"),
            )
        except (ValueError, OSError) as exc:
            result["main_shared_failure"] = "Failed source snapshot: " + str(
                exc
            )
            return False
    return attempts < attempts_max


def load_entry(path, case):
    value = read(path)
    if value.get("case") != case:
        raise ValueError("Shared LR/main case identity changed")
    return value


def prior_inventory():
    protected = pilot45.protection()
    if pilot45.REGISTRY.exists():
        with pilot45.REGISTRY.open(encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                protected["records"].append(value)
    return protected


def catalog_path(task, cfg):
    return Path(task) / "catalog" if cfg.get("fresh_start") else CATALOG


def interrupted_raw_identity(task, cfg, entry, attempt):
    """Identify an interrupted owned flight without changing its original outcome."""
    if attempt.get("run_id") or attempt.get("state") != "RUNNING":
        return None
    folder = task / "pilot/execution" / entry["case"]["case_id"]
    request = folder / f"flight-{attempt['number']:02}.request.json"
    process = request.with_suffix(".process.json")
    claim = folder / "socket-claim.json"
    if not all(p.is_file() for p in (request, process, claim)):
        return None
    for path in (request, process, claim):
        ai19.no_links(path)
    job, owner, claimed = read(request), read(process), read(claim)
    rid = claimed.get("run_id", "")
    if not re.fullmatch(r"[0-9a-f]{32}", rid):
        raise ValueError("Invalid interrupted owned run UUID")
    raw = ROOT / "simulation/runs" / rid
    if not raw.exists():
        return None
    raw = ai19.no_links(raw)
    meta_path, scenario_path = (
        raw / "run.json",
        raw / "evaluation/scenario.json",
    )
    meta, scenario = read(meta_path), read(scenario_path)
    case = entry["case"]
    if case not in read(task / "manifest.json")["cases"]:
        raise ValueError("Interrupted request must retain its frozen case/role")
    group = "/" + cfg["service_unit"]
    if (
        job.get("task") != str(task)
        or job.get("case") != case
        or job.get("action") != "flight"
        or job.get("instance") != attempt["instance"]
        or job.get("parent_attempt_id") != attempt["parent_attempt_id"]
        or owner.get("request") != str(request)
        or owner.get("owner_uid") != os.getuid()
        or not owner.get("pid_start_ticks")
        or not claimed.get("owner", {}).get("start")
        or not str(owner.get("cgroup", "")).strip().endswith(group)
        or not str(claimed.get("cgroup", "")).strip().endswith(group)
        or claimed.get("owner", {}).get("uid") != os.getuid()
        or claimed.get("instance") != attempt["instance"]
        or claimed.get("path") != f"/tmp/px4-sock-{attempt['instance']}"
        or meta.get("run_id") != rid
        or meta.get("instance") != attempt["instance"]
        or meta.get("parent_attempt_id") != attempt["parent_attempt_id"]
        or scenario.get("name") != case["scenario"]
        or scenario.get("seed") != case["seed"]
        or scenario.get("operation_variation") != case["operation_variation"]
        or scenario.get("paired_case_id") != case["paired_case_id"]
        or scenario.get("producer_lock_sha256")
        != cfg["producer"]["runtime_sha256"]
    ):
        raise ValueError(
            "Interrupted raw does not bind this exact owned request/case"
        )
    if identity(owner.get("pid")) == owner.get("pid_start_ticks") or identity(
        claimed["owner"]["pid"]
    ) == claimed["owner"].get("start"):
        raise pilot45.Blocked("Previous flight owner is still running")
    evidence = {
        str(p): sha(p)
        for p in (request, process, claim, meta_path, scenario_path)
    }
    return rid, evidence


def fresh_history(binding):
    """Only previously approved identical cases may be collected into a separate campaign."""
    verify({str(FRESH_AUTH): FRESH_AUTH_SHA})
    verify({str(AUTH): AUTH_SHA})
    compatible = read(AUTH).get("recollection_prior_producers", [])
    tasks, run_cases, interrupted = {}, {}, {}
    for config in sorted((HOME / "tasks").glob("*/config.json")):
        cfg = read(config)
        if cfg.get("mode") != "main19":
            continue
        task = config.parent
        validate(task)
        verify(
            {
                str(config): (task / "config.sha256")
                .read_text(encoding="utf-8")
                .strip()
            }
        )
        if cfg.get("owner_uid") != os.getuid() or (
            cfg.get("producer") != binding
            and cfg.get("producer") not in compatible
        ):
            continue
        ledger = read(task / "ledger.json")
        previous = [read(p) for p in ledger["cases"].values()]
        ids = {
            a["run_id"]
            for e in previous
            for a in e["attempts"]
            if a.get("run_id")
        }
        for entry in previous:
            for attempt in entry["attempts"]:
                resolved = interrupted_raw_identity(task, cfg, entry, attempt)
                rid = attempt.get("run_id") or (
                    resolved[0] if resolved else None
                )
                if rid:
                    if (
                        rid in run_cases
                        and run_cases[rid] != entry["case"]["case_id"]
                    ):
                        raise ValueError(
                            "Prior raw belongs to different frozen cases"
                        )
                    run_cases[rid] = entry["case"]["case_id"]
                    ids.add(rid)
                    if resolved:
                        interrupted[rid] = dict(
                            task=str(task),
                            case_id=entry["case"]["case_id"],
                            files=resolved[1],
                            original_state_preserved=True,
                        )
        tasks[task.name] = dict(
            config_sha256=sha(config),
            namespace=cfg["lineage_namespace"],
            manifest_sha256=sha(task / "manifest.json"),
            run_ids=sorted(ids),
        )
    return dict(
        schema="drone-data.main19.fresh-start/1",
        previous_tasks=tasks,
        previous_run_ids=sorted(run_cases),
        previous_run_cases=run_cases,
        interrupted_owned_runs=interrupted,
        authority=dict(path=str(FRESH_AUTH), sha256=FRESH_AUTH_SHA),
        reuse_previous_inputs=False,
    )


def recollection_record(record, history, rows):
    previous = history["previous_tasks"].get(record.get("task"))
    cid = (
        record.get("case_id")
        if previous
        else history["previous_run_cases"].get(record.get("run_id"))
    )
    case = rows.get(cid)
    if case is None or (previous and record.get("cohort") != case["cohort"]):
        return False
    for key in ("seed", "scenario", "paired_case_id", "role"):
        if key in record and record[key] != case[key]:
            return False
    return not previous or all(
        key not in record
        or record[key] == previous["namespace"] + ":" + case[key]
        for key in ("group_id", "retry_family_id")
    )


def create(scope, *, storage_root=STORE, name=None, fresh=False):
    auth, plan = plans(scope)
    spec = dependencies()
    storage_root = Path(storage_root).expanduser().absolute()
    policy = capture(storage_root)
    guard(storage_root, policy)
    if not storage_root.exists():
        raise ValueError("Create/select an existing storage directory")
    pointer = HOME / ("main19-" + scope + "-latest.json")
    if pointer.exists() and not fresh:
        old = Path(read(pointer)["task"])
        validate(old)
        if Path(read(old / "config.json")["external_storage"]) != storage_root:
            raise ValueError(
                "Existing task uses another storage root; specify that task explicitly"
            )
        return old
    name = name or (
        "main19-"
        + scope
        + "-"
        + time.strftime("%Y%m%d-%H%M%S")
        + "-"
        + secrets.token_hex(3)
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("Safe task name required")
    task = pilot45.task_path(HOME / "tasks" / name)
    if task.exists():
        raise ValueError("Existing task/evidence preserved")
    pilot_parallel.released()
    baseline = prior_inventory()
    binding = ai19.producer_binding()
    history = fresh_history(binding) if fresh else None
    rows = {c["case_id"]: c for c in plan["cases"]}
    if fresh:
        # Old raw UUIDs/hashes remain protected. The user explicitly requested
        # a new acquisition of the same frozen conditions, not old raw reuse.
        pilot45.fresh_cases(
            plan["cases"],
            dict(
                records=[
                    r
                    for r in baseline["records"]
                    if not recollection_record(r, history, rows)
                ]
            ),
        )
    entries = {}
    for case in plan["cases"]:
        cached = CATALOG / (case["case_id"] + ".json")
        entry = (
            load_entry(cached, case)
            if cached.exists() and not fresh
            else dict(case=case, attempts=[])
        )
        if any(a["state"] == "RUNNING" for a in entry["attempts"]):
            raise ValueError(
                "Unreconciled previous physical attempt; resume its recorded owner task"
            )
        if entry.get("run") and entry.get("status") != "COMMITTED":
            raise ValueError(
                "Resume the recorded owner task for uncommitted successful raw; do not migrate or refly it"
            )
        if entry.get("commit"):
            verify({entry["commit"]["path"]: entry["commit"]["sha256"]})
            main_storage.committed(entry["commit"]["path"], deep=False)
        elif not entry["attempts"] and not fresh:
            pilot45.fresh_cases([case], baseline)
        entries[case["case_id"]] = entry
    task.mkdir()
    (task / "operator").mkdir()
    if fresh:
        write(task / "fresh-start.json", history)
    budget = storage_budget(len(plan["cases"]))
    if fresh:
        budget["continuous_refill_raw_limit"] = None
    reused = sum(e.get("status") == "COMMITTED" for e in entries.values())
    budget["already_committed_cases"] = reused
    budget["new_cases"] = len(entries) - reused
    remaining_permanent = (
        budget["permanent_planning_upper_bytes"]
        * (len(entries) - reused)
        / len(entries)
    )
    budget["required_initial_free_bytes"] = (
        int(
            max(
                remaining_permanent + budget["aggregate_scratch_bytes"],
                remaining_permanent
                - budget["permanent_planning_upper_bytes"]
                / len(entries)
                * min(16, len(entries) - reused)
                + budget["batch_scratch_bytes"],
            )
        )
        + budget["reserve_bytes"]
    )
    if fresh:
        budget["peak_growth_planning_bytes"] = (
            budget["required_initial_free_bytes"] - budget["reserve_bytes"]
        )
        budget["assumptions"] += (
            " Outstanding raw count is uncapped; actual disk reserve and next-flight scratch control admission."
        )
    cfg = dict(
        mode="main19",
        scope=scope,
        collection_profile="production_common19",
        planned_cases=len(plan["cases"]),
        task_id="MAIN_COLLECTION_CONNECTION_20261107",
        owner_uid=os.getuid(),
        service_unit="drone-pretrain-" + name + ".service",
        python_runtime=runtime_identity(),
        external_storage=str(storage_root),
        storage_policy=policy,
        lineage_namespace=NAMESPACE + ":" + name if fresh else NAMESPACE,
        fresh_start=dict(
            path=str(task / "fresh-start.json"),
            sha256=sha(task / "fresh-start.json"),
        )
        if fresh
        else None,
        flight_workers=16,
        requested_real_time_factor=2,
        cpu_workers_total_max=4,
        export_workers=4,
        io_workers=4,
        workers=4,
        audit_workers=4,
        unit_attempts_max=3,
        min_available_ram_gb=1,
        use_available_ram=True,
        use_swap=True,
        wall_seconds=7 * 24 * 3600,
        parallel16=True,
        capture_timing=False,  # Full per-poll timing traces are diagnostics, not training input.
        nonflight_when_flights_idle=not fresh,
        continuous_refill=fresh,
        max_outstanding_raw_runs=None if fresh else 16,
        stop_on_first_physical_failure=False,
        training_authorized=False,
        fit_authorized=False,
        normalization_authorized_after_user_start=True,
        producer=binding,
        plan_sha256=sha(
            auth["frozen_plans"]["main" if scope == "main" else "LR"]["path"]
        ),
        dependency_handoff=str(task / "dependency-handoff.json"),
        policy_bundle=pilot_bridge.policy_bundle(spec),
        paired_cli_options=spec["paired_cli_options"],
        ai_ready=True,
        sealed=True,
        source_sha256={},
    )
    write(task / "config.json", cfg)
    write(task / "manifest.json", plan)
    write(task / "dependency-handoff.json", spec)
    write(task / "budget.json", budget)
    write(task / "protection.json", baseline)
    # Global prior identities remain excluded from all production uses; current identities retain their approved split.
    owned = {
        a.get("run_id")
        for e in entries.values()
        for a in e["attempts"]
        if a.get("run_id")
    }
    own_tasks = {
        Path(a.get("execution_task", a.get("source_task", ""))).name
        for e in entries.values()
        for a in e["attempts"]
    }
    protected = dict(schema="drone-ai.global-protection/1", entries=[])
    for record in baseline["records"]:
        if (
            record.get("run_id") in owned
            or record.get("task") in own_tasks
            and record.get("case_id") in entries
        ):
            continue
        ids = [
            prefix + str(record[key])
            for key, prefix in (
                ("run_id", "run:"),
                ("parent_attempt_id", "run:"),
                ("paired_case_id", "pair:"),
                ("group_id", "group:"),
                ("raw_sha256", "raw-sha256:"),
            )
            if record.get(key)
            and not (
                fresh
                and key == "paired_case_id"
                and recollection_record(record, history, rows)
            )
        ]
        if ids:
            protected["entries"].append(
                dict(
                    identities=ids,
                    forbidden_uses=[
                        "input_check",
                        "fit",
                        "calibration",
                        "final",
                        "final_test",
                        "final_input_check",
                    ],
                    purpose=record.get("cohort", "preexisting_excluded"),
                )
            )
    write(task / "global-protection.json", protected)
    write(
        task / "expected-producer.json",
        {
            k: v
            for k, v in binding.items()
            if k not in {"source_hashes", "vehicle_profile_revision"}
        },
    )
    for cid, e in entries.items():
        main_storage.durable(task / "cases" / (cid + ".json"), e)
    main_storage.durable(
        task / "ledger.json",
        dict(
            schema="drone-data.main19.ledger/1",
            cases={
                cid: str(task / "cases" / (cid + ".json")) for cid in entries
            },
            physical_attempts=sum(len(e["attempts"]) for e in entries.values()),
        ),
    )
    sources = {
        str(p): sha(p) for p in (ROOT / "Data/generation/pretrain").glob("*.py")
    }
    for path in (
        ROOT / "Data/generation/drone-pretrain",
        ROOT / "Data/generation/tools/collect_long_runs.py",
        ROOT / "Data/generation/tools/collection_support.py",
        ROOT / "Data/generation/pretrain/resources/main19-storage-basis.json",
        ROOT / "ai/x500_storage.py",
        PACKAGE / "PACKAGE.json",
    ):
        sources[str(ai19.no_links(path))] = sha(path)
    sources.update(
        {
            str(PACKAGE / name): digest
            for name, digest in read(PACKAGE / "PACKAGE.json")["files"].items()
        }
    )
    sources.update(
        {str(ROOT / "simulation" / p): h for p, h in producer_sources().items()}
    )
    sources.update(
        {
            str(ROOT / p): h
            for p, h in read(spec["source_freeze"]["path"])["files"].items()
        }
    )
    for p in [
        AUTH,
        HANDOFF,
        Path(spec["source_freeze"]["path"]),
        *(Path(v["path"]) for v in auth["frozen_plans"].values()),
        task / "manifest.json",
        task / "budget.json",
        task / "dependency-handoff.json",
        task / "global-protection.json",
        task / "expected-producer.json",
        *map(Path, spec["paths"].values()),
    ]:
        sources[str(p)] = sha(p)
    if fresh:
        sources[str(FRESH_AUTH)] = FRESH_AUTH_SHA
        sources[str(task / "fresh-start.json")] = sha(task / "fresh-start.json")
    cfg["source_sha256"] = sources
    write(task / "config.json", cfg)
    (task / "config.sha256").write_text(
        sha(task / "config.json") + "\n", encoding="utf-8"
    )
    write(
        task / "operator/state.json",
        dict(
            status="READY_NOT_STARTED",
            stage="prepared",
            total=len(entries),
            done=sum(e.get("status") == "COMMITTED" for e in entries.values()),
            completed=[],
            active=[],
            updated_utc=now(),
        ),
    )
    main_storage.durable(pointer, dict(task=str(task)))
    return task


STORAGE_BASIS = (
    ROOT / "Data/generation/pretrain/resources/main19-storage-basis.json"
)


def storage_budget(count):
    basis = read(STORAGE_BASIS)
    scale = 480 / basis["measured_mission_seconds"]
    mean_export = basis["export_bytes_total"] / basis["measured_raw_runs"]
    point = int((basis["xz_capsule_mean_bytes"] + mean_export) * scale * count)
    permanent = int(
        (
            basis["xz_capsule_max_bytes"] * basis["capsule_allowance_factor"]
            + basis["export_max_bytes"] * basis["export_allowance_factor"]
        )
        * scale
        * count
    )
    scratch = int(
        sum(
            basis[k]
            for k in (
                "raw_bytes_total",
                "gzip_bytes_total",
                "consumer_bytes_total",
                "export_bytes_total",
            )
        )
        * scale
        * basis["scratch_allowance_factor"]
    )
    aggregate = int(
        basis["online_npz_max_bytes"]
        * scale
        * (621 + 108 if count == 1674 else 108)
        * 2
    )
    peak = max(
        permanent + aggregate, permanent - permanent / count * 16 + scratch
    )
    return dict(
        measured_basis=basis,
        mission_seconds=480,
        extrapolation=scale,
        raw_all_cases_extrapolation_bytes=int(
            basis["raw_bytes_total"] / 16 * scale * count
        ),
        permanent_point_estimate_bytes=point,
        permanent_planning_upper_bytes=permanent,
        aggregate_scratch_bytes=aggregate,
        batch_scratch_bytes=scratch,
        local_raw_batch_scratch_bytes=int(
            basis["raw_bytes_total"] * scale * 1.25
        ),
        peak_growth_planning_bytes=int(peak),
        reserve_bytes=50_000_000_000,
        required_initial_free_bytes=int(peak) + 50_000_000_000,
        no_full_raw_accumulation=False,
        storage_policy="single_xz0_write_no_readback_user_20261108",
        storage_size_measured_for_current_policy=False,
        assumptions="New XZ preset0 size is unmeasured; the following XZ3 historical estimate is provisional. Intermediate gzip/consumer copies are now omitted. Pending raw has no count cap, subject to actual disk admission. 180→480 linear scaling; max of4 full regular raw+export xz samples with15% allowance and max16 exports with5%; bounded batch scratch1.25 and combined arrays allowance2. These are planning assumptions, not measured1674 guarantees; rolling actual sizes/free space are recorded.",
    )


def groups(rows):
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(row)
    batch = []
    for members in by_group.values():
        if len(batch) + len(members) > 16:
            yield batch
            batch = []
        batch.extend(members)
    if batch:
        yield batch


def registry_identity(task, cfg, old, record):
    """Only the same frozen production case or its exact counterpart may continue."""
    validate(task)
    previous = HOME / "tasks" / old["task"]
    if not (previous / "config.json").exists():
        return False
    oldcfg = read(previous / "config.json")
    if oldcfg.get("mode") != "main19":
        return False
    old_namespace = oldcfg.get("lineage_namespace")
    namespace = cfg.get("lineage_namespace")
    if not old_namespace or not namespace:
        return False
    _, plan = plans("main")
    rows = {c["case_id"]: c for c in plan["cases"]}
    a, b = rows.get(old.get("case_id")), rows.get(record.get("case_id"))
    if a is None or b is None:
        return False
    if old.get("cohort") != a["cohort"] or record.get("cohort") != b["cohort"]:
        return False
    for event, case, ns in ((old, a, old_namespace), (record, b, namespace)):
        for key in ("seed", "scenario", "role", "paired_case_id"):
            if key in event and event[key] != case[key]:
                return False
        for key in ("group_id", "retry_family_id"):
            if key in event and event[key] != ns + ":" + case[key]:
                return False
    if cfg.get("fresh_start") and old_namespace != namespace:
        proof = read(cfg["fresh_start"]["path"])
        prior = proof["previous_tasks"].get(old["task"])
        same_pair = (
            a["group_id"] == b["group_id"]
            and {a["category"], b["category"]} == {"paired_normal", "anomaly"}
            and all(
                a[k] == b[k]
                for k in (
                    "seed",
                    "cohort",
                    "paired_case_id",
                    "operation_variation",
                )
            )
        )
        return bool(
            prior
            and prior["config_sha256"] == sha(previous / "config.json")
            and prior["namespace"] == old_namespace
            and (a == b or same_pair)
            and not any(
                old.get(k) and old[k] == record.get(k)
                for k in ("run_id", "raw_sha256", "retry_family_id")
            )
        )
    if old_namespace != namespace:
        return False
    if a == b:
        entry = load_entry(task / "cases" / (b["case_id"] + ".json"), b)
        return not old.get("run_id") or old["run_id"] in {
            p.get("run_id") for p in entry["attempts"]
        }
    return (
        a["group_id"] == b["group_id"]
        and {a["category"], b["category"]} == {"paired_normal", "anomaly"}
        and all(
            a[k] == b[k]
            for k in ("seed", "cohort", "paired_case_id", "operation_variation")
        )
        and not any(
            old.get(k) and old[k] == record.get(k)
            for k in ("run_id", "raw_sha256", "retry_family_id")
        )
    )


class Campaign(pilot_parallel.Coordinator):
    """Reuse native children, identity checks and per-run processing from the finite runner."""

    def __init__(self, task):
        self.task = Path(task)
        self.cfg, self.manifest = validate(self.task)
        self.flight_limits = {}
        if self.cfg.get("recovery_authorization"):
            from .main_quality import attempts_max

            approved = read(self.cfg["recovery_authorization"]["path"])["cases"]
            self.flight_limits = {
                c["case_id"]: attempts_max(self.cfg, c)
                for c in self.manifest["cases"]
                if c["case_id"] in approved
            }
        self.catalog = catalog_path(self.task, self.cfg)
        cases_by_id = {c["case_id"]: c for c in self.manifest["cases"]}
        entries = {
            cid: load_entry(Path(path), cases_by_id[cid])
            for cid, path in read(self.task / "ledger.json")["cases"].items()
        }
        self.ledger = dict(
            cases=entries,
            attempts=[
                dict(case_id=cid, **a)
                for cid, e in entries.items()
                for a in e["attempts"]
            ],
        )
        for cid, entry in list(entries.items()):
            shared = self.catalog / (cid + ".json")
            if not shared.exists():
                continue
            newest = load_entry(shared, entry["case"])
            owner = newest.get("last_owner_task") or next(
                (
                    a.get("execution_task")
                    for a in reversed(newest["attempts"])
                    if a.get("execution_task")
                ),
                None,
            )
            if owner == str(self.task):
                continue
            if newest.get("status") == "COMMITTED":
                verify({newest["commit"]["path"]: newest["commit"]["sha256"]})
                main_storage.committed(newest["commit"]["path"], deep=False)
                entries[cid] = newest
            elif newest.get("run") or any(
                a["state"] == "RUNNING" for a in newest["attempts"]
            ):
                raise ValueError(
                    "Another recorded owner has an unfinished case; resume that owner"
                )
            elif len(newest["attempts"]) > len(entry["attempts"]):
                entries[cid] = newest
        self.ledger["attempts"] = [
            dict(case_id=cid, **a)
            for cid, e in entries.items()
            for a in e["attempts"]
        ]
        self.events = queue.SimpleQueue()
        self.active = {}
        self.gazebo_priority_done = set()
        self.errors = []
        self.denied = {
            e.get("run_id")
            for e in read(self.task / "protection.json")["records"]
        }
        self.batch_ids = set(entries)
        self.saved = {}
        self.saved_ledger = None
        self.case_paths = {
            cid: str(self.task / "cases" / (cid + ".json")) for cid in entries
        }
        self.budget = read(self.task / "budget.json")
        self.resource_lock = threading.Lock()
        self.resource_snapshot = None
        self.resource_sample_time = 0.0
        self.spec = dependencies(self.cfg["dependency_handoff"])
        self.state = dict(
            status="RUNNING",
            stage="collection",
            task=self.task.name,
            pid=os.getpid(),
            pid_start_ticks=identity(os.getpid()),
            owner_uid=os.getuid(),
            invocation_id=os.environ.get("INVOCATION_ID"),
            started=time.time(),
            service_unit=service_unit(self.task),
            max_active_flights=0,
            max_active_nonflight=0,
            completed=[],
            active=[],
            total=len(entries),
            training_authorized=False,
            whole_task_retry=False,
        )

    def save(self):
        for cid in self.batch_ids:
            e = self.ledger["cases"][cid]
            encoded = json.dumps(e, sort_keys=True)
            if self.saved.get(cid) == encoded:
                continue
            main_storage.durable(self.task / "cases" / (cid + ".json"), e)
            main_storage.durable(self.catalog / (cid + ".json"), e)
            self.saved[cid] = encoded
        summary = dict(
            physical_attempts=len(self.ledger["attempts"]),
            failed_physical_attempts=sum(
                a["state"] == "FAILED_PRESERVED"
                for a in self.ledger["attempts"]
            ),
            counts=dict(
                Counter(
                    e.get("status", "PENDING")
                    for e in self.ledger["cases"].values()
                )
            ),
        )
        if summary != self.saved_ledger:
            main_storage.durable(
                self.task / "ledger.json",
                dict(
                    schema="drone-data.main19.ledger/1",
                    cases=self.case_paths,
                    **summary,
                ),
            )
            self.saved_ledger = summary

    def child_resources(self):
        # All lane supervisors are threads in this process. Share one fresh
        # /proc scan per second instead of repeating it for every flight/CPU job.
        with self.resource_lock:
            if (
                self.resource_snapshot is None
                or time.monotonic() - self.resource_sample_time >= 1
            ):
                self.resource_snapshot = resources(self.cfg)
                self.resource_sample_time = time.monotonic()
            return self.resource_snapshot

    def child(self, request, log):
        return pilot45.child(
            self.task,
            request,
            log,
            on_progress=self.events.put,
            resource_snapshot=self.child_resources,
        )

    def tick(self, flights, cpu):
        while not self.events.empty():
            event = self.events.get()
            entry = self.ledger["cases"][event["case_id"]]
            event.update(
                kind=entry["case"]["kind"],
                environment=entry["case"]["environment"],
                attempt=entry.get("processing_attempts", 1)
                if event.get("action") in {"prepare", "audit"}
                else len(entry["attempts"]),
                attempt_limit=self.flight_limit(entry["case"])
                if event.get("action") == "flight"
                else 3,
            )
            self.active[event["case_id"]] = event
            self.prioritize_controller(event)
        ids = {j["case"]["case_id"] for j in [*flights.values(), *cpu.values()]}
        self.active = {k: v for k, v in self.active.items() if k in ids}
        progress = []
        targets = self.task / "operator/aggregate-targets.json"
        for scope, target in read(targets).items() if targets.exists() else []:
            source = Path(target) / "progress.json"
            if source.exists():
                progress.append(dict(read(source), dataset_scope=scope))
        self.state["aggregate_progress"] = progress
        busy = set(ids)
        pair_groups = {
            j["case"]["group_id"]
            for j in cpu.values()
            if j.get("phase") == "pair"
        }
        waiting = dict(prepare=0, pair=0, commit=0)
        for cid, entry in self.ledger["cases"].items():
            if (
                cid in busy
                or entry["case"]["group_id"] in pair_groups
                or not entry.get("run")
                or entry.get("status")
                in {"COMMITTED", "UNAVAILABLE", "FAILED_PRESERVED"}
            ):
                continue
            if entry.get("status") != "COMPLETE":
                waiting["prepare"] += 1
            elif entry["case"]["category"] == "standalone_normal" or entry.get(
                "pair_receipt"
            ):
                waiting["commit"] += 1
            else:
                waiting["pair"] += 1
        self.state.update(
            postprocess_waiting=waiting, raw_queue_waiting=sum(waiting.values())
        )
        permanent = sum(
            e.get("permanent_bytes", 0) for e in self.ledger["cases"].values()
        )
        accepted = sum(
            e.get("status") == "COMMITTED"
            for e in self.ledger["cases"].values()
        )
        budget = self.budget
        self.state["storage_projection"] = dict(
            actual_committed_cases=accepted,
            actual_permanent_bytes=permanent,
            measured_remaining_point_bytes=int(
                permanent / accepted * (len(self.ledger["cases"]) - accepted)
            )
            if accepted
            else None,
            initial_permanent_planning_upper_bytes=budget.get(
                "permanent_planning_upper_bytes"
            ),
            initial_peak_growth_planning_bytes=budget.get(
                "peak_growth_planning_bytes"
            ),
            reserve_bytes=50_000_000_000,
        )
        self.state.update(
            updated_utc=now(),
            active=list(self.active.values()),
            active_flights=len(flights),
            active_nonflight=len(cpu),
            done=sum(
                e.get("status") == "COMMITTED"
                for e in self.ledger["cases"].values()
            ),
            failed=sum(
                e.get("status") in {"FAILED_PRESERVED", "UNAVAILABLE"}
                for e in self.ledger["cases"].values()
            ),
            physical_attempts=len(self.ledger["attempts"]),
            failed_physical_attempts=sum(
                a["state"] == "FAILED_PRESERVED"
                for a in self.ledger["attempts"]
            ),
            completed_physical_attempts=sum(
                a["state"].startswith("COMPLETED")
                for a in self.ledger["attempts"]
            ),
            quality_rejected_attempts=sum(
                bool(a.get("quality_rejection"))
                for a in self.ledger["attempts"]
            ),
            not_started=sum(
                not e["attempts"] for e in self.ledger["cases"].values()
            ),
            unavailable_cases=sum(
                e.get("status") == "UNAVAILABLE"
                for e in self.ledger["cases"].values()
            ),
            max_active_flights=max(
                self.state["max_active_flights"], len(flights)
            ),
            max_active_nonflight=max(
                self.state["max_active_nonflight"], len(cpu)
            ),
            resources=self.child_resources(),
        )
        write(self.task / "operator/state.json", self.state)

    def prioritize_controller(self, event):
        """Lower only this run's Gazebo group, leaving PX4/control at normal priority."""
        target = self.cfg.get("gazebo_nice")
        rid = event.get("producer_run_id")
        if (
            target is None
            or event.get("action") != "flight"
            or not isinstance(rid, str)
            or not re.fullmatch(r"[0-9a-f]{32}", rid)
            or rid in self.gazebo_priority_done
            or (event.get("producer_px4_boot_s") or 0) < 20
        ):
            return
        if type(target) is not int or target != 3:
            raise ValueError(
                "Only the explicitly selected Gazebo nice3 policy is supported"
            )
        raw = ROOT / "simulation/runs" / rid
        meta = read(raw / "run.json")
        group = meta.get("owned_process_groups", {}).get("gazebo")
        if type(group) is not int or group <= 0:
            return
        result = dict(
            run_id=rid,
            gazebo_group=group,
            target_nice=target,
            changed_threads=[],
            observed_utc=now(),
        )
        todo, seen = [group], set()
        try:
            while todo:
                pid = todo.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                proc = Path("/proc") / str(pid)
                if (
                    proc.stat().st_uid != os.getuid()
                    or os.getpgid(pid) != group
                    or not (proc / "cgroup")
                    .read_text(encoding="utf-8")
                    .rstrip()
                    .endswith("/" + self.cfg["service_unit"])
                ):
                    raise ValueError(
                        "Gazebo priority target is outside the owned service/group"
                    )
                todo.extend(
                    map(
                        int,
                        (proc / "task" / str(pid) / "children")
                        .read_text(encoding="utf-8")
                        .split(),
                    )
                )
                for thread in (proc / "task").iterdir():
                    tid = int(thread.name)
                    before = os.getpriority(os.PRIO_PROCESS, tid)
                    if before < target:
                        os.setpriority(os.PRIO_PROCESS, tid, target)
                        result["changed_threads"].append(
                            dict(
                                tid=tid,
                                before=before,
                                after=os.getpriority(os.PRIO_PROCESS, tid),
                            )
                        )
            result["status"] = "APPLIED"
        except (OSError, ValueError) as exc:
            result.update(
                status="PRIORITY_ADJUSTMENT_INCOMPLETE", reason=str(exc)
            )
        main_storage.durable(
            self.folder(self.ledger["cases"][event["case_id"]]["case"])
            / (rid + "-gazebo-priority.json"),
            result,
        )
        self.gazebo_priority_done.add(rid)

    def stopped(self):
        return (self.task / "operator/stop-request.json").exists()

    def begin_flight(self, case, instance):
        # Preflight verifies the frozen sources once before the owned service runs.
        # Source edits require a stopped service and a new task, not per-run hashing.
        if case not in self.manifest["cases"]:
            raise ValueError("Unplanned production case")
        job = super().begin_flight(case, instance)
        e = self.ledger["cases"][case["case_id"]]
        e["last_owner_task"] = str(self.task)
        e["attempts"][-1]["execution_task"] = str(self.task)
        if len(e["attempts"]) > 1 and e["attempts"][-2].get(
            "quality_rejection"
        ):
            e["processing_attempts"] = 0
            e.pop("status", None)
        from .main_quality import recovery_entry

        recovery = recovery_entry(self.cfg, case)
        if recovery and recovery["mode"] == "reflight":
            e["processing_attempts"] = 0
            e.pop("status", None)
        self.ledger["attempts"][-1].update(e["attempts"][-1])
        self.save()
        return job

    def flight_limit(self, case):
        return self.flight_limits.get(case["case_id"], case["attempts_max"])

    def begin_processing(self, case):
        e = self.ledger["cases"][case["case_id"]]
        n = e.get("processing_attempts", 0) + 1
        if n > 3:
            raise ValueError(
                "Processing maximum3 reached; retain same successful raw"
            )
        e["processing_attempts"] = n
        self.save()
        physical = len(e["attempts"])
        label = f"physical-{physical:02}-attempt-{n:02}"
        output = (
            Path(self.cfg["external_storage"])
            / "processed/main19"
            / self.task.name
            / case["case_id"]
            / f"physical-{physical:02}"
            / f"attempt-{n:02}"
        )
        e["archive_cache"] = str(
            self.folder(case) / f"physical-{physical:02}-archive-cache.json"
        )
        selection = dict(
            case_id=self.cfg["lineage_namespace"] + ":" + case["case_id"],
            path=e["run"],
            run_id=Path(e["run"]).name,
            role=case["role"],
            cohort=case["cohort"],
            purpose=case["purpose"],
            seed=case["seed"],
            group_id=self.cfg["lineage_namespace"] + ":" + case["group_id"],
            retry_family_id=self.cfg["lineage_namespace"]
            + ":"
            + case["retry_family_id"],
            paired_case_id=case["paired_case_id"],
            previous_attempts=[
                a["run_id"] for a in e["attempts"][:-1] if a.get("run_id")
            ],
            execution_task=self.task.name,
        )
        self.save()
        request, log = self.request(
            case,
            "prepare",
            "prepare-" + label,
            selection=selection,
            output=str(output),
            archive_cache=e["archive_cache"],
        )
        audit, audit_log = self.request(
            case, "audit", "audit-" + label, output=str(output)
        )
        return dict(
            case=case,
            request=request,
            log=log,
            audit=audit,
            audit_log=audit_log,
            output=output,
            selection=selection,
        )

    def native_batch(self, rows, pool):
        pending = [
            c
            for c in rows
            if not self.ledger["cases"][c["case_id"]].get("run")
            and len(self.ledger["cases"][c["case_id"]]["attempts"])
            < self.flight_limit(c)
            and self.ledger["cases"][c["case_id"]].get("status")
            != "FAILED_PRESERVED"
        ]
        running = {}
        free = set(range(16))
        shared = None
        while pending or running:
            if self.stopped() or shared:
                pending.clear()
            while pending and free:
                c = pending.pop(0)
                i = min(free)
                free.remove(i)
                job = self.begin_flight(c, i)
                running[pool.submit(self.child, job["request"], job["log"])] = (
                    job
                )
            self.tick(running, {})
            if not running:
                break
            done, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
            for f in done:
                j = running.pop(f)
                c = j["case"]
                e = self.ledger["cases"][c["case_id"]]
                try:
                    success = self.finish_flight(j, f.exception())
                    if not success and not self.stopped():
                        pending.append(c)
                except Exception as exc:
                    e["failure_reason"] = f"{type(exc).__name__}: {exc}"
                    # Only a known completed or failed UUID is an independent run outcome.
                    a = e["attempts"][-1]
                    if a.get("result", {}).get("main_shared_failure"):
                        shared = pilot45.Blocked(
                            a["result"]["main_shared_failure"]
                        )
                    elif a.get("state") == "FAILED_PRESERVED" and a.get(
                        "run_id"
                    ):
                        e["status"] = "FAILED_PRESERVED"
                    else:
                        shared = exc
                free.add(j["instance"])
                self.save()
        if shared:
            raise shared

    def finish_preparation(self, j, error):
        e = self.ledger["cases"][j["case"]["case_id"]]
        try:
            if error is not None:
                raise error
            self.finish_processing(j)
            if not (e["online_available"] and e["run_available"]):
                e["status"] = "UNAVAILABLE"
        except Exception as exc:
            e["failure_reason"] = f"{type(exc).__name__}: {exc}"
            failures = []
            for request in (j["request"], j["audit"]):
                p = Path(request).with_suffix(".failure.json")
                if p.exists():
                    failures.append(read(p))
            quality = next(
                (
                    f
                    for f in failures
                    if f.get("failure_scope") == "quality_rejection"
                ),
                None,
            )
            if quality:
                a = e["attempts"][-1]
                a["quality_rejection"] = quality["quality_rejection"]
                for flat in self.ledger["attempts"]:
                    if (
                        flat["case_id"] == j["case"]["case_id"]
                        and flat["number"] == a["number"]
                    ):
                        flat.update(a)
                e.setdefault("rejected_raw", []).append(
                    dict(
                        run=e.pop("run"),
                        quality_rejection=a["quality_rejection"],
                        processing_attempts=e["processing_attempts"],
                    )
                )
                e["status"] = (
                    "QUALITY_REJECTED_PENDING_REFLIGHT"
                    if len(e["attempts"]) < self.flight_limit(j["case"])
                    else "UNAVAILABLE"
                )
            elif (
                any(f.get("failure_scope") == "shared" for f in failures)
                or shared_failure(exc)
                or (
                    isinstance(exc, pilot45.Blocked)
                    and not (
                        str(exc).startswith("Case child exit")
                        and any(
                            f.get("schema")
                            == "drone-data.main19.child-failure/1"
                            and f.get("failure_scope") == "unit"
                            for f in failures
                        )
                    )
                )
            ):
                raise pilot45.Blocked(e["failure_reason"]) from exc
            elif e["processing_attempts"] < 3:
                pass
            else:
                e["status"] = "UNAVAILABLE"
        self.save()

    def process_batch(self, rows, pool):
        pending = [
            c
            for c in rows
            if self.ledger["cases"][c["case_id"]].get("run")
            and self.ledger["cases"][c["case_id"]].get("status")
            not in {"COMPLETE", "COMMITTED", "UNAVAILABLE"}
            and self.ledger["cases"][c["case_id"]].get("processing_attempts", 0)
            < 3
        ]
        running, shared = {}, None
        while pending or running:
            if self.stopped() or shared:
                pending.clear()
            while pending and len(running) < 4:
                job = self.begin_processing(pending.pop(0))
                running[pool.submit(self.process, job)] = job
            self.tick({}, running)
            if not running:
                break
            done, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                job = running.pop(future)
                try:
                    self.finish_preparation(job, future.exception())
                    entry = self.ledger["cases"][job["case"]["case_id"]]
                    if (
                        entry.get("run")
                        and entry.get("status")
                        not in {"COMPLETE", "COMMITTED", "UNAVAILABLE"}
                        and entry.get("processing_attempts", 0) < 3
                    ):
                        pending.append(job["case"])
                except Exception as exc:
                    shared = exc
                self.save()
        if shared:
            raise shared

    def postprocess_batch(self, rows, pool):
        controls = [c for c in rows if c["category"] == "paired_normal"]
        for normal in controls:
            members = [c for c in rows if c["group_id"] == normal["group_id"]]
            if all(
                self.ledger["cases"][c["case_id"]].get("pair_receipt")
                for c in members
            ):
                continue
            if not all(
                self.ledger["cases"][c["case_id"]].get("status") == "COMPLETE"
                for c in members
            ):
                for c in members:
                    e = self.ledger["cases"][c["case_id"]]
                    if e.get("status") == "COMPLETE":
                        e.update(
                            status="UNAVAILABLE",
                            failure_reason="Paired member not available; pair denominator retained",
                        )
                self.save()
                continue
            anomaly = next(c for c in members if c["category"] == "anomaly")
            request, log = self.request(
                normal,
                "pair",
                "pair",
                normal_id=normal["case_id"],
                anomaly_id=anomaly["case_id"],
            )
            try:
                self.child(request, log)
                pair = read(read(request)["receipt"])["pair_receipt"]
                for c in members:
                    self.ledger["cases"][c["case_id"]]["pair_receipt"] = pair
            except Exception as exc:
                failure = Path(request).with_suffix(".failure.json")
                if (
                    shared_failure(exc)
                    or failure.exists()
                    and read(failure).get("failure_scope") == "shared"
                ):
                    raise
                for c in members:
                    self.ledger["cases"][c["case_id"]].update(
                        status="UNAVAILABLE",
                        failure_reason="Independent pair verification: "
                        + str(exc),
                    )
            self.save()
        eligible = [
            c
            for c in rows
            if self.ledger["cases"][c["case_id"]].get("status") == "COMPLETE"
        ]
        pending = [
            c
            for c in eligible
            if self.ledger["cases"][c["case_id"]].get("storage_attempts", 0) < 3
        ]
        running = {}
        shared = None
        while pending or running:
            if self.stopped() or shared:
                pending.clear()
            while pending and len(running) < 4:
                c = pending.pop(0)
                e = self.ledger["cases"][c["case_id"]]
                e["storage_attempts"] = e.get("storage_attempts", 0) + 1
                self.save()
                request, log = self.request(
                    c, "commit", f"commit-{e['storage_attempts']:02}", entry=e
                )
                job = dict(case=c, request=request, log=log)
                running[pool.submit(self.child, request, log)] = job
            self.tick({}, running)
            if not running:
                break
            done, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
            for f in done:
                job = running.pop(f)
                e = self.ledger["cases"][job["case"]["case_id"]]
                try:
                    f.result()
                    result = read(read(job["request"])["receipt"])
                    e.update(
                        status="COMMITTED",
                        commit=result["commit"],
                        producer=self.cfg["producer"],
                    )
                    committed = read(result["commit"]["path"])
                    if committed.get("artifact"):
                        e["permanent_bytes"] = committed["artifact"][
                            "bytes"
                        ] + sum(
                            Path(p).stat().st_size
                            for p in committed["export_files"]
                        )
                except Exception as exc:
                    failure = Path(job["request"]).with_suffix(".failure.json")
                    e["failure_reason"] = str(exc)
                    if (
                        shared_failure(exc)
                        or failure.exists()
                        and read(failure).get("failure_scope") == "shared"
                    ):
                        shared = exc
                    elif e["storage_attempts"] < 3:
                        pending.append(job["case"])
                    else:
                        e["status"] = "UNAVAILABLE"
                self.save()
        if shared:
            raise shared

    def write_index(self):
        index = self.task / "exports.jsonl"
        with index.with_suffix(".tmp").open("w", encoding="utf-8") as stream:
            for c in self.manifest["cases"]:
                e = self.ledger["cases"][c["case_id"]]
                lineage = [
                    "pair:" + c["paired_case_id"],
                    "group:"
                    + self.cfg["lineage_namespace"]
                    + ":"
                    + c["group_id"],
                    "retry:"
                    + self.cfg["lineage_namespace"]
                    + ":"
                    + c["retry_family_id"],
                ]
                attempts = [
                    dict(
                        index=a["number"],
                        run_id=a.get("run_id"),
                        seed=c["seed"],
                        parent_attempt_id=a["parent_attempt_id"],
                        status="completed"
                        if a["state"].startswith("COMPLETED")
                        else "failed"
                        if a["state"] == "FAILED_PRESERVED"
                        else "running",
                        lineage=lineage,
                        failure_reason=a.get("error")
                        or e.get("failure_reason"),
                    )
                    for a in e["attempts"]
                ]
                for projected, original in zip(attempts, e["attempts"]):
                    if original.get("quality_rejection"):
                        projected.update(
                            status="quality_rejected",
                            physical_status="completed",
                            quality_rejection=original["quality_rejection"],
                        )
                ledger = self.task / "case-ledgers" / (c["case_id"] + ".json")
                main_storage.durable(
                    ledger,
                    dict(
                        schema="drone-ai.collection19.case-ledger/1",
                        case_id=c["case_id"],
                        accepted_run_id=Path(e["run"]).name
                        if e.get("status") == "COMMITTED"
                        else None,
                        attempts=attempts,
                        **(
                            {
                                "recovery_authorization": e[
                                    "recovery_authorization"
                                ]
                            }
                            if e.get("recovery_authorization") is not None
                            else {}
                        ),
                    ),
                )
                row = dict(
                    schema="drone-ai.collection19.index-entry/1",
                    case_id=c["case_id"],
                    status="accepted"
                    if e.get("status") == "COMMITTED"
                    else "unavailable"
                    if e.get("run") or e.get("status") == "UNAVAILABLE"
                    else "failed",
                    case_ledger=dict(path=str(ledger), sha256=sha(ledger)),
                    failure_reason=e.get(
                        "failure_reason",
                        "Not completed in this finite campaign",
                    ),
                )
                if row["status"] == "accepted":
                    if e.get("pair_receipt"):
                        row["pair_evidence"] = read(e["pair_receipt"])[
                            "native_evidence"
                        ]
                    row.update(
                        export_root=e["output"],
                        audit=dict(
                            path=e["audit_receipt"],
                            sha256=sha(e["audit_receipt"]),
                        ),
                        commit=e["commit"],
                    )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        index.with_suffix(".tmp").replace(index)
        return index

    def continuous_collect(self):
        """One coordinator owns all mutations; children only execute bounded jobs."""
        rows = [
            case for batch in groups(self.manifest["cases"]) for case in batch
        ]
        entries = self.ledger["cases"]
        grouped = {}
        for case in rows:
            grouped.setdefault(case["group_id"], []).append(case)
        terminal = {"COMMITTED", "UNAVAILABLE", "FAILED_PRESERVED"}
        flights, cpu, free = {}, {}, set(range(16))
        shared = None
        self.state["stage"] = "continuous-collection-and-preparation"
        limit = self.cfg["max_outstanding_raw_runs"]

        def shared_error(job, error):
            evidence = Path(job["request"]).with_suffix(".failure.json")
            detail = read(evidence) if evidence.exists() else {}
            unit_exit = (
                str(error).startswith("Case child exit")
                and detail.get("schema") == "drone-data.main19.child-failure/1"
                and detail.get("failure_scope") == "unit"
            )
            return (
                shared_failure(error)
                or detail.get("failure_scope") == "shared"
                or isinstance(error, pilot45.Blocked)
                and not unit_exit
            )

        with (
            ThreadPoolExecutor(max_workers=16) as flight_pool,
            ThreadPoolExecutor(max_workers=4) as cpu_pool,
        ):
            while True:
                self.batch_ids = set()
                busy = {
                    j["case"]["case_id"]
                    for j in [*flights.values(), *cpu.values()]
                }
                busy_groups = {
                    j["case"]["group_id"]
                    for j in cpu.values()
                    if j["phase"] == "pair"
                }
                if shared is not None:
                    self.state.update(
                        status="DRAINING",
                        stage="error-draining",
                        reason=f"{type(shared).__name__}: {shared}",
                    )
                admitting = not self.stopped() and shared is None
                if admitting:
                    # Do not leave a successful control waiting forever for a terminal partner.
                    for members in grouped.values():
                        if len(members) != 2 or any(
                            c["case_id"] in busy for c in members
                        ):
                            continue
                        if any(
                            entries[c["case_id"]].get("status")
                            in {"UNAVAILABLE", "FAILED_PRESERVED"}
                            for c in members
                        ):
                            for c in members:
                                e = entries[c["case_id"]]
                                if e.get("status") == "COMPLETE" and not e.get(
                                    "pair_receipt"
                                ):
                                    e.update(
                                        status="UNAVAILABLE",
                                        failure_reason="Paired member unavailable; denominator retained",
                                    )
                                    self.batch_ids.add(c["case_id"])
                    # Commit first to release temporary disk, then pair-check, then prepare.
                    candidates = []
                    for order, c in enumerate(rows):
                        cid, e = c["case_id"], entries[c["case_id"]]
                        if cid in busy or e.get("status") in terminal:
                            continue
                        if e.get("status") == "COMPLETE":
                            if c["category"] == "standalone_normal" or e.get(
                                "pair_receipt"
                            ):
                                if e.get("storage_attempts", 0) < 3:
                                    candidates.append((0, order, "commit", c))
                            elif (
                                c["category"] == "paired_normal"
                                and c["group_id"] not in busy_groups
                            ):
                                if all(
                                    entries[m["case_id"]].get("status")
                                    == "COMPLETE"
                                    for m in grouped[c["group_id"]]
                                ):
                                    candidates.append((1, order, "pair", c))
                        elif (
                            e.get("run") and e.get("processing_attempts", 0) < 3
                        ):
                            candidates.append((2, order, "prepare", c))
                    if self.cfg.get(
                        "defer_nonflight_until_flights_complete"
                    ) and (
                        flights
                        or any(
                            not e.get("run")
                            and e.get("status") not in terminal
                            and len(e["attempts"])
                            < self.flight_limit(e["case"])
                            for e in entries.values()
                        )
                    ):
                        candidates = []
                    for _, _, phase, c in sorted(candidates):
                        if len(cpu) >= 4:
                            break
                        cid, e = c["case_id"], entries[c["case_id"]]
                        self.batch_ids.add(cid)
                        if phase == "prepare":
                            job = self.begin_processing(c)
                            future = cpu_pool.submit(self.process, job)
                        else:
                            if phase == "pair":
                                members = grouped[c["group_id"]]
                                anomaly = next(
                                    m
                                    for m in members
                                    if m["category"] == "anomaly"
                                )
                                request, log = self.request(
                                    c,
                                    "pair",
                                    "pair",
                                    normal_id=cid,
                                    anomaly_id=anomaly["case_id"],
                                )
                            else:
                                e["storage_attempts"] = (
                                    e.get("storage_attempts", 0) + 1
                                )
                                self.save()
                                request, log = self.request(
                                    c,
                                    "commit",
                                    f"commit-{e['storage_attempts']:02}",
                                    entry=e,
                                )
                            job = dict(case=c, request=request, log=log)
                            future = cpu_pool.submit(self.child, request, log)
                        cpu[future] = dict(job, phase=phase)
                        busy.add(cid)
                    outstanding = len(flights) + sum(
                        bool(e.get("run")) and e.get("status") not in terminal
                        for e in entries.values()
                    )
                    pending = [
                        c
                        for c in (rows if free else ())
                        if c["case_id"] not in busy
                        and not entries[c["case_id"]].get("run")
                        and entries[c["case_id"]].get("status") not in terminal
                        and len(entries[c["case_id"]]["attempts"])
                        < self.flight_limit(c)
                    ]
                    pending.sort(
                        key=lambda c: not bool(
                            entries[c["case_id"]]["attempts"]
                        )
                    )
                    for c in pending:
                        if not free or (
                            limit is not None and outstanding >= limit
                        ):
                            break
                        try:
                            resources(
                                self.cfg,
                                scratch=self.budget["batch_scratch_bytes"],
                            )
                        except pilot45.Blocked as exc:
                            if "Disk50GB reserve" not in str(exc):
                                raise
                            self.state["admission_wait_reason"] = str(exc)
                            break
                        else:
                            self.state.pop("admission_wait_reason", None)
                        lane = min(free)
                        self.batch_ids.add(c["case_id"])
                        job = self.begin_flight(c, lane)
                        free.remove(lane)
                        flights[
                            flight_pool.submit(
                                self.child, job["request"], job["log"]
                            )
                        ] = job
                        outstanding += 1
                    self.state["raw_queue_waiting"] = outstanding - len(flights)
                    self.state["raw_queue_limit"] = limit
                self.save()
                self.tick(flights, cpu)
                futures = set(flights) | set(cpu)
                if not futures:
                    break
                done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in flights:
                        job = flights.pop(future)
                        c = job["case"]
                        e = entries[c["case_id"]]
                        self.batch_ids.add(c["case_id"])
                        try:
                            self.finish_flight(job, future.exception())
                        except Exception as exc:
                            e["failure_reason"] = f"{type(exc).__name__}: {exc}"
                            attempt = e["attempts"][-1]
                            if attempt.get("result", {}).get(
                                "main_shared_failure"
                            ):
                                shared = pilot45.Blocked(
                                    attempt["result"]["main_shared_failure"]
                                )
                            elif attempt.get(
                                "state"
                            ) == "FAILED_PRESERVED" and attempt.get("run_id"):
                                e["status"] = "FAILED_PRESERVED"
                            else:
                                shared = exc
                        free.add(job["instance"])
                    else:
                        job = cpu.pop(future)
                        c, phase = job["case"], job["phase"]
                        e = entries[c["case_id"]]
                        self.batch_ids.add(c["case_id"])
                        try:
                            if phase == "prepare":
                                self.finish_preparation(job, future.exception())
                            else:
                                future.result()
                                result = read(read(job["request"])["receipt"])
                                if phase == "pair":
                                    for member in grouped[c["group_id"]]:
                                        entries[member["case_id"]][
                                            "pair_receipt"
                                        ] = result["pair_receipt"]
                                        self.batch_ids.add(member["case_id"])
                                else:
                                    verify(
                                        {
                                            result["commit"]["path"]: result[
                                                "commit"
                                            ]["sha256"]
                                        }
                                    )
                                    e.update(
                                        status="COMMITTED",
                                        commit=result["commit"],
                                        producer=self.cfg["producer"],
                                    )
                                    committed = read(result["commit"]["path"])
                                    e["permanent_bytes"] = committed[
                                        "artifact"
                                    ]["bytes"] + sum(
                                        Path(p).stat().st_size
                                        for p in committed["export_files"]
                                    )
                        except Exception as exc:
                            if phase == "prepare" or shared_error(job, exc):
                                shared = exc
                            elif phase == "pair":
                                for member in grouped[c["group_id"]]:
                                    entries[member["case_id"]].update(
                                        status="UNAVAILABLE",
                                        failure_reason="Independent pair verification: "
                                        + str(exc),
                                    )
                                    self.batch_ids.add(member["case_id"])
                            elif e.get("storage_attempts", 0) >= 3:
                                e.update(
                                    status="UNAVAILABLE",
                                    failure_reason=str(exc),
                                )
                    self.save()
                if shared and not flights and not cpu:
                    break
            self.tick({}, {})
        if shared:
            raise shared

    def run(self):
        if self.cfg.get("continuous_refill"):
            self.continuous_collect()
        else:
            with (
                ThreadPoolExecutor(max_workers=16) as flight_pool,
                ThreadPoolExecutor(max_workers=4) as cpu_pool,
            ):
                batches = list(groups(self.manifest["cases"]))
                for n, rows in enumerate(batches, 1):
                    if self.stopped():
                        break
                    if all(
                        self.ledger["cases"][c["case_id"]].get("status")
                        == "COMMITTED"
                        for c in rows
                    ):
                        continue
                    verify(self.cfg["source_sha256"])
                    self.batch_ids = {c["case_id"] for c in rows}
                    self.state.update(
                        batch=n, batches=len(batches), stage="collection"
                    )
                    resources(
                        self.cfg,
                        scratch=read(self.task / "budget.json")[
                            "batch_scratch_bytes"
                        ],
                    )
                    for _ in range(3):
                        self.native_batch(rows, flight_pool)
                        if self.stopped():
                            break
                        self.state["stage"] = "preprocess-and-independent-audit"
                        self.process_batch(rows, cpu_pool)
                        if self.stopped() or not any(
                            self.ledger["cases"][c["case_id"]].get("status")
                            == "QUALITY_REJECTED_PENDING_REFLIGHT"
                            for c in rows
                        ):
                            break
                    if self.stopped():
                        break
                    self.state["stage"] = "pair-and-durable-retirement"
                    self.postprocess_batch(rows, cpu_pool)
                    self.tick({}, {})
        index = self.write_index()
        all_ready = all(
            e.get("status") == "COMMITTED"
            for e in self.ledger["cases"].values()
        )
        if all_ready and not self.stopped():
            case = self.manifest["cases"][0]
            previous = list(self.folder(case).glob("aggregate-*.request.json"))
            aggregate_attempt = len(previous) + 1
            if aggregate_attempt > 3:
                raise ValueError(
                    "Maximum3 explicit aggregate attempts reached; preserve all datasets"
                )
            request, log = self.request(
                case,
                "aggregate",
                f"aggregate-{aggregate_attempt:02}",
                index=str(index),
                aggregate_attempt=aggregate_attempt,
            )
            self.state["stage"] = "bounded-aggregate-and-independent-audit"
            self.tick({}, {})
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.batch(pool, [dict(case=case, request=request, log=log)])
            self.state["status"] = "MAIN19_INPUTS_PASSED_TERMINATING"
        else:
            self.state["status"] = (
                "USER_STOPPED"
                if self.stopped()
                else "MAIN19_INCOMPLETE_STOPPED"
            )
        self.tick({}, {})


def check_aggregate_result(result, scope, index):
    if (
        result.get("passed") is not True
        or result.get("actual_data_verified") is not True
        or result.get("scope") != scope
        or result.get("counts", {}).get("planned_cases")
        != (1674 if scope == "main" else 108)
    ):
        raise ValueError(
            "Production dataset aggregate lost actual audit or frozen denominator"
        )
    for key in ("receipt", "independent_audit"):
        ref = result[key]
        verify({str(ai19.no_links(ref["path"])): ref["sha256"]})
        check = read(ref["path"])
        if (
            check.get("passed") is not True
            or check.get("actual_data_verified") is not True
            or check.get("scope") != scope
            or check.get("index") != dict(path=str(index), sha256=sha(index))
        ):
            raise ValueError("Sealed dataset input/audit receipt mismatch")


def aggregate_scope(task, cfg, scope, job, aggregate):
    """Reuse a sealed successful scope; only its failed sibling needs another attempt."""
    checkpoint = task / "operator" / ("aggregate-" + scope + "-complete.json")
    context = dict(
        config_sha256=sha(task / "config.json"), index_sha256=sha(job["index"])
    )
    if checkpoint.exists():
        saved = read(ai19.no_links(checkpoint))
        if saved.get("context") != context or saved.get("scope") != scope:
            raise ValueError(
                "Prior aggregate source/index identity changed; preserve evidence"
            )
        verify(saved["dataset_files"])
        check_aggregate_result(saved["aggregate"], scope, job["index"])
        return dict(
            scope=scope,
            output=saved["output"],
            aggregate=saved["aggregate"],
            reused=True,
        )
    output = (
        Path(cfg["external_storage"])
        / "datasets/main19"
        / task.name
        / f"aggregate-{job.get('aggregate_attempt', 1):02}"
        / scope
    )
    ai19.no_links(output)
    targets = task / "operator/aggregate-targets.json"
    known = read(targets) if targets.exists() else {}
    known[scope] = str(output)
    write(targets, known)
    if output.exists():
        raise ValueError(
            "Preserve previous aggregate; explicit recovery required"
        )
    result = aggregate(
        job["index"],
        output,
        scope=scope,
        producer=read(task / "expected-producer.json"),
        protection=read(task / "global-protection.json"),
        policy_bundle=cfg["policy_bundle"],
    )
    check_aggregate_result(result, scope, job["index"])
    files = {
        str(ai19.no_links(p)): sha(p) for p in output.rglob("*") if p.is_file()
    }
    main_storage.durable(
        checkpoint,
        dict(
            scope=scope,
            output=str(output),
            aggregate=result,
            context=context,
            dataset_files=files,
        ),
    )
    return dict(scope=scope, output=str(output), aggregate=result, reused=False)


def execute_case(request):
    job = read(request)
    task = pilot45.task_path(job["task"])
    cfg, manifest = validate(task)
    verify(
        {
            str(task / "config.json"): (task / "config.sha256")
            .read_text(encoding="utf-8")
            .strip()
        }
    )
    # Retain this child's task/config binding; full source verification belongs to
    # service preflight, before any lane starts under the immutable-source policy.
    if job["case"] not in manifest["cases"]:
        raise ValueError("Only approved production cases")
    handoff = cfg["dependency_handoff"]
    verify({handoff: cfg["source_sha256"][handoff]})
    spec = read(
        handoff
    )  # Full source/runtime validation already ran in service preflight.
    action = job["action"]
    if action == "prepare":
        from .main_views import prepare

        result = prepare(
            job["selection"],
            job["output"],
            task / "expected-producer.json",
            job["archive_cache"],
            task / "global-protection.json",
            policy_bundle=cfg["policy_bundle"],
            case=job["case"],
            producer_binding=cfg["producer"],
        )
    elif action == "audit":
        ai19.ai_modules()
        from learning.collection19 import audit_case

        output = Path(job["output"])
        prepared = read(output / "receipt.json")
        result = audit_case(
            output,
            producer=read(task / "expected-producer.json"),
            protection=read(task / "global-protection.json"),
            policy_bundle=cfg["policy_bundle"],
            evidence_root=prepared["run"]["path"],
            assessment_ref=dict(
                path=str(output / "assessment.json"),
                sha256=sha(output / "assessment.json"),
            ),
        )
        if (
            result.get("passed") is not True
            or result.get("actual_data_verified") is not True
        ):
            raise ValueError(
                "Actual production per-case independent audit required"
            )
        result.update(
            export_receipt_sha256=sha(output / "receipt.json"),
            run_id=prepared["run"]["run_id"],
            online_available=prepared["online"]["available"],
            run_available=prepared["run_health"]["status"] == "available",
        )
    elif action == "pair":
        # The existing official native verifier runs while both raw members still exist.
        paths = read(task / "ledger.json")["cases"]
        entries = {
            cid: read(paths[cid])
            for cid in (job["normal_id"], job["anomaly_id"])
        }
        # pilot_bridge accepts the same runtime representation without changing production schemas.
        normal, anomaly = entries[job["normal_id"]], entries[job["anomaly_id"]]
        directory = task / "pilot/pairs" / normal["case"]["group_id"]
        directory.mkdir(parents=True, exist_ok=True)
        native = directory / "native.json"
        if not native.exists():
            import subprocess

            argv = [
                s.format(
                    normal_run=normal["run"],
                    anomaly_run=anomaly["run"],
                    output=str(native),
                )
                for s in spec["pair_verifier_argv"]
            ]
            subprocess.run(argv, cwd=ROOT, check=True, timeout=3600)
        ai19.ai_modules()
        from learning.history19.pilot_connector import project_pair_receipt

        projected = project_pair_receipt(
            native,
            policy_path=spec["paths"]["paired_policy"],
            normal_run=normal["run"],
            anomaly_run=anomaly["run"],
        )
        projected.update(
            schema="drone-ai.pilot-pair-check/1",
            producer=read(task / "expected-producer.json"),
            paired_policy_sha256=sha(spec["paths"]["paired_policy"]),
            pair_id=normal["case"]["paired_case_id"],
            native_evidence=dict(path=str(native), sha256=sha(native)),
        )
        write(directory / "projection.json", projected)
        result = dict(pair_receipt=str(directory / "projection.json"))
    elif action == "commit":
        entry = job["entry"]
        commit = main_storage.commit(
            task,
            entry,
            Path(cfg["external_storage"])
            / "capsules/main19"
            / task.name
            / entry["case"]["case_id"],
            {
                r.get("run_id")
                for r in read(task / "protection.json")["records"]
            },
        )
        result = dict(commit=commit)
    elif action == "aggregate":
        ai19.ai_modules()
        from learning.collection19 import aggregate

        scopes = [cfg["scope"]] + (["lr108"] if cfg["scope"] == "main" else [])
        results = [
            aggregate_scope(task, cfg, scope, job, aggregate)
            for scope in scopes
        ]
        result = dict(
            status="FINITE_DATASET_INPUTS_CHECKED",
            results=results,
            models_started=False,
        )
    else:
        raise ValueError(
            "No model/training/calibration/performance action exists"
        )
    write(job["receipt"], result)


def run_local_assessment_failure(exc):
    """A complete producer assessment can reject one run without a shared fault."""
    prefix = "Producer assessment failed: "
    if not isinstance(exc, ValueError) or not str(exc).startswith(prefix):
        return False
    try:
        assessment = json.loads(str(exc)[len(prefix) :])
    except (ValueError, TypeError):
        return False
    if (
        not isinstance(assessment, dict)
        or assessment.get("schema") != "drone.run-assessment/1"
        or assessment.get("instance_isolation_passed") is not True
    ):
        return False
    errors = assessment.get("errors")
    # Known input-quality checks may fail together. Unknown source/contract,
    # integrity and isolation errors still stop the campaign.
    known = (
        "normal health eligibility",
        "normal mission model envelope",
        "Joint native consumption: Incomplete actual reference consumption coverage",
    )
    flags = ("contract_passed", "physical_gate_passed", "missing_data_passed")
    return (
        isinstance(errors, list)
        and all(error in known for error in errors)
        and all(type(assessment.get(key)) is bool for key in flags)
        and assessment["contract_passed"] == (not errors)
        and any(assessment[key] is False for key in flags)
    )


def case_entry(request):
    try:
        execute_case(request)
    except BaseException as exc:
        from .main_quality import QualityRejected

        scope = (
            "quality_rejection"
            if isinstance(exc, QualityRejected)
            else "unit"
            if run_local_assessment_failure(exc)
            else "shared"
            if shared_failure(exc)
            or isinstance(exc, pilot45.Blocked)
            or any(
                t in str(exc).lower()
                for t in (
                    "source",
                    "integrity",
                    "producer",
                    "contract",
                    "protection",
                    "globally excluded",
                    "policy",
                    "storage",
                    "symlink",
                )
            )
            else "unit"
        )
        failure = dict(
            schema="drone-data.main19.child-failure/1",
            failure_scope=scope,
            error_type=type(exc).__name__,
            errno=getattr(exc, "errno", None),
            failure_reason=str(exc),
        )
        if isinstance(exc, QualityRejected):
            failure["quality_rejection"] = exc.ref
        main_storage.durable(
            Path(request).with_suffix(".failure.json"), failure
        )
        raise


def worker(task):
    task = pilot45.task_path(task)
    cfg = read(task / "config.json")
    if (
        cfg.get("owner_uid") != os.getuid()
        or cfg.get("mode") != "main19"
        or cfg.get("service_unit") != "drone-pretrain-" + task.name + ".service"
    ):
        raise ValueError("Owned registered production task required")
    try:
        worker_body(task)
    except BaseException as exc:
        path = task / "operator/state.json"
        state = read(path) if path.exists() else dict(task=task.name)
        main_storage.durable(
            task
            / "operator"
            / ("start-or-run-failure-" + str(time.time_ns()) + ".json"),
            dict(
                prior_state=state,
                error_type=type(exc).__name__,
                reason=str(exc),
            ),
        )
        state.update(
            status="BLOCKED_STOPPED",
            reason=f"{type(exc).__name__}: {exc}",
            finished_utc=now(),
            next_stage_authorized=False,
        )
        write(path, state)
        raise


def worker_body(task):
    task = pilot45.task_path(task)
    preflight(task)
    with ExitStack() as stack:
        for p in (
            HOME / "pretrain.lock",
            ROOT / "Data/operations/collection.lock",
            CATALOG.parent / "catalog.lock",
        ):
            p.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(p.open("a", encoding="utf-8"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (task / "operator/claimed").exists():
            request = task / "operator/resume-request.json"
            if not request.exists() or read(request).get(
                "config_sha256"
            ) != sha(task / "config.json"):
                raise ValueError("Explicit owned resume required")
            request.rename(
                request.with_name(
                    "consumed-resume-" + str(time.time_ns()) + ".json"
                )
            )
            stop = task / "operator/stop-request.json"
            if stop.exists():
                stop.rename(
                    stop.with_name(
                        "consumed-stop-" + str(time.time_ns()) + ".json"
                    )
                )
        else:
            (task / "operator/claimed").write_text(now(), encoding="utf-8")
        record_worker_identity(task)
        campaign = Campaign(task)
        try:
            campaign.run()
        except BaseException as exc:
            campaign.state.update(
                status="BLOCKED_STOPPED", reason=f"{type(exc).__name__}: {exc}"
            )
            raise
        finally:
            campaign.save()
            campaign.state.update(
                finished_utc=now(), next_stage_authorized=False
            )
            write(task / "operator/state.json", campaign.state)
            write(
                task / "completion.json",
                dict(
                    campaign.state,
                    counts=dict(
                        Counter(
                            e.get("status", "PENDING")
                            for e in campaign.ledger["cases"].values()
                        )
                    ),
                    failure_denominator=len(campaign.ledger["cases"]),
                    model_forward_steps=0,
                    GPU_jobs=0,
                ),
            )


def resume(task):
    import subprocess

    from .cli import start

    task = pilot45.task_path(task)
    state_path = task / "operator/state.json"
    previous_state = read(state_path) if state_path.exists() else {}
    if (
        previous_state.get("data_discarded") is True
        or previous_state.get("status") == "USER_CANCELLED_DISCARDED"
    ):
        raise ValueError(
            "Discarded inputs cannot be resumed; create a new collection"
        )
    cfg, _ = validate(task)
    unit, pid, props = pilot45.service_identity(task)
    if pid or props.get("ActiveState") in {"activating", "deactivating"}:
        raise ValueError("Do not resume a running task")
    if reconcile_termination(task)["status"] == "MAIN19_INPUTS_PASSED_STOPPED":
        raise ValueError("Completed finite task; do not regenerate")
    verify(cfg["source_sha256"])
    pilot_parallel.released()
    campaign = Campaign(task)
    for cid, e in campaign.ledger["cases"].items():
        for a in e["attempts"]:
            if a["state"] != "RUNNING":
                continue
            folder = task / "pilot/execution" / cid
            request = folder / f"flight-{a['number']:02}.request.json"
            job = read(request)
            receipt = Path(job["receipt"])
            if not receipt.exists():
                stdout = request.with_suffix(".stdout.json")
                if not stdout.exists():
                    raise ValueError(
                        "Unknown physical outcome preserved; no blind reflight"
                    )
                result = native_flight.read_result(stdout)
                raw = Path(result["run"])
                meta = read(raw / "run.json")
                if any(
                    identity(p) is not None
                    for p in meta.get("owned_process_groups", {}).values()
                ):
                    raise ValueError("Native owner process still alive")
                write(
                    receipt,
                    dict(
                        result,
                        reconciled_stdout=dict(
                            path=str(stdout), sha256=sha(stdout)
                        ),
                    ),
                )
            try:
                campaign.finish_flight(
                    dict(
                        case=e["case"],
                        number=a["number"],
                        instance=a["instance"],
                        request=request,
                    ),
                    None,
                )
            except pilot45.Blocked:
                if not a.get("run_id") or a["state"] == "RUNNING":
                    raise
    campaign.save()
    write(
        task / "operator/resume-request.json",
        dict(
            config_sha256=sha(task / "config.json"),
            owner_uid=os.getuid(),
            requested_utc=now(),
        ),
    )
    subprocess.run(
        ["systemctl", "--user", "reset-failed", unit],
        capture_output=True,
        check=False,
    )
    start(task, pilot_resume=True)


def user_start(scope, *, storage_root=STORE, name=None):
    from .cli import start

    lock_path = CATALOG.parent / "submission.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        task = create(scope, storage_root=storage_root, name=name, fresh=True)
        if (task / "operator/claimed").exists() or (
            task / "operator/service.json"
        ).exists():
            print(
                "기존 작업:",
                task,
                "· status/watch 또는 명시 resume 사용; 자동 재시작 없음",
            )
            return task
        start(task)
    print("진행 조회: ./Data/generation/drone-pretrain main19 watch", task)
    return task


def record_worker_identity(task):
    """The worker records its own invocation before any physical dispatch."""
    cfg = read(task / "config.json")
    invocation = os.environ.get("INVOCATION_ID", "")
    group = Path("/proc/self/cgroup").read_text(encoding="utf-8").strip()
    unit = service_unit(task, cfg)
    if not re.fullmatch(r"[0-9a-f]{32}", invocation) or not group.endswith(
        "/" + unit
    ):
        raise ValueError("Actual owned systemd worker invocation required")
    record = dict(
        schema="drone-data.main19.worker-identity/1",
        task=str(task),
        owner_uid=os.getuid(),
        unit=unit,
        config_sha256=sha(task / "config.json"),
        pid=os.getpid(),
        pid_start_ticks=identity(os.getpid()),
        invocation_id=invocation,
        cgroup=group,
        started_utc=now(),
    )
    snapshot = (
        task / "operator/invocations" / invocation / "worker-identity.json"
    )
    if snapshot.exists():
        raise ValueError(
            "Invocation already recorded; preserve its original identity"
        )
    main_storage.durable(snapshot, record)
    main_storage.durable(task / "operator/worker-identity.json", record)


def cleanup_finished(task, report):
    cfg = read(task / "config.json")
    if cfg.get("owner_uid") != os.getuid() or cfg.get("mode") != "main19":
        return
    state = read(task / "operator/state.json")
    proof = dict(
        schema="drone-data.main19.cleanup-exit/1",
        task=str(task),
        owner_uid=os.getuid(),
        config_sha256=sha(task / "config.json"),
        invocation_id=os.environ.get("INVOCATION_ID"),
        service_result=report["service_result"],
        worker_exit_code=os.environ.get("EXIT_CODE"),
        worker_exit_status=os.environ.get("EXIT_STATUS"),
        cleanup_sha256=sha(task / "operator/socket-cleanup.json"),
        observed_utc=now(),
        passed=False,
    )
    try:
        if (
            report.get("status") != "CLEANUP_COMPLETE"
            or report.get("errors")
            or report.get("service_result") != "success"
        ):
            raise ValueError("Owned cleanup/service exit failed")
        pilot_parallel.released()
        owner_path = ai19.no_links(
            task
            / "operator/invocations"
            / str(proof["invocation_id"])
            / "worker-identity.json"
        )
        owner = read(owner_path)
        if owner.get("invocation_id") != proof["invocation_id"]:
            raise ValueError("Cleanup invocation differs from actual worker")
        proof["worker_identity"] = dict(
            path=str(owner_path), sha256=sha(owner_path)
        )
        cleanup_copy = owner_path.with_name("socket-cleanup.json")
        main_storage.durable(cleanup_copy, report)
        proof["cleanup_report"] = dict(
            path=str(cleanup_copy), sha256=sha(cleanup_copy)
        )
        if state.get("status") == "MAIN19_INPUTS_PASSED_TERMINATING":
            proof["passed"] = True
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        proof["reason"] = str(exc)
    main_storage.durable(task / "operator/cleanup-exit.json", proof)
    if re.fullmatch(r"[0-9a-f]{32}", str(proof["invocation_id"])):
        main_storage.durable(
            task
            / "operator/invocations"
            / proof["invocation_id"]
            / "cleanup-exit.json",
            proof,
        )
    if (
        not proof["passed"]
        and state.get("status") == "MAIN19_INPUTS_PASSED_TERMINATING"
    ):
        state.update(
            status="CLEANUP_FAILED_STOPPED",
            reason=proof.get("reason", "Cleanup not verified"),
        )
        write(task / "operator/state.json", state)


def termination_verified(
    cfg, state, proof, properties, owner=None, *, processes_released=False
):
    invocation = state.get("invocation_id")
    owner = owner or {}
    common = (
        cfg.get("owner_uid") == os.getuid()
        and cfg.get("mode") == "main19"
        and state.get("status") == "MAIN19_INPUTS_PASSED_TERMINATING"
        and proof.get("passed") is True
        and proof.get("owner_uid") == os.getuid()
        and proof.get("service_result") == "success"
        and proof.get("worker_exit_code") == "exited"
        and proof.get("worker_exit_status") == "0"
        and invocation
        == proof.get("invocation_id")
        == owner.get("invocation_id")
        and bool(re.fullmatch(r"[0-9a-f]{32}", str(invocation)))
        and owner.get("owner_uid") == os.getuid()
        and owner.get("pid") == state.get("pid")
        and isinstance(owner.get("pid"), int)
        and owner["pid"] > 0
        and owner.get("pid_start_ticks") == state.get("pid_start_ticks")
        and bool(owner.get("pid_start_ticks"))
        and owner.get("unit") == cfg.get("service_unit")
        and owner.get("cgroup", "").endswith("/" + cfg.get("service_unit", "?"))
        and processes_released
        and properties.get("MainPID") == "0"
        and properties.get("ActiveState") == "inactive"
        and properties.get("SubState") == "dead"
    )
    if properties.get("LoadState") == "not-found":
        # Garbage-collected defaults cannot prove success. The bound stop-hook
        # exit and the worker's own invocation/identity above are the evidence.
        return common and properties.get("InvocationID", "") in {"", invocation}
    return (
        common
        and properties.get("LoadState") == "loaded"
        and properties.get("InvocationID") == invocation
        and properties.get("Result") == "success"
        and properties.get("ExecMainCode") == "1"
        and properties.get("ExecMainStatus") == "0"
    )


def worker_processes_released(owner):
    if identity(owner["pid"]) == owner["pid_start_ticks"]:
        return False
    group = owner["cgroup"]
    if (
        not group.startswith("0::/")
        or ".." in Path(group[3:]).parts
        or "\n" in group
    ):
        return False
    folder = Path("/sys/fs/cgroup") / group[3:].lstrip("/")
    return not folder.exists() or all(
        not p.read_text(encoding="utf-8").strip()
        for p in folder.rglob("cgroup.procs")
    )


def reconcile_termination(task):
    cfg = read(task / "config.json")
    state = read(task / "operator/state.json")
    path = task / "operator/cleanup-exit.json"
    if (
        state.get("status") != "MAIN19_INPUTS_PASSED_TERMINATING"
        or not path.exists()
    ):
        return state
    proof = read(path)
    try:
        owner_path = ai19.no_links(
            task
            / "operator/invocations"
            / str(state.get("invocation_id"))
            / "worker-identity.json"
        )
        if proof.get("worker_identity", {}).get("path") != str(owner_path):
            return state
        verify({str(owner_path): proof["worker_identity"]["sha256"]})
        owner = read(owner_path)
        submission = read(ai19.no_links(task / "operator/service.json"))
        if (
            owner.get("task") != str(task)
            or owner.get("config_sha256") != sha(task / "config.json")
            or submission.get("config_sha256") != owner["config_sha256"]
            or submission.get("unit") != owner["unit"]
            or submission.get("owner_uid") != os.getuid()
        ):
            return state
        _, _, properties = pilot45.service_identity(task)
        if not termination_verified(
            cfg,
            state,
            proof,
            properties,
            owner,
            processes_released=worker_processes_released(owner),
        ):
            return state
        pilot_parallel.released()
        verify(
            {
                str(task / "config.json"): proof["config_sha256"],
                str(task / "operator/socket-cleanup.json"): proof[
                    "cleanup_sha256"
                ],
                str(ai19.no_links(proof["cleanup_report"]["path"])): proof[
                    "cleanup_report"
                ]["sha256"],
            }
        )
        receipt = task / "operator/final-termination.json"
        main_storage.durable(
            receipt,
            dict(
                passed=True,
                properties=properties,
                cleanup_exit=dict(path=str(path), sha256=sha(path)),
                observed_utc=now(),
            ),
        )
        state.update(
            status="MAIN19_INPUTS_PASSED_STOPPED",
            termination_receipt=dict(path=str(receipt), sha256=sha(receipt)),
        )
        write(task / "operator/state.json", state)
        completion = read(task / "completion.json")
        completion.update(
            status=state["status"],
            termination_receipt=state["termination_receipt"],
        )
        write(task / "completion.json", completion)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        subprocess.SubprocessError,
    ):
        return state
    return state


def render(task, tick=0):
    from .cli import duration
    from .display import grid

    task = Path(task).absolute()
    cfg = read(task / "config.json")
    s = reconcile_termination(task)
    if s.get("status") in {"RUNNING", "DRAINING"} and (
        not s.get("pid_start_ticks")
        or identity(s.get("pid")) != s.get("pid_start_ticks")
    ):
        s = dict(
            s,
            status="UNKNOWN_NOT_RUNNING",
            reason="Recorded worker is not running; no success inferred",
        )
    status = s.get("status", "NOT_STARTED")
    running = status in {"RUNNING", "DRAINING"}
    passed = status == "MAIN19_INPUTS_PASSED_STOPPED"
    summary = (
        "전체 검산·실제 종료 확인"
        if passed
        else "진행 중"
        if running
        else "전체 미완료 · 중지"
    )
    if status == "MAIN19_INPUTS_PASSED_TERMINATING":
        summary = "입력 검산 완료 · 실제 종료 확인 대기"
    elif status == "DRAINING":
        summary = "오류로 신규 투입 차단 · 실행 중 작업 종료 대기"
    elif status == "USER_CANCELLED_DISCARDED":
        summary = "사용자 중지 · 이번 시도 자료 폐기"
    elif status in {"READY_NOT_STARTED", "NOT_STARTED"}:
        summary = "시작 전"
    timestamp = datetime.now(ZoneInfo("Asia/Seoul"))
    end = (
        datetime.fromisoformat(s["finished_utc"]).timestamp()
        if s.get("finished_utc")
        else time.time()
    )
    wall_elapsed = max(0, end - s.get("started", end))
    done, total = s.get("done", 0), cfg["planned_cases"]
    percent = min(100, 100 * done / total) if total else 0
    filled = int(percent * 24 / 100)
    bar = "█" * filled + "·" * (24 - filled)
    stages = {
        "collection": "비행 수집",
        "continuous-collection-and-preparation": "빈 비행 슬롯 보충 · 후처리 동시 실행",
        "error-draining": "오류 중지 대기 · 신규 비행/후처리 시작 안 함",
        "preprocess-and-independent-audit": "원본 직접 정제·입력 검산",
        "pair-and-durable-retirement": "대조 확인·최종 압축 1회·중간 정리",
        "bounded-aggregate-and-independent-audit": "공통 scaler·통합·독립 검산",
    }
    kinds = {
        "hover": "호버",
        "waypoint": "경로",
        "horizontal": "수평",
        "yaw": "회전",
        "altitude": "고도",
    }
    operations = {
        "prepare": "정제",
        "audit": "검산",
        "commit": "보관",
        "pair": "대조",
        "aggregate": "통합",
    }
    remaining = []
    cards = []
    for x in sorted(
        s.get("active", []),
        key=lambda x: (x.get("instance") is None, x.get("instance") or 0),
    ):
        elapsed, target, rtf = (
            x.get("observed_simulation_elapsed_s"),
            x.get("producer_target_s"),
            x.get("live_rtf"),
        )
        clock = (
            f"{elapsed:.0f}/{target or 480:.0f}s"
            if elapsed is not None
            else "--/480s"
        )
        rate = f"{rtf:.2f}x" if rtf is not None else "--x"
        lane = (
            str(x.get("instance")) if x.get("instance") is not None else "CPU"
        )
        kind = kinds.get(x.get("kind"), str(x.get("kind", "비행"))[:4])
        if x.get("action") == "flight":
            cards.append(
                f"{lane:>3} {kind} {clock} {rate} {x.get('attempt', '?')}/{x.get('attempt_limit', 3)}"
            )
            if elapsed is not None and rtf is not None and rtf > 0:
                remaining.append(max(0, (target or 480) - elapsed) / rtf)
        else:
            operation = operations.get(
                x.get("action"), str(x.get("action", "처리"))[:4]
            )
            cards.append(
                f"CPU {operation} {str(x.get('case_id', '-'))[-20:]} {x.get('attempt', '?')}/3"
            )
    projection = s.get("storage_projection", {})
    resource = s.get("resources", {})

    def gb(key):
        return f"{resource[key] / 1e9:.1f}GB" if key in resource else "미확인"

    eta = (
        duration(max(remaining)) + " (관측 RTF 기준, 후처리 제외)"
        if running and remaining
        else "미확인"
    )
    phase = stages.get(s.get("stage"), s.get("stage", "준비"))
    label = "실행" if running else "마지막 기록"
    lines = [
        f"X500  /  학습 전 통합 콘솔  /  {task.name}",
        f"  {status}  |  {summary}",
        f"  {timestamp:%m/%d %H:%M:%S} KST · 화면 #{tick} · 경과 {duration(wall_elapsed)} · PID {s.get('pid', '-')}",
        "─" * 78,
        "흐름   수집 → 원본 직접 정제 → 대조 확인·보관 → 통합·공통 scaler → 검산 → 정지",
        "설정   비행16 / 2배속 / 임무480초 · 후처리 전체4 동시 실행 · 최대3시도"
        if cfg.get("continuous_refill")
        else "설정   비행16 / 2배속 / 임무480초 · 비행 종료 후 비비행 전체4 · 최대3시도",
        f"수집·정제 [{bar}] {percent:5.1f}%  {done}/{total}건 인수·보관 완료",
        f"물리비행 완료 {s.get('completed_physical_attempts', 0)}/{s.get('physical_attempts', 0)}시도 · 실패 {s.get('failed_physical_attempts', 0)} · 품질거절 {s.get('quality_rejected_attempts', 0)}",
        f"불가용/실패 {s.get('failed', 0)}건 · 미시작 {s.get('not_started', total)}건 · 단계100% ≠ 전체완료",
        f"현재   {phase} · {label} 비행{s.get('active_flights', 0)}/16 처리{s.get('active_nonflight', 0)}/4"
        if cfg.get("continuous_refill")
        else f"현재   {phase} · 배치 {s.get('batch', 0)}/{s.get('batches', '-')} · {label} 비행{s.get('active_flights', 0)}/16 처리{s.get('active_nonflight', 0)}/4",
        *(
            [
                f"후처리 대기 {s.get('raw_queue_waiting', 0)}건 · "
                f"정제 {s.get('postprocess_waiting', {}).get('prepare', '?')} / "
                f"대조쌍 {s.get('postprocess_waiting', {}).get('pair', '?')} / "
                f"보관 {s.get('postprocess_waiting', {}).get('commit', '?')} · 실행 중 제외"
            ]
            if cfg.get("continuous_refill")
            else []
        ),
        f"현재 비행 잔여 예상 {eta}",
        f"마지막 자원  가용 RAM {gb('available_bytes')} · RSS {gb('owned_tree_rss_bytes')} · 저장 여유 {gb('disk_free_bytes')}",
        *grid(cards),
        *[
            f"합산/검산 {p.get('dataset_scope', '-')} · {p.get('stage', '-')} · {p.get('done', 0)}/{p.get('total', '-')} · {p.get('role', '')} {p.get('status', '')}"
            for p in s.get("aggregate_progress", [])
        ],
        f"보관   {projection.get('actual_permanent_bytes', 0) / 1e9:.1f}GB · 저장 {cfg.get('external_storage', STORE)}",
    ]
    failure, failure_log = None, None
    if not running and s.get("reason"):
        for item in s.get("active", [])[:4]:
            if not item.get("log"):
                continue
            log = Path(item.get("log", ""))
            receipt = log.with_suffix(".request.failure.json")
            if (
                receipt.is_relative_to(task)
                and receipt.is_file()
                and not receipt.is_symlink()
            ):
                failure, failure_log = read(receipt), log
                break
    if s.get("admission_wait_reason"):
        lines.append(
            "비행 투입 대기: "
            + str(s["admission_wait_reason"])
            + " · 후처리 계속"
        )
    if failure:
        lines.append(
            f"사유   {failure.get('error_type')}: {failure.get('failure_reason')}"
        )
        lines.append(f"오류 로그 {failure_log}")
    elif s.get("reason"):
        lines.append("사유   " + str(s["reason"]))
    lines.extend(
        [
            f"로그   {task / 'operator/run.log'}",
            "q/Ctrl+C: 화면만 닫기 · 작업 독립 실행 · 자동재시작·학습·LLM 호출 없음",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("collect", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("scope", choices=("main", "lr108"))
        p.add_argument("--storage-root", type=Path, default=STORE)
        p.add_argument("--name")
        if name == "collect":
            p.add_argument(
                "--no-watch",
                action="store_true",
                help="작업 제출 후 화면을 열지 않음",
            )
    for name in ("watch", "status", "stop", "cancel", "resume", "preflight"):
        p = sub.add_parser(name)
        p.add_argument("task", type=Path)
    p = sub.add_parser("_case")
    p.add_argument("request", type=Path)
    a = parser.parse_args(argv)
    if a.action == "_case":
        case_entry(a.request)
    elif a.action in {"collect", "prepare"}:
        task = (
            user_start(a.scope, storage_root=a.storage_root, name=a.name)
            if a.action == "collect"
            else create(
                a.scope, storage_root=a.storage_root, name=a.name, fresh=True
            )
        )
        print(task)
        if (
            a.action == "collect"
            and not a.no_watch
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        ):
            from .cli import watch

            watch(task)
    elif a.action in {"stop", "cancel"}:
        print(pilot45.stop(a.task, immediate=a.action == "cancel"))
    elif a.action == "resume":
        resume(a.task)
    elif a.action == "preflight":
        print(preflight(a.task))
    else:
        from .cli import watch

        watch(a.task, once=a.action == "status")


if __name__ == "__main__":
    main()
