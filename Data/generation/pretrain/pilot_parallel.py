"""Finite continuation of the same45 cases:16 native lanes and one4-slot CPU pool."""

import argparse
import copy
import fcntl
import os
import queue
import socket
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path

from . import ai19, pilot45, pilot_bridge
from .common import (
    HOME,
    ROOT,
    STORE,
    identity,
    now,
    read,
    service_unit,
    sha,
    verify,
    write,
)
from .pilot19 import COHORT, FINAL16_COHORT, SHORT_COHORT, audit_one

AUTH = ROOT / "ai/reports/p27-pilot45-20261107/parallel16-01/authorization.json"
REPORT = ROOT / "docs/data/reports/p27-pilot45-20261107/parallel16-01"
CONTROL_CHANGES = {
    str(ROOT / "Data/generation/pretrain" / n)
    for n in ("pilot45.py", "native_flight.py")
}


def authorization():
    value = read(AUTH)
    if value.get("task_id") != pilot45.TASK_ID or value.get(
        "requested"
    ) != dict(
        flight_workers=16,
        real_time_factor=2,
        aggregate_nonflight_workers=4,
        pilot_logical_cases=45,
    ):
        raise pilot45.Blocked(
            "Explicit same45/16flight/RTF2/CPU4 authorization missing"
        )
    if (
        not all(
            value["unchanged"].get(k) is True
            for k in (
                "same_producer_source_runtime_physics_and_common19",
                "success_regeneration_forbidden",
                "no_training_fit_forward_evaluation_gpu",
                "no_main_collection_or_next_round",
                "no_LLM_monitoring_or_callbacks",
            )
        )
        or value["unchanged"].get("physical_attempts_max") != 3
    ):
        raise pilot45.Blocked("Parallel transition scope changed")
    return value


def validate_config(cfg):
    if cfg.get("final16"):
        from .final16 import validate_config as validate_final16

        return validate_final16(cfg)
    value = (
        pilot45.short_authorization()
        if cfg.get("short_cycle")
        else authorization()
    )
    if type(cfg.get("nonflight_when_flights_idle", False)) is not bool:
        raise pilot45.Blocked(
            "Explicit boolean required for nonflight admission policy"
        )
    if (
        cfg.get("flight_workers"),
        cfg.get("requested_real_time_factor"),
        cfg.get("cpu_workers_total_max"),
    ) != (16, 2, 4):
        raise pilot45.Blocked("No silent concurrency/RTF downgrade")
    if (
        cfg.get("planned_cases") != 45
        or cfg.get("training_authorized")
        or cfg.get("fit_authorized")
    ):
        raise pilot45.Blocked(
            "Only the approved45 no-fit continuation is executable"
        )
    if cfg.get("short_cycle"):
        if cfg.get("short_resume") and (
            cfg.get("nonflight_when_flights_idle") is not True
            or cfg.get("stop_on_first_physical_failure") is not True
        ):
            raise pilot45.Blocked(
                "Reviewed continuation requires idle-only and stop-on-first-failure"
            )
        if (
            cfg.get("short_repair")
            and cfg.get("stop_on_first_physical_failure") is not True
        ):
            raise pilot45.Blocked(
                "Short repair must drain on first new physical failure"
            )
        if cfg.get("mission_seconds") != 180 or cfg.get("continuation"):
            raise pilot45.Blocked(
                "Short-cycle must be180s in a new namespace without480s continuation"
            )
    elif (
        cfg.get("continuation", {}).get("previous_task")
        != value["previous_pilot"]
    ):
        raise pilot45.Blocked("Continuation belongs to a different task")


def clean_short_inputs(quality, ledger):
    return (
        quality.get("status") == "PILOT45_QUALITY_PASSED_STOPPED"
        and all(
            quality.get(k) == 45
            for k in (
                "planned_cases",
                "completed_raw",
                "prepared_and_audited",
                "online_available",
                "run_available",
                "ai_case_checks_passed",
            )
        )
        and len(ledger["attempts"]) == 45
        and all(
            a["state"] == "COMPLETED" and a["number"] == 1
            for a in ledger["attempts"]
        )
        and not any(e.get("inherited_from") for e in ledger["cases"].values())
        and quality.get("short_ai_final_passed") is True
    )


def current_execution_counts(task, ledger, *, planned_case_ids=None):
    """Keep cumulative attempts intact; report only this execution in a second denominator."""
    from collections import Counter

    current = [
        a for a in ledger["attempts"] if a.get("execution_task") == str(task)
    ]
    counts = Counter(a["case_id"] for a in current)
    expected = (
        set(ledger["cases"])
        if planned_case_ids is None
        else set(planned_case_ids)
    )
    if not expected <= set(ledger["cases"]):
        raise pilot45.Blocked(
            "Current native denominator must be a subset of the frozen45"
        )
    return dict(
        current_execution_planned_cases=len(expected),
        current_execution_reused_accepted_cases=len(ledger["cases"])
        - len(expected),
        physical_completed_attempts=sum(
            a["state"] == "COMPLETED" for a in ledger["attempts"]
        ),
        historical_quality_rejected_attempts=sum(
            bool(a.get("source_task")) and bool(a.get("quality_rejection"))
            for a in ledger["attempts"]
        ),
        administrative_not_submitted_attempts=len(
            ledger.get("administrative_attempts", [])
        ),
        current_execution_task=str(task),
        current_execution_physical_attempts=len(current),
        current_execution_completed_cases=len(
            {a["case_id"] for a in current if a["state"] == "COMPLETED"}
        ),
        current_execution_failed_attempts=sum(
            a["state"] == "FAILED_PRESERVED" for a in current
        ),
        current_execution_unknown_attempts=sum(
            a["state"] not in {"COMPLETED", "FAILED_PRESERVED"} for a in current
        )
        + sum(
            not a.get("source_task") and not a.get("execution_task")
            for a in ledger["attempts"]
        ),
        current_execution_retries=sum(max(0, n - 1) for n in counts.values()),
        current_execution_all_case_ids_match=set(counts) == expected,
        historical_physical_attempts=sum(
            bool(a.get("source_task")) for a in ledger["attempts"]
        ),
        historical_failed_attempts=sum(
            bool(a.get("source_task")) and a["state"] == "FAILED_PRESERVED"
            for a in ledger["attempts"]
        ),
    )


def current_execution_inputs_passed(quality):
    planned = quality.get("current_execution_planned_cases", 45)
    reused = quality.get("current_execution_reused_accepted_cases", 0)
    return (
        (planned, reused) in {(45, 0), (17, 28), (14, 31)}
        and quality.get("inherited_successes") == reused
        and quality.get("current_execution_physical_attempts") == planned
        and quality.get("current_execution_completed_cases") == planned
        and quality.get("status") == "PILOT45_QUALITY_PASSED_STOPPED"
        and quality.get("short_ai_final_passed") is True
        and all(
            quality.get(k) == 45
            for k in (
                "planned_cases",
                "completed_raw",
                "prepared_and_audited",
                "online_available",
                "run_available",
                "ai_case_checks_passed",
            )
        )
        and all(
            quality.get(k) == 0
            for k in (
                "current_execution_failed_attempts",
                "current_execution_unknown_attempts",
                "current_execution_retries",
            )
        )
        and quality.get("current_execution_all_case_ids_match") is True
    )


def ports(instance):
    return [base + instance for base in (14540, 14580, 18570, 14280, 13030)]


def check_ports(instances=range(16)):
    with ExitStack() as stack:
        for i in instances:
            for port in ports(i):
                sock = stack.enter_context(
                    socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                )
                sock.bind(("127.0.0.1", port))


def released():
    with ExitStack() as stack:
        for p in (
            HOME / "pretrain.lock",
            ROOT / "Data/operations/collection.lock",
            ROOT / "simulation/.cache/setup.lock",
            *(
                ROOT / f"simulation/.cache/instances/{i}.lock"
                for i in range(16)
            ),
        ):
            handle = stack.enter_context(p.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for i in range(16):
            pilot45.collector().check_instance_socket(i)
        check_ports()


def closed_previous(task):
    unit, pid, props = pilot45.service_identity(task)
    cfg, state = read(task / "config.json"), read(task / "operator/state.json")
    if (
        cfg["owner_uid"] != os.getuid()
        or pid
        or props.get("ControlGroup")
        or props.get("ActiveState") not in {"inactive", "failed"}
    ):
        raise pilot45.Blocked("Previous owned service has not drained")
    if identity(state["pid"]) == state["pid_start_ticks"]:
        raise pilot45.Blocked("Previous recorded worker is still alive")
    cleanup = read(task / "operator/socket-cleanup.json")
    if (
        cleanup.get("status") != "CLEANUP_COMPLETE"
        or cleanup.get("errors")
        or cleanup.get("remaining_unremoved_sockets")
    ):
        raise pilot45.Blocked(
            "Previous stop hook did not release its resources"
        )
    return dict(unit=unit, properties=props, state=state, cleanup=cleanup)


def compatible_sources(task, accepted=None):
    reviewed = {}
    if accepted is not None:
        value = read(accepted)
        if (
            value.get("task") != str(task)
            or value.get("status") != "ACCEPTED_EXISTING_INTEGRATION_NO_FLIGHT"
        ):
            raise pilot45.Blocked(
                "Prior reviewed integration acceptance does not match"
            )
        verify(value["evidence_sha256"])
        reviewed = value["source_compatibility"]["reviewed_control_changes"]
    changed = {}
    for name, digest in read(task / "config.json")["source_sha256"].items():
        current = sha(ai19.no_links(name))
        if current != digest:
            prior = reviewed.get(name, {})
            certified_old_fix = (
                name == str(ROOT / "Data/generation/pretrain/pilot_flow.py")
                and prior.get("old_sha256") == digest
                and prior.get("current_sha256") == current
            )
            if name not in CONTROL_CHANGES and not certified_old_fix:
                raise pilot45.Blocked(
                    "Producer/AI/signal processing changed: " + name
                )
            changed[name] = dict(previous_sha256=digest, current_sha256=current)
    return changed


def check_raw(raw):
    raw = ai19.no_links(raw)
    if (
        raw.parent != ROOT / "simulation/runs"
        or read(raw / "run.json").get("status") != "completed"
    ):
        raise pilot45.Blocked("Verified completed native raw required")
    for name, item in read(raw / "integrity.json").items():
        path = ai19.no_links(raw / name)
        if path.stat().st_size != item["size"] or sha(path) != item["sha256"]:
            raise pilot45.Blocked("Reused raw integrity changed: " + str(path))


def create(name):
    """New immutable source capsule; old raw/NPZ/task/attempts stay in place."""
    from . import integration_reuse, pilot_flow

    auth = authorization()
    previous = pilot45.task_path(auth["previous_pilot"])
    parent = pilot45.task_path(auth["previous_flow"])
    termination = [closed_previous(p) for p in (parent, previous)]
    released()
    old_cfg, manifest, old_ledger = (
        read(previous / n)
        for n in ("config.json", "manifest.json", "ledger.json")
    )
    verify(
        {
            str(previous / "config.json"): (previous / "config.sha256")
            .read_text()
            .strip(),
            str(previous / "manifest.json"): (previous / "manifest.sha256")
            .read_text()
            .strip(),
        }
    )
    pilot45.validate_manifest(manifest)
    if any(a["state"] == "RUNNING" for a in old_ledger["attempts"]):
        raise pilot45.Blocked(
            "Unknown physical attempt requires explicit reconciliation"
        )
    changes = compatible_sources(previous)
    spec = pilot_bridge.dependencies(old_cfg["dependency_handoff"])
    integration = Path(
        read(parent / "ledger.json")["phases"]["integration"]["task"]
    )
    integration_reuse.historical_termination(integration)
    integration_changes = compatible_sources(
        integration, parent / "accepted-integration.json"
    )
    audits = {}
    for source in (integration, previous):
        for cid, entry in read(source / "ledger.json")["cases"].items():
            if entry.get("run"):
                check_raw(entry["run"])
            if entry.get("status") == "COMPLETE":
                verify(entry["files"])
                verify(
                    read(Path(entry["output"]) / "binding.json")[
                        "consumer_sources"
                    ]
                )
                audits[str(source) + ":" + cid] = audit_one(
                    entry["output"], source / "expected-producer.json"
                )
    # The fit/reference namespace and exact planned identities survive the move.
    task = pilot45.task_path(HOME / "tasks" / (name + "-45"))
    flow = pilot45.task_path(HOME / "tasks" / name)
    if (
        task.exists()
        or flow.exists()
        or not pilot45.re.fullmatch(r"[A-Za-z0-9_-]{1,53}", name)
    ):
        raise ValueError(
            "New safe task name required; existing tasks are never overwritten"
        )
    task.mkdir(parents=True)
    namespace = old_cfg.get("lineage_namespace", previous.name)
    preserved = {
        str(p): sha(p)
        for source in (parent, previous, integration)
        for p in source.rglob("*")
        if p.is_file()
    }
    old_files = {
        str(previous / n): sha(previous / n)
        for n in ("config.json", "manifest.json", "ledger.json")
    }
    write(
        task / "continuation-acceptance.json",
        dict(
            status="VERIFIED_SUCCESS_REUSED_NO_FLIGHT",
            checked_utc=now(),
            previous=str(previous),
            source_changes=changes,
            integration_source_changes=integration_changes,
            termination=termination,
            audits=audits,
            preserved_sha256=preserved,
            new_flights=0,
        ),
    )
    write(task / "manifest.json", manifest)
    (task / "manifest.sha256").write_text(sha(task / "manifest.json") + "\n")
    ledger = copy.deepcopy(old_ledger)
    ledger["status"] = "NOT_STARTED_CONTINUATION"
    for case in manifest["cases"]:
        e = ledger["cases"].setdefault(
            case["case_id"], dict(case=case, attempts=[])
        )
        if e["case"] != case or len(e["attempts"]) > 3:
            raise pilot45.Blocked("Frozen case/retry count mismatch")
        e["processing_attempts"] = len(
            list(
                (previous / "pilot/execution" / case["case_id"]).glob(
                    "prepare-*.request.json"
                )
            )
        )
        if e.get("status") == "COMPLETE":
            e["inherited_from"] = str(previous)
    write(task / "ledger.json", ledger)
    cfg = copy.deepcopy(old_cfg)
    cfg.update(
        parallel16=True,
        flight_workers=16,
        cpu_workers_total_max=4,
        requested_real_time_factor=2,
        validated_flight_workers=None,
        dispatch_validation="regression_only_native16_not_yet_observed",
        lineage_namespace=namespace,
        service_unit="drone-pretrain-" + task.name + ".service",
        continuation=dict(previous_task=str(previous), files=old_files),
        source_sha256={},
        report_directory=str(REPORT / task.name),
        ai_ready=None,
        authorization=dict(path=str(AUTH), sha256=sha(AUTH)),
    )
    write(task / "config.json", cfg)
    write(task / "budget.json", read(previous / "budget.json"))
    write(task / "dependency-handoff.json", spec)
    cfg["dependency_handoff"] = str(task / "dependency-handoff.json")
    write(task / "config.json", cfg)
    write(
        task / "expected-producer.json",
        read(previous / "expected-producer.json"),
    )
    protected = pilot45.protection()
    own_raw = {a["run_id"] for a in ledger["attempts"] if a.get("run_id")}
    # Keep old pilot claims in their own no-fit cohort instead of excluding them
    # from this same pilot's continuation. All other raw remain excluded.
    protected["records"] = [
        r for r in protected["records"] if r.get("run_id") not in own_raw
    ]
    write(task / "protection.json", protected)
    native = read(spec["paths"]["native_completion"])
    dev = native["actual_run_ids"] + [
        Path(e["run"]).name
        for e in read(integration / "ledger.json")["cases"].values()
    ]
    gp = pilot45.global_protection(protected, manifest["cases"], namespace, dev)
    for rid in own_raw:
        gp["entries"].append(
            dict(
                identities=["run:" + rid],
                forbidden_uses=[
                    "fit",
                    "input_check",
                    "calibration",
                    "final",
                    "final_test",
                    "final_input_check",
                ],
                purpose=COHORT,
            )
        )
    write(task / "global-protection.json", gp)
    write(task / "ai-manifest.json", pilot_bridge.plan(task))
    write(task / "consumer-capsule.json", pilot_bridge.capsule(task, spec))
    checked = pilot_bridge.check(task, spec)
    if checked["result"].get("ai_prerequisites_passed") is not True:
        raise pilot45.Blocked("New continuation capsule prerequisites failed")
    cfg["ai_ready"] = checked
    extra = [
        AUTH,
        task / "continuation-acceptance.json",
        task / "dependency-handoff.json",
        task / "ai-manifest.json",
        task / "consumer-capsule.json",
        Path(checked["receipt"]),
        *map(Path, old_files),
        *map(Path, spec["paths"].values()),
    ]
    cfg["source_sha256"] = pilot45.frozen_sources(task, extra)
    cfg["source_sha256"].update(
        {
            str(ROOT / p): h
            for p, h in read(task / "consumer-capsule.json")["files"].items()
        }
    )
    write(task / "config.json", cfg)
    (task / "config.sha256").write_text(sha(task / "config.json") + "\n")
    write(
        task / "operator/state.json",
        dict(status="READY_NOT_STARTED", task=task.name, updated_utc=now()),
    )
    flow = pilot_flow.init(
        name,
        integration,
        previous,
        cfg["dependency_handoff"],
        reuse_integration=integration,
    )
    fc = read(flow / "config.json")
    fc.update(
        parallel_flow=True,
        pilot_task=str(task),
        planned_cases=45,
        flight_workers=16,
        requested_real_time_factor=2,
        cpu_workers_total_max=4,
    )
    fc["source_sha256"].update(
        {
            str(AUTH): sha(AUTH),
            str(task / "config.json"): sha(task / "config.json"),
        }
    )
    write(flow / "config.json", fc)
    (flow / "config.sha256").write_text(sha(flow / "config.json") + "\n")
    write(
        flow / "ledger.json",
        dict(
            cases={},
            attempts=[],
            phases=dict(
                pilot45=dict(task=str(task)),
                integration=dict(task=str(integration)),
            ),
        ),
    )
    verify(preserved)
    return flow, task


class Coordinator:
    """Only this thread owns ledger, registry, admission and shared state writes."""

    def __init__(self, task):
        self.task = task
        self.cfg, self.manifest, self.ledger = (
            read(task / n)
            for n in ("config.json", "manifest.json", "ledger.json")
        )
        validate_config(self.cfg)
        if self.cfg.get("final16"):
            from .final16 import validate_ledger

            validate_ledger(task)
        self.events = queue.SimpleQueue()
        self.active = {}
        self.errors = []
        self.denied = {
            r["run_id"] for r in read(task / "protection.json")["records"]
        }
        self.state = dict(
            status="RUNNING",
            stage="final16" if self.cfg.get("final16") else "parallel45",
            task=task.name,
            pid=os.getpid(),
            pid_start_ticks=identity(os.getpid()),
            owner_uid=os.getuid(),
            started=time.time(),
            service_unit=service_unit(task),
            cgroup=Path("/proc/self/cgroup").read_text(),
            requested_flight_workers=16,
            requested_real_time_factor=2,
            cpu_workers_total_max=4,
            max_active_flights=0,
            max_active_nonflight=0,
            training_authorized=False,
        )

    def save(self):
        write(self.task / "ledger.json", self.ledger)

    def folder(self, case):
        folder = self.task / "pilot/execution" / case["case_id"]
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def request(self, case, action, label, **values):
        folder = self.folder(case)
        request, receipt = (
            folder / (label + ".request.json"),
            folder / (label + ".receipt.json"),
        )
        if request.exists() or receipt.exists():
            raise pilot45.Blocked(
                "Existing phase evidence is never overwritten"
            )
        write(
            request,
            dict(
                task=str(self.task),
                case=case,
                action=action,
                receipt=str(receipt),
                **values,
            ),
        )
        return request, folder / (label + ".log")

    def begin_flight(self, case, instance):
        if self.cfg.get("final16") and case not in self.manifest["cases"]:
            raise pilot45.Blocked(
                "Only the exact frozen16 may enter flight admission"
            )
        entry = self.ledger["cases"][case["case_id"]]
        if entry.get("run") or entry.get("status") == "COMPLETE":
            raise pilot45.Blocked("Successful raw must not be regenerated")
        prior = entry["attempts"][-1] if entry["attempts"] else None
        corrected = pilot45.short_repair_allows(self.cfg, case, prior)
        if self.cfg.get("short_resume"):
            from .short_resume import allowed

            corrected = allowed(self.cfg, case, prior)
        if self.cfg.get("mode") == "main19":
            from .main_quality import allowed

            corrected = allowed(self.cfg, case, prior)
        maximum = case["attempts_max"]
        if self.cfg.get("mode") == "main19":
            from .main_quality import attempts_max

            maximum = attempts_max(self.cfg, case)
        number = len(entry["attempts"]) + 1
        if number > maximum or (
            prior
            and (
                (not prior.get("retry_allowed") and not corrected)
                or not prior.get("run_id")
            )
        ):
            raise pilot45.Blocked(
                "Logical retry limit/parent/independent-failure gate"
            )
        parent = prior["run_id"] if prior else None
        attempt = dict(
            number=number,
            state="RUNNING",
            instance=instance,
            parent_attempt_id=parent,
            started_utc=now(),
        )
        if self.cfg.get("short_cycle") or self.cfg.get("final16"):
            attempt["execution_task"] = str(self.task)
        pilot45.register(
            self.task,
            dict(
                event="physical_attempt_started",
                cohort=case["cohort"],
                case_id=case["case_id"],
                attempt=number,
                scenario=case["scenario"],
                seed=case["seed"],
                paired_case_id=case["paired_case_id"],
                group_id=self.cfg["lineage_namespace"] + ":" + case["group_id"],
                retry_family_id=self.cfg["lineage_namespace"]
                + ":"
                + case["retry_family_id"],
                parent_attempt_id=parent,
                role=case["role"],
            ),
        )
        entry["attempts"].append(attempt)
        self.ledger["attempts"].append(dict(case_id=case["case_id"], **attempt))
        self.save()

        request, log = self.request(
            case,
            "flight",
            f"flight-{number:02}",
            parent_attempt_id=parent,
            instance=instance,
        )
        return dict(
            case=case,
            instance=instance,
            number=number,
            request=request,
            log=log,
        )

    def finish_flight(self, job, failure):
        case = job["case"]
        entry = self.ledger["cases"][case["case_id"]]
        attempt = entry["attempts"][job["number"] - 1]
        path = Path(read(job["request"])["receipt"])
        result = read(path) if path.exists() else {}
        raw = ai19.no_links(result["run"]) if result.get("run") else None
        if raw is not None:
            if (
                raw.parent != ROOT / "simulation/runs"
                or raw.name in self.denied
            ):
                raise pilot45.Blocked("Raw outside the continued pilot")
            meta, scenario = (
                read(raw / "run.json"),
                read(raw / "evaluation/scenario.json"),
            )
            if (
                meta.get("instance"),
                meta.get("parent_attempt_id"),
                scenario.get("name"),
                scenario.get("seed"),
                scenario.get("operation_variation"),
            ) != (
                job["instance"],
                attempt["parent_attempt_id"],
                case["scenario"],
                case["seed"],
                case["operation_variation"],
            ):
                raise pilot45.Blocked(
                    "Lane/case/seed/parent/operation mismatch"
                )
            attempt["run_id"] = raw.name
        success = raw is not None and result.get("status") == "completed"
        if success and meta.get("status") != "completed":
            raise pilot45.Blocked("Wrapper/raw completion disagreement")
        attempt.update(
            state="COMPLETED" if success else "FAILED_PRESERVED",
            ended_utc=now(),
            error=str(failure) if failure else None,
        )
        if success:
            entry["run"] = str(
                raw
            )  # durable before any preprocessing or cleanup error
        else:
            check_retry = pilot45.retryable
            if self.cfg.get("mode") == "main19":
                from functools import partial

                from .main_collection import retryable as main_retryable
                from .main_quality import attempts_max

                check_retry = partial(
                    main_retryable,
                    binding=self.cfg["producer"],
                    attempts_max=attempts_max(self.cfg, case),
                )
            attempt.update(
                result=result,
                retry_allowed=not self.cfg.get(
                    "stop_on_first_physical_failure", False
                )
                and check_retry(
                    result,
                    job["number"],
                    (self.task / "operator/stop-request.json").exists(),
                ),
            )
            pilot45.register(
                self.task,
                dict(
                    event="physical_attempt_failed",
                    cohort=case["cohort"],
                    case_id=case["case_id"],
                    run_id=raw.name if raw else None,
                    attempt=job["number"],
                    failure=attempt["error"],
                    retry_allowed=attempt["retry_allowed"],
                ),
            )
        # Never update attempts[-1]: concurrent lanes may have appended later cases.
        found = [
            a
            for a in self.ledger["attempts"]
            if a["case_id"] == case["case_id"] and a["number"] == job["number"]
        ]
        if len(found) != 1:
            raise pilot45.Blocked("Ambiguous logical attempt record")
        found[0].update(attempt)
        self.save()
        from .socket_cleanup import release

        claim = job["request"].parent / "socket-claim.json"
        if claim.exists():
            release(read(claim), service_unit(self.task))
        pilot45.collector().check_instance_socket(job["instance"])
        check_ports([job["instance"]])
        if success and failure is not None:
            raise pilot45.Blocked(
                "Successful raw retained after child error; no re-flight"
            ) from failure
        if not success and not attempt["retry_allowed"]:
            raise pilot45.Blocked(
                "Physical failure preserved; automatic retry not allowed"
            ) from failure
        return success

    def begin_processing(self, case):
        entry = self.ledger["cases"][case["case_id"]]
        if entry.get("status") == "COMPLETE":
            raise pilot45.Blocked("Complete views must not be overwritten")
        n = entry.get("processing_attempts", 0) + 1
        if n > (1 if self.cfg.get("final16") else 3):
            raise pilot45.Blocked(
                "Processing attempt limit reached, raw retained"
            )
        entry["processing_attempts"] = n
        self.save()
        output = (
            STORE
            / (
                "processed/final16"
                if self.cfg.get("final16")
                else "processed/pilot45"
            )
            / self.task.name
            / case["case_id"]
            / f"attempt-{n:02}"
        )
        ns = self.cfg["lineage_namespace"]
        raw = Path(entry["run"])
        selection = dict(
            case_id=ns + ":" + case["case_id"],
            path=str(raw),
            run_id=raw.name,
            role=case["role"],
            cohort=FINAL16_COHORT
            if self.cfg.get("final16")
            else SHORT_COHORT
            if self.cfg.get("short_cycle")
            else COHORT,
            group_id=ns + ":" + case["group_id"],
            retry_family_id=ns + ":" + case["retry_family_id"],
            previous_attempts=[
                a["run_id"] for a in entry["attempts"][:-1] if a.get("run_id")
            ],
            paired_case_id=case["paired_case_id"],
        )
        request, log = self.request(
            case,
            "prepare",
            f"prepare-{n:02}",
            selection=selection,
            output=str(output),
            archive_cache=str(self.folder(case) / "archive-cache.json"),
        )
        audit, audit_log = self.request(
            case, "audit", f"audit-{n:02}", output=str(output)
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

    def child(self, request, log):
        return pilot45.child(
            self.task, request, log, on_progress=self.events.put
        )

    def process(self, job):
        # Archive verification, extraction, both views and independent audit all
        # occupy this same CPU slot, never another per-stage executor.
        self.child(job["request"], job["log"])
        self.child(job["audit"], job["audit_log"])

    def finish_processing(self, job):
        case, output = job["case"], job["output"]
        receipt, ar = (
            Path(read(job["request"])["receipt"]),
            Path(read(job["audit"])["receipt"]),
        )
        prepared, audit = read(receipt), read(ar)
        if audit["export_receipt_sha256"] != sha(output / "receipt.json"):
            raise pilot45.Blocked("Independent audit binds another export")
        entry = self.ledger["cases"][case["case_id"]]
        sel = job["selection"]
        pilot45.register(
            self.task,
            dict(
                event="completed_raw",
                cohort=case["cohort"],
                case_id=case["case_id"],
                scenario=case["scenario"],
                seed=case["seed"],
                run_id=Path(entry["run"]).name,
                raw_sha256=prepared["run"]["raw_sha256"],
                paired_case_id=sel["paired_case_id"],
                group_id=sel["group_id"],
                retry_family_id=sel["retry_family_id"],
                role=case["role"],
                parent_attempt_id=entry["attempts"][-1]["parent_attempt_id"],
            ),
        )
        files = {str(output / p): h for p, h in prepared["files"].items()}
        files.update(
            {
                str(output / p): sha(output / p)
                for p in ("receipt.json", "seal.json")
            }
        )
        files[str(ar)] = sha(ar)
        entry.update(
            status="COMPLETE",
            receipt=str(receipt),
            output=str(output),
            files=files,
            audit_receipt=str(ar),
            online_available=prepared["online"]["available"],
            run_available=prepared["run_health"]["status"] == "available",
        )
        self.save()

    def tick(self, flights, cpu):
        while not self.events.empty():
            event = self.events.get()
            self.active[event["case_id"]] = event
        ids = {j["case"]["case_id"] for j in [*flights.values(), *cpu.values()]}
        self.active = {k: v for k, v in self.active.items() if k in ids}
        self.state.update(
            updated_utc=now(),
            active=list(self.active.values()),
            active_flights=len(flights),
            active_nonflight=len(cpu),
            total=len(self.manifest["cases"]),
            done=sum(
                e.get("status") == "COMPLETE"
                for e in self.ledger["cases"].values()
            ),
            max_active_flights=max(
                self.state["max_active_flights"], len(flights)
            ),
            max_active_nonflight=max(
                self.state["max_active_nonflight"], len(cpu)
            ),
            errors=self.errors,
        )
        write(self.task / "operator/state.json", self.state)

    def run(self):
        pending = deque(
            c
            for c in self.manifest["cases"]
            if not self.ledger["cases"][c["case_id"]].get("run")
        )
        processing = deque(
            c
            for c in self.manifest["cases"]
            if self.ledger["cases"][c["case_id"]].get("run")
            and self.ledger["cases"][c["case_id"]].get("status") != "COMPLETE"
        )
        free = set(range(16))
        flights, cpu = {}, {}
        deadline = time.monotonic() + 24 * 3600
        with (
            ThreadPoolExecutor(max_workers=16) as flight_pool,
            ThreadPoolExecutor(max_workers=4) as cpu_pool,
        ):
            while pending or processing or flights or cpu:
                stopped = (self.task / "operator/stop-request.json").exists()
                if stopped and not self.errors:
                    self.errors.append(
                        "USER_STOP: drain active lanes; no new case/retry"
                    )
                if self.errors:
                    pending.clear()
                    if self.cfg.get("final16"):
                        processing.clear()
                try:
                    self.state["resources"] = pilot45.resources()
                    if time.monotonic() > deadline:
                        raise pilot45.Blocked("Finite parallel campaign24h cap")
                    while (
                        pending
                        and free
                        and not self.errors
                        and (
                            not self.cfg.get(
                                "nonflight_when_flights_idle", False
                            )
                            or not cpu
                        )
                    ):
                        case, instance = pending.popleft(), min(free)
                        free.remove(instance)
                        job = self.begin_flight(case, instance)
                        flights[
                            flight_pool.submit(
                                self.child, job["request"], job["log"]
                            )
                        ] = job
                    while (
                        processing
                        and not (self.cfg.get("final16") and self.errors)
                        and len(cpu) < 4
                        and (
                            not self.cfg.get(
                                "nonflight_when_flights_idle", False
                            )
                            or not flights
                        )
                    ):
                        job = self.begin_processing(processing.popleft())
                        cpu[cpu_pool.submit(self.process, job)] = job
                except Exception as exc:
                    self.errors.append(
                        f"admission: {type(exc).__name__}: {exc}"
                    )
                    pending.clear()
                    if not flights and not cpu:
                        break
                self.tick(flights, cpu)
                if not flights and not cpu:
                    break
                done, _ = wait(
                    [*flights, *cpu], timeout=1, return_when=FIRST_COMPLETED
                )
                for future in done:
                    if future in flights:
                        job = flights.pop(future)
                        failure = future.exception()
                        try:
                            success = self.finish_flight(job, failure)
                            if success:
                                processing.append(job["case"])
                            elif not self.errors:
                                pending.appendleft(job["case"])
                            free.add(job["instance"])
                        except Exception as exc:
                            self.errors.append(
                                f"flight {job['case']['case_id']}: {type(exc).__name__}: {exc}"
                            )
                            if self.ledger["cases"][job["case"]["case_id"]].get(
                                "run"
                            ):
                                processing.append(job["case"])
                    else:
                        job = cpu.pop(future)
                        try:
                            future.result()
                            self.finish_processing(job)
                        except Exception as exc:
                            self.errors.append(
                                f"processing {job['case']['case_id']}: {type(exc).__name__}: {exc}"
                            )
            self.tick(flights, cpu)
            if self.errors:
                raise pilot45.Blocked("; ".join(self.errors))
            self.final_checks(cpu_pool)

    def batch(self, pool, jobs):
        pending, running = deque(jobs), {}
        results = []
        stopped = False
        while pending or running:
            if (self.task / "operator/stop-request.json").exists():
                stopped = True
                pending.clear()
            while pending and len(running) < 4:
                j = pending.popleft()
                running[pool.submit(self.child, j["request"], j["log"])] = j
            if not running:
                break
            done, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
            self.tick({}, running)
            for future in done:
                job = running.pop(future)
                future.result()
                results.append((job, read(read(job["request"])["receipt"])))
        if stopped:
            raise pilot45.Blocked("USER_STOP: nonflight jobs drained")
        return results

    def final_checks(self, pool):
        # The full45 denominator, including inherited successes, is checked by
        # another process in the same global CPU pool before any pair checks.
        case = self.manifest["cases"][0]
        request, log = self.request(case, "cohort_audit", "cohort-audit")
        self.batch(pool, [dict(case=case, request=request, log=log)])
        pairs, jobs = {}, []
        for control in (
            c
            for c in self.manifest["cases"]
            if c["category"] == "paired_normal"
        ):
            anomaly = next(
                c
                for c in self.manifest["cases"]
                if c["category"] == "anomaly"
                and c["group_id"] == control["group_id"]
            )
            request, log = self.request(
                control,
                "pair",
                "pair",
                normal_id=control["case_id"],
                anomaly_id=anomaly["case_id"],
            )
            jobs.append(dict(case=control, request=request, log=log))
        for job, result in self.batch(pool, jobs):
            pairs[job["case"]["group_id"]] = result["pair_receipt"]
        jobs = []
        for case in self.manifest["cases"]:
            request, log = self.request(
                case,
                "ai_case",
                "ai-case",
                pair_receipt=pairs.get(case["group_id"]),
            )
            jobs.append(dict(case=case, request=request, log=log))
        for job, _ in self.batch(pool, jobs):
            self.ledger["cases"][job["case"]["case_id"]]["ai_receipt"] = read(
                job["request"]
            )["receipt"]
        self.save()
        if self.cfg.get("short_cycle") or self.cfg.get("final16"):
            request, log = self.request(
                self.manifest["cases"][0], "ai_final", "ai-final"
            )
            self.batch(
                pool,
                [
                    dict(
                        case=self.manifest["cases"][0], request=request, log=log
                    )
                ],
            )
            self.ledger["ai_final_receipt"] = read(request)["receipt"]
            self.save()


def cohort_audit(task):
    ai19.ai_modules()
    from learning.admission19 import check_protection, validate_lineages

    ledger = read(task / "ledger.json")
    rows = read(task / "manifest.json")["cases"]
    final16 = read(task / "config.json").get("final16") is True
    if final16:
        from .final16 import validate_ledger

        validate_ledger(task, complete=True)
    total = 16 if final16 else 45
    if len(rows) != total or set(ledger["cases"]) != {
        c["case_id"] for c in rows
    }:
        raise pilot45.Blocked("Complete frozen case denominator changed")
    reports = []
    for case in rows:
        entry = ledger["cases"][case["case_id"]]
        if entry.get("status") != "COMPLETE" or entry["case"] != case:
            raise pilot45.Blocked("A planned case is incomplete")
        verify(entry["files"])
        reports.append(read(entry["receipt"]))
    validate_lineages([r["run"] for r in reports])
    check_protection(
        [r["run"] for r in reports],
        read(task / "global-protection.json"),
        usage="final16_input_check"
        if final16
        else "short_cycle_input_check"
        if read(task / "config.json").get("short_cycle")
        else "pilot_input_check",
    )
    if not all(
        r["online"]["available"] and r["run_health"]["status"] == "available"
        for r in reports
    ):
        raise pilot45.Blocked(
            "One or both views unavailable; retain full45 denominator"
        )
    return dict(
        status="ALL16_INDEPENDENTLY_CHECKED"
        if final16
        else "ALL45_INDEPENDENTLY_CHECKED",
        planned=total,
        no_fit=True,
    )


def worker(task):
    cfg = read(task / "config.json")
    verify(
        {
            str(task / "config.json"): (task / "config.sha256")
            .read_text()
            .strip()
        }
    )
    pilot45.preflight(task)
    coordinator = Coordinator(task)
    with ExitStack() as stack:
        for p in (
            HOME / "pretrain.lock",
            ROOT / "Data/operations/collection.lock",
        ):
            h = stack.enter_context(p.open("a"))
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (task / "operator/claimed").exists():
            if cfg.get("final16"):
                raise pilot45.Blocked(
                    "FINAL16 cannot resume or retry a claimed cycle"
                )
            request = task / "operator/resume-request.json"
            if (
                not request.exists()
                or read(request).get("config_sha256")
                != sha(task / "config.json")
                or read(request).get("owner_uid") != os.getuid()
            ):
                raise pilot45.Blocked(
                    "Explicit identity-checked resume required"
                )
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
            with (task / "operator/claimed").open("x") as f:
                f.write(now())
        try:
            coordinator.run()
            coordinator.state.update(
                status="FINAL16_QUALITY_PASSED_STOPPED"
                if cfg.get("final16")
                else "PILOT45_QUALITY_PASSED_STOPPED",
                no_fit=True,
                next_stage_authorized=False,
            )
        except BaseException as exc:
            coordinator.state.update(
                status="BLOCKED_STOPPED",
                reason=f"{type(exc).__name__}: {exc}",
                next_stage_authorized=False,
            )
            raise
        finally:
            coordinator.state.update(finished=time.time(), updated_utc=now())
            coordinator.ledger["status"] = coordinator.state["status"]
            coordinator.save()
            write(task / "operator/state.json", coordinator.state)
            if cfg.get("final16"):
                from .final16 import quality as final16_quality

                quality = final16_quality(
                    task, coordinator.ledger, coordinator.state
                )
            else:
                quality = pilot45.quality_snapshot(
                    task, coordinator.ledger, coordinator.state
                )
            quality.update(
                requested_flight_workers=16,
                requested_real_time_factor=2,
                cpu_workers_total_max=4,
                max_active_flights=coordinator.state["max_active_flights"],
                max_active_nonflight=coordinator.state["max_active_nonflight"],
                inherited_successes=sum(
                    bool(e.get("inherited_from"))
                    for e in coordinator.ledger["cases"].values()
                ),
            )
            if cfg.get("short_cycle"):
                quality.update(
                    current_execution_counts(
                        task,
                        coordinator.ledger,
                        planned_case_ids=cfg.get("short_resume", {}).get(
                            "pending_native_cases"
                        ),
                    )
                )
                final = (
                    read(coordinator.ledger["ai_final_receipt"])["result"]
                    if coordinator.ledger.get("ai_final_receipt")
                    else {}
                )
                quality["short_ai_final_passed"] = (
                    final.get("status")
                    == "SHORT45_INPUT_AND_PAIR_CHECKS_PASSED"
                    and final.get("actual_data_verified") is True
                    and final.get("cases_checked") == 45
                    and final.get("pairs_checked") == 9
                )
                quality.update(
                    scope_id="SHORT_CYCLE_20261107",
                    mission_seconds=180,
                    clean_inputs_without_retry=clean_short_inputs(
                        quality, coordinator.ledger
                    ),
                    clean_cycle=False,
                    actual_service_termination_pending=True,
                    LR_main_connected=False,
                )
                quality["current_execution_inputs_passed_without_retry"] = (
                    current_execution_inputs_passed(quality)
                )
            write(task / "quality.json", quality)
            write(Path(cfg["report_directory"]) / "completion.json", quality)


def flow_worker(flow):
    from .pilot_flow import execute, preflight

    cfg = read(flow / "config.json")
    preflight(flow)
    state = dict(
        status="RUNNING",
        stage="final16" if cfg.get("final16") else "parallel45",
        task=flow.name,
        pid=os.getpid(),
        pid_start_ticks=identity(os.getpid()),
        owner_uid=os.getuid(),
        updated_utc=now(),
        next_stage_authorized=False,
    )
    with (HOME / "pilot45-flow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (flow / "operator/claimed").open("x") as f:
            f.write(now())
        write(flow / "operator/state.json", state)
        try:
            result = execute(flow, Path(cfg["pilot_task"]), state)
            released()
            result.update(
                status="PILOT45_COMPLETED_AND_RESOURCES_RELEASED",
                flight_workers_requested=16,
                real_time_factor_requested=2,
                aggregate_nonflight_workers_max=4,
                next_stage_authorized=False,
            )
            if cfg.get("final16"):
                result.update(
                    status="FINAL16_COMPLETED_AND_RESOURCES_RELEASED",
                    scope_id="FINAL16_20261107",
                    mission_seconds=180,
                    LR_main_connected=False,
                )
            if cfg.get("short_cycle"):
                clean = (
                    result["quality"].get("clean_inputs_without_retry") is True
                )
                result.update(
                    status="SHORT_CYCLE_CLEAN_PASSED_STOPPED"
                    if clean
                    else "SHORT_CYCLE_PASSED_WITH_RETRIES_STOPPED",
                    clean_cycle=clean,
                    mission_seconds=180,
                    scope_id="SHORT_CYCLE_20261107",
                    LR_main_connected=False,
                    current_execution_completed_and_resources_released=result[
                        "quality"
                    ].get("current_execution_inputs_passed_without_retry")
                    is True,
                    clean_scope="clean_cycle retains all-history first-attempt criterion; current_execution fields retain a separate current-cycle denominator",
                )
            write(flow / "completion.json", result)
            (flow / "FINAL_REPORT.md").write_text(
                "신규16편·두19입력·독립품질·8pair·AI16검사와 실제서비스 종료/16인스턴스 반환을 확인했습니다. 학습·성능평가 결과가 아닙니다. 세부 근거는 completion.json을 따릅니다.\n"
                if cfg.get("final16")
                else (
                    (
                        "# 180초 short-cycle 종료\n\n재시도 없는 clean 판정은 completion.json의 clean_cycle을 따릅니다.\n\n"
                        if cfg.get("short_cycle")
                        else "# 예비45 유한 수집 완료\n\n"
                    )
                    + "같은45 case의 수집·두19·독립품질·9pair·AI45검사와 실제서비스종료/16인스턴스반환을 확인했습니다. 학습·모델성능평가 결과가 아닙니다.\n\n세부 수치와 실패/시도 계보는 completion.json과 하위 task/quality.json을 따릅니다.\n"
                )
            )
            state["status"] = result["status"]
        except BaseException as exc:
            state.update(
                status="BLOCKED_STOPPED", reason=f"{type(exc).__name__}: {exc}"
            )
            write(flow / "failure.json", state)
            raise
        finally:
            state.update(finished_utc=now(), next_stage_authorized=False)
            write(flow / "operator/state.json", state)


def prepare_short(task, handoff):
    """Seal a new short draft into the same finite runner; never reuse480s case data."""
    pilot45.short_authorization()
    task = pilot45.task_path(task)
    cfg = read(task / "config.json")
    manifest = read(task / "manifest.json")
    pilot45.validate_manifest(manifest)
    if (
        not cfg.get("short_cycle")
        or cfg.get("sealed")
        or (
            read(task / "ledger.json")["attempts"]
            and not (cfg.get("short_repair") or cfg.get("short_resume"))
        )
    ):
        raise ValueError(
            "Only a new unclaimed180s draft can receive a final source capsule"
        )
    if cfg.get("short_repair"):
        verify(cfg["short_repair"]["files"])
    if cfg.get("short_resume"):
        verify(cfg["short_resume"]["files"])
    name = task.name.removesuffix("-45")
    if name == task.name:
        raise ValueError("Short worker task must have explicit -45 suffix")
    flow = pilot45.task_path(HOME / "tasks" / name)
    if flow.exists():
        raise ValueError(
            "Existing short flow must never be overwritten or resubmitted"
        )
    spec = pilot_bridge.dependencies(handoff)
    if spec.get("task_id") != "SHORT_CYCLE_20261107":
        raise ValueError(
            "Explicit new short contract required; historical native6 cannot replace it"
        )
    if any(
        k in spec["paths"] for k in ("native_completion", "development_receipt")
    ):
        raise ValueError(
            "New short contract must not consume historical development6/pilot READY"
        )
    binding = ai19.producer_binding()
    from learning.admission19 import check_producer

    check_producer(spec["producer"], binding)
    snapshot = task / "dependency-handoff.json"
    write(snapshot, spec)
    write(
        task / "expected-producer.json",
        {
            k: v
            for k, v in binding.items()
            if k not in {"source_hashes", "vehicle_profile_revision"}
        },
    )
    write(task / "ai-manifest.json", pilot_bridge.plan(task))
    write(task / "consumer-capsule.json", pilot_bridge.capsule(task, spec))
    checked = pilot_bridge.check(task, spec)
    if (
        checked["result"].get("actual_data_verified") is not True
        or checked["result"].get("phase") != "prerequisites"
    ):
        raise ValueError(
            "Actual static short capsule prerequisites did not pass"
        )
    sources = pilot45.frozen_sources(
        task,
        [
            snapshot,
            pilot45.SHORT_AUTH,
            task / "ai-manifest.json",
            task / "consumer-capsule.json",
            Path(checked["receipt"]),
            *map(Path, spec["paths"].values()),
            *map(Path, cfg.get("short_repair", {}).get("files", {})),
            *map(Path, cfg.get("short_resume", {}).get("files", {})),
            *(
                Path(e["reuse_audit"]["path"])
                for e in read(task / "ledger.json")["cases"].values()
                if e.get("reuse_audit")
            ),
        ],
    )
    sources.update(
        {
            str(ROOT / p): h
            for p, h in read(task / "consumer-capsule.json")["files"].items()
        }
    )
    cfg.update(
        sealed=True,
        parallel16=True,
        capture_timing=True,
        stop_on_first_physical_failure=True,
        producer=binding,
        source_sha256=sources,
        dependency_handoff=str(snapshot),
        ai_ready=checked,
        policy_bundle=pilot_bridge.policy_bundle(spec),
        paired_cli_options=spec["paired_cli_options"],
        validated_flight_workers=None,
        report_directory=str(
            ROOT
            / "docs/data/reports/short-cycle-20261107/attempt-01"
            / task.name
        ),
    )
    write(task / "config.json", cfg)
    (task / "config.sha256").write_text(sha(task / "config.json") + "\n")
    write(
        task / "operator/state.json",
        dict(
            status="READY_NOT_STARTED",
            task=task.name,
            scope_id="SHORT_CYCLE_20261107",
            updated_utc=now(),
            no_fit=True,
        ),
    )
    flow.mkdir()
    fc = copy.deepcopy(cfg)
    fc.update(
        flow=True,
        parallel_flow=True,
        parallel16=False,
        pilot_task=str(task),
        service_unit="drone-pretrain-" + name + ".service",
        wall_seconds=24 * 3600,
    )
    fc["source_sha256"][str(task / "config.json")] = sha(task / "config.json")
    write(flow / "config.json", fc)
    (flow / "config.sha256").write_text(sha(flow / "config.json") + "\n")
    write(flow / "budget.json", dict(per_case_conservative_increment_bytes=0))
    write(
        flow / "ledger.json",
        dict(cases={}, attempts=[], phases=dict(pilot45=dict(task=str(task)))),
    )
    write(
        flow / "operator/state.json",
        dict(
            status="READY_NOT_STARTED",
            stage="short180",
            task=name,
            updated_utc=now(),
        ),
    )
    return flow


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    args = p.parse_args()
    flow, task = create(args.name)
    print(str(flow))
    print(str(task))
