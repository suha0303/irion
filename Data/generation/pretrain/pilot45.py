"""Finite45-case no-fit pilot. No LLM callbacks, scheduler, training or old ctx6 path."""

import argparse
import fcntl
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from x500_storage import capture

from . import ai19, native_flight
from .common import (
    HOME,
    PYTHON,
    ROOT,
    STORE,
    collector,
    identity,
    now,
    producer_sources,
    pythonpath,
    read,
    runtime_identity,
    service_unit,
    sha,
    verify,
    write,
)
from .pilot19 import (
    COHORT,
    SHORT_COHORT,
    audit_one,
    helper_inventory,
    prepare_one,
)

TASK_ID = "P27_PILOT45_20261107"
AUTH = ROOT / "ai/reports/p27-pilot45-20261107/attempt-01/authorization.json"
RECOVERY_AUTH = (
    ROOT / "ai/reports/p27-pilot45-20261107/recovery-01/authorization.json"
)
SHORT_AUTH = (
    ROOT / "ai/reports/short-cycle-20261107/attempt-01/authorization.json"
)
REPORT = ROOT / "docs/data/reports/p27-pilot45-20261107/attempt-01"
REGISTRY = ROOT / "Data/operations/claims/pilot45-registry.jsonl"
KINDS = (
    "hover",
    "altitude",
    "line",
    "ellipse",
    "figure8",
    "lissajous",
    "yaw",
    "mixed",
    "random",
)
ENVIRONMENTS = ("calm", "low", "strong")


class Blocked(RuntimeError):
    pass


def task_path(path):
    path = ai19.no_links(path)
    if path.parent != HOME / "tasks":
        raise ValueError("Registered pretrain task required")
    return path


def authorization(path=AUTH):
    value = read(ai19.no_links(path))
    if (
        value.get("task_id") != TASK_ID
        or value.get("authorized", {}).get("pilot_runs") != 45
        or value.get("authorized", {}).get("per_run_requested_mission_seconds")
        != 480
        or not value.get("not_authorized", {}).get("training")
        or not value.get("not_authorized", {}).get("llm_monitoring")
    ):
        raise Blocked(
            "Explicit finite45/no-training/no-LLM-monitor authorization required"
        )
    return value


def mission_contract(duration):
    if type(duration) is not int or duration not in (180, 480):
        raise ValueError(
            "Only explicit180s short-cycle or default480s missions are supported"
        )
    return (45, 60, 75) if duration == 180 else (180, 240, 300)


def short_authorization():
    value = read(SHORT_AUTH)
    scope = value.get("scope", {})
    if value.get("task_id") != "SHORT_CYCLE_20261107" or any(
        scope.get(k) != v
        for k, v in dict(
            proposed_mission_seconds=180,
            logical_cases_per_cycle=45,
            flight_workers=16,
            requested_rtf=2,
            aggregate_nonflight_workers=4,
            physical_attempts_max_per_logical_case=3,
            training_fit_model_forward_gpu=False,
            LR108_collection=False,
            main1674_collection=False,
            production_default_mission_seconds=480,
        ).items()
    ):
        raise Blocked(
            "Explicit no-fit180s45-case short-cycle authority required"
        )
    return value


def build_cases(seed, *, duration=480, case_prefix="p27"):
    """Freeze geometry before flight; pairs share inputs, never raw or retry identities."""
    sys.path.insert(0, str(ROOT / "simulation/sim/src"))
    from capstone_sim import (
        bounded_wind,
        flight_program,
        normal_variation,
        random_flight,
    )

    offsets = mission_contract(duration)
    rows = []

    def operation(kind, env, value):
        spec = normal_variation.sample("hover_normal", value)
        spec.update(
            soc=0.9,
            temperature=298.15,
            payload=0.0,
            wind_e=0.0,
            wind_n=0.0,
            gust_sigma=0.0,
            wind_profile=bounded_wind.profile(env, value % 360, 1.0),
        )
        mode = kind if kind in {"mixed", "random"} else "basic"
        index = flight_program.BASIC.index(kind) if mode == "basic" else 0
        return flight_program.build(
            spec,
            mode,
            value,
            duration,
            index,
            random_revision=random_flight.BOUNDED_REVISION,
        )

    def add(
        cid,
        value,
        kind,
        env,
        category,
        group,
        op,
        family=None,
        timing=None,
        offset=None,
    ):
        abnormal = category == "anomaly"
        scenario = (
            ("battery_aging_" if family == "battery" else "propulsion_loss_")
            + timing
            if abnormal
            else "hover_normal"
        )
        rows.append(
            dict(
                case_id=cid,
                role="anomaly_test" if abnormal else "normal_test",
                cohort=SHORT_COHORT if duration == 180 else COHORT,
                category=category,
                group_id=group,
                retry_family_id=cid,
                paired_case_id=f"x500-{kind}-seed{value}",
                seed=value,
                kind=kind,
                environment=env,
                operation_variation=op,
                scenario=scenario,
                duration=duration,
                battery_stage="rpt_16"
                if abnormal and family == "combined"
                else None,
                fault_after_mission_s=offset
                if abnormal and timing in {"step", "ramp"}
                else None,
                anomaly_family=family,
                anomaly_timing=timing,
                attempts_max=3,
            )
        )

    for mi, kind in enumerate(KINDS):
        for ei, env in enumerate(ENVIRONMENTS):
            value = (seed + 1009 * (3 * mi + ei + 1)) % 2**32
            cid = f"{case_prefix}-normal-{mi:02}-{ei}"
            add(
                cid,
                value,
                kind,
                env,
                "standalone_normal",
                cid,
                operation(kind, env, value),
            )
    for i in range(9):
        family, timing = (
            ("battery", "propulsion", "combined")[i // 3],
            ("existing", "step", "ramp")[i % 3],
        )
        kind, env = KINDS[i], ENVIRONMENTS[(i + i // 3) % 3]
        value = (seed + 1009 * (100 + i)) % 2**32
        group = f"{case_prefix}-pair-{i:02}"
        op = operation(kind, env, value)
        offset = offsets[(i + i // 3) % 3]
        add(group + "-normal", value, kind, env, "paired_normal", group, op)
        add(
            group + "-anomaly",
            value,
            kind,
            env,
            "anomaly",
            group,
            op,
            family,
            timing,
            offset,
        )
    validate_cases(rows, duration=duration)
    return rows


def validate_cases(rows, *, duration=480):
    from collections import Counter

    offsets = mission_contract(duration)
    if (
        len(rows) != 45
        or len({r["case_id"] for r in rows}) != 45
        or Counter(r["role"] for r in rows)
        != {"normal_test": 36, "anomaly_test": 9}
        or Counter(r["category"] for r in rows)
        != {"standalone_normal": 27, "paired_normal": 9, "anomaly": 9}
        or any(
            r["cohort"] != (SHORT_COHORT if duration == 180 else COHORT)
            or r["duration"] != duration
            or r["attempts_max"] != 3
            for r in rows
        )
    ):
        raise ValueError("Exact frozen45/no-fit composition required")
    base = [r for r in rows if r["category"] == "standalone_normal"]
    if {(r["kind"], r["environment"]) for r in base} != {
        (k, e) for k in KINDS for e in ENVIRONMENTS
    }:
        raise ValueError("All27 operation cells required")
    anomalies = [r for r in rows if r["category"] == "anomaly"]
    if {(r["anomaly_family"], r["anomaly_timing"]) for r in anomalies} != {
        (f, t)
        for f in ("battery", "propulsion", "combined")
        for t in ("existing", "step", "ramp")
    }:
        raise ValueError("All9 anomaly meanings required")
    for r in anomalies:
        controls = [
            c
            for c in rows
            if c["category"] == "paired_normal"
            and c["group_id"] == r["group_id"]
        ]
        if len(controls) != 1 or any(
            controls[0][k] != r[k]
            for k in (
                "seed",
                "kind",
                "environment",
                "operation_variation",
                "paired_case_id",
                "duration",
            )
        ):
            raise ValueError(
                "Independent one-to-one control with matched requested conditions required"
            )
    sys.path.insert(0, str(ROOT / "simulation/sim/src"))
    from capstone_sim.normal_variation import validate

    for row in rows:
        spec = row["operation_variation"]
        if (
            spec["soc"] != 0.9
            or spec["temperature"] != 298.15
            or spec["payload"] != 0
            or spec["path"]["kind"] != row["kind"]
            or spec["wind_profile"]["environment"] != row["environment"]
            or any(spec[k] != 0 for k in ("wind_e", "wind_n", "gust_sigma"))
        ):
            raise ValueError(
                "Requested operating condition differs from pilot contract"
            )
        validate(spec)
        if row["category"] == "anomaly":
            prefix = (
                "battery_aging_"
                if row["anomaly_family"] == "battery"
                else "propulsion_loss_"
            )
            stage = "rpt_16" if row["anomaly_family"] == "combined" else None
            if (
                row["scenario"] != prefix + row["anomaly_timing"]
                or row["battery_stage"] != stage
                or (
                    row["anomaly_timing"] != "existing"
                    and row["fault_after_mission_s"] not in offsets
                )
            ):
                raise ValueError("Anomaly endpoint/timing meaning changed")
        elif (
            row["scenario"] != "hover_normal"
            or row["battery_stage"] is not None
        ):
            raise ValueError(
                "Normal control must keep healthy source selection"
            )
    return rows


def protection():
    """Integrate all existing raw/parent/hash/pair identities and persistent old claims."""
    records = []
    for p in sorted((ROOT / "simulation/runs").glob("*/run.json")):
        if p.is_symlink() or p.parent.is_symlink():
            raise Blocked("Symlink in prior raw inventory")
        meta = read(p)
        scenario_path = p.parent / "evaluation/scenario.json"
        integrity_path = p.parent / "integrity.json"
        scenario = read(scenario_path) if scenario_path.exists() else {}
        integrity = read(integrity_path) if integrity_path.exists() else {}
        records.append(
            dict(
                run_id=p.parent.name,
                parent_attempt_id=meta.get("parent_attempt_id"),
                paired_case_id=scenario.get("paired_case_id"),
                seed=scenario.get("seed"),
                scenario=scenario.get("name"),
                cohort="preexisting_excluded",
                raw_sha256=integrity.get("observations/streams.jsonl", {}).get(
                    "sha256"
                ),
            )
        )
    claims = {}
    for p in sorted((ROOT / "Data/operations/claims").glob("*.claim")):
        claims[str(p)] = sha(p)
        prior = read(p)
        if "seed" in prior and "scenario" in prior:
            records.append(
                dict(
                    run_id=prior.get("run_id"),
                    seed=prior["seed"],
                    scenario=prior["scenario"],
                    group_id=prior.get("group_id"),
                    paired_case_id=prior.get("pair_id"),
                    parent_attempt_id=prior.get("parent_attempt_id"),
                    cohort="preexisting_claim_excluded",
                    claim=str(p),
                )
            )
    return dict(
        schema="drone-data.pilot45.protection/1",
        records=records,
        existing_claims=claims,
        policy="All prior raw including development/failed/training are excluded; never delete",
    )


def global_protection(
    protected, rows, name, development_ids=(), *, cohort=COHORT
):
    reference_ids = set(development_ids)
    for rel in (
        "wind5-support-20261107/development-reference-runs.json",
        "wind5-gust-support-20261107/development-reference-runs.json",
        "wind5-gust-support-20261107/superseded-mean-ou-runs.json",
    ):
        source = ROOT / "docs/data/reports" / rel
        if not source.exists():
            raise Blocked(
                "Required development exclusion index missing: " + str(source)
            )
        value = read(source)
        for item in value.get("records", []) + value.get("failed_runs", []):
            rid = item.get("run_id")
            if not rid:
                rid = Path(
                    item.get("raw", item.get("run", item.get("raw_path", "")))
                ).name
            if rid:
                reference_ids.add(rid)
    entries = []
    for row in protected["records"]:
        keys = []
        for field, prefix in (
            ("run_id", "run:"),
            ("parent_attempt_id", "run:"),
            ("paired_case_id", "pair:"),
            ("group_id", "group:"),
            ("raw_sha256", "raw-sha256:"),
        ):
            if row.get(field):
                keys.append(prefix + row[field])
        if not keys:
            continue
        forbidden = [
            "pilot_input_check",
            "final",
            "final_test",
            "final_input_check",
        ]
        if cohort == SHORT_COHORT:
            forbidden.append("short_cycle_input_check")
        if cohort == "final16_development_reference":
            forbidden.append("final16_input_check")
        if row.get("run_id") in reference_ids:
            forbidden += ["fit", "input_check", "calibration"]
        entries.append(
            dict(
                identities=keys, forbidden_uses=forbidden, purpose=row["cohort"]
            )
        )
    for rid in reference_ids:
        entries.append(
            dict(
                identities=["run:" + rid],
                forbidden_uses=[
                    "pilot_input_check",
                    "fit",
                    "input_check",
                    "calibration",
                    "final",
                    "final_test",
                    "final_input_check",
                ],
                purpose="development_reference",
            )
        )
        if cohort == SHORT_COHORT:
            entries[-1]["forbidden_uses"].append("short_cycle_input_check")
        if cohort == "final16_development_reference":
            entries[-1]["forbidden_uses"].append("final16_input_check")
    for row in rows:
        entries.append(
            dict(
                identities=["case:" + name + ":" + row["case_id"]],
                forbidden_uses=[
                    "fit",
                    "input_check",
                    "calibration",
                    "final",
                    "final_test",
                    "final_input_check",
                ],
                purpose=cohort,
            )
        )
    return dict(
        schema="drone-ai.global-protection/1",
        entries=entries,
        development_reference_ids=sorted(reference_ids),
        task=name,
        scope="pilot intake and no-fit/no-final",
    )


def resources(remaining=0, per_case=0):
    snapshot = collector().memory_snapshot()
    snapshot["disk_free_bytes"] = shutil.disk_usage(STORE).free
    if snapshot["available_bytes"] < 1_000_000_000:
        raise Blocked("Available RAM below1GB; no new admission")
    if snapshot["disk_free_bytes"] < 50_000_000_000 + remaining * per_case:
        raise Blocked(
            "Disk50GB reserve plus remaining conservative incremental budget unavailable"
        )
    return snapshot


def estimate():
    """Measured file sizes plus an explicit conservative duplication/output budget."""
    index = (
        ROOT
        / "docs/data/reports/wind5-gust-support-20261107/development-reference-runs.json"
    )
    sizes, consumer = [], []
    if index.exists():
        for row in read(index)["records"]:
            raw = Path(row["raw_path"])
            if raw.exists():
                sizes.append(
                    sum(p.stat().st_size for p in raw.rglob("*") if p.is_file())
                )
                consumer.append(
                    sum((raw / p).stat().st_size for p in ai19.CONSUMER)
                )
    raw_max = max(sizes, default=2_000_000_000)
    consumer_max = max(consumer, default=2_000_000_000)
    per_case = int(1.25 * (2.05 * raw_max + consumer_max + 128 * 1024**2))
    return dict(
        measured_reference_runs=len(sizes),
        measured_max_raw_bytes=raw_max if sizes else None,
        measured_max_consumer_bytes=consumer_max if consumer else None,
        per_case_conservative_increment_bytes=per_case,
        total_increment_bytes=45 * per_case,
        reserve_bytes=50_000_000_000,
        basis="1.25*(raw + archive upper1.05raw + consumer +128MiB outputs); not pilot measurement",
    )


def validate_manifest(manifest):
    if manifest.get("final16"):
        from .final16 import validate_manifest as validate_final16

        return validate_final16(manifest)
    if manifest.get("short_cycle"):
        short_authorization()
        duration = manifest.get("mission_seconds")
        if duration != 180 or manifest["cases"] != build_cases(
            manifest["seed_base"], duration=180, case_prefix="s180"
        ):
            raise ValueError(
                "Short-cycle must retain the exact new180s case/seed/condition definition"
            )
        return validate_cases(manifest["cases"], duration=duration)
    if not manifest.get("single_integration"):
        return validate_cases(manifest["cases"])
    expected = build_cases(manifest["seed_base"])[
        8
    ]  # line, strong bounded wind
    expected.update(
        case_id="integration19-normal",
        group_id="integration19-normal",
        attempts_max=1,
    )
    expected["retry_family_id"] = expected["case_id"]
    if manifest.get("recovery_of"):
        previous = task_path(manifest["recovery_of"])
        old = read(previous / "manifest.json")
        if old["seed_base"] != manifest["seed_base"] or old["cases"] != [
            expected
        ]:
            raise ValueError(
                "Recovery must retain the exact old case, seed and conditions"
            )
        expected["attempts_max"] = 3
    if manifest["cases"] != [expected]:
        raise ValueError(
            "Integration scope is exactly one healthy480s line/strong case, one attempt"
        )
    return manifest["cases"]


def fresh_cases(rows, protected):
    prior = {(r.get("scenario"), r.get("seed")) for r in protected["records"]}
    pairs = {
        r.get("paired_case_id")
        for r in protected["records"]
        if r.get("paired_case_id")
    }
    if any(
        (r["scenario"], r["seed"]) in prior or r["paired_case_id"] in pairs
        for r in rows
    ):
        raise Blocked(
            "Seed/scenario or paired family already exists; no regeneration"
        )


def frozen_sources(task, extra=()):
    sources = helper_inventory()
    sources.update(
        {
            str(p): sha(p)
            for p in (ROOT / "Data/generation/pretrain").glob("*.py")
        }
    )
    sources.update(
        {str(ROOT / "simulation" / p): h for p, h in producer_sources().items()}
    )
    for p in (
        ROOT / "ai/x500_storage.py",
        ROOT / "Data/generation/drone-pretrain",
        ROOT / "Data/generation/tools/collect_long_runs.py",
        AUTH,
        task / "manifest.json",
        task / "protection.json",
        task / "global-protection.json",
        task / "budget.json",
        task / "expected-producer.json",
        *extra,
    ):
        sources[str(p)] = sha(p)
    return sources


def recovery_authorization():
    value = read(RECOVERY_AUTH)
    if value.get("task_id") != TASK_ID or any(
        value.get("authorized", {}).get(k) is not True
        for k in (
            "Data_launch_environment_fix",
            "standalone_integration_retest",
            "pilot_preprocessing_and_independent_input_quality",
        )
    ):
        raise Blocked(
            "Explicit source-fix/integration/pilot recovery authorization required"
        )
    return value


def recovery_context(previous, receipt):
    recovery_authorization()
    previous = task_path(previous)
    ledger = read(previous / "ledger.json")
    manifest = read(previous / "manifest.json")
    evidence = read(receipt)
    if (
        evidence.get("status") != "PASSED_NO_FLIGHT"
        or evidence.get("probe", {}).get("run_flight_imported") is not True
    ):
        raise Blocked(
            "Actual Sim interpreter/import preflight must pass before recovery"
        )
    if len(ledger["attempts"]) != 1 or len(manifest["cases"]) != 1:
        raise Blocked(
            "This recovery only covers the recorded single launcher failure"
        )
    cid = manifest["cases"][0]["case_id"]
    entry = ledger["cases"][cid]
    log = previous / "pilot/execution" / cid / "flight-01.log"
    if (
        entry.get("run")
        or entry["attempts"][0].get("run_id")
        or "ModuleNotFoundError: No module named 'pymavlink'"
        not in log.read_text()
    ):
        raise Blocked(
            "Prior completed/unknown raw cannot be replaced by a new flight"
        )
    unit, pid, props = service_identity(previous)
    if pid or props.get("ActiveState") in {
        "active",
        "activating",
        "deactivating",
    }:
        raise Blocked("Previous owned task must have terminated")
    audit = (
        ROOT
        / "ai/reports/ai19-single-run-integration-20261107/attempt-02/result-review-01/termination-audit.json"
    )
    proof = read(audit)
    if proof.get("actual_raw_runs") != 0 or not proof.get("checks", {}).get(
        "shared_locks_free"
    ):
        raise Blocked("Previous no-raw/termination proof missing")
    paths = [
        previous / n for n in ("config.json", "manifest.json", "ledger.json")
    ]
    paths += [log, audit, Path(receipt), RECOVERY_AUTH]
    return dict(
        previous_task=str(previous),
        case_id=cid,
        previous_attempt=1,
        namespace=previous.name,
        native_parent_run_id=None,
        native_parent_reason="Previous launcher failed before a native run/UUID existed",
        files={str(p): sha(p) for p in paths},
        environment_receipt=str(Path(receipt).absolute()),
    )


def short_repair_context(previous, receipt):
    """Admit only hash-proven seed-FFI failures after an explicit producer correction."""
    short_authorization()
    previous = task_path(previous)
    receipt = ai19.no_links(receipt)
    accepted = read(receipt)
    if (
        accepted.get("schema")
        != "drone-data.short-cycle.deterministic-repair/1"
        or accepted.get("task_id") != "SHORT_CYCLE_20261107"
        or accepted.get("status") != "ACCEPTED_STATIC_REPAIR_RETRY_ONLY"
        or accepted.get("reason") != "gazebo_uint32_seed_bridge"
        or accepted.get("previous_task") != str(previous)
        or accepted.get("physical_attempts_max") != 3
        or accepted.get("authorization")
        != dict(path=str(SHORT_AUTH), sha256=sha(SHORT_AUTH))
    ):
        raise Blocked(
            "Explicit bounded same-seed producer repair acceptance required"
        )
    verify(accepted["files"])
    for member in (
        "config.json",
        "manifest.json",
        "ledger.json",
        "operator/socket-cleanup.json",
    ):
        path = previous / member
        if accepted["files"].get(str(path)) != sha(path):
            raise Blocked(
                "Complete previous task identity and cleanup must be hash-pinned"
            )
    if accepted["files"].get(str(previous / "ledger.json")) != sha(
        previous / "ledger.json"
    ):
        raise Blocked(
            "Repair acceptance must pin the complete prior attempt ledger"
        )
    cfg, manifest, ledger = (
        read(previous / n)
        for n in ("config.json", "manifest.json", "ledger.json")
    )
    validate_manifest(manifest)
    if (
        not cfg.get("short_cycle")
        or set(ledger["cases"]) != {c["case_id"] for c in manifest["cases"]}
        or any(
            ledger["cases"][c["case_id"]].get("case") != c
            for c in manifest["cases"]
        )
        or any(e.get("run") for e in ledger["cases"].values())
    ):
        raise Blocked(
            "This repair admits failed startup only; no successful raw may be regenerated"
        )
    unit, pid, props = service_identity(previous)
    if pid or props.get("ActiveState") != "failed" or props.get("ControlGroup"):
        raise Blocked(
            "Prior owned startup-failure service must be completely stopped"
        )
    cleanup = read(previous / "operator/socket-cleanup.json")
    if cleanup.get("status") != "CLEANUP_COMPLETE" or cleanup.get("errors"):
        raise Blocked("Prior instance cleanup must be proven")
    ai19.ai_modules()
    from learning.admission19 import check_producer

    check_producer(accepted["old_producer"], cfg["producer"])
    check_producer(accepted["new_producer"], ai19.producer_binding())
    if any(
        accepted["old_producer"][key] == accepted["new_producer"][key]
        for key in ("source_manifest_sha256", "runtime_sha256")
    ):
        raise Blocked(
            "A corrected producer source and runtime seal are both required"
        )
    for key in ("simulation_repair", "AI_handoff"):
        item = accepted[key]
        if sha(ai19.no_links(item["path"])) != item["sha256"]:
            raise Blocked("Reviewed producer repair evidence changed: " + key)
    indexed = {r["case_id"]: r for r in accepted["failures"]}
    prior_attempts = ledger["attempts"]
    if len(indexed) != 16 or len(prior_attempts) != 16:
        raise Blocked(
            "Exactly the16 observed C02 startup failures must be preserved"
        )
    if set(indexed) != {a["case_id"] for a in prior_attempts}:
        raise Blocked(
            "Repair case denominator differs from the complete failure ledger"
        )
    for a in prior_attempts:
        item = indexed[a["case_id"]]
        case = ledger["cases"][a["case_id"]]["case"]
        raw = ROOT / "simulation/runs" / a["run_id"]
        log = raw / "native_logs/gazebo.log"
        for member in (
            "run.json",
            "evaluation/scenario.json",
            "reference/source_lock.json",
        ):
            path = raw / member
            if accepted["files"].get(str(path)) != sha(path):
                raise Blocked(
                    "Exact failed raw identity/producer snapshot must be hash-pinned"
                )
        meta, scenario = (
            read(raw / "run.json"),
            read(raw / "evaluation/scenario.json"),
        )
        if (
            a["state"] != "FAILED_PRESERVED"
            or a["number"] != 1
            or a.get("retry_allowed") is not False
            or a.get("parent_attempt_id") is not None
            or item["run_id"] != a["run_id"]
            or item["attempt"] != a["number"]
            or item["seed"] != case["seed"]
            or not 2**31 <= case["seed"] < 2**32
            or scenario["seed"] != case["seed"]
            or scenario.get("producer_lock_sha256")
            != cfg["producer"]["runtime_sha256"]
            or scenario["operation_variation"] != case["operation_variation"]
            or meta["status"] != "failed"
            or meta.get("primary_failure", {}).get("stage") != "waiting"
            or accepted["files"].get(str(log)) != sha(log)
            or f"integer {case['seed']} too big to convert to `int'"
            not in log.read_text()
            or any(
                identity(p) is not None
                for p in meta.get("owned_process_groups", {}).values()
            )
        ):
            raise Blocked(
                "Only the exact proven no-flight uint32 bridge failure may be retried"
            )
    files = dict(accepted["files"], **{str(receipt): sha(receipt)})
    return dict(
        previous_task=str(previous),
        namespace=cfg["lineage_namespace"],
        receipt=str(receipt),
        files=files,
        failures=indexed,
        source_ledger_sha256=sha(previous / "ledger.json"),
    )


def short_repair_allows(config, case, prior):
    repair = config.get("short_repair")
    if (
        not repair
        or not prior
        or prior.get("source_task") != repair["previous_task"]
    ):
        return False
    expected = repair["failures"].get(case["case_id"])
    if (
        not expected
        or prior["state"] != "FAILED_PRESERVED"
        or prior.get("retry_allowed") is not False
        or prior["number"] != expected["attempt"]
        or prior.get("run_id") != expected["run_id"]
        or case["seed"] != expected["seed"]
    ):
        raise Blocked(
            "Repair retry differs from the exact accepted failed parent"
        )
    verify(repair["files"])
    return True


def init(
    name,
    seed=None,
    *,
    single=False,
    recovery_from=None,
    launch_receipt=None,
    dependency_handoff=None,
    short_cycle=False,
    short_recovery_from=None,
    correction_receipt=None,
):
    if short_cycle:
        short_authorization()
        if single or recovery_from is not None:
            raise ValueError(
                "Short-cycle is a new independent45-case test, not480s recovery"
            )
    else:
        authorization()
    if bool(short_recovery_from) != bool(correction_receipt) or (
        short_recovery_from and not short_cycle
    ):
        raise ValueError(
            "Short repair requires its previous task and exact correction receipt"
        )
    if recovery_from and launch_receipt is None:
        raise ValueError(
            "Recovery requires the actual launch-environment receipt"
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("Safe task name required")
    recovery = (
        recovery_context(recovery_from, launch_receipt)
        if recovery_from
        else None
    )
    short_repair = (
        short_repair_context(short_recovery_from, correction_receipt)
        if short_recovery_from
        else None
    )
    if short_repair:
        if seed is not None:
            raise Blocked(
                "Short repair retains the original seed without replacement"
            )
        seed = read(Path(short_repair["previous_task"]) / "manifest.json")[
            "seed_base"
        ]
    if recovery:
        if not single or seed is not None:
            raise Blocked(
                "Recovery reuses the prior frozen seed; no replacement seed"
            )
        seed = read(Path(recovery["previous_task"]) / "manifest.json")[
            "seed_base"
        ]
    random_seed = seed is None
    seed = secrets.randbits(32) if random_seed else seed
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must fit uint32")
    task = task_path(HOME / "tasks" / name)
    if task.exists():
        raise ValueError("Existing task preserved; use status/seal/resume")
    rows = (
        build_cases(seed, duration=180, case_prefix="s180")
        if short_cycle
        else build_cases(seed)
    )
    if single:
        rows = [rows[8]]
        rows[0].update(
            case_id="integration19-normal",
            group_id="integration19-normal",
            attempts_max=1,
        )
        rows[0]["retry_family_id"] = rows[0]["case_id"]
        if recovery:
            rows[0]["attempts_max"] = 3
    protected = protection()
    if short_repair:
        owned = {r["run_id"] for r in short_repair["failures"].values()}
        protected["records"] = [
            r for r in protected["records"] if r.get("run_id") not in owned
        ]
    fresh_cases(rows, protected)
    task.mkdir(parents=True)
    (task / "operator").mkdir()
    write(
        task / "manifest.json",
        dict(
            schema="drone-data.pilot45.manifest/1",
            task_id="SHORT_CYCLE_20261107" if short_cycle else TASK_ID,
            seed_base=seed,
            seed_origin="preserved_predecessor"
            if recovery or short_repair
            else "OS_random"
            if random_seed
            else "user",
            single_integration=single,
            recovery_of=recovery["previous_task"] if recovery else None,
            cases=rows,
            short_cycle=short_cycle,
            mission_seconds=180 if short_cycle else 480,
        ),
    )
    (task / "manifest.sha256").write_text(sha(task / "manifest.json") + "\n")
    write(task / "protection.json", protected)
    write(
        task / "global-protection.json",
        global_protection(
            protected,
            rows,
            recovery["namespace"]
            if recovery
            else short_repair["namespace"]
            if short_repair
            else name,
            **(dict(cohort=SHORT_COHORT) if short_cycle else {}),
        ),
    )
    budget = estimate()
    budget["planned_cases"] = len(rows)
    budget["total_increment_bytes"] = (
        len(rows) * budget["per_case_conservative_increment_bytes"]
    )
    write(task / "budget.json", budget)
    config = dict(
        mode="pilot45",
        collection_profile="integration19_common19"
        if single
        else "pilot45_common19",
        single_integration=single,
        planned_cases=len(rows),
        task_id=TASK_ID,
        owner_uid=os.getuid(),
        service_unit="drone-pretrain-" + name + ".service",
        python_runtime=runtime_identity(),
        external_storage=str(STORE),
        storage_policy=capture(STORE),
        seed_base=seed,
        flight_workers=1,
        export_workers=1,
        io_workers=1,
        workers=1,
        audit_workers=1,
        cpu_workers_total_max=1,
        validated_flight_workers=None,
        min_available_ram_gb=1,
        use_available_ram=True,
        use_swap=True,
        unit_attempts_max=1 if single else 3,
        wall_seconds=7 * 24 * 3600,
        training_authorized=False,
        fit_authorized=False,
        sealed=False,
        source_sha256={},
        authorization=dict(
            path=str(SHORT_AUTH if short_cycle else AUTH),
            sha256=sha(SHORT_AUTH if short_cycle else AUTH),
        ),
    )
    if short_cycle:
        config.update(
            short_cycle=True,
            mission_seconds=180,
            lineage_namespace=name,
            flight_workers=16,
            requested_real_time_factor=2,
            cpu_workers_total_max=4,
            export_workers=4,
            io_workers=4,
            workers=4,
            audit_workers=4,
            scope_id="SHORT_CYCLE_20261107",
            task_id="SHORT_CYCLE_20261107",
            collection_profile="short180_common19",
            authorization=dict(path=str(SHORT_AUTH), sha256=sha(SHORT_AUTH)),
        )
        if short_repair:
            config.update(
                short_repair=short_repair,
                lineage_namespace=short_repair["namespace"],
            )
    if recovery:
        config.update(
            recovery=recovery,
            lineage_namespace=recovery["namespace"],
            unit_attempts_max=3,
            authorization=dict(
                path=str(RECOVERY_AUTH), sha256=sha(RECOVERY_AUTH)
            ),
        )
    if single:
        binding = ai19.producer_binding()
        expected = {
            k: binding[k]
            for k in (
                "battery_policy",
                "battery_revision",
                "catalog_sha256",
                "propulsion_revision",
                "propulsion_catalog_sha256",
                "runtime_sha256",
                "source_manifest_sha256",
            )
        }
        write(task / "expected-producer.json", expected)
        extra = [Path(p) for p in recovery["files"]] if recovery else []
        if dependency_handoff:
            from . import pilot_bridge

            spec = pilot_bridge.dependencies(dependency_handoff)
            snapshot = task / "dependency-handoff.json"
            write(snapshot, spec)
            config.update(
                dependency_handoff=str(snapshot),
                policy_bundle=pilot_bridge.policy_bundle(spec),
                paired_cli_options=spec["paired_cli_options"],
            )
            extra += [snapshot, *(Path(p) for p in spec["paths"].values())]
        config.update(
            sealed=True,
            producer=binding,
            source_sha256=frozen_sources(task, extra),
            paired_run_options={},
            integration_scope="one standalone raw -> shared frames -> two19 readers -> audit -> STOP; no paired admission",
        )
    write(task / "config.json", config)
    (task / "config.sha256").write_text(sha(task / "config.json") + "\n")
    write(
        task / "operator/state.json",
        dict(
            status="READY_NOT_STARTED" if single else "BLOCKED_DEPENDENCIES",
            stage="prepare",
            reason="Coordinator starts one standalone integration only"
            if single
            else "Explicit short180 source/policy seal and AI static gate required"
            if short_cycle
            else "Final paired6 seal and AI/Data intake receipts required",
            task=name,
            updated_utc=now(),
        ),
    )
    ledger = dict(
        schema="drone-data.pilot45.ledger/1",
        cases={c["case_id"]: dict(case=c, attempts=[]) for c in rows}
        if short_cycle
        else {},
        attempts=[],
        cohort=SHORT_COHORT if short_cycle else COHORT,
        status="NOT_STARTED",
    )
    if recovery:
        prior = read(Path(recovery["previous_task"]) / "ledger.json")
        old = prior["cases"][recovery["case_id"]]
        attempts = [
            dict(
                a,
                source_task=recovery["previous_task"],
                source_ledger_sha256=recovery["files"][
                    str(Path(recovery["previous_task"]) / "ledger.json")
                ],
            )
            for a in old["attempts"]
        ]
        ledger.update(
            cases={recovery["case_id"]: dict(case=rows[0], attempts=attempts)},
            attempts=[dict(a, case_id=recovery["case_id"]) for a in attempts],
        )
    if short_repair:
        previous_ledger = read(
            Path(short_repair["previous_task"]) / "ledger.json"
        )
        for cid, entry in previous_ledger["cases"].items():
            imported = [
                dict(
                    a,
                    source_task=short_repair["previous_task"],
                    source_ledger_sha256=short_repair["source_ledger_sha256"],
                )
                for a in entry["attempts"]
            ]
            ledger["cases"][cid]["attempts"] = imported
            ledger["attempts"].extend(dict(a, case_id=cid) for a in imported)
    write(task / "ledger.json", ledger)
    return task


def seal(task, sim_receipt, ai_receipt):
    """Current native gate failed; no fabricated PASSED summary can unlock45."""
    task = task_path(task)
    if read(task / "config.json").get("single_integration"):
        raise Blocked("Standalone integration cannot be upgraded to paired45")
    reason = "Pilot45 blocked: actual paired6 and reviewed native-to-AI mapping are not implemented"
    write(
        task / "operator/state.json",
        dict(
            status="BLOCKED_DEPENDENCIES",
            task=task.name,
            stage="seal",
            reason=reason,
            simulation_receipt=str(sim_receipt),
            ai_receipt=str(ai_receipt),
            updated_utc=now(),
        ),
    )
    raise Blocked(reason)


def prepare_pilot(
    name, draft, integration, handoff, *, integration_acceptance=None
):
    """Create a new sealed task; the old draft and all failed tasks stay unchanged."""
    from . import pilot_bridge

    recovery_authorization()
    spec = pilot_bridge.dependencies(handoff)
    draft, integration = task_path(draft), task_path(integration)
    old = read(draft / "manifest.json")
    verify(
        {
            str(draft / "manifest.json"): (draft / "manifest.sha256")
            .read_text()
            .strip()
        }
    )
    quality = read(integration / "quality.json")
    if (
        quality["status"] != "INTEGRATION19_QUALITY_PASSED_STOPPED"
        or quality["prepared_and_audited"] != 1
    ):
        raise Blocked(
            "Same-producer standalone integration and both actual views must pass first"
        )
    binding = ai19.producer_binding()
    from learning.admission19 import check_producer

    check_producer(read(integration / "expected-producer.json"), binding)
    if integration_acceptance is None:
        verify(read(integration / "config.json")["source_sha256"])
    else:
        from .integration_reuse import validate

        validate(integration, integration_acceptance)
    task = init(name, old["seed_base"])
    if read(task / "manifest.json")["cases"] != old["cases"]:
        raise Blocked(
            "New final producer changed frozen planned cases; review before flight"
        )
    cfg = read(task / "config.json")
    snapshot = task / "dependency-handoff.json"
    write(snapshot, spec)
    native = read(spec["paths"]["native_completion"])
    integration_raw = [
        Path(e["run"]).name
        for e in read(integration / "ledger.json")["cases"].values()
    ]
    protected = protection()
    write(task / "protection.json", protected)
    write(
        task / "global-protection.json",
        global_protection(
            protected,
            old["cases"],
            task.name,
            [*native["actual_run_ids"], *integration_raw],
        ),
    )
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
    if checked["result"].get("ai_prerequisites_passed") is not True:
        raise Blocked("AI actual prerequisites are not passed")
    extra = [
        snapshot,
        RECOVERY_AUTH,
        integration / "quality.json",
        integration / "expected-producer.json",
        draft / "manifest.json",
        task / "ai-manifest.json",
        task / "consumer-capsule.json",
        Path(checked["receipt"]),
        *(Path(p) for p in spec["paths"].values()),
    ]
    if integration_acceptance is not None:
        extra.append(Path(integration_acceptance))
    sources = frozen_sources(task, extra)
    sources.update(
        {
            str(ROOT / p): h
            for p, h in read(task / "consumer-capsule.json")["files"].items()
        }
    )
    cfg.update(
        sealed=True,
        producer=binding,
        source_sha256=sources,
        ai_ready=checked,
        dependency_handoff=str(snapshot),
        policy_bundle=pilot_bridge.policy_bundle(spec),
        paired_cli_options=spec["paired_cli_options"],
        validated_flight_workers=1,
        authorization=dict(path=str(RECOVERY_AUTH), sha256=sha(RECOVERY_AUTH)),
        report_directory=str(
            ROOT
            / "docs/data/reports/p27-pilot45-20261107/recovery-01"
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
            stage="final_gates",
            updated_utc=now(),
            ai_prerequisites=checked["receipt"],
        ),
    )
    return task


def service_identity(task, *, require_running=False):
    cfg = read(task / "config.json")
    if cfg.get("owner_uid") != os.getuid() or cfg.get("mode") not in {
        "pilot45",
        "main19",
    }:
        raise Blocked("This user's owned collection task required")
    if cfg.get("mode") == "main19":
        from .main_collection import validate

        validate(task)
    unit = service_unit(task, cfg)
    record = task / "operator/service.json"
    if not record.exists() or read(record)["unit"] != unit:
        raise Blocked("No owned service submission receipt")
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=MainPID",
            "--property=ControlGroup",
            "--property=ActiveState",
            "--property=ExecStart",
            "--property=LoadState",
            "--property=SubState",
            "--property=Result",
            "--property=ExecMainCode",
            "--property=ExecMainStatus",
            "--property=InvocationID",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    props = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    unloaded = (
        props.get("LoadState") == "not-found"
        and props.get("ActiveState") == "inactive"
    )
    if not unloaded and (
        str(task) not in props.get("ExecStart", "")
        or "pretrain.cli" not in props.get("ExecStart", "")
    ):
        raise Blocked("Service command does not belong to this frozen task")
    if not unloaded:
        required = {
            "MainPID",
            "ControlGroup",
            "ActiveState",
            "LoadState",
            "SubState",
            "Result",
            "ExecMainCode",
            "ExecMainStatus",
            "InvocationID",
        }
        if required - set(props):
            raise Blocked(
                "Actual service properties missing: "
                + str(sorted(required - set(props)))
            )
    pid = int(props.get("MainPID", 0))
    if pid:
        if Path(f"/proc/{pid}").stat().st_uid != os.getuid() or not any(
            line.startswith("0::") and line.rstrip().endswith("/" + unit)
            for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        ):
            raise Blocked("Service PID/owner/cgroup mismatch")
    if require_running and (not pid or identity(pid) is None):
        raise Blocked("Owned pilot service is not running")
    return unit, pid, props


def preflight(task):
    task = task_path(task)
    cfg = read(task / "config.json")
    if cfg.get("parallel16"):
        from .pilot_parallel import validate_config

        validate_config(cfg)
        if not cfg.get("short_cycle") and not cfg.get("final16"):
            verify(cfg["continuation"]["files"])
    if cfg.get("flow"):
        from .pilot_flow import preflight as flow_preflight

        return flow_preflight(task)
    if not cfg.get("sealed") or cfg.get("owner_uid") != os.getuid():
        raise Blocked("Final dependency seal/owned task missing")
    verify(cfg["source_sha256"])
    if cfg.get("final16"):
        from .final16 import authorization as final16_authorization
        from .final16 import validate_ledger

        final16_authorization()
        validate_ledger(task, pristine=not (task / "operator/claimed").exists())
    else:
        short_authorization() if cfg.get("short_cycle") else authorization()
    if cfg.get("recovery"):
        if not cfg.get("short_cycle"):
            recovery_authorization()
        verify(cfg["recovery"]["files"])
    if cfg.get("short_repair"):
        verify(cfg["short_repair"]["files"])
    if cfg.get("short_resume"):
        verify(cfg["short_resume"]["files"])
    manifest = read(task / "manifest.json")
    validate_manifest(manifest)
    if cfg.get("single_integration") != manifest.get(
        "single_integration"
    ) or cfg.get("planned_cases") != len(manifest["cases"]):
        raise Blocked("Frozen task scope mismatch")
    if not cfg.get("single_integration"):
        from .pilot_bridge import dependencies

        if not cfg.get("short_cycle") and not cfg.get("final16"):
            recovery_authorization()
        dependencies(cfg["dependency_handoff"])
        checked = cfg.get("ai_ready", {})
        if (
            not checked
            or sha(checked["receipt"]) != checked["sha256"]
            or not (
                checked["result"].get("actual_data_verified") is True
                and checked["result"].get("phase") == "prerequisites"
                if cfg.get("short_cycle") or cfg.get("final16")
                else checked["result"].get("ai_prerequisites_passed")
            )
        ):
            raise Blocked("Pilot45 actual AI prerequisites missing")
    write(
        task / "operator/native-environment.json",
        native_flight.environment_check(),
    )
    ledger = read(task / "ledger.json")
    if cfg.get("parallel16") and (
        set(ledger["cases"]) != {c["case_id"] for c in manifest["cases"]}
        or any(
            ledger["cases"][c["case_id"]].get("case") != c
            for c in manifest["cases"]
        )
    ):
        raise Blocked(
            "Full parallel planned case ledger must be initialized before submission"
        )
    unstarted = [
        c
        for c in manifest["cases"]
        if not ledger["cases"].get(c["case_id"], {}).get("attempts")
    ]
    if unstarted:
        prior = protection()
        if cfg.get("parallel16"):
            # A resumed pair may already have its other member. Exclude only
            # native IDs durably owned by this same logical pilot ledger.
            owned = {a["run_id"] for a in ledger["attempts"] if a.get("run_id")}
            prior["records"] = [
                r for r in prior["records"] if r.get("run_id") not in owned
            ]
        fresh_cases(unstarted, prior)
    remaining = cfg["planned_cases"] - sum(
        e.get("status") == "COMPLETE" for e in ledger["cases"].values()
    )
    snapshot = resources(
        remaining,
        read(task / "budget.json")["per_case_conservative_increment_bytes"],
    )
    with ExitStack() as stack:
        for p in (
            HOME / "pretrain.lock",
            ROOT / "Data/operations/collection.lock",
            ROOT / "simulation/.cache/setup.lock",
            *(
                ROOT / f"simulation/.cache/instances/{i}.lock"
                for i in range(cfg.get("flight_workers", 1))
            ),
        ):
            handle = stack.enter_context(p.open("rb"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for i in range(cfg.get("flight_workers", 1)):
            collector().check_instance_socket(i)
        if cfg.get("parallel16"):
            from .pilot_parallel import check_ports

            check_ports()
    write(
        task / "operator/preflight.json",
        dict(
            status="PASSED",
            observed_utc=now(),
            resources=snapshot,
            flight_workers=cfg.get("flight_workers", 1),
            cpu_workers_total_max=cfg.get("cpu_workers_total_max", 1),
            no_fit=True,
        ),
    )
    return snapshot


def start(task, resume=False):
    from .cli import start as submit

    task = task_path(task)
    cfg = read(task / "config.json")
    if (
        not cfg.get("flow")
        and not cfg.get("single_integration")
        and not cfg.get("ai_ready")
    ):
        raise Blocked(
            "Pilot45 execution remains blocked; coordinator integration19 is separate"
        )
    if not cfg.get("sealed"):
        raise Blocked("Final dependency seal missing; use pilot45 seal")
    verify(cfg["source_sha256"])
    ledger = read(task / "ledger.json")
    remaining = cfg["planned_cases"] - sum(
        e.get("status") == "COMPLETE" for e in ledger["cases"].values()
    )
    resources(
        remaining,
        read(task / "budget.json")["per_case_conservative_increment_bytes"],
    )
    if resume:
        unit, pid, props = service_identity(task)
        if pid or props.get("ActiveState") in {"activating", "deactivating"}:
            raise Blocked("Do not resume an active service")
        if props.get("ActiveState") == "active":
            # RemainAfterExit has no PID; release only this identity-checked own unit.
            subprocess.run(["systemctl", "--user", "stop", unit], check=True)
        state = read(task / "operator/state.json")
        if state["status"] in {
            "PILOT45_QUALITY_PASSED_STOPPED",
            "INTEGRATION19_QUALITY_PASSED_STOPPED",
        }:
            raise Blocked(
                "Pilot already completed; never regenerate successful cases"
            )
        ledger = read(task / "ledger.json")
        reconcile_completed(task, ledger)
        if any(a.get("state") == "RUNNING" for a in ledger["attempts"]):
            raise Blocked(
                "Interrupted physical attempt requires explicit reconciliation, not blind retry"
            )
        request = task / "operator/resume-request.json"
        if request.exists():
            raise Blocked("Unconsumed resume request preserved")
        write(
            request,
            dict(
                config_sha256=sha(task / "config.json"),
                prior_state=state,
                owner_uid=os.getuid(),
                requested_utc=now(),
            ),
        )
        subprocess.run(
            ["systemctl", "--user", "reset-failed", unit],
            check=False,
            capture_output=True,
        )
    submit(task, pilot_resume=resume)


def reconcile_completed(task, ledger):
    """One-shot reconciliation; never equate an unknown/stopped flight with success."""
    for cid, entry in ledger["cases"].items():
        for attempt in entry["attempts"]:
            if attempt["state"] != "RUNNING":
                continue
            receipt = (
                task
                / "pilot/execution"
                / cid
                / f"flight-{attempt['number']:02}.receipt.json"
            )
            if not receipt.exists():
                continue
            result = read(receipt)
            if result.get("status") != "completed":
                continue
            raw = ai19.no_links(result["run"])
            meta = read(raw / "run.json")
            scenario = read(raw / "evaluation/scenario.json")
            if (
                raw.parent != ROOT / "simulation/runs"
                or meta.get("status") != "completed"
                or meta.get("parent_attempt_id") != attempt["parent_attempt_id"]
                or scenario["name"] != entry["case"]["scenario"]
                or scenario["seed"] != entry["case"]["seed"]
                or scenario["operation_variation"]
                != entry["case"]["operation_variation"]
            ):
                raise Blocked("Interrupted completed-raw identity mismatch")
            if any(
                identity(pid) is not None
                for pid in meta.get("owned_process_groups", {}).values()
            ):
                raise Blocked("Raw-owned native process still alive")
            entry["run"] = str(raw)
            attempt.update(
                state="COMPLETED_RECONCILED_UNAUDITED",
                run_id=raw.name,
                reconciled_utc=now(),
            )
            for previous in ledger["attempts"]:
                if (
                    previous["case_id"] == cid
                    and previous["number"] == attempt["number"]
                ):
                    previous.update(attempt)
    write(task / "ledger.json", ledger)


def stop(task, immediate=False):
    task = task_path(task)
    unit, pid, props = service_identity(task, require_running=True)
    write(
        task / "operator/stop-request.json",
        dict(
            unit=unit,
            pid=pid,
            pid_start_ticks=identity(pid),
            owner_uid=os.getuid(),
            config_sha256=sha(task / "config.json"),
            requested_utc=now(),
        ),
    )
    if read(task / "config.json").get("flow"):
        active = read(task / "operator/state.json").get("active_task")
        if active:
            stop(task_path(active), immediate=immediate)
    if immediate:
        subprocess.run(["systemctl", "--user", "stop", unit], check=True)
    return dict(
        status="OWNED_SERVICE_STOP_REQUESTED"
        if immediate
        else "DRAIN_CURRENT_CASE_REQUESTED",
        unit=unit,
    )


def resumed_registry_identity(task, config, old, record):
    """Only frozen SHORT members may share pair/group identity, never raw or retry IDs."""
    resumed = config["short_resume"]
    if (
        not config.get("short_cycle")
        or old.get("cohort") != SHORT_COHORT
        or record.get("cohort") != SHORT_COHORT
        or old["task"] not in resumed["previous_task_names"]
        or config.get("lineage_namespace") != resumed["namespace"]
    ):
        return False
    verify(resumed["files"])
    current = read(task / "manifest.json")["cases"]
    original = read(Path(resumed["previous_task"]) / "manifest.json")["cases"]
    by_id = {c["case_id"]: c for c in current}
    canonical = {c["case_id"]: c for c in original}
    a, b = by_id.get(old.get("case_id")), by_id.get(record.get("case_id"))
    if (
        not a
        or not b
        or a != canonical.get(a["case_id"])
        or b != canonical.get(b["case_id"])
    ):
        return False
    for item, case in ((old, a), (record, b)):
        expected = dict(
            scenario=case["scenario"],
            seed=case["seed"],
            paired_case_id=case["paired_case_id"],
            group_id=resumed["namespace"] + ":" + case["group_id"],
            retry_family_id=resumed["namespace"]
            + ":"
            + case["retry_family_id"],
            role=case["role"],
        )
        if any(
            key in item and item[key] != value
            for key, value in expected.items()
        ):
            return False
    if a["case_id"] == b["case_id"]:
        return True
    if any(
        old.get(key) and old[key] == record.get(key)
        for key in ("run_id", "raw_sha256", "retry_family_id")
    ):
        return False
    members = [c for c in current if c["group_id"] == b["group_id"]]
    if len(members) != 2 or {c["case_id"] for c in members} != {
        a["case_id"],
        b["case_id"],
    }:
        return False
    if {a["category"], b["category"]} != {"paired_normal", "anomaly"} or {
        a["role"],
        b["role"],
    } != {
        "normal_test",
        "anomaly_test",
    }:
        return False
    return all(
        a[key] == b[key]
        for key in (
            "paired_case_id",
            "group_id",
            "seed",
            "operation_variation",
            "kind",
            "environment",
            "duration",
        )
    )


def register(task, record):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        for line in stream:
            old = json.loads(line)
            collision = any(
                record.get(k) and record.get(k) == old.get(k)
                for k in (
                    "run_id",
                    "raw_sha256",
                    "paired_case_id",
                    "group_id",
                    "retry_family_id",
                )
            )
            collision |= (
                record.get("scenario") is not None
                and record.get("seed") is not None
                and (record["scenario"], record["seed"])
                == (old.get("scenario"), old.get("seed"))
            )
            if old["task"] != task.name and collision:
                cfg = read(task / "config.json")
                recovery = cfg.get("recovery", {})
                allowed = old["task"] == Path(
                    recovery.get("previous_task", "")
                ).name and old.get("case_id") == record.get(
                    "case_id"
                ) == recovery.get("case_id")
                continuation = cfg.get("continuation", {})
                repair = cfg.get("short_repair", {})
                if repair and old["task"] == Path(repair["previous_task"]).name:
                    verify(repair["files"])
                    allowed = (
                        old.get("case_id")
                        == record.get("case_id")
                        in repair["failures"]
                    )
                resumed = cfg.get("short_resume", {})
                if resumed and old["task"] in resumed["previous_task_names"]:
                    allowed = resumed_registry_identity(task, cfg, old, record)
                if (
                    cfg.get("parallel16")
                    and old["task"]
                    == Path(continuation.get("previous_task", "")).name
                ):
                    verify(continuation["files"])
                    rows = read(
                        Path(continuation["previous_task"]) / "manifest.json"
                    )["cases"]
                    allowed = any(
                        c["case_id"]
                        == record.get("case_id")
                        == old.get("case_id")
                        for c in rows
                    )
                if cfg.get("mode") == "main19":
                    from .main_collection import registry_identity

                    allowed = registry_identity(task, cfg, old, record)
                if not allowed:
                    raise Blocked(
                        "Global raw/hash/retry/pair/group cohort reuse"
                    )
                if not cfg.get("parallel16"):
                    verify(recovery["files"])
        stream.seek(0, 2)
        stream.write(
            json.dumps(
                dict(
                    record,
                    task=task.name,
                    cohort=record.get("cohort", COHORT),
                    recorded_utc=now(),
                )
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def retryable(result, attempts, stopped=False):
    """Only the existing, explicit recorder-clock stall is independent infrastructure."""
    reason = result.get("reason", "")
    return (
        not stopped
        and attempts < 3
        and result.get("status") == "failed"
        and isinstance(reason, str)
        and reason.startswith("recording_stalled:")
        and "PX4 clock advanced but raw capture did not" in reason
    )


def physical(task, case, entry, ledger, folder, denied):
    """Preserve every attempt and parent. Successful raw survives preprocessing recovery."""
    while not entry.get("run"):
        cfg = read(task / "config.json")
        namespace = cfg.get("lineage_namespace", task.name)
        recovery = cfg.get("recovery", {})
        previous = entry["attempts"][-1] if entry["attempts"] else None
        corrected = bool(
            previous
            and previous.get("source_task") == recovery.get("previous_task")
            and previous["number"] == recovery.get("previous_attempt")
        )
        if corrected:
            verify(recovery["files"])
        if previous and not previous.get("retry_allowed") and not corrected:
            raise Blocked(
                "Prior physical failure is not approved independent infrastructure"
            )
        number = len(entry["attempts"]) + 1
        if (
            number > case["attempts_max"]
            or (task / "operator/stop-request.json").exists()
        ):
            raise Blocked("Stop/max3 physical attempts; counters preserved")
        parent = previous.get("run_id") if previous else None
        if previous and not parent and not corrected:
            raise Blocked("Missing failed raw parent; no blind retry")
        attempt = dict(
            number=number,
            state="RUNNING",
            parent_attempt_id=parent,
            started_utc=now(),
        )
        if corrected:
            attempt["parent_execution_attempt"] = dict(
                task=recovery["previous_task"],
                number=previous["number"],
                native_run_id=None,
            )
        entry["attempts"].append(attempt)
        ledger["attempts"].append(dict(case_id=case["case_id"], **attempt))
        write(task / "ledger.json", ledger)
        receipt = folder / f"flight-{number:02}.receipt.json"
        request = folder / f"flight-{number:02}.request.json"
        write(
            request,
            dict(
                task=str(task),
                action="flight",
                case=case,
                receipt=str(receipt),
                parent_attempt_id=parent,
            ),
        )
        register(
            task,
            dict(
                event="physical_attempt_started",
                cohort=case["cohort"],
                case_id=case["case_id"],
                attempt=number,
                scenario=case["scenario"],
                seed=case["seed"],
                paired_case_id=case["paired_case_id"],
                retry_family_id=namespace + ":" + case["retry_family_id"],
                group_id=namespace + ":" + case["group_id"],
                parent_attempt_id=parent,
                role=case["role"],
            ),
        )
        failure = None
        try:
            child(task, request, folder / f"flight-{number:02}.log")
        except BaseException as exc:
            failure = exc
        result = read(receipt) if receipt.exists() else {}
        raw = ai19.no_links(result["run"]) if result.get("run") else None
        if raw is not None:
            if raw.parent != ROOT / "simulation/runs" or raw.name in denied:
                raise Blocked("Returned raw is outside newpilot scope")
            meta = read(raw / "run.json")
            scenario = read(raw / "evaluation/scenario.json")
            if (
                meta.get("parent_attempt_id") != parent
                or scenario["name"] != case["scenario"]
                or scenario["seed"] != case["seed"]
                or scenario["operation_variation"]
                != case["operation_variation"]
            ):
                raise Blocked("Attempt parent/seed/operation changed")
            attempt["run_id"] = raw.name
        if result.get("status") == "completed" and raw is not None:
            if meta.get("status") != "completed":
                raise Blocked("Wrapper and actual raw completion disagree")
            entry["run"] = str(raw)
            attempt.update(
                state="COMPLETED"
                if failure is None
                else "COMPLETED_RAW_CHILD_ERROR",
                ended_utc=now(),
                error=str(failure) if failure else None,
            )
        else:
            attempt.update(
                state="FAILED_PRESERVED",
                ended_utc=now(),
                error=str(failure),
                result=result,
                retry_allowed=retryable(
                    result,
                    number,
                    (task / "operator/stop-request.json").exists(),
                )
                and number < case["attempts_max"],
            )
            register(
                task,
                dict(
                    event="physical_attempt_failed",
                    cohort=case["cohort"],
                    case_id=case["case_id"],
                    run_id=raw.name if raw else None,
                    parent_attempt_id=parent,
                    attempt=number,
                    failure=attempt["error"],
                    retry_allowed=attempt["retry_allowed"],
                ),
            )
        ledger["attempts"][-1].update(attempt)
        write(task / "ledger.json", ledger)
        if entry.get("run") and failure is not None:
            raise Blocked(
                "Completed raw retained; recover processing without another flight"
            ) from failure
        if not entry.get("run"):
            if not attempt["retry_allowed"]:
                raise Blocked(
                    "Physical failure preserved; not blindly retried"
                ) from failure
            verify(read(task / "config.json")["source_sha256"])
            resources()
            claim = folder / "socket-claim.json"
            if claim.exists() and os.path.lexists("/tmp/px4-sock-0"):
                from .socket_cleanup import release

                release(read(claim), service_unit(task))
            collector().check_instance_socket(0)


def child(task, request, log, *, on_progress=None, resource_snapshot=None):
    """Only this independent finite program polls its own child; never an LLM callback."""
    job = read(request)
    instance = job.get("instance", 0)
    c = collector()
    progress = None
    if job["action"] == "flight":
        raw_root = ROOT / "simulation/runs"
        progress = c.MissionProgress(
            ROOT / "simulation",
            {p.name for p in raw_root.iterdir()},
            job["case"],
            dict(instance=instance),
            job["case"]["duration"],
            runs_root=raw_root,
        )
        old = Path(request).parent / "socket-claim.json"
        if old.exists():
            old.rename(
                old.with_name(
                    "socket-claim-preserved-" + str(time.time_ns()) + ".json"
                )
            )
    cfg = read(task / "config.json")
    argv = (
        native_flight.command(
            job["case"],
            job.get("parent_attempt_id"),
            cfg.get("paired_cli_options", []),
            instance,
            *([True] if cfg.get("capture_timing") else []),
        )
        if job["action"] == "flight"
        else [
            str(PYTHON),
            "-B",
            "-m",
            "pretrain.main_collection"
            if cfg.get("mode") == "main19"
            else "pretrain.pilot45",
            "_case",
            str(request),
        ]
    )
    with ExitStack() as streams:
        stream = streams.enter_context(Path(log).open("x"))
        stdout_path = (
            Path(request).with_suffix(".stdout.json")
            if job["action"] == "flight"
            else None
        )
        stdout = (
            streams.enter_context(stdout_path.open("x"))
            if stdout_path
            else stream
        )
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=dict(os.environ, PYTHONPATH=pythonpath()),
            stdout=stdout,
            stderr=stream,
        )
        receipt = task / "operator/child.json"
        try:
            group = Path(f"/proc/{process.pid}/cgroup").read_text()
        except FileNotFoundError:
            group = None
        process_record = dict(
            pid=process.pid,
            pid_start_ticks=identity(process.pid),
            owner_uid=os.getuid(),
            cgroup=group,
            request=str(request),
            log=str(log),
            argv=argv,
            stdout_path=str(stdout_path) if stdout_path else None,
            started_utc=now(),
            instance=instance if progress is not None else None,
            requested_real_time_factor=2 if progress is not None else None,
        )
        if not cfg.get("parallel16"):
            write(receipt, process_record)
        write(Path(request).with_suffix(".process.json"), process_record)
        start_time = time.monotonic()
        blocked = None
        terminating = None
        while process.poll() is None:
            try:
                snapshot = (
                    resources()
                    if resource_snapshot is None
                    else resource_snapshot()
                )
                process_record["min_available_ram_bytes"] = min(
                    process_record.get(
                        "min_available_ram_bytes", snapshot["available_bytes"]
                    ),
                    snapshot["available_bytes"],
                )
                process_record["min_free_disk_bytes"] = min(
                    process_record.get(
                        "min_free_disk_bytes", snapshot["disk_free_bytes"]
                    ),
                    snapshot["disk_free_bytes"],
                )
                process_record["max_owned_rss_bytes"] = max(
                    process_record.get("max_owned_rss_bytes", 0),
                    snapshot["owned_tree_rss_bytes"],
                )
                view = progress.sample() if progress is not None else {}
                if progress is not None and len(progress.matches) == 1:
                    from .socket_cleanup import claim

                    raw = (
                        ROOT / "simulation/runs" / next(iter(progress.matches))
                    )
                    meta = read(raw / "run.json")
                    scenario = read(raw / "evaluation/scenario.json")
                    if (
                        meta.get("parent_attempt_id")
                        != job.get("parent_attempt_id")
                        or scenario.get("operation_variation")
                        != job["case"]["operation_variation"]
                    ):
                        raise Blocked(
                            "Progress run parent/operation does not belong to this lane"
                        )
                    claim(
                        raw,
                        instance,
                        Path(request).parent / "socket-claim.json",
                    )
                observation = dict(
                    case_id=job["case"]["case_id"],
                    pid=process.pid,
                    log=str(log),
                    instance=instance if progress is not None else None,
                    action=job["action"],
                    **view,
                )
                if cfg.get("parallel16"):
                    if on_progress is not None:
                        on_progress(observation)
                else:
                    state = read(task / "operator/state.json")
                    state.update(
                        stage=job["action"],
                        active=[observation],
                        resources=snapshot,
                        updated_utc=now(),
                    )
                    write(task / "operator/state.json", state)
                if time.monotonic() - start_time > 14400:
                    raise Blocked("Finite per-case4h cap")
            except Exception as exc:
                blocked = str(exc)
                if terminating is None:
                    process.terminate()
                    terminating = time.monotonic()
            if terminating is not None and time.monotonic() - terminating > 5:
                process.kill()
            time.sleep(1)
        process_record.update(
            exit_code=process.returncode,
            elapsed_s=time.monotonic() - start_time,
            finished_utc=now(),
        )
        write(Path(request).with_suffix(".process.json"), process_record)
        if stdout_path is not None:
            stdout.flush()
            if stdout_path.stat().st_size:
                try:
                    write(
                        job["receipt"], native_flight.read_result(stdout_path)
                    )
                except (ValueError, OSError) as exc:
                    process_record["stdout_error"] = str(exc)
                    write(
                        Path(request).with_suffix(".process.json"),
                        process_record,
                    )
        if progress is not None and not Path(job["receipt"]).exists():
            terminal = progress.terminal_evidence()
            if terminal.get("producer_run_id"):
                raw = ROOT / "simulation/runs" / terminal["producer_run_id"]
                meta = read(raw / "run.json")
                write(
                    job["receipt"],
                    dict(
                        run=str(raw),
                        status=meta["status"],
                        reason=meta.get("termination_reason"),
                        recovered_from_owned_progress=terminal,
                    ),
                )
        if blocked:
            raise Blocked(blocked)
        if process.returncode:
            raise Blocked(
                f"Case child exit{process.returncode}; preserve log and no blind retry: {log}"
            )
    if cfg.get("mode") != "main19":
        verify(read(task / "config.json")["source_sha256"])
    # Production main19 freezes sources at service preflight; each owned phase
    # need not reread every native binary when the child finishes.
    return process_record


def run_case(request):
    value = read(request)
    task = task_path(value["task"])
    cfg, case = read(task / "config.json"), value["case"]
    verify(
        {
            str(task / "config.json"): (task / "config.sha256")
            .read_text()
            .strip()
        }
    )
    verify(cfg["source_sha256"])
    manifest = read(task / "manifest.json")
    validate_manifest(manifest)
    if case not in manifest["cases"] or (
        not cfg.get("single_integration") and not cfg.get("ai_ready")
    ):
        raise Blocked(
            "Only frozen cases with actual final readiness are executable"
        )
    if value["action"] == "prepare":
        result = prepare_one(
            value["selection"],
            value["output"],
            task / "expected-producer.json",
            value["archive_cache"],
            task / "global-protection.json",
            policy_bundle=cfg.get("policy_bundle"),
            **(
                dict(final16=True)
                if cfg.get("final16")
                else dict(short_cycle=True)
                if cfg.get("short_cycle")
                else {}
            ),
        )
        write(value["receipt"], result)
    elif value["action"] == "audit":
        write(
            value["receipt"],
            audit_one(
                value["output"],
                task / "expected-producer.json",
                **(
                    dict(final16=True)
                    if cfg.get("final16")
                    else dict(short_cycle=True)
                    if cfg.get("short_cycle")
                    else {}
                ),
            ),
        )
    elif value["action"] == "cohort_audit" and cfg.get("parallel16"):
        from .pilot_parallel import cohort_audit

        write(value["receipt"], cohort_audit(task))
    elif value["action"] in {"pair", "ai_case", "ai_final"}:
        from . import pilot_bridge

        spec = pilot_bridge.dependencies(cfg["dependency_handoff"])
        if value["action"] == "pair":
            result = dict(
                pair_receipt=str(
                    pilot_bridge.pair(
                        task, spec, value["normal_id"], value["anomaly_id"]
                    )
                )
            )
        elif value["action"] == "ai_final":
            if not cfg.get("short_cycle") and not cfg.get("final16"):
                raise ValueError(
                    "Explicit short45 or FINAL16 final gate required"
                )
            result = pilot_bridge.check(task, spec, phase="final")
        else:
            result = pilot_bridge.check(
                task,
                spec,
                case_id=case["case_id"],
                pair_path=value.get("pair_receipt"),
            )
        write(value["receipt"], result)
    else:
        raise ValueError("No training/scoring action exists")


def worker(task):
    task = task_path(task)
    cfg = read(task / "config.json")
    if cfg.get("parallel16") or cfg.get("parallel_flow"):
        from .pilot_parallel import flow_worker
        from .pilot_parallel import worker as parallel_worker

        return (
            flow_worker(task)
            if cfg.get("parallel_flow")
            else parallel_worker(task)
        )
    if cfg.get("flow"):
        from .pilot_flow import worker as flow_worker

        return flow_worker(task)
    total = cfg["planned_cases"]
    label = "INTEGRATION19" if cfg.get("single_integration") else "PILOT45"
    report_dir = Path(
        cfg.get(
            "report_directory",
            REPORT / "integration19" / task.name
            if cfg.get("single_integration")
            else REPORT,
        )
    )
    verify(
        {
            str(task / "config.json"): (task / "config.sha256")
            .read_text()
            .strip()
        }
    )
    verify(cfg["source_sha256"])
    preflight(task)
    previous_state = read(task / "operator/state.json")
    state = dict(
        status="RUNNING",
        stage="pilot45",
        task=task.name,
        pid=os.getpid(),
        pid_start_ticks=identity(os.getpid()),
        owner_uid=os.getuid(),
        updated_utc=now(),
        started=previous_state.get("started", time.time()),
        service_unit=service_unit(task, cfg),
        cgroup=Path("/proc/self/cgroup").read_text(),
        training_authorized=False,
        completed=[],
        active=[],
    )
    ledger = read(task / "ledger.json")
    with ExitStack() as stack:
        for path in (
            HOME / "pretrain.lock",
            ROOT / "Data/operations/collection.lock",
        ):
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (task / "operator/claimed").exists():
            request = task / "operator/resume-request.json"
            if (
                not request.exists()
                or read(request)["config_sha256"] != sha(task / "config.json")
                or read(request)["owner_uid"] != os.getuid()
            ):
                raise Blocked("Already claimed; explicit owned resume required")
            request.rename(
                task
                / "operator"
                / ("consumed-resume-" + str(time.time_ns()) + ".json")
            )
            stop_request = task / "operator/stop-request.json"
            if stop_request.exists():
                stop_request.rename(
                    stop_request.with_name(
                        "consumed-stop-" + str(time.time_ns()) + ".json"
                    )
                )
        else:
            with (task / "operator/claimed").open("x") as stream:
                stream.write(now())
        write(task / "operator/state.json", state)
        manifest = read(task / "manifest.json")
        validate_manifest(manifest)
        protected = read(task / "protection.json")
        denied = {r["run_id"] for r in protected["records"]}
        try:
            for index, case in enumerate(manifest["cases"]):
                entry = ledger["cases"].setdefault(
                    case["case_id"], dict(case=case, attempts=[])
                )
                if entry.get("status") == "COMPLETE":
                    verify(entry["files"])
                    continue
                if (task / "operator/stop-request.json").exists():
                    raise Blocked(
                        "USER_STOP: current case drained; no new case/retry"
                    )
                verify(cfg["source_sha256"])
                resources(
                    total - index,
                    read(task / "budget.json")[
                        "per_case_conservative_increment_bytes"
                    ],
                )
                folder = task / "pilot/execution" / case["case_id"]
                folder.mkdir(parents=True, exist_ok=True)
                if not entry.get("run"):
                    physical(task, case, entry, ledger, folder, denied)
                raw = Path(entry["run"])
                processing = (
                    len(list(folder.glob("prepare-*.request.json"))) + 1
                )
                if processing > 3:
                    raise Blocked(
                        "Maximum3 preprocessing attempts; successful raw retained"
                    )
                output = (
                    STORE
                    / "processed/pilot45"
                    / task.name
                    / case["case_id"]
                    / f"attempt-{processing:02}"
                )
                selection = dict(
                    case_id=cfg.get("lineage_namespace", task.name)
                    + ":"
                    + case["case_id"],
                    path=str(raw),
                    run_id=raw.name,
                    role=case["role"],
                    cohort=COHORT,
                    group_id=cfg.get("lineage_namespace", task.name)
                    + ":"
                    + case["group_id"],
                    retry_family_id=cfg.get("lineage_namespace", task.name)
                    + ":"
                    + case["retry_family_id"],
                    execution_ancestors=[
                        dict(
                            task=a["source_task"],
                            number=a["number"],
                            ledger_sha256=a["source_ledger_sha256"],
                        )
                        for a in entry["attempts"]
                        if a.get("source_task")
                    ],
                    previous_attempts=[
                        a["run_id"]
                        for a in entry["attempts"][:-1]
                        if a.get("run_id")
                    ],
                    paired_case_id=case["paired_case_id"],
                )
                request = folder / f"prepare-{processing:02}.request.json"
                receipt = folder / f"prepare-{processing:02}.receipt.json"
                write(
                    request,
                    dict(
                        task=str(task),
                        action="prepare",
                        case=case,
                        selection=selection,
                        output=str(output),
                        archive_cache=str(folder / "archive-cache.json"),
                        receipt=str(receipt),
                    ),
                )
                child(task, request, folder / f"prepare-{processing:02}.log")
                prepared = read(receipt)
                audit_request = folder / f"audit-{processing:02}.request.json"
                audit_receipt = folder / f"audit-{processing:02}.receipt.json"
                write(
                    audit_request,
                    dict(
                        task=str(task),
                        action="audit",
                        case=case,
                        output=str(output),
                        receipt=str(audit_receipt),
                    ),
                )
                child(
                    task, audit_request, folder / f"audit-{processing:02}.log"
                )
                audit = read(audit_receipt)
                if audit["export_receipt_sha256"] != sha(
                    output / "receipt.json"
                ):
                    raise Blocked(
                        "Independent quality receipt does not bind this export"
                    )
                register(
                    task,
                    dict(
                        event="completed_raw",
                        cohort=case["cohort"],
                        case_id=case["case_id"],
                        scenario=case["scenario"],
                        seed=case["seed"],
                        run_id=raw.name,
                        raw_sha256=prepared["run"]["raw_sha256"],
                        paired_case_id=selection["paired_case_id"],
                        group_id=selection["group_id"],
                        retry_family_id=selection["retry_family_id"],
                        role=case["role"],
                        parent_attempt_id=entry["attempts"][-1][
                            "parent_attempt_id"
                        ],
                    ),
                )
                entry.update(
                    status="COMPLETE",
                    receipt=str(receipt),
                    output=str(output),
                    files={
                        str(output / p): h for p, h in prepared["files"].items()
                    },
                    audit_receipt=str(audit_receipt),
                    online_available=prepared["online"]["available"],
                    run_available=prepared["run_health"]["status"]
                    == "available",
                )
                entry["files"].update(
                    {
                        str(output / p): sha(output / p)
                        for p in ("receipt.json", "seal.json")
                    }
                )
                entry["files"][str(audit_receipt)] = sha(audit_receipt)
                write(task / "ledger.json", ledger)
                state.update(
                    done=sum(
                        e.get("status") == "COMPLETE"
                        for e in ledger["cases"].values()
                    ),
                    total=total,
                    active=[],
                    updated_utc=now(),
                )
                write(task / "operator/state.json", state)
            reports = [
                read(Path(e["receipt"])) for e in ledger["cases"].values()
            ]
            from learning.admission19 import check_protection, validate_lineages

            validate_lineages([r["run"] for r in reports])
            for entry in ledger["cases"].values():
                verify(entry["files"])
            check_protection(
                [r["run"] for r in reports],
                read(task / "global-protection.json"),
                usage="pilot_input_check",
            )
            passed = len(reports) == total and all(
                r["online"]["available"]
                and r["run_health"]["status"] == "available"
                for r in reports
            )
            if passed and not cfg.get("single_integration"):
                pairs = {}
                for control in (
                    c
                    for c in manifest["cases"]
                    if c["category"] == "paired_normal"
                ):
                    anomaly = next(
                        c
                        for c in manifest["cases"]
                        if c["category"] == "anomaly"
                        and c["group_id"] == control["group_id"]
                    )
                    folder = task / "pilot/execution" / control["case_id"]
                    request, receipt = (
                        folder / "pair.request.json",
                        folder / "pair.receipt.json",
                    )
                    write(
                        request,
                        dict(
                            task=str(task),
                            action="pair",
                            case=control,
                            normal_id=control["case_id"],
                            anomaly_id=anomaly["case_id"],
                            receipt=str(receipt),
                        ),
                    )
                    child(task, request, folder / "pair.log")
                    pairs[control["group_id"]] = read(receipt)["pair_receipt"]
                for case in manifest["cases"]:
                    folder = task / "pilot/execution" / case["case_id"]
                    request, receipt = (
                        folder / "ai-case.request.json",
                        folder / "ai-case.receipt.json",
                    )
                    write(
                        request,
                        dict(
                            task=str(task),
                            action="ai_case",
                            case=case,
                            pair_receipt=pairs.get(case["group_id"]),
                            receipt=str(receipt),
                        ),
                    )
                    child(task, request, folder / "ai-case.log")
                    ledger["cases"][case["case_id"]]["ai_receipt"] = str(
                        receipt
                    )
                    write(task / "ledger.json", ledger)
            state.update(
                status=label + "_QUALITY_PASSED_STOPPED"
                if passed
                else label + "_QUALITY_REVIEW_REQUIRED_STOPPED",
                finished=time.time(),
                no_fit=True,
                next_stage_authorized=False,
            )
            quality = dict(
                status=state["status"],
                planned=total,
                physical_attempts=len(ledger["attempts"]),
                completed_raw=len(reports),
                not_started=total - len(reports),
                online_available=sum(r["online"]["available"] for r in reports),
                run_available=sum(
                    r["run_health"]["status"] == "available" for r in reports
                ),
                valid_target_seconds=sum(
                    r["online"]["target_valid_seconds"] for r in reports
                ),
                requested_mission_seconds=sum(
                    c["duration"] for c in manifest["cases"]
                ),
                cohort=COHORT,
                no_fit=True,
                outputs=[
                    dict(case_id=k, receipt=e["receipt"])
                    for k, e in ledger["cases"].items()
                ],
            )
            write(task / "quality.json", quality)
            write(report_dir / "execution-completion.json", quality)
        except BaseException as exc:
            state.update(
                status="BLOCKED_STOPPED",
                reason=f"{type(exc).__name__}: {exc}",
                finished=time.time(),
                next_stage_authorized=False,
            )
            raise
        finally:
            ledger["status"] = state["status"]
            write(task / "ledger.json", ledger)
            write(task / "operator/state.json", state)
            quality = quality_snapshot(task, ledger, state)
            write(task / "quality.json", quality)
            write(report_dir / "execution-completion.json", quality)
            write(report_dir / "completion.json", quality)


def quality_snapshot(task, ledger, state):
    from collections import Counter

    manifest = read(task / "manifest.json")
    total = len(manifest["cases"])
    reports = [
        read(Path(e["receipt"]))
        for e in ledger["cases"].values()
        if e.get("status") == "COMPLETE"
    ]
    processes = [
        read(p) for p in (task / "pilot/execution").glob("*/*.process.json")
    ]
    receipts = [
        read(p) for p in (task / "pilot/execution").glob("*/archive-cache.json")
    ]
    raw = [Path(e["run"]) for e in ledger["cases"].values() if e.get("run")]
    result = dict(
        schema="drone-data.pilot45.completion/1",
        task_id=read(task / "config.json").get("task_id", TASK_ID),
        task=str(task),
        status=state["status"],
        reason=state.get("reason"),
        updated_utc=now(),
        planned_cases=total,
        physical_attempts=len(ledger["attempts"]),
        failed_physical_attempts=sum(
            a["state"] == "FAILED_PRESERVED" for a in ledger["attempts"]
        ),
        unknown_physical_attempts=sum(
            a["state"] == "RUNNING" for a in ledger["attempts"]
        ),
        completed_raw=len(raw),
        prepared_and_audited=len(reports),
        ai_case_checks_passed=sum(
            bool(e.get("ai_receipt")) for e in ledger["cases"].values()
        ),
        not_started=total
        - sum(bool(e["attempts"]) for e in ledger["cases"].values()),
        planned_role_counts=dict(Counter(c["role"] for c in manifest["cases"])),
        cohort=SHORT_COHORT
        if read(task / "config.json").get("short_cycle")
        else COHORT,
        online_available=sum(r["online"]["available"] for r in reports),
        run_available=sum(
            r["run_health"]["status"] == "available" for r in reports
        ),
        valid_online_windows=sum(r["online"]["valid_windows"] for r in reports),
        valid_target_seconds=sum(
            r["online"]["target_valid_seconds"] for r in reports
        ),
        run_post_initial100_seconds=sum(
            r["run_health"]["post_initial100_seconds"] for r in reports
        ),
        requested_mission_seconds=sum(c["duration"] for c in manifest["cases"]),
        raw_bytes=sum(
            p.stat().st_size for r in raw for p in r.rglob("*") if p.is_file()
        ),
        failed_raw_bytes=sum(
            p.stat().st_size
            for a in ledger["attempts"]
            if a.get("run_id") and a["state"] == "FAILED_PRESERVED"
            for p in (ROOT / "simulation/runs" / a["run_id"]).rglob("*")
            if p.is_file()
        ),
        verified_archive_bytes=sum(r["archive"]["bytes"] for r in receipts),
        consumer_bytes=sum(r["consumer_bytes"] for r in reports),
        final_export_bytes=sum(
            p.stat().st_size
            for e in ledger["cases"].values()
            if e.get("status") == "COMPLETE"
            for p in Path(e["output"]).rglob("*")
            if p.is_file()
        ),
        child_wall_seconds=sum(
            p["elapsed_s"] for p in processes if p.get("elapsed_s") is not None
        ),
        child_wall_seconds_by_stage={
            stage: sum(
                p.get("elapsed_s", 0)
                for p in processes
                if Path(p["request"]).name.startswith(stage + "-")
            )
            for stage in ("flight", "prepare", "audit")
        },
        min_available_ram_bytes=min(
            (
                p["min_available_ram_bytes"]
                for p in processes
                if "min_available_ram_bytes" in p
            ),
            default=None,
        ),
        max_owned_rss_bytes=max(
            (
                p["max_owned_rss_bytes"]
                for p in processes
                if "max_owned_rss_bytes" in p
            ),
            default=None,
        ),
        source=read(task / "config.json").get("producer"),
        no_fit=True,
        training_started=False,
        model_inference_started=False,
        next_stage_authorized=False,
        cases={
            cid: dict(
                status=e.get("status", "INCOMPLETE"),
                run=e.get("run"),
                attempts=e["attempts"],
                receipt=e.get("receipt"),
                audit_receipt=e.get("audit_receipt"),
            )
            for cid, e in ledger["cases"].items()
        },
    )
    return result


def main(argv=None, *, single_scope=False):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("init")
    p.add_argument("--name", required=True)
    p.add_argument("--seed-base", type=int)
    if single_scope:
        p.add_argument("--recover-from", type=Path)
        p.add_argument("--launch-receipt", type=Path)
    if not single_scope:
        p = sub.add_parser("flow-init")
        p.add_argument("--name", required=True)
        p.add_argument("--previous-integration", type=Path, required=True)
        p.add_argument("--draft", type=Path, required=True)
        p.add_argument("--handoff", type=Path)
        p.add_argument("--reuse-integration", type=Path)
    for action in (
        ("start", "status", "stop", "resume")
        if single_scope
        else ("seal", "start", "status", "stop", "resume")
    ):
        p = sub.add_parser(action)
        p.add_argument("task", type=Path)
        if action == "seal":
            p.add_argument("--sim-receipt", type=Path, required=True)
            p.add_argument("--ai-receipt", type=Path, required=True)
        if action == "stop":
            p.add_argument("--immediate", action="store_true")
    p = sub.add_parser("_case")
    p.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "flow-init":
            from .pilot_flow import DEFAULT_HANDOFF
            from .pilot_flow import init as flow_init

            result = dict(
                status="FINITE_FLOW_READY_NOT_STARTED",
                task=str(
                    flow_init(
                        args.name,
                        args.previous_integration,
                        args.draft,
                        args.handoff or DEFAULT_HANDOFF,
                        reuse_integration=args.reuse_integration,
                    )
                ),
            )
        elif args.action == "init":
            result = dict(
                status="INTEGRATION19_READY_NOT_STARTED"
                if single_scope
                else "DRAFT45_FROZEN_DEPENDENCIES_PENDING",
                task=str(
                    init(
                        args.name,
                        args.seed_base,
                        single=single_scope,
                        recovery_from=getattr(args, "recover_from", None),
                        launch_receipt=getattr(args, "launch_receipt", None),
                    )
                ),
            )
        elif args.action == "seal":
            result = seal(args.task, args.sim_receipt, args.ai_receipt)
        elif args.action == "status":
            task = task_path(args.task)
            result = dict(
                state=read(task / "operator/state.json"),
                ledger=read(task / "ledger.json"),
            )
            state = result["state"]
            if state.get("status") in {"RUNNING", "WAITING_DEPENDENCIES"} and (
                not state.get("pid")
                or identity(state.get("pid")) is None
                or identity(state.get("pid")) != state.get("pid_start_ticks")
            ):
                result["live_status"] = (
                    "UNKNOWN_NOT_RUNNING_REQUIRES_RECONCILIATION"
                )
            result["status_is_one_shot"] = True
        elif args.action == "stop":
            result = stop(args.task, args.immediate)
        elif args.action in {"start", "resume"}:
            start(args.task, resume=args.action == "resume")
            result = dict(status="SUBMITTED_NOT_COMPLETED", task=str(args.task))
        else:
            run_case(args.request)
            result = dict(status="CASE_FINISHED")
        print(json.dumps(result))
        return 0
    except (Blocked, ValueError, OSError) as exc:
        print(
            json.dumps(dict(status="BLOCKED", reason=str(exc))), file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
