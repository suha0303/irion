"""One independent16-flight input check; reuse the finite runner without45 exemptions."""

import copy
import os
import re
from pathlib import Path

from x500_storage import capture

from . import ai19, pilot45, pilot_bridge
from .common import (
    HOME,
    ROOT,
    STORE,
    now,
    read,
    runtime_identity,
    sha,
    verify,
    write,
)

TASK = "FINAL16_20261107"
COHORT = "final16_development_reference"
USAGE = "final16_input_check"
AUTH = ROOT / "ai/reports/final16-20261107/attempt-01/authorization.json"
AUTH_SHA = "05017f2adc41caf14906ecab4ce82e02d9313b097da8be4609e18a3b9ca3041b"
EXPECTED = AUTH.with_name("expected-cases.json")
REPORT = ROOT / "docs/data/reports/final16-20261107/attempt-01"


def authorization():
    verify({str(AUTH): AUTH_SHA})
    value = read(AUTH)
    if value.get("task_id") != TASK:
        raise pilot45.Blocked("Explicit FINAL16 authorization required")
    return value


def cases():
    auth = authorization()
    source = read(EXPECTED)
    generated = pilot45.build_cases(
        auth["scope"]["seed_base"], duration=180, case_prefix="final16"
    )
    rows = [
        c
        for c in generated
        if c["group_id"] in {f"final16-pair-{i:02}" for i in range(1, 9)}
    ]
    for c in rows:
        c["attempts_max"] = 1
    if (
        source.get("task_id") != TASK
        or source.get("authorization_sha256") != AUTH_SHA
        or source.get("seed_base") != auth["scope"]["seed_base"]
        or rows != source.get("cases")
    ):
        raise pilot45.Blocked(
            "FINAL16 must match the independently frozen original8 pairs"
        )
    for c in rows:
        c["cohort"] = COHORT
    return rows


def validate_manifest(manifest):
    if (
        manifest.get("schema") != "drone-data.final16.manifest/1"
        or manifest.get("task_id") != TASK
        or manifest.get("final16") is not True
        or manifest.get("short_cycle")
        or manifest.get("single_integration") is not False
        or manifest.get("mission_seconds") != 180
        or manifest.get("seed_base") != authorization()["scope"]["seed_base"]
        or manifest.get("authorization")
        != dict(path=str(AUTH), sha256=AUTH_SHA)
        or manifest.get("cases") != cases()
        or any(
            manifest.get(k)
            for k in ("recovery_of", "short_resume", "reused_cases")
        )
    ):
        raise pilot45.Blocked(
            "Exact new FINAL16/max1/8pair/no-reuse manifest required"
        )
    return manifest["cases"]


def validate_config(cfg):
    authorization()
    expected = dict(
        task_id=TASK,
        scope_id=TASK,
        planned_cases=16,
        mission_seconds=180,
        final16=True,
        flight_workers=16,
        requested_real_time_factor=2,
        cpu_workers_total_max=4,
        unit_attempts_max=1,
        stop_on_first_physical_failure=True,
        nonflight_when_flights_idle=True,
        training_authorized=False,
        fit_authorized=False,
    )
    if any(
        type(cfg.get(k)) is not type(v) or cfg[k] != v
        for k, v in expected.items()
    ) or any(
        cfg.get(k)
        for k in (
            "short_cycle",
            "short_resume",
            "short_repair",
            "continuation",
            "recovery",
        )
    ):
        raise pilot45.Blocked(
            "Exact FINAL16/16flight/RTF2/CPU4-after-flight/max1 contract required"
        )
    if cfg.get("authorization") != dict(
        path=str(AUTH), sha256=AUTH_SHA
    ) or not re.fullmatch(
        r"final16[A-Za-z0-9_-]*", cfg.get("lineage_namespace", "")
    ):
        raise pilot45.Blocked("New authorized FINAL16 namespace required")


def validate_ledger(task, *, complete=False, pristine=False):
    cfg, manifest, ledger = (
        read(task / n) for n in ("config.json", "manifest.json", "ledger.json")
    )
    validate_config(cfg)
    rows = validate_manifest(manifest)
    if (
        ledger.get("schema") != "drone-data.final16.ledger/1"
        or ledger.get("task_id") != TASK
        or set(ledger["cases"]) != {c["case_id"] for c in rows}
        or ledger.get("administrative_attempts")
        or len(ledger["attempts"]) > 16
    ):
        raise pilot45.Blocked(
            "FINAL16 complete denominator without inherited/admin attempts required"
        )
    attempts = []
    for c in rows:
        entry = ledger["cases"][c["case_id"]]
        if entry.get("case") != c or any(
            entry.get(k)
            for k in ("inherited_from", "external_audit", "reuse_audit")
        ):
            raise pilot45.Blocked("FINAL16 cannot substitute inherited data")
        if len(entry["attempts"]) > 1 or any(
            a.get("number") != 1
            or a.get("parent_attempt_id") is not None
            or a.get("source_task")
            or a.get("quality_rejection")
            for a in entry["attempts"]
        ):
            raise pilot45.Blocked(
                "FINAL16 admits one new physical attempt per case"
            )
        attempts.extend(
            dict(case_id=c["case_id"], **a) for a in entry["attempts"]
        )
        if complete and (
            entry.get("status") != "COMPLETE"
            or not entry.get("run")
            or len(entry["attempts"]) != 1
            or entry["attempts"][0].get("state") != "COMPLETED"
            or entry["attempts"][0].get("run_id") != Path(entry["run"]).name
        ):
            raise pilot45.Blocked("All16 new actual completed cases required")
    if attempts != ledger["attempts"]:
        # Completion order may differ, identities and content must not.
        def by_id(xs):
            return {a["case_id"]: a for a in xs}

        if len(attempts) != len(ledger["attempts"]) or by_id(attempts) != by_id(
            ledger["attempts"]
        ):
            raise pilot45.Blocked(
                "FINAL16 flat/individual physical ledgers disagree"
            )
    if pristine and (
        attempts
        or any(
            e.get("run") or e.get("status") for e in ledger["cases"].values()
        )
    ):
        raise pilot45.Blocked(
            "FINAL16 submission must be unclaimed and entirely new"
        )
    ids = [a["run_id"] for a in attempts if a.get("run_id")]
    if len(ids) != len(set(ids)):
        raise pilot45.Blocked("Duplicate FINAL16 native UUID")
    return ledger


def dependencies(value):
    """Only the new owner's exact16 static handoff can enable this scope."""
    authorization()
    if (
        value.get("schema") != "drone-ai.final16.data-handoff/1"
        or value.get("task_id") != TASK
        or value.get("status") != "READY"
        or value.get("code_frozen") is not True
    ):
        raise pilot45.Blocked(
            "Final AI FINAL16 READY source-frozen handoff required"
        )
    for key in (
        "authorization",
        "expected_cases",
        "original_source_definitions",
        "reviewed_Data_contract",
    ):
        item = value[key]
        verify({str(ai19.no_links(item["path"])): item["sha256"]})
    if value["authorization"] != dict(path=str(AUTH), sha256=AUTH_SHA):
        raise pilot45.Blocked(
            "Owner handoff must bind this FINAL16 authorization"
        )
    freeze = value["source_freeze"]
    verify({freeze["path"]: freeze["sha256"]})
    frozen = read(freeze["path"])
    ai19.ai_modules()
    from learning.history19.final16 import CRITICAL

    if (
        frozen.get("task_id") != TASK
        or frozen.get("code_frozen") is not True
        or set(frozen["files"]) != set(CRITICAL)
    ):
        raise pilot45.Blocked(
            "Complete final AI FINAL16 source freeze required"
        )
    for name, digest in frozen["files"].items():
        if not name.startswith("ai/") or ".." in Path(name).parts:
            raise pilot45.Blocked("AI freeze must bind actual owner sources")
        verify({str(ai19.no_links(ROOT / name)): digest})
    for name, path in value["paths"].items():
        verify({str(ai19.no_links(path)): value["sha256"][name]})
    if set(value["paths"]) != {
        "native_policy",
        "paired_policy",
        "expected_producer",
        "source_manifest",
        "runtime_manifest",
    }:
        raise pilot45.Blocked(
            "FINAL16 does not import prior45/native6 READY receipts"
        )
    ai19.ai_modules()
    from learning.admission19 import check_producer

    binding = ai19.producer_binding()
    check_producer(value["producer"], binding)
    check_producer(read(value["paths"]["expected_producer"]), binding)
    if (
        ai19.no_links(value["paths"]["source_manifest"])
        != ROOT / "simulation/config/producer-sources.json"
        or ai19.no_links(value["paths"]["runtime_manifest"])
        != ROOT / "simulation/artifacts/runtime-manifest.json"
        or value["sha256"]["source_manifest"]
        != binding["source_manifest_sha256"]
        or value["sha256"]["runtime_manifest"] != binding["runtime_sha256"]
    ):
        raise pilot45.Blocked(
            "FINAL16 source/runtime aliases must be the actual authorized producer"
        )
    if (binding["source_manifest_sha256"], binding["runtime_sha256"]) != (
        authorization()["producer"]["source_sha256"],
        authorization()["producer"]["runtime_sha256"],
    ):
        raise pilot45.Blocked(
            "FINAL16 producer differs from coordinator authority"
        )
    policy, native = (
        read(value["paths"][n]) for n in ("paired_policy", "native_policy")
    )
    check_producer(policy["producer"], binding)
    if (
        policy.get("scope_task") != TASK
        or policy.get("approved_for_final16") is not True
        or policy.get("purpose") != COHORT
        or policy.get("authorization_sha256") != AUTH_SHA
        or policy.get("fixture_only") is not False
        or policy.get("native_policy_sha256")
        != sha(value["paths"]["native_policy"])
        or policy.get("native_adapter") != "joint-native-consumption-c2-v2"
        or native.get("revision") != "joint-native-consumption-c2-v2"
        or value.get("origin_relative_path") != policy.get("mission_source")
        or value.get("paired_cli_options")
        != [
            "--phase-policy",
            native["revision"],
            "--paired-case-id",
            "{paired_case_id}",
        ]
        or value.get("pair_verifier_argv")
        != [
            str(ROOT / "simulation/capstone-sim"),
            "pair-check",
            "--normal-run",
            "{normal_run}",
            "--anomaly-run",
            "{anomaly_run}",
            "--output",
            "{output}",
        ]
    ):
        raise pilot45.Blocked("FINAL16 native policy/CLI binding changed")
    return value


def quality(task, ledger, state):
    q = pilot45.quality_snapshot(task, ledger, state)
    final = (
        read(ledger["ai_final_receipt"])["result"]
        if ledger.get("ai_final_receipt")
        else {}
    )
    q.update(
        schema="drone-data.final16.completion/1",
        cohort=COHORT,
        mission_seconds=180,
        scope_id=TASK,
        final16_ai_final=final,
        LR_main_connected=False,
        inherited_successes=sum(
            bool(e.get("inherited_from")) for e in ledger["cases"].values()
        ),
    )
    if state["status"] == "FINAL16_QUALITY_PASSED_STOPPED":
        validate_ledger(task, complete=True)
        if not (
            final.get("status") == "FINAL16_INPUT_AND_PAIR_CHECKS_PASSED"
            and final.get("actual_data_verified") is True
            and final.get("cases_checked") == 16
            and final.get("pairs_checked") == 8
        ):
            raise pilot45.Blocked("Actual AI final16/8pair pass missing")
    return q


def create(name, handoff):
    """Freeze a new task and parent flow; never submit a service here."""
    auth = authorization()
    if not re.fullmatch(r"final16[A-Za-z0-9_-]{1,42}", name):
        raise ValueError("New final16 namespace required")
    task, flow = (
        pilot45.task_path(HOME / "tasks" / n) for n in (name + "-16", name)
    )
    if task.exists() or flow.exists():
        raise ValueError("Existing FINAL16 evidence must be preserved")
    spec = pilot_bridge.dependencies(handoff)
    rows, protected = cases(), pilot45.protection()
    pilot45.fresh_cases(rows, protected)
    binding = ai19.producer_binding()
    cfg = dict(
        mode="pilot45",
        collection_profile="final16_common19",
        final16=True,
        single_integration=False,
        planned_cases=16,
        task_id=TASK,
        scope_id=TASK,
        owner_uid=os.getuid(),
        service_unit="drone-pretrain-" + task.name + ".service",
        python_runtime=runtime_identity(),
        external_storage=str(STORE),
        storage_policy=capture(STORE),
        seed_base=auth["scope"]["seed_base"],
        lineage_namespace=name,
        flight_workers=16,
        requested_real_time_factor=2,
        cpu_workers_total_max=4,
        export_workers=4,
        io_workers=4,
        workers=4,
        audit_workers=4,
        validated_flight_workers=None,
        min_available_ram_gb=1,
        use_available_ram=True,
        use_swap=True,
        unit_attempts_max=1,
        wall_seconds=24 * 3600,
        training_authorized=False,
        fit_authorized=False,
        sealed=False,
        source_sha256={},
        authorization=dict(path=str(AUTH), sha256=AUTH_SHA),
        mission_seconds=180,
        nonflight_when_flights_idle=True,
        stop_on_first_physical_failure=True,
        capture_timing=True,
        parallel16=True,
        producer=binding,
        dependency_handoff=str(task / "dependency-handoff.json"),
        policy_bundle=pilot_bridge.policy_bundle(spec),
        paired_cli_options=spec["paired_cli_options"],
        report_directory=str(REPORT / name),
    )
    manifest = dict(
        schema="drone-data.final16.manifest/1",
        task_id=TASK,
        final16=True,
        short_cycle=False,
        single_integration=False,
        mission_seconds=180,
        seed_base=cfg["seed_base"],
        seed_origin="coordinator_OS_entropy_frozen",
        authorization=cfg["authorization"],
        cases=rows,
    )
    validate_manifest(manifest)
    validate_config(cfg)
    budget = pilot45.estimate()
    budget.update(
        planned_cases=16,
        total_increment_bytes=16
        * budget["per_case_conservative_increment_bytes"],
    )
    pilot45.resources(16, budget["per_case_conservative_increment_bytes"])
    task.mkdir()
    for member, value in {
        "manifest.json": manifest,
        "config.json": cfg,
        "protection.json": protected,
        "global-protection.json": pilot45.global_protection(
            protected, rows, name, cohort=COHORT
        ),
        "budget.json": budget,
        "dependency-handoff.json": spec,
        "expected-producer.json": {
            k: v
            for k, v in binding.items()
            if k not in {"source_hashes", "vehicle_profile_revision"}
        },
        "ledger.json": dict(
            schema="drone-data.final16.ledger/1",
            task_id=TASK,
            status="DRAFT_NOT_STARTED",
            cases={c["case_id"]: dict(case=c, attempts=[]) for c in rows},
            attempts=[],
        ),
    }.items():
        write(task / member, value)
    (task / "manifest.sha256").write_text(
        sha(task / "manifest.json") + "\n", encoding="utf-8"
    )
    write(task / "ai-manifest.json", pilot_bridge.plan(task))
    write(task / "consumer-capsule.json", pilot_bridge.capsule(task, spec))
    checked = pilot_bridge.check(task, spec)
    sources = pilot45.frozen_sources(
        task,
        [
            AUTH,
            EXPECTED,
            task / "dependency-handoff.json",
            task / "ai-manifest.json",
            task / "consumer-capsule.json",
            Path(checked["receipt"]),
            EXPECTED.with_name("expected-manifest-cases.json"),
            Path(handoff),
            Path(spec["source_freeze"]["path"]),
            REPORT / "CONTRACT.json",
            *map(Path, spec["paths"].values()),
        ],
    )
    sources.update(
        {
            str(ROOT / p): h
            for p, h in read(task / "consumer-capsule.json")["files"].items()
        }
    )
    cfg.update(sealed=True, ai_ready=checked, source_sha256=sources)
    write(task / "config.json", cfg)
    (task / "config.sha256").write_text(
        sha(task / "config.json") + "\n", encoding="utf-8"
    )
    validate_ledger(task, pristine=True)
    write(
        task / "operator/state.json",
        dict(
            status="READY_NOT_STARTED",
            task=task.name,
            scope_id=TASK,
            updated_utc=now(),
        ),
    )
    fc = copy.deepcopy(cfg)
    fc.update(
        flow=True,
        parallel_flow=True,
        parallel16=False,
        pilot_task=str(task),
        service_unit="drone-pretrain-" + name + ".service",
    )
    fc["source_sha256"][str(task / "config.json")] = sha(task / "config.json")
    flow.mkdir()
    write(flow / "config.json", fc)
    (flow / "config.sha256").write_text(
        sha(flow / "config.json") + "\n", encoding="utf-8"
    )
    write(
        flow / "ledger.json",
        dict(cases={}, attempts=[], phases=dict(final16=dict(task=str(task)))),
    )
    write(
        flow / "operator/state.json",
        dict(
            status="READY_NOT_STARTED",
            task=name,
            scope_id=TASK,
            updated_utc=now(),
        ),
    )
    return flow, task
