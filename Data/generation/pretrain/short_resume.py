"""Prepare the reviewed C03 continuation; use the existing finite runner, never submit."""

import ast
import copy
import os
from pathlib import Path

from . import ai19, pilot19, pilot45
from .common import HOME, ROOT, now, read, sha, verify, write


def physical_projection(previous, value, original):
    """Keep the immutable52-row source; exclude only one strictly proven non-submission."""
    result = copy.deepcopy(original)
    reference = value.get("administrative_not_submitted")
    unknown = [a for a in original["attempts"] if not a.get("run_id")]
    if not reference:
        if unknown:
            raise pilot45.Blocked("Unknown UUID-less attempt cannot be discarded")
        return result
    if sha(ai19.no_links(reference["path"])) != reference["sha256"]:
        raise pilot45.Blocked("Administrative non-submission proof changed")
    proof = read(reference["path"])
    verify(proof["files"])
    if (
        proof.get("schema") != "drone-data.not-submitted-administrative-attempt/1"
        or proof.get("status") != "NOT_SUBMITTED_ADMINISTRATIVE_ATTEMPT_PROVEN"
        or proof.get("source_task") != str(previous)
        or proof.get("source_ledger")
        != dict(path=str(previous / "ledger.json"), sha256=sha(previous / "ledger.json"))
        or proof.get("physical_budget_consumed") is not False
        or proof.get("original_state_preserved") != "RUNNING"
        or len(unknown) != 1
        or unknown[0] != proof.get("original_flat_attempt")
        or proof.get("matching_native_parent_run_ids") != []
    ):
        raise pilot45.Blocked(
            "Only the exact closed C04 unsubmitted administrative row is excludable"
        )
    row = unknown[0]
    cid = row["case_id"]
    if (
        row["state"] != "RUNNING"
        or row["number"] != 2
        or row.get("run_id")
        or row.get("pid")
        or result["cases"][cid]["attempts"][-1] != proof["original_case_attempt"]
    ):
        raise pilot45.Blocked("Administrative row and original per-case evidence differ")
    control = proof["controlflow"]
    cfg = read(previous / "config.json")
    if (
        sha(control["source_snapshot"]) != control["sha256"]
        or control["sha256"]
        != cfg["source_sha256"][str(ROOT / "Data/generation/pretrain/pilot_parallel.py")]
        or control["register_line"] >= control["request_line"]
        or control.get("no_exception_handler_before_request") is not True
        or any(Path(path).exists() for path in proof["absent_paths"])
    ):
        raise pilot45.Blocked(
            "Exact frozen before-submit controlflow and absent dispatch evidence required"
        )
    snapshot = proof["registry_snapshot"]
    if sha(snapshot["path"]) != snapshot["sha256"]:
        raise pilot45.Blocked("Closed registry snapshot changed")
    import json

    if any(
        r.get("task") == previous.name
        and r.get("case_id") == cid
        and r.get("event") == "physical_attempt_started"
        and r.get("attempt") == 2
        for r in (json.loads(line) for line in Path(snapshot["path"]).read_text().splitlines())
    ):
        raise pilot45.Blocked("A submitted/claimed attempt cannot be classified administrative")
    result["cases"][cid]["attempts"].pop()
    result["attempts"] = [a for a in result["attempts"] if a != row]
    result.setdefault("administrative_attempts", []).append(
        dict(
            classification="NOT_SUBMITTED_ADMINISTRATIVE_ATTEMPT",
            proof=reference,
            original_flat_attempt=copy.deepcopy(row),
            original_case_attempt=proof["original_case_attempt"],
            source_ledger=proof["source_ledger"],
            physical_budget_consumed=False,
        )
    )
    if len(result["attempts"]) != 51 or any(not a.get("run_id") for a in result["attempts"]):
        raise pilot45.Blocked(
            "C05 retains all51 actual physical UUIDs and the separate administrative proof"
        )
    return result


def context(previous, acceptance):
    """Verify exact28 accepted/3 quality-rejected/1 native-failed/13 unstarted cases."""
    pilot45.short_authorization()
    previous = pilot45.task_path(previous)
    acceptance = ai19.no_links(acceptance)
    value = read(acceptance)
    if (
        value.get("schema") != "drone-data.short-cycle.resume-acceptance/1"
        or value.get("status") != "ACCEPTED_IDLE_ONLY_COUNTEREXPERIMENT_PREPARATION"
        or value.get("task_id") != "SHORT_CYCLE_20261107"
        or value.get("previous_task") != str(previous)
        or value.get("nonflight_when_flights_idle") is not True
        or value.get("stop_on_first_physical_failure") is not True
        or value.get("physical_attempts_max") != 3
        or value.get("authorization")
        != dict(path=str(pilot45.SHORT_AUTH), sha256=sha(pilot45.SHORT_AUTH))
    ):
        raise pilot45.Blocked("Explicit same-producer idle-only continuation acceptance required")
    verify(value["files"])
    for member in ("config.json", "manifest.json", "ledger.json", "operator/socket-cleanup.json"):
        path = previous / member
        if value["files"].get(str(path)) != sha(path):
            raise pilot45.Blocked("Complete original ledger/config/cleanup must be pinned")
    cfg, manifest, recorded = (
        read(previous / n) for n in ("config.json", "manifest.json", "ledger.json")
    )
    profile = value.get("profile", "c04_idle_only")
    if profile not in {"c04_idle_only", "c05_registration_repair"}:
        raise pilot45.Blocked("Only the specifically reviewed continuation profiles are permitted")
    ledger = physical_projection(previous, value, recorded)
    pilot45.validate_manifest(manifest)
    from .pilot_parallel import closed_previous

    closed_previous(previous)
    ai19.ai_modules()
    from learning.admission19 import check_producer

    check_producer(value["producer"], cfg["producer"])
    check_producer(value["producer"], ai19.producer_binding())
    if set(ledger["cases"]) != {c["case_id"] for c in manifest["cases"]}:
        raise pilot45.Blocked("All45 original cases must remain")
    groups = {
        k: set(value[k])
        for k in ("accepted_cases", "quality_rejections", "native_retries", "unstarted_cases")
    }
    expected = [28, 3, 1, 13] if profile == "c04_idle_only" else [31, 1, 0, 13]
    if [len(groups[k]) for k in groups] != expected:
        raise pilot45.Blocked("Continuation classes differ from the exact reviewed profile")
    union = set().union(*groups.values())
    if union != set(ledger["cases"]) or sum(map(len, groups.values())) != len(union):
        raise pilot45.Blocked("Continuation classes must partition all45 cases")
    for case in manifest["cases"]:
        cid = case["case_id"]
        entry = ledger["cases"][cid]
        if entry.get("case") != case:
            raise pilot45.Blocked("Original case/seed/group/condition changed")
        attempts = entry["attempts"]
        if cid in groups["accepted_cases"]:
            if (
                entry.get("status") != "COMPLETE"
                or not entry.get("run")
                or attempts[-1]["state"] != "COMPLETED"
            ):
                raise pilot45.Blocked("Accepted raw/view identity must be retained")
        elif cid in groups["unstarted_cases"]:
            if attempts or entry.get("run"):
                raise pilot45.Blocked("Unstarted case already has an attempt")
        else:
            if len(attempts) != 1 or attempts[0]["number"] != 1 or not attempts[0].get("run_id"):
                raise pilot45.Blocked("The four reviewed retries retain their original attempt1")
            raw = ROOT / "simulation/runs" / attempts[0]["run_id"]
            meta, scenario = read(raw / "run.json"), read(raw / "evaluation/scenario.json")
            if (
                scenario["seed"] != case["seed"]
                or scenario["operation_variation"] != case["operation_variation"]
            ):
                raise pilot45.Blocked("Raw and original requested condition differ")
            if scenario.get("producer_lock_sha256") != value["producer"]["runtime_sha256"]:
                raise pilot45.Blocked("Retry parent belongs to another producer")
            if cid in groups["quality_rejections"]:
                if (
                    meta["status"] != "completed"
                    or attempts[0]["state"] != "COMPLETED"
                    or (
                        Path(entry.get("run", "")) != raw
                        and not any(
                            Path(r["run"]) == raw for r in entry.get("quality_rejected_raws", [])
                        )
                    )
                ):
                    raise pilot45.Blocked("Preserve physical COMPLETED for rejected data")
                item = value["quality_rejections"][cid]
                if sha(ai19.no_links(item["path"])) != item["sha256"]:
                    raise pilot45.Blocked("AI quality rejection receipt changed")
            else:
                expected = value["native_retries"][cid]
                if (
                    meta["status"] != "failed"
                    or attempts[0]["state"] != "FAILED_PRESERVED"
                    or attempts[0].get("retry_allowed") is not False
                    or expected["run_id"] != raw.name
                    or expected["reason"] != meta.get("termination_reason")
                    or value["files"].get(str(raw / "native_logs/px4.log"))
                    != sha(raw / "native_logs/px4.log")
                    or "joint_phase_failure: late native hover readiness"
                    not in (raw / "native_logs/px4.log").read_text()
                ):
                    raise pilot45.Blocked(
                        "Only the exact reviewed native readiness failure is admissible"
                    )
    expected_physical = 48 if profile == "c04_idle_only" else 51
    if len(ledger["attempts"]) != expected_physical:
        raise pilot45.Blocked("All original physical attempts, including C02 failures, must remain")
    return dict(
        previous_task=str(previous),
        profile=profile,
        administrative_not_submitted=value.get("administrative_not_submitted"),
        namespace=cfg["lineage_namespace"],
        previous_task_names=sorted(
            {
                previous.name,
                *(Path(a["source_task"]).name for a in ledger["attempts"] if a.get("source_task")),
            }
        ),
        acceptance=str(acceptance),
        files=dict(value["files"], **{str(acceptance): sha(acceptance)}),
        quality_rejections=value["quality_rejections"],
        native_retries=value["native_retries"],
        accepted_cases=sorted(groups["accepted_cases"]),
        pending_native_cases=sorted(union - groups["accepted_cases"]),
        source_ledger_sha256=sha(previous / "ledger.json"),
    )


def allowed(config, case, prior):
    """One reviewed retry, never a successful raw or an unrelated new failure."""
    resume = config.get("short_resume")
    if (
        not resume
        or not prior
        or (
            resume.get("profile") != "c05_registration_repair"
            and resume["previous_task"]
            not in {prior.get("source_task"), prior.get("inherited_via_task")}
        )
    ):
        return False
    cid = case["case_id"]
    if cid not in resume["quality_rejections"] and cid not in resume["native_retries"]:
        return False
    verify(resume["files"])
    previous = Path(resume["previous_task"])
    source = read(previous / "ledger.json")
    if resume.get("administrative_not_submitted"):
        source = physical_projection(previous, read(resume["acceptance"]), source)
    original = source["cases"][cid]
    old = original["attempts"][-1]
    if (
        any(prior.get(k) != old.get(k) for k in ("number", "run_id", "state", "parent_attempt_id"))
        or prior["number"] != 1
    ):
        raise pilot45.Blocked("Continuation cannot replace the exact prior attempt/parent")
    if cid in resume["quality_rejections"]:
        if (
            prior["state"] != "COMPLETED"
            or prior.get("quality_rejection") != resume["quality_rejections"][cid]
        ):
            raise pilot45.Blocked(
                "Explicit quality rejection required for this physically completed parent"
            )
    elif prior["state"] != "FAILED_PRESERVED" or prior.get("retry_allowed") is not False:
        raise pilot45.Blocked("Reviewed native failure must remain immutable")
    return True


def current_reader_evidence(spec, previous, ledger, resume):
    """Reuse the owner's actual28 current audit result; preserve original export bindings."""
    validation = spec["validation"]["current_readers_existing28"]
    item = validation["evidence"]
    if (
        validation.get("exit_code") != 0
        or validation.get("count") != 28
        or sha(item["path"]) != item["sha256"]
    ):
        raise pilot45.Blocked("Hash-bound actual28 current reader audit required")
    checked = read(item["path"])
    results = {row["case_id"]: row for row in checked["results"]}
    if (
        checked.get("exit_code") != 0
        or checked.get("passed") != 28
        or checked.get("torch_imported") is not False
        or not set(results) <= set(resume["accepted_cases"])
        or len(checked["results"]) != 28
    ):
        raise pilot45.Blocked("Current reader report must cover exactly the28 preserved exports")
    for cid, row in results.items():
        e = ledger["cases"][cid]
        result = row["result"]
        if (
            Path(row["raw"]) != Path(e["run"])
            or Path(row["export"]) != Path(e["output"])
            or row["raw_integrity_sha256"] != sha(Path(e["run"]) / "integrity.json")
            or result.get("status") != "INDEPENDENT_INPUT_QUALITY_CHECKED_NO_FIT"
            or result.get("run_id") != Path(e["run"]).name
            or result.get("export_receipt_sha256") != sha(Path(e["output"]) / "receipt.json")
            or result.get("cohort") != pilot19.SHORT_COHORT
            or result.get("online_available") is not True
            or result.get("run_available") is not True
            or result.get("mission_frames", 0) <= 100
            or result.get("online_windows", 0) <= 0
        ):
            raise pilot45.Blocked("Current reader audit does not bind this exact original export")
    additional = set(resume["accepted_cases"]) - set(results)
    expected_additional = 3 if resume.get("profile") == "c05_registration_repair" else 0
    if len(additional) != expected_additional:
        raise pilot45.Blocked("Only the exact additional3 completed C04 exports may be reused")
    audits = {}
    if additional:
        old_sources = read(previous / "config.json")["source_sha256"]
        for name, digest in read(spec["source_freeze"]["path"])["files"].items():
            if old_sources.get(str(ROOT / name)) != digest:
                raise pilot45.Blocked("Current AI19 reader differs from the C04 successful audits")
    for cid in additional:
        e = ledger["cases"][cid]
        binding = read(Path(e["output"]) / "binding.json")
        audit = read(e["audit_receipt"])
        if (
            binding.get("consumer_sources") != pilot19.helper_inventory()
            or audit.get("status") != "INDEPENDENT_INPUT_QUALITY_CHECKED_NO_FIT"
            or audit.get("run_id") != Path(e["run"]).name
            or audit.get("export_receipt_sha256") != sha(Path(e["output"]) / "receipt.json")
            or audit.get("online_available") is not True
            or audit.get("run_available") is not True
            or audit.get("mission_frames", 0) <= 100
            or audit.get("online_windows", 0) <= 0
        ):
            raise pilot45.Blocked(
                "Additional C04 success must retain its same-reader independent audit"
            )
        audits[cid] = dict(path=e["audit_receipt"], sha256=sha(e["audit_receipt"]))
    return dict(item, additional_audits=audits)


def prepare(name, previous, acceptance, handoff):
    """Build and statically seal a new task; no service/native/model invocation."""
    from . import pilot_bridge, pilot_parallel

    if not pilot45.re.fullmatch(r"[A-Za-z0-9_-]{1,60}", name):
        raise pilot45.Blocked("Safe new continuation name required")
    previous = pilot45.task_path(previous)
    resume = context(previous, acceptance)
    spec = pilot_bridge.dependencies(handoff)
    cfg, manifest, old = (
        read(previous / n) for n in ("config.json", "manifest.json", "ledger.json")
    )
    task = pilot45.task_path(HOME / "tasks" / (name + "-45"))
    if task.exists() or (HOME / "tasks" / name).exists():
        raise pilot45.Blocked("Existing continuation evidence must never be overwritten")
    old = physical_projection(previous, read(acceptance), old)
    ledger = copy.deepcopy(old)
    reader_proof = current_reader_evidence(spec, previous, old, resume)
    before_ref = read(acceptance)["prior_Data_audit_source"]
    prior_cfg = read(previous / "config.json")
    if sha(before_ref["path"]) != before_ref["sha256"] or before_ref["sha256"] != prior_cfg[
        "source_sha256"
    ].get(str(ROOT / "Data/generation/pretrain/pilot19.py")):
        raise pilot45.Blocked("Exact original Data reader source beforeimage required")

    def audit_ast(path):
        tree = ast.parse(Path(path).read_text())
        return ast.dump(
            next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "audit_one"),
            include_attributes=False,
        )

    if audit_ast(before_ref["path"]) != audit_ast(ROOT / "Data/generation/pretrain/pilot19.py"):
        raise pilot45.Blocked("Data independent audit function changed beyond this reuse review")
    for numeric in (
        "ai/src/capstone_ai/mapping.py",
        "ai/src/capstone_ai/source.py",
        "ai/src/capstone_ai/dataset.py",
        "ai/learning/history19/preprocessing.py",
        "ai/learning/history19/preprocessing.json",
        "ai/learning/history19/data.py",
        "ai/learning/run_health19/data.py",
    ):
        path = ROOT / numeric
        if cfg["source_sha256"].get(str(path)) != sha(path):
            raise pilot45.Blocked(
                "Numerical reader/source changed beyond this ledger-only review: " + numeric
            )
    for cid, entry in ledger["cases"].items():
        for a in entry["attempts"]:
            a.setdefault("source_task", str(previous))
            a.setdefault("source_ledger_sha256", resume["source_ledger_sha256"])
            a.setdefault("inherited_via_task", str(previous))
        if cid in resume["quality_rejections"]:
            entry["attempts"][-1]["quality_rejection"] = resume["quality_rejections"][cid]
            if entry.get("run"):
                entry.setdefault("quality_rejected_raws", []).append(
                    dict(
                        run=entry.pop("run"),
                        physical_status="completed",
                        quality_rejection=resume["quality_rejections"][cid],
                    )
                )
        elif cid in resume["accepted_cases"]:
            pilot_parallel.check_raw(entry["run"])
            verify(entry["files"])
            entry.setdefault("inherited_from", str(previous))
            entry.setdefault(
                "prior_processing",
                [
                    {
                        k: copy.deepcopy(entry[k])
                        for k in (
                            "receipt",
                            "output",
                            "audit_receipt",
                            "files",
                            "processing_attempts",
                        )
                    }
                ],
            )
            ref = reader_proof.get("additional_audits", {}).get(cid) or {
                k: reader_proof[k] for k in ("path", "sha256")
            }
            entry["reuse_audit"] = dict(ref, case_id=cid)
            entry["consumer_compatibility"] = (
                "original_binding_and_arrays_preserved_current_readers_reaudited"
            )
    indexed = {(cid, a["number"]): a for cid, e in ledger["cases"].items() for a in e["attempts"]}
    ledger["attempts"] = [
        dict(indexed[a["case_id"], a["number"]], case_id=a["case_id"]) for a in old["attempts"]
    ]
    ledger["status"] = "NOT_STARTED"
    task.mkdir(parents=True)
    cfg = copy.deepcopy(cfg)
    cfg.pop("short_repair", None)
    cfg.update(
        sealed=False,
        source_sha256={},
        ai_ready=None,
        short_resume=resume,
        lineage_namespace=resume["namespace"],
        nonflight_when_flights_idle=True,
        stop_on_first_physical_failure=True,
        parallel16=False,
        service_unit="drone-pretrain-" + name + "-45.service",
        owner_uid=os.getuid(),
    )
    write(task / "config.json", cfg)
    write(task / "manifest.json", manifest)
    (task / "manifest.sha256").write_text(sha(task / "manifest.json") + "\n")
    write(task / "ledger.json", ledger)
    write(task / "budget.json", read(previous / "budget.json"))
    protected = pilot45.protection()
    owned = {a["run_id"] for a in ledger["attempts"] if a.get("run_id")}
    protected["records"] = [r for r in protected["records"] if r.get("run_id") not in owned]
    write(task / "protection.json", protected)
    write(
        task / "global-protection.json",
        pilot45.global_protection(
            protected, manifest["cases"], resume["namespace"], cohort=pilot19.SHORT_COHORT
        ),
    )
    for cid in resume["accepted_cases"]:
        old_cache = previous / "pilot/execution" / cid / "archive-cache.json"
        cached = read(old_cache)
        raw = Path(ledger["cases"][cid]["run"])
        if cached["raw_integrity_sha256"] != sha(raw / "integrity.json") or cached["archive"][
            "sha256"
        ] != sha(cached["archive"]["archive"]):
            raise pilot45.Blocked("Existing verified archive/cache changed")
        write(task / "pilot/execution" / cid / "archive-cache.json", cached)
    write(
        task / "operator/state.json",
        dict(status="DRAFT_NOT_STARTED", task=task.name, updated_utc=now()),
    )
    flow = pilot_parallel.prepare_short(task, handoff)
    return flow, task
