"""Hash-bound external full-gate evidence; historical generation is never relabelled."""

from pathlib import Path

from .pilot import json_file, regular, required, sha

REVISION = "native-consumption-identical-clock-v2"
GATES = (
    "contract_passed",
    "instance_isolation_passed",
    "missing_data_passed",
    "physical_gate_passed",
)
RAW_MEMBERS = {
    "run.json",
    "evaluation/scenario.json",
    "reference/source_lock.json",
    "evaluation/joint-origin.json",
    "evaluation/joint-landing.json",
    "evaluation/joint-consumption.csv",
    "evaluation/joint-reference.csv",
    "evaluation/joint-physics-origin.json",
    "evaluation/joint-phase-policy.json",
}


def reference(value):
    path = regular(required(value, "path"))
    if sha(path) != required(value, "sha256"):
        raise ValueError("External audit reference changed: " + str(path))
    return path


def verified_check(
    run_path, external_audit, *, generation, policy_sha256, duration_us, fixture_only=False
):
    """Verify Sim audit/2 plus the separate full assessment. Does not invoke Sim APIs.

    auditor_binding is an explicitly pinned AI hash-verification receipt created
    while the six historical generation dependencies were available. Their bytes
    need not equal a different, currently installed short-cycle producer.
    """
    run_path = regular(run_path)
    raw = run_path.parent
    receipt_path = reference(required(external_audit, "receipt"))
    binding_path = reference(required(external_audit, "auditor_binding"))
    if receipt_path.is_relative_to(raw):
        raise ValueError("External audit evidence cannot backfill the original raw")
    binding, receipt = json_file(binding_path), json_file(receipt_path)
    if (
        binding.get("schema") != "drone-ai.verified-auditor-binding/1"
        or binding.get("revision") != REVISION
        or binding.get("generation_dependencies_verified_at_freeze") is not True
        or receipt.get("schema") != "drone.native-consumption-audit/2"
        or receipt.get("status") != "passed"
        or receipt.get("full_gate_passed") is not True
    ):
        raise ValueError("Verified auditor binding and actual full-gate audit/2 required")
    manifest_path = reference(binding["manifest"])
    manifest = json_file(manifest_path)
    if (
        manifest.get("schema") != "drone.native-consumption-audit-manifest/1"
        or manifest.get("revision") != REVISION
        or binding["audit_files"] != manifest["audit_files"]
        or binding["generation_dependencies"] != manifest["generation_dependencies"]
    ):
        raise ValueError("Historical auditor manifest/dependency binding changed")
    for name, digest in manifest["audit_files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Auditor file must stay in its frozen package")
        if sha(regular(manifest_path.parent / name)) != digest:
            raise ValueError("Auditor implementation differs from frozen manifest: " + name)
    expected_generation = dict(
        source_sha256=generation["source_manifest_sha256"],
        runtime_sha256=generation["runtime_sha256"],
        policy_sha256=policy_sha256,
    )
    if binding["generation"] != expected_generation or any(
        manifest.get("generation_" + key) != value for key, value in expected_generation.items()
    ):
        raise ValueError("External audit belongs to another generation, not the current producer")
    proof = required(receipt, "audit")
    declared_manifest = regular(required(proof, "manifest_path"))
    if sha(declared_manifest) != sha(manifest_path):
        raise ValueError("Historical auditor relocation must preserve the manifest content hash")
    for name, digest in manifest["audit_files"].items():
        if sha(regular(declared_manifest.parent / name)) != digest:
            raise ValueError("Declared historical auditor package differs: " + name)
    state, scenario = json_file(run_path), json_file(raw / "evaluation/scenario.json")
    if state.get("origin") != ("synthetic_unit_fixture" if fixture_only else "px4_sitl"):
        raise ValueError("External audit fixture cannot pass as actual raw")
    integrity_path = regular(raw / "integrity.json")
    integrity = json_file(integrity_path)
    if (
        proof.get("revision") != REVISION
        or proof.get("manifest_sha256") != sha(manifest_path)
        or proof.get("run_id") != state["run_id"]
        or Path(required(proof, "raw_path")).absolute() != raw
        or proof.get("raw_integrity_sha256") != sha(integrity_path)
        or proof.get("original_raw_unchanged") is not True
        or proof.get("original_termination_reason") != state.get("termination_reason")
        or proof.get("original_primary_failure") != state.get("primary_failure")
        or scenario.get("producer_lock_sha256") != generation["runtime_sha256"]
        or scenario.get("joint_phase_policy", {}).get("policy_sha256") != policy_sha256
        or any(
            proof.get("generation_" + key) != value for key, value in expected_generation.items()
        )
    ):
        raise ValueError("External audit run/raw integrity/original failure/generation mismatch")
    original_check_present = (raw / "evaluation/joint-check.json").is_file()
    if proof.get("raw_joint_check_present") is not original_check_present:
        raise ValueError("Original CHECK presence must not be rewritten")
    inputs = required(proof, "raw_files_sha256")
    if set(inputs) != RAW_MEMBERS:
        raise ValueError("All original native audit inputs must be bound")
    for name, digest in inputs.items():
        path = regular(raw / name)
        if (
            integrity[name]["sha256"] != digest
            or integrity[name]["size"] != path.stat().st_size
            or sha(path) != digest
        ):
            raise ValueError(
                "External audit input differs from original closing integrity: " + name
            )
    full_path = reference(required(receipt, "full_gate_receipt"))
    if full_path.is_relative_to(raw):
        raise ValueError("Full assessment must remain external to the original raw")
    full = json_file(full_path)
    if (
        full.get("schema") != "drone.run-assessment/1"
        or full.get("run_id") != state["run_id"]
        or full.get("audit_full_gate_passed") is not True
        or any(full.get(key) is not True for key in GATES)
        or full.get("cpu_audit") != proof
    ):
        raise ValueError("api.verify alone is insufficient: actual four full gates are required")
    checked = required(full, "native_consumption_audit")
    if checked.get("status") != "passed" or any(
        receipt.get(key) != value for key, value in checked.items()
    ):
        raise ValueError("External native and full assessment summaries differ")
    first, last = checked["actual_start_us"], checked["actual_landing_us"]
    total, duplicates, unique = (
        checked[k]
        for k in ("consumed_samples", "same_source_time_consumptions", "unique_source_time_samples")
    )
    if (
        any(type(v) is not int for v in (first, last, total, duplicates, unique))
        or total != duplicates + unique
        or duplicates < 0
        or unique < 1
        or checked.get("audit_revision") != REVISION
        or checked.get("backward_source_time_count") != 0
        or checked.get("conflicting_same_source_time_count") != 0
        or checked.get("first_consumption_source_us") != first
        or not first + duration_us <= checked.get("last_consumption_source_us", -1) <= last
        or not first + duration_us <= last <= first + duration_us + 100000
        or checked.get("terminal_hold_tail_us") != last - first - duration_us
        or checked.get("reference_rows_seen") != duration_us // 100000 + 1
        or not 0 <= checked.get("max_consumption_gap_us", -1) <= 40000
    ):
        raise ValueError("External native counts/order/full duration/coverage failed")
    return checked, dict(
        receipt_sha256=sha(receipt_path),
        full_gate_receipt_sha256=sha(full_path),
        auditor_binding_sha256=sha(binding_path),
        auditor_manifest_sha256=sha(manifest_path),
        generation=expected_generation,
        raw_integrity_sha256=sha(integrity_path),
    )
