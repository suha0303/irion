"""Independent case input audit, including a lossless native-evidence capsule."""

import sys
from pathlib import Path

import numpy as np
from capstone_ai.dataset import feature_transform, target_transform
from capstone_ai.mapping import FEATURES, TARGETS
from execution.common import child

from learning.admission19 import (
    check_producer,
    check_protection,
    source_evidence,
    validate_lineages,
)
from learning.history19.data import inspect_run
from learning.history19.pilot import ROOT, json_file, regular, sha
from learning.history19.pilot_connector import native_mission_interval
from learning.run_health19.data import iter_manifests

from .contract import (
    bound,
    case_key,
    check_case_record,
    check_scope,
    read_protection,
    validate_policy,
    write_new,
)
from .streams import array_blocks, canonical_matrix, grid

ATOL = RTOL = 1e-6


def reference(path):
    path = regular(path)
    return dict(path=str(path), sha256=sha(path))


def verify_source_snapshot(raw, producer):
    reviewed = ROOT / "simulation/config/producer-sources.json"
    runtime_path = ROOT / "simulation/artifacts/runtime-manifest.json"
    snapshot = json_file(Path(raw) / "reference/source_lock.json")
    expected_sources = json_file(reviewed)["files"]
    runtime = json_file(runtime_path)
    binaries = {
        "vendor/PX4-Autopilot/build/px4_sitl_default/bin/px4": runtime[
            "px4_binary_sha256"
        ],
        "build/current/libDronePowertrain.so": runtime["physics_plugin_sha256"],
    }
    transport = "runtime/gazebo-transport/lib/libgz-transport13.so.13.6.0"
    if transport in runtime["files"]:
        binaries[transport] = runtime["files"][transport]
    if (
        sha(reviewed) != producer["source_manifest_sha256"]
        or sha(runtime_path) != producer["runtime_sha256"]
        or any(
            snapshot.get("local_sources", {}).get(k) != v
            for k, v in expected_sources.items()
            if k.startswith(("sim/", "profiles/", "contracts/"))
        )
        or snapshot.get("runtime_binary_sha256") != binaries
    ):
        raise ValueError(
            "Original source/runtime snapshot differs from current producer"
        )


def _row_paths(export_root):
    root = Path(export_root).absolute()
    online, whole = (
        json_file(root / "online.json"),
        json_file(root / "run.json"),
    )
    if len(whole.get("runs", [])) != 1:
        raise ValueError("A per-case export must contain one whole-run record")
    row, other = online["run"], whole["runs"][0]
    for key in (
        "case_id",
        "run_id",
        "role",
        "cohort",
        "purpose",
        "lineage",
        "mission_interval",
        "source_evidence",
        "origin",
    ):
        if row.get(key) != other.get(key):
            raise ValueError(
                "Both19 views must have identical scope/mission/source identity"
            )
    return root, online, whole, row


def compare_views(export_root, *, producer, protection):
    """One bounded run, using current model readers and every canonical eligible origin."""
    root, online, whole, row = _row_paths(export_root)
    keys = {"case:" + row["case_id"], "run:" + row["run_id"], *row["lineage"]}
    registry = read_protection(protection, keys)
    validate_lineages([row])
    check_protection([row], registry, usage="input_check")
    check_producer(online["producer"], producer)
    check_producer(whole["producer"], producer)
    checked = inspect_run(
        root / "online.json", expected_producer=producer, protection=registry
    )
    iterator = iter_manifests(
        [root / "run.json"], expected_producer=producer, protection=registry
    )
    run = next(iterator)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("Per-case whole manifest contained another run")
    physical, times, segment = canonical_matrix(root / "frames.jsonl", row)
    if not np.array_equal(run.time_us, times) or not np.allclose(
        run.values, physical, atol=ATOL, rtol=RTOL
    ):
        raise ValueError(
            "Whole-run array differs from the complete canonical mission/tail"
        )
    config = json_file(ROOT / "ai/learning/history19/preprocessing.json")
    if (
        config.get("nominal_capacity_ah") != 12
        or config.get("history") != 100
        or config.get("future") != 10
    ):
        raise ValueError(
            "Fixed common19/history100/future10/nameplate contract changed"
        )
    transformed = feature_transform(physical, config)
    target = target_transform(physical[:, [FEATURES.index(k) for k in TARGETS]])
    source = child(root, online["file"]["path"])
    with np.load(source, allow_pickle=False) as arrays:
        origins, segments = arrays["origin_us"], arrays["segment_id"]
    eligible = [
        int(t)
        for t in times
        if t % 1000000 == 0
        and t - 9900000 >= times[0]
        and t + 1000000 <= times[-1]
    ]
    if (
        origins.tolist() != eligible
        or not eligible
        or not np.all(segments == segment)
    ):
        raise ValueError(
            "Online windows omitted/duplicated eligible mission origins or mixed segments"
        )
    at = {int(t): i for i, t in enumerate(times)}
    for key in ("X", "Y"):
        for start, block in array_blocks(source, key):
            expected = np.empty_like(block, dtype=np.float64)
            for j, origin in enumerate(origins[start : start + len(block)]):
                index = at[int(origin)]
                expected[j] = (
                    transformed[index - 99 : index + 1]
                    if key == "X"
                    else target[index + 1 : index + 11]
                )
            if not np.allclose(block, expected, atol=ATOL, rtol=RTOL):
                raise ValueError(
                    "Window features/arrived targets differ from causal canonical values"
                )
    first, last, count = grid(row)
    return dict(
        row=row,
        frames=count,
        first_time_us=first,
        last_time_us=last,
        segment_id=segment,
        clock_id=row["mission_interval"]["clock_id"],
        windows=checked["windows"],
        input_refs={
            "canonical": reference(root / "frames.jsonl"),
            "online_manifest": reference(root / "online.json"),
            "whole_manifest": reference(root / "run.json"),
            "online_array": reference(source),
            "whole_array": reference(child(root, whole["runs"][0]["path"])),
        },
    )


def audit_case(
    export_root,
    *,
    producer,
    protection,
    policy_bundle,
    evidence_root,
    assessment_ref,
    output=None,
):
    scope = check_scope()
    policy = json_file(policy_bundle["paired_policy"])
    validate_policy(policy, policy_bundle["native_policy"])
    check_producer(policy["producer"], producer)
    checked = compare_views(
        export_root, producer=producer, protection=protection
    )
    row = checked["row"]
    case = scope["cases"].get(case_key(row["case_id"]))
    if case is None:
        raise ValueError("Run is outside the approved1674 plan")
    check_case_record(row, case)
    fixture = policy.get("fixture_only") is True
    if row.get("origin") != (
        "synthetic_unit_fixture" if fixture else "px4_sitl"
    ):
        raise ValueError(
            "Synthetic data cannot be promoted to actual collection admission"
        )
    if not fixture:
        source_evidence(row, producer)
    raw = Path(evidence_root).absolute()
    state = json_file(raw / "run.json")
    scenario = json_file(raw / "evaluation/scenario.json")
    integrity = json_file(raw / "integrity.json")
    required = (
        "run.json",
        "evaluation/scenario.json",
        "evaluation/joint-origin.json",
        "evaluation/joint-landing.json",
        "evaluation/joint-physics-origin.json",
        "evaluation/joint-check.json",
        "evaluation/joint-consumption.csv",
        "evaluation/joint-reference.csv",
        "evaluation/battery-selection.json",
        "evaluation/propulsion-selection.json",
        "reference/source_lock.json",
    )
    hashes = {}
    for member in required:
        path = regular(raw / member)
        hashes[member] = sha(path)
        if integrity[member] != dict(
            sha256=hashes[member], size=path.stat().st_size
        ):
            raise ValueError(
                "Evidence capsule differs from original closing: " + member
            )
    if (
        state.get("run_id") != row["run_id"]
        or state.get("status") != "completed"
    ):
        raise ValueError(
            "Native evidence is not the accepted original physical run"
        )
    if any(scenario.get(k) != case[k] for k in ("seed", "operation_variation")):
        raise ValueError(
            "Actual seed/path/wind differs from immutable planned definition"
        )
    if scenario.get("name") != case["scenario"]:
        raise ValueError(
            "Source scenario/fault differs from immutable planned case"
        )
    for name, member in (
        ("battery_selection", "evaluation/battery-selection.json"),
        ("propulsion_selection", "evaluation/propulsion-selection.json"),
    ):
        if row["source_evidence"][name] != json_file(raw / member):
            raise ValueError(
                "Selection provenance differs from original native source"
            )
    if not fixture:
        verify_source_snapshot(raw, producer)
    mission = native_mission_interval(
        raw / policy_bundle["origin_relative_path"],
        policy_path=policy_bundle["paired_policy"],
        native_policy_path=policy_bundle["native_policy"],
        run_path=raw / "run.json",
    )
    if (
        mission != row["mission_interval"]
        or mission.get("mission_duration_us") != 480000000
    ):
        raise ValueError(
            "Both19 views must preserve the actual480 native start/LAND/tail"
        )
    assessment_path = bound(assessment_ref)
    assessment = json_file(assessment_path)
    if (
        assessment.get("schema") != "drone.run-assessment/1"
        or assessment.get("run_id") != row["run_id"]
        or any(
            assessment.get(k) is not True
            for k in (
                "contract_passed",
                "physical_gate_passed",
                "missing_data_passed",
                "instance_isolation_passed",
            )
        )
        or assessment.get("errors")
    ):
        raise ValueError(
            "Preserved official pre-retirement full assessment must pass all gates"
        )
    source_binding = assessment.get("source_binding")
    if source_binding is not None:
        expected_binding = dict(
            schema="drone-data.native-assessment-binding/1",
            raw_integrity_sha256=sha(raw / "integrity.json"),
            producer_source_manifest_sha256=producer["source_manifest_sha256"],
            producer_runtime_sha256=producer["runtime_sha256"],
            native_policy_sha256=policy["native_policy_sha256"],
            native_check_sha256=hashes["evaluation/joint-check.json"],
            native_verifier_executed=True,
        )
        if source_binding != expected_binding:
            raise ValueError("Completed native assessment binding changed")
        native_policy_path = regular(raw / "evaluation/joint-phase-policy.json")
        if (
            integrity.get("evaluation/joint-phase-policy.json")
            != dict(
                sha256=expected_binding["native_policy_sha256"],
                size=native_policy_path.stat().st_size,
            )
            or sha(native_policy_path)
            != expected_binding["native_policy_sha256"]
        ):
            raise ValueError(
                "Assessment native policy differs from original closing"
            )
        consumed = json_file(raw / "evaluation/joint-check.json")
        native_verification = "reused_hash_bound_full_assessment"
    else:
        # Legacy receipts lack the required binding and still need the verifier.
        # Never call joint_phase.check(), which would write the original raw.
        prior = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(ROOT / "simulation/sim/src"))
        try:
            from capstone_sim.joint_phase import verify

            consumed = verify(raw)
        finally:
            sys.dont_write_bytecode = prior
        if consumed != json_file(raw / "evaluation/joint-check.json"):
            raise ValueError(
                "Preserved CHECK differs from actual native CSV/reference verification"
            )
        native_verification = "legacy_receipt_native_csv_rechecked"
    result = dict(
        checked,
        schema="drone-ai.collection19.case-audit/1",
        task_id=scope["authorization"]["task_id"],
        passed=True,
        status="CASE_INPUTS_AUDITED_NO_MODEL",
        actual_data_verified=not fixture,
        fixture_only=fixture,
        producer=producer,
        case_id=case["case_id"],
        policy_sha256=sha(regular(policy_bundle["paired_policy"])),
        assessment=reference(assessment_path),
        native_sources_sha256=hashes,
        raw_integrity_sha256=sha(raw / "integrity.json"),
        native_check=consumed,
        native_verification=native_verification,
        comparison_tolerance=dict(atol=ATOL, rtol=RTOL),
        eligible_for_fit=False,
        audit_scope="Canonical/arrays checked once; hash-bound full assessment reused for native evidence, legacy receipts rechecked",
        model_forward_steps=0,
        optimizer_steps=0,
        GPU_jobs=0,
    )
    receipt = write_new(output, result) if output is not None else None
    return dict(result, receipt=receipt)
