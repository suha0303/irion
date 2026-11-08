"""MAIN-only known native-gap rejection proof; no retry submission or raw writes."""

import csv
import sys
from pathlib import Path

from learning.admission19 import check_producer
from learning.history19.pilot import ROOT, json_file, regular, sha

from .cases import reference, verify_source_snapshot
from .contract import (
    TASK,
    bound,
    case_key,
    check_scope,
    validate_policy,
    write_new,
)

ERROR = (
    "Joint native consumption: Incomplete actual reference consumption coverage"
)


def _inspect(
    evidence_root, *, case_id, producer, policy_bundle, assessment_ref
):
    plans = check_scope()
    case = plans["cases"][case_key(case_id)]
    policy = json_file(policy_bundle["paired_policy"])
    gap_limit = policy.get("native_consumption_max_gap_us", 40000)
    validate_policy(policy, policy_bundle["native_policy"])
    check_producer(policy["producer"], producer)
    raw = Path(evidence_root).absolute()
    state, scenario = (
        json_file(raw / "run.json"),
        json_file(raw / "evaluation/scenario.json"),
    )
    fixture = policy.get("fixture_only") is True
    if state.get("status") != "completed" or state.get("origin") != (
        "synthetic_unit_fixture" if fixture else "px4_sitl"
    ):
        raise ValueError("Physical completed status must remain original")
    if (
        any(scenario.get(k) != case[k] for k in ("seed", "operation_variation"))
        or scenario.get("producer_lock_sha256") != producer["runtime_sha256"]
        or scenario.get("name") != case["scenario"]
        or scenario.get("requested_duration_s") != 480
    ):
        raise ValueError(
            "MAIN rejected raw does not match original source/case/seed"
        )
    if not fixture:
        verify_source_snapshot(raw, producer)
    integrity = json_file(raw / "integrity.json")
    members = (
        "run.json",
        "evaluation/scenario.json",
        "evaluation/joint-origin.json",
        "evaluation/joint-landing.json",
        "evaluation/joint-physics-origin.json",
        "evaluation/joint-consumption.csv",
        "evaluation/joint-reference.csv",
        "reference/source_lock.json",
    )
    refs = {}
    for name in members:
        path = regular(raw / name)
        if integrity[name] != dict(sha256=sha(path), size=path.stat().st_size):
            raise ValueError(
                "Rejected raw differs from original closing integrity"
            )
        refs[name] = reference(path)
    start_receipt = json_file(raw / "evaluation/joint-origin.json")
    end_receipt = json_file(raw / "evaluation/joint-landing.json")
    physics = json_file(raw / "evaluation/joint-physics-origin.json")
    first_us = start_receipt.get("actual_start_us")
    if (
        type(first_us) is not int
        or first_us < 60000000
        or state.get("clock", {}).get("id") != "px4-boot"
        or state.get("clock", {}).get("unit") != "us"
        or start_receipt.get("clock") != "px4-boot"
        or end_receipt.get("clock") != "px4-boot"
        or start_receipt.get("kind") != "actual_reference_consumption_start"
        or end_receipt.get("kind") != "actual_auto_land_reference_consumption"
        or start_receipt.get("actual_time_us") != first_us
        or end_receipt.get("actual_start_us") != first_us
        or physics.get("schema") != "drone.joint-physics-origin/1"
        or physics.get("origin_us") != first_us
        or type(physics.get("notice_us")) is not int
        or not first_us <= physics["notice_us"] <= first_us + 500000
    ):
        raise ValueError(
            "Invalid native clock/start/LAND/physics notice is not a gap-only rejection"
        )
    assessment_path = bound(assessment_ref)
    assessment = json_file(assessment_path)
    if (
        assessment.get("schema") != "drone.run-assessment/1"
        or assessment.get("run_id") != state["run_id"]
        or assessment.get("errors") != [ERROR]
        or assessment.get("contract_passed") is not False
        or assessment.get("physical_gate_passed") is not False
        or assessment.get("missing_data_passed") is not True
        or assessment.get("instance_isolation_passed") is not True
    ):
        raise ValueError(
            "Only the known official native-gap rejection is retryable"
        )
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "simulation/sim/src"))
    try:
        from capstone_sim.joint_phase import verify

        try:
            verify(raw)
        except ValueError as error:
            if str(error) != "Incomplete actual reference consumption coverage":
                raise ValueError(
                    "Unknown native failure cannot authorize quality reflight"
                ) from error
        else:
            raise ValueError(
                "Valid native raw cannot be quality-rejected for reflight"
            )
    finally:
        sys.dont_write_bytecode = prior
    start = json_file(raw / "evaluation/joint-origin.json")["actual_start_us"]
    end = json_file(raw / "evaluation/joint-landing.json")["actual_time_us"]
    first = previous = None
    maximum, over, indices = 0, 0, set()
    with (raw / "evaluation/joint-consumption.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            stamp = int(row["source_us"])
            first = stamp if first is None else first
            gap = 0 if previous is None else stamp - previous
            maximum, over = max(maximum, gap), over + (gap > gap_limit)
            previous = stamp
            indices.add(int(row["index"]))
    if (
        first != start
        or indices != set(range(4801))
        or previous is None
        or not start + 480000000 <= previous <= end <= start + 480100000
        or maximum <= gap_limit
    ):
        raise ValueError(
            "Native gap alone, with complete4801reference/fullmission, required"
        )
    return dict(
        schema="drone-ai.collection19.quality-rejection/1",
        task_id=TASK,
        case_id=case["case_id"],
        run_id=state["run_id"],
        seed=case["seed"],
        physical_status="completed",
        reason="native_consumption_gap_exceeded",
        failure_reason=ERROR,
        producer=policy["producer"],
        evidence_root=str(raw),
        policy_sha256=sha(regular(policy_bundle["paired_policy"])),
        policy_refs={
            "paired_policy": reference(policy_bundle["paired_policy"]),
            "native_policy": reference(policy_bundle["native_policy"]),
            "origin_relative_path": policy_bundle["origin_relative_path"],
        },
        assessment=reference(assessment_path),
        physical_run=refs["run.json"],
        refs=refs,
        raw_integrity=reference(raw / "integrity.json"),
        measured=dict(
            max_gap_us=maximum,
            gaps_over_limit=over,
            reference_rows_seen=4801,
            gap_limit_us=gap_limit,
        ),
        fixture_only=fixture,
        actual_data_verified=not fixture,
        authorizes_flight=False,
    )


def publish_quality_rejection(
    evidence_root, *, case_id, producer, policy_bundle, assessment_ref, output
):
    value = _inspect(
        evidence_root,
        case_id=case_id,
        producer=producer,
        policy_bundle=policy_bundle,
        assessment_ref=assessment_ref,
    )
    return write_new(output, value)


def verify_quality_rejection(
    ref, attempt, producer, policy_bundle=None, *, case_id=None
):
    value = json_file(bound(ref))
    if attempt.get("status") not in {"completed", "quality_rejected"} or (
        case_id is not None and case_key(case_id) != value["case_id"]
    ):
        raise ValueError(
            "Physical COMPLETED or logical case cannot be disguised during retry"
        )
    if policy_bundle is None:
        policy_bundle = {
            key: str(bound(value["policy_refs"][key]))
            for key in ("paired_policy", "native_policy")
        }
        policy_bundle["origin_relative_path"] = value["policy_refs"][
            "origin_relative_path"
        ]
    expected = _inspect(
        value["evidence_root"],
        case_id=value["case_id"],
        producer=producer,
        policy_bundle=policy_bundle,
        assessment_ref=value["assessment"],
    )
    if (
        value != expected
        or value["run_id"] != attempt["run_id"]
        or value["seed"] != attempt["seed"]
    ):
        raise ValueError("Explicit MAIN quality rejection evidence differs")
    state = json_file(bound(value["physical_run"]))
    if state.get("parent_attempt_id") != attempt.get("parent_attempt_id"):
        raise ValueError("Original rejected physical parent chain changed")
    return True


check_quality_rejection = verify_quality_rejection
