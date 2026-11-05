"""Full480 synthetic native/public API integration; no producer command or physical run."""

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from test_collection19 import export
from test_pilot_publisher import fixture as native_fixture

from learning.collection19 import (
    audit_case,
    check_quality_rejection,
    contract,
    prepare_policy,
    publish_quality_rejection,
)
from learning.history19.pilot import ROOT, json_file, sha
from learning.history19.pilot_connector import native_mission_interval

ag = importlib.import_module("learning.collection19.aggregate")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return dict(path=str(path), sha256=sha(path))


def refresh(raw):
    members = {
        str(p.relative_to(raw)): dict(sha256=sha(p), size=p.stat().st_size)
        for p in raw.rglob("*")
        if p.is_file() and p.name != "integrity.json"
    }
    put(raw / "integrity.json", members)


@pytest.fixture
def full480(tmp_path, monkeypatch):
    args = native_fixture(tmp_path / "native")
    binding = json_file(args["producer_binding"])
    native = json_file(ROOT / "simulation/profiles/collection/joint-phase-policy.json")
    args["native_policy"].write_bytes(
        (ROOT / "simulation/profiles/collection/joint-phase-policy.json").read_bytes()
    )
    auth = json_file(contract.AUTH)
    auth.update(
        fixture_only=True,
        producer=dict(
            source_sha256=binding["source_manifest_sha256"],
            runtime_sha256=binding["runtime_sha256"],
        ),
    )
    auth_path = tmp_path / "fixture-authorization.json"
    put(auth_path, auth)
    monkeypatch.setattr(contract, "AUTH", auth_path)
    monkeypatch.setattr(contract, "AUTH_SHA256", sha(auth_path))
    policy_path = tmp_path / "policy.json"
    prepare_policy(args["native_policy"], args["producer_binding"], policy_path, fixture_only=True)
    bundle = dict(
        paired_policy=str(policy_path),
        native_policy=str(args["native_policy"]),
        origin_relative_path="evaluation/joint-origin.json",
    )
    case = contract.check_scope()["cases"]["lr-normal_train-00-0-0"]
    raw = tmp_path / "native/synthetic-0-normal"
    first, last = 60000000, 540024000
    start = json_file(raw / "evaluation/joint-origin.json")
    start.update(
        revision=native["revision"], policy_sha256=sha(args["native_policy"]), duration_us=480000000
    )
    landing = dict(start, kind="actual_auto_land_reference_consumption", actual_time_us=last)
    put(raw / "evaluation/joint-origin.json", start)
    put(raw / "evaluation/joint-landing.json", landing)
    sys.path.insert(0, str(ROOT / "simulation/sim/src"))
    from capstone_sim import joint_phase

    rows = list(joint_phase.reference_rows(case["operation_variation"], 480))
    reference = raw / "evaluation/joint-reference.csv"
    with reference.open("w") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    scenario = dict(
        name=case["scenario"],
        seed=case["seed"],
        paired_case_id=case["paired_case_id"],
        operation_variation=case["operation_variation"],
        wind_profile=case["operation_variation"]["wind_profile"],
        producer_lock_sha256=binding["runtime_sha256"],
        requested_duration_s=480,
        health_eligibility="confirmed_healthy",
        joint_phase_policy=dict(
            revision=native["revision"],
            policy_sha256=sha(args["native_policy"]),
            duration_us=480000000,
            reference_period_us=100000,
            reference_rows=4801,
            expected_weather_ticks=480001,
            reference_sha256=sha(reference),
            actual_start=start,
            actual_landing=landing,
        ),
    )
    put(raw / "evaluation/scenario.json", scenario)
    path = raw / "evaluation/joint-consumption.csv"
    with path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "source_us",
                "phase_us",
                "index",
                "px",
                "py",
                "pz",
                "vx",
                "vy",
                "vz",
                "ax",
                "ay",
                "az",
                "yaw",
            ]
        )
        for seq, phase in enumerate(range(0, 480000001, 25000), 1):
            index = phase // 100000
            writer.writerow([seq, first + phase, phase, index, *rows[index][1:]])
    put(raw / "evaluation/joint-check.json", joint_phase.verify(raw))
    selection = dict(fixture_only=True)
    put(raw / "evaluation/battery-selection.json", selection)
    put(raw / "evaluation/propulsion-selection.json", selection)
    put(raw / "reference/source_lock.json", dict(fixture_only=True))
    refresh(raw)
    interval = native_mission_interval(
        raw / "evaluation/joint-origin.json",
        policy_path=policy_path,
        native_policy_path=args["native_policy"],
        run_path=raw / "run.json",
    )
    rid = json_file(raw / "run.json")["run_id"]
    namespace = "main1674-20261107-01:"
    row = dict(
        case_id=namespace + case["case_id"],
        cohort=case["cohort"],
        purpose=case["purpose"],
        paired_case_id=case["paired_case_id"],
        group_id=namespace + case["group_id"],
        retry_family_id=namespace + case["retry_family_id"],
        mission_interval=interval,
        source_evidence=dict(battery_selection=selection, propulsion_selection=selection),
        lineage=[
            "run:" + rid,
            "pair:" + case["paired_case_id"],
            "group:" + namespace + case["group_id"],
            "retry:" + namespace + case["retry_family_id"],
        ],
    )
    values = np.tile(np.arange(19) * 0.001, (4801, 1))
    values[:, 17] = 16
    values[:, 18] = 12
    record, _ = export(
        tmp_path / "export",
        rid,
        case["role"],
        n=4801,
        row_override=row,
        producer_binding=binding,
        physical_override=values,
    )
    assessment = put(
        tmp_path / "assessment.json",
        dict(
            schema="drone.run-assessment/1",
            run_id=rid,
            contract_passed=True,
            physical_gate_passed=True,
            missing_data_passed=True,
            instance_isolation_passed=True,
            errors=[],
        ),
    )
    return dict(
        raw=raw,
        export=record["export"],
        case=case,
        producer=binding,
        bundle=bundle,
        assessment=assessment,
        row=record["row"],
    )


def test_public_full480_native_case_keeps_all_frames_windows_and_land_tail(full480, tmp_path):
    result = audit_case(
        full480["export"],
        producer=full480["producer"],
        protection=dict(schema="drone-ai.global-protection/1", entries=[]),
        policy_bundle=full480["bundle"],
        evidence_root=full480["raw"],
        assessment_ref=full480["assessment"],
        output=tmp_path / "audit.json",
    )
    assert result["passed"] is True and result["actual_data_verified"] is False
    assert result["frames"] == 4801 and result["windows"] == 470
    assert result["row"]["mission_interval"]["landing_tail_us"] == 24000
    assert result["native_check"]["reference_rows_seen"] == 4801
    assert result["native_check"]["max_consumption_gap_us"] == 25000
    assert sha(Path(result["receipt"]["path"])) == result["receipt"]["sha256"]
    assert "torch" not in sys.modules


@pytest.mark.parametrize(
    "damage", ["pilot_policy", "assessment", "csv", "scope_case", "origin_clock"]
)
def test_public_main_case_rejects_wrong_old_authority_or_source(full480, tmp_path, damage):
    if damage == "pilot_policy":
        p = Path(full480["bundle"]["paired_policy"])
        v = json_file(p)
        v.update(
            scope_task="SHORT_CYCLE_20261107",
            approved_for_pilot=True,
            approved_for_collection=False,
        )
        put(p, v)
    if damage == "assessment":
        p = Path(full480["assessment"]["path"])
        v = json_file(p)
        v["physical_gate_passed"] = False
        full480["assessment"] = put(p, v)
    if damage == "csv":
        with (full480["raw"] / "evaluation/joint-consumption.csv").open("a") as f:
            f.write("\n")
    if damage == "scope_case":
        p = full480["export"] / "online.json"
        v = json_file(p)
        v["run"]["case_id"] = "oldpilot:unapproved"
        put(p, v)
    if damage == "origin_clock":
        p = full480["raw"] / "evaluation/joint-origin.json"
        v = json_file(p)
        v["clock"] = "wall"
        put(p, v)
        refresh(full480["raw"])
    with pytest.raises(ValueError):
        audit_case(
            full480["export"],
            producer=full480["producer"],
            protection=dict(schema="drone-ai.global-protection/1", entries=[]),
            policy_bundle=full480["bundle"],
            evidence_root=full480["raw"],
            assessment_ref=full480["assessment"],
        )


def rejected(full480):
    raw = full480["raw"]
    path = raw / "evaluation/joint-consumption.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    rows.pop(1)
    for i, row in enumerate(rows, 1):
        row["sequence"] = str(i)
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    refresh(raw)
    p = Path(full480["assessment"]["path"])
    a = json_file(p)
    a.update(
        contract_passed=False,
        physical_gate_passed=False,
        errors=["Joint native consumption: Incomplete actual reference consumption coverage"],
    )
    full480["assessment"] = put(p, a)


def test_main_quality_rejection_is_independently_rechecked_and_preserves_physical_completed(
    full480, tmp_path
):
    rejected(full480)
    before = sha(full480["raw"] / "integrity.json")
    ref = publish_quality_rejection(
        full480["raw"],
        case_id=full480["case"]["case_id"],
        producer=full480["producer"],
        policy_bundle=full480["bundle"],
        assessment_ref=full480["assessment"],
        output=tmp_path / "rejection.json",
    )
    a = dict(
        index=1,
        run_id=full480["row"]["run_id"],
        parent_attempt_id=None,
        status="quality_rejected",
        physical_status="completed",
        quality_rejection=ref,
        seed=full480["case"]["seed"],
        lineage=full480["row"]["lineage"],
    )
    assert check_quality_rejection(ref, a, full480["producer"]) is True
    doc = json_file(ref["path"])
    assert doc["measured"]["max_gap_us"] == 50000 and doc["measured"]["reference_rows_seen"] == 4801
    assert (
        json_file(full480["raw"] / "run.json")["status"] == "completed"
        and sha(full480["raw"] / "integrity.json") == before
    )
    second = dict(
        a, index=2, run_id="synthetic-retry", parent_attempt_id=a["run_id"], status="completed"
    )
    second.pop("quality_rejection")
    second.pop("physical_status")
    ledger = put(
        tmp_path / "ledger.json",
        dict(
            schema="drone-ai.collection19.case-ledger/1",
            case_id=full480["case"]["case_id"],
            attempts=[a, second],
            accepted_run_id=second["run_id"],
        ),
    )
    _, counts = ag._ledger(ledger, full480["case"], full480["producer"], set(), full480["bundle"])
    assert counts == dict(
        physical_attempts=2, physical_failed_attempts=0, quality_rejected_attempts=1
    )


def test_main_quality_rejection_never_accepts_unknown_failure_or_valid_success(full480, tmp_path):
    with pytest.raises(ValueError):
        publish_quality_rejection(
            full480["raw"],
            case_id=full480["case"]["case_id"],
            producer=full480["producer"],
            policy_bundle=full480["bundle"],
            assessment_ref=full480["assessment"],
            output=tmp_path / "bad.json",
        )
    rejected(full480)
    p = Path(full480["assessment"]["path"])
    a = json_file(p)
    a["errors"] = ["source snapshot mismatch"]
    ref = put(p, a)
    with pytest.raises(ValueError):
        publish_quality_rejection(
            full480["raw"],
            case_id=full480["case"]["case_id"],
            producer=full480["producer"],
            policy_bundle=full480["bundle"],
            assessment_ref=ref,
            output=tmp_path / "unknown.json",
        )
