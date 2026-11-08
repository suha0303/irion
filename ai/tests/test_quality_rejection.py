"""Explicit three-run rejection fixtures, never flight or accepted-data
evidence.
"""

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest
from test_pilot45_admission import make_request

from learning.admission19 import SHORT
from learning.history19 import quality_rejection as qr
from learning.history19.pilot import json_file, plan_and_ledger, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return dict(path=str(path), sha256=sha(path))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    catalog = json_file(
        Path(__file__).parent / "fixtures/quality-rejection-catalog.json"
    )
    catalog["fixture_only"] = True
    for name in (
        "root_review",
        "original_physical_ledger",
        "official_rejection_index",
    ):
        catalog[name] = write(
            tmp_path / (name + ".json"), dict(fixture_only=True)
        )
    for rid, entry in catalog["entries"].items():
        raw = tmp_path / "raw" / rid
        # Bind generated metadata, so this regression needs no archived flight.
        operation = {"fixture_only": True, "run_id": rid}
        entry["case"]["operation_sha256"] = hashlib.sha256(
            json.dumps(
                operation, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        scenario = dict(
            seed=entry["case"]["seed"],
            operation_variation=operation,
            producer_lock_sha256=catalog["producer"]["runtime_sha256"],
            requested_duration_s=180,
            joint_phase_policy=dict(
                revision="joint-native-consumption-c2-v2",
                duration_us=180000000,
                policy_sha256=catalog["native_policy_sha256"],
            ),
        )
        refs = dict(
            run=write(
                raw / "run.json",
                dict(
                    run_id=rid,
                    status="completed",
                    origin="synthetic_unit_fixture",
                ),
            ),
            scenario=write(raw / "evaluation/scenario.json", scenario),
            native_origin=write(
                raw / "evaluation/joint-origin.json",
                dict(actual_start_us=60000000),
            ),
            native_landing=write(
                raw / "evaluation/joint-landing.json",
                dict(actual_time_us=240000000),
            ),
        )
        path = raw / "evaluation/joint-consumption.csv"
        gap = entry["expected_gap_us"]
        stamps = [0, gap, *range((gap // 10000 + 1) * 10000, 180000001, 10000)]
        with path.open("w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "source_us", "phase_us", "index"])
            for i, t in enumerate(stamps, 1):
                writer.writerow([i, 60000000 + t, t, t // 100000])
        refs["native_consumption"] = dict(path=str(path), sha256=sha(path))
        closing = {
            str(Path(v["path"]).relative_to(raw)): dict(
                sha256=v["sha256"], size=Path(v["path"]).stat().st_size
            )
            for v in refs.values()
        }
        refs["raw_integrity"] = write(raw / "integrity.json", closing)
        assessment = dict(
            schema="drone.run-assessment/1",
            run_id=rid,
            contract_passed=False,
            physical_gate_passed=False,
            missing_data_passed=True,
            instance_isolation_passed=True,
            errors=[qr.ERROR],
        )
        refs["official_failed_assessment"] = write(
            tmp_path / (rid + "-assessment.json"), assessment
        )
        refs["assessment_log"] = write(
            tmp_path / (rid + ".log"),
            dict(
                status="BLOCKED",
                reason="Producer assessment failed: " + json.dumps(assessment),
            ),
        )
        entry["refs"] = refs
    path = tmp_path / "catalog.json"
    write(path, catalog)
    monkeypatch.setattr(qr, "CATALOG", path)
    monkeypatch.setattr(qr, "CATALOG_SHA256", sha(path))
    return catalog


class Context:
    def __init__(self, docs=None):
        self.request = dict(fixture_only=True)
        self.hashes, self.checked = {}, []
        self.docs = docs or {}

    def doc(self, name, ref=None):
        if ref is None:
            return self.docs[name]
        path = Path(ref["path"])
        assert sha(path) == ref["sha256"]
        return json_file(path)


def attempt(entry, receipt):
    return dict(
        entry["original_attempt"],
        status="quality_rejected",
        quality_rejection=receipt,
    )


def verify(ctx, catalog, case, a):
    return qr.verify_attempt(
        ctx,
        case,
        a,
        expected_producer=catalog["producer"],
        native_policy_sha256=catalog["native_policy_sha256"],
    )


def test_three_explicit_rejections_publish_and_verify(evidence, tmp_path):
    for rid, entry in evidence["entries"].items():
        receipt = qr.publish(
            rid, tmp_path / (rid + "-receipt.json"), fixture_only=True
        )
        assert (
            verify(Context(), evidence, entry["case"], attempt(entry, receipt))
            is True
        )
        value = json_file(receipt["path"])
        assert (
            value["actual_data_verified"] is False
            and value["authorizes_flight"] is False
        )
        assert value["measured"]["reference_rows_seen"] == 1801
        assert value["measured"]["max_gap_us"] == entry["expected_gap_us"]
        with pytest.raises(FileExistsError):
            qr.publish(rid, receipt["path"], fixture_only=True)


@pytest.mark.parametrize(
    "damage",
    [
        "unapproved",
        "physical_status",
        "seed",
        "parent",
        "index",
        "lineage",
        "case",
        "receipt_limit",
        "receipt_ref",
        "csv",
        "official_assessment",
        "integrity",
        "catalog",
        "producer",
        "policy",
        "fake_failed",
        "actual_promotion",
    ],
)
def test_evidence_and_identity_cannot_be_weakened(evidence, tmp_path, damage):
    rid, entry = next(iter(evidence["entries"].items()))
    receipt = qr.publish(rid, tmp_path / "receipt.json", fixture_only=True)
    a, case, ctx = (
        attempt(entry, receipt),
        copy.deepcopy(entry["case"]),
        Context(),
    )
    producer, policy = (
        copy.deepcopy(evidence["producer"]),
        evidence["native_policy_sha256"],
    )
    if damage == "unapproved":
        a["run_id"] = "unapproved-completed-raw"
    if damage == "physical_status":
        a["physical_status"] = "failed"
    if damage == "seed":
        a["seed"] += 1
    if damage == "parent":
        a["parent_attempt_id"] = "other-parent"
    if damage == "index":
        a["index"] = 2
    if damage == "lineage":
        a["lineage"] = []
    if damage == "case":
        case["case_id"] += "-renamed"
    if damage in {"receipt_limit", "receipt_ref"}:
        doc = json_file(receipt["path"])
        if damage == "receipt_limit":
            doc["criterion_max_gap_us"] = 60000
        else:
            doc["refs"].pop("official_failed_assessment")
        a["quality_rejection"] = write(Path(receipt["path"]), doc)
    if damage in {"csv", "official_assessment", "integrity"}:
        key = dict(
            csv="native_consumption",
            official_assessment="official_failed_assessment",
            integrity="raw_integrity",
        )[damage]
        with Path(entry["refs"][key]["path"]).open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write("\n")
    if damage == "catalog":
        qr.CATALOG.write_text("{}", encoding="utf-8")
    if damage == "producer":
        producer["runtime_sha256"] = "0" * 64
    if damage == "policy":
        policy = "0" * 64
    if damage == "fake_failed":
        a["status"] = "failed"
    if damage == "actual_promotion":
        ctx.request["fixture_only"] = False
    with pytest.raises((ValueError, AssertionError)):
        qr.verify_attempt(
            ctx,
            case,
            a,
            expected_producer=producer,
            native_policy_sha256=policy,
        )


def campaign(evidence, tmp_path):
    request = json_file(make_request(tmp_path / "plan"))
    path = tmp_path / "plan"
    manifest = json_file(path / request["refs"]["pilot_manifest"]["path"])
    manifest.update(task_id=qr.TASK, purpose=SHORT, cohort=SHORT)
    for entry in evidence["entries"].values():
        case = entry["case"]
        index = next(
            i
            for i, c in enumerate(manifest["cases"])
            if c["case_kind"] == case["case_kind"]
            and (
                c["case_kind"] == "paired_normal"
                or (c["motion"], c["environment"])
                == (case["motion"], case["environment"])
            )
        )
        old_pair = manifest["cases"][index]["pair_id"]
        manifest["cases"][index] = case
        if case["case_kind"] == "paired_normal":
            paired = next(
                c
                for c in manifest["cases"]
                if c["case_kind"] == "anomaly" and c["pair_id"] == old_pair
            )
            paired.update(
                {
                    k: case[k]
                    for k in (
                        "seed",
                        "motion",
                        "environment",
                        "operation_sha256",
                        "pair_id",
                    )
                }
            )
    registry = dict(schema="drone-ai.global-protection/1", entries=[])
    rows, failed = [], 0
    by_case = {
        e["case"]["case_id"]: (rid, e) for rid, e in evidence["entries"].items()
    }
    for i, case in enumerate(manifest["cases"]):
        rid = "synthetic-valid-" + str(i)
        completed = dict(
            index=1,
            run_id=rid,
            status="completed",
            seed=case["seed"],
            lineage=case["lineage"],
            parent_attempt_id=None,
        )
        attempts = [completed]
        if case["case_id"] in by_case:
            prior_rid, e = by_case[case["case_id"]]
            receipt = qr.publish(
                prior_rid,
                tmp_path / (prior_rid + "-receipt.json"),
                fixture_only=True,
            )
            attempts.insert(0, attempt(e, receipt))
            completed.update(index=2, parent_attempt_id=prior_rid)
        elif failed < 17:
            old = "synthetic-physical-failure-" + str(i)
            attempts.insert(
                0,
                dict(
                    completed,
                    run_id=old,
                    status="failed",
                    failure_reason="preserved native failure",
                ),
            )
            completed.update(index=2, parent_attempt_id=old)
            failed += 1
        rows.append(
            dict(case_id=case["case_id"], status="completed", attempts=attempts)
        )
        registry["entries"].append(
            dict(
                identities=[
                    "case:" + case["case_id"],
                    *["run:" + a["run_id"] for a in attempts],
                ],
                forbidden_uses=["fit", "calibration", "final_test"],
            )
        )
    ctx = Context(
        dict(
            pilot_manifest=manifest,
            ledger=dict(
                schema="drone-ai.pilot45.ledger/1", task_id=qr.TASK, cases=rows
            ),
        )
    )
    return ctx, registry


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "ordinary_success_retry",
        "wrong_parent",
        "max4",
        "missing_exclusion",
        "wrong_cohort",
        "unresolved_quality",
        "no_callback",
    ],
)
def test_short_retry_counts_and_ordinary_success_guard(
    evidence, tmp_path, damage
):
    ctx, registry = campaign(evidence, tmp_path)
    rows = ctx.docs["ledger"]["cases"]
    rejected = next(
        r for r in rows if r["attempts"][0]["status"] == "quality_rejected"
    )
    if damage == "ordinary_success_retry":
        rejected["attempts"][0]["status"] = "completed"
    if damage == "wrong_parent":
        rejected["attempts"][1]["parent_attempt_id"] = None
    if damage == "max4":
        rejected["attempts"] *= 2
    if damage == "missing_exclusion":
        registry["entries"] = registry["entries"][1:]
    if damage == "unresolved_quality":
        rejected["attempts"].pop()
        # Case still claiming completed must be refused.
    callback = (
        None
        if damage == "no_callback"
        else lambda c, a: verify(ctx, evidence, c, a)
    )
    kwargs = dict(
        task_id=qr.TASK,
        cohort=SHORT if damage != "wrong_cohort" else "other",
        usage="short_cycle_input_check",
        quality_rejection_check=callback,
    )
    if damage is not None:
        with pytest.raises(ValueError):
            plan_and_ledger(ctx, registry, **kwargs)
    else:
        _, _, counts = plan_and_ledger(ctx, registry, **kwargs)
        assert (
            counts["planned_cases"] == 45 and counts["physical_attempts"] == 65
        )
        assert (
            counts["failed_attempts"]
            == counts["physical_failed_attempts"]
            == 17
        )
        assert counts["quality_rejected_attempts"] == 3
        assert counts["quality_rejected_cases"] == counts["pending_cases"] == 0
