"""Narrow short180 retry evidence for three physically completed, rejected C03 raws."""

import csv
import json
from pathlib import Path

from learning.admission19 import check_producer

from .pilot import ROOT, json_file, regular, sha

TASK = "SHORT_CYCLE_20261107"
CATALOG = (
    ROOT
    / "ai/reports/quality-rejection-20261107/attempt-01/approved-rejections-v2.json"
)
CATALOG_SHA256 = (
    "a2b35fe0b41a3d524d7172de1d30ff52109c207f113b539ee3e39819d6ed9e79"
)
APPROVED_RUNS = frozenset(
    (
        "12a8f7764990425f94cc2d55a4e0cdb0",
        "a92abf2e82ff4911a50d4cc966cf7d1e",
        "6046a91f7d2b429cad8a1b72652158ca",
    )
)
ERROR = (
    "Joint native consumption: Incomplete actual reference consumption coverage"
)


def _bound(ref):
    path = regular(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("Quality rejection evidence changed: " + str(path))
    return path


def _catalog(fixture_only):
    if sha(regular(CATALOG)) != CATALOG_SHA256:
        raise ValueError("Approved quality rejection catalogue changed")
    value = json_file(CATALOG)
    if (
        value.get("schema")
        != "drone-ai.short-cycle.quality-rejection-catalog/1"
        or value.get("task_id") != TASK
        or value.get("fixture_only") is not fixture_only
        or set(value.get("entries", {})) != APPROVED_RUNS
        or set(value.get("approved_run_ids", [])) != APPROVED_RUNS
        or value.get("max_gap_us") != 40000
        or value.get("max_physical_attempts") != 3
    ):
        raise ValueError(
            "Only the three explicit short180 rejections are authorized"
        )
    for key in (
        "root_review",
        "original_physical_ledger",
        "official_rejection_index",
    ):
        _bound(value[key])
    return value


def _evidence(catalog, run_id, fixture_only):
    entry = catalog["entries"][run_id]
    paths = {name: _bound(ref) for name, ref in entry["refs"].items()}
    run_path = paths["run"]
    raw = run_path.parent
    run, scenario = json_file(run_path), json_file(paths["scenario"])
    integrity = json_file(paths["raw_integrity"])
    expected_origin = "synthetic_unit_fixture" if fixture_only else "px4_sitl"
    if (
        run.get("run_id") != run_id
        or run.get("status") != "completed"
        or run.get("origin") != expected_origin
    ):
        raise ValueError(
            "Original physical completed raw identity must remain unchanged"
        )
    members = {
        "run": "run.json",
        "scenario": "evaluation/scenario.json",
        "native_consumption": "evaluation/joint-consumption.csv",
        "native_origin": "evaluation/joint-origin.json",
        "native_landing": "evaluation/joint-landing.json",
    }
    for name, member in members.items():
        if paths[name] != raw / member or integrity[member] != dict(
            sha256=entry["refs"][name]["sha256"],
            size=paths[name].stat().st_size,
        ):
            raise ValueError(
                "Rejected raw differs from original closing integrity"
            )
    if paths["raw_integrity"] != raw / "integrity.json":
        raise ValueError("Original raw integrity location required")
    case = entry["case"]
    selected = scenario.get("joint_phase_policy", {})
    operation = json.dumps(
        scenario.get("operation_variation"),
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    if (
        scenario.get("seed") != case["seed"]
        or hashlib.sha256(operation.encode()).hexdigest()
        != case["operation_sha256"]
        or scenario.get("producer_lock_sha256")
        != catalog["producer"]["runtime_sha256"]
        or scenario.get("requested_duration_s") != 180
        or selected.get("revision") != "joint-native-consumption-c2-v2"
        or selected.get("duration_us") != 180000000
        or selected.get("policy_sha256") != catalog["native_policy_sha256"]
    ):
        raise ValueError(
            "Rejected original seed/operation/producer/short180 policy changed"
        )
    assessment = json_file(paths["official_failed_assessment"])
    blocked = json_file(paths["assessment_log"])
    if blocked.get("status") != "BLOCKED" or not blocked.get(
        "reason", ""
    ).startswith("Producer assessment failed: "):
        raise ValueError("Original official assessment failure log required")
    original_assessment = json.loads(
        blocked["reason"].removeprefix("Producer assessment failed: ")
    )
    if (
        assessment != original_assessment
        or assessment.get("schema") != "drone.run-assessment/1"
        or assessment.get("run_id") != run_id
        or assessment.get("contract_passed") is not False
        or assessment.get("physical_gate_passed") is not False
        or assessment.get("missing_data_passed") is not True
        or assessment.get("instance_isolation_passed") is not True
        or assessment.get("errors") != [ERROR]
    ):
        raise ValueError(
            "Hash-bound official native-gap full-gate rejection required"
        )
    start = json_file(paths["native_origin"])["actual_start_us"]
    end = json_file(paths["native_landing"])["actual_time_us"]
    count, max_gap, over, previous, payload = 0, 0, 0, None, None
    indices = set()
    with paths["native_consumption"].open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stamp, index = int(row["source_us"]), int(row["index"])
            current = {k: v for k, v in row.items() if k != "sequence"}
            if (
                int(row["sequence"]) != count + 1
                or stamp - start != int(row["phase_us"])
                or not 0 <= index <= 1800
                or index != min(1800, (stamp - start) // 100000)
                or (previous is None and stamp != start)
                or (
                    previous is not None
                    and (
                        stamp < previous
                        or (stamp == previous and current != payload)
                    )
                )
            ):
                raise ValueError(
                    "Rejected CSV sequence/clock/coverage/conflict differs"
                )
            gap = 0 if previous is None else stamp - previous
            max_gap, over = max(max_gap, gap), over + (gap > 40000)
            previous, payload = stamp, current
            indices.add(index)
            count += 1
    if (
        indices != set(range(1801))
        or previous is None
        or not start + 180000000 <= previous <= end <= start + 180100000
        or max_gap != entry["expected_gap_us"]
        or max_gap <= 40000
        or over != 1
    ):
        raise ValueError(
            "Actual one-gap rejection and complete1801 reference coverage required"
        )
    return dict(
        max_gap_us=max_gap,
        gaps_over_limit=over,
        reference_rows_seen=len(indices),
        consumed_samples=count,
    )


def publish(run_id, output, *, fixture_only=False):
    """Write new evidence only; never change physical metadata, data, or execution state."""
    if run_id not in APPROVED_RUNS:
        raise ValueError(
            "Run is not one of the three authorized quality rejections"
        )
    catalog = _catalog(fixture_only)
    measured = _evidence(catalog, run_id, fixture_only)
    entry = catalog["entries"][run_id]
    value = dict(
        schema="drone-ai.short-cycle.quality-rejection/1",
        task_id=TASK,
        run_id=run_id,
        case_id=entry["case"]["case_id"],
        physical_status="completed",
        data_status="quality_rejected",
        reason="native_consumption_gap_exceeded",
        criterion_max_gap_us=40000,
        original_attempt=entry["original_attempt"],
        producer=catalog["producer"],
        native_policy_sha256=catalog["native_policy_sha256"],
        catalog=dict(path=str(CATALOG), sha256=CATALOG_SHA256),
        refs=entry["refs"],
        measured=measured,
        fixture_only=fixture_only,
        actual_data_verified=not fixture_only,
        authorizes_flight=False,
        eligible_for_fit=False,
    )
    output = Path(output).absolute()
    if ".." in output.parts or any(
        p.is_symlink() for p in (output, *output.parents)
    ):
        raise ValueError("Separate new regular rejection output required")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    return dict(path=str(output), sha256=sha(output))


def verify_attempt(
    ctx, case, attempt, *, expected_producer, native_policy_sha256
):
    """Return True for verified quality_rejected, otherwise preserve the old guard."""
    rejected = attempt.get("status") == "quality_rejected"
    if not rejected:
        if attempt.get("quality_rejection") is not None or (
            attempt["run_id"] in APPROVED_RUNS
            and attempt.get("status") in {"failed", "stopped"}
        ):
            raise ValueError(
                "Physical COMPLETED cannot be disguised as a failed attempt"
            )
        return False
    if (
        attempt["run_id"] not in APPROVED_RUNS
        or attempt.get("physical_status") != "completed"
    ):
        raise ValueError(
            "Only explicit completed quality rejections can be retried"
        )
    fixture = ctx.request.get("fixture_only") is True
    catalog = _catalog(fixture)
    check_producer(catalog["producer"], expected_producer)
    if catalog["native_policy_sha256"] != native_policy_sha256:
        raise ValueError(
            "Quality rejection cannot be moved to a different native policy"
        )
    entry = catalog["entries"][attempt["run_id"]]
    if case != entry["case"] or any(
        attempt.get(k) != v for k, v in entry["original_attempt"].items()
    ):
        raise ValueError(
            "Original quality-rejected case/index/seed/lineage/parent changed"
        )
    receipt = ctx.doc(
        "quality_rejection:" + attempt["run_id"],
        attempt.get("quality_rejection"),
    )
    measured = _evidence(catalog, attempt["run_id"], fixture)
    if (
        receipt.get("schema") != "drone-ai.short-cycle.quality-rejection/1"
        or receipt.get("task_id") != TASK
        or receipt.get("run_id") != attempt["run_id"]
        or receipt.get("case_id") != case["case_id"]
        or receipt.get("physical_status") != "completed"
        or receipt.get("data_status") != "quality_rejected"
        or receipt.get("reason") != "native_consumption_gap_exceeded"
        or receipt.get("criterion_max_gap_us") != 40000
        or receipt.get("original_attempt") != entry["original_attempt"]
        or receipt.get("producer") != catalog["producer"]
        or receipt.get("native_policy_sha256") != native_policy_sha256
        or receipt.get("refs") != entry["refs"]
        or receipt.get("catalog")
        != dict(path=str(CATALOG), sha256=CATALOG_SHA256)
        or receipt.get("measured") != measured
        or receipt.get("fixture_only") is not fixture
        or receipt.get("actual_data_verified") is not (not fixture)
        or receipt.get("authorizes_flight") is not False
        or receipt.get("eligible_for_fit") is not False
    ):
        raise ValueError(
            "Explicit hash-bound quality rejection receipt differs"
        )
    ctx.hashes["quality_rejection_catalog"] = CATALOG_SHA256
    return True
