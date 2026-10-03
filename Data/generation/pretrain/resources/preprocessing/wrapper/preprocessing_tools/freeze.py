"""Freeze a completed R3 collection; read identity metadata, never event truth."""

import argparse
import hashlib
import json
import signal
import tarfile
from pathlib import Path, PurePosixPath

from .control import (
    RAW_FILES,
    STORAGE,
    read,
    regular,
    sha,
    validate_cases,
    validate_plan,
    validate_roles,
)


def identity(archive):
    path = regular(archive["path"])
    if not path.is_relative_to(STORAGE / "archives"):
        raise ValueError("R3 archive path required")
    if path.stat().st_size != archive["bytes"] or sha(path) != archive["sha256"]:
        raise ValueError("archive hash/size differs from completed ledger")
    wanted = {"run.json", "evaluation/scenario.json", "reference/source_lock.json"}
    result, seen = {}, set()
    with tarfile.open(path, mode="r|gz") as stream:
        for member in stream:
            key = member.name
            name = PurePosixPath(key)
            if name.is_absolute() or ".." in name.parts or "\\" in key or key in seen:
                raise ValueError("unsafe/duplicate member in completed archive")
            seen.add(key)
            if key not in wanted:
                continue
            expected = archive["members"][key]
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
                or member.sparse
                or not 0 <= member.size <= 32_000_000
                or member.size != expected["bytes"]
            ):
                raise ValueError("identity member must be bounded regular JSON")
            with stream.extractfile(member) as data:
                body = data.read(32_000_001)
            if len(body) != member.size or hashlib.sha256(body).hexdigest() != expected["sha256"]:
                raise ValueError("identity member digest mismatch")
            result[key] = json.loads(body)
    if set(result) != wanted or sha(path) != archive["sha256"]:
        raise ValueError("missing identity or changed archive")
    return result


def make_freeze(plan_path, cases_path, ledger_path, revision_path, protected_path):
    plan, cases_record, ledger, revision, protected = map(
        read, (plan_path, cases_path, ledger_path, revision_path, protected_path)
    )
    cases = cases_record["cases"]
    validate_plan(plan)
    validate_cases(cases)
    if (
        plan.get("revision") != "R3_LONG15"
        or cases_record.get("revision") != "R3_LONG15"
        or plan["cases_sha256"] != sha(cases_path)
        or revision.get("battery_profile") != "VTC6_4S4P"
        or protected.get("complete") is not True
        or not protected.get("runs")
        or not protected.get("deny_paths")
        or "R2_1GB" not in protected.get("includes_revisions", [])
    ):
        raise ValueError("R3 plan/cases/new producer/protected revision lock required")
    if (
        ledger.get("status") != "COLLECTED_PENDING_QUALITY_STOPPED"
        or ledger.get("phase") != "collection"
        or ledger.get("not_started")
        or ledger.get("stop_reason")
        or set(ledger["cases"]) != {c["case_id"] for c in cases}
    ):
        raise ValueError(
            "all50 collection cases must finish; failed/unstarted cases cannot be dropped"
        )
    runs = []
    for case in cases:
        record = ledger["cases"][case["case_id"]]
        if record.get("case") != case or record.get("status") != "ARCHIVED_PENDING_QUALITY":
            raise ValueError("case identity/status changed or source drift")
        assessment = record["assessment"]
        if assessment.get("run_id") != record["run_id"] or any(
            assessment.get(k) is not True
            for k in (
                "contract_passed",
                "instance_isolation_passed",
                "missing_data_passed",
                "physical_gate_passed",
            )
        ):
            raise ValueError(
                "producer contract/isolation/data/physical gate must pass for every case"
            )
        archive = record["export"]
        reference = dict(
            path=archive["archive"],
            sha256=archive["sha256"],
            bytes=archive["bytes"],
            members=archive["members"],
        )
        values = identity(reference)
        run, scenario, source = (
            values[n]
            for n in ("run.json", "evaluation/scenario.json", "reference/source_lock.json")
        )
        if (
            run.get("run_id") != record["run_id"]
            or run.get("origin") != "px4_sitl"
            or run.get("status") != "completed"
            or run.get("parent_attempt_id")
            or run.get("previous_attempts")
            or run.get("continuation_id")
            or scenario.get("parent_attempt_id")
            or scenario.get("previous_attempts")
            or scenario.get("continuation_id")
            or scenario.get("name") != case["scenario"]
            or scenario.get("seed") != case["seed"]
            or scenario.get("continuity") != "independent_reset"
            or scenario.get("requested_duration_s") != case["mission_seconds"]
        ):
            raise ValueError("archive run/scenario/duration/origin identity mismatch")
        if (
            case["role"] != "anomaly_dev"
            and scenario.get("health_eligibility") != "confirmed_healthy"
        ):
            raise ValueError("normal health eligibility not confirmed")
        members = reference["members"]
        if members["reference/source_lock.json"]["sha256"] != revision["producer_lock_sha256"]:
            raise ValueError("new producer source-lock digest differs")
        source_digests = set(source.get("local_sources", {}).values()) | set(
            source.get("runtime_binary_sha256", {}).values()
        )
        for field in ("px4_binary_sha256", "physics_plugin_sha256", "x500_profile_sha256"):
            if revision[field] not in source_digests:
                raise ValueError(
                    "new firmware/physics/profile digest absent from source-lock: " + field
                )
        files = {name: members[name]["sha256"] for name in RAW_FILES}
        if any(members[n].get("kind") != "file" for n in RAW_FILES):
            raise ValueError("six regular consumer members required")
        runs.append(
            dict(
                case,
                run_id=run["run_id"],
                path=str(STORAGE / "consumer" / run["run_id"]),
                raw_flight_id=run.get("raw_flight_id", run["run_id"]),
                raw_sha256=files["observations/streams.jsonl"],
                paired_case_id=scenario.get("paired_case_id") or case["group_id"],
                producer_paired_case_id=scenario.get("paired_case_id"),
                previous_attempts=[],
                continuity="independent_reset",
                origin="px4_sitl",
                status="completed",
                health_eligibility="abnormal"
                if case["role"] == "anomaly_dev"
                else "confirmed_healthy",
                scenario_parameters_snapshot=scenario,
                archive=reference,
                files=files,
                revision_lock=revision,
                assessment=assessment,
            )
        )
    validate_roles(runs, cases, protected["runs"])
    return dict(
        status="DATA_FROZEN",
        revision="R3_LONG15",
        plan_sha256=sha(plan_path),
        revision_lock=revision,
        runs=runs,
        ledger_sha256=sha(ledger_path),
        protected_inventory_sha256=sha(protected_path),
        cases_sha256=sha(cases_path),
        derived_quality_verified=False,
        actual_GPU_execution=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "cases", "ledger", "revision", "protected", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or any(p.is_symlink() for p in (args.out, *args.out.parents)):
        raise ValueError("new nonsymlink freeze output required")

    def expired(*_):
        raise TimeoutError("finite freeze verification7200s exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 7200)
    try:
        result = make_freeze(args.plan, args.cases, args.ledger, args.revision, args.protected)
        with args.out.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print("DATA_FROZEN; derived/model checks still required", sha(args.out))


if __name__ == "__main__":
    main()
