"""Independent existing numeric audit plus R3 lineage and current CPU RAM gate."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from . import derived
from .control import (
    artifact,
    linked,
    package_check,
    read,
    sha,
    validate_roles,
    verify_inventory,
    write,
)


def check(root, prepared, freeze, protected, ram):
    package = package_check(root)
    frozen, exclusion = read(freeze), read(protected)
    if frozen.get("revision") != "R3_LONG15" or frozen.get("status") != "DATA_FROZEN":
        raise ValueError("R3 freeze required")
    if (
        frozen["plan_sha256"] != package["plan_sha256"]
        or frozen["revision_lock"] != read(root / "required-revision.json")
        or exclusion.get("complete") is not True
        or "R2_1GB" not in exclusion.get("includes_revisions", [])
    ):
        raise ValueError("source/plan/protected identity lock mismatch")
    validate_roles(frozen["runs"], read(root / "cases.json")["cases"], exclusion["runs"])
    reference = {"path": str(prepared), "seal_sha256": sha(prepared / "artifact.json")}
    _, info = artifact(reference, "prepare", sha(freeze))
    end, release = read(prepared / "exit.json"), read(prepared / "release.json")
    if (
        end.get("exit_code") != 0
        or release.get("passed") is not True
        or release.get("released") is not True
        or release.get("artifact_sha256") != reference["seal_sha256"]
        or info["package_sha256"] != sha(root / "PACKAGE.json")
    ):
        raise ValueError("prepare exit/release/package proof required")
    dataset = prepared / "dataset"
    canonical = read(dataset / "canonical-seal.json")
    expected = {
        r["run_id"] + "/" + name
        for r in frozen["runs"]
        for name in (
            "frames.jsonl",
            "windows.jsonl",
            "exclusions.jsonl",
            "summary.json",
            "checkpoint.json",
        )
    }
    if set(canonical) != expected:
        raise ValueError("all50 canonical runs required")
    verify_inventory(dataset / "canonical", canonical)
    for row in frozen["runs"]:
        summary = read(dataset / "canonical" / row["run_id"] / "summary.json")
        if (
            summary["config"] != read(root / "preprocessing.json")
            or summary["input_hashes"] != row["files"]
            or summary["contract_sha256"]
            != package["files"]["CODEX_REBUILD_PACKAGE_V4_2/00_SHARED_DATA_CONTRACT.md"]
        ):
            raise ValueError("canonical source/preprocessing/contract mismatch")
        for name, digest in summary["output_hashes"].items():
            if canonical.get(row["run_id"] + "/" + name + ".jsonl") != digest:
                raise ValueError("canonical summary output digest mismatch")
    result = derived.inspect_dataset(dataset, frozen)
    if ram is None:
        # Quality-only preparation cannot authorize a later GPU training stage.
        memory = {"memory_audit_passed": False, "status": "NOT_REQUESTED"}
    else:
        evidence = linked(ram)
        memory = derived.memory_assessment(result["dimensions"], evidence)
        age = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(evidence["observed_utc"].replace("Z", "+00:00"))
        ).total_seconds()
        if not -60 <= age <= 900:
            memory.update(memory_audit_passed=False, status="stale_CPU_RAM")
    verify_inventory(dataset / "canonical", canonical)
    verify_inventory(dataset, read(dataset / "manifest.json")["hashes"])
    artifact(reference, "prepare", sha(freeze))
    return dict(
        **result,
        memory=memory,
        passed=ram is None or memory["memory_audit_passed"],
        status=(
            "DERIVED_QUALITY_PASSED_PENDING_GPU_ADMISSION"
            if ram is None
            else "DERIVED_AUDIT_PASSED"
            if memory["memory_audit_passed"]
            else "DERIVED_AUDIT_PENDING_MEMORY"
        ),
        scope="quality_only" if ram is None else "quality_and_training_memory",
        lineage_passed=True,
        memory_audit_passed=memory["memory_audit_passed"],
        train_windows=result["dimensions"]["normal_train"]["windows"],
        validation_windows=result["dimensions"]["normal_validation"]["windows"],
        calibration_windows=result["dimensions"]["normal_calibration"]["windows"],
        frozen_manifest_sha256=sha(freeze),
        prepared_artifact_sha256=reference["seal_sha256"],
        package_sha256=sha(root / "PACKAGE.json"),
        ram_evidence=ram,
        retry=False,
        next_stage_authorized=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "prepared", "freeze", "protected", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ram", type=Path)
    mode.add_argument(
        "--quality-only",
        action="store_true",
        help="Local numeric/lineage/scaler checks only; no GPU admission",
    )
    args = parser.parse_args()
    from .control import external

    external(args.out, "metadata")
    if args.out.exists():
        raise ValueError("new audit report required")
    derived.fence(read(args.protected)["deny_paths"])
    import signal

    def expired(*_):
        raise TimeoutError("finite audit1800s exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 1800)
    try:
        result = check(
            args.root,
            args.prepared,
            args.freeze,
            args.protected,
            {"path": str(args.ram.resolve()), "sha256": sha(args.ram)} if args.ram else None,
        )
    except Exception as error:
        result = {"passed": False, "status": "DERIVED_AUDIT_INVALID", "error": str(error)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    write(args.out, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
