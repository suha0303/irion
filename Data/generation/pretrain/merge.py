"""Merge frozen portable datasets offline, using one global train-only scaler."""

import argparse
import copy
import shutil
import time
from collections import Counter
from pathlib import Path

from .common import HOME, RESOURCES, STORE, Budget, external, read, sha, write

ROLES = {
    "normal_train",
    "normal_validation",
    "normal_calibration",
    "normal_dev_eval",
    "anomaly_dev",
}
COMMON = (
    "battery_profile",
    "px4_binary_sha256",
    "physics_plugin_sha256",
    "x500_profile_sha256",
    "preprocessing_sha256",
)


def merge_records(parts):
    from .composition import validate_roles

    if len(parts) < 2:
        raise ValueError("At least two verified source datasets required")
    first = parts[0]["frozen"]
    shared = {key: first["revision_lock"][key] for key in COMMON}
    rows, cases, groups = [], [], []
    for part in parts:
        frozen = part["frozen"]
        if (
            frozen.get("status") != "DATA_FROZEN"
            or frozen.get("collection_profile") != "balanced300"
            or frozen.get("revision") != first["revision"]
            or frozen["protected_inventory_sha256"] != first["protected_inventory_sha256"]
            or any(frozen["revision_lock"][key] != shared[key] for key in COMMON)
        ):
            raise ValueError("Incompatible source physics/preprocessing/protection revision")
        if any(row["role"] not in ROLES for row in frozen["runs"]):
            raise ValueError("Only training/validation/calibration/development roles may merge")
        rows.extend(copy.deepcopy(frozen["runs"]))
        cases.extend(copy.deepcopy(part["cases"]))
        groups.extend(copy.deepcopy(frozen["collection_set_case_ids"]))
    validate_roles(
        rows,
        cases,
        read(RESOURCES / "protected-inventory.json")["runs"],
        "balanced300",
        case_sets=groups,
    )
    return dict(
        status="DATA_FROZEN",
        revision=first["revision"],
        revision_lock=shared,
        producer_revision_scope="per_source_revision_locks; heterogeneous producer source hashes",
        source_revision_locks=[p["frozen"]["revision_lock"] for p in parts],
        protected_inventory_sha256=first["protected_inventory_sha256"],
        runs=rows,
        collection_profile="balanced300",
        collection_sets=len(groups),
        collection_set_case_ids=groups,
        scaler_scope="all_selected_normal_train",
        training_authorized=False,
        actual_GPU_execution=False,
    ), cases


def inputs(bundles):
    parts, pins = [], {}
    for bundle in bundles:
        from x500_storage import guard

        bundle = guard(bundle)
        receipt = read(bundle / "receipt.json")
        prepared = bundle.parent / "prepared"
        task = HOME / "tasks" / bundle.parent.name
        frozen = read(prepared / "frozen-manifest.json")
        quality = read(prepared / "quality.json")
        state = read(task / "operator/state.json")
        if (
            receipt.get("status") != "PORTABLE_DATASET_VERIFIED"
            or receipt.get("final_test_included") is not False
            or quality.get("passed") is not True
            or quality.get("train_only_scaler_verified") is not True
            or state.get("status") != "PRETRAIN_COMPACT_READY_STOPPED_RESTORE_REQUIRED"
        ):
            raise ValueError("Completed independently verified portable source required")
        for name in ("frozen-manifest.json", "quality.json"):
            p = prepared / name
            if sha(p) != receipt["dataset"]["members"]["evidence/" + name]["sha256"]:
                raise ValueError("Source evidence differs from portable seal")
        cases_path = task / "cases.json"
        if sha(cases_path) != frozen["cases_sha256"]:
            raise ValueError("Frozen source case manifest changed")
        for key in ("dataset", "evaluator_truth"):
            ref = receipt[key]
            p = bundle / ref["path"]
            if p.is_symlink() or p.stat().st_size != ref["bytes"]:
                raise ValueError("Source archive missing/replaced")
        paths = [
            bundle / "receipt.json",
            prepared / "frozen-manifest.json",
            prepared / "quality.json",
            cases_path,
            task / "config.json",
        ]
        for row in frozen["runs"]:
            if row["role"] in {"normal_dev_eval", "anomaly_dev"}:
                p = Path(row["compact_source"]["receipt"])
                if sha(p) != row["compact_source"]["sha256"]:
                    raise ValueError("Original development event receipt changed")
                event = Path(read(p)["event_path"])
                expected = row["archive"]["members"]["evaluation/events.jsonl"]
                if event.stat().st_size != expected["bytes"] or sha(event) != expected["sha256"]:
                    raise ValueError("Original development event changed")
                paths.extend([p, event])
        pins.update({str(p): sha(p) for p in paths})
        parts.append(
            dict(
                bundle=str(bundle), receipt=receipt, frozen=frozen, cases=read(cases_path)["cases"]
            )
        )
    frozen, cases = merge_records(parts)
    return parts, frozen, cases, pins


def create(bundles, name):
    from . import cli

    parts, frozen, cases, pins = inputs(bundles)
    required = sum(p["receipt"]["uncompressed_dataset_bytes"] for p in parts)
    if shutil.disk_usage(STORE).free < 3 * required + 50_000_000_000:
        raise ValueError("Insufficient external scratch space for bounded offline merge")
    args = cli.parser().parse_args(
        [
            "init",
            "--name",
            name,
            "--mode",
            "collect",
            "--storage-profile",
            "compact",
            "--collection-sets",
            str(frozen["collection_sets"]),
        ]
    )
    task = cli.create(args)
    config = read(task / "config.json")
    config.update(
        mode="merge_prepared",
        merge_bundles=[p["bundle"] for p in parts],
        collection_run_count=len(frozen["runs"]),
        gates=False,
        pipeline_overlap=False,
        collection_authorized=False,
        training_authorized=False,
        seed_base=None,
        seed_source="preserved_source_manifests",
        continue_after_case_failure=False,
        unit_attempts_max=3,
        external_additional_bytes_max=3 * required + 50_000_000_000,
        scope="Merge existing portable datasets; reuse canonical preprocessing, refit "
        "one train scaler, independently audit, pack and stop. No flight/training.",
    )
    by_id = {row["case_id"]: row for row in cases}
    planned = Path(config["planned_cases"])
    write(
        planned,
        dict(
            profile="balanced300",
            execution_started=False,
            scope="Existing source cases only; no collection authorized",
            sets=[[by_id[key] for key in group] for group in frozen["collection_set_case_ids"]],
        ),
    )
    config["source_sha256"][str(planned)] = sha(planned)
    config["source_sha256"].update(pins)
    write(task / "config.json", config)
    (task / "config.sha256").write_text(sha(task / "config.json") + "\n")
    write(
        task / "cases.json",
        dict(cases=cases, collection_set_case_ids=frozen["collection_set_case_ids"]),
    )
    write(
        task / "merge-admission.json",
        dict(
            runs=len(frozen["runs"]),
            by_role=dict(Counter(r["role"] for r in frozen["runs"])),
            source_bundles=config["merge_bundles"],
            restore_bytes=required,
            scratch_cap_bytes=config["external_additional_bytes_max"],
            source_pins=pins,
            compatibility="Same battery/PX4/physics/profile/preprocessing; producer revisions kept per source",
        ),
    )
    return task


def run(runner):
    from .storage import restore, verify_bundle
    from .streaming import retire_owned, tree_stamp

    runner.update(planned_stage_count=7)
    runner.begin("merge-inputs", workers=1)
    parts, frozen, cases, _ = inputs(runner.config["merge_bundles"])
    root = external(STORE / "processed/pretrain" / runner.task.name)
    root.mkdir(parents=True, exist_ok=False)
    prepared = root / "prepared"
    prepared.mkdir()
    write(
        prepared / "frozen-manifest.json",
        dict(
            frozen,
            cases_sha256=sha(runner.task / "cases.json"),
            source_freezes=[
                dict(
                    bundle=p["bundle"],
                    revision_lock=p["frozen"]["revision_lock"],
                    dataset_sha256=p["receipt"]["dataset"]["sha256"],
                    receipt_sha256=sha(Path(p["bundle"]) / "receipt.json"),
                    frozen_manifest_sha256=sha(
                        Path(p["bundle"]).parent / "prepared/frozen-manifest.json"
                    ),
                )
                for p in parts
            ],
        ),
    )
    frozen = read(prepared / "frozen-manifest.json")
    runner.finish("merge-inputs", dict(passed=True, runs=len(frozen["runs"])))
    total = sum(len(p["receipt"]["dataset"]["members"]) for p in parts)
    runner.begin("merge-restore", total=total, workers=1)
    (prepared / "canonical-work").mkdir()
    offset, last_update = 0, 0.0

    class RestoreBudget(Budget):
        last_check = 0.0

        def check(inner):
            super().check()
            if time.monotonic() - inner.last_check > 1:
                runner.resources()
                inner.last_check = time.monotonic()

    def progress(*, done, total):
        nonlocal last_update
        if time.monotonic() - last_update > 1 or done == total:
            runner.update(done=offset + done)
            last_update = time.monotonic()

    for index, part in enumerate(parts):
        target = root / ("source-restore-" + str(index + 1))
        restore(part["bundle"], target, budget=RestoreBudget(), progress=progress)
        archived = read(target / "evidence/frozen-manifest.json")
        if archived != part["frozen"]:
            raise ValueError("Restored source freeze differs from admission")
        for row in part["frozen"]["runs"]:
            (target / "dataset/canonical" / row["run_id"]).rename(
                prepared / "canonical-work" / row["run_id"]
            )
        offset += len(part["receipt"]["dataset"]["members"])
        runner.update(done=offset)
        runner.resources()
    runner.finish(
        "merge-restore",
        dict(
            passed=True, completed=total, source_datasets=len(parts), preprocessing_repeated=False
        ),
    )
    runner.finalize_prepared(frozen, prepared, "merged", 4)
    runner.begin("merge-retire", total=1, workers=1)
    portable = root / "prepared-portable"
    receipt = read(portable / "receipt.json")
    for key in ("dataset", "evaluator_truth"):
        ref = receipt[key]
        if sha(portable / ref["path"]) != ref["sha256"]:
            raise ValueError("New portable merge archive changed; preserve working data")
        verify_bundle(portable / ref["path"], ref["members"], Budget())
    paths = [prepared / "dataset", prepared / "audit-work"]
    paths += [root / ("source-restore-" + str(i + 1)) for i in range(len(parts))]
    allowed = {str(p): tree_stamp(p) for p in paths}
    retire_owned(paths, allowed, portable / "working-cleanup.json", proof=receipt, budget=Budget())
    write(
        prepared / "handoff.json",
        dict(
            status="PORTABLE_DATASET_RESTORE_REQUIRED",
            portable=str(portable),
            training_eligible=False,
            training_authorized=False,
            archived_handoff_sha256=receipt["source_handoff_sha256"],
        ),
    )
    runner.finish("merge-retire", dict(passed=True, source_bundles_preserved=True))
    runner.update(
        portable_dataset=str(portable),
        dataset_requires_restore=True,
        result=str(portable / "receipt.json"),
    )


def main():
    from .cli import start

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--bundle", action="append", required=True, type=Path)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    task = create(args.bundle, args.name)
    print(task)
    if args.start:
        start(task)


if __name__ == "__main__":
    main()
