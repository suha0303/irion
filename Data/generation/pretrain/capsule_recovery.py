"""Finish a stopped streaming collection from its verified per-run capsules."""

import re
import shutil
import time
from pathlib import Path

from .common import (
    HOME,
    PACKAGE,
    RESOURCES,
    STORE,
    Budget,
    external,
    identity,
    read,
    sha,
    verify,
    write,
)


def regular(path):
    path = Path(path).absolute()
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Regular, unlinked recovery input required: " + str(path))
    return path


def capsule_row(source, store, label, row, pins):
    """Bind a capsule to the original freeze and independently completed CPU units."""
    rid = row["run_id"]
    if not re.fullmatch(r"[0-9a-f]{32}", rid):
        raise ValueError("Invalid capsule run identity")
    root = store / "streamed/pretrain" / source.name
    receipt_path = regular(root / "runs" / rid / "receipt.json")
    receipt = read(receipt_path)
    expected_row = dict(row, path=str(store / "consumer/pretrain" / source.name / label / rid))
    if (
        receipt.get("status") != "RUN_CAPSULE_VERIFIED"
        or receipt.get("task") != str(source)
        or receipt.get("run_id") != rid
        or receipt.get("row") != expected_row
        or receipt.get("raw_archive_sha256") != row["archive"]["sha256"]
    ):
        raise ValueError("Capsule receipt differs from original frozen run: " + rid)
    artifact = receipt["artifact"]
    archive = regular(root / "runs" / rid / "artifacts.tar.xz")
    if artifact["path"] != str(archive) or archive.stat().st_size != artifact["bytes"]:
        raise ValueError("Capsule archive missing/replaced: " + rid)
    if not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
        raise ValueError("Invalid capsule archive digest")
    members = artifact["members"]
    required = {
        "canonical/checkpoint.json",
        "canonical/summary.json",
        "canonical/frames.jsonl",
        "canonical/windows.jsonl",
        "canonical/exclusions.jsonl",
        "audit/numeric.npz",
        "audit/report.json",
        "audit/status.json",
        "identity/run.json",
        "identity/reference/source_lock.json",
        "identity/observations/metadata.json",
        "identity/scenario.json",
    }
    if set(members) != required:
        raise ValueError("Unexpected capsule member inventory")
    for member in members.values():
        if (
            type(member["bytes"]) is not int
            or member["bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", member["sha256"])
        ):
            raise ValueError("Invalid capsule member size/digest")
    for name, original in (
        ("run.json", "run.json"),
        ("reference/source_lock.json", "reference/source_lock.json"),
        ("observations/metadata.json", "observations/metadata.json"),
        ("scenario.json", "evaluation/scenario.json"),
    ):
        ref = row["archive"]["members"][original]
        if members["identity/" + name] != {key: ref[key] for key in ("bytes", "sha256")}:
            raise ValueError("Capsule identity differs from original archive")
    statuses = {}
    for action in ("preprocess", "audit"):
        path = regular(source / "units" / (label + "-" + action) / (rid + ".status.json"))
        status = read(path)
        if (
            status.get("status") != "COMPLETE"
            or status.get("action") != action
            or status.get("run_id") != rid
        ):
            raise ValueError("Completed matching preprocessing/audit required")
        statuses[action] = status
        pins[str(path)] = sha(path)
    audit_path = source / "units" / (label + "-audit") / (rid + ".status.json")
    if (
        receipt["audit_status_sha256"] != pins[str(audit_path)]
        or members["audit/status.json"]["sha256"] != pins[str(audit_path)]
        or members["audit/status.json"]["bytes"] != audit_path.stat().st_size
        or members["audit/numeric.npz"]["sha256"] != statuses["audit"]["result"]["numeric_sha256"]
        or members["audit/report.json"]["sha256"] != statuses["audit"]["result"]["report_sha256"]
    ):
        raise ValueError("Capsule audit does not match completed source audit")
    outputs = statuses["preprocess"]["result"]["output_hashes"]
    if set(outputs) != {"frames", "windows", "exclusions"} or any(
        members["canonical/" + name + ".jsonl"]["sha256"] != digest
        for name, digest in outputs.items()
    ):
        raise ValueError("Capsule canonical outputs differ from completed preprocessing")
    if row["role"] in {"normal_dev_eval", "anomaly_dev"}:
        event = regular(root / "evaluator-only" / (rid + ".jsonl"))
        expected = row["archive"]["members"]["evaluation/events.jsonl"]
        if (
            receipt.get("event_path") != str(event)
            or event.stat().st_size != expected["bytes"]
            or not 0 <= expected["bytes"] <= 10_000_000
            or sha(event) != expected["sha256"]
        ):
            raise ValueError("Development evaluator evidence changed")
        pins[str(event)] = expected["sha256"]
    elif receipt.get("event_path") is not None:
        raise ValueError("Unexpected evaluator truth on a fit run")
    pins[str(receipt_path)] = sha(receipt_path)
    return dict(
        expected_row,
        compact_source=dict(receipt=str(receipt_path), sha256=pins[str(receipt_path)]),
        raw_retained=False,
        raw_retirement_journal=str(receipt_path.parent / "cleanup.json"),
    ), sum(member["bytes"] for member in members.values())


def inputs(source):
    """Read stopped-task metadata; large archive hashes are checked during restore."""
    from x500_storage import guard, task_root

    from .composition import validate_roles
    from .flow import homogeneous_batches

    source = Path(source).absolute()
    regular(source / "config.json")
    if source.parent != HOME / "tasks":
        raise ValueError("Registered local source task required")
    verify({str(source / "config.json"): regular(source / "config.sha256").read_text().strip()})
    config = read(source / "config.json")
    state = read(regular(source / "operator/state.json"))
    cpu = read(regular(source / "operator/processing-state.json"))
    if (
        config.get("mode") != "collect"
        or config.get("storage_profile") != "stream_compact"
        or config.get("collection_profile") != "balanced300"
        or state.get("status") != "FAILED_STOPPED"
        or cpu.get("status") != "PROCESSING_COMPLETE"
        or cpu.get("stream_completed") != config.get("collection_run_count")
    ):
        raise ValueError("Stopped streaming task with all CPU capsules complete required")
    for record in (state, cpu):
        if (
            record.get("pid_start_ticks")
            and identity(record.get("pid", -1)) == record["pid_start_ticks"]
        ):
            raise ValueError("Source task process still exists")
    planned_path = regular(source / "planned-cases.json")
    if config.get("planned_cases") != str(planned_path):
        raise ValueError("Source planned case path mismatch")
    verify({str(planned_path): config["source_sha256"][str(planned_path)]})
    # Execution code may have the approved fix; preprocessing semantics must remain identical.
    immutable = {
        p: digest
        for p, digest in config["source_sha256"].items()
        if Path(p).is_relative_to(PACKAGE) or Path(p) == RESOURCES / "protected-inventory.json"
    }
    if (
        str(PACKAGE / "preprocessing.json") not in immutable
        or str(RESOURCES / "protected-inventory.json") not in immutable
    ):
        raise ValueError("Missing historical preprocessing/protection pins")
    verify(immutable)
    planned = read(planned_path)["sets"]
    if (
        len(planned) != config["collection_sets"]
        or sum(map(len, planned)) != config["collection_run_count"]
    ):
        raise ValueError("Source approved run count mismatch")
    store = guard(task_root(config), config.get("storage_policy"))
    pins = {
        str(source / name): sha(regular(source / name))
        for name in (
            "config.json",
            "config.sha256",
            "operator/state.json",
            "operator/processing-state.json",
            "planned-cases.json",
        )
    }
    protected = read(RESOURCES / "protected-inventory.json")["runs"]
    all_rows, parts, seen, restore_bytes = [], [], set(), 0
    for set_index, cases in enumerate(planned):
        rows, refs, template = [], [], None
        for index, batch in enumerate(homogeneous_batches(cases, config["flight_workers"])):
            label = f"stream-{set_index + 1:03}-{index + 1:03}"
            folder = source / label
            frozen_dir = folder / "fragment-freeze"
            paths = [folder / "manifest.json", folder / "execution/ledger.json"]
            paths += [
                frozen_dir / name
                for name in (
                    "frozen-fragment.json",
                    "cases.json",
                    "verified-fragment-ledger.json",
                    "required-revision.json",
                    "collection-plan.json",
                )
            ]
            pins.update({str(path): sha(regular(path)) for path in paths})
            ledger, fragment = read(paths[1]), read(paths[2])
            if (
                read(paths[0])["cases"] != batch
                or read(frozen_dir / "cases.json")["cases"] != batch
                or ledger.get("status") != "COLLECTED_PENDING_QUALITY_STOPPED"
                or ledger.get("phase") != "collection"
                or ledger.get("not_started")
                or ledger.get("failed_cases")
                or ledger.get("stop_reason")
                or set(ledger["cases"]) != {c["case_id"] for c in batch}
                or fragment.get("status") != "VERIFIED_FRAGMENT_PENDING_GLOBAL_SPLIT"
                or fragment["original_ledger_sha256"] != pins[str(paths[1])]
                or fragment["ledger_sha256"]
                != pins[str(frozen_dir / "verified-fragment-ledger.json")]
                or fragment["cases_sha256"] != pins[str(frozen_dir / "cases.json")]
                or fragment["plan_sha256"] != pins[str(frozen_dir / "collection-plan.json")]
                or fragment["revision_lock"] != read(frozen_dir / "required-revision.json")
                or fragment["revision_lock"]["preprocessing_sha256"]
                != immutable[str(PACKAGE / "preprocessing.json")]
                or fragment["protected_inventory_sha256"]
                != immutable[str(RESOURCES / "protected-inventory.json")]
                or len(fragment["runs"]) != len(batch)
            ):
                raise ValueError("Source fragment/ledger binding mismatch: " + label)
            by_case = {row["case_id"]: row for row in fragment["runs"]}
            if set(by_case) != {c["case_id"] for c in batch}:
                raise ValueError("Fragment case membership mismatch")
            for case in batch:
                record, row = ledger["cases"][case["case_id"]], by_case[case["case_id"]]
                exported = record["export"]
                if (
                    record.get("case") != case
                    or record.get("status") != "ARCHIVED_PENDING_QUALITY"
                    or row["run_id"] != record["run_id"]
                    or row["run_id"] in seen
                    or row["assessment"] != record["assessment"]
                    or any(row[key] != case[key] for key in case)
                    or row["archive"]
                    != {
                        "path": exported["archive"],
                        **{k: exported[k] for k in ("bytes", "sha256", "members")},
                    }
                ):
                    raise ValueError("Frozen run differs from completed collection")
                recovered, size = capsule_row(source, store, label, row, pins)
                seen.add(row["run_id"])
                rows.append(recovered)
                restore_bytes += size
            if template is None:
                template = fragment
            elif any(
                fragment[key] != template[key]
                for key in ("revision", "revision_lock", "protected_inventory_sha256")
            ):
                raise ValueError("Mixed fragment source/protection revisions")
            refs.append(dict(path=str(paths[2]), sha256=pins[str(paths[2])]))
        lookup = {row["case_id"]: row for row in rows}
        rows = [lookup[case["case_id"]] for case in cases]
        validate_roles(rows, cases, protected, config["collection_profile"])
        parts.append(
            dict(
                cases=cases,
                frozen=dict(
                    template,
                    status="DATA_FROZEN",
                    runs=rows,
                    collection_profile=config["collection_profile"],
                    stream_fragments=refs,
                ),
            )
        )
        all_rows.extend(rows)
    groups = [[case["case_id"] for case in cases] for cases in planned]
    validate_roles(
        all_rows,
        [case for cases in planned for case in cases],
        protected,
        config["collection_profile"],
        case_sets=groups,
    )
    if len(seen) != config["collection_run_count"]:
        raise ValueError("Recovered capsule count differs from approved collection")
    return dict(
        source=str(source),
        config=config,
        parts=parts,
        source_pins=pins,
        restore_bytes=restore_bytes,
    )


def create_recovery(source, name):
    from x500_storage import task_root

    from . import cli

    admitted = inputs(source)
    original = admitted["config"]
    args = cli.parser().parse_args(["init", "--name", name, "--mode", "collect"])
    args.storage_root = task_root(original)
    args.storage_profile = "compact"
    for key in (
        "workers",
        "io_workers",
        "audit_workers",
        "flight_workers",
        "export_workers",
        "collection_sets",
        "collection_profile",
        "seed_base",
        "use_available_ram",
        "use_swap",
    ):
        setattr(args, key, original[key])
    required = 3 * admitted["restore_bytes"] + 50_000_000_000
    if shutil.disk_usage(args.storage_root).free < required:
        raise ValueError("Insufficient scratch space for capsule recovery")
    task = cli.create(args)
    planned = task / "planned-cases.json"
    write(
        planned,
        dict(
            profile=original["collection_profile"],
            sets=[part["cases"] for part in admitted["parts"]],
            execution_started=False,
            scope="Preserved source cases; no collection authorized",
        ),
    )
    manifest = task / "recovery-inputs.json"
    write(
        manifest,
        dict(
            source_task=admitted["source"],
            source_pins=admitted["source_pins"],
            run_count=original["collection_run_count"],
            restore_bytes=admitted["restore_bytes"],
            source_capsules_preserved=True,
            archive_hashes_verified_during_restore=True,
        ),
    )
    config = read(task / "config.json")
    config.update(
        mode="recover_capsules",
        gates=False,
        pipeline_overlap=False,
        collection_authorized=False,
        training_authorized=False,
        unit_attempts_max=1,
        python_allocator="debug",
        continue_after_case_failure=False,
        recovery_source_task=admitted["source"],
        recovery_inputs=str(manifest),
        external_additional_bytes_max=required,
        scope="Restore existing verified capsules; assemble one train-only scaler, independently verify, package and stop. No collection/training/automatic retry; preserve source task and capsules.",
    )
    config["source_sha256"].update({str(planned): sha(planned), str(manifest): sha(manifest)})
    write(task / "config.json", config)
    (task / "config.sha256").write_text(sha(task / "config.json") + "\n")
    return task


def run(runner):
    from .collection import combine_frozen
    from .storage import verify_bundle
    from .streaming import restore_capsules, retire_owned, tree_stamp

    runner.update(planned_stage_count=8)
    runner.begin("capsule-admission", workers=1)
    expected = read(runner.config["recovery_inputs"])
    verify(expected["source_pins"])
    admitted = inputs(Path(runner.config["recovery_source_task"]))
    if (
        admitted["source_pins"] != expected["source_pins"]
        or admitted["restore_bytes"] != expected["restore_bytes"]
    ):
        raise ValueError("Recovery evidence changed since admission")
    parts = []
    for index, part in enumerate(admitted["parts"]):
        folder = runner.task / f"recovered-set-{index + 1:03}"
        write(folder / "cases.json", dict(cases=part["cases"]))
        frozen = dict(part["frozen"], cases_sha256=sha(folder / "cases.json"))
        parts.append((folder, frozen))
    runner.finish(
        "capsule-admission",
        dict(passed=True, runs=expected["run_count"], source_task_preserved=True),
    )
    runner.begin("combine", workers=1)
    frozen = combine_frozen(runner.task, parts)
    runner.finish("combine", dict(passed=True, runs=len(frozen["runs"])))
    prepared = external(STORE / "processed/pretrain" / runner.task.name / "prepared")
    prepared.mkdir(parents=True, exist_ok=False)
    write(prepared / "frozen-manifest.json", frozen)

    class RecoveryBudget(Budget):
        last_check = 0.0

        def check(self):
            super().check()
            if time.monotonic() - self.last_check > 1:
                runner.resources()
                self.last_check = time.monotonic()

    budget = RecoveryBudget()
    audits = restore_capsules(runner, frozen["runs"], prepared, budget)
    runner.finalize_prepared(
        frozen, prepared, "prepared", runner.config["workers"], preaudited=audits
    )
    runner.begin("recovery-retire", workers=1)
    portable = prepared.parent / "prepared-portable"
    receipt = read(portable / "receipt.json")
    for key in ("dataset", "evaluator_truth"):
        ref = receipt[key]
        path = portable / ref["path"]
        if sha(path) != ref["sha256"]:
            raise ValueError("New portable archive changed; preserve working data")
        verify_bundle(path, ref["members"], budget)
    # Only this fresh task's working copies may be retired. Source capsules remain intact.
    paths = [prepared / "dataset", prepared / "audit-work"]
    retire_owned(
        paths,
        {str(p): tree_stamp(p) for p in paths},
        portable / "working-cleanup.json",
        proof=receipt,
        budget=budget,
    )
    write(
        prepared / "handoff.json",
        dict(
            status="PORTABLE_DATASET_RESTORE_REQUIRED",
            portable=str(portable),
            training_eligible=False,
            training_authorized=False,
            archived_handoff_sha256=receipt["source_handoff_sha256"],
            source_capsules_preserved=True,
        ),
    )
    runner.finish("recovery-retire", dict(passed=True, source_capsules_preserved=True))
    runner.update(
        portable_dataset=str(portable),
        dataset_requires_restore=True,
        result=str(portable / "receipt.json"),
        raw_deleted=False,
    )
