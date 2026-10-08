"""Process exactly one failed run, then assemble the complete existing dataset."""

import shutil
from pathlib import Path

from .common import HOME, PACKAGE, ROOT, STORE, external, identity, read, sha, verify, write


def inputs(source):
    source = Path(source).resolve()
    if source.parent != (HOME / "tasks").resolve():
        raise ValueError("Registered task required")
    config = read(source / "config.json")
    verify({str(source / "config.json"): (source / "config.sha256").read_text().strip()})
    state = read(source / "operator/state.json")
    if (
        config["mode"] != "resume_preprocess"
        or state.get("status") != "FAILED_STOPPED"
        or state.get("stage") != "prepared-preprocess"
    ):
        raise ValueError("Stopped preprocessing failure required")
    if state.get("pid_start_ticks") and identity(state["pid"]) == state["pid_start_ticks"]:
        raise ValueError("Previous worker still running")
    # Workflow changes are allowed; preprocessing implementation/package must match.
    verify(
        {
            p: h
            for p, h in config["source_sha256"].items()
            if Path(p).is_relative_to(PACKAGE) or p == str(ROOT / "Data/generation/pretrain/worker.py")
        }
    )
    from x500_storage import guard, task_root

    config = read(source / "config.json")
    root = guard(
        task_root(config) / "processed/pretrain" / source.name / "prepared",
        config.get("storage_policy"),
    )
    frozen = read(root / "frozen-manifest.json")
    result = read(source / "units/prepared-preprocess/result.json")
    extracted = read(source / "units/prepared-extract/result.json")
    rows = {r["run_id"]: r for r in frozen["runs"]}
    completed = {r["run_id"]: r for r in result["unit_records"]}
    failed = result["failure_records"]
    if (
        len(rows) != len(frozen["runs"])
        or len(failed) != 1
        or result["failed"] != 1
        or result["completed"] != len(completed)
        or extracted["completed"] != len(rows)
        or extracted.get("failure")
        or extracted.get("failed")
        or len(result["unit_records"]) != len(completed)
        or len(completed) != len(rows) - 1
        or failed[0]["run_id"] in completed
        or set(rows) != set(completed) | {failed[0]["run_id"]}
    ):
        raise ValueError("Exactly one failed run and complete extraction required")
    evidence = [
        source / n
        for n in (
            "config.json",
            "config.sha256",
            "operator/state.json",
            "units/prepared-preprocess/result.json",
            "units/prepared-extract/result.json",
            "cases.json",
        )
    ]
    evidence.append(root / "frozen-manifest.json")
    for rid, record in completed.items():
        summary_path = root / "canonical-work" / rid / "summary.json"
        summary = read(summary_path)
        if (
            record["status"] != "COMPLETE"
            or record["action"] != "preprocess"
            or summary["output_hashes"] != record["result"]["output_hashes"]
            or set(summary["output_hashes"]) != {"frames", "windows", "exclusions"}
            or summary["input_hashes"] != rows[rid]["files"]
            or summary["config"] != read(PACKAGE / "preprocessing.json")
            or not summary["closed"]
        ):
            raise ValueError("Successful result identity/config mismatch: " + rid)
        evidence.append(summary_path)
        for name in ("summary.json", "frames.jsonl", "windows.jsonl", "exclusions.jsonl"):
            p = summary_path.parent / name
            if not p.is_file() or any(q.is_symlink() for q in (p, *p.parents)):
                raise ValueError("Missing/linked completed result: " + str(p))
    return root, frozen, completed, failed[0]["run_id"], evidence


def create_partial(source, name):
    from . import cli

    source = Path(source).resolve()
    root, frozen, completed, failed, evidence = inputs(source)
    original = read(source / "config.json")
    args = cli.parser().parse_args(["init", "--name", name, "--mode", "collect"])
    from x500_storage import task_root

    args.storage_root = task_root(original)
    args.storage_profile = original.get("storage_profile", "standard")
    for key in (
        "workers",
        "io_workers",
        "audit_workers",
        "flight_workers",
        "export_workers",
        "collection_sets",
        "collection_profile",
        "seed_base",
    ):
        setattr(args, key, original[key])
    task = cli.create(args)
    config = read(task / "config.json")
    config.update(
        mode="finish_preprocess",
        partial_source_task=str(source),
        partial_failed_run=failed,
        reused_run_count=len(completed),
        collection_run_count=len(frozen["runs"]),
        gates=False,
        collection_authorized=False,
        training_authorized=False,
        scope="One failed run with maximum3 attempts; reuse successful results; assemble/audit all runs; stop. No extraction, collection or training.",
    )
    config["source_sha256"].update({str(p): sha(p) for p in evidence})
    write(task / "config.json", config)
    (task / "config.sha256").write_text(sha(task / "config.json") + "\n")
    return task


def finish_partial(runner):
    runner.begin("validate-existing-results", workers=1)
    source = Path(runner.config["partial_source_task"])
    old, frozen, completed, failed, _ = inputs(source)
    if (
        failed != runner.config["partial_failed_run"]
        or len(completed) != runner.config["reused_run_count"]
    ):
        raise ValueError("Partial recovery identity changed")
    from .common import RESOURCES
    from .composition import validate_roles

    validate_roles(
        frozen["runs"],
        read(source / "cases.json")["cases"],
        read(RESOURCES / "protected-inventory.json")["runs"],
        frozen["collection_profile"],
        case_sets=frozen.get("collection_set_case_ids"),
    )
    root = external(STORE / "processed/pretrain" / runner.task.name / "prepared")
    root.mkdir(parents=True, exist_ok=False)
    write(root / "frozen-manifest.json", frozen)
    shutil.copy2(source / "cases.json", runner.task / "cases.json")
    runner.finish(
        "validate-existing-results", dict(passed=True, reused=len(completed), retry_run=failed)
    )
    row = next(r for r in frozen["runs"] if r["run_id"] == failed)
    # Existing extracted inputs are hash-checked by the worker; no extraction or freeze repeats.
    runner.batch("prepared-preprocess-one", "preprocess", [row], 1, root / "canonical-work")
    sources = {rid: old / "canonical-work" / rid for rid in completed}
    write(
        root / "reuse-evidence.json",
        dict(
            source_task=str(source),
            reused_runs=list(completed),
            processed_runs=[failed],
            extraction_repeated=False,
            training_authorized=False,
        ),
    )
    result = runner.finalize_prepared(
        frozen, root, "prepared", runner.config["workers"], canonical_sources=sources
    )
    runner.update(result=str(result / "handoff.json"))
