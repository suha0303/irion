"""Recover one failed audit, reusing completed preprocessing and audit outputs."""

from pathlib import Path

from .common import (
    ADMISSION,
    HOME,
    PACKAGE,
    ROOT,
    STORE,
    external,
    identity,
    read,
    sha,
    verify,
    write,
)


def inputs(source):
    source = Path(source).resolve()
    if source.parent != (HOME / "tasks").resolve():
        raise ValueError("Registered source task required")
    config = read(source / "config.json")
    verify({str(source / "config.json"): (source / "config.sha256").read_text().strip()})
    state = read(source / "operator/state.json")
    if (
        config["mode"] not in {"resume_preprocess", "finish_preprocess"}
        or state.get("status") != "FAILED_STOPPED"
        or state.get("stage") != "prepared-audit"
        or "prepared-assemble" not in state.get("completed", [])
    ):
        raise ValueError("Stopped audit with completed assembly required")
    if state.get("pid_start_ticks") and identity(state["pid"]) == state["pid_start_ticks"]:
        raise ValueError("Previous process still exists")
    verify(
        {
            p: h
            for p, h in config["source_sha256"].items()
            if Path(p).is_relative_to(PACKAGE)
            or p in {str(ROOT / "Data/generation/pretrain/worker.py"), str(ADMISSION / "derived.py")}
        }
    )
    from x500_storage import guard, task_root

    config = read(source / "config.json")
    root = guard(
        task_root(config) / "processed/pretrain" / source.name / "prepared",
        config.get("storage_policy"),
    )
    frozen = read(root / "frozen-manifest.json")
    dataset = root / "dataset"
    manifest = read(dataset / "manifest.json")
    result = read(source / "units/prepared-audit/result.json")
    rows = {r["run_id"]: r for r in frozen["runs"]}
    completed = {r["run_id"]: r for r in result["unit_records"]}
    failed = result["failure_records"]
    if (
        len(failed) != 1
        or result["failed"] != 1
        or len(rows) != len(frozen["runs"])
        or len(completed) != len(result["unit_records"])
        or result["completed"] != len(completed)
        or len(completed) != len(rows) - 1
        or failed[0]["run_id"] in completed
        or set(rows) != set(completed) | {failed[0]["run_id"]}
        or manifest["runs"] != frozen["runs"]
    ):
        raise ValueError("Exactly one audit failure and complete dataset required")
    evidence = [
        source / n
        for n in (
            "config.json",
            "config.sha256",
            "operator/state.json",
            "units/prepared-audit/result.json",
        )
    ]
    evidence += [root / "frozen-manifest.json", dataset / "manifest.json"]
    summaries = {r["run_id"]: r["summary"] for r in manifest["audit"]}
    if len(summaries) != len(manifest["audit"]) or set(summaries) != set(rows):
        raise ValueError("Assembly summary membership mismatch")
    for rid, row in rows.items():
        summary_path = dataset / "canonical" / rid / "summary.json"
        summary = read(summary_path)
        if (
            summary != summaries[rid]
            or summary["input_hashes"] != row["files"]
            or not summary["closed"]
            or summary["config"] != read(PACKAGE / "preprocessing.json")
            or set(summary["output_hashes"]) != {"frames", "windows", "exclusions"}
        ):
            raise ValueError("Canonical summary/config mismatch: " + rid)
        evidence.append(summary_path)
        for name in ("summary.json", "frames.jsonl", "windows.jsonl", "exclusions.jsonl"):
            p = summary_path.parent / name
            if not p.is_file() or any(q.is_symlink() for q in (p, *p.parents)):
                raise ValueError("Missing/linked canonical file: " + str(p))
        if rid in completed:
            status_path = source / "units/prepared-audit" / (rid + ".status.json")
            status = read(status_path)
            if (
                status != completed[rid]
                or status.get("status") != "COMPLETE"
                or status.get("action") != "audit"
            ):
                raise ValueError("Audit completion evidence mismatch: " + rid)
            evidence.append(status_path)
    return root, frozen, completed, failed[0]["run_id"], evidence


def create_audit_recovery(source, name):
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
        mode="finish_audit",
        audit_source_task=str(source),
        audit_failed_run=failed,
        reused_audit_count=len(completed),
        collection_run_count=len(frozen["runs"]),
        gates=False,
        collection_authorized=False,
        training_authorized=False,
        scope="One failed audit with maximum3 attempts; reuse existing dataset/scaler and successful audits; global verification and stop. No collection, extraction, preprocessing, assembly or training.",
    )
    config["source_sha256"].update({str(p): sha(p) for p in evidence})
    write(task / "config.json", config)
    (task / "config.sha256").write_text(sha(task / "config.json") + "\n")
    return task


def finish_audit(runner):
    runner.begin("validate-existing-audits", workers=1)
    source = Path(runner.config["audit_source_task"])
    old, frozen, completed, failed, _ = inputs(source)
    if (
        failed != runner.config["audit_failed_run"]
        or len(completed) != runner.config["reused_audit_count"]
    ):
        raise ValueError("Audit recovery input changed")
    root = external(STORE / "processed/pretrain" / runner.task.name / "prepared")
    root.mkdir(parents=True, exist_ok=False)
    write(root / "frozen-manifest.json", frozen)
    runner.finish(
        "validate-existing-audits", dict(passed=True, reused=len(completed), retry_run=failed)
    )
    row = next(r for r in frozen["runs"] if r["run_id"] == failed)
    runner.batch(
        "prepared-audit-one",
        "audit",
        [row],
        1,
        root / "audit-work",
        canonical=old / "dataset/canonical",
    )
    sources = {
        rid: (old / "audit-work" / rid, source / "units/prepared-audit" / (rid + ".status.json"))
        for rid in completed
    }
    sources[failed] = (
        root / "audit-work" / failed,
        runner.task / "units/prepared-audit-one" / (failed + ".status.json"),
    )
    write(
        root / "reuse-evidence.json",
        dict(
            source_task=str(source),
            dataset=str(old / "dataset"),
            reused_audits=list(completed),
            processed_audits=[failed],
            preprocessing_repeated=False,
            assembly_repeated=False,
            training_authorized=False,
        ),
    )
    result = runner.finish_quality(frozen, root, "prepared", old / "dataset", audit_sources=sources)
    runner.update(result=str(result / "handoff.json"))
