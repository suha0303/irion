"""Explicit recovery of a completed collection, with no collection execution path."""

from pathlib import Path

from .common import HOME, identity, read, sha, verify, write


def validated_collections(source):
    source = Path(source).resolve()
    if source.parent != (HOME / "tasks").resolve():
        raise ValueError("Registered local source task required")
    config = read(source / "config.json")
    verify({str(source / "config.json"): (source / "config.sha256").read_text().strip()})
    state = read(source / "operator/state.json")
    if config["mode"] != "collect" or state.get("status") != "FAILED_STOPPED":
        raise ValueError("Stopped collection task required; never duplicate an active task")
    if identity(state.get("pid", -1)) == state.get("pid_start_ticks") and state.get(
        "pid_start_ticks"
    ):
        raise ValueError("Source task process still exists")
    folders = [Path(p).resolve() for p in state.get("collected_sets", [])]
    if len(folders) != config.get("collection_sets", 1) or len(set(folders)) != len(folders):
        raise ValueError("Every planned collection must be complete")
    total = 0
    for folder in folders:
        if folder.parent != source:
            raise ValueError("Collection must belong to source task")
        ledger = read(folder / "execution/ledger.json")
        manifest = read(folder / "manifest.json")["cases"]
        if (
            ledger.get("status") != "COLLECTED_PENDING_QUALITY_STOPPED"
            or ledger.get("phase") != "collection"
            or ledger.get("not_started")
            or ledger.get("failed_cases")
            or ledger.get("stop_reason")
            or set(ledger["cases"]) != {c["case_id"] for c in manifest}
        ):
            raise ValueError("Incomplete collection cannot be recovered as a full dataset")
        for case in manifest:
            record = ledger["cases"][case["case_id"]]
            if record.get("status") != "ARCHIVED_PENDING_QUALITY" or record.get("case") != case:
                raise ValueError("Collection case/status mismatch")
        total += len(manifest)
    if total != config["collection_run_count"]:
        raise ValueError("Recovery run count differs from approved collection")
    return folders


def create_recovery(source, name):
    from . import cli

    source = Path(source).resolve()
    folders = validated_collections(source)
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
        mode="resume_preprocess",
        gates=False,
        collection_authorized=False,
        training_authorized=False,
        resume_source_task=str(source),
        resume_collections=[str(p) for p in folders],
        scope="Existing completed archives only: verify, preprocess, audit, stop. Per-run maximum3 attempts. No collection/training/task restart.",
    )
    evidence = [
        source / n
        for n in ("config.json", "config.sha256", "operator/state.json", "operator/run.log")
    ]
    evidence += [
        folder / n
        for folder in folders
        for n in ("manifest.json", "plan.json", "execution/ledger.json")
    ]
    config["source_sha256"].update({str(p): sha(p) for p in evidence})
    write(task / "config.json", config)
    (task / "config.sha256").write_text(sha(task / "config.json") + "\n")
    return task
