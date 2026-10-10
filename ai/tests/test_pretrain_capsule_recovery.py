"""Stopped capsule recovery never recollects or modifies the failed source task."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from pretrain import capsule_recovery as recovery
from pretrain import common, composition, storage, streaming


@pytest.fixture
def source(tmp_path, monkeypatch):
    import x500_storage

    home, store, package = tmp_path / "operations", tmp_path / "store", tmp_path / "package"
    resources = tmp_path / "resources"
    source = home / "tasks/source"
    source.mkdir(parents=True)
    store.mkdir()
    common.write(package / "preprocessing.json", dict(fixture=True))
    common.write(resources / "protected-inventory.json", dict(runs=[]))
    monkeypatch.setattr(recovery, "HOME", home)
    monkeypatch.setattr(recovery, "PACKAGE", package)
    monkeypatch.setattr(recovery, "RESOURCES", resources)
    monkeypatch.setattr(x500_storage, "guard", lambda p, policy=None: Path(p))
    monkeypatch.setattr(composition, "validate_roles", lambda *args, **kwargs: None)
    rid, label = "a" * 32, "stream-001-001"
    case = dict(case_id="case-one", role="normal_train", seed=17, mission_seconds=900)
    ref = dict(kind="file", bytes=3, sha256="a" * 64)
    original = dict(
        path="retired.tar.gz",
        bytes=20,
        sha256="b" * 64,
        members={
            name: ref
            for name in (
                "run.json",
                "reference/source_lock.json",
                "observations/metadata.json",
                "evaluation/scenario.json",
            )
        },
    )
    revision = dict(preprocessing_sha256=common.sha(package / "preprocessing.json"))
    row = dict(
        case,
        run_id=rid,
        path="retired-consumer",
        archive=original,
        assessment=dict(passed=True),
        files={},
        revision_lock=revision,
    )
    exported = dict(
        archive=original["path"], **{k: original[k] for k in ("bytes", "sha256", "members")}
    )
    ledger = dict(
        status="COLLECTED_PENDING_QUALITY_STOPPED",
        phase="collection",
        cases={
            case["case_id"]: dict(
                case=case,
                run_id=rid,
                status="ARCHIVED_PENDING_QUALITY",
                export=exported,
                assessment=row["assessment"],
            )
        },
    )
    folder, frozen_dir = source / label, source / label / "fragment-freeze"
    common.write(folder / "manifest.json", dict(cases=[case]))
    common.write(folder / "execution/ledger.json", ledger)
    common.write(frozen_dir / "cases.json", dict(cases=[case]))
    common.write(frozen_dir / "verified-fragment-ledger.json", ledger)
    common.write(frozen_dir / "required-revision.json", revision)
    common.write(frozen_dir / "collection-plan.json", dict(fixture=True))
    fragment = dict(
        status="VERIFIED_FRAGMENT_PENDING_GLOBAL_SPLIT",
        revision="COLLECTION",
        runs=[row],
        revision_lock=revision,
        original_ledger_sha256=common.sha(folder / "execution/ledger.json"),
        ledger_sha256=common.sha(frozen_dir / "verified-fragment-ledger.json"),
        cases_sha256=common.sha(frozen_dir / "cases.json"),
        plan_sha256=common.sha(frozen_dir / "collection-plan.json"),
        protected_inventory_sha256=common.sha(resources / "protected-inventory.json"),
    )
    common.write(frozen_dir / "frozen-fragment.json", fragment)
    output_hashes = dict.fromkeys(("frames", "windows", "exclusions"), "c" * 64)
    statuses = {}
    for action in ("preprocess", "audit"):
        path = source / "units" / (label + "-" + action) / (rid + ".status.json")
        result = (
            dict(output_hashes=output_hashes)
            if action == "preprocess"
            else dict(numeric_sha256="d" * 64, report_sha256="e" * 64)
        )
        common.write(path, dict(status="COMPLETE", action=action, run_id=rid, result=result))
        statuses[action] = path
    root = store / "streamed/pretrain/source/runs" / rid
    root.mkdir(parents=True)
    artifact = root / "artifacts.tar.xz"
    artifact.write_bytes(b"fixture artifact, deliberately not read during metadata admission")
    members = {
        "canonical/" + name + ".jsonl": dict(bytes=3, sha256=digest)
        for name, digest in output_hashes.items()
    }
    for name in ("canonical/checkpoint.json", "canonical/summary.json"):
        members[name] = dict(bytes=3, sha256="f" * 64)
    for name, original_name in (
        ("run.json", "run.json"),
        ("reference/source_lock.json", "reference/source_lock.json"),
        ("observations/metadata.json", "observations/metadata.json"),
        ("scenario.json", "evaluation/scenario.json"),
    ):
        members["identity/" + name] = {
            k: original["members"][original_name][k] for k in ("bytes", "sha256")
        }
    members["audit/status.json"] = dict(
        bytes=statuses["audit"].stat().st_size, sha256=common.sha(statuses["audit"])
    )
    members["audit/numeric.npz"] = dict(bytes=3, sha256="d" * 64)
    members["audit/report.json"] = dict(bytes=3, sha256="e" * 64)
    receipt_path = root / "receipt.json"
    common.write(
        receipt_path,
        dict(
            status="RUN_CAPSULE_VERIFIED",
            task=str(source),
            run_id=rid,
            artifact=dict(
                path=str(artifact),
                bytes=artifact.stat().st_size,
                sha256=common.sha(artifact),
                members=members,
            ),
            audit_status_sha256=common.sha(statuses["audit"]),
            raw_archive_sha256=original["sha256"],
            row=dict(row, path=str(store / "consumer/pretrain/source" / label / rid)),
            event_path=None,
        ),
    )
    common.write(
        source / "operator/state.json",
        dict(
            status="FAILED_STOPPED",
            pid=-1,
            pid_start_ticks="dead",
            reason="Service exited: core-dump",
        ),
    )
    common.write(
        source / "operator/processing-state.json",
        dict(status="PROCESSING_COMPLETE", stream_completed=1),
    )
    planned = source / "planned-cases.json"
    common.write(planned, dict(sets=[[case]]))
    config = dict(
        mode="collect",
        storage_profile="stream_compact",
        collection_profile="balanced300",
        collection_sets=1,
        collection_run_count=1,
        flight_workers=16,
        workers=4,
        io_workers=4,
        audit_workers=4,
        export_workers=4,
        use_available_ram=True,
        use_swap=True,
        seed_base=17,
        planned_cases=str(planned),
        external_storage=str(store),
        source_sha256={
            str(p): common.sha(p)
            for p in (
                planned,
                package / "preprocessing.json",
                resources / "protected-inventory.json",
            )
        },
    )
    common.write(source / "config.json", config)
    (source / "config.sha256").write_text(common.sha(source / "config.json"))
    return SimpleNamespace(
        path=source,
        store=store,
        row=row,
        receipt=receipt_path,
        archive=artifact,
        fragment=frozen_dir / "frozen-fragment.json",
        status=statuses,
        ledger=folder / "execution/ledger.json",
    )


def test_metadata_admission_preserves_source_and_defers_large_archive_hash(source, monkeypatch):
    before = {p: p.read_bytes() for p in source.path.rglob("*") if p.is_file()}
    real_sha = recovery.sha

    def metadata_sha(path):
        assert Path(path).suffix != ".xz", "large archive must be verified only when restored"
        return real_sha(path)

    monkeypatch.setattr(recovery, "sha", metadata_sha)
    result = recovery.inputs(source.path)
    assert len(result["parts"][0]["frozen"]["runs"]) == 1
    assert result["parts"][0]["frozen"]["runs"][0]["compact_source"]["receipt"] == str(
        source.receipt
    )
    assert str(source.archive) not in result["source_pins"]
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize(
    "kind",
    [
        "receipt_row",
        "archive_path",
        "archive_size",
        "audit_digest",
        "canonical_digest",
        "identity_digest",
        "inventory",
        "ledger",
        "unfinished",
        "duplicate",
        "linked",
    ],
)
def test_invalid_capsule_evidence_rejected(source, kind, tmp_path):
    receipt = common.read(source.receipt)
    if kind == "receipt_row":
        receipt["row"]["seed"] += 1
    elif kind == "archive_path":
        receipt["artifact"]["path"] = str(tmp_path / "outside.xz")
    elif kind == "archive_size":
        receipt["artifact"]["bytes"] += 1
    elif kind == "audit_digest":
        receipt["audit_status_sha256"] = "0" * 64
    elif kind == "canonical_digest":
        receipt["artifact"]["members"]["canonical/windows.jsonl"]["sha256"] = "0" * 64
    elif kind == "identity_digest":
        receipt["artifact"]["members"]["identity/run.json"]["sha256"] = "0" * 64
    elif kind == "inventory":
        receipt["artifact"]["members"]["../escape"] = dict(bytes=0, sha256="0" * 64)
    elif kind == "ledger":
        data = common.read(source.ledger)
        data["failed_cases"] = ["case-one"]
        common.write(source.ledger, data)
    elif kind == "unfinished":
        common.write(source.status["audit"], dict(status="FAILED", run_id="a" * 32, action="audit"))
    elif kind == "duplicate":
        data = common.read(source.fragment)
        data["runs"].append(copy.deepcopy(data["runs"][0]))
        common.write(source.fragment, data)
    elif kind == "linked":
        outside = tmp_path / "outside.xz"
        source.archive.rename(outside)
        source.archive.symlink_to(outside)
    common.write(source.receipt, receipt)
    with pytest.raises(ValueError):
        recovery.inputs(source.path)
    assert source.archive.exists()


def test_live_source_rejected(source, monkeypatch):
    monkeypatch.setattr(recovery, "identity", lambda _: "dead")
    with pytest.raises(ValueError, match="still exists"):
        recovery.inputs(source.path)


def test_new_task_freezes_preserved_inputs_without_collection(source, tmp_path, monkeypatch):
    from pretrain import cli

    created = tmp_path / "new"
    created.mkdir()

    def create(args):
        assert args.workers == args.io_workers == args.audit_workers == 4
        assert args.use_available_ram and args.use_swap
        common.write(created / "config.json", dict(source_sha256={}, workers=args.workers))
        return created

    monkeypatch.setattr(cli, "create", create)
    monkeypatch.setattr(recovery.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**15))
    before = source.path.joinpath("operator/state.json").read_bytes()
    assert recovery.create_recovery(source.path, "new") == created
    config = common.read(created / "config.json")
    assert config["mode"] == "recover_capsules" and config["unit_attempts_max"] == 1
    assert not config["collection_authorized"] and not config["training_authorized"]
    assert not config["pipeline_overlap"]
    assert source.path.joinpath("operator/state.json").read_bytes() == before
    assert common.read(created / "recovery-inputs.json")["source_capsules_preserved"]


def test_real_restore_and_final_package_keep_source_capsule(tmp_path, monkeypatch):
    fixture = common.load(
        "recovery_stream_fixture", common.ROOT / "ai/tests/test_pretrain_streaming.py"
    )
    f = fixture.setup_run(tmp_path, monkeypatch)
    row = f.pipeline.capsule(f.row, f.chunk, f.label, f.folder)
    original_capsule = Path(common.read(row["compact_source"]["receipt"])["artifact"]["path"])
    original_hash = common.sha(original_capsule)
    task = tmp_path / "fresh-task"
    task.mkdir()
    common.write(task / "recovery-inputs.json", dict(source_pins={}, restore_bytes=0, run_count=1))
    r = SimpleNamespace(
        task=task,
        state={},
        config=dict(
            recovery_inputs=str(task / "recovery-inputs.json"),
            recovery_source_task=str(f.runner.task),
            workers=4,
        ),
    )
    r.update = lambda **kw: r.state.update(kw)
    r.begin = lambda *a, **kw: None
    r.finish = lambda *a, **kw: None
    r.resources = lambda: None
    frozen = dict(runs=[row])
    monkeypatch.setattr(
        recovery,
        "inputs",
        lambda _: dict(source_pins={}, restore_bytes=0, parts=[dict(cases=[], frozen=frozen)]),
    )
    from pretrain import collection

    monkeypatch.setattr(collection, "combine_frozen", lambda *a: frozen)
    monkeypatch.setattr(recovery, "STORE", streaming.STORE)
    monkeypatch.setattr(recovery, "external", lambda p: Path(p))
    monkeypatch.setattr(storage, "external", lambda p: Path(p))

    def finalize(frozen, prepared, label, workers, preaudited):
        assert workers == 4 and row["run_id"] in preaudited
        dataset = prepared / "dataset"
        dataset.mkdir()
        (prepared / "canonical-work").rename(dataset / "canonical")
        common.write(dataset / "manifest.json", frozen)
        (dataset / "normal_train.npz").write_bytes(b"fixture")
        common.write(
            prepared / "dataset-seal.json",
            {str(p.relative_to(dataset)): common.sha(p) for p in dataset.rglob("*") if p.is_file()},
        )
        common.write(prepared / "quality.json", dict(passed=True, train_only_scaler_verified=True))
        common.write(
            prepared / "handoff.json",
            dict(
                training_eligible=True,
                dataset=str(dataset),
                dataset_seal_sha256=common.sha(prepared / "dataset-seal.json"),
                quality_sha256=common.sha(prepared / "quality.json"),
                frozen_manifest_sha256=common.sha(prepared / "frozen-manifest.json"),
            ),
        )
        storage.pack(prepared, prepared.parent / "prepared-portable")

    r.finalize_prepared = finalize
    recovery.run(r)
    assert original_capsule.exists() and common.sha(original_capsule) == original_hash
    assert r.state["dataset_requires_restore"] is True
    assert common.read(r.state["result"])["status"] == "PORTABLE_DATASET_VERIFIED"
    prepared = streaming.STORE / "processed/pretrain" / task.name / "prepared"
    assert not (prepared / "dataset").exists()
    assert not (prepared / "audit-work").exists()


def test_archive_corruption_is_rejected_before_assembly(tmp_path, monkeypatch):
    fixture = common.load(
        "recovery_corrupt_fixture", common.ROOT / "ai/tests/test_pretrain_streaming.py"
    )
    f = fixture.setup_run(tmp_path, monkeypatch)
    row = f.pipeline.capsule(f.row, f.chunk, f.label, f.folder)
    artifact = Path(common.read(row["compact_source"]["receipt"])["artifact"]["path"])
    artifact.write_bytes(b"corrupt")
    prepared = tmp_path / "restore"
    prepared.mkdir()
    f.runner.begin = lambda *a, **kw: None
    f.runner.finish = lambda *a, **kw: pytest.fail("corrupt restore reported completion")
    with pytest.raises(ValueError, match="archive changed"):
        streaming.restore_capsules(f.runner, [row], prepared, common.Budget())
    assert artifact.exists()
