"""Real synthetic preprocessing/audit and recoverable owned-only retirement."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from pretrain import common, storage, streaming


def setup_run(tmp_path, monkeypatch, role="normal_train"):
    common.package_imports()
    root, store = tmp_path / "root", tmp_path / "store"
    (root / "runs" / ("b" * 32)).mkdir(parents=True)
    (root / "runs" / ("b" * 32) / "old.txt").write_text("protected")
    task = root / "task"
    task.mkdir()
    common.write(task / "config.json", dict(mode="collect", storage_profile="stream_compact"))
    monkeypatch.setattr(streaming, "ROOT", root)
    monkeypatch.setattr(streaming, "STORE", store)
    monkeypatch.setattr(streaming, "external", lambda p: Path(p))

    class Runner:
        config = dict(mode="collect", storage_profile="stream_compact")
        state = {}

        def resources(self):
            return {}

        def update(self, **fields):
            self.state.update(fields)

    runner = Runner()
    runner.task = task
    pipeline = streaming.Pipeline(runner)
    rid, label = "a" * 32, "stream-001-001"
    raw = root / "runs" / rid
    fixture = common.load("stream_raw_fixture", common.ROOT / "ai/tests/test_preprocessing.py")
    meta = copy.deepcopy(fixture.META)
    meta["run_id"] = rid
    meta["vehicle"].update(
        nominal_capacity_ah=12.0,
        nominal_parallel_cells=4,
        battery_profile_id="vtc6_4s4p",
        profile_revision="vtc6-4s4p",
    )
    common.write(raw / "observations/metadata.json", meta)
    common.write(raw / "run.json", dict(run_id=rid, origin="synthetic_fixture", status="completed"))
    common.write(raw / "reference/source_lock.json", dict(synthetic_fixture=True))
    events = list(fixture.events(72))
    for event in events:
        event["run_id"] = rid
    (raw / "observations/streams.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    common.write(
        raw / "observations/watermarks.jsonl",
        dict(
            schema="drone.raw.watermark/2.0",
            run_id=rid,
            clock_id="px4-boot",
            through_available_time_us=72_000_000,
            recorded_events=len(events),
            closed=True,
        ),
    )
    marks = raw / "observations/watermarks.jsonl"
    marks.write_text(json.dumps(common.read(marks)) + "\n")
    (raw / "observations/diagnostics.jsonl").touch()
    files = {str(p.relative_to(raw)): common.sha(p) for p in raw.rglob("*") if p.is_file()}
    common.write(raw / "evaluation/scenario.json", dict(name="hover_normal", seed=17))
    (raw / "evaluation/events.jsonl").write_text("")
    archive = store / "archives" / label / (rid + ".tar.gz")
    archive.parent.mkdir(parents=True)
    members = {}
    with tarfile.open(archive, "w:gz") as tar:
        for p in raw.rglob("*"):
            if p.is_file():
                name = str(p.relative_to(raw))
                tar.add(p, arcname=name)
                members[name] = dict(kind="file", bytes=p.stat().st_size, sha256=common.sha(p))
    folder = task / label
    common.write(folder / "plan.json", dict(external_output=str(archive.parent)))
    chunk = pipeline.root / label
    consumer = store / "consumer/pretrain" / task.name / label / rid
    for name in files:
        target = consumer / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw / name, target)
    canonical = chunk / "canonical" / rid
    row = dict(
        run_id=rid,
        path=str(consumer),
        files=files,
        role=role,
        archive=dict(
            path=str(archive),
            bytes=archive.stat().st_size,
            sha256=common.sha(archive),
            members=members,
        ),
    )
    preprocess_request = tmp_path / "preprocess-request.json"
    common.write(
        preprocess_request,
        dict(
            action="preprocess",
            row=row,
            out=str(canonical),
            status=str(task / "units" / (label + "-preprocess") / (rid + ".status.json")),
        ),
    )
    # Match production isolation: every CPU unit has its own interpreter.
    processed = subprocess.run(
        [
            str(common.PYTHON),
            "-B",
            "-X",
            "faulthandler",
            "-m",
            "pretrain.worker",
            str(preprocess_request),
        ],
        env={**os.environ, "PYTHONPATH": common.pythonpath()},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert processed.returncode == 0, processed.stderr
    status = task / "units" / (label + "-audit") / (rid + ".status.json")
    request = dict(
        action="audit",
        row=row,
        out=str(chunk / "audit" / rid),
        status=str(status),
        canonical=str(canonical.parent),
    )
    (chunk / "audit").mkdir()
    common.write(tmp_path / "audit-request.json", request)
    run = subprocess.run(
        [str(common.PYTHON), "-B", "-m", "pretrain.worker", str(tmp_path / "audit-request.json")],
        env={**os.environ, "PYTHONPATH": common.pythonpath()},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert run.returncode == 0, run.stderr
    return SimpleNamespace(
        pipeline=pipeline,
        row=row,
        chunk=chunk,
        label=label,
        folder=folder,
        raw=raw,
        consumer=consumer,
        canonical=canonical,
        archive=archive,
        runner=runner,
    )


@pytest.mark.parametrize("role", ["normal_train", "normal_dev_eval"])
def test_verified_capsule_retires_only_new_success_and_restores_identical_data(
    tmp_path, monkeypatch, role
):
    f = setup_run(tmp_path, monkeypatch, role)
    original = (f.canonical / "frames.jsonl").read_bytes()
    failed = f.canonical.parent / ".failed" / f.row["run_id"] / "attempt-01"
    failed.mkdir(parents=True)
    (failed / "evidence.txt").write_text("failed evidence")
    row = f.pipeline.capsule(f.row, f.chunk, f.label, f.folder)
    assert not f.raw.exists() and not f.archive.exists() and not f.consumer.exists()
    assert not f.canonical.exists() and failed.exists()
    assert (streaming.ROOT / "runs" / ("b" * 32) / "old.txt").read_text() == "protected"
    receipt = common.read(row["compact_source"]["receipt"])
    restore = tmp_path / "restored"
    restore.mkdir()
    storage.verify_bundle(
        receipt["artifact"]["path"], receipt["artifact"]["members"], common.Budget(), restore
    )
    assert (restore / "canonical/frames.jsonl").read_bytes() == original
    assert (restore / "audit/numeric.npz").exists()
    assert all("streams.jsonl" not in n for n in receipt["artifact"]["members"])
    if role == "normal_dev_eval":
        assert storage.event_body(row, common.Budget()) == b""


def test_capsule_corruption_never_deletes_originals(tmp_path, monkeypatch):
    f = setup_run(tmp_path, monkeypatch)
    real = streaming.pack_files

    def corrupt(*args):
        result = real(*args)
        Path(result["path"]).write_bytes(b"corrupt")
        return result

    monkeypatch.setattr(streaming, "pack_files", corrupt)
    with pytest.raises(Exception):
        f.pipeline.capsule(f.row, f.chunk, f.label, f.folder)
    assert f.raw.exists() and f.archive.exists() and f.consumer.exists()


def test_px4_internal_links_are_not_followed_or_deleted_outside_owned_run(tmp_path, monkeypatch):
    f = setup_run(tmp_path, monkeypatch)
    external = tmp_path / "vendor/test_data"
    external.mkdir(parents=True)
    (external / "protected.txt").write_text("vendor untouched")
    (f.raw / "px4").mkdir()
    (f.raw / "reference/px4_etc").mkdir()
    (f.raw / "px4/etc").symlink_to(f.raw / "reference/px4_etc", target_is_directory=True)
    (f.raw / "px4/test_data").symlink_to(external, target_is_directory=True)
    (f.raw / "px4/dangling").symlink_to(tmp_path / "absent")
    stamp = streaming.tree_stamp(f.raw)
    assert str(f.raw / "px4/test_data/protected.txt") not in stamp
    assert stamp[str(f.raw / "px4/test_data")][-1] == str(external)
    f.pipeline.capsule(f.row, f.chunk, f.label, f.folder)
    assert not f.raw.exists()
    assert (external / "protected.txt").read_text() == "vendor untouched"


def test_changed_internal_link_blocks_retirement(tmp_path):
    owned = tmp_path / "owned"
    owned.mkdir()
    link = owned / "link"
    link.symlink_to(tmp_path / "first")
    allowed = {str(owned): streaming.tree_stamp(owned)}
    link.unlink()
    link.symlink_to(tmp_path / "second")
    with pytest.raises(ValueError, match="replaced cleanup target"):
        streaming.retire_owned(
            [owned],
            allowed,
            tmp_path / "cleanup.json",
            proof=dict(status="RUN_CAPSULE_VERIFIED"),
            budget=common.Budget(),
        )
    assert link.is_symlink() and owned.exists()


def test_changed_canonical_after_audit_never_deletes_originals(tmp_path, monkeypatch):
    f = setup_run(tmp_path, monkeypatch)
    with (f.canonical / "frames.jsonl").open("a") as s:
        s.write("{}\n")
    with pytest.raises(ValueError):
        f.pipeline.capsule(f.row, f.chunk, f.label, f.folder)
    assert f.raw.exists() and f.archive.exists()


def test_final_package_replaces_working_dataset_and_remains_restorable(tmp_path, monkeypatch):
    f = setup_run(tmp_path, monkeypatch)
    row = f.pipeline.capsule(f.row, f.chunk, f.label, f.folder)
    prepared = streaming.STORE / "processed/pretrain" / f.runner.task.name / "prepared"
    dataset = prepared / "dataset"
    dataset.mkdir(parents=True)
    common.write(dataset / "manifest.json", dict(runs=[row]))
    (dataset / "normal_train.npz").write_bytes(b"lossless-array-fixture")
    common.write(prepared / "dataset-seal.json", {p.name: common.sha(p) for p in dataset.iterdir()})
    common.write(prepared / "quality.json", dict(passed=True, train_only_scaler_verified=True))
    common.write(prepared / "frozen-manifest.json", dict(runs=[row]))
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
    audit = prepared / "audit-work" / row["run_id"]
    audit.mkdir(parents=True)
    (audit / "cache.npz").write_bytes(b"cache")
    failure = prepared / "audit-work/.failed/old-evidence"
    failure.mkdir(parents=True)
    (failure / "log.txt").write_text("preserve")
    monkeypatch.setattr(storage, "external", lambda p: Path(p))
    portable = prepared.parent / "prepared-portable"
    storage.pack(prepared, portable)
    f.runner.begin = lambda *a, **k: None
    f.runner.finish = lambda *a, **k: None
    f.pipeline.retire_final(prepared, [row])
    assert not dataset.exists() and not audit.exists() and failure.exists()
    assert common.read(prepared / "handoff.json")["training_eligible"] is False
    assert f.runner.state["dataset_requires_restore"] is True
    restored = storage.restore(portable, tmp_path / "restored-final")
    assert (restored / "dataset/normal_train.npz").read_bytes() == b"lossless-array-fixture"


def test_replacement_and_symlink_cleanup_targets_are_preserved(tmp_path):
    p = tmp_path / "new"
    p.write_text("data")
    allowed = {str(p): streaming.tree_stamp(p)}
    p.unlink()
    p.write_text("replaced data")
    with pytest.raises(ValueError):
        streaming.retire_owned(
            [p],
            allowed,
            tmp_path / "cleanup.json",
            proof=dict(status="RUN_CAPSULE_VERIFIED"),
            budget=common.Budget(),
        )
    assert p.exists()
    link = tmp_path / "link"
    link.symlink_to(p)
    with pytest.raises(ValueError):
        streaming.tree_stamp(link)


def test_missing_runs_never_reach_global_assembly(tmp_path, monkeypatch):
    import pretrain.collection as collection

    fake = SimpleNamespace(
        task=tmp_path, config=dict(flight_workers=2, collection_profile="diverse104"), state={}
    )
    common.write(tmp_path / "planned.json", dict(sets=[[dict(case_id=str(i)) for i in range(5)]]))
    fake.config["planned_cases"] = str(tmp_path / "planned.json")
    fake.update = lambda **kw: None
    fake.begin = lambda *a, **kw: None
    fake.finish = lambda *a, **kw: None
    calls = []
    fake.flight = lambda phase, count, rows, label: calls.append([r["case_id"] for r in rows])

    class Pipeline:
        def __init__(self, r):
            pass

    monkeypatch.setattr(streaming, "Pipeline", Pipeline)
    monkeypatch.setattr(streaming, "fragment_freeze", lambda *a: (None, []))
    monkeypatch.setattr(
        collection, "combine_frozen", lambda *a: pytest.fail("incomplete dataset admitted")
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        streaming.run(fake)
    assert calls == [["0", "1"], ["2", "3"], ["4"]]


def test_fragment_freeze_checks_real_identity_but_never_admits_partial_split(tmp_path, monkeypatch):
    from pretrain import collection

    common.package_imports()
    from preprocessing_tools import control

    store = tmp_path / "store"
    monkeypatch.setattr(control, "STORAGE", store)
    monkeypatch.setattr(common, "STORE", store)
    monkeypatch.setattr(streaming, "STORE", store)
    base = tmp_path / "resources"
    base.mkdir()
    shutil.copy2(common.RESOURCES / "collection-plan.json", base / "collection-plan.json")
    plan = common.read(base / "collection-plan.json")
    common.write(base / "collection-plan.json", dict(plan, storage_root=str(store)))
    (base / "admission").mkdir(parents=True)
    shutil.copy2(common.RESOURCES / "admission/freeze.py", base / "admission/freeze.py")
    common.write(
        base / "protected-inventory.json",
        dict(
            complete=True,
            runs=[dict(run_id="protected")],
            deny_paths=["protected"],
            includes_revisions=["R2_1GB"],
        ),
    )
    monkeypatch.setattr(streaming, "RESOURCES", base)
    monkeypatch.setattr(streaming, "ADMISSION", base / "admission")
    source = dict(local_sources=dict(px4="a", plugin="b", profile="c"), runtime_binary_sha256={})
    revision = dict(
        battery_profile="VTC6_4S4P",
        producer_lock_hash_format="sorted-json-v1",
        producer_lock_sha256=hashlib.sha256(
            (json.dumps(source, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
        px4_binary_sha256="a",
        physics_plugin_sha256="b",
        x500_profile_sha256="c",
    )
    monkeypatch.setattr(collection, "source_revision", lambda: revision)
    case = dict(
        case_id="only",
        role="normal_train",
        scenario="hover_normal",
        seed=17,
        mission_seconds=900,
        group_id="g",
    )
    rid = "c" * 32
    values = {name: {} for name in control.RAW_FILES}
    values.update(
        {
            "run.json": dict(run_id=rid, origin="px4_sitl", status="completed"),
            "evaluation/scenario.json": dict(
                name="hover_normal",
                seed=17,
                continuity="independent_reset",
                requested_duration_s=900,
                health_eligibility="confirmed_healthy",
            ),
            "reference/source_lock.json": source,
        }
    )
    raw = tmp_path / "synthetic"
    for name, value in values.items():
        common.write(raw / name, value)
    archive = store / "archives/new.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as tar:
        for name in values:
            tar.add(raw / name, arcname=name)
    members = {
        name: dict(kind="file", bytes=(raw / name).stat().st_size, sha256=common.sha(raw / name))
        for name in values
    }
    record = dict(
        case=case,
        run_id=rid,
        status="ARCHIVED_PENDING_QUALITY",
        assessment=dict(
            run_id=rid,
            **dict.fromkeys(
                [
                    "contract_passed",
                    "instance_isolation_passed",
                    "missing_data_passed",
                    "physical_gate_passed",
                ],
                True,
            ),
        ),
        export=dict(
            archive=str(archive),
            bytes=archive.stat().st_size,
            sha256=common.sha(archive),
            members=members,
        ),
    )
    folder = tmp_path / "chunk"
    common.write(folder / "manifest.json", dict(cases=[case]))
    ledger = dict(
        status="COLLECTED_WITH_ERRORS_STOPPED",
        phase="collection",
        cases=dict(only=record),
        failed_cases=["other"],
    )
    common.write(folder / "execution/ledger.json", ledger)
    original = common.sha(folder / "execution/ledger.json")
    out, rows = streaming.fragment_freeze(
        SimpleNamespace(config=dict(flight_workers=2)), folder, [case]
    )
    assert len(rows) == 1
    assert (
        common.read(out / "frozen-fragment.json")["status"]
        == "VERIFIED_FRAGMENT_PENDING_GLOBAL_SPLIT"
    )
    assert common.sha(folder / "execution/ledger.json") == original
    assert archive.exists()
