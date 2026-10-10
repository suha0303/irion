"""Recovery never submits flights or rewrites the failed task."""

import pytest

from pretrain import cli, common, recovery, runner


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "HOME", tmp_path)
    task = tmp_path / "tasks/source"
    common.write(
        task / "config.json", dict(mode="collect", collection_sets=1, collection_run_count=1)
    )
    (task / "config.sha256").write_text(common.sha(task / "config.json"))
    common.write(
        task / "operator/state.json",
        dict(
            status="FAILED_STOPPED",
            pid=-1,
            pid_start_ticks="not-running",
            collected_sets=[str(task / "collection")],
        ),
    )
    case = dict(case_id="one")
    common.write(task / "collection/manifest.json", dict(cases=[case]))
    common.write(
        task / "collection/execution/ledger.json",
        dict(
            status="COLLECTED_PENDING_QUALITY_STOPPED",
            phase="collection",
            cases=dict(one=dict(case=case, status="ARCHIVED_PENDING_QUALITY")),
            not_started=[],
            failed_cases=[],
            stop_reason=[],
        ),
    )
    return task


def test_complete_collection_is_read_only(source):
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert recovery.validated_collections(source) == [source / "collection"]
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "FAILED_DRAINED_STOPPED"),
        ("failed_cases", ["one"]),
        ("not_started", ["two"]),
        ("stop_reason", ["failure"]),
    ],
)
def test_partial_collection_rejected(source, field, value):
    path = source / "collection/execution/ledger.json"
    ledger = common.read(path)
    ledger[field] = value
    common.write(path, ledger)
    with pytest.raises(ValueError, match="Incomplete"):
        recovery.validated_collections(source)


def test_running_source_rejected(source, monkeypatch):
    monkeypatch.setattr(recovery, "identity", lambda pid: "not-running")
    with pytest.raises(ValueError, match="process still exists"):
        recovery.validated_collections(source)


def test_resume_dispatch_cannot_collect(tmp_path, monkeypatch):
    task = tmp_path / "new"
    config = dict(
        mode="resume_preprocess",
        source_sha256={},
        resume_source_task=str(tmp_path / "source"),
        resume_collections=[str(tmp_path / "source/collection")],
    )
    common.write(task / "config.json", config)
    (task / "config.sha256").write_text(common.sha(task / "config.json"))
    (task / "operator").mkdir()
    r = runner.Runner(task)
    monkeypatch.setattr(r, "resources", lambda: {})
    monkeypatch.setattr(runner, "external", lambda path: tmp_path / "external")
    monkeypatch.setattr(
        recovery, "validated_collections", lambda path: [tmp_path / "source/collection"]
    )
    prepared = []
    monkeypatch.setattr(r, "prepare_collections", lambda folders: prepared.extend(folders))
    monkeypatch.setattr(r, "collection_sets", lambda: pytest.fail("recollection invoked"))
    monkeypatch.setattr(r, "benchmark", lambda: pytest.fail("flight test invoked"))
    r.run()
    assert prepared == [tmp_path / "source/collection"]
    assert r.state["status"] == "PRETRAIN_READY_STOPPED_PENDING_USER_REVIEW"
    assert r.state["training_authorized"] is False
    with pytest.raises(ValueError, match="Recollection forbidden"):
        r.flight("collection", 16, [], "forbidden")
    with pytest.raises(FileExistsError):
        r.run()


def test_resume_console_has_no_collection_flow(tmp_path):
    task = tmp_path / "task"
    common.write(
        task / "config.json",
        dict(
            mode="resume_preprocess",
            collection_sets=1,
            collection_run_count=104,
            io_workers=16,
            workers=16,
            audit_workers=16,
        ),
    )
    common.write(
        task / "operator/state.json",
        dict(status="FAILED_STOPPED", stage="freeze", started=0, finished=1, completed=[]),
    )
    result = cli.render(task)
    assert "재수집 없음" in result
    assert "압축본 검증·인수" in result
    assert "0/7" in result


def test_recovery_freeze_output_is_new_not_original(source, tmp_path, monkeypatch):
    from pretrain import collection

    task = tmp_path / "new"
    common.write(task / "config.json", dict(mode="resume_preprocess", source_sha256={}, workers=16))
    r = runner.Runner(task)
    original = source / "collection"
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}

    def freeze(task_arg, folder, *, output, progress):
        assert task_arg == task and folder == original
        assert output == task / "recovery-freeze-1"
        common.write(output / "frozen-evidence.json", dict(passed=True))
        progress(1)
        return dict(runs=[dict(run_id="one")])

    monkeypatch.setattr(collection, "freeze_collection", freeze)
    monkeypatch.setattr(collection, "combine_frozen", lambda task, parts: parts[0][1])
    monkeypatch.setattr(r, "prepare", lambda *a, **k: tmp_path / "external/prepared")
    r.prepare_collections([original])
    assert r.state["completed"] == ["freeze", "combine"]
    assert before == {p: p.read_bytes() for p in before}
    assert r.state["result"] == str(tmp_path / "external/prepared/handoff.json")
