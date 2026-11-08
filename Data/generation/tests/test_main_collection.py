"""Production menu, finite retries, checkpoint and retirement fixtures;
native dispatch0.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from pretrain import cli, common, main_storage, pilot45, pilot_parallel
from pretrain import main_collection as main


@pytest.mark.local_evidence
def test_frozen_main_contains_lr_once_and_all_approved_roles():
    _, full = main.plans("main")
    _, lr = main.plans("lr108")
    assert len(full["cases"]) == 1674 and len(lr["cases"]) == 108
    by_id = {c["case_id"]: c for c in full["cases"]}
    assert all(by_id[c["case_id"]] == c for c in lr["cases"])
    chunks = list(main.groups(full["cases"]))
    assert sum(map(len, chunks)) == 1674 and max(map(len, chunks)) == 16
    for c in full["cases"]:
        if c["category"] == "anomaly":
            assert any(
                c in chunk
                and any(
                    n["group_id"] == c["group_id"]
                    and n["category"] == "paired_normal"
                    for n in chunk
                )
                for chunk in chunks
            )


@pytest.mark.parametrize("choice,scope", [("1", "lr108"), ("2", "main")])
def test_real_menu_dispatches_selected_finite_scope_once(
    monkeypatch, choice, scope
):
    calls = []
    task = Path("fixture-task")
    monkeypatch.setattr("builtins.input", lambda _: choice)
    monkeypatch.setattr(
        main, "user_start", lambda selected: calls.append(selected) or task
    )
    monkeypatch.setattr(cli, "watch", calls.append)
    cli.menu()
    assert calls == [scope, task]


def test_menu_can_open_existing_screen_without_submitting(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "choose", lambda *a: "3")
    monkeypatch.setattr(cli, "existing_task", lambda: calls.append("existing"))
    monkeypatch.setattr(
        main, "user_start", lambda *a: pytest.fail("No resubmission")
    )
    cli.menu()
    assert calls == ["existing"]


@pytest.mark.parametrize("no_watch", [False, True])
def test_direct_collection_opens_screen_only_when_requested(
    monkeypatch, tmp_path, no_watch
):
    calls = []
    monkeypatch.setattr(
        main,
        "user_start",
        lambda *a, **k: calls.append("submitted") or tmp_path,
    )
    monkeypatch.setattr(
        cli, "watch", lambda path: calls.append(("screen", path))
    )
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(main.sys.stdout, "isatty", lambda: True)
    main.main(["collect", "main"] + (["--no-watch"] if no_watch else []))
    assert calls == ["submitted"] + ([] if no_watch else [("screen", tmp_path)])


def test_closed_raw_is_reused_without_intermediate_archive(tmp_path):
    from pretrain import ai19, main_views

    raw = tmp_path / "raw"
    for name in ai19.CONSUMER:
        common.write(raw / name, {"fixture_only": True})
    members = {
        str(p.relative_to(raw)): {
            "size": p.stat().st_size,
            "sha256": common.sha(p),
        }
        for p in raw.rglob("*")
        if p.is_file()
    }
    common.write(raw / "integrity.json", members)
    before = {name: common.sha(raw / name) for name in members}
    source = main_views.closed_source(raw)
    assert source["kind"] == "closed_raw_directory"
    assert source["intermediate_archive"] is False
    assert source["replay_source"] == "original_read_only"
    assert {name: common.sha(raw / name) for name in members} == before
    assert not list(tmp_path.rglob("*.tar.*"))
    changed = raw / next(iter(members))
    changed.write_bytes(b"changed size")
    with pytest.raises(ValueError, match="Closed raw size changed"):
        main_views.closed_source(raw)


@pytest.mark.local_evidence
@pytest.mark.parametrize(
    "message,expected_calls",
    [("Case child exit1; fixture", 3), ("Available RAM below1GB", 1)],
)
def test_unit_exit_retries_but_parent_resource_stop_does_not(
    monkeypatch, tmp_path, message, expected_calls
):
    from concurrent.futures import ThreadPoolExecutor

    task, rows, _ = runtime(monkeypatch, tmp_path, count=1)
    c = main.Campaign(task)
    entry = c.ledger["cases"][rows[0]["case_id"]]
    raw = str(tmp_path / "valid-raw")
    entry.update(
        run=raw, attempts=[dict(number=1, state="COMPLETED", run_id="fixture")]
    )
    calls = []

    def fail(job):
        calls.append(job)
        common.write(
            Path(job["request"]).with_suffix(".failure.json"),
            dict(
                schema="drone-data.main19.child-failure/1",
                failure_scope="unit",
                error_type="FileNotFoundError",
                failure_reason="fixture unit error",
            ),
        )
        raise pilot45.Blocked(message)

    monkeypatch.setattr(c, "process", fail)
    with ThreadPoolExecutor(max_workers=4) as pool:
        if expected_calls == 1:
            with pytest.raises(pilot45.Blocked):
                c.process_batch(rows, pool)
        else:
            c.process_batch(rows, pool)
    assert len(calls) == expected_calls
    assert entry["run"] == raw and len(entry["attempts"]) == 1
    if expected_calls == 3:
        assert (
            entry["status"] == "UNAVAILABLE"
            and entry["processing_attempts"] == 3
        )


def runtime(monkeypatch, tmp_path, count=17):
    _, plan = main.plans("main")
    rows = plan["cases"][:count]
    task = tmp_path / "task"
    store = tmp_path / "Data"
    cfg = dict(
        mode="main19",
        scope="lr108",
        external_storage=str(store),
        source_sha256={},
        lineage_namespace=main.NAMESPACE,
        stop_on_first_physical_failure=False,
        parallel16=True,
        service_unit="drone-pretrain-task.service",
        producer={},
        dependency_handoff="fixture",
    )
    monkeypatch.setattr(main, "validate", lambda t: (cfg, dict(cases=rows)))
    monkeypatch.setattr(main, "dependencies", lambda *a: {})
    monkeypatch.setattr(main, "resources", lambda *a, **k: {})
    monkeypatch.setattr(main, "CATALOG", tmp_path / "catalog")
    monkeypatch.setattr(pilot_parallel, "ROOT", tmp_path)
    monkeypatch.setattr(pilot_parallel, "check_ports", lambda *a: None)
    monkeypatch.setattr(
        pilot45,
        "collector",
        lambda: SimpleNamespace(check_instance_socket=lambda i: None),
    )
    monkeypatch.setattr(pilot45, "register", lambda *a: None)
    common.write(task / "config.json", cfg)
    common.write(task / "budget.json", dict(batch_scratch_bytes=0))
    common.write(task / "protection.json", dict(records=[]))
    paths = {}
    for c in rows:
        p = task / "cases" / (c["case_id"] + ".json")
        common.write(p, dict(case=c, attempts=[]))
        paths[c["case_id"]] = str(p)
    common.write(task / "ledger.json", dict(cases=paths))
    return task, rows, cfg


@pytest.mark.local_evidence
@pytest.mark.parametrize("failures", [1, 3])
def test_general_native_failure_gets_total3_and_preserved_parent_then_next_case(
    monkeypatch, tmp_path, failures
):
    task, rows, _ = runtime(monkeypatch, tmp_path)
    counts = {}
    events = []

    def child(
        unused_task, request, log, *, on_progress=None, resource_snapshot=None
    ):
        j = common.read(request)
        c = j["case"]
        cid = c["case_id"]
        events.append((j["action"], cid))
        if j["action"] == "flight":
            n = counts[cid] = counts.get(cid, 0) + 1
            failed = cid == rows[0]["case_id"] and n <= failures
            rid = hashlib.md5(
                (cid + str(n)).encode(), usedforsecurity=False
            ).hexdigest()
            raw = tmp_path / "simulation/runs" / rid
            common.write(
                raw / "run.json",
                dict(
                    run_id=rid,
                    status="failed" if failed else "completed",
                    instance=j["instance"],
                    parent_attempt_id=j["parent_attempt_id"],
                    owned_process_groups={},
                    primary_failure=dict(reason="native_wait_timeout"),
                ),
            )
            common.write(
                raw / "evaluation/scenario.json",
                dict(
                    name=c["scenario"],
                    seed=c["seed"],
                    operation_variation=c["operation_variation"],
                ),
            )
            common.write(
                j["receipt"],
                dict(
                    status="failed" if failed else "completed",
                    run=str(raw),
                    reason="native_wait_timeout" if failed else None,
                ),
            )
            if failed:
                raise RuntimeError("Recorded independent native timeout")
        elif j["action"] == "prepare":
            output = Path(j["output"])
            common.write(output / "receipt.json", dict(fixture_only=True))
            common.write(output / "seal.json", {})
            common.write(
                j["receipt"],
                dict(
                    files={},
                    run=dict(raw_sha256="a" * 64),
                    online=dict(available=True),
                    run_health=dict(status="available"),
                ),
            )
        elif j["action"] == "audit":
            common.write(
                j["receipt"],
                dict(
                    export_receipt_sha256=common.sha(
                        Path(j["output"]) / "receipt.json"
                    )
                ),
            )
        elif j["action"] == "commit":
            p = tmp_path / "commits" / (cid + ".json")
            common.write(p, dict(fixture_only=True))
            common.write(
                j["receipt"],
                dict(commit=dict(path=str(p), sha256=common.sha(p))),
            )
        elif j["action"] == "aggregate":
            common.write(j["receipt"], dict(fixture_only=True))
        else:
            raise AssertionError(j["action"])

    monkeypatch.setattr(pilot45, "child", child)
    c = main.Campaign(task)
    c.run()
    first = c.ledger["cases"][rows[0]["case_id"]]
    assert counts[rows[0]["case_id"]] == (2 if failures == 1 else 3)
    assert all(counts[r["case_id"]] == 1 for r in rows[1:])
    assert first["attempts"][0]["state"] == "FAILED_PRESERVED"
    for before, after in zip(first["attempts"], first["attempts"][1:]):
        assert after["parent_attempt_id"] == before["run_id"]
    assert all(a["number"] <= 3 for a in c.ledger["attempts"])
    if failures == 1:
        assert (
            first["status"] == "COMMITTED"
            and c.state["status"] == "MAIN19_INPUTS_PASSED_TERMINATING"
        )
    else:
        assert (
            first["status"] == "FAILED_PRESERVED"
            and c.state["status"] == "MAIN19_INCOMPLETE_STOPPED"
        )
        assert not any(action == "aggregate" for action, cid in events)
    index = [
        json.loads(line)
        for line in (task / "exports.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(index) == 17
    assert index[0]["status"] == ("accepted" if failures == 1 else "failed")
    assert not list((tmp_path / "simulation/runs").glob("*/px4"))


@pytest.mark.parametrize(
    "reason",
    [
        "disk full",
        "integrity mismatch",
        "source changed",
        "contract mismatch",
        "MemoryError",
    ],
)
def test_shared_failures_are_never_retried(tmp_path, reason):
    raw = tmp_path / "run"
    common.write(
        raw / "run.json",
        dict(status="failed", primary_failure=dict(reason=reason)),
    )
    result = dict(status="failed", run=str(raw), reason=reason)
    assert main.retryable(result, 1) is False and result.get(
        "main_shared_failure"
    )


def storage_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(main_storage.ai19, "ROOT", tmp_path)
    task = tmp_path / "tasks/current"
    store = tmp_path / "Data"
    raw = tmp_path / "simulation/runs" / ("a" * 32)
    out = store / "processed/main19/current/case/attempt-01"
    archive = store / "archives/main19/current/run.tar.gz"
    consumer = store / "consumer/main19/current/run/attempt-01"
    raw.mkdir(parents=True)
    consumer.mkdir(parents=True)
    archive.parent.mkdir(parents=True)
    (raw / "run.json").write_text('{"status":"completed"}', encoding="utf-8")
    (consumer / "streams").write_bytes(b"consumer")
    archive.write_bytes(b"verified gzip")
    common.write(task / "config.json", dict(external_storage=str(store)))
    prepared = dict(
        run=dict(run_id=raw.name),
        archive=dict(path=str(archive), sha256=common.sha(archive)),
        consumer_path=str(consumer),
    )
    common.write(out / "receipt.json", prepared)
    common.write(out / "frames.jsonl", {"fixture_only": True})
    audit = task / "audit.json"
    common.write(
        audit, dict(export_receipt_sha256=common.sha(out / "receipt.json"))
    )
    common.write(
        task / "pilot/execution/case/archive-cache.json",
        dict(
            archive=dict(
                members={
                    "run.json": dict(
                        kind="file",
                        bytes=(raw / "run.json").stat().st_size,
                        sha256=common.sha(raw / "run.json"),
                    )
                }
            )
        ),
    )
    entry = dict(
        case=dict(case_id="case"),
        run=str(raw),
        output=str(out),
        receipt=str(out / "receipt.json"),
        audit_receipt=str(audit),
        online_available=True,
        run_available=True,
        attempts=[
            dict(
                number=1,
                state="COMPLETED",
                run_id=raw.name,
                parent_attempt_id=None,
            )
        ],
        files={
            str(p): common.sha(p)
            for p in [out / "receipt.json", out / "frames.jsonl", audit]
        },
    )
    target = store / "capsules/main19/current/case"
    protected = tmp_path / "protected"
    protected.write_text("prior data", encoding="utf-8")
    return task, entry, target, raw, archive, consumer, protected


def test_partial_capsule_failure_is_preserved_and_resume_never_reflies(
    monkeypatch, tmp_path
):
    task, e, target, raw, archive, consumer, protected = storage_fixture(
        monkeypatch, tmp_path
    )
    actual = main_storage.pack_files

    def crash(sources, path, budget, **kwargs):
        Path(path).write_bytes(b"interrupted partial")
        raise RuntimeError("power loss before publication")

    monkeypatch.setattr(main_storage, "pack_files", crash)
    with pytest.raises(RuntimeError):
        main_storage.commit(task, e, target, set())
    assert (
        raw.exists()
        and archive.exists()
        and not (target / "capsule.tar.xz").exists()
    )
    monkeypatch.setattr(main_storage, "pack_files", actual)
    main_storage.commit(task, e, target, set())
    assert (
        not raw.exists()
        and not archive.exists()
        and not consumer.exists()
        and protected.read_text(encoding="utf-8") == "prior data"
    )
    discarded = list(target.glob("*.partial.tar.xz.discarded.json"))
    assert len(discarded) == 1
    assert common.read(discarded[0])["discarded_bytes"] == len(
        b"interrupted partial"
    )
    assert not list(target.glob("*.partial.tar.xz"))
    main_storage.committed(target / "commit.json")


def test_interrupted_deletion_has_durable_intent_and_exact_resume(
    monkeypatch, tmp_path
):
    task, e, target, raw, archive, consumer, protected = storage_fixture(
        monkeypatch, tmp_path
    )
    actual = main_storage.retire_owned

    def crash(paths, allowed, journal, **kwargs):
        actual(paths, allowed, journal, **kwargs)
        raise RuntimeError("power loss after first owned deletion")

    monkeypatch.setattr(main_storage, "retire_owned", crash)
    with pytest.raises(RuntimeError):
        main_storage.commit(task, e, target, set())
    assert not raw.exists() and archive.exists() and consumer.exists()
    assert common.read(target / "cleanup-current.json")["in_progress"] == str(
        raw
    )
    monkeypatch.setattr(main_storage, "retire_owned", actual)
    main_storage.commit(task, e, target, set())
    assert (
        not archive.exists()
        and not consumer.exists()
        and protected.read_text(encoding="utf-8") == "prior data"
    )
    assert len(common.read(target / "cleanup.json")["deleted"]) == 3


@pytest.mark.local_evidence
def test_worker_preflight_failure_is_visible_without_any_dispatch(
    monkeypatch, tmp_path
):
    task, _, cfg = runtime(monkeypatch, tmp_path, count=1)
    cfg.update(owner_uid=os.getuid())
    common.write(task / "config.json", cfg)
    common.write(task / "operator/state.json", dict(status="READY_NOT_STARTED"))
    monkeypatch.setattr(pilot45, "task_path", Path)

    def blocked(task):
        raise ValueError("source preflight mismatch")

    monkeypatch.setattr(main, "preflight", blocked)
    monkeypatch.setattr(
        pilot45,
        "child",
        lambda *a, **k: pytest.fail("No child on preflight failure"),
    )
    with pytest.raises(ValueError, match="preflight"):
        main.worker(task)
    state = common.read(task / "operator/state.json")
    assert state["status"] == "BLOCKED_STOPPED" and "source" in state["reason"]


@pytest.mark.local_evidence
def test_real_subprocess_shared_errno_survives_wrapped_child_error(
    monkeypatch, tmp_path
):
    task, rows, cfg = runtime(monkeypatch, tmp_path, count=1)
    handoff = tmp_path / "handoff.json"
    common.write(handoff, {"fixture_only": True})
    cfg.update(
        policy_bundle={},
        dependency_handoff=str(handoff),
        source_sha256={str(handoff): common.sha(handoff)},
    )
    common.write(task / "config.json", cfg)
    (task / "config.sha256").write_text(
        common.sha(task / "config.json"), encoding="utf-8"
    )
    common.write(task / "manifest.json", dict(cases=rows))
    raw = tmp_path / "simulation/runs" / ("c" * 32)
    entry = common.read(task / "cases" / (rows[0]["case_id"] + ".json"))
    entry.update(
        run=str(raw),
        attempts=[
            dict(
                number=1,
                state="COMPLETED",
                run_id=raw.name,
                parent_attempt_id=None,
            )
        ],
    )
    common.write(task / "cases" / (rows[0]["case_id"] + ".json"), entry)
    c = main.Campaign(task)
    job = c.begin_processing(rows[0])
    script = """
import errno,sys
from pathlib import Path
from pretrain import common,main_collection as m,pilot45,main_views
p=Path(sys.argv[1]);t=Path(common.read(p)["task"])
pilot45.task_path=lambda value:Path(value)
m.validate=lambda task:(common.read(t/"config.json"),common.read(t/"manifest.json"))
m.dependencies=lambda path:{}
def fail(*args,**kwargs):raise OSError(errno.ENOSPC,"fixture disk full")
main_views.prepare=fail
m.case_entry(p)
"""
    result = subprocess.run(
        [str(common.PYTHON), "-B", "-c", script, str(job["request"])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert Path(job["request"]).with_suffix(".failure.json").exists(), (
        result.stderr
    )
    failure = common.read(Path(job["request"]).with_suffix(".failure.json"))
    assert failure["failure_scope"] == "shared" and failure["errno"] == 28
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(1)
        raise RuntimeError(
            "Case child exit1; original errno is only in its receipt"
        )

    monkeypatch.setattr(c, "begin_processing", lambda case: job)
    monkeypatch.setattr(c, "process", wrapped)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        with pytest.raises(pilot45.Blocked):
            c.process_batch(rows, pool)
    assert (
        calls == [1]
        and entry["run"] == c.ledger["cases"][rows[0]["case_id"]]["run"]
    )


@pytest.mark.local_evidence
@pytest.mark.parametrize(
    "quality_failures,transform_failures", [(1, 0), (3, 0), (0, 2)]
)
def test_quality_rejection_preserves_completed_and_transform_retry_keeps_raw(
    monkeypatch, tmp_path, quality_failures, transform_failures
):
    from pretrain import main_quality

    task, rows, _ = runtime(monkeypatch, tmp_path, count=1)
    native = []
    prepare = []
    monkeypatch.setattr(
        main_quality,
        "allowed",
        lambda cfg, case, prior: bool(prior and prior.get("quality_rejection")),
    )

    def child(
        unused_task, request, log, *, on_progress=None, resource_snapshot=None
    ):
        j = common.read(request)
        c = j["case"]
        action = j["action"]
        if action == "flight":
            rid = f"{len(native) + 1:032x}"
            native.append(rid)
            raw = tmp_path / "simulation/runs" / rid
            common.write(
                raw / "run.json",
                dict(
                    run_id=rid,
                    status="completed",
                    instance=j["instance"],
                    parent_attempt_id=j["parent_attempt_id"],
                    owned_process_groups={},
                ),
            )
            common.write(
                raw / "evaluation/scenario.json",
                dict(
                    name=c["scenario"],
                    seed=c["seed"],
                    operation_variation=c["operation_variation"],
                ),
            )
            common.write(j["receipt"], dict(run=str(raw), status="completed"))
        elif action == "prepare":
            prepare.append(j["selection"]["run_id"])
            if len(native) <= quality_failures:
                q = Path(request).with_suffix(".quality-fixture.json")
                common.write(
                    q,
                    dict(
                        fixture_only=True,
                        run_id=native[-1],
                        physical_status="completed",
                    ),
                )
                common.write(
                    Path(request).with_suffix(".failure.json"),
                    dict(
                        failure_scope="quality_rejection",
                        quality_rejection=dict(
                            path=str(q), sha256=common.sha(q)
                        ),
                    ),
                )
                raise RuntimeError(
                    "Case child exit1: official rejection fixture"
                )
            if len(prepare) <= transform_failures:
                raise RuntimeError("Independent conversion fixture error")
            out = Path(j["output"])
            common.write(out / "receipt.json", dict(fixture_only=True))
            common.write(out / "seal.json", {})
            common.write(
                j["receipt"],
                dict(
                    files={},
                    run=dict(raw_sha256="a" * 64),
                    online=dict(available=True),
                    run_health=dict(status="available"),
                ),
            )
        elif action == "audit":
            common.write(
                j["receipt"],
                dict(
                    export_receipt_sha256=common.sha(
                        Path(j["output"]) / "receipt.json"
                    )
                ),
            )
        elif action == "commit":
            q = tmp_path / "committed.json"
            common.write(q, dict(fixture_only=True))
            common.write(
                j["receipt"],
                dict(commit=dict(path=str(q), sha256=common.sha(q))),
            )
        elif action == "aggregate":
            common.write(j["receipt"], dict(fixture_only=True))
        else:
            raise AssertionError(action)

    monkeypatch.setattr(pilot45, "child", child)
    c = main.Campaign(task)
    c.run()
    e = c.ledger["cases"][rows[0]["case_id"]]
    assert len(native) == (
        1 if not quality_failures else min(quality_failures + 1, 3)
    )
    assert all(a["state"] == "COMPLETED" for a in e["attempts"])
    for i, a in enumerate(e["attempts"]):
        assert a["parent_attempt_id"] == (native[i - 1] if i else None)
    projected = common.read(
        task / "case-ledgers" / (rows[0]["case_id"] + ".json")
    )
    assert (
        sum(a["status"] == "quality_rejected" for a in projected["attempts"])
        == quality_failures
    )
    if quality_failures == 3:
        assert (
            e["status"] == "UNAVAILABLE"
            and c.state["status"] == "MAIN19_INCOMPLETE_STOPPED"
        )
    else:
        assert e["status"] == "COMMITTED"
    if transform_failures:
        assert prepare == [native[0]] * 3
    for rid in native[:quality_failures]:
        assert (
            common.read(tmp_path / "simulation/runs" / rid / "run.json")[
                "status"
            ]
            == "completed"
        )


@pytest.mark.parametrize(
    "change",
    [
        None,
        "gc",
        "cleanup",
        "pid",
        "load",
        "invocation",
        "worker_exit",
        "unit_exit",
        "active",
        "live_owner",
        "owner_invocation",
    ],
)
def test_input_pass_is_not_final_stopped_without_actual_exit_and_cleanup(
    change,
):
    cfg = dict(
        mode="main19",
        owner_uid=os.getuid(),
        service_unit="drone-pretrain-test.service",
    )
    state = dict(
        status="MAIN19_INPUTS_PASSED_TERMINATING",
        invocation_id="a" * 32,
        pid=123,
        pid_start_ticks="456",
    )
    owner = dict(
        owner_uid=os.getuid(),
        pid=123,
        pid_start_ticks="456",
        invocation_id="a" * 32,
        unit=cfg["service_unit"],
        cgroup="0::/user.slice/" + cfg["service_unit"],
    )
    proof = dict(
        passed=True,
        owner_uid=os.getuid(),
        service_result="success",
        worker_exit_code="exited",
        worker_exit_status="0",
        invocation_id="a" * 32,
    )
    properties = dict(
        LoadState="loaded",
        MainPID="0",
        ActiveState="inactive",
        SubState="dead",
        Result="success",
        ExecMainCode="1",
        ExecMainStatus="0",
        InvocationID="a" * 32,
    )
    if change == "gc":
        properties.update(
            LoadState="not-found",
            InvocationID="",
            ExecMainCode="0",
            Result="success",
        )
    elif change == "cleanup":
        proof["passed"] = False
    elif change == "pid":
        properties["MainPID"] = "10"
    elif change == "load":
        properties["LoadState"] = "unknown"
    elif change == "invocation":
        properties["InvocationID"] = "b" * 32
    elif change == "worker_exit":
        proof["worker_exit_code"] = "killed"
    elif change == "unit_exit":
        properties["ExecMainStatus"] = "1"
    elif change == "active":
        properties["ActiveState"] = "deactivating"
    elif change == "owner_invocation":
        owner["invocation_id"] = "b" * 32
    assert main.termination_verified(
        cfg,
        state,
        proof,
        properties,
        owner,
        processes_released=change != "live_owner",
    ) is (change in {None, "gc"})


@pytest.mark.parametrize(
    "broken", [None, "worker_exit", "owner_hash", "submission"]
)
def test_collected_unit_gc_requires_bound_real_exit_and_released_owner(
    monkeypatch, tmp_path, broken
):
    task = tmp_path / "task"
    unit, inv = "drone-pretrain-task.service", "c" * 32
    cfg = dict(mode="main19", owner_uid=os.getuid(), service_unit=unit)
    common.write(task / "config.json", cfg)
    owner = dict(
        task=str(task),
        unit=unit,
        owner_uid=os.getuid(),
        config_sha256=common.sha(task / "config.json"),
        invocation_id=inv,
        pid=999,
        pid_start_ticks="123",
        cgroup="0::/user.slice/" + unit,
    )
    owner_path = task / "operator/invocations" / inv / "worker-identity.json"
    common.write(owner_path, owner)
    cleanup = task / "operator/socket-cleanup.json"
    common.write(cleanup, dict(status="CLEANUP_COMPLETE"))
    common.write(
        task / "operator/service.json",
        dict(
            unit=unit,
            owner_uid=os.getuid(),
            config_sha256=owner["config_sha256"],
        ),
    )
    proof = dict(
        passed=True,
        owner_uid=os.getuid(),
        invocation_id=inv,
        service_result="success",
        worker_exit_code="exited",
        worker_exit_status="0",
        config_sha256=owner["config_sha256"],
        worker_identity=dict(
            path=str(owner_path), sha256=common.sha(owner_path)
        ),
        cleanup_sha256=common.sha(cleanup),
        cleanup_report=dict(path=str(cleanup), sha256=common.sha(cleanup)),
    )
    if broken == "worker_exit":
        proof["worker_exit_status"] = "1"
    elif broken == "owner_hash":
        common.write(owner_path, dict(owner, invocation_id="d" * 32))
    elif broken == "submission":
        common.write(task / "operator/service.json", dict(unit="other.service"))
    common.write(task / "operator/cleanup-exit.json", proof)
    common.write(
        task / "operator/state.json",
        dict(
            status="MAIN19_INPUTS_PASSED_TERMINATING",
            invocation_id=inv,
            pid=999,
            pid_start_ticks="123",
        ),
    )
    common.write(task / "completion.json", {})
    monkeypatch.setattr(
        pilot45,
        "service_identity",
        lambda _: (
            unit,
            0,
            dict(
                LoadState="not-found",
                MainPID="0",
                ActiveState="inactive",
                SubState="dead",
                Result="success",
            ),
        ),
    )
    monkeypatch.setattr(main, "worker_processes_released", lambda _: True)
    monkeypatch.setattr(pilot_parallel, "released", lambda: None)
    assert main.reconcile_termination(task)["status"] == (
        "MAIN19_INPUTS_PASSED_STOPPED"
        if broken is None
        else "MAIN19_INPUTS_PASSED_TERMINATING"
    )


@pytest.mark.parametrize("linked", ["freeze", "source", "alias"])
def test_same_bytes_symlink_cannot_be_a_main_handoff_input(
    monkeypatch, tmp_path, linked
):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    source = tmp_path / "ai/source.py"
    source.parent.mkdir()
    source.write_text("# fixture\n", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    common.write(freeze, dict(files={"ai/source.py": common.sha(source)}))
    alias = tmp_path / "alias.json"
    common.write(alias, {})
    handoff = tmp_path / "handoff.json"
    value = dict(
        schema="drone-ai.collection19.data-handoff/1",
        status="READY",
        code_frozen=True,
        source_freeze=dict(path=str(freeze), sha256=common.sha(freeze)),
        paths=dict(native_policy=str(alias)),
        sha256=dict(native_policy=common.sha(alias)),
    )
    target = {"freeze": freeze, "source": source, "alias": alias}[linked]
    original = target.with_name(target.name + ".original")
    target.rename(original)
    target.symlink_to(original)
    common.write(handoff, value)
    with pytest.raises(ValueError, match="Symlink"):
        main.dependencies(handoff)


def test_checkpoint_reuses_main_and_rejects_changed_bytes(
    tmp_path,
):
    task = tmp_path / "task"
    cfg = dict(external_storage=str(tmp_path / "store"), policy_bundle={})
    common.write(task / "config.json", cfg)
    common.write(task / "expected-producer.json", {})
    common.write(task / "global-protection.json", {})
    index = task / "exports.jsonl"
    index.write_text("fixture\n", encoding="utf-8")
    calls = []

    def aggregate(index, output, *, scope, **_):
        calls.append(scope)
        output.mkdir(parents=True)
        if scope == "lr108" and calls.count(scope) == 1:
            raise RuntimeError("preserved LR failure")
        result = dict(
            passed=True,
            actual_data_verified=True,
            scope=scope,
            counts=dict(planned_cases=1674 if scope == "main" else 108),
            index=dict(path=str(index), sha256=common.sha(index)),
        )
        for key in ("independent_audit", "receipt"):
            p = output / (key + ".json")
            common.write(p, result)
            result[key] = dict(path=str(p), sha256=common.sha(p))
        (output / "array.npz").write_bytes(b"fixture-array")
        return result

    job = dict(index=str(index), aggregate_attempt=1)
    first = main.aggregate_scope(task, cfg, "main", job, aggregate)
    with pytest.raises(RuntimeError):
        main.aggregate_scope(task, cfg, "lr108", job, aggregate)
    job["aggregate_attempt"] = 2
    assert (
        main.aggregate_scope(task, cfg, "main", job, aggregate)["reused"]
        is True
    )
    main.aggregate_scope(task, cfg, "lr108", job, aggregate)
    assert calls == ["main", "lr108", "lr108"]
    (Path(first["output"]) / "array.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        main.aggregate_scope(task, cfg, "main", job, aggregate)


def test_main_status_reuses_eight_by_two_grid_and_actual_progress_keys(
    monkeypatch, tmp_path
):
    cfg = dict(scope="main", planned_cases=1674)
    common.write(tmp_path / "config.json", cfg)
    state = dict(
        status="RUNNING",
        pid=123,
        pid_start_ticks="x",
        active=[
            dict(
                instance=i,
                action="flight",
                observed_simulation_elapsed_s=123,
                producer_target_s=480,
                live_rtf=1.5,
            )
            for i in range(16)
        ],
        aggregate_progress=[
            dict(
                dataset_scope="main",
                stage="train_statistics",
                done=4,
                total=540,
            )
        ],
    )
    monkeypatch.setattr(main, "identity", lambda _: "x")
    monkeypatch.setattr(main, "reconcile_termination", lambda _: state)
    text = main.render(tmp_path)
    assert text.count("123/480s") == 16 and text.count("1.50x") == 16
    assert len([x for x in text.splitlines() if x.startswith("│")]) == 8
    assert "train_statistics · 4/540" in text and '{"' not in text


def test_detailed_screen_keeps_all_lanes_and_honest_stage_progress(
    monkeypatch, tmp_path
):
    from pretrain.display import screen

    common.write(
        tmp_path / "config.json", dict(scope="main", planned_cases=1674)
    )
    state = dict(
        status="RUNNING",
        pid=123,
        pid_start_ticks="x",
        started=100,
        done=16,
        physical_attempts=32,
        completed_physical_attempts=32,
        active_flights=16,
        active_nonflight=0,
        batch=2,
        batches=105,
        stage="collection",
        resources=dict(
            available_bytes=12e9,
            owned_tree_rss_bytes=2e9,
            disk_free_bytes=450e9,
        ),
        active=[
            dict(
                instance=i,
                action="flight",
                kind="hover",
                attempt=1,
                observed_simulation_elapsed_s=120,
                producer_target_s=480,
                live_rtf=2,
            )
            for i in range(16)
        ],
    )
    monkeypatch.setattr(main, "identity", lambda _: "x")
    monkeypatch.setattr(main, "reconcile_termination", lambda _: state)
    monkeypatch.setattr(main.time, "time", lambda: 400)
    text = main.render(tmp_path, tick=7)
    for expected in (
        "화면 #7",
        "경과 00:05:00",
        "16/1674",
        "32/32",
        "배치 2/105",
        "00:03:00",
        "가용 RAM 12.0GB",
        "RSS 2.0GB",
        "450.0GB",
        "단계100% ≠ 전체완료",
    ):
        assert expected in text
    fitted = screen(text, columns=100, rows=24)
    assert len(fitted.splitlines()) <= 24
    assert fitted.count("120/480s") == 16 and "화면 #7" in fitted
    assert "q/Ctrl+C" in fitted


def test_stopped_screen_freezes_elapsed_and_does_not_claim_live_work(
    monkeypatch, tmp_path
):
    common.write(
        tmp_path / "config.json", dict(scope="main", planned_cases=1674)
    )
    state = dict(
        status="BLOCKED_STOPPED",
        started=100,
        finished_utc="1970-01-01T00:05:00+00:00",
        active_flights=0,
        active_nonflight=1,
        done=0,
        reason="fixture prepare failure",
    )
    monkeypatch.setattr(main, "reconcile_termination", lambda _: state)
    text = main.render(tmp_path)
    assert "경과 00:03:20" in text and "전체 미완료 · 중지" in text
    assert "마지막 기록 비행0/16 처리1/4" in text
    assert "잔여 예상 미확인" in text and "fixture prepare failure" in text
    assert "전체 검산·실제 종료 확인" not in text


def test_unknown_worker_keeps_screen_available_without_child_log(
    monkeypatch, tmp_path
):
    common.write(
        tmp_path / "config.json", dict(scope="main", planned_cases=1674)
    )
    state = dict(
        status="RUNNING",
        pid=123,
        pid_start_ticks="old",
        active=[dict(instance=0)],
    )
    monkeypatch.setattr(main, "identity", lambda _: None)
    monkeypatch.setattr(main, "reconcile_termination", lambda _: state)
    text = main.render(tmp_path)
    assert "UNKNOWN_NOT_RUNNING" in text and "전체 미완료 · 중지" in text


def test_stopped_screen_shows_recorded_child_cause_not_only_wrapper(
    monkeypatch, tmp_path
):
    common.write(
        tmp_path / "config.json", dict(scope="main", planned_cases=1674)
    )
    log = tmp_path / "pilot/execution/case/prepare-01.log"
    common.write(
        log.with_suffix(".request.failure.json"),
        dict(
            error_type="FileNotFoundError",
            failure_reason="archive-evidence parent missing",
        ),
    )
    state = dict(
        status="BLOCKED_STOPPED",
        reason="Case child exit1",
        active=[dict(log=str(log))],
    )
    monkeypatch.setattr(main, "reconcile_termination", lambda _: state)
    text = main.render(tmp_path)
    assert "FileNotFoundError: archive-evidence parent missing" in text
    assert str(log) in text


@pytest.mark.parametrize(
    "command,immediate", [("stop", False), ("cancel", True)]
)
def test_main_stop_and_cancel_address_only_the_requested_owned_task(
    monkeypatch, tmp_path, command, immediate
):
    calls = []
    monkeypatch.setattr(
        pilot45, "stop", lambda task, **kw: calls.append((task, kw))
    )
    main.main([command, str(tmp_path)])
    assert calls == [(tmp_path, dict(immediate=immediate))]


@pytest.mark.local_evidence
def test_explicit_resume_recovers_known_completed_raw_without_a_new_flight(
    monkeypatch, tmp_path
):
    task, rows, cfg = runtime(monkeypatch, tmp_path, count=1)
    case = rows[0]
    monkeypatch.setattr(pilot45, "task_path", Path)
    monkeypatch.setattr(
        pilot45,
        "service_identity",
        lambda _: (cfg["service_unit"], 0, dict(ActiveState="inactive")),
    )
    monkeypatch.setattr(pilot_parallel, "released", lambda: None)
    monkeypatch.setattr(main.subprocess, "run", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        cli, "start", lambda selected, **kw: calls.append((selected, kw))
    )
    monkeypatch.setattr(
        pilot45,
        "child",
        lambda *a, **k: pytest.fail(
            "Resume reconciliation must not dispatch a flight"
        ),
    )
    common.write(task / "operator/state.json", dict(status="USER_STOPPED"))
    raw = tmp_path / "simulation/runs" / ("e" * 32)
    common.write(
        raw / "run.json",
        dict(
            run_id=raw.name,
            status="completed",
            instance=0,
            parent_attempt_id=None,
            owned_process_groups={},
        ),
    )
    common.write(
        raw / "evaluation/scenario.json",
        dict(
            name=case["scenario"],
            seed=case["seed"],
            operation_variation=case["operation_variation"],
        ),
    )
    record = task / "cases" / (case["case_id"] + ".json")
    common.write(
        record,
        dict(
            case=case,
            attempts=[
                dict(
                    number=1,
                    state="RUNNING",
                    instance=0,
                    parent_attempt_id=None,
                    execution_task=str(task),
                )
            ],
        ),
    )
    folder = task / "pilot/execution" / case["case_id"]
    receipt = folder / "known-completed.json"
    common.write(receipt, dict(status="completed", run=str(raw)))
    common.write(
        folder / "flight-01.request.json", dict(case=case, receipt=str(receipt))
    )
    main.resume(task)
    saved = common.read(record)
    assert saved["run"] == str(raw) and len(saved["attempts"]) == 1
    assert saved["attempts"][0]["state"] == "COMPLETED"
    assert calls == [(task, dict(pilot_resume=True))]
    assert common.read(task / "operator/resume-request.json")[
        "config_sha256"
    ] == common.sha(task / "config.json")


def test_budget_separates_estimates_from_measurements():
    value = main.storage_budget(1674)
    assert value["raw_all_cases_extrapolation_bytes"] > 1_000_000_000_000
    assert (
        value["permanent_point_estimate_bytes"]
        < value["permanent_planning_upper_bytes"]
    )
    assert (
        value["required_initial_free_bytes"]
        == value["peak_growth_planning_bytes"] + 50_000_000_000
    )
    assert value["measured_basis"]["runtime_requires_prior_tasks"] is False
