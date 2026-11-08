"""Scheduler fixtures exercise concurrency/ownership boundaries, never native flight."""

import hashlib
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from pretrain import common, native_flight, pilot45
from pretrain import pilot_parallel as parallel


def fixture(monkeypatch, tmp_path):
    cases = pilot45.build_cases(98765)
    task = tmp_path / "task"
    cfg = dict(
        parallel16=True,
        flight_workers=16,
        requested_real_time_factor=2,
        cpu_workers_total_max=4,
        planned_cases=45,
        training_authorized=False,
        fit_authorized=False,
        continuation=dict(previous_task="previous"),
        lineage_namespace="previous",
        service_unit="drone-pretrain-task.service",
        source_sha256={},
    )
    monkeypatch.setattr(
        parallel, "authorization", lambda: dict(previous_pilot="previous")
    )
    monkeypatch.setattr(parallel, "ROOT", tmp_path)
    monkeypatch.setattr(parallel, "STORE", tmp_path)
    monkeypatch.setattr(parallel, "check_ports", lambda *args: None)
    monkeypatch.setattr(pilot45, "resources", lambda *args: {})
    monkeypatch.setattr(
        pilot45,
        "collector",
        lambda: SimpleNamespace(check_instance_socket=lambda i: None),
    )
    monkeypatch.setattr(pilot45, "register", lambda *args: None)
    entries = {c["case_id"]: dict(case=c, attempts=[]) for c in cases}
    for case in cases[:3]:
        e = entries[case["case_id"]]
        e.update(
            status="COMPLETE",
            run=str(tmp_path / "inherited" / case["case_id"]),
            inherited_from="previous",
            attempts=[
                dict(
                    number=1,
                    state="COMPLETED",
                    parent_attempt_id=None,
                    run_id=case["case_id"],
                )
            ],
        )
    common.write(task / "config.json", cfg)
    common.write(task / "manifest.json", dict(cases=cases))
    common.write(
        task / "ledger.json",
        dict(
            cases=entries,
            attempts=[
                dict(
                    case_id=c["case_id"], **entries[c["case_id"]]["attempts"][0]
                )
                for c in cases[:3]
            ],
        ),
    )
    common.write(task / "protection.json", dict(records=[]))
    return task, cases


@pytest.mark.parametrize(
    "key,value",
    [
        ("flight_workers", 1),
        ("flight_workers", 4),
        ("requested_real_time_factor", 1),
        ("cpu_workers_total_max", 16),
        ("planned_cases", 44),
        ("fit_authorized", True),
    ],
)
def test_no_silent_downgrade_or_scope_expansion(
    monkeypatch, tmp_path, key, value
):
    task, _ = fixture(monkeypatch, tmp_path)
    cfg = common.read(task / "config.json")
    cfg[key] = value
    with pytest.raises(pilot45.Blocked):
        parallel.validate_config(cfg)


def test_real_native_argv_has16_distinct_instances_and_requested_rtf2():
    case = pilot45.build_cases(1)[0]
    argv = [
        native_flight.command(
            case, "parent", ["--paired-case-id", "{paired_case_id}"], i
        )
        for i in range(16)
    ]
    assert [int(a[a.index("--instance") + 1]) for a in argv] == list(range(16))
    assert all(a[a.index("--real-time-factor") + 1] == "2" for a in argv)
    assert all(a[a.index("--parent-attempt-id") + 1] == "parent" for a in argv)
    assert len({p for i in range(16) for p in parallel.ports(i)}) == 80


@pytest.mark.parametrize("instance", [-1, 16, 1.0, True])
def test_invalid_instance_is_rejected(instance):
    with pytest.raises(ValueError):
        native_flight.command(pilot45.build_cases(1)[0], instance=instance)


def test_pipeline_actually_dispatches16_and_one_cpu4_pool_preserves3_successes(
    monkeypatch, tmp_path
):
    task, cases = fixture(monkeypatch, tmp_path)
    owner = threading.get_ident()
    writes, registry, launches, actions = [], [], [], []
    lock, barrier = threading.Lock(), threading.Barrier(16)
    active = dict(flight=0, cpu=0)
    peak = dict(flight=0, cpu=0)
    original_write = common.write

    def write(path, value):
        if Path(path).name in {"ledger.json", "state.json"}:
            writes.append(threading.get_ident())
        original_write(path, value)

    monkeypatch.setattr(parallel, "write", write)
    monkeypatch.setattr(
        pilot45,
        "register",
        lambda *args: registry.append(threading.get_ident()),
    )

    def child(task, request, log, *, on_progress=None):
        job = common.read(request)
        kind = "flight" if job["action"] == "flight" else "cpu"
        with lock:
            active[kind] += 1
            peak[kind] = max(peak[kind], active[kind])
            if kind == "flight":
                launches.append((job["case"]["case_id"], job["instance"]))
                first_wave = len(launches) <= 16
            else:
                actions.append(job["action"])
        try:
            if kind == "flight":
                if first_wave:
                    barrier.wait(timeout=10)
                case = job["case"]
                rid = hashlib.md5(
                    case["case_id"].encode(), usedforsecurity=False
                ).hexdigest()
                raw = tmp_path / "simulation/runs" / rid
                common.write(
                    raw / "run.json",
                    dict(
                        status="completed",
                        instance=job["instance"],
                        parent_attempt_id=job["parent_attempt_id"],
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
                common.write(
                    job["receipt"], dict(status="completed", run=str(raw))
                )
                if on_progress:
                    on_progress(
                        dict(
                            case_id=case["case_id"],
                            instance=job["instance"],
                            action="flight",
                            live_rtf=None,
                        )
                    )
                time.sleep(0.03)
            else:
                time.sleep(0.015)
                if job["action"] == "prepare":
                    output = Path(job["output"])
                    common.write(
                        output / "receipt.json", dict(fixture_only=True)
                    )
                    common.write(output / "seal.json", dict(fixture_only=True))
                    result = dict(
                        files={},
                        run=dict(raw_sha256="fixture"),
                        online=dict(available=True),
                        run_health=dict(status="available"),
                    )
                elif job["action"] == "audit":
                    result = dict(
                        export_receipt_sha256=common.sha(
                            Path(job["output"]) / "receipt.json"
                        )
                    )
                elif job["action"] == "pair":
                    result = dict(
                        pair_receipt="fixture-pair:" + job["case"]["group_id"]
                    )
                else:
                    result = dict(fixture_only=True)
                common.write(job["receipt"], result)
        finally:
            with lock:
                active[kind] -= 1

    monkeypatch.setattr(pilot45, "child", child)
    coordinator = parallel.Coordinator(task)
    inherited = {
        k: e.copy()
        for k, e in coordinator.ledger["cases"].items()
        if e.get("status") == "COMPLETE"
    }
    coordinator.run()
    assert len(launches) == len({c for c, i in launches}) == 42
    assert {i for c, i in launches[:16]} == set(range(16))
    assert peak == dict(flight=16, cpu=4)
    assert set(writes + registry) == {owner}
    assert Counter(actions) == dict(
        prepare=42, audit=42, pair=9, ai_case=45, cohort_audit=1
    )
    assert (
        len(coordinator.ledger["cases"])
        == len(coordinator.ledger["attempts"])
        == 45
    )
    assert all(
        e["status"] == "COMPLETE" and e.get("ai_receipt")
        for e in coordinator.ledger["cases"].values()
    )
    assert (
        coordinator.state["max_active_flights"] == 16
        and coordinator.state["max_active_nonflight"] == 4
    )
    for cid, old in inherited.items():
        assert all(
            coordinator.ledger["cases"][cid][k] == v for k, v in old.items()
        )


def test_complete_success_cannot_be_reflown(monkeypatch, tmp_path):
    task, cases = fixture(monkeypatch, tmp_path)
    c = parallel.Coordinator(task)
    with pytest.raises(pilot45.Blocked, match="regenerated"):
        c.begin_flight(cases[0], 0)


@pytest.mark.parametrize(
    "prior",
    [
        dict(
            number=1,
            state="FAILED_PRESERVED",
            retry_allowed=False,
            run_id="failed",
        ),
        dict(number=1, state="FAILED_PRESERVED", retry_allowed=True),
        dict(
            number=3,
            state="FAILED_PRESERVED",
            retry_allowed=True,
            run_id="failed",
        ),
    ],
)
def test_retry_policy_keeps_parent_and_cap(monkeypatch, tmp_path, prior):
    task, cases = fixture(monkeypatch, tmp_path)
    c = parallel.Coordinator(task)
    e = c.ledger["cases"][cases[3]["case_id"]]
    e["attempts"] = [dict(prior, number=i + 1) for i in range(prior["number"])]
    with pytest.raises(pilot45.Blocked, match="retry"):
        c.begin_flight(cases[3], 0)


def test_retry_reuses_case_seed_and_failed_native_parent(monkeypatch, tmp_path):
    task, cases = fixture(monkeypatch, tmp_path)
    c = parallel.Coordinator(task)
    e = c.ledger["cases"][cases[3]["case_id"]]
    e["attempts"] = [
        dict(
            number=1,
            state="FAILED_PRESERVED",
            retry_allowed=True,
            run_id="failed",
            parent_attempt_id=None,
        )
    ]
    j = c.begin_flight(cases[3], 15)
    r = common.read(j["request"])
    assert (
        j["number"] == 2
        and r["parent_attempt_id"] == "failed"
        and r["case"] == cases[3]
        and r["instance"] == 15
    )


def test_full45_denominator_cannot_drop_missing_case(monkeypatch, tmp_path):
    task, cases = fixture(monkeypatch, tmp_path)
    ledger = common.read(task / "ledger.json")
    del ledger["cases"][cases[-1]["case_id"]]
    common.write(task / "ledger.json", ledger)
    with pytest.raises(pilot45.Blocked, match="denominator"):
        parallel.cohort_audit(task)


def result_for(job, tmp_path):
    case = job["case"]
    raw = tmp_path / "simulation/runs" / ("a" * 30 + f"{job['instance']:02}")
    common.write(
        raw / "run.json",
        dict(
            status="completed", instance=job["instance"], parent_attempt_id=None
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
    common.write(
        common.read(job["request"])["receipt"],
        dict(status="completed", run=str(raw)),
    )


def test_out_of_order_completion_updates_case_and_attempt_instead_of_last_row(
    monkeypatch, tmp_path
):
    task, cases = fixture(monkeypatch, tmp_path)
    c = parallel.Coordinator(task)
    first, second = c.begin_flight(cases[3], 0), c.begin_flight(cases[4], 1)
    result_for(first, tmp_path)
    result_for(second, tmp_path)
    assert c.finish_flight(first, None)
    assert c.ledger["attempts"][-2]["state"] == "COMPLETED"
    assert c.ledger["attempts"][-1]["state"] == "RUNNING"
    assert c.finish_flight(second, None)
    assert c.ledger["attempts"][-1]["state"] == "COMPLETED"
    assert (
        c.ledger["attempts"][-2]["run_id"] != c.ledger["attempts"][-1]["run_id"]
    )


def test_resource_block_never_silently_runs_fewer_lanes_or_discards_denominator(
    monkeypatch, tmp_path
):
    task, _ = fixture(monkeypatch, tmp_path)
    c = parallel.Coordinator(task)

    def blocked(*args):
        raise pilot45.Blocked("Available RAM below1GB")

    monkeypatch.setattr(pilot45, "resources", blocked)
    monkeypatch.setattr(
        pilot45,
        "child",
        lambda *a, **kw: pytest.fail("No flight on blocked resources"),
    )
    with pytest.raises(pilot45.Blocked, match="RAM"):
        c.run()
    assert len(c.ledger["cases"]) == 45 and len(c.ledger["attempts"]) == 3


@pytest.mark.local_evidence
def test_actual_child_processes_have_separate_records_and_do_not_write_shared_state(
    monkeypatch, tmp_path
):
    import json

    task = tmp_path / "task"
    (tmp_path / "simulation/runs").mkdir(parents=True)
    common.write(task / "config.json", dict(parallel16=True, source_sha256={}))
    common.write(
        task / "operator/state.json", dict(coordinator_only="untouched")
    )
    monkeypatch.setattr(pilot45, "ROOT", tmp_path)
    monkeypatch.setattr(native_flight, "ROOT", tmp_path)
    monkeypatch.setattr(
        pilot45,
        "resources",
        lambda: dict(
            available_bytes=2_000_000_000,
            disk_free_bytes=100_000_000_000,
            owned_tree_rss_bytes=1,
        ),
    )
    lanes = []

    def progress(*args, **kwargs):
        lanes.append(args[3]["instance"])
        return SimpleNamespace(
            matches={}, sample=lambda: {}, terminal_evidence=lambda: {}
        )

    monkeypatch.setattr(
        pilot45, "collector", lambda: SimpleNamespace(MissionProgress=progress)
    )

    def command(case, parent, options, instance):
        raw = tmp_path / "simulation/runs" / str(instance)
        raw.mkdir(parents=True, exist_ok=True)
        result = dict(status="completed", run=str(raw))
        program = (
            "import time;time.sleep(.15);print("
            + repr(json.dumps(result))
            + ")"
        )
        return [
            str(common.ROOT / "simulation/.venv/bin/python"),
            "-B",
            "-c",
            program,
        ]

    monkeypatch.setattr(native_flight, "command", command)
    requests = []
    for instance in (0, 15):
        case = pilot45.build_cases(1)[instance]
        folder = task / case["case_id"]
        request = folder / "flight-01.request.json"
        common.write(
            request,
            dict(
                case=case,
                action="flight",
                instance=instance,
                receipt=str(folder / "receipt.json"),
            ),
        )
        requests.append(request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda p: pilot45.child(
                    task, p, p.parent / "flight.log", on_progress=lambda e: None
                ),
                requests,
            )
        )
    assert set(lanes) == {0, 15} and {r["instance"] for r in results} == {0, 15}
    assert len({r["pid"] for r in results}) == 2
    assert all(r["exit_code"] == 0 for r in results)
    assert common.read(task / "operator/state.json") == dict(
        coordinator_only="untouched"
    )
    assert not (task / "operator/child.json").exists()
    assert all(p.with_suffix(".process.json").exists() for p in requests)


@pytest.mark.parametrize("fault", [None, "current_hash", "other_file"])
def test_prior_termination_fix_requires_exact_historical_acceptance(
    monkeypatch, tmp_path, fault
):
    monkeypatch.setattr(parallel, "ROOT", tmp_path)
    p = (
        tmp_path
        / "Data/generation/pretrain"
        / ("ai19.py" if fault == "other_file" else "pilot_flow.py")
    )
    p.parent.mkdir(parents=True)
    p.write_text("reviewed control fix\n", encoding="utf-8")
    task = tmp_path / "old-integration"
    old = "0" * 64
    common.write(task / "config.json", dict(source_sha256={str(p): old}))
    receipt = tmp_path / "acceptance.json"
    common.write(
        receipt,
        dict(
            task=str(task),
            status="ACCEPTED_EXISTING_INTEGRATION_NO_FLIGHT",
            evidence_sha256={},
            source_compatibility=dict(
                reviewed_control_changes={
                    str(p): dict(
                        old_sha256=old,
                        current_sha256="1" * 64
                        if fault == "current_hash"
                        else common.sha(p),
                    )
                }
            ),
        ),
    )
    if fault:
        with pytest.raises(pilot45.Blocked, match="processing changed"):
            parallel.compatible_sources(task, receipt)
    else:
        assert parallel.compatible_sources(task, receipt)[str(p)][
            "current_sha256"
        ] == common.sha(p)
