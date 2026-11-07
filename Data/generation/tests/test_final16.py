"""FINAL16 admission and finite scheduler checks; no native process is started."""

import copy
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from pretrain import common, final16, pilot45, pilot_bridge, pilot_parallel


def fixture(tmp_path):
    task = tmp_path / "final16-isolated-16"
    rows = final16.cases()
    cfg = dict(
        final16=True,
        parallel16=True,
        task_id=final16.TASK,
        scope_id=final16.TASK,
        planned_cases=16,
        mission_seconds=180,
        flight_workers=16,
        requested_real_time_factor=2,
        cpu_workers_total_max=4,
        unit_attempts_max=1,
        stop_on_first_physical_failure=True,
        nonflight_when_flights_idle=True,
        training_authorized=False,
        fit_authorized=False,
        lineage_namespace="final16-isolated",
        authorization=dict(path=str(final16.AUTH), sha256=final16.AUTH_SHA),
        service_unit="drone-pretrain-final16-isolated-16.service",
    )
    manifest = dict(
        schema="drone-data.final16.manifest/1",
        task_id=final16.TASK,
        final16=True,
        short_cycle=False,
        single_integration=False,
        mission_seconds=180,
        seed_base=2938306477,
        authorization=cfg["authorization"],
        cases=rows,
    )
    ledger = dict(
        schema="drone-data.final16.ledger/1",
        task_id=final16.TASK,
        cases={c["case_id"]: dict(case=c, attempts=[]) for c in rows},
        attempts=[],
    )
    for name, value in (
        ("config.json", cfg),
        ("manifest.json", manifest),
        ("ledger.json", ledger),
        ("protection.json", dict(records=[])),
    ):
        common.write(task / name, value)
    return task, rows


@pytest.mark.local_evidence
def test_actual_original_eight_pairs_match_coordinator_projection(tmp_path):
    task, rows = fixture(tmp_path)
    expected = common.read(
        final16.EXPECTED.with_name("expected-manifest-cases.json")
    )
    assert rows == expected["cases"]
    assert len(rows) == 16 and len({c["paired_case_id"] for c in rows}) == 8
    assert all(c["attempts_max"] == 1 for c in rows)
    final16.validate_ledger(task, pristine=True)
    plan = pilot_bridge.plan(task)
    assert plan["schema"] == "drone-ai.final16.manifest/1"
    assert plan["cohort"] == plan["purpose"] == final16.COHORT
    assert plan["no_fit"] is True and len(plan["cases"]) == 16
    from learning.history19.final16 import expected_plan

    assert plan == expected_plan("final16-isolated")
    with pytest.raises(ValueError, match="45"):
        pilot45.validate_cases(rows, duration=180)


@pytest.mark.local_evidence
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "attempt2",
        "physics",
        "cohort",
        "seed",
        "short",
        "reuse",
    ],
)
def test_strict_manifest_rejects_changes(tmp_path, mutation):
    task, _ = fixture(tmp_path)
    m = common.read(task / "manifest.json")
    if mutation == "missing":
        m["cases"].pop()
    elif mutation == "extra":
        m["cases"].append(copy.deepcopy(m["cases"][0]))
    elif mutation == "attempt2":
        m["cases"][0]["attempts_max"] = 2
    elif mutation == "physics":
        m["cases"][0]["operation_variation"]["soc"] = 0.8
    elif mutation == "cohort":
        m["cases"][0]["cohort"] = "short_cycle_development_reference"
    elif mutation == "seed":
        m["seed_base"] += 1
    elif mutation == "short":
        m["short_cycle"] = True
    elif mutation == "reuse":
        m["recovery_of"] = "old45"
    with pytest.raises(pilot45.Blocked):
        pilot45.validate_manifest(m)


@pytest.mark.local_evidence
@pytest.mark.parametrize(
    "key,value",
    [
        ("planned_cases", 45),
        ("flight_workers", 8),
        ("requested_real_time_factor", 1),
        ("cpu_workers_total_max", 16),
        ("unit_attempts_max", 3),
        ("stop_on_first_physical_failure", False),
        ("nonflight_when_flights_idle", False),
        ("fit_authorized", True),
        ("training_authorized", True),
        ("final16", "yes"),
        ("short_cycle", True),
        ("short_resume", {"previous": "old45"}),
    ],
)
def test_strict_config_rejects_scope_expansion(tmp_path, key, value):
    task, _ = fixture(tmp_path)
    cfg = common.read(task / "config.json")
    cfg[key] = value
    with pytest.raises(pilot45.Blocked):
        pilot_parallel.validate_config(cfg)


@pytest.mark.local_evidence
@pytest.mark.parametrize(
    "bad", ["inherited", "parent", "source", "second", "admin", "raw_duplicate"]
)
def test_ledger_rejects_reuse_retry_and_unknown_admin(tmp_path, bad):
    task, rows = fixture(tmp_path)
    ledger = common.read(task / "ledger.json")
    first = ledger["cases"][rows[0]["case_id"]]
    a = dict(number=1, state="RUNNING", parent_attempt_id=None, run_id="a" * 32)
    first["attempts"] = [a]
    if bad == "inherited":
        first["inherited_from"] = "old45"
    elif bad == "parent":
        a["parent_attempt_id"] = "b" * 32
    elif bad == "source":
        a["source_task"] = "old45"
    elif bad == "second":
        first["attempts"].append(dict(a, number=2))
    elif bad == "admin":
        ledger["administrative_attempts"] = [dict(state="RUNNING")]
    else:
        ledger["cases"][rows[1]["case_id"]]["attempts"] = [dict(a)]
    ledger["attempts"] = [
        dict(case_id=cid, **a)
        for cid, e in ledger["cases"].items()
        for a in e["attempts"]
    ]
    common.write(task / "ledger.json", ledger)
    with pytest.raises(pilot45.Blocked):
        final16.validate_ledger(task)


@pytest.mark.local_evidence
def test_all16_real_registration_boundaries_in_isolated_registry(
    monkeypatch, tmp_path
):
    task, rows = fixture(tmp_path)
    registry = tmp_path / "registry.jsonl"
    # Replay the recorded pre-FINAL16 boundary after the actual authorized cycle completed.
    registry.write_text(
        "".join(
            line
            for line in pilot45.REGISTRY.read_text().splitlines(keepends=True)
            if json.loads(line)["task"] != "final16-20261107-01-16"
        )
    )
    monkeypatch.setattr(pilot45, "REGISTRY", registry)
    c = pilot_parallel.Coordinator(task)
    for i, case in enumerate(rows):
        c.begin_flight(case, i)
    final16.validate_ledger(task)
    events = [
        json.loads(line)
        for line in registry.read_text().splitlines()
        if json.loads(line)["task"] == task.name
    ]
    assert len(events) == 16 and len(c.ledger["attempts"]) == 16
    assert {r["attempt"] for r in events} == {1}
    assert all(r["parent_attempt_id"] is None for r in events)
    assert len({r["retry_family_id"] for r in events}) == 16
    assert not list(task.rglob("*.process.json"))
    with pytest.raises(pilot45.Blocked):
        c.begin_flight(rows[0], 0)
    with pytest.raises(pilot45.Blocked):
        c.begin_flight(dict(rows[0], case_id="third"), 0)


@pytest.mark.local_evidence
@pytest.mark.parametrize("wrong", ["native", "adapter", "both"])
def test_native_revision_must_be_v2_even_if_both_wrong_values_match(
    monkeypatch, tmp_path, wrong
):
    """The original chained comparison missed two equal wrong revisions."""
    spec = common.read(
        common.ROOT / "ai/reports/final16-20261107/attempt-01/data-handoff.json"
    )
    from learning.history19.final16 import CRITICAL

    # This negative field test uses a synthetic current-code freeze, not old release READY.
    freeze = tmp_path / "freeze.json"
    common.write(
        freeze,
        dict(
            task_id=final16.TASK,
            code_frozen=True,
            files={p: common.sha(common.ROOT / p) for p in CRITICAL},
        ),
    )
    spec["source_freeze"] = dict(path=str(freeze), sha256=common.sha(freeze))
    native = common.read(spec["paths"]["native_policy"])
    policy = common.read(spec["paths"]["paired_policy"])
    producer = dict(policy["producer"])
    for name, key in (
        ("source_manifest", "source_manifest_sha256"),
        ("runtime_manifest", "runtime_sha256"),
    ):
        spec["sha256"][name] = common.sha(spec["paths"][name])
        producer[key] = spec["sha256"][name]
    expected = tmp_path / "expected-producer.json"
    common.write(expected, producer)
    spec["paths"]["expected_producer"] = str(expected)
    spec["sha256"]["expected_producer"] = common.sha(expected)
    spec["producer"] = policy["producer"] = producer
    fixture_authority = copy.deepcopy(final16.authorization())
    fixture_authority["producer"].update(
        source_sha256=producer["source_manifest_sha256"],
        runtime_sha256=producer["runtime_sha256"],
    )
    monkeypatch.setattr(final16, "authorization", lambda: fixture_authority)
    if wrong in {"native", "both"}:
        native["revision"] = "wrong-equal-revision"
    if wrong in {"adapter", "both"}:
        policy["native_adapter"] = "wrong-equal-revision"
    n, p = tmp_path / "native.json", tmp_path / "paired.json"
    common.write(n, native)
    policy["native_policy_sha256"] = common.sha(n)
    common.write(p, policy)
    spec["paths"].update(native_policy=str(n), paired_policy=str(p))
    spec["sha256"].update(
        native_policy=common.sha(n), paired_policy=common.sha(p)
    )
    spec["paired_cli_options"][1] = native["revision"]
    monkeypatch.setattr(final16.ai19, "producer_binding", lambda: producer)
    with pytest.raises(pilot45.Blocked, match="policy/CLI"):
        final16.dependencies(spec)


@pytest.mark.local_evidence
@pytest.mark.parametrize("fail", [False, True])
def test_exact16_concurrent_then_cpu4_or_failure_drain(
    monkeypatch, tmp_path, fail
):
    task, rows = fixture(tmp_path)
    monkeypatch.setattr(pilot_parallel, "ROOT", tmp_path)
    monkeypatch.setattr(pilot_parallel, "STORE", tmp_path)
    monkeypatch.setattr(pilot_parallel, "check_ports", lambda *a: None)
    monkeypatch.setattr(
        pilot45,
        "collector",
        lambda: SimpleNamespace(check_instance_socket=lambda i: None),
    )
    monkeypatch.setattr(pilot45, "resources", lambda *a: {})
    monkeypatch.setattr(pilot45, "register", lambda *a: None)
    barrier, lock = threading.Barrier(16), threading.Lock()
    active, peak = dict(flight=0, cpu=0), dict(flight=0, cpu=0)
    calls, overlap = [], []

    def child(_task, request, log, *, on_progress=None):
        j = common.read(request)
        kind = "flight" if j["action"] == "flight" else "cpu"
        with lock:
            active[kind] += 1
            peak[kind] = max(peak[kind], active[kind])
            calls.append(j["action"])
            if all(active.values()):
                overlap.append(True)
        try:
            if kind == "flight":
                barrier.wait(timeout=10)
                time.sleep(0.01 + j["instance"] * 0.001)
                case = j["case"]
                rid = hashlib.md5(
                    case["case_id"].encode(), usedforsecurity=False
                ).hexdigest()
                raw = tmp_path / "simulation/runs" / rid
                status = (
                    "failed" if fail and j["instance"] == 0 else "completed"
                )
                common.write(
                    raw / "run.json",
                    dict(
                        status=status,
                        instance=j["instance"],
                        parent_attempt_id=None,
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
                common.write(j["receipt"], dict(status=status, run=str(raw)))
                if status == "failed":
                    raise RuntimeError("fixture physical failure")
            else:
                time.sleep(0.01)
                if j["action"] == "prepare":
                    output = Path(j["output"])
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
                elif j["action"] == "audit":
                    result = dict(
                        export_receipt_sha256=common.sha(
                            Path(j["output"]) / "receipt.json"
                        )
                    )
                elif j["action"] == "pair":
                    result = dict(
                        pair_receipt="fixture:" + j["case"]["group_id"]
                    )
                else:
                    result = dict(fixture_only=True)
                common.write(j["receipt"], result)
        finally:
            with lock:
                active[kind] -= 1

    monkeypatch.setattr(pilot45, "child", child)
    coordinator = pilot_parallel.Coordinator(task)
    if fail:
        with pytest.raises(pilot45.Blocked, match="Physical failure"):
            coordinator.run()
    else:
        coordinator.run()
    assert calls.count("flight") == 16 and peak["flight"] == 16 and not overlap
    assert len(coordinator.ledger["attempts"]) == 16 and not any(
        active.values()
    )
    assert all(
        len(e["attempts"]) == 1 for e in coordinator.ledger["cases"].values()
    )
    if fail:
        assert set(calls) == {"flight"} and peak["cpu"] == 0
        assert (
            sum(
                a["state"] == "FAILED_PRESERVED"
                for a in coordinator.ledger["attempts"]
            )
            == 1
        )
        assert (
            sum(
                bool(e.get("run")) for e in coordinator.ledger["cases"].values()
            )
            == 15
        )
    else:
        assert (
            peak["cpu"] == 4
            and calls.count("prepare") == calls.count("audit") == 16
        )
        assert (
            calls.count("pair") == 8
            and calls.count("ai_case") == 16
            and calls.count("ai_final") == 1
        )
