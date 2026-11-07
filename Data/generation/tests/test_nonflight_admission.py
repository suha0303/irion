"""Actual finite scheduler fixtures:16 flight lanes/one4-slot pool, no native execution."""

import copy
import hashlib
import threading
import time
from pathlib import Path

import pytest
from pretrain import common, pilot45
from pretrain import pilot_parallel as parallel
from test_pilot_parallel import fixture


@pytest.mark.parametrize("idle_only", [False, True])
def test_policy_controls_real_scheduler_overlap_without_reflying_successes(
    monkeypatch, tmp_path, idle_only
):
    task, _ = fixture(monkeypatch, tmp_path)
    cfg = common.read(task / "config.json")
    cfg["nonflight_when_flights_idle"] = idle_only
    common.write(task / "config.json", cfg)
    lock, barrier = threading.Lock(), threading.Barrier(16)
    active, peak = dict(flight=0, cpu=0), dict(flight=0, cpu=0)
    overlap, flights = [], []

    def child(_task, request, log, *, on_progress=None):
        job = common.read(request)
        kind = "flight" if job["action"] == "flight" else "cpu"
        with lock:
            active[kind] += 1
            peak[kind] = max(peak[kind], active[kind])
            if all(active.values()):
                overlap.append((job["action"], dict(active)))
            if kind == "flight":
                flights.append(job["case"]["case_id"])
                first_wave = len(flights) <= 16
        try:
            if kind == "flight":
                if first_wave:
                    barrier.wait(timeout=10)
                time.sleep(0.025 + job["instance"] * 0.001)
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
                result = dict(status="completed", run=str(raw))
            else:
                time.sleep(0.005)
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
                        pair_receipt="fixture:" + job["case"]["group_id"]
                    )
                else:
                    result = dict(fixture_only=True)
            common.write(job["receipt"], result)
        finally:
            with lock:
                active[kind] -= 1

    monkeypatch.setattr(pilot45, "child", child)
    coordinator = parallel.Coordinator(task)
    before = {
        cid: copy.deepcopy(e)
        for cid, e in coordinator.ledger["cases"].items()
        if e.get("status") == "COMPLETE"
    }
    coordinator.run()
    assert len(flights) == len(set(flights)) == 42
    assert not set(flights) & set(before)
    assert peak == dict(flight=16, cpu=4)
    assert bool(overlap) is (not idle_only)
    assert (
        len(coordinator.ledger["cases"])
        == len(coordinator.ledger["attempts"])
        == 45
    )
    assert all(
        e.get("status") == "COMPLETE"
        for e in coordinator.ledger["cases"].values()
    )
    for cid, old in before.items():
        assert all(
            coordinator.ledger["cases"][cid][k] == v for k, v in old.items()
        )


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_admission_policy_rejects_ambiguous_configuration(
    monkeypatch, tmp_path, value
):
    task, _ = fixture(monkeypatch, tmp_path)
    cfg = common.read(task / "config.json")
    cfg["nonflight_when_flights_idle"] = value
    with pytest.raises(pilot45.Blocked, match="boolean"):
        parallel.validate_config(cfg)
