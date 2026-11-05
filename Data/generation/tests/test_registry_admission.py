"""Isolated actual registry/admission boundaries; no native/process submission."""

import copy
import fcntl
import importlib.util
import json
from pathlib import Path

import pytest
from pretrain import common, pilot45, pilot_parallel

REPORT = common.ROOT / "docs/data/reports/short-cycle-20261107/attempt-01/resume-c05"


def initial_queue(monkeypatch, tmp_path):
    previous = common.HOME / "tasks/short180-20261107-c03-45"
    actual = common.HOME / "tasks/short180-20261107-c04-45"
    cfg = common.read(actual / "config.json")
    manifest = common.read(actual / "manifest.json")
    ledger = common.read(previous / "ledger.json")
    for cid, e in ledger["cases"].items():
        for a in e["attempts"]:
            a.setdefault("source_task", str(previous))
        if cid in cfg["short_resume"]["quality_rejections"]:
            e["attempts"][-1]["quality_rejection"] = cfg["short_resume"]["quality_rejections"][cid]
            e.pop("run")
    index = {(cid, a["number"]): a for cid, e in ledger["cases"].items() for a in e["attempts"]}
    ledger["attempts"] = [
        dict(index[a["case_id"], a["number"]], case_id=a["case_id"]) for a in ledger["attempts"]
    ]
    task = tmp_path / actual.name
    for name, value in (
        ("config.json", cfg),
        ("manifest.json", manifest),
        ("ledger.json", ledger),
        ("protection.json", common.read(actual / "protection.json")),
    ):
        common.write(task / name, value)
    # The actual registry now contains later C05 events; replay only its frozen C04 boundary.
    live = REPORT / "registry-at-C04-closure.jsonl"
    with live.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        lines = [row for row in stream if json.loads(row)["task"] != actual.name]
    registry = tmp_path / "registry.jsonl"
    registry.write_text("".join(lines))
    monkeypatch.setattr(pilot45, "REGISTRY", registry)
    pending = [c for c in manifest["cases"] if not ledger["cases"][c["case_id"]].get("run")]
    assert len(pending) == 17 and pending[3]["case_id"] == "s180-pair-00-normal"
    return task, registry, pending


def before_module(name, file):
    spec = importlib.util.spec_from_file_location("pretrain." + name, REPORT / "before" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_red_original_source_reproduces_fourth_pair_collision_and_nonphysical_phantom(
    monkeypatch, tmp_path
):
    task, registry, pending = initial_queue(monkeypatch, tmp_path)
    old = before_module("_before_pair_guard", "pilot45.py")
    old.REGISTRY = registry
    oldparallel = before_module("_before_append", "pilot_parallel.py")
    monkeypatch.setattr(pilot45, "register", old.register)
    c = oldparallel.Coordinator(task)
    for i, case in enumerate(pending[:3]):
        c.begin_flight(case, i)
    with pytest.raises(RuntimeError, match="cohort reuse"):
        c.begin_flight(pending[3], 3)
    assert len(c.ledger["attempts"]) == 52
    c.save()  # original finally/another completed child could persist this memory row
    phantom = c.ledger["cases"][pending[3]["case_id"]]["attempts"][-1]
    assert phantom["state"] == "RUNNING" and not phantom.get("run_id")
    assert not (
        task / "pilot/execution" / pending[3]["case_id"] / "flight-02.request.json"
    ).exists()


def test_green_all17_actual_registrations_in_original_order_keep_limits_and_no_native_submission(
    monkeypatch, tmp_path
):
    task, registry, pending = initial_queue(monkeypatch, tmp_path)
    c = pilot_parallel.Coordinator(task)
    for i, case in enumerate(pending):
        c.begin_flight(case, i % 16)
    assert len(c.ledger["attempts"]) == 65
    requests = list((task / "pilot/execution").glob("*/flight-*.request.json"))
    assert len(requests) == 17
    assert not list((task / "pilot/execution").glob("*/*.process.json"))
    events = [
        json.loads(x)
        for x in registry.read_text().splitlines()
        if json.loads(x)["task"] == task.name
    ]
    assert [e["case_id"] for e in events] == [c["case_id"] for c in pending]
    assert all(e["attempt"] == (2 if i < 4 else 1) for i, e in enumerate(events))
    assert len({e["retry_family_id"] for e in events}) == 17


def test_register_exception_changes_neither_memory_nor_disk_ledger(monkeypatch, tmp_path):
    task, _, pending = initial_queue(monkeypatch, tmp_path)
    c = pilot_parallel.Coordinator(task)
    before = copy.deepcopy(c.ledger)
    digest = common.sha(task / "ledger.json")

    def deny(*args):
        raise pilot45.Blocked("fixture registry refusal")

    monkeypatch.setattr(pilot45, "register", deny)
    with pytest.raises(pilot45.Blocked):
        c.begin_flight(pending[3], 3)
    assert c.ledger == before and common.sha(task / "ledger.json") == digest
    assert not (task / "pilot/execution").exists()


@pytest.mark.parametrize(
    "mutation",
    [None, "cohort", "third", "seed", "op", "raw_hash", "raw_uuid", "retry_id", "namespace"],
)
def test_only_exact_two_member_pair_identity_may_be_shared(monkeypatch, tmp_path, mutation):
    task, registry, pending = initial_queue(monkeypatch, tmp_path)
    cfg = common.read(task / "config.json")
    manifest = common.read(task / "manifest.json")
    cases = {c["case_id"]: c for c in manifest["cases"]}
    normal = cases["s180-pair-00-normal"]
    anomaly = cases["s180-pair-00-anomaly"]

    def event(case):
        return dict(
            task=Path(cfg["short_resume"]["previous_task"]).name,
            cohort=pilot45.SHORT_COHORT,
            case_id=case["case_id"],
            scenario=case["scenario"],
            seed=case["seed"],
            paired_case_id=case["paired_case_id"],
            group_id=cfg["lineage_namespace"] + ":" + case["group_id"],
            retry_family_id=cfg["lineage_namespace"] + ":" + case["retry_family_id"],
            role=case["role"],
        )

    old = event(anomaly)
    new = event(normal)
    if mutation == "cohort":
        old["cohort"] = "final"
    elif mutation == "third":
        manifest["cases"].append(dict(normal, case_id="third-member"))
    elif mutation == "seed":
        new["seed"] += 1
    elif mutation == "op":
        normal["operation_variation"]["payload"] = 0.5
    elif mutation in ("raw_hash", "raw_uuid", "retry_id"):
        key = {"raw_hash": "raw_sha256", "raw_uuid": "run_id", "retry_id": "retry_family_id"}[
            mutation
        ]
        old[key] = new[key] = "reused-identity"
    elif mutation == "namespace":
        cfg["lineage_namespace"] = "another-namespace"
    common.write(task / "manifest.json", manifest)
    common.write(task / "config.json", cfg)
    registry.write_text(json.dumps(old) + "\n")
    if mutation is None:
        pilot45.register(task, new)
        assert len(registry.read_text().splitlines()) == 2
    else:
        with pytest.raises(pilot45.Blocked):
            pilot45.register(task, new)
        assert len(registry.read_text().splitlines()) == 1
