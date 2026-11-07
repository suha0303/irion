"""New user-selected campaigns preserve old evidence and never dispatch native flights."""

import copy
import json

import pytest
from pretrain import common, pilot45
from pretrain import main_collection as main
from test_main_factory import factory


def old_campaign(monkeypatch, tmp_path):
    home, store = factory(monkeypatch, tmp_path)
    task = main.create("main", storage_root=store, name="old-main")
    cfg = common.read(task / "config.json")
    cases = common.read(task / "manifest.json")["cases"]
    case = cases[0]
    raw = tmp_path / "old-raw"
    common.write(raw / "run.json", dict(status="completed"))
    entry = dict(
        case=case,
        attempts=[
            dict(
                number=1,
                state="COMPLETED",
                run_id=raw.name,
                parent_attempt_id=None,
                execution_task=str(task),
            )
        ],
        run=str(raw),
        processing_attempts=1,
    )
    common.write(task / "cases" / (case["case_id"] + ".json"), entry)
    common.write(main.CATALOG / (case["case_id"] + ".json"), entry)
    record = dict(
        task=task.name,
        case_id=case["case_id"],
        seed=case["seed"],
        scenario=case["scenario"],
        cohort=case["cohort"],
        role=case["role"],
        paired_case_id=case["paired_case_id"],
        group_id=cfg["lineage_namespace"] + ":" + case["group_id"],
        retry_family_id=cfg["lineage_namespace"]
        + ":"
        + case["retry_family_id"],
    )
    inventory = dict(
        records=[
            record,
            dict(
                run_id=raw.name,
                seed=case["seed"],
                scenario=case["scenario"],
                paired_case_id=case["paired_case_id"],
                raw_sha256="a" * 64,
                cohort="preexisting_excluded",
            ),
        ]
    )
    monkeypatch.setattr(main, "prior_inventory", lambda: inventory)
    return task, store, case, record, raw


def test_fresh_start_does_not_import_old_raw_attempts_or_modify_old_files(
    monkeypatch, tmp_path
):
    previous, store, case, record, raw = old_campaign(monkeypatch, tmp_path)
    preserved = {
        str(p): common.sha(p) for p in previous.rglob("*") if p.is_file()
    }
    preserved[str(raw / "run.json")] = common.sha(raw / "run.json")
    preserved.update(
        {str(p): common.sha(p) for p in main.CATALOG.glob("*.json")}
    )
    fresh = main.create(
        "main", storage_root=store, name="fresh-main", fresh=True
    )
    cfg = common.read(fresh / "config.json")
    entries = [
        common.read(p)
        for p in common.read(fresh / "ledger.json")["cases"].values()
    ]
    assert len(entries) == 1674 and all(
        not e["attempts"] and "run" not in e for e in entries
    )
    assert (
        cfg["unit_attempts_max"] == 3
        and cfg["lineage_namespace"] != main.NAMESPACE
    )
    assert common.read(fresh / "budget.json")["new_cases"] == 1674
    assert all(common.sha(p) == h for p, h in preserved.items())
    protection = common.read(fresh / "global-protection.json")
    identities = {x for e in protection["entries"] for x in e["identities"]}
    assert (
        "run:" + raw.name in identities
        and "raw-sha256:" + "a" * 64 in identities
    )
    assert "pair:" + case["paired_case_id"] not in identities
    assert not (fresh / "operator/service.json").exists()


def test_new_campaign_cannot_read_or_write_legacy_catalog(
    monkeypatch, tmp_path
):
    previous, store, case, _, _ = old_campaign(monkeypatch, tmp_path)
    fresh = main.create(
        "main", storage_root=store, name="fresh-catalog", fresh=True
    )
    original = common.sha(main.CATALOG / (case["case_id"] + ".json"))
    c = main.Campaign(fresh)
    assert "run" not in c.ledger["cases"][case["case_id"]]
    c.save()
    assert (fresh / "catalog" / (case["case_id"] + ".json")).exists()
    assert common.sha(main.CATALOG / (case["case_id"] + ".json")) == original


@pytest.mark.parametrize(
    "mutation",
    [None, "cohort", "seed", "uuid", "hash", "retry", "unknown_task"],
)
def test_fresh_registry_accepts_same_condition_but_keeps_identity_exclusions(
    monkeypatch, tmp_path, mutation
):
    previous, store, case, old, raw = old_campaign(monkeypatch, tmp_path)
    fresh = main.create(
        "main", storage_root=store, name="fresh-registry", fresh=True
    )
    cfg = common.read(fresh / "config.json")
    new = copy.deepcopy(old)
    new.update(
        task=fresh.name,
        group_id=cfg["lineage_namespace"] + ":" + case["group_id"],
        retry_family_id=cfg["lineage_namespace"]
        + ":"
        + case["retry_family_id"],
    )
    if mutation == "cohort":
        new["cohort"] = "final"
    elif mutation == "seed":
        new["seed"] += 1
    elif mutation == "uuid":
        old["run_id"] = new["run_id"] = "same-uuid"
    elif mutation == "hash":
        old["raw_sha256"] = new["raw_sha256"] = "same-hash"
    elif mutation == "retry":
        new["retry_family_id"] = old["retry_family_id"]
    elif mutation == "unknown_task":
        old["task"] = "unknown-task"
    assert main.registry_identity(fresh, cfg, old, new) is (mutation is None)


def test_same_case_registry_append_uses_real_guard_without_a_child(
    monkeypatch, tmp_path
):
    previous, store, case, old, raw = old_campaign(monkeypatch, tmp_path)
    fresh = main.create(
        "main", storage_root=store, name="fresh-append", fresh=True
    )
    registry = tmp_path / "registry.jsonl"
    registry.write_text(json.dumps(old) + "\n")
    monkeypatch.setattr(pilot45, "REGISTRY", registry)
    c = main.Campaign(fresh)
    job = c.begin_flight(case, 0)
    assert common.read(job["request"])["parent_attempt_id"] is None
    assert len(registry.read_text().splitlines()) == 2
    assert len(c.ledger["cases"][case["case_id"]]["attempts"]) == 1
    assert not list(fresh.glob("pilot/execution/*/*.process.json"))


def test_second_user_start_creates_another_empty_task_without_claim_reset(
    monkeypatch, tmp_path
):
    previous, store, case, old, raw = old_campaign(monkeypatch, tmp_path)
    submitted = []
    monkeypatch.setattr(
        "pretrain.cli.start", lambda task: submitted.append(task)
    )
    first = main.user_start("main", storage_root=store, name="fresh-a")
    common.write(first / "operator/service.json", dict(unit="fixture-stopped"))
    (first / "operator/claimed").write_text("previous attempt preserved")
    second = main.user_start("main", storage_root=store, name="fresh-b")
    assert submitted == [first, second] and first != second
    assert (
        first / "operator/claimed"
    ).read_text() == "previous attempt preserved"
    assert common.read(second / "ledger.json")["physical_attempts"] == 0


def test_existing_unknown_seed_remains_blocked_before_task_creation(
    monkeypatch, tmp_path
):
    home, store = factory(monkeypatch, tmp_path)
    _, plan = main.plans("main")
    case = plan["cases"][0]
    monkeypatch.setattr(
        main,
        "prior_inventory",
        lambda: dict(
            records=[dict(seed=case["seed"], scenario=case["scenario"])]
        ),
    )
    with pytest.raises(pilot45.Blocked, match="already exists"):
        main.create(
            "main", storage_root=store, name="must-not-exist", fresh=True
        )
    assert not (home / "tasks/must-not-exist").exists()
