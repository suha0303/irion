"""Plan/menu checks only; no raw processing, collection, training or model
imports.
"""

from collections import Counter
from pathlib import Path

import pytest

from pretrain import cli, common, training19


def test_lr108_exact_conditions_roles_and_independent_identities():
    rows = training19.cases(12345)
    training19.validate(
        dict(purpose=training19.LR_PURPOSE, seed_base=12345, cases=rows)
    )
    assert Counter(c["role"] for c in rows) == dict(
        normal_train=81, normal_validation=27
    )
    cells = Counter((c["role"], c["kind"], c["environment"]) for c in rows)
    assert len(cells) == 54
    assert all(
        n == (3 if role == "normal_train" else 1)
        for (role, k, e), n in cells.items()
    )
    assert all(
        c["duration"] == 480
        and c["attempts_max"] == 3
        and c["anomaly_family"] is None
        for c in rows
    )
    for c in rows:
        op = c["operation_variation"]
        assert (
            op["soc"] == 0.9
            and op["temperature"] == 298.15
            and op["payload"] == 0
        )
        assert (
            c["cohort"] == c["role"] and c["purpose"] == training19.LR_PURPOSE
        )
    assert {c["kind"] for c in rows} == set(training19.KINDS)


@pytest.mark.parametrize("change", ["count", "role", "path", "seed", "purpose"])
def test_wrong_or_relabelled_plan_is_rejected(change):
    rows = training19.cases(1)
    plan = dict(purpose=training19.LR_PURPOSE, seed_base=1, cases=rows)
    if change == "count":
        rows.pop()
    elif change == "role":
        rows[0]["role"] = "normal_test"
    elif change == "path":
        rows[0]["operation_variation"]["payload"] = 1
    elif change == "seed":
        rows[0]["seed"] += 1
    else:
        plan["purpose"] = "pilot_development_reference"
    with pytest.raises(ValueError):
        training19.validate(plan)


def test_prepare_only_creates_lr_plan_never_main_task_or_data(
    monkeypatch, tmp_path
):
    root = tmp_path / "plans"
    monkeypatch.setattr(training19, "PLANS", root)
    monkeypatch.setattr(training19, "prior_seeds", lambda: set())
    path = training19.prepare_plan(root / "lr", seed=123)
    m = common.read(path / "manifest.json")
    assert len(m["cases"]) == 108 and m["preliminary_data_connected"] is False
    assert (
        m["existing_data_reused"] is False
        and m["main_training_approved"] is False
    )
    assert not (path / "collection-task.json").exists()
    before = {str(p): common.sha(p) for p in path.iterdir() if p.is_file()}
    assert training19.prepare_plan(path) == path
    assert before == {
        str(p): common.sha(p) for p in path.iterdir() if p.is_file()
    }
    assert set(root.iterdir()) == {root / "planning.lock", path}


def test_explicit_duplicate_seed_is_blocked(monkeypatch, tmp_path):
    root = tmp_path / "plans"
    monkeypatch.setattr(training19, "PLANS", root)
    prior = training19.cases(1)[0]
    monkeypatch.setattr(
        training19, "prior_seeds", lambda: {(prior["scenario"], prior["seed"])}
    )
    with pytest.raises(ValueError, match="collides"):
        training19.prepare_plan(root / "lr", seed=1)
    assert not (root / "lr").exists()


def test_new_menu_keeps_lr_main_existing_screen_and_exit(monkeypatch, capsys):
    inputs = iter(["0"])
    monkeypatch.setattr("builtins.input", lambda p: next(inputs))
    monkeypatch.setattr(
        training19,
        "menu_lr",
        lambda: pytest.fail("Exit must not prepare/start"),
    )
    cli.menu()
    output = capsys.readouterr().out
    assert "1. LR용108런" in output and "2. 본학습용" in output
    assert "3. 기존 작업 화면" in output
    assert all(s not in output for s in ("300run", "324run", "예비", "테스트"))


def test_menu1_lr_dispatch_is_user_selected_and_does_not_start_main(
    monkeypatch,
):
    from pretrain import main_collection

    calls = []
    task = Path("fixture-task")
    monkeypatch.setattr("builtins.input", lambda p: "1")
    monkeypatch.setattr(
        main_collection, "user_start", lambda scope: calls.append(scope) or task
    )
    monkeypatch.setattr(cli, "watch", lambda selected: calls.append(selected))
    cli.menu()
    assert calls == ["lr108", task]


@pytest.mark.local_evidence
def test_missing_main_approval_blocks_before_task_creation(
    monkeypatch, tmp_path
):
    from pretrain import main_collection, training19_main

    monkeypatch.setattr(
        training19_main, "AUTH", tmp_path / "missing-approval.json"
    )
    with pytest.raises(ValueError, match="approval"):
        main_collection.plans("main")


def test_approved_main_menu_dispatches_the_whole_scope(monkeypatch):
    from pretrain import main_collection

    calls = []
    task = Path("fixture-task")
    monkeypatch.setattr("builtins.input", lambda p: "2")
    monkeypatch.setattr(
        main_collection, "user_start", lambda scope: calls.append(scope) or task
    )
    monkeypatch.setattr(cli, "watch", lambda selected: calls.append(selected))
    cli.menu()
    assert calls == ["main", task]


def test_main_builder_has_exact1674_and_byte_equivalent_lr_subset():
    from pretrain import training19_main

    lr = dict(
        purpose=training19.LR_PURPOSE, seed_base=1, cases=training19.cases(1)
    )
    rows = training19_main.validate(
        training19_main.build(lr, 123456789, set()), lr
    )
    assert len(rows) == 1674
    assert rows[:108] == lr["cases"]
    assert Counter(c["cohort"] for c in rows) == dict(
        normal_train=540,
        normal_validation=81,
        normal_calibration=81,
        development=243,
        final=729,
    )
    assert len({c["group_id"] for c in rows}) == 1350
    for cohort, count in (("development", 3), ("final", 9)):
        selected = [
            c
            for c in rows
            if c["cohort"] == cohort
            and c["category"] == "anomaly"
            and c["anomaly_timing"] != "existing"
        ]
        by_type = Counter(
            (
                c["anomaly_family"],
                c["anomaly_timing"],
                c["fault_after_mission_s"],
            )
            for c in selected
        )
        assert set(by_type.values()) == {count}
    combined = [c for c in rows if c["anomaly_family"] == "combined"]
    assert all(c["battery_stage"] == "rpt_16" for c in combined)
    assert not any(c.get("source_task") or c.get("pilot_task") for c in rows)


@pytest.mark.parametrize("fault", ["drop", "lr_role", "pair_wind"])
def test_main_rejects_missing_case_lr_repartition_or_changed_pair(fault):
    from pretrain import training19_main

    lr = dict(
        purpose=training19.LR_PURPOSE, seed_base=1, cases=training19.cases(1)
    )
    rows = training19_main.build(lr, 123456789, set())
    if fault == "drop":
        rows.pop()
    elif fault == "lr_role":
        rows[0]["cohort"] = "normal_validation"
    else:
        next(c for c in rows if c["category"] == "anomaly")[
            "operation_variation"
        ]["wind_profile"]["direction_deg"] += 1
    with pytest.raises(ValueError):
        training19_main.validate(rows, lr)


def test_main_prepare_cannot_create_anything_before_user_final_approval(
    monkeypatch, tmp_path
):
    from pretrain import training19_main

    monkeypatch.setattr(
        training19_main, "AUTH", tmp_path / "missing-approval.json"
    )
    monkeypatch.setattr(
        training19,
        "prepare_plan",
        lambda *a: pytest.fail("Approval must be checked first"),
    )
    with pytest.raises(ValueError, match="approval"):
        training19_main.prepare(tmp_path / "main")
    assert not list(tmp_path.iterdir())
