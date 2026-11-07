"""Actual user menu -> complete frozen task factory, with submission blocked by a mock."""

import hashlib
from pathlib import Path

from pretrain import cli, common, pilot45, pilot_parallel
from pretrain import main_collection as main


def factory(monkeypatch, tmp_path):
    home = tmp_path / "operations/pretrain"
    (home / "tasks").mkdir(parents=True)
    store = tmp_path / "store"
    store.mkdir()
    handoff = tmp_path / "handoff.json"
    freeze = tmp_path / "freeze.json"
    common.write(handoff, dict(fixture_only=True))
    common.write(freeze, dict(files={}))
    policy = tmp_path / "policy.json"
    common.write(policy, {})
    spec = dict(
        source_freeze=dict(path=str(freeze), sha256=common.sha(freeze)),
        paths=dict(native_policy=str(policy), paired_policy=str(policy)),
        origin_relative_path="evaluation/joint-origin.json",
        paired_cli_options=[
            "--phase-policy",
            "joint-native-consumption-c2-v2",
            "--paired-case-id",
            "{paired_case_id}",
        ],
    )
    monkeypatch.setattr(main, "HOME", home)
    monkeypatch.setattr(pilot45, "HOME", home)
    monkeypatch.setattr(main, "CATALOG", tmp_path / "catalog")
    monkeypatch.setattr(main, "HANDOFF", handoff)
    monkeypatch.setattr(main, "dependencies", lambda *a: spec)
    monkeypatch.setattr(main, "prior_inventory", lambda: dict(records=[]))
    monkeypatch.setattr(pilot_parallel, "released", lambda: None)
    monkeypatch.setattr(main, "producer_sources", lambda: {})
    monkeypatch.setattr(
        main.native_flight,
        "command",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("Native dispatch prohibited")
        ),
    )
    return home, store


def test_main_factory_includes_existing_lr108_once_and_user_menu_submits_one_scope(
    monkeypatch, tmp_path
):
    home, store = factory(monkeypatch, tmp_path)
    lr = main.create("lr108", storage_root=store, name="main19-fixture-lr108")
    cfg = common.read(lr / "config.json")
    # Boundary fixture: the already independently accepted catalogue is immutable input.
    for case in common.read(lr / "manifest.json")["cases"]:
        rid = hashlib.md5(
            case["case_id"].encode(), usedforsecurity=False
        ).hexdigest()
        proof = tmp_path / "proofs" / (rid + ".json")
        common.write(proof, dict(fixture_only=True))
        entry = dict(
            case=case,
            status="COMMITTED",
            run=str(tmp_path / "old-runs" / rid),
            producer=cfg["producer"],
            attempts=[
                dict(
                    number=1,
                    state="COMPLETED",
                    run_id=rid,
                    parent_attempt_id=None,
                    execution_task=str(lr),
                )
            ],
            commit=dict(path=str(proof), sha256=common.sha(proof)),
        )
        common.write(main.CATALOG / (case["case_id"] + ".json"), entry)
    monkeypatch.setattr(
        main.main_storage, "committed", lambda p, **k: common.read(p)
    )
    task = main.create("main", storage_root=store, name="main19-fixture-main")
    manifest = common.read(task / "manifest.json")
    ledger = common.read(task / "ledger.json")
    assert len(manifest["cases"]) == len(ledger["cases"]) == 1674
    entries = [common.read(p) for p in ledger["cases"].values()]
    assert sum(e.get("status") == "COMMITTED" for e in entries) == 108
    assert sum(not e["attempts"] for e in entries) == 1566
    assert common.read(task / "budget.json")["new_cases"] == 1566
    assert (
        common.read(task / "budget.json")["required_initial_free_bytes"]
        < main.storage_budget(1674)["required_initial_free_bytes"]
    )
    calls = []
    screens = []
    monkeypatch.setattr(cli, "start", lambda t: calls.append(Path(t)))
    monkeypatch.setattr(cli, "watch", lambda t: screens.append(Path(t)))
    # The actual factory is reused; no service/native/model can be invoked by this test.
    monkeypatch.setattr("builtins.input", lambda _: "2")
    original_start = main.user_start
    monkeypatch.setattr(
        main,
        "user_start",
        lambda scope: original_start(scope, storage_root=store),
    )
    cli.menu()
    assert len(calls) == 1 and calls[0] != task
    assert screens == calls
    new_entries = [
        common.read(p)
        for p in common.read(calls[0] / "ledger.json")["cases"].values()
    ]
    assert len(new_entries) == 1674 and all(
        not e["attempts"] for e in new_entries
    )
    assert main.create("main", storage_root=store) == calls[0]
    assert not list(home.glob("tasks/*/operator/service.json"))
