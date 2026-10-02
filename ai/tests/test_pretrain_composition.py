"""Fixed104 composition, independent sets, sealed legacy checks, fail-stop scheduling."""

from collections import Counter

import pytest

from pretrain import cli, collection, common, composition, runner


def test_104_roles_exposure_and_independent_seeds():
    rows = collection.cases(500000, "train104")["cases"]
    composition.validate_cases(rows, "train104")
    assert Counter(r["role"] for r in rows) == dict(
        normal_train=63,
        normal_validation=9,
        normal_calibration=9,
        normal_dev_eval=11,
        anomaly_dev=12,
    )
    exposure = Counter()
    for r in rows:
        exposure[r["role"]] += r["mission_seconds"]
    assert exposure["normal_train"] == 7 * 7380
    assert exposure["normal_validation"] == exposure["normal_calibration"] == 7380
    assert exposure["normal_dev_eval"] == 7380 + 2 * 480
    next_rows = collection.cases(500100, "train104")["cases"]
    for key in ("seed", "group_id", "case_id"):
        assert not {r[key] for r in rows} & {r[key] for r in next_rows}


@pytest.mark.parametrize(
    "damage", ["seed", "group_id", "duration", "condition", "fault", "missing"]
)
def test_changed_membership_or_leakage_rejected(damage):
    rows = collection.cases(500000, "train104")["cases"]
    train = [r for r in rows if r["role"] == "normal_train"]
    if damage in {"seed", "group_id"}:
        train[-1][damage] = train[0][damage]
    elif damage == "duration":
        train[-1]["mission_seconds"] = 42
    elif damage == "condition":
        train[-1]["scenario"] = (
            "waypoint_normal" if train[-1]["scenario"] != "waypoint_normal" else "hover_normal"
        )
    elif damage == "fault":
        next(r for r in rows if r["role"] == "anomaly_dev")["scenario"] = "hover_normal"
    else:
        rows.pop()
    with pytest.raises(ValueError):
        composition.validate_cases(rows, "train104")


def test_multiplier_and_no_legacy_selection():
    args = cli.parser().parse_args(
        ["init", "--name", "fixture", "--mode", "collect", "--collection-sets", "3"]
    )
    assert args.collection_sets == 3 and args.collection_profile == "balanced300"
    for value in ("0", "-1", "1.5"):
        with pytest.raises(SystemExit):
            cli.parser().parse_args(["init", "--name", "fixture", "--collection-sets", value])
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["init", "--name", "fixture", "--collection-profile", "legacy50"])


def fixture_runner(tmp_path):
    task = tmp_path / "task"
    common.write(
        task / "config.json",
        dict(
            mode="collect",
            source_sha256={},
            flight_workers=16,
            workers=4,
            seed_base=500000,
            collection_sets=2,
            collection_profile="train104",
        ),
    )
    return runner.Runner(task)


def synthetic_rows(cases):
    return [
        dict(
            c,
            run_id=f"run-{c['case_id']}",
            raw_sha256=f"hash-{c['case_id']}",
            raw_flight_id=f"flight-{c['case_id']}",
            paired_case_id=f"pair-{c['seed']}",
            status="completed",
            origin="px4_sitl",
            continuity="independent_reset",
            previous_attempts=[],
            health_eligibility="abnormal" if c["role"] == "anomaly_dev" else "confirmed_healthy",
            revision_lock=dict(battery_profile="VTC6_4S4P"),
        )
        for c in cases
    ]


def test_two_sets_prepare_once_with_one_global_handoff(tmp_path, monkeypatch):
    r = fixture_runner(tmp_path)
    calls = []

    def flight(phase, count, rows, label):
        assert len(rows) == 104
        calls.append(("flight", label))
        folder = r.task / label
        common.write(folder / "manifest.json", dict(cases=rows))
        return folder

    monkeypatch.setattr(r, "flight", flight)

    def freeze(task, folder, *, output, progress=None):
        record = common.read(folder / "manifest.json")
        common.write(output / "cases.json", record)
        return dict(
            status="DATA_FROZEN",
            collection_profile="train104",
            revision="COLLECTION",
            revision_lock={"fixture": True},
            protected_inventory_sha256="fixture",
            runs=synthetic_rows(record["cases"]),
        )

    monkeypatch.setattr(collection, "freeze_collection", freeze)

    def prepare(frozen, label, workers, *, cases_path):
        assert cases_path == r.task / "cases.json"
        assert len(common.read(cases_path)["cases"]) == len(frozen["runs"]) == 208
        assert Counter(row["role"] for row in frozen["runs"])["normal_train"] == 126
        assert len(frozen["source_freezes"]) == 2
        assert frozen["scaler_scope"] == "all_selected_normal_train"
        assert label == "prepared"
        calls.append(("prepare", label))
        return tmp_path / label

    monkeypatch.setattr(r, "prepare", prepare)
    r.collection_sets()
    assert [c[0] for c in calls] == ["flight", "flight", "prepare"]
    assert r.state["result"] == str(tmp_path / "prepared/handoff.json")
    assert not (r.task / "collection-handoffs.json").exists()


@pytest.mark.parametrize(
    "damage", ["none", "seed", "group_id", "paired_case_id", "membership", "raw_sha256"]
)
def test_global_set_identity_guards(damage):
    first = collection.cases(500000, "train104")["cases"]
    second = collection.cases(500100, "train104")["cases"]
    cases = first + second
    rows = synthetic_rows(cases)
    sets = [[c["case_id"] for c in part] for part in (first, second)]
    if damage == "none":
        composition.validate_roles(rows, cases, [], "train104", case_sets=sets)
        return
    if damage == "membership":
        sets[1][-1] = sets[0][-1]
    else:
        rows[-1][damage] = rows[0][damage]
    with pytest.raises(ValueError):
        composition.validate_roles(rows, cases, [], "train104", case_sets=sets)


def test_failed_first_collection_never_launches_second_or_preparation(tmp_path, monkeypatch):
    r = fixture_runner(tmp_path)
    calls = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(r, "flight", fail)
    monkeypatch.setattr(r, "prepare", lambda *a, **k: pytest.fail("must not prepare"))
    with pytest.raises(RuntimeError, match="fixture failure"):
        r.collection_sets()
    assert len(calls) == 1


def test_legacy_validation_is_still_sealed():
    common.package_imports()
    from preprocessing_tools.control import validate_cases

    validate_cases(collection.cases(500000)["cases"])
    with pytest.raises(ValueError, match="exactly 50"):
        validate_cases(collection.cases(500000, "train104")["cases"])


def test_expanded_roles_use_original_lineage_health_and_pair_guards():
    cases = collection.cases(500000, "train104")["cases"]
    rows = []
    for i, case in enumerate(cases):
        rows.append(
            dict(
                case,
                run_id=f"fixture-{i}",
                raw_sha256=f"hash-{i}",
                raw_flight_id=f"flight-{i}",
                paired_case_id=f"pair-{case['seed']}",
                status="completed",
                origin="px4_sitl",
                continuity="independent_reset",
                previous_attempts=[],
                health_eligibility="abnormal"
                if case["role"] == "anomaly_dev"
                else "confirmed_healthy",
                revision_lock=dict(battery_profile="VTC6_4S4P"),
            )
        )
    composition.validate_roles(rows, cases, [], "train104")
    rows[-1]["raw_sha256"] = rows[0]["raw_sha256"]
    with pytest.raises(ValueError, match="global identity"):
        composition.validate_roles(rows, cases, [], "train104")
    rows[-1]["raw_sha256"] = "restored-unique-hash"
    rows[-1]["health_eligibility"] = "unknown"
    with pytest.raises(ValueError, match="health"):
        composition.validate_roles(rows, cases, [], "train104")
