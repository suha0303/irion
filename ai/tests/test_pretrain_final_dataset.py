"""Final holdout design and menu integration; no real flights, services or scoring."""

import copy
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from pretrain import cli, collection, common, composition
from pretrain import final_dataset as final


def test_final_counts_independence_lengths_and_pairing():
    rows = collection.cases(170001000, "final324")["cases"]
    final.validate(rows)
    assert Counter(row["final_cohort"] for row in rows) == final.COHORTS
    assert Counter(row["role"] for row in rows) == {"normal_test": 216, "anomaly_test": 108}
    assert len({row["case_id"] for row in rows}) == 324
    assert all(row["attempts_max"] == 3 for row in rows)
    normal = [row for row in rows if row["final_cohort"] == "standalone_normal"]
    assert Counter((r["motion"], r["environment"]) for r in normal) == {
        (motion, environment): 3 for motion in final.MOTIONS for environment in final.ENVIRONMENTS
    }
    assert len([row for row in rows if row["mission_seconds"] == 180]) == 54
    assert all(
        row["mission_seconds"] == (180 if row["environment"] == "low_soc" else 480) for row in rows
    )
    for cohort in final.COHORTS:
        subset = [r for r in rows if r["final_cohort"] == cohort]
        assert Counter(r["motion"] for r in subset) == dict.fromkeys(final.MOTIONS, 18)
        assert Counter(r["environment"] for r in subset) == dict.fromkeys(final.ENVIRONMENTS, 18)
        assert sum(r["mission_seconds"] for r in subset) == 46440
    pairs = defaultdict(list)
    for row in rows:
        if row.get("pair_id"):
            pairs[row["pair_id"]].append(row)
    assert len(pairs) == 108
    for pair in pairs.values():
        assert len(pair) == 2
        assert {x["final_cohort"] for x in pair} == {"paired_control", "anomaly"}
        assert len({x["seed"] for x in pair}) == len({x["mission_seconds"] for x in pair}) == 1
        assert all("normal_variation" not in x and x.get("fault_at_s") is None for x in pair)
        assert pair[0]["operation_variation"] == pair[1]["operation_variation"]
        assert pair[0]["motion"] == pair[1]["motion"]
        assert pair[0]["environment"] == pair[1]["environment"]
        healthy = next(r for r in pair if r["final_cohort"] == "paired_control")
        abnormal = next(r for r in pair if r["final_cohort"] == "anomaly")
        assert healthy.get("fault_after_mission_s") is None
        if abnormal["scenario"].endswith(("step", "ramp")):
            assert abnormal["fault_after_mission_s"] == healthy["pair_anchor_after_mission_s"]
            assert abnormal["fault_after_mission_s"] in (
                (100,) if abnormal["mission_seconds"] == 180 else (180, 240, 300)
            )
        else:
            assert abnormal["fault_after_mission_s"] is None
    assert set(
        Counter(row["scenario"] for row in rows if row["role"] == "anomaly_test").values()
    ) == {18}
    assert len({row["group_id"] for row in rows}) == 216
    other = final.cases(170001000 + composition.seed_stride("final324"))
    assert not {r["seed"] for r in rows} & {r["seed"] for r in other}


@pytest.mark.parametrize(
    "field,value",
    [("mission_seconds", 900), ("role", "normal_train"), ("seed", 3), ("group_id", "foreign")],
)
def test_final_membership_cannot_be_relabelled_or_tuned(field, value):
    rows = final.cases(170001000)
    rows[0][field] = value
    with pytest.raises(ValueError, match="changed"):
        final.validate(rows)
    with pytest.raises(ValueError, match="training/development"):
        composition.validation_subsets(final.cases(170001000), "final324")


def test_protection_includes_latest_dataset_and_previously_planned_runs(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    home = root / "Data/operations/pretrain"
    resources = tmp_path / "resources"
    common.write(resources / "protected-inventory.json", dict(runs=[dict(seed=1, run_id="old")]))
    common.write(
        root / "Data/processed/2026-10-31/prepared/frozen-manifest.json",
        dict(runs=[dict(seed=2, group_id="old-group", raw_sha256="old-hash")]),
    )
    common.write(home / "tasks/previous/planned-cases.json", dict(sets=[[dict(seed=3)]]))
    common.write(home / "tasks/current/planned-cases.json", dict(sets=[[dict(seed=4)]]))
    monkeypatch.setattr(final, "ROOT", root)
    monkeypatch.setattr(final, "HOME", home)
    common.write(
        root / "Data/processed/final-test/earlier/sealed-manifest.json",
        dict(
            runs=[
                dict(
                    seed=5,
                    group_id="sealed-final",
                    raw_sha256="sealed-hash",
                    raw_flight_id="sealed-raw",
                )
            ]
        ),
    )
    monkeypatch.setattr(final, "STORE", root / "Data")
    monkeypatch.setattr(final, "RESOURCES", resources)
    protected = final.protection_inputs(home / "tasks/current")
    assert protected["seeds"] == [1, 2, 3, 5]
    for seed in (1, 2, 3, 5):
        with pytest.raises(ValueError, match="overlaps"):
            final.reject_overlap([dict(seed=seed, group_id="new")], protected)
    assert "sealed-raw" in protected["raw_ids"] and "sealed-hash" in protected["raw_sha256"]
    final.reject_overlap([dict(seed=4, group_id="new")], protected)


def test_final_create_is_inert_and_disables_fit_with_bounded_retries(tmp_path, monkeypatch):
    args = cli.parser().parse_args(
        [
            "init",
            "--name",
            "final-fixture",
            "--mode",
            "collect",
            "--collection-profile",
            "final324",
            "--seed-base",
            "170001000",
            "--storage-root",
            str(tmp_path / "data"),
        ]
    )
    monkeypatch.setattr(cli, "HOME", tmp_path / "ops")
    producer_manifest = tmp_path / "runtime-manifest.json"
    common.write(producer_manifest, {"synthetic_fixture": True})
    monkeypatch.setattr(cli, "PRODUCER_MANIFEST", producer_manifest)
    monkeypatch.setattr(cli, "producer_sources", lambda: {})
    monkeypatch.setattr(
        final,
        "protection_inputs",
        lambda **kw: dict(sources={}, seeds=[], group_ids=[], raw_ids=[], raw_sha256=[]),
    )
    monkeypatch.setattr(cli, "start", lambda *a: pytest.fail("create must not start"))
    task = cli.create(args)
    config = common.read(task / "config.json")
    assert config["collection_run_count"] == 324
    assert not config["pipeline_overlap"] and config["storage_profile"] == "standard"
    assert config["unit_attempts_max"] == 3 and config["continue_after_case_failure"] is True
    assert not config["scaler_fit_authorized"] and not config["final_test_scoring_authorized"]
    assert config["dataset_purpose"] == "locked_final_test"
    assert config["design_revision"] == final.DESIGN_REVISION
    assert not (task / "operator/claimed").exists()
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("extra", [["--gates"], ["--collection-sets", "2"], ["--mode", "test"]])
def test_final_cannot_request_training_multipliers_or_gate_fit(extra):
    args = cli.parser().parse_args(
        ["init", "--name", "fixture", "--mode", "collect", "--collection-profile", "final324"]
        + extra
    )
    with pytest.raises(ValueError, match="최종 평가"):
        cli.create(args)


def test_menu_four_selects_final_without_start_until_user_confirmation(
    monkeypatch, capsys, tmp_path
):
    choices = iter(["4", "n"])
    monkeypatch.setattr(cli, "choose", lambda *a: next(choices))
    monkeypatch.setattr(cli, "input_integer", lambda prompt, default, *rest: default)
    monkeypatch.setattr(cli, "input_seed", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *a: str(tmp_path / "data"))
    monkeypatch.setattr(cli, "create", lambda *a: pytest.fail("declined menu must not create"))
    monkeypatch.setattr(cli, "start", lambda *a: pytest.fail("declined menu must not start"))
    cli.menu()
    text = capsys.readouterr().out
    assert "4. 최종 평가 데이터324run" in text
    assert "paired 정상대조108" in text
    assert "최초1+재시도2" in text


def test_final_run_only_uses_existing_extract_preprocess_audit(tmp_path, monkeypatch):
    task = tmp_path / "task"
    output = tmp_path / "store"
    rows = final.cases(170001000)
    planned = task / "planned-cases.json"
    common.write(planned, dict(sets=[rows]))
    protection = task / "protection.json"
    empty = dict(sources={}, seeds=[], group_ids=[], raw_ids=[], raw_sha256=[])
    common.write(protection, empty)
    folder = task / "final-collection"
    common.write(folder / "plan.json", dict(source_hashes={"fixture": "bound"}))
    common.write(folder / "execution/ledger.json", dict(status="fixture"))
    monkeypatch.setattr(final, "STORE", output)
    monkeypatch.setenv("X500_STORAGE_ROOT", str(output))
    monkeypatch.setattr(final, "protection_inputs", lambda **kw: empty)
    admitted = [dict(row, run_id=f"{i:032x}", archive={}) for i, row in enumerate(rows, 1)]
    monkeypatch.setattr(final, "admit_rows", lambda *a: admitted)
    calls = []
    state = {}

    def batch(name, action, items, workers, out, **kwargs):
        calls.append(action)
        if action == "extract":
            assert (output / "consumer/final-test" / task.name).is_dir()
        for row in items:
            base = Path(out) / row["run_id"]
            base.mkdir(parents=True)
            if action == "preprocess":
                common.write(base / "summary.json", dict(output_hashes={}))
            if action == "audit":
                common.write(
                    base / "report.json", dict(accepted_targets=0 if row == admitted[0] else 10)
                )

    fake = SimpleNamespace(
        task=task,
        config=dict(
            planned_cases=str(planned),
            unit_attempts_max=3,
            pipeline_overlap=False,
            final_protection=str(protection),
            flight_workers=4,
            io_workers=2,
            workers=2,
            audit_workers=2,
        ),
        update=lambda **kw: state.update(kw),
        flight=lambda *a: folder,
        begin=lambda *a, **kw: None,
        finish=lambda *a: None,
        batch=batch,
    )
    final.run(fake)
    assert calls == ["extract", "preprocess", "audit"]
    result = common.read(output / "processed/final-test" / task.name / "sealed-manifest.json")
    assert result["status"] == "FINAL_DATASET_SEALED_STOPPED"
    assert not result["scaler_fit_performed"] and not result["final_test_scoring_authorized"]
    assert result["zero_target_runs"] == [admitted[0]["run_id"]]
    assert len(result["pairs"]) == 108
    assert state["final_dataset_sealed"] is True


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "failed_gate",
        "identity",
        "duplicate_raw",
        "health",
        "source",
        "conditions",
        "onset",
        "missing_actual_onset",
        "wrong_schedule",
        "early_application",
        "late_application",
        "retry_history",
        "third_attempt",
    ],
)
def test_final_admission_checks_metadata_roles_and_protection(tmp_path, monkeypatch, damage):
    rows = final.cases(170001000)
    pair = next(r["pair_id"] for r in rows if r.get("fault_after_mission_s"))
    planned = [r for r in rows if r.get("pair_id") == pair]
    records = {}
    metadata = {}
    common.package_imports()
    from preprocessing_tools.control import RAW_FILES

    for i, row in enumerate(planned, 1):
        rid = f"{i:032x}"
        files = {name: dict(kind="file", sha256=f"hash-{i}-{name}", bytes=1) for name in RAW_FILES}
        records[row["case_id"]] = dict(
            status="ARCHIVED_PENDING_QUALITY",
            case=row,
            run_id=rid,
            assessment={
                k: True
                for k in (
                    "contract_passed",
                    "instance_isolation_passed",
                    "missing_data_passed",
                    "physical_gate_passed",
                )
            },
            export=dict(archive=f"fixture-{rid}", sha256="fixture", bytes=1, members=files),
        )
        metadata[f"fixture-{rid}"] = {
            "run.json": dict(run_id=rid, status="completed", origin="px4_sitl"),
            "evaluation/scenario.json": dict(
                name=row["scenario"],
                seed=row["seed"],
                requested_duration_s=row["mission_seconds"],
                operation_variation=copy.deepcopy(row["operation_variation"]),
                fault_after_mission_s=row.get("fault_after_mission_s"),
                fault_schedule=None
                if not row.get("fault_after_mission_s")
                else dict(
                    clock="px4-boot",
                    origin="airborne_path_zero",
                    acknowledged=True,
                    mission_origin_time_us=50_000_000,
                    requested_offset_us=row["fault_after_mission_s"] * 1_000_000,
                    scheduled_time_us=(50 + row["fault_after_mission_s"]) * 1_000_000,
                    applied_time_us=(50 + row["fault_after_mission_s"]) * 1_000_000 + 4000,
                ),
                continuity="independent_reset",
                health_eligibility="confirmed_healthy"
                if row["role"] == "normal_test"
                else "abnormal",
            ),
            "reference/source_lock.json": {"fixture": "valid"},
        }
    first = planned[0]["case_id"]
    if damage in {"retry_history", "third_attempt"}:
        records[first].update(
            attempt=3,
            attempts_max=3,
            previous_attempts=[
                dict(attempt=1, run_id="failed-1", evidence="first-failure.json"),
                dict(attempt=2, run_id="failed-2", evidence="second-failure.json"),
            ]
            if damage == "third_attempt"
            else [],
        )
    if damage == "failed_gate":
        records[first]["assessment"]["physical_gate_passed"] = False
    if damage == "identity":
        metadata[records[first]["export"]["archive"]]["evaluation/scenario.json"]["seed"] = -1
    if damage == "conditions":
        metadata[records[first]["export"]["archive"]]["evaluation/scenario.json"][
            "operation_variation"
        ]["soc"] = 0.01
    if damage == "onset":
        metadata[records[first]["export"]["archive"]]["evaluation/scenario.json"][
            "fault_after_mission_s"
        ] = -1
    if damage in {
        "missing_actual_onset",
        "wrong_schedule",
        "early_application",
        "late_application",
    }:
        row = next(r for r in planned if r.get("fault_after_mission_s"))
        schedule = metadata[records[row["case_id"]]["export"]["archive"]][
            "evaluation/scenario.json"
        ]["fault_schedule"]
        if damage == "missing_actual_onset":
            schedule["applied_time_us"] = None
        elif damage == "wrong_schedule":
            schedule["scheduled_time_us"] += 1
        elif damage == "early_application":
            schedule["applied_time_us"] = schedule["scheduled_time_us"] - 1
        else:
            schedule["applied_time_us"] = (50 + row["mission_seconds"]) * 1_000_000
    if damage == "health":
        row = next(x for x in planned if x["role"] == "normal_test")
        metadata[records[row["case_id"]]["export"]["archive"]]["evaluation/scenario.json"][
            "health_eligibility"
        ] = "uncertain"
    if damage == "source":
        metadata[records[first]["export"]["archive"]]["reference/source_lock.json"]["fixture"] = (
            "wrong"
        )
    folder = tmp_path / "collection"
    common.write(
        folder / "execution/ledger.json",
        dict(status="COLLECTED_PENDING_QUALITY_STOPPED", cases=records, not_started=[]),
    )
    common.write(folder / "plan.json", dict(unit_attempts_max=3))
    monkeypatch.setattr(
        final, "load", lambda *args: SimpleNamespace(identity=lambda ref: metadata[ref["path"]])
    )

    def source_check(plan, source):
        if source["fixture"] != "valid":
            raise ValueError("Source mismatch")

    monkeypatch.setattr(
        final, "collector", lambda: SimpleNamespace(verify_run_source_snapshot=source_check)
    )
    monkeypatch.setattr(final, "STORE", tmp_path / "store")
    protected = dict(
        raw_ids=["00000000000000000000000000000001"] if damage == "duplicate_raw" else [],
        raw_sha256=[],
    )
    if damage and damage != "third_attempt":
        with pytest.raises(ValueError):
            final.admit_rows(folder, planned, protected, "task")
    else:
        admitted = final.admit_rows(folder, planned, protected, "task")
        assert len(admitted) == 2
        assert all(set(row["files"]) == RAW_FILES for row in admitted)
        assert all("evaluation/events.jsonl" not in row["files"] for row in admitted)
        if damage == "third_attempt":
            row = next(row for row in admitted if row["case_id"] == first)
            assert row["collection_attempt"] == row["collection_attempts_max"] == 3
            assert [h["attempt"] for h in row["collection_previous_attempts"]] == [1, 2]


def test_final_roles_preprocess_and_audit_without_training_dependencies(tmp_path):
    import json
    import os
    import subprocess

    fixture = common.load("final_raw_fixture", common.ROOT / "ai/tests/test_preprocessing.py")
    rows = []
    for index, role in enumerate(("normal_test", "anomaly_test"), 1):
        rid = f"{index:032x}"
        raw = tmp_path / "consumer" / rid
        metadata = copy.deepcopy(fixture.META)
        metadata["run_id"] = rid
        metadata["vehicle"].update(
            nominal_capacity_ah=12.0,
            nominal_parallel_cells=4,
            battery_profile_id="vtc6_4s4p",
            profile_revision="vtc6-4s4p",
        )
        common.write(raw / "observations/metadata.json", metadata)
        common.write(
            raw / "run.json", dict(run_id=rid, origin="synthetic_fixture", status="completed")
        )
        common.write(raw / "reference/source_lock.json", dict(synthetic_fixture=True))
        events = list(fixture.events(72))
        for event in events:
            event["run_id"] = rid
        (raw / "observations/streams.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        (raw / "observations/watermarks.jsonl").write_text(
            json.dumps(
                dict(
                    schema="drone.raw.watermark/2.0",
                    run_id=rid,
                    clock_id="px4-boot",
                    through_available_time_us=72_000_000,
                    recorded_events=len(events),
                    closed=True,
                )
            )
            + "\n",
        )
        (raw / "observations/diagnostics.jsonl").touch()
        files = {str(p.relative_to(raw)): common.sha(p) for p in raw.rglob("*") if p.is_file()}
        rows.append(dict(run_id=rid, role=role, path=str(raw), files=files))
    common.write(tmp_path / "rows.json", rows)
    (tmp_path / "canonical").mkdir()
    (tmp_path / "quality").mkdir()
    script = r"""
import importlib.abc,json,sys
from pathlib import Path
root,folder,action,index=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
sys.path[:0]=[str(root/'Data/generation'),str(root/'ai')]
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','safetensors','learning','execution'}:
            raise RuntimeError('Training dependency forbidden: '+fullname)
sys.meta_path.insert(0,NoTraining())
from pretrain.worker import execute
from pretrain.common import read
row=read(folder/'rows.json')[index]
execute(dict(action=action,row=row,out=str(folder/('canonical' if action=='preprocess' else 'quality')/row['run_id']),status=str(folder/'status'/f'{action}-{index}.json'),canonical=str(folder/'canonical')))
assert 'torch' not in sys.modules
if action=='audit':
    report=read(folder/'quality'/row['run_id']/'report.json')
    assert report['role'] in {'normal_test','anomaly_test'} and report['accepted_targets']>0
"""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for key in ("X500_STORAGE_ROOT", "X500_STORAGE_POLICY"):
        env.pop(key, None)
    for action in ("preprocess", "audit"):
        for index in range(2):
            result = subprocess.run(
                [
                    str(common.PYTHON),
                    "-I",
                    "-B",
                    "-c",
                    script,
                    str(common.ROOT),
                    str(tmp_path),
                    action,
                    str(index),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stdout + result.stderr


def test_menu_four_user_confirmation_routes_to_final_task(tmp_path, monkeypatch):
    choices = iter(["4", "y"])
    monkeypatch.setattr(cli, "choose", lambda *a: next(choices))
    monkeypatch.setattr(cli, "input_integer", lambda prompt, default, *rest: default)
    monkeypatch.setattr(cli, "input_seed", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *a: str(tmp_path / "data"))
    calls = []
    task = tmp_path / "only-a-mock"
    monkeypatch.setattr(cli, "create", lambda args: calls.append(args) or task)
    monkeypatch.setattr(cli, "start", lambda path: calls.append(("start", path)))
    monkeypatch.setattr(cli, "watch", lambda path: calls.append(("watch", path)))
    cli.menu()
    args = calls[0]
    assert args.collection_profile == "final324" and args.mode == "collect"
    assert args.collection_sets == 1 and not args.gates
    assert args.flight_workers == 4 and args.export_workers == 2
    assert calls[1:] == [("start", task), ("watch", task)]
    assert not task.exists()


def test_failed_final_collection_cannot_enter_preprocessing(tmp_path, monkeypatch):
    task = tmp_path / "task"
    common.write(task / "planned-cases.json", dict(sets=[final.cases(170001000)]))
    empty = dict(sources={}, seeds=[], group_ids=[], raw_ids=[], raw_sha256=[])
    common.write(task / "protection.json", empty)
    monkeypatch.setattr(final, "protection_inputs", lambda **kw: empty)
    monkeypatch.setattr(final, "STORE", tmp_path / "storage")

    def failed(*args):
        raise RuntimeError("fixture collection failure")

    fake = SimpleNamespace(
        task=task,
        config=dict(
            planned_cases=str(task / "planned-cases.json"),
            unit_attempts_max=3,
            pipeline_overlap=False,
            final_protection=str(task / "protection.json"),
            flight_workers=4,
        ),
        update=lambda **kw: None,
        flight=failed,
        batch=lambda *a, **kw: pytest.fail("failed final set cannot preprocess"),
    )
    with pytest.raises(RuntimeError, match="fixture collection failure"):
        final.run(fake)
    assert not (tmp_path / "storage").exists()


def test_final_conditions_preserve_health_and_requested_motion():
    from capstone_sim.normal_variation import FIELDS, target, validate

    rows = final.cases(170001000)
    for row in rows:
        spec = row["operation_variation"]
        assert set(spec) == FIELDS
        validate(spec)
        motion, environment = row["motion"], row["environment"]
        kind = spec["path"]["kind"]
        assert (
            kind
            in {
                "hover": {"hover"},
                "vertical": {"altitude"},
                "horizontal": {"line", "ellipse"},
                "yaw": {"yaw"},
                "composite": {"mixed"},
                "random": {"random"},
            }[motion]
        )
        if environment not in {"reference", "middle_soc", "low_soc"}:
            assert spec["soc"] == 0.9
        if environment != "wind":
            assert spec["wind_e"] == spec["wind_n"] == spec["gust_sigma"] == 0
        if environment != "cold":
            assert spec["temperature"] == 298.15
        if environment != "payload":
            assert spec["payload"] == 0
        if motion == "composite":
            program = spec["path"]["program"]
            assert len(program) == (2 if row["mission_seconds"] == 180 else 6)
            assert sum(p["duration_s"] for p in program) == row["mission_seconds"]
            assert {p["path"]["kind"] for p in program} >= {"altitude", "line"}
        # Command programs must actually vary the declared axes, not just labels.
        samples = [target(t, spec) for t in range(30, row["mission_seconds"] - 10, 5)]
        spans = [max(s[i] for s in samples) - min(s[i] for s in samples) for i in range(4)]
        if motion in {"vertical", "composite", "random"}:
            assert spans[2] > 0.05
        if motion in {"horizontal", "composite", "random"}:
            assert max(spans[:2]) > 0.05
        if motion in {"yaw", "random"}:
            assert spans[3] > 0.05
        if motion == "hover":
            assert max(spans) < 1e-9


def test_final_dynamic_onsets_are_balanced_and_not_fixed_to_environment():
    rows = final.cases(170001000)
    for scenario in {r["scenario"] for r in rows if r.get("fault_after_mission_s")}:
        long = [r for r in rows if r["scenario"] == scenario and r["mission_seconds"] == 480]
        assert Counter(r["fault_after_mission_s"] for r in long) == {180: 5, 240: 5, 300: 5}
        for environment in set(final.ENVIRONMENTS) - {"low_soc"}:
            assert (
                len({r["fault_after_mission_s"] for r in long if r["environment"] == environment})
                >= 2
            )


def test_final_old_hover_recipe_is_rejected():
    rows = final.cases(170001000)
    rows[0].pop("design_revision")
    with pytest.raises(ValueError, match="changed"):
        final.validate(rows)
