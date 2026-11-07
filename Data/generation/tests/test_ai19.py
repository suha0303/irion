"""Synthetic19-signal adapter checks, never actual SITL or training evidence."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pretrain import ai19, common

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "raw_fixture", ROOT / "ai/tests/test_preprocessing.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
CONFIG = json.loads(
    (ROOT / "ai/learning/history19/preprocessing.json").read_text()
)


def mission(start_us=0, end_us=26_100_000):
    return dict(
        policy=ai19.MISSION_POLICY,
        clock_id="px4-boot",
        start_us=start_us,
        end_us_exclusive=end_us,
    )


def raw_fixture(path, rid="synthetic-raw", seconds=26):
    path.mkdir(parents=True)
    metadata = copy.deepcopy(fixture.META)
    metadata["run_id"] = rid
    metadata["vehicle"].update(
        profile_id="x500_reference",
        profile_revision=ai19.PROFILE_REVISION,
        nominal_capacity_ah=12,
        nominal_series_cells=4,
        nominal_parallel_cells=4,
        battery_profile_id="vtc6_4s4p",
    )
    common.write(
        path / "run.json",
        dict(run_id=rid, status="completed", origin="synthetic_unit_fixture"),
    )
    common.write(path / "observations/metadata.json", metadata)
    events = list(fixture.events(seconds))
    for e in events:
        e["run_id"] = rid
    stream = path / "observations/streams.jsonl"
    stream.write_text("".join(json.dumps(e) + "\n" for e in events))
    (path / "observations/watermarks.jsonl").write_text(
        json.dumps(
            dict(
                schema="drone.raw.watermark/2.0",
                run_id=rid,
                clock_id="px4-boot",
                through_available_time_us=seconds * 1000000,
                recorded_events=len(events),
                closed=True,
            )
        )
        + "\n"
    )
    (path / "observations/diagnostics.jsonl").touch()
    common.write(
        path / "reference/source_lock.json", dict(synthetic_unit_fixture=True)
    )
    common.write(
        path / "evaluation/scenario.json", dict(synthetic_unit_fixture=True)
    )
    common.write(
        path / "integrity.json",
        {
            str(p.relative_to(path)): dict(
                size=p.stat().st_size, sha256=common.sha(p)
            )
            for p in path.rglob("*")
            if p.is_file()
        },
    )
    return path


def row(rid, role="normal_train", **changes):
    return dict(
        run_id=rid,
        raw_flight_id=rid,
        role=role,
        paired_case_id=rid,
        group_id=rid,
        retry_family_id=rid,
        health_eligibility="confirmed_healthy",
        origin="synthetic_unit_fixture",
        **changes,
    )


def test_shared_engine_no_context_and_physical_vs_online_units(tmp_path):
    raw = raw_fixture(tmp_path / "raw")
    before = common.sha(raw / "observations/metadata.json")
    engine, alias = ai19.replay(raw, CONFIG)
    assert engine.__class__.__module__ == "learning.history19.preprocessing"
    assert len(engine.frames) == 261 and len(engine.windows) > 0
    arrays, excluded = ai19.online_arrays(engine)
    physical, coverage = ai19.run_array(engine, mission())
    assert set(arrays) == {
        "X",
        "Y",
        "run_id",
        "clock_id",
        "segment_id",
        "origin_us",
    }
    assert arrays["X"].shape[1:] == (100, 19) and arrays["Y"].shape[1:] == (
        10,
        4,
    )
    assert arrays["origin_us"][0] == 10_000_000
    assert np.diff(arrays["origin_us"]).tolist() == [1_000_000] * (
        len(arrays["X"]) - 1
    )
    np.testing.assert_allclose(arrays["X"][:, :, 17], 3.7)
    np.testing.assert_allclose(arrays["X"][:, :, 18], 0.5)
    np.testing.assert_allclose(physical["values"][:, 17], 14.8)
    np.testing.assert_allclose(physical["values"][:, 18], 6.0)
    assert (
        len(physical["values"]) == 261 and len(physical["values"]) % 100 == 61
    )
    assert physical["time_us"].dtype == np.int64 and np.all(
        np.diff(physical["time_us"]) == 100000
    )
    assert (
        common.sha(raw / "observations/metadata.json") == before
        and alias["metadata_sha256"] == before
    )
    assert coverage["status"] == "available"
    assert all("C" not in w for w in engine.windows)


@pytest.mark.parametrize(
    "fault", ["gap", "segment", "clock", "missing", "ground", "short"]
)
def test_run_unavailable_preserves_bad_interval(tmp_path, fault):
    engine, _ = ai19.replay(raw_fixture(tmp_path / "raw"), CONFIG)
    engine = copy.deepcopy(engine)
    if fault == "gap":
        engine.frames.pop(123)
    elif fault == "segment":
        engine.frames[123]["segment_id"] += 1
    elif fault == "clock":
        engine.frames[123]["clock_id"] = "new-clock"
    elif fault == "missing":
        engine.frames[123]["quality"]["pack_voltage"] = "unavailable"
        engine.frames[123]["values"]["pack_voltage"] = None
    elif fault == "ground":
        engine.frames[123]["operating_state"] = "ground"
    elif fault == "short":
        engine.frames = engine.frames[:100]
    values, report = ai19.run_array(engine, mission())
    assert (
        values is None
        and report["status"] == "unavailable"
        and report["reasons"]
    )
    assert report["interval_frames"] == len(engine.frames)


def test_mission_phase_is_explicit_and_landing_missing_values_are_preserved(
    tmp_path,
):
    engine, _ = ai19.replay(raw_fixture(tmp_path / "raw"), CONFIG)
    last = engine.frames[-1]
    last["quality"]["climb_rate_error"] = "unavailable"
    last["values"]["climb_rate_error"] = None
    before = copy.deepcopy(engine.frames)
    scope = mission(start_us=2_050_000, end_us=22_370_000)
    values, report = ai19.run_array(engine, scope)
    assert report["status"] == "available"
    assert (
        values["time_us"][0] == 2_100_000
        and values["time_us"][-1] == 22_300_000
    )
    assert values["values"].shape == (203, 19)
    assert len(values["values"]) % 100 == 3
    assert report["excluded_airborne_before_mission"] == 21
    assert report["excluded_airborne_after_mission"] == 37
    assert engine.frames == before
    engine.frames[100]["quality"]["pack_voltage"] = "invalid"
    rejected, coverage = ai19.run_array(engine, scope)
    assert (
        rejected is None
        and "missing_or_nonvalid_feature" in coverage["reasons"]
    )


def test_mission_missing_boundary_or_edge_frames_never_fall_back_to_good_subset(
    tmp_path,
):
    engine, _ = ai19.replay(raw_fixture(tmp_path / "raw"), CONFIG)
    assert ai19.run_array(engine)[1]["reasons"] == [
        "missing_mission_boundaries"
    ]
    engine.frames.pop()
    values, report = ai19.run_array(engine, mission())
    assert values is None and "incomplete_mission_interval" in report["reasons"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "reverse", "nan"])
def test_actual_operational_boundaries_are_unambiguous(tmp_path, damage):
    raw = tmp_path / "raw"
    raw.mkdir()
    events = [
        dict(state="mission", sim_time=2.05),
        dict(state="landing", sim_time=22.37),
    ]
    if damage == "missing":
        events.pop()
    elif damage == "duplicate":
        events.append(events[0])
    elif damage == "reverse":
        events[1]["sim_time"] = 1
    else:
        events[0]["sim_time"] = float("nan")
    path = raw / "evaluation/execution.json"
    path.parent.mkdir()
    path.write_text(json.dumps(events))
    with pytest.raises(ValueError):
        ai19.mission_interval(raw)
    events = [
        dict(state="mission", sim_time=2.05),
        dict(state="landing", sim_time=22.37),
    ]
    path.write_text(json.dumps(events))
    interval = ai19.mission_interval(raw)
    assert (
        interval["start_us"] == 2_050_000
        and interval["end_us_exclusive"] == 22_370_000
    )
    assert interval["source_sha256"] == common.sha(path)


def test_no_missing_future_targets_filled_with_zeros(tmp_path):
    engine, _ = ai19.replay(raw_fixture(tmp_path / "raw"), CONFIG)
    before, _ = ai19.online_arrays(engine)
    engine.frames[-5]["quality"]["pack_voltage"] = "invalid"
    after, excluded = ai19.online_arrays(engine)
    assert len(after["X"]) < len(before["X"]) and excluded
    assert all(
        e["reason"] == "missing_or_nonvalid_future_target" for e in excluded
    )


@pytest.mark.parametrize(
    "key", ["raw_flight_id", "paired_case_id", "group_id", "retry_family_id"]
)
def test_lineage_rejects_split_leakage(key):
    train = row("train")
    validation = row("validation", "normal_validation")
    validation[key] = train[key]
    with pytest.raises(ValueError, match="leakage"):
        ai19.validate_lineage([train, validation])


def test_raw_id_and_retry_parent_share_namespace():
    a = row("a")
    b = row("b", "normal_validation")
    b["raw_flight_id"] = "a"
    with pytest.raises(ValueError, match="leakage"):
        ai19.validate_lineage([a, b])
    b = row("b", "normal_validation")
    b["previous_attempts"] = ["a"]
    with pytest.raises(ValueError, match="leakage"):
        ai19.validate_lineage([a, b])


def test_pair_test_roles_share_test_partition():
    a = row("healthy", "normal_test")
    b = row("abnormal", "anomaly_test")
    b["paired_case_id"] = a["paired_case_id"]
    ai19.validate_lineage([a, b])


def test_old_metadata_not_relabelled_as_current(tmp_path):
    raw = raw_fixture(tmp_path / "raw")
    m = common.read(raw / "observations/metadata.json")
    m["vehicle"]["profile_revision"] = "vtc6-4s4p"
    common.write(raw / "observations/metadata.json", m)
    with pytest.raises(ValueError, match="Current X500"):
        ai19.replay(raw, CONFIG)


def exported_fixture(
    tmp_path, monkeypatch, roles=("normal_train", "normal_validation")
):
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setenv("X500_STORAGE_ROOT", str(store))
    monkeypatch.delenv("X500_STORAGE_POLICY", raising=False)
    monkeypatch.setattr(ai19, "STORE", store)
    ai19.ai_modules()
    common.package_imports()
    from pretrain import archive

    monkeypatch.setattr(archive, "STORAGE", store)
    selection = []
    for i, role in enumerate(roles):
        rid = f"synthetic-{i}"
        raw = raw_fixture(tmp_path / rid, rid)
        selection.append(dict(row(rid, role), path=str(raw)))
    path = tmp_path / "selection.json"
    common.write(
        path, dict(schema="drone-data.ai19.selection/1", runs=selection)
    )
    monkeypatch.setattr(
        ai19,
        "producer_binding",
        lambda: dict(
            battery_policy=ai19.POLICY,
            source_hashes={},
            runtime_sha256="synthetic-binding",
        ),
    )
    monkeypatch.setattr(
        ai19,
        "admit",
        lambda r, b: (
            r,
            dict(synthetic_unit_fixture=True, mission_interval=mission()),
        ),
    )
    output = store / "processed/export"
    result = ai19.prepare(path, output)
    return output, result


def test_archive_extract_two_views_and_train_only_unique_scaler(
    tmp_path, monkeypatch
):
    output, result = exported_fixture(tmp_path, monkeypatch)
    assert ai19.check_export(output)["runs"] == 2
    assert result["history19_fit"]["status"] == "FIT_INPUT_EXPORTED_NOT_TRAINED"
    manifest = common.read(output / "history19-manifest.json")
    assert manifest["scaler_fit_run_ids"] == ["synthetic-0"]
    with np.load(output / "scaler.npz") as s:
        np.testing.assert_allclose(s["x_mean"][17:], [3.7, 0.5])
    with np.load(output / "normal_train_frames.npz") as a:
        assert len(a["values"]) == 261 and set(a["run_id"]) == {"synthetic-0"}
    for r in result["records"]:
        consumer = Path(r["metadata_evidence"]["consumer_path"])
        assert {
            str(p.relative_to(consumer))
            for p in consumer.rglob("*")
            if p.is_file()
        } == ai19.CONSUMER
        assert not (consumer / "evaluation").exists()
    env = dict(
        os.environ, PYTHONPATH=f"{ROOT}/ai:{ROOT}/ai/src:{ROOT}/Data/generation"
    )
    script = """from pathlib import Path
from learning.history19.data import load
from learning.run_health19.data import load_manifest
import sys
p=Path(sys.argv[1]);data,scaler,binding=load(p/'history19-manifest.json')
assert data['normal_train']['X'].shape[1:]==(100,19)
for role in ('normal_train','normal_validation'):
 r=load_manifest(p/('run-health19-'+role+'.json'))[0]
 assert r.values.shape==(261,19) and abs(r.values[100,17]-14.8)<1e-10
"""
    subprocess.run(
        [str(ROOT / "ai/.venv/bin/python"), "-B", "-c", script, str(output)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    (output / "scaler.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        ai19.check_export(output)


def test_one_run_is_two_views_but_not_fake_train_validation(
    tmp_path, monkeypatch
):
    output, result = exported_fixture(tmp_path, monkeypatch, ("normal_train",))
    check = ai19.check_export(output)
    assert check["online_available"] == check["run_available"] == 1
    assert result["history19_fit"]["status"] == "NOT_FIT_READY"
    assert not (output / "history19-manifest.json").exists()


def test_one_run_shell_submits_one_scoped_task_and_rejects_unsafe_names(
    tmp_path,
):
    tool = tmp_path / "Data/generation/tools/ai19_one_run.sh"
    tool.parent.mkdir(parents=True)
    tool.write_bytes(
        (ROOT / "Data/generation/tools/ai19_one_run.sh").read_bytes()
    )
    stub = tool.parent.parent / "drone-pretrain"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> calls.txt\n')
    stub.chmod(0o700)
    subprocess.run(["bash", str(tool), "one"], check=True, cwd=tmp_path)
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls == [
        "integration19 init --name one",
        "integration19 start "
        + str(tmp_path / "Data/operations/pretrain/tasks/one"),
    ]
    assert (
        subprocess.run(
            ["bash", str(tool), "../escape"], cwd=tmp_path, capture_output=True
        ).returncode
        == 2
    )
    assert (tmp_path / "calls.txt").read_text().splitlines() == calls


def test_data_subprocess_imports_no_training_dependencies():
    env = dict(
        os.environ, PYTHONPATH=f"{ROOT}/Data/generation:{ROOT}/ai:{ROOT}/ai/src"
    )
    command = 'import sys;from pretrain.ai19 import ai_modules;ai_modules();assert "torch" not in sys.modules'
    subprocess.run(
        [str(ROOT / "Data/generation/.venv/bin/python"), "-B", "-c", command],
        env=env,
        check=True,
    )


def test_legacy_fresh_cli_blocks_current_battery_before_task_creation(
    monkeypatch, tmp_path
):
    from pretrain import cli

    args = cli.parser().parse_args(
        ["init", "--name", "blocked", "--mode", "collect"]
    )
    monkeypatch.setattr(
        cli,
        "producer_sources",
        lambda: {"profiles/battery-aging/cmu-vah01.json": "fixture"},
    )
    monkeypatch.setattr(cli, "HOME", tmp_path)
    with pytest.raises(ValueError, match="19신호"):
        cli.create(args)
    assert not list(tmp_path.iterdir())


def admission_fixture(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "simulation/sim/src"))
    from capstone_sim import battery_aging, collection_config, run_assessment

    binding = ai19.producer_binding()
    raw = raw_fixture(tmp_path / "synthetic-source", "synthetic-source")
    common.write(
        raw / "run.json",
        dict(
            run_id=raw.name,
            status="completed",
            origin="px4_sitl",
            configuration="collection",
            instance=0,
            synthetic_unit_fixture=True,
        ),
    )
    common.write(
        raw / "evaluation/scenario.json",
        dict(
            name="hover_normal",
            seed=1,
            paired_case_id="fixture-pair",
            health_eligibility="confirmed_healthy",
            requested_duration_s=26,
            producer_lock_sha256=binding["runtime_sha256"],
        ),
    )
    common.write(
        raw / "reference/source_lock.json",
        dict(
            local_sources={
                n: h
                for n, h in binding["source_hashes"].items()
                if n.startswith(("sim/", "profiles/", "contracts/"))
            },
            runtime_binary_sha256={
                n: binding["source_hashes"][n]
                for n in [
                    "vendor/PX4-Autopilot/build/px4_sitl_default/bin/px4",
                    "build/current/libDronePowertrain.so",
                    "runtime/gazebo-transport/lib/libgz-transport13.so.13.6.0",
                ]
            },
        ),
    )
    (raw / "evaluation/battery-catalog.json").write_bytes(
        battery_aging.CATALOG.read_bytes()
    )
    common.write(
        raw / "evaluation/battery-selection.json",
        battery_aging.resolve(
            "hover_normal", None, None, collection_config.battery()
        ),
    )
    common.write(
        raw / "evaluation/execution.json",
        [
            dict(state="mission", sim_time=0),
            dict(state="landing", sim_time=26.1),
        ],
    )

    def integrity():
        common.write(
            raw / "integrity.json",
            {
                str(p.relative_to(raw)): dict(
                    size=p.stat().st_size, sha256=common.sha(p)
                )
                for p in raw.rglob("*")
                if p.is_file() and p.name != "integrity.json"
            },
        )

    integrity()
    monkeypatch.setattr(
        run_assessment,
        "assess",
        lambda *a: dict.fromkeys(
            [
                "contract_passed",
                "instance_isolation_passed",
                "missing_data_passed",
                "physical_gate_passed",
            ],
            True,
        ),
    )
    selection = dict(
        path=str(raw),
        role="normal_train",
        group_id="fixture-group",
        retry_family_id="fixture-retry",
    )
    return raw, binding, selection, integrity, run_assessment


@pytest.mark.parametrize(
    "damage", [None, "old_producer", "wrong_policy", "tamper", "failed_quality"]
)
def test_current_producer_battery_and_quality_admission(
    tmp_path, monkeypatch, damage
):
    raw, binding, selection, integrity, assessment = admission_fixture(
        tmp_path, monkeypatch
    )
    if damage == "old_producer":
        d = common.read(raw / "evaluation/scenario.json")
        d["producer_lock_sha256"] = "0" * 64
        common.write(raw / "evaluation/scenario.json", d)
        integrity()
    if damage == "wrong_policy":
        d = common.read(raw / "evaluation/battery-selection.json")
        d["flight_policy"]["revision"] = "old"
        common.write(raw / "evaluation/battery-selection.json", d)
        integrity()
    if damage == "tamper":
        (raw / "observations/streams.jsonl").write_text("tampered")
    if damage == "failed_quality":
        monkeypatch.setattr(
            assessment, "assess", lambda *a: dict(contract_passed=False)
        )
    if damage:
        with pytest.raises(ValueError):
            ai19.admit(selection, binding)
    else:
        r, e = ai19.admit(selection, binding)
        assert (
            r["battery_policy"] == ai19.POLICY
            and r["producer_runtime_sha256"] == binding["runtime_sha256"]
        )
        assert e["battery_selection"]["target_stage"] == "rpt_00"


@pytest.mark.parametrize(
    "field,value",
    [
        ("previous_attempts", "run-id"),
        ("previous_attempts", [None]),
        ("parent_attempt_id", 123),
    ],
)
def test_malformed_retry_identity_is_rejected(field, value):
    r = row("current")
    r[field] = value
    with pytest.raises(ValueError, match="attempt"):
        ai19.validate_lineage([r])
