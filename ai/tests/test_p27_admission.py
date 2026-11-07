"""Precollection software checks only: no flight, model fit or scaler/threshold fit."""

import copy
import dataclasses
import gc
import hashlib
import json
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from capstone_ai.mapping import FEATURES, TARGETS
from learning import admission19 as admission
from learning.history19.data import (
    arrived_samples,
    inspect_run,
    verify_train_frames,
)
from learning.run_health19 import data


def run(role="normal_test", cohort="development", rid="normal"):
    x = np.zeros((205, 19))
    x[:, 17:19] = [16.0, 24.0]
    return data.Run(
        rid,
        role,
        ("run:" + rid, "retry:" + rid, "pair:paired", "group:grouped"),
        "abnormal" if role == "anomaly_test" else "confirmed_healthy",
        "synthetic_unit_fixture",
        x,
        np.arange(205, dtype=np.int64) * 100000,
        cohort,
    )


def manifest(tmp_path, instance):
    source = tmp_path / (instance.run_id + ".npz")
    np.savez(source, values=instance.values, time_us=instance.time_us)
    row = dict(
        run_id=instance.run_id,
        role=instance.role,
        cohort=instance.cohort,
        lineage=list(instance.lineage),
        health=instance.health,
        origin=instance.origin,
        path=source.name,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mission_interval=dict(
            policy=data.INTERVAL_POLICY, start_us=0, end_us_exclusive=20500000
        ),
    )
    value = dict(
        schema=data.SCHEMA,
        feature_order=FEATURES,
        interval_policy=data.INTERVAL_POLICY,
        units=admission.RUN_UNITS,
        step_us=100000,
        vehicle_profile="x500_reference",
        battery_policy=admission.BATTERY_POLICY,
        runs=[row],
    )
    path = tmp_path / (instance.run_id + ".json")
    path.write_text(json.dumps(value))
    return path, value


def test_paired_roles_share_only_one_explicit_evaluation_cohort():
    normal, anomaly = run(), run("anomaly_test", rid="abnormal")
    assert data.validate_splits([normal, anomaly]) == [normal, anomaly]
    with pytest.raises(ValueError, match="cohorts"):
        data.validate_splits(
            [normal, dataclasses.replace(anomaly, cohort="final")]
        )
    with pytest.raises(ValueError, match="cohort"):
        dataclasses.replace(normal, cohort=None).validate()
    with pytest.raises(ValueError, match="Only pair/group"):
        data.validate_splits(
            [normal, dataclasses.replace(anomaly, lineage=normal.lineage)]
        )
    train = dataclasses.replace(
        normal, run_id="train", role="normal_train", cohort=None
    )
    with pytest.raises(ValueError, match="splits"):
        data.validate_splits([train, anomaly])


def test_per_role_manifests_must_be_checked_together(tmp_path):
    n, _ = manifest(tmp_path, run())
    a, av = manifest(tmp_path, run("anomaly_test", rid="abnormal"))
    assert len(list(data.iter_manifests([n, a]))) == 2
    av["runs"][0]["cohort"] = "final"
    a.write_text(json.dumps(av))
    (tmp_path / "abnormal.npz").unlink()
    with pytest.raises(ValueError, match="cohorts"):
        list(
            data.iter_manifests([n, a])
        )  # Fails metadata before loading any run.


def test_units_tail_boundaries_and_stream_release(tmp_path):
    n, value = manifest(tmp_path, run())
    first = data.load_manifest(n)[0]
    np.testing.assert_array_equal(
        data.physical_features(first.values)[:, 17:19],
        np.tile([4.0, 2.0], (205, 1)),
    )
    assert len(first.values) - 100 == 105 and len(first.values) % 100 == 5
    value["units"] = admission.ONLINE_UNITS
    n.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="units"):
        data.load_manifest(n)
    value["units"] = admission.RUN_UNITS
    value["runs"][0]["mission_interval"]["start_us"] = 100000
    n.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Complete declared mission"):
        data.load_manifest(n)
    n, _ = manifest(tmp_path, run())
    a, _ = manifest(tmp_path, run("anomaly_test", rid="abnormal"))
    iterator = data.iter_manifests([n, a])
    previous = next(iterator)
    ref = weakref.ref(previous.values)
    del previous
    next(iterator)
    gc.collect()
    assert ref() is None


def expected_producer():
    return dict(
        battery_policy=admission.BATTERY_POLICY,
        battery_revision="cmu-vah01-baseline/1",
        propulsion_revision=admission.PROPULSION_REVISION,
        **{
            k: c * 64
            for k, c in (
                ("catalog_sha256", "a"),
                ("propulsion_catalog_sha256", "b"),
                ("runtime_sha256", "c"),
                ("source_manifest_sha256", "d"),
            )
        },
    )


def metadata_fixture():
    # Contract fixture declaring source semantics; this is not an actual SITL run.
    producer = expected_producer()
    row = dict(
        run_id="metadata-fixture",
        role="anomaly_test",
        cohort="development",
        origin="px4_sitl",
        health="abnormal",
        lineage=[
            "run:metadata-fixture",
            "pair:p",
            "group:g",
            "retry:r",
            "raw-sha256:" + "e" * 64,
        ],
        mission_interval=dict(
            policy=admission.MISSION_POLICY,
            start_us=20_000_000,
            end_us_exclusive=500_000_000,
            clock_id="px4-boot",
            source="evaluation/execution.json",
            source_sha256="f" * 64,
        ),
        source_evidence=dict(
            producer_runtime_sha256=producer["runtime_sha256"],
            scenario_name="propulsion_loss_step",
            battery_selection=dict(
                catalog_revision=producer["battery_revision"],
                catalog_sha256=producer["catalog_sha256"],
                flight_policy=dict(revision=admission.BATTERY_POLICY),
                initial_stage="rpt_16",
                target_stage="rpt_16",
                transition_s=0.0,
                fixed_for_flight=True,
            ),
            propulsion_selection=dict(
                catalog_revision=producer["propulsion_revision"],
                catalog_sha256=producer["propulsion_catalog_sha256"],
                initial_state="normal",
                target_state="abnormal",
                transition_s=1.0,
                fixed_for_flight=False,
            ),
            transition_evidence=dict(
                propulsion=dict(
                    requested_offset_s=180.0,
                    actual_onset_us=200_000_000,
                    actual_endpoint_us=201_000_000,
                    clock_id="px4-boot",
                )
            ),
        ),
    )
    return producer, row


def test_producer_revision_hash_and_mixed_step_semantics():
    producer, row = metadata_fixture()
    admission.check_producer(producer, producer)
    result = admission.source_evidence(row, producer)
    assert result["initial_anomaly_systems"] == ["battery"]
    assert result["pre_injection_is_normal"] is False
    assert result["raw_physical_audit_verified"] is False
    with pytest.raises(ValueError, match="frozen expected"):
        admission.check_producer(producer, None)
    old = {**producer, "propulsion_revision": "old-component-faults"}
    with pytest.raises(ValueError, match="joint binary"):
        admission.check_producer(old, old)
    with pytest.raises(ValueError, match="runtime_sha256"):
        admission.check_producer(
            {**producer, "runtime_sha256": "0" * 64}, producer
        )
    changed = copy.deepcopy(row)
    changed["source_evidence"]["battery_selection"]["initial_stage"] = "rpt_00"
    with pytest.raises(ValueError, match="fixed battery"):
        admission.source_evidence(changed, producer)
    changed = copy.deepcopy(row)
    changed["source_evidence"]["scenario_name"] = "motor_friction_step"
    with pytest.raises(ValueError, match="old component"):
        admission.source_evidence(changed, producer)
    changed = copy.deepcopy(row)
    changed["source_evidence"]["transition_evidence"]["propulsion"].pop(
        "actual_endpoint_us"
    )
    with pytest.raises(ValueError, match="onset/endpoint"):
        admission.source_evidence(changed, producer)


@pytest.mark.parametrize("family", ["battery", "propulsion", "combined"])
@pytest.mark.parametrize("kind", ["existing", "step", "ramp"])
def test_all_nine_anomaly_selection_metadata_paths(family, kind):
    producer, row = metadata_fixture()
    evidence = row["source_evidence"]
    dynamic = kind != "existing"
    evidence["scenario_name"] = (
        "battery_aging_" if family == "battery" else "propulsion_loss_"
    ) + kind
    battery, propulsion = (
        evidence["battery_selection"],
        evidence["propulsion_selection"],
    )
    battery_active, propulsion_active = (
        family == "battery" and dynamic,
        family != "battery" and dynamic,
    )
    battery["initial_stage"] = (
        "rpt_00" if battery_active or family == "propulsion" else "rpt_16"
    )
    battery["target_stage"] = "rpt_00" if family == "propulsion" else "rpt_16"
    propulsion["initial_state"] = (
        "abnormal" if family != "battery" and not dynamic else "normal"
    )
    propulsion["target_state"] = "normal" if family == "battery" else "abnormal"
    evidence["transition_evidence"] = {}
    for system, selected, active in (
        ("battery", battery, battery_active),
        ("propulsion", propulsion, propulsion_active),
    ):
        selected["fixed_for_flight"] = not active
        selected["transition_s"] = (
            (1.0 if kind == "step" else 20.0) if active else 0.0
        )
        if active:
            evidence["transition_evidence"][system] = dict(
                requested_offset_s=180.0,
                actual_onset_us=200000000,
                actual_endpoint_us=200000000
                + int(selected["transition_s"] * 1000000),
                clock_id="px4-boot",
            )
    result = admission.source_evidence(row, producer)
    expected = (
        ["battery"]
        if family == "combined" or (family == "battery" and not dynamic)
        else []
    )
    if family != "battery" and not dynamic:
        expected += ["propulsion"]
    assert result["initial_anomaly_systems"] == expected
    assert result["pre_injection_is_normal"] == (not expected)


def test_normal_selection_metadata_path():
    producer, row = metadata_fixture()
    row.update(role="normal_test", health="confirmed_healthy")
    evidence = row["source_evidence"]
    evidence["scenario_name"] = "hover_normal"
    evidence["battery_selection"].update(
        initial_stage="rpt_00", target_stage="rpt_00"
    )
    evidence["propulsion_selection"].update(
        initial_state="normal",
        target_state="normal",
        fixed_for_flight=True,
        transition_s=0.0,
    )
    evidence["transition_evidence"] = {}
    assert (
        admission.source_evidence(row, producer)["initial_anomaly_systems"]
        == []
    )


def test_actual_run_cannot_enter_without_an_external_seal(tmp_path):
    path, value = manifest(tmp_path, run())
    value["runs"][0]["origin"] = "px4_sitl"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frozen expected"):
        data.load_manifest(path)


@pytest.mark.parametrize(
    "damage", ["gap", "reset", "missing", "short", "oversized_header"]
)
def test_bad_run_is_rejected_without_quality_trimming(tmp_path, damage):
    instance = run()
    if damage == "gap":
        instance.time_us[110:] += 100000
    elif damage == "reset":
        instance.time_us[110:] -= 100000
    elif damage == "missing":
        instance.values[110, 17] = np.nan
    elif damage == "short":
        instance = dataclasses.replace(
            instance,
            values=instance.values[:100],
            time_us=instance.time_us[:100],
        )
    elif damage == "oversized_header":
        instance = dataclasses.replace(
            instance,
            values=np.zeros((10001, 19)),
            time_us=np.arange(10001, dtype=np.int64) * 100000,
        )
    path, value = manifest(tmp_path, instance)
    # Match declared end so internal gap/reset/NaN checks, not edge cropping, decide.
    value["runs"][0]["mission_interval"]["end_us_exclusive"] = (
        int(instance.time_us[-1]) + 100000
    )
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        data.load_manifest(path)


def online_engine():
    frames, windows = [], []
    for step in range(1, 201):
        frames.append(
            dict(
                run_id="toy",
                clock_id="boot",
                segment_id=0,
                time_us=step * 100000,
                quality={n: "valid" for n in FEATURES},
                values={
                    n: (
                        16.0
                        if n == "pack_voltage"
                        else 24.0
                        if n == "pack_current"
                        else 1.0
                    )
                    for n in FEATURES
                },
            )
        )
    for second in range(10, 21):
        indices = list(range(second * 10 - 99, second * 10 + 1))
        windows.append(
            dict(
                run_id="toy",
                clock_id="boot",
                segment_id=0,
                origin_us=second * 1000000,
                history_times_us=[i * 100000 for i in indices],
                target_times_us=[
                    second * 1000000 + i * 100000 for i in range(1, 11)
                ],
                X=[
                    [frames[i - 1]["values"][n] for n in FEATURES]
                    for i in indices
                ],
            )
        )
    return SimpleNamespace(
        frames=frames,
        windows=windows,
        config=dict(
            battery_profile_id="vtc6_4s4p_r3_long15", nominal_capacity_ah=12
        ),
    )


def test_online_mission_history_target_arrival_and_transform_once():
    engine = online_engine()
    before = copy.deepcopy(engine.windows)
    mission = dict(
        policy=admission.MISSION_POLICY,
        clock_id="boot",
        start_us=2050000,
        end_us_exclusive=18050000,
    )
    arrays, excluded = arrived_samples(engine, mission)
    assert arrays["origin_us"].tolist() == [
        12_000_000,
        13_000_000,
        14_000_000,
        15_000_000,
        16_000_000,
        17_000_000,
    ]
    assert [e["reason"] for e in excluded].count("history_outside_mission") == 2
    assert [e["reason"] for e in excluded].count("target_outside_mission") == 3
    assert np.all(arrays["X"][:, :, 17] == 4.0) and np.all(
        arrays["X"][:, :, 18] == 2.0
    )
    assert np.all(arrays["Y"][:, :, TARGETS.index("pack_voltage")] == 4.0)
    engine.frames[144]["quality"]["pack_voltage"] = "stale"
    fewer, excluded = arrived_samples(engine, mission)
    assert 14_000_000 not in fewer["origin_us"]
    assert any(
        e["reason"] == "missing_or_nonvalid_future_target" for e in excluded
    )
    assert (
        engine.windows == before
    )  # No retrospective editing of issued inputs.


def test_per_run_online_inspection_needs_no_train_val_or_scaler(tmp_path):
    engine = online_engine()
    mission = dict(
        policy=admission.MISSION_POLICY,
        clock_id="boot",
        start_us=2050000,
        end_us_exclusive=18050000,
    )
    arrays, _ = arrived_samples(engine, mission)
    source = tmp_path / "online.npz"
    np.savez(source, **arrays)
    value = dict(
        schema="drone-ai.history19.run/1",
        units=admission.ONLINE_UNITS,
        interval_policy=admission.MISSION_POLICY,
        feature_order=FEATURES,
        target_order=TARGETS,
        step_us=100000,
        history=100,
        future=10,
        emit_every_us=1000000,
        run=dict(
            run_id="toy",
            role="normal_test",
            cohort="preliminary",
            lineage=["run:toy"],
            origin="synthetic_unit_fixture",
            mission_interval=mission,
        ),
        file=dict(
            path=source.name,
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
    )
    path = tmp_path / "online.json"
    path.write_text(json.dumps(value))
    result = inspect_run(path)
    assert result["windows"] == 6 and result["target_points"] == 60
    assert result["status"] == "INPUT_SCHEMA_VALIDATED_NOT_TRAINED"
    value["units"] = admission.RUN_UNITS
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="units"):
        inspect_run(path)
    value["units"] = admission.ONLINE_UNITS
    arrays["Y"][0, 0, 0] = np.nan
    np.savez(source, **arrays)
    value["file"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="arrived target"):
        inspect_run(path)


def test_bounded_unique_train_frame_verification_no_scaler_fit(tmp_path):
    source = tmp_path / "frames.npz"
    values = np.tile(np.array([-1.0, 1.0, -1.0, 1.0])[:, None], (1, 19))
    arrays = dict(
        values=values,
        run_id=np.array(["train"] * 4),
        clock_id=np.array(["boot"] * 4),
        segment_id=np.zeros(4, dtype=np.int64),
        time_us=np.arange(4, dtype=np.int64) * 100000,
    )
    scaler = dict(
        x_mean=np.zeros(19), x_scale=np.ones(19)
    )  # Fixed expected constants, no fit.
    np.savez(source, **arrays)
    verify_train_frames(source, ["train"], scaler, chunk=2)
    arrays["time_us"][-1] = arrays["time_us"][-2]
    np.savez(source, **arrays)
    with pytest.raises(ValueError, match="Duplicate"):
        verify_train_frames(source, ["train"], scaler, chunk=2)
