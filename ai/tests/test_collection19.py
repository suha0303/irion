"""Software fixtures only: bounded datasets, independent statistics, no real collection/fit."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from capstone_ai.dataset import feature_transform, target_transform
from capstone_ai.mapping import FEATURES, TARGETS
from learning.admission19 import MISSION_POLICY, ONLINE_UNITS
from learning.collection19.cases import compare_views
from learning.collection19.contract import check_scope, write_new
from learning.collection19.streams import Moments, array_blocks, write_npz
from learning.history19.data import load
from learning.history19.pilot import ROOT, json_file, sha
from learning.run_health19.data import iter_manifests

ag = importlib.import_module("learning.collection19.aggregate")


def producer():
    return dict(
        battery_policy="cmu-vah01-rpt16-only/1",
        battery_revision="cmu-vah01-baseline/1",
        catalog_sha256="a" * 64,
        propulsion_revision="x500-propulsion-binary-v1",
        propulsion_catalog_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
        runtime_sha256="d" * 64,
    )


def export(
    root,
    rid,
    role,
    offset=0,
    *,
    n=131,
    row_override=None,
    producer_binding=None,
    physical_override=None,
):
    root.mkdir(parents=True)
    binding = producer_binding or producer()
    times = 60000000 + np.arange(n, dtype=np.int64) * 100000
    values = (
        np.arange(19)[None, :] * 0.01 + offset + np.arange(n)[:, None] * 0.001
    )
    values[:, 17] = 16 + offset + np.arange(n) * 0.2
    values[:, 18] = 24 + offset + np.arange(n) * 0.01
    if physical_override is not None:
        values = physical_override
    row = dict(
        run_id=rid,
        case_id="fixture:" + rid,
        raw_flight_id=rid,
        paired_case_id="pair-" + rid,
        group_id="group-" + rid,
        retry_family_id="retry-" + rid,
        previous_attempts=[],
        parent_attempt_id=None,
        raw_sha256=hashlib.sha256(rid.encode()).hexdigest(),
        role=role,
        cohort=role,
        purpose="software_fixture",
        health="confirmed_healthy",
        health_eligibility="confirmed_healthy",
        origin="synthetic_unit_fixture",
        lineage=[
            "run:" + rid,
            "pair:pair-" + rid,
            "group:group-" + rid,
            "retry:retry-" + rid,
        ],
        source_evidence=dict(fixture_only=True),
        mission_interval=dict(
            policy=MISSION_POLICY,
            source="evaluation/execution.json",
            source_sha256="f" * 64,
            start_us=int(times[0]),
            end_us_exclusive=int(times[-1] + 100000),
            clock_id="px4-boot",
        ),
    )
    if row_override is not None:
        row.update(row_override)
    config = json_file(ROOT / "ai/learning/history19/preprocessing.json")
    transformed = feature_transform(values, config)
    targets = target_transform(values[:, [FEATURES.index(k) for k in TARGETS]])
    origins = np.array(
        [
            t
            for t in times
            if t % 1000000 == 0
            and t - 9900000 >= times[0]
            and t + 1000000 <= times[-1]
        ],
        dtype=np.int64,
    )
    at = {int(t): i for i, t in enumerate(times)}
    X = np.array(
        [transformed[at[int(t)] - 99 : at[int(t)] + 1] for t in origins]
    )
    Y = np.array([targets[at[int(t)] + 1 : at[int(t)] + 11] for t in origins])
    np.savez(
        root / "online.npz",
        X=X,
        Y=Y,
        run_id=np.full(len(origins), rid),
        clock_id=np.full(len(origins), "px4-boot"),
        segment_id=np.zeros(len(origins), dtype=np.int64),
        origin_us=origins,
    )
    np.savez(root / "run.npz", values=values, time_us=times)
    write_new(
        root / "online.json",
        dict(
            schema="drone-ai.history19.run/1",
            units=ONLINE_UNITS,
            interval_policy=MISSION_POLICY,
            feature_order=FEATURES,
            target_order=TARGETS,
            step_us=100000,
            history=100,
            future=10,
            emit_every_us=1000000,
            producer=binding,
            run=row,
            file=dict(path="online.npz", sha256=sha(root / "online.npz")),
        ),
    )
    write_new(
        root / "run.json",
        dict(
            schema="drone-ai.run-health19.dataset/1",
            feature_order=FEATURES,
            units="physical_common19_pack_V_A",
            step_us=100000,
            interval_policy=MISSION_POLICY,
            vehicle_profile="x500_reference",
            battery_policy=binding["battery_policy"],
            producer=binding,
            runs=[dict(row, path="run.npz", sha256=sha(root / "run.npz"))],
        ),
    )
    with (root / "frames.jsonl").open("w", encoding="utf-8") as f:
        for t, vector in [
            (times[0] - 100000, np.full(19, 1e9)),
            *zip(times, values),
            (times[-1] + 100000, np.full(19, -1e9)),
        ]:
            frame = dict(
                run_id=rid,
                clock_id="px4-boot",
                segment_id=0,
                time_us=int(t),
                feature_profile="common19_ctx0",
                feature_order=FEATURES,
                operating_state="airborne",
                quality=dict.fromkeys(FEATURES, "valid"),
                values=dict(zip(FEATURES, vector.tolist())),
                lineage=dict(
                    sample=dict(sample_time_us=int(t), available_time_us=int(t))
                ),
            )
            f.write(json.dumps(frame) + "\n")
    checked = compare_views(
        root,
        producer=binding,
        protection=dict(schema="drone-ai.global-protection/1", entries=[]),
    )
    return dict(row=row, export=root, audit=checked), values


@pytest.fixture
def records(tmp_path):
    first, _ = export(tmp_path / "train0", "fixture-t0", "normal_train", 0)
    second, _ = export(tmp_path / "train1", "fixture-t1", "normal_train", 2)
    validation, _ = export(
        tmp_path / "val", "fixture-v", "normal_validation", 1000
    )
    return [first, second, validation]


def registry():
    return dict(schema="drone-ai.global-protection/1", entries=[])


def test_bounded_dataset_matches_current_readers_and_exact_train_only_statistics(
    records, tmp_path
):
    events = []
    root, prov = ag._compile(
        records,
        tmp_path / "dataset",
        scope="main",
        producer=producer(),
        registry=registry(),
        fixture_only=True,
        progress=events.append,
    )
    checked = ag._audit_compiled(
        records, root, scope="main", producer=producer(), registry=registry()
    )
    assert checked["passed"] and checked["actual_data_verified"] is False
    assert prov["unique_valid_mission_frames"] == 262 and prov[
        "fit_run_ids"
    ] == [
        "fixture-t0",
        "fixture-t1",
    ]
    # Independent analytic expectation; validation offset1000 and outside-mission +/-1e9 are excluded.
    expected = np.arange(19) * 0.01 + 1 + 0.065
    expected[17] = (16 + 1 + 65 * 0.2) / 4
    expected[18] = (24 + 1 + 65 * 0.01) / 12
    with np.load(root / "scaler.npz", allow_pickle=False) as a:
        np.testing.assert_allclose(a["x_mean"], expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(a["y_mean"], a["x_mean"][[4, 5, 6, 17]])
    arrays, scaler, _ = load(
        root / "history19-manifest.json",
        cache=tmp_path / "normalized",
        expected_producer=producer(),
        protection=registry(),
    )
    assert isinstance(arrays["normal_train"]["X"], np.memmap)
    assert arrays["normal_train"]["X"].shape == (6, 100, 19)
    assert (
        len(
            list(
                iter_manifests(
                    ag.whole_manifest_paths(root),
                    expected_producer=producer(),
                    protection=registry(),
                )
            )
        )
        == 3
    )
    shared = ag.load_shared_scaler(root)
    np.testing.assert_array_equal(shared.mean, scaler["x_mean"])
    assert events and all(e["whole_task_complete"] is False for e in events)
    assert any(
        e["stage"] == "train_statistics" and e["done"] == 2 for e in events
    )


@pytest.mark.parametrize(
    "damage",
    [
        "frames_segment",
        "frames_duplicate",
        "scaler",
        "val_as_train",
        "scope",
        "canonical_gap",
        "canonical_future",
        "target_zero",
    ],
)
def test_independent_audit_rejects_wrong_statistics_identity_or_inputs(
    records, tmp_path, damage
):
    root, _ = ag._compile(
        records,
        tmp_path / "dataset",
        scope="main",
        producer=producer(),
        registry=registry(),
        fixture_only=True,
    )
    if damage in {"frames_segment", "frames_duplicate"}:
        path = root / "normal_train_frames.npz"
        with np.load(path, allow_pickle=False) as a:
            fields = dict(a)
        if damage == "frames_segment":
            fields["segment_id"] += 1
        else:
            # Same numeric moments, but false additional segment identities must fail.
            fields = {
                k: np.tile(v, (2, 1) if k == "values" else 2)
                for k, v in fields.items()
            }
            fields["segment_id"][262:] += 1
        np.savez(path, **fields)
    if damage == "scaler":
        with np.load(root / "scaler.npz", allow_pickle=False) as a:
            fields = dict(a)
        fields["x_mean"][0] += 1
        np.savez(root / "scaler.npz", **fields)
    if damage == "val_as_train":
        manifest = json_file(root / "history19-manifest.json")
        manifest["scaler_fit_run_ids"].append("fixture-v")
        (root / "history19-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    if damage in {"canonical_gap", "canonical_future"}:
        path = records[0]["export"] / "frames.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        if damage == "canonical_gap":
            lines.pop(35)
        else:
            f = json.loads(lines[35])
            f["lineage"]["sample"]["available_time_us"] = f["time_us"] + 100000
            lines[35] = json.dumps(f)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if damage == "target_zero":
        p = records[0]["export"] / "online.npz"
        with np.load(p, allow_pickle=False) as a:
            fields = dict(a)
        fields["Y"][:] = 0
        np.savez(p, **fields)
        m = json_file(p.parent / "online.json")
        m["file"]["sha256"] = sha(p)
        (p.parent / "online.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError):
        ag._audit_compiled(
            records,
            root,
            scope="lr108" if damage == "scope" else "main",
            producer=producer(),
            registry=registry(),
        )


def test_lr_scaler_uses_same_source_subset_without_new_collection(
    records, tmp_path
):
    main, _ = ag._compile(
        records,
        tmp_path / "main",
        scope="main",
        producer=producer(),
        registry=registry(),
        fixture_only=True,
    )
    lr, _ = ag._compile(
        [records[0], records[2]],
        tmp_path / "lr",
        scope="lr108",
        producer=producer(),
        registry=registry(),
        fixture_only=True,
    )
    main_scale = ag.load_shared_scaler(main)
    lr_scale = ag.load_shared_scaler(lr)
    assert main_scale.fit_run_ids == (
        "fixture-t0",
        "fixture-t1",
    ) and lr_scale.fit_run_ids == ("fixture-t0",)
    assert main_scale.mean[0] - lr_scale.mean[0] == pytest.approx(1)
    assert json_file(lr / "scaler-provenance.json")["scope"] == "lr108"


def test_aggregate_calls_independent_audit_once_and_returns_both_sealed_receipts(
    records, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        ag,
        "_records",
        lambda *a, **kw: (
            records,
            registry(),
            dict(planned_cases=3, accepted=3),
            dict(fixture_only=True),
        ),
    )
    monkeypatch.setattr(ag, "_pairs", lambda *a: 0)

    def audit(records, root, **kwargs):
        calls.append(str(root))
        result = dict(
            passed=True,
            actual_data_verified=False,
            fixture_only=True,
            scope="lr108",
        )
        return result

    monkeypatch.setattr(ag, "_audit_compiled", audit)
    write_new(tmp_path / "fixture-index", {"fixture_only": True})
    result = ag.aggregate(
        tmp_path / "fixture-index",
        tmp_path / "aggregate",
        scope="lr108",
        producer=producer(),
        protection=registry(),
        policy_bundle={},
    )
    assert len(calls) == 1
    assert sha(Path(result["receipt"]["path"])) == result["receipt"]["sha256"]
    assert (
        sha(Path(result["independent_audit"]["path"]))
        == result["independent_audit"]["sha256"]
    )
    assert (
        json_file(tmp_path / "aggregate/progress.json")["whole_task_complete"]
        is False
    )


def test_frozen_main_and_lr_are_same_role_exact_subset_without_mutation():
    checked = check_scope()
    assert len(checked["cases"]) == 1674 and len(checked["lr_ids"]) == 108
    assert (
        checked["lr"]["native_phase_policy"] == "joint-native-consumption-c2-v1"
    )


@pytest.mark.parametrize(
    "states",
    [
        ["failed", "completed"],
        ["failed", "failed", "completed"],
        ["failed", "failed", "failed"],
    ],
)
def test_retry_chain_keeps_original_failure_budget_and_next_case(
    tmp_path, states
):
    allids = set()
    case = dict(case_id="fixture-case", seed=7)
    attempts = []
    for i, state in enumerate(states, 1):
        attempts.append(
            dict(
                index=i,
                run_id="run" + str(i),
                seed=7,
                parent_attempt_id="run" + str(i - 1) if i > 1 else None,
                status=state,
                lineage=["group:fixture-case"],
                failure_reason="preserved" if state == "failed" else None,
            )
        )
    ref = write_new(
        tmp_path / "ledger.json",
        dict(
            schema="drone-ai.collection19.case-ledger/1",
            case_id="fixture-case",
            attempts=attempts,
        ),
    )
    _, counts = ag._ledger(ref, case, producer(), allids, {})
    assert counts["physical_attempts"] == len(states) and counts[
        "physical_failed_attempts"
    ] == states.count("failed")
    nextcase = dict(case_id="next", seed=8)
    ref = write_new(
        tmp_path / "next.json",
        dict(
            schema="drone-ai.collection19.case-ledger/1",
            case_id="next",
            attempts=[
                dict(
                    index=1,
                    run_id="next-run",
                    seed=8,
                    parent_attempt_id=None,
                    status="completed",
                    lineage=["group:next"],
                )
            ],
        ),
    )
    _, after = ag._ledger(ref, nextcase, producer(), allids, {})
    assert after["physical_attempts"] == 1


@pytest.mark.parametrize(
    "damage", ["four", "parent", "seed", "completed_reflight"]
)
def test_retry_guard_rejects_budget_or_identity_changes(tmp_path, damage):
    a = dict(
        index=1,
        run_id="one",
        seed=7,
        parent_attempt_id=None,
        status="failed",
        lineage=["group:g"],
        failure_reason="kept",
    )
    attempts = [
        a,
        dict(
            a,
            index=2,
            run_id="two",
            parent_attempt_id="one",
            status="completed",
        ),
    ]
    if damage == "four":
        attempts = attempts * 2
    if damage == "parent":
        attempts[1]["parent_attempt_id"] = None
    if damage == "seed":
        attempts[1]["seed"] = 8
    if damage == "completed_reflight":
        attempts[0]["status"] = "completed"
    ref = write_new(
        tmp_path / "ledger.json",
        dict(
            schema="drone-ai.collection19.case-ledger/1",
            case_id="fixture",
            attempts=attempts,
        ),
    )
    with pytest.raises(ValueError):
        ag._ledger(ref, dict(case_id="fixture", seed=7), producer(), set(), {})


def test_zip64_writer_and_moments_match_independent_small_array(tmp_path):
    array = np.arange(131 * 19, dtype=np.float64).reshape(131, 19)
    moments = Moments()

    def blocks():
        for i in range(0, len(array), 17):
            part = array[i : i + 17]
            moments.add(part)
            yield part

    p = tmp_path / "stream.npz"
    write_npz(p, dict(values=(array.shape, "<f8", blocks())))
    read = []
    for _, part in array_blocks(p, "values"):
        read.extend(part.tolist())
    np.testing.assert_array_equal(read, array)
    mean, scale = moments.finish()
    np.testing.assert_allclose(mean, array.mean(0), atol=1e-12)
    np.testing.assert_allclose(scale, array.std(0), atol=1e-12)


def test_large_whole_input_is_sharded_and_existing_iterator_keeps_small_loader_guard(
    tmp_path,
):
    from learning.run_health19.data import load_manifest

    records = []
    for i in range(66):
        role = "normal_validation" if i == 65 else "normal_train"
        record, _ = export(
            tmp_path / ("case" + str(i)), "fixture-" + str(i), role, i * 0.01
        )
        records.append(record)
    root, _ = ag._compile(
        records,
        tmp_path / "many",
        scope="main",
        producer=producer(),
        registry=registry(),
        fixture_only=True,
    )
    paths = ag.whole_manifest_paths(root)
    assert len(paths) == 2
    assert (
        len(json_file(paths[0])["runs"]) == 64
        and len(json_file(paths[1])["runs"]) == 2
    )
    with pytest.raises(ValueError, match="iter_manifests"):
        load_manifest(
            paths[0], expected_producer=producer(), protection=registry()
        )
    count = 0
    for run in iter_manifests(
        paths, expected_producer=producer(), protection=registry()
    ):
        count += 1
        assert len(run.values) == 131
        del run
    assert count == 66


def test_canonical_stats_never_include_calibration_or_final_data(
    records, tmp_path
):
    calibration, _ = export(
        tmp_path / "cal", "fixture-cal", "normal_calibration", 99999
    )
    final, _ = export(
        tmp_path / "final",
        "fixture-final",
        "normal_test",
        999999,
        row_override=dict(cohort="final", purpose="independent_final_test"),
    )
    records = records + [calibration, final]
    root, prov = ag._compile(
        records,
        tmp_path / "allroles",
        scope="main",
        producer=producer(),
        registry=registry(),
        fixture_only=True,
    )
    assert prov["unique_valid_mission_frames"] == 262 and prov[
        "fit_run_ids"
    ] == [
        "fixture-t0",
        "fixture-t1",
    ]
    checked = ag._audit_compiled(
        records, root, scope="main", producer=producer(), registry=registry()
    )
    assert checked["passed"] and checked["fit_run_ids"] == [
        "fixture-t0",
        "fixture-t1",
    ]


def test_preserved_native_pair_proof_is_auditable_after_original_raw_paths_are_retired(
    tmp_path,
):
    proof = dict(
        schema="drone.joint-phase-pair-check/1",
        status="passed",
        errors=[],
        source_sha256=producer()["source_manifest_sha256"],
        runtime_sha256=producer()["runtime_sha256"],
        policy=dict(
            revision="joint-native-consumption-c2-v2", policy_sha256="e" * 64
        ),
        original_interval_us=[0, 480000000],
        interpolation_used=False,
        wind=dict(
            samples=480001,
            expected_samples=480001,
            missing_samples=0,
            max_vector_difference_mps=0.0,
            rms_vector_difference_mps=0.0,
        ),
        reference=dict(
            rows=4801,
            missing_rows=0,
            max_position_difference_m=0.0,
            max_velocity_difference_mps=0.0,
            max_circular_yaw_difference_rad=0.0,
        ),
        wind_function_checks=[
            dict(
                status="passed",
                mission_samples=480001,
                max_abs_component_error_mps=0.0,
            )
        ]
        * 2,
        normal_run="/retired/raw/normal-id",
        anomaly_run="/retired/raw/anomaly-id",
        native_consumption=dict(normal={"fixture": 1}, anomaly={"fixture": 2}),
        origins=dict(normal=60000000, anomaly=60180000),
    )
    p = tmp_path / "pair.json"
    ref = write_new(p, proof)
    policy = tmp_path / "policy.json"
    write_new(
        policy,
        dict(
            native_policy_sha256="e" * 64,
            native_adapter="joint-native-consumption-c2-v2",
        ),
    )
    records = []
    for role, rid, native in [
        ("normal_test", "normal-id", 1),
        ("anomaly_test", "anomaly-id", 2),
    ]:
        records.append(
            dict(
                record=dict(pair_evidence=ref),
                row=dict(
                    paired_case_id="pair",
                    run_id=rid,
                    role=role,
                    mission_interval=dict(
                        start_us=proof["origins"][
                            "anomaly" if native == 2 else "normal"
                        ]
                    ),
                ),
                audit=dict(native_check={"fixture": native}),
            )
        )
    assert ag._pairs(records, producer(), dict(paired_policy=str(policy))) == 1
    proof["wind"]["missing_samples"] = 1
    p.write_text(json.dumps(proof), encoding="utf-8")
    bad = dict(path=str(p), sha256=sha(p))
    for record in records:
        record["record"]["pair_evidence"] = bad
    with pytest.raises(ValueError):
        ag._pairs(records, producer(), dict(paired_policy=str(policy)))
