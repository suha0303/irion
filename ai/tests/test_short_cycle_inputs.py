"""Short180 schema/value fixtures only; never actual flights or model
execution.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from test_pilot45_admission import make_request
from test_pilot_publisher import fixture as native_fixture
from test_wind_metadata import fixture as wind_fixture

from capstone_ai.mapping import FEATURES, TARGETS
from learning.admission19 import SHORT
from learning.history19 import joint_native, short_cycle
from learning.history19.pilot import CRITICAL, ROOT, json_file, sha
from learning.history19.pilot_connector import native_mission_interval


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return dict(path=str(path.absolute()), sha256=sha(path))


def refresh(raw):
    integrity = json_file(raw / "integrity.json")
    for name in integrity:
        p = raw / name
        integrity[name] = dict(sha256=sha(p), size=p.stat().st_size)
    write(raw / "integrity.json", integrity)


def native180(folder):
    args = native_fixture(folder)
    native = json_file(
        ROOT / "simulation/profiles/collection/joint-phase-policy.json"
    )
    assert native["revision"] == "joint-native-consumption-c2-v2"
    write(args["native_policy"], native)
    for i in range(3):
        report = json_file(args["pair_checks"][i])
        report["policy"].update(
            revision=native["revision"],
            policy_sha256=sha(args["native_policy"]),
        )
        report["original_interval_us"] = [0, 180000000]
        report["wind"].update(samples=180001, expected_samples=180001)
        report["reference"]["rows"] = 1801
        for item in report["wind_function_checks"]:
            item["mission_samples"] = 180001
        for role in ("normal", "anomaly"):
            raw = Path(report[role + "_run"])
            start = json_file(raw / joint_native.ORIGIN)
            first = start["actual_start_us"]
            start.update(
                revision=native["revision"],
                policy_sha256=sha(args["native_policy"]),
                duration_us=180000000,
            )
            end = dict(
                start,
                kind="actual_auto_land_reference_consumption",
                actual_time_us=first + 180024000,
            )
            check = json_file(raw / joint_native.CHECK)
            check.update(
                actual_landing_us=end["actual_time_us"],
                actual_duration_s=180.024,
                duration_us=180000000,
                reference_rows_seen=1801,
                same_source_time_consumptions=2,
                unique_source_time_samples=120000,
                backward_source_time_count=0,
                conflicting_same_source_time_count=0,
            )
            scenario = json_file(raw / "evaluation/scenario.json")
            scenario.update(requested_duration_s=180)
            scenario["joint_phase_policy"].update(
                revision=native["revision"],
                policy_sha256=sha(args["native_policy"]),
                duration_us=180000000,
                reference_rows=1801,
                expected_weather_ticks=180001,
                actual_start=start,
                actual_landing=end,
            )
            for name, value in (
                (joint_native.ORIGIN, start),
                (joint_native.LANDING, end),
                (joint_native.CHECK, check),
                ("evaluation/scenario.json", scenario),
                (
                    "evaluation/execution.json",
                    [
                        dict(state="mission", sim_time=first / 1e6),
                        dict(
                            state="landing",
                            sim_time=end["actual_time_us"] / 1e6,
                        ),
                    ],
                ),
            ):
                write(raw / name, value)
            refresh(raw)
            report["native_consumption"][role] = check
        write(args["pair_checks"][i], report)
    policy_path = folder / "short-policy.json"
    policy = short_cycle.prepare_policy(
        args["native_policy"],
        args["producer_binding"],
        policy_path,
        fixture_only=True,
    )
    return args, policy_path, policy


def short_request(folder, *, phase="case"):
    folder.mkdir()
    args, policy_path, policy = native180(folder / "native")
    path = make_request(folder / "request", phase=phase)
    request = json_file(path)
    refs = request["refs"]
    request.update(
        schema="drone-ai.short-cycle.check/1",
        task_id=short_cycle.TASK,
        purpose=SHORT,
    )
    for key in ("native_completion", "development_receipt"):
        refs.pop(key)
    for key, keyarg in (
        ("expected_producer", "producer_binding"),
        ("source_manifest", "source_manifest"),
        ("runtime_manifest", "runtime_manifest"),
        ("native_policy", "native_policy"),
    ):
        refs[key] = dict(
            path=str(args[keyarg].absolute()), sha256=sha(args[keyarg])
        )
    refs["paired_policy"] = dict(
        path=str(policy_path.absolute()), sha256=sha(policy_path)
    )

    def update_ref(key, value):
        old = refs[key]
        p = path.parent / old["path"]
        refs[key] = write(p, value)

    manifest = json_file(path.parent / refs["pilot_manifest"]["path"])
    manifest.update(task_id=short_cycle.TASK, purpose=SHORT, cohort=SHORT)
    update_ref("pilot_manifest", manifest)
    ledger = json_file(path.parent / refs["ledger"]["path"])
    ledger["task_id"] = short_cycle.TASK
    raw = folder / "native/synthetic-0-normal"
    state = json_file(raw / "run.json")
    rid = state["run_id"]
    ledger["cases"][0]["attempts"][0]["run_id"] = rid
    update_ref("ledger", ledger)
    registry = json_file(path.parent / refs["registry"]["path"])
    registry["entries"].append(
        dict(
            identities=["run:" + rid],
            forbidden_uses=["fit", "calibration", "final_test"],
        )
    )
    update_ref("registry", registry)
    capsule = json_file(path.parent / refs["capsule"]["path"])
    capsule.update(
        producer=policy["producer"],
        paired_policy_sha256=sha(policy_path),
        Data_entrypoint="Data/generation/pretrain/common.py",
        files={
            n: sha(ROOT / n)
            for n in (*CRITICAL, "Data/generation/pretrain/common.py")
        },
    )
    update_ref("capsule", capsule)
    scenario = wind_fixture("calm", "altitude")
    scenario.update(
        seed=manifest["cases"][0]["seed"],
        producer_lock_sha256=policy["producer"]["runtime_sha256"],
        health_eligibility="confirmed_healthy",
        requested_duration_s=180,
        joint_phase_policy=json_file(raw / "evaluation/scenario.json")[
            "joint_phase_policy"
        ],
    )
    scenario["wind_model"]["clock"] = policy["wind_model_clock"]
    write(raw / "evaluation/scenario.json", scenario)
    for key, member in (
        ("battery_selection", "evaluation/battery-selection.json"),
        ("propulsion_selection", "evaluation/propulsion-selection.json"),
    ):
        value = json_file(path.parent / refs[key]["path"])
        refs[key] = write(raw / member, value)
    integrity = json_file(raw / "integrity.json")
    for member in (
        "evaluation/battery-selection.json",
        "evaluation/propulsion-selection.json",
    ):
        p = raw / member
        integrity[member] = dict(sha256=sha(p), size=p.stat().st_size)
    write(raw / "integrity.json", integrity)
    refresh(raw)
    interval = native_mission_interval(
        raw / joint_native.ORIGIN,
        policy_path=policy_path,
        native_policy_path=args["native_policy"],
        run_path=raw / "run.json",
    )
    for key, member in (
        ("scenario", "evaluation/scenario.json"),
        ("raw_integrity", "integrity.json"),
    ):
        refs[key] = dict(
            path=str((raw / member).absolute()), sha256=sha(raw / member)
        )
    request["case_origin"] = {
        key: dict(path=str((raw / member).absolute()), sha256=sha(raw / member))
        for key, member in (
            ("run", "run.json"),
            ("execution", "evaluation/execution.json"),
            ("origin", joint_native.ORIGIN),
        )
    }
    online = json_file(path.parent / refs["online_manifest"]["path"])
    run = json_file(path.parent / refs["run_manifest"]["path"])
    row = online["run"]
    row.update(
        run_id=rid, purpose=SHORT, cohort=SHORT, mission_interval=interval
    )
    row["source_evidence"].update(
        producer_runtime_sha256=policy["producer"]["runtime_sha256"]
    )
    times = np.arange(60000000, 240024000, 100000, dtype=np.int64)
    physical = np.tile(np.arange(19) * 0.01, (len(times), 1))
    physical[:, 17:] = [16, 24]
    canonical = path.parent / "short-frames.jsonl"
    with canonical.open("w", encoding="utf-8") as stream:
        for t, values in zip(times, physical):
            stream.write(
                json.dumps(
                    dict(
                        run_id=rid,
                        clock_id="px4-boot",
                        segment_id=0,
                        time_us=int(t),
                        feature_profile="common19_ctx0",
                        feature_order=FEATURES,
                        values=dict(zip(FEATURES, values.tolist())),
                        quality=dict.fromkeys(FEATURES, "valid"),
                        operating_state="airborne",
                        lineage=dict(
                            sample=dict(
                                sample_time_us=int(t), available_time_us=int(t)
                            )
                        ),
                    )
                )
                + "\n"
            )
    refs["canonical_frames"] = dict(
        path=str(canonical.absolute()), sha256=sha(canonical)
    )
    rnpz = path.parent / "short-run.npz"
    np.savez(rnpz, values=physical, time_us=times)
    run.update(
        producer=policy["producer"],
        runs=[dict(row, path=rnpz.name, sha256=sha(rnpz))],
    )
    update_ref("run_manifest", run)
    origins = np.arange(70000000, 240000000, 1000000, dtype=np.int64)
    at = {int(t): i for i, t in enumerate(times)}
    values = physical.copy()
    values[:, 17] /= 4
    values[:, 18] /= 12
    X = np.array(
        [
            [values[at[int(t) + j * 100000]] for j in range(-99, 1)]
            for t in origins
        ]
    )
    Y = np.array(
        [
            [
                values[at[int(t) + j * 100000]][
                    [FEATURES.index(n) for n in TARGETS]
                ]
                for j in range(1, 11)
            ]
            for t in origins
        ]
    )
    onpz = path.parent / "short-online.npz"
    np.savez(
        onpz,
        X=X,
        Y=Y,
        origin_us=origins,
        segment_id=np.zeros(len(origins), dtype=np.int64),
        run_id=np.full(len(origins), rid),
        clock_id=np.full(len(origins), "px4-boot"),
    )
    online.update(
        producer=policy["producer"],
        run=row,
        file=dict(path=onpz.name, sha256=sha(onpz)),
    )
    update_ref("online_manifest", online)
    write(path, request)
    return path


def test_complete_short180_case_has_positive_full_frames_windows_and_tail(
    tmp_path, capsys
):
    path = short_request(tmp_path / "unit")
    code = short_cycle.main(path)
    receipt = json.loads(capsys.readouterr().out)
    assert (
        code == 0
        and receipt["status"] == "SYNTHETIC_SHORT180_CASE_INPUTS_CHECKED_NO_FIT"
    )
    assert (
        receipt["frames"] == 1801
        and receipt["online"]["windows"] == 170
        and receipt["tail_frames"] == 1
    )
    assert (
        receipt["actual_data_verified"] is False
        and receipt["expected_case_denominator"] == 45
    )


def test_short_static_gate_has_no_native6_dependency(tmp_path, capsys):
    path = short_request(tmp_path / "unit", phase="prerequisites")
    code = short_cycle.main(path)
    r = json.loads(capsys.readouterr().out)
    assert (
        code == 0
        and r["status"] == "SYNTHETIC_SHORT180_STATIC_INPUTS_CHECKED_NO_FLIGHTS"
    )


@pytest.mark.parametrize(
    "damage",
    [
        "reference_rows",
        "requested_duration_s",
        "receipt_duration",
        "counter_conflict",
        "gap",
        "revision",
    ],
)
def test_short_native_mismatch_is_rejected_without_cropping_or_relabelling(
    tmp_path, damage
):
    args, policy_path, policy = native180(tmp_path / "unit")
    raw = tmp_path / "unit/synthetic-0-normal"
    if damage in {"reference_rows", "counter_conflict", "gap"}:
        p = raw / joint_native.CHECK
        v = json_file(p)
        v.update(
            **(
                {"reference_rows_seen": 4801}
                if damage == "reference_rows"
                else {"conflicting_same_source_time_count": 1}
                if damage == "counter_conflict"
                else {"max_consumption_gap_us": 40001}
            )
        )
    elif damage == "requested_duration_s":
        p = raw / "evaluation/scenario.json"
        v = json_file(p)
        v["requested_duration_s"] = 480
    else:
        p = raw / joint_native.ORIGIN
        v = json_file(p)
        v.update(
            **(
                {"duration_us": 480000000}
                if damage == "receipt_duration"
                else {"revision": "joint-native-consumption-c2-v1"}
            )
        )
    write(p, v)
    refresh(raw)
    with pytest.raises(ValueError):
        joint_native.origin(raw / "run.json", policy, fixture=True)


def test_short_pair_uses_all180001_weather_ticks(tmp_path):
    args, pp, policy = native180(tmp_path / "unit")
    assert joint_native.pair(args["pair_checks"][0], policy, fixture=True)[
        "passed"
    ]
    p = args["pair_checks"][0]
    v = json_file(p)
    v["wind"]["samples"] -= 1
    write(p, v)
    with pytest.raises(ValueError, match="coverage"):
        joint_native.pair(p, policy, fixture=True)


def test_short_final_cannot_pass_with_missing_actual_case_receipts(
    tmp_path, capsys
):
    path = short_request(tmp_path / "unit", phase="final")
    assert short_cycle.main(path) == 2
    assert (
        json.loads(capsys.readouterr().out)["status"] == "BLOCKED_PENDING_INPUT"
    )


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "drop_case",
        "empty_windows",
        "missing_pair",
        "reused_pair",
        "stale_scope",
    ],
)
def test_final45_and_nine_pairs_are_required_even_for_a_synthetic_campaign(
    tmp_path, capsys, damage
):
    import hashlib

    path = short_request(tmp_path / "unit", phase="final")
    r = json_file(path)
    plan = json_file(r["refs"]["pilot_manifest"]["path"])
    ledger = json_file(r["refs"]["ledger"]["path"])
    registry = json_file(r["refs"]["registry"]["path"])
    receipts = []
    for index, case in enumerate(plan["cases"]):
        rid = "synthetic-final-" + str(index)
        ledger["cases"][index].update(
            status="completed",
            attempts=[
                dict(
                    index=1,
                    run_id=rid,
                    parent_attempt_id=None,
                    status="completed",
                    seed=case["seed"],
                    lineage=case["lineage"] + ["run:" + rid],
                )
            ],
        )
        registry["entries"].append(
            dict(
                identities=["run:" + rid],
                forbidden_uses=["fit", "calibration", "final_test"],
            )
        )
        evidence = {
            key: r["refs"][key]["sha256"]
            for key in (
                "source_manifest",
                "runtime_manifest",
                "paired_policy",
                "capsule",
            )
        }
        if case["case_kind"] != "normal_base":
            evidence["pilot_pair_native_evidence"] = hashlib.sha256(
                case["pair_id"].encode()
            ).hexdigest()
        # Real-receipt-shaped metadata exercises final aggregation only. The
        # outer
        # request stays fixture_only=True and can never produce actual
        # acceptance.
        record = dict(
            task_id=short_cycle.TASK,
            purpose=SHORT,
            phase="case",
            status="SHORT180_CASE_INPUTS_CHECKED_NO_FIT",
            fixture_only=False,
            actual_data_verified=True,
            exit_code=0,
            model_forward_steps=0,
            case_id=case["case_id"],
            run_id=rid,
            frames=1801,
            online=dict(windows=170),
            duration_profile="short180",
            mission_duration_us=180000000,
            pair_verified=case["case_kind"] != "normal_base",
            evidence_sha256=evidence,
        )
        if damage == "empty_windows" and index == 0:
            record["online"]["windows"] = 0
        if damage == "missing_pair" and case["case_kind"] == "anomaly":
            record["pair_verified"] = False
        if damage == "reused_pair" and case["case_kind"] != "normal_base":
            evidence["pilot_pair_native_evidence"] = "0" * 64
        if damage == "stale_scope" and index == 0:
            record["task_id"] = "P27_PILOT45_20261107"
        receipts.append(
            write(path.parent / ("receipt-" + str(index) + ".json"), record)
        )
    r["refs"]["ledger"] = write(Path(r["refs"]["ledger"]["path"]), ledger)
    r["refs"]["registry"] = write(Path(r["refs"]["registry"]["path"]), registry)
    r["case_receipts"] = receipts[:-1] if damage == "drop_case" else receipts
    write(path, r)
    code = short_cycle.main(path)
    result = json.loads(capsys.readouterr().out)
    assert code == (0 if damage is None else 1)
    assert result["actual_data_verified"] is False
    if damage is None:
        assert result["cases_checked"] == 45 and result["pairs_checked"] == 9
