"""Synthetic consumer projections only; never native6, actual pilot, fit or forward."""

import copy
import json

import numpy as np
import pytest
from test_wind_metadata import fixture as wind_fixture

from capstone_ai.mapping import FEATURES, TARGETS
from learning.admission19 import (
    MISSION_POLICY,
    ONLINE_UNITS,
    PILOT,
    check_protection,
    partition,
)
from learning.history19 import pilot
from learning.history19.pilot_connector import (
    native_mission_interval,
    prepare_request,
)
from learning.wind_metadata import KINDS


def make_request(folder, *, phase="case", paired=False):
    folder.mkdir()

    def put(name, value):
        path = folder / (name + ".json")
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return dict(path=path.name, sha256=pilot.sha(path))

    source = put("source", dict(files={"unit.py": "a" * 64}))
    runtime = put("runtime", dict(source_manifest_sha256=source["sha256"]))
    producer = dict(
        battery_policy="cmu-vah01-rpt16-only/1",
        battery_revision="cmu-vah01-baseline/1",
        catalog_sha256="a" * 64,
        propulsion_revision="x500-propulsion-binary-v1",
        propulsion_catalog_sha256="b" * 64,
        source_manifest_sha256=source["sha256"],
        runtime_sha256=runtime["sha256"],
    )
    native_policy = put(
        "native-policy", dict(schema="unit.native.policy/1", fixture_only=True)
    )
    policy = dict(
        schema="drone-ai.paired-source-clock-policy/1",
        status="FROZEN",
        approved_for_pilot=True,
        producer=producer,
        policy_id="synthetic-only",
        wind_model_clock="unit_source_clock",
        mission_policy=MISSION_POLICY,
        alignment_mode="producer_source_clock",
        posthoc_adjustment=False,
        native_policy_sha256=native_policy["sha256"],
        native_receipt_schema="unit.native.origin/1",
        native_receipt_field_paths={k: [k] for k in pilot.ORIGIN_FIELDS},
        native_comparison_field_paths={k: [k] for k in pilot.COMPARISON_FIELDS},
        limits=dict(
            origin_delta_us=0,
            wind_rms_mps=0.0,
            wind_peak_mps=0.0,
            receipt_min_offset_us=0,
            receipt_max_offset_us=0,
        ),
    )
    policy_ref = put("policy", policy)
    cases = []

    def case(i, kind, motion, environment, pair_id, seed):
        cid = "case-" + str(i)
        return dict(
            case_id=cid,
            role="anomaly_test" if kind == "anomaly" else "normal_test",
            case_kind=kind,
            motion=motion,
            environment=environment,
            pair_id=pair_id,
            seed=seed,
            operation_sha256="d" * 64,
            lineage=[
                "case:" + cid,
                "pair:" + pair_id,
                "group:" + pair_id,
                "retry:" + cid,
            ],
        )

    for kind in sorted(KINDS):
        for env in ("calm", "low", "strong"):
            i = len(cases)
            cases.append(
                case(i, "normal_base", kind, env, "base-" + str(i), 1000 + i)
            )
    for i in range(9):
        cases.append(
            case(
                27 + i,
                "paired_normal",
                sorted(KINDS)[i],
                ("calm", "low", "strong")[i % 3],
                "pair-" + str(i),
                2000 + i,
            )
        )
    for i in range(9):
        c = case(
            36 + i,
            "anomaly",
            sorted(KINDS)[i],
            ("calm", "low", "strong")[i % 3],
            "pair-" + str(i),
            2000 + i,
        )
        c.update(
            health_family=("battery", "propulsion", "combined")[i // 3],
            timing=("existing", "step", "ramp")[i % 3],
        )
        cases.append(c)
    chosen = cases[27 if paired else 0]
    registry = dict(
        schema="drone-ai.global-protection/1",
        entries=[
            dict(
                identities=["case:" + c["case_id"]],
                forbidden_uses=["fit", "calibration", "final_test"],
            )
            for c in cases
        ],
    )
    ledger = dict(
        schema="drone-ai.pilot45.ledger/1",
        task_id=pilot.TASK,
        cases=[
            dict(case_id=c["case_id"], status="pending", attempts=[])
            for c in cases
        ],
    )

    def member(rid):
        run = put(
            rid + "-run",
            dict(
                run_id=rid,
                status="completed",
                origin="synthetic_unit_fixture",
                clock=dict(id="unit-clock", unit="us"),
            ),
        )
        execution = put(
            rid + "-execution",
            [
                dict(state="mission", sim_time=1.0),
                dict(state="landing", sim_time=22.0),
            ],
        )
        origin = dict(
            schema=policy["native_receipt_schema"],
            run_id=rid,
            clock_id="unit-clock",
            policy_id=policy["policy_id"],
            source_manifest_sha256=producer["source_manifest_sha256"],
            runtime_sha256=producer["runtime_sha256"],
            actual_mission_start_us=1000000,
            actual_landing_start_us=22000000,
            weather_origin_us=1000000,
            path_origin_us=1000000,
            actual_source_ack_us=1000000,
            source_acknowledged=True,
        )
        return dict(
            run=run, execution=execution, origin=put(rid + "-origin", origin)
        )

    members = [member("dev" + str(i)) for i in range(6)]
    comparisons = []
    for i in range(3):
        metrics = dict(origin_delta_us=0, wind_rms_mps=0.0, wind_peak_mps=0.0)
        comparisons.append(
            dict(
                passed=True,
                normal_run_id="dev" + str(i * 2),
                anomaly_run_id="dev" + str(i * 2 + 1),
                native_evidence=put(
                    "metrics" + str(i),
                    dict(
                        **metrics,
                        passed=True,
                        normal_run_id="dev" + str(i * 2),
                        anomaly_run_id="dev" + str(i * 2 + 1),
                        policy_id=policy["policy_id"],
                        source_manifest_sha256=producer[
                            "source_manifest_sha256"
                        ],
                        runtime_sha256=producer["runtime_sha256"],
                    ),
                ),
                **metrics,
            )
        )
    development = put(
        "development",
        dict(
            schema="drone-ai.paired-development-evidence/1",
            status="PASSED",
            producer=producer,
            paired_policy_sha256=policy_ref["sha256"],
            members=members,
            comparisons=comparisons,
        ),
    )
    registry["entries"] += [
        dict(
            identities=["run:dev" + str(i)],
            forbidden_uses=[
                "pilot_input_check",
                "fit",
                "calibration",
                "final_test",
            ],
        )
        for i in range(6)
    ]
    rid = "pilot-raw"
    registry["entries"] += [
        dict(
            identities=["run:" + rid],
            forbidden_uses=[
                "pilot_input_check",
                "fit",
                "calibration",
                "final_test",
            ],
        )
        for rid in pilot.json_file(pilot.HISTORICAL_EXCLUSIONS)[
            "raw_attempt_ids"
        ]
    ]
    case_member = member(rid)
    current = next(
        r for r in ledger["cases"] if r["case_id"] == chosen["case_id"]
    )
    current.update(
        status="completed",
        attempts=[
            dict(
                index=1,
                run_id=rid,
                parent_attempt_id=None,
                status="completed",
                seed=chosen["seed"],
                lineage=chosen["lineage"] + ["run:" + rid],
            )
        ],
    )
    registry["entries"].append(
        dict(
            identities=["run:" + rid],
            forbidden_uses=["fit", "calibration", "final_test"],
        )
    )
    capsule = put(
        "capsule",
        dict(
            schema="drone-ai.consumer-capsule/1",
            producer=producer,
            paired_policy_sha256=policy_ref["sha256"],
            Data_entrypoint="Data/generation/pretrain/ai19.py",
            files={
                n: pilot.sha(pilot.ROOT / n)
                for n in (*pilot.CRITICAL, "Data/generation/pretrain/ai19.py")
            },
        ),
    )
    request = dict(
        schema="drone-ai.pilot45.check/1",
        task_id=pilot.TASK,
        purpose=PILOT,
        phase=phase,
        fixture_only=True,
        case_id=chosen["case_id"],
        case_origin=case_member,
        refs=dict(
            expected_producer=put("producer", producer),
            source_manifest=source,
            runtime_manifest=runtime,
            paired_policy=policy_ref,
            native_policy=native_policy,
            native_completion=put(
                "native-completion",
                dict(
                    schema="unit.native.completion/1",
                    task_id=pilot.TASK,
                    passed=True,
                    allow_pilot45=True,
                    actual_runs=6,
                    passed_runs=6,
                    passed_pairs=3,
                    actual_run_ids=["dev" + str(i) for i in range(6)],
                    source_sha256=producer["source_manifest_sha256"],
                    runtime_sha256=producer["runtime_sha256"],
                    policy_sha256=native_policy["sha256"],
                ),
            ),
            capsule=capsule,
            development_receipt=development,
            registry=put("registry", registry),
            pilot_manifest=put(
                "manifest",
                dict(
                    schema="drone-ai.pilot45.manifest/1",
                    task_id=pilot.TASK,
                    purpose=PILOT,
                    cohort=PILOT,
                    no_fit=True,
                    status="FROZEN",
                    cases=cases,
                ),
            ),
            ledger=put("ledger", ledger),
        ),
    )
    scenario = wind_fixture(chosen["environment"], chosen["motion"])
    scenario["wind_model"]["clock"] = policy["wind_model_clock"]
    scenario.update(
        seed=chosen["seed"],
        producer_lock_sha256=producer["runtime_sha256"],
        health_eligibility="confirmed_healthy",
    )
    request["refs"]["scenario"] = put("scenario", scenario)
    battery = dict(
        catalog_revision=producer["battery_revision"],
        catalog_sha256=producer["catalog_sha256"],
        flight_policy=dict(revision=producer["battery_policy"]),
        initial_stage="rpt_00",
        target_stage="rpt_00",
        fixed_for_flight=True,
        transition_s=0.0,
    )
    propulsion = dict(
        catalog_revision=producer["propulsion_revision"],
        catalog_sha256=producer["propulsion_catalog_sha256"],
        initial_state="normal",
        target_state="normal",
        fixed_for_flight=True,
        transition_s=0.0,
    )
    request["refs"].update(
        battery_selection=put("battery", battery),
        propulsion_selection=put("propulsion", propulsion),
    )
    integrity = {}
    for alias, name in [
        ("run", "run.json"),
        ("execution", "evaluation/execution.json"),
    ]:
        ref = case_member[alias]
        integrity[name] = dict(
            size=(folder / ref["path"]).stat().st_size, sha256=ref["sha256"]
        )
    for alias, name in [
        ("scenario", "evaluation/scenario.json"),
        ("battery_selection", "evaluation/battery-selection.json"),
        ("propulsion_selection", "evaluation/propulsion-selection.json"),
    ]:
        ref = request["refs"][alias]
        integrity[name] = dict(
            size=(folder / ref["path"]).stat().st_size, sha256=ref["sha256"]
        )
    request["refs"]["raw_integrity"] = put("integrity", integrity)
    mission = dict(
        policy=MISSION_POLICY,
        start_us=1000000,
        end_us_exclusive=22000000,
        clock_id="unit-clock",
        source="evaluation/execution.json",
        source_sha256=case_member["execution"]["sha256"],
    )
    row = dict(
        run_id=rid,
        case_id=chosen["case_id"],
        role=chosen["role"],
        purpose=PILOT,
        cohort=PILOT,
        origin="synthetic_unit_fixture",
        health="confirmed_healthy",
        lineage=chosen["lineage"] + ["run:" + rid],
        mission_interval=mission,
        source_evidence=dict(
            scenario_name=scenario["name"],
            battery_selection=battery,
            propulsion_selection=propulsion,
            producer_runtime_sha256=producer["runtime_sha256"],
        ),
    )
    times = np.arange(1000000, 22000000, 100000, dtype=np.int64)
    values = np.tile(np.arange(19) * 0.01, (len(times), 1))
    values[:, 17:] = [16.0, 24.0]
    canonical = folder / "frames.jsonl"
    with canonical.open("w", encoding="utf-8") as f:
        for t, x in zip(times, values):
            frame = dict(
                run_id=rid,
                clock_id="unit-clock",
                segment_id=0,
                time_us=int(t),
                feature_order=FEATURES,
                feature_profile="common19_ctx0",
                quality=dict.fromkeys(FEATURES, "valid"),
                values=dict(zip(FEATURES, x.tolist())),
                operating_state="airborne",
                lineage=dict(
                    sample=dict(sample_time_us=int(t), available_time_us=int(t))
                ),
            )
            f.write(json.dumps(frame) + "\n")
    request["refs"]["canonical_frames"] = dict(
        path=canonical.name, sha256=pilot.sha(canonical)
    )
    run_npz = folder / "run.npz"
    np.savez(run_npz, values=values, time_us=times)
    run_row = {**row, "path": run_npz.name, "sha256": pilot.sha(run_npz)}
    request["refs"]["run_manifest"] = put(
        "run-manifest",
        dict(
            schema="drone-ai.run-health19.dataset/1",
            feature_order=FEATURES,
            interval_policy=MISSION_POLICY,
            units="physical_common19_pack_V_A",
            step_us=100000,
            vehicle_profile="x500_reference",
            battery_policy=producer["battery_policy"],
            producer=producer,
            runs=[run_row],
        ),
    )
    origins = np.arange(11, 21, dtype=np.int64) * 1000000
    transformed = values.copy()
    transformed[:, 17] /= 4
    transformed[:, 18] /= 12
    at = {int(t): i for i, t in enumerate(times)}
    X = np.array(
        [
            [transformed[at[int(t) + j * 100000]] for j in range(-99, 1)]
            for t in origins
        ]
    )
    Y = np.array(
        [
            [
                transformed[at[int(t) + j * 100000]][
                    [FEATURES.index(n) for n in TARGETS]
                ]
                for j in range(1, 11)
            ]
            for t in origins
        ]
    )
    online = folder / "online.npz"
    np.savez(
        online,
        X=X,
        Y=Y,
        origin_us=origins,
        segment_id=np.zeros(len(origins), dtype=np.int64),
        run_id=np.full(len(origins), rid),
        clock_id=np.full(len(origins), "unit-clock"),
    )
    request["refs"]["online_manifest"] = put(
        "online-manifest",
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
            producer=producer,
            run=row,
            file=dict(path=online.name, sha256=pilot.sha(online)),
        ),
    )
    path = folder / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def edit(path, name, change):
    request = json.loads(path.read_text(encoding="utf-8"))
    ref = request["refs"][name]
    p = path.parent / ref["path"]
    obj = json.loads(p.read_text(encoding="utf-8"))
    change(obj)
    p.write_text(json.dumps(obj), encoding="utf-8")
    ref["sha256"] = pilot.sha(p)
    path.write_text(json.dumps(request), encoding="utf-8")
    return obj


def result(path, capsys):
    code = pilot.main(path)
    value = json.loads(capsys.readouterr().out)
    assert value["exit_code"] == code and value["eligible_for_fit"] is False
    assert (
        value["model_forward_steps"]
        == value["optimizer_steps"]
        == value["scaler_threshold_fits"]
        == 0
    )
    return code, value


def test_synthetic_prerequisites_never_start_real_pilot(tmp_path, capsys):
    path = make_request(tmp_path / "data", phase="prerequisites")
    code, r = result(path, capsys)
    assert (
        code == 0
        and r["status"] == "SYNTHETIC_PREREQUISITES_CHECKED_NO_CASE_DATA"
    )
    assert (
        r["actual_data_verified"] is False
        and r["usable_to_start_real_pilot"] is False
    )
    assert r["ledger_counts"]["planned_cases"] == 45


def test_case_both_views_compare_to_unscaled_canonical_without_model(
    tmp_path, capsys
):
    code, r = result(make_request(tmp_path / "data"), capsys)
    assert code == 0, r
    assert (
        r["frames"] == 210
        and r["initial100"] == 100
        and r["post_initial_frames"] == 110
        and r["tail_frames"] == 10
    )
    assert r["online"]["windows"] == 10 and r["online"]["target_points"] == 100


@pytest.mark.parametrize(
    "damage",
    [
        "missing_policy",
        "null_criterion",
        "missing_development",
        "null_actual_ack",
    ],
)
def test_missing_or_null_native_evidence_stays_pending(
    tmp_path, capsys, damage
):
    path = make_request(tmp_path / "data")
    if damage in {"missing_policy", "missing_development"}:
        r = json.loads(path.read_text(encoding="utf-8"))
        r["refs"][
            "paired_policy"
            if damage == "missing_policy"
            else "development_receipt"
        ] = None
        path.write_text(json.dumps(r), encoding="utf-8")
    elif damage == "null_criterion":
        edit(
            path,
            "paired_policy",
            lambda d: d["limits"].update(wind_rms_mps=None),
        )
    else:
        r = json.loads(path.read_text(encoding="utf-8"))
        ref = r["case_origin"]["origin"]
        p = path.parent / ref["path"]
        d = json.loads(p.read_text(encoding="utf-8"))
        d["actual_source_ack_us"] = None
        p.write_text(json.dumps(d), encoding="utf-8")
        ref["sha256"] = pilot.sha(p)
        path.write_text(json.dumps(r), encoding="utf-8")
    code, r = result(path, capsys)
    assert code == 2 and r["status"] == "BLOCKED_PENDING_INPUT"


@pytest.mark.parametrize(
    "damage",
    [
        "capsule",
        "drop_case",
        "reuse_success",
        "unprotected_pilot",
        "fake_pair_pass",
    ],
)
def test_capsule_denominators_lineage_and_global_exclusion_cannot_be_bypassed(
    tmp_path, capsys, damage
):
    path = make_request(tmp_path / "data")
    if damage == "capsule":
        edit(
            path,
            "capsule",
            lambda d: d["files"].update({pilot.CRITICAL[0]: "0" * 64}),
        )
    elif damage == "drop_case":
        edit(path, "ledger", lambda d: d["cases"].pop())
    elif damage == "reuse_success":

        def mutate(d):
            a = copy.deepcopy(d["cases"][0]["attempts"][0])
            a.update(
                index=2, run_id="regenerated", parent_attempt_id="pilot-raw"
            )
            d["cases"][0]["attempts"].append(a)

        edit(path, "ledger", mutate)
    elif damage == "unprotected_pilot":
        edit(path, "registry", lambda d: d["entries"].pop())
    elif damage == "fake_pair_pass":
        edit(
            path,
            "development_receipt",
            lambda d: d["comparisons"][0].update(wind_peak_mps=0.1),
        )
    code, r = result(path, capsys)
    assert code == 1 and r["status"] == "FAILED_INPUT_CHECK"


def test_prior_failure_remains_in_full_denominator(tmp_path, capsys):
    path = make_request(tmp_path / "data")
    request = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (path.parent / request["refs"]["pilot_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    case = manifest["cases"][1]

    def failed(d):
        d["cases"][1].update(
            status="failed",
            attempts=[
                dict(
                    index=1,
                    run_id="failed-raw",
                    parent_attempt_id=None,
                    status="failed",
                    failure_reason="synthetic preserved launch failure",
                    seed=case["seed"],
                    lineage=case["lineage"] + ["run:failed-raw"],
                )
            ],
        )

    edit(path, "ledger", failed)
    edit(
        path,
        "registry",
        lambda d: d["entries"].append(
            dict(
                identities=["run:failed-raw"],
                forbidden_uses=["fit", "calibration", "final_test"],
            )
        ),
    )
    code, r = result(path, capsys)
    assert (
        code == 0
        and r["ledger_counts"]["failed_attempts"] == 1
        and r["ledger_counts"]["planned_cases"] == 45
    )


@pytest.mark.parametrize(
    "damage", ["double_voltage", "gap", "future_available", "receipt_shift"]
)
def test_no_double_transform_zero_fill_shift_or_future_input(
    tmp_path, capsys, damage
):
    path = make_request(tmp_path / "data")
    request = json.loads(path.read_text(encoding="utf-8"))
    if damage == "double_voltage":
        mpath = path.parent / request["refs"]["run_manifest"]["path"]
        m = json.loads(mpath.read_text(encoding="utf-8"))
        npz = path.parent / m["runs"][0]["path"]
        with np.load(npz) as a:
            values = a["values"].copy()
            times = a["time_us"].copy()
        values[:, 17] /= 4
        np.savez(npz, values=values, time_us=times)
        m["runs"][0]["sha256"] = pilot.sha(npz)
        mpath.write_text(json.dumps(m), encoding="utf-8")
        request["refs"]["run_manifest"]["sha256"] = pilot.sha(mpath)
    elif damage in {"gap", "future_available"}:
        ref = request["refs"]["canonical_frames"]
        p = path.parent / ref["path"]
        rows = [
            json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
        ]
        if damage == "gap":
            rows.pop(110)
        else:
            rows[110]["lineage"]["sample"]["available_time_us"] = (
                rows[110]["time_us"] + 1
            )
        p.write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        ref["sha256"] = pilot.sha(p)
    else:
        ref = request["case_origin"]["origin"]
        p = path.parent / ref["path"]
        d = json.loads(p.read_text(encoding="utf-8"))
        d["actual_mission_start_us"] += 100000
        p.write_text(json.dumps(d), encoding="utf-8")
        ref["sha256"] = pilot.sha(p)
    path.write_text(json.dumps(request), encoding="utf-8")
    code, r = result(path, capsys)
    assert code == (3 if damage == "gap" else 1)


def test_paired_case_never_passes_without_completed_partner(tmp_path, capsys):
    code, r = result(make_request(tmp_path / "data", paired=True), capsys)
    assert code == 2 and r["status"] == "BLOCKED_PENDING_INPUT"


@pytest.mark.parametrize("damage", [None, "other_pair"])
def test_pair_receipt_binds_both_real_members_and_native_comparison(
    tmp_path, capsys, damage
):
    path = make_request(tmp_path / "data", paired=True)
    request = json.loads(path.read_text(encoding="utf-8"))

    def read(ref):
        return json.loads(
            (path.parent / ref["path"]).read_text(encoding="utf-8")
        )

    def put(name, obj):
        p = path.parent / (name + ".json")
        p.write_text(json.dumps(obj), encoding="utf-8")
        return dict(path=p.name, sha256=pilot.sha(p))

    producer = read(request["refs"]["expected_producer"])
    policy = read(request["refs"]["paired_policy"])
    cases = read(request["refs"]["pilot_manifest"])["cases"]
    partner = cases[36]
    raw_id = "independent-pilot-anomaly"
    edit(
        path,
        "ledger",
        lambda d: d["cases"][36].update(
            status="completed",
            attempts=[
                dict(
                    index=1,
                    run_id=raw_id,
                    parent_attempt_id=None,
                    status="completed",
                    seed=partner["seed"],
                    lineage=partner["lineage"] + ["run:" + raw_id],
                )
            ],
        ),
    )
    edit(
        path,
        "registry",
        lambda d: d["entries"].append(
            dict(
                identities=["run:" + raw_id],
                forbidden_uses=["fit", "calibration", "final_test"],
            )
        ),
    )
    request = json.loads(path.read_text(encoding="utf-8"))
    metrics = dict(origin_delta_us=0, wind_rms_mps=0.0, wind_peak_mps=0.0)
    native = dict(
        **metrics,
        passed=True,
        normal_run_id="pilot-raw",
        anomaly_run_id=raw_id,
        policy_id=policy["policy_id"],
        source_manifest_sha256=producer["source_manifest_sha256"],
        runtime_sha256=producer["runtime_sha256"],
    )
    if damage == "other_pair":
        native["anomaly_run_id"] = "some-other-flight"
    request["refs"]["pair_receipt"] = put(
        "pilot-pair",
        dict(
            schema="drone-ai.pilot-pair-check/1",
            producer=producer,
            paired_policy_sha256=request["refs"]["paired_policy"]["sha256"],
            pair_id="pair-0",
            normal_run_id="pilot-raw",
            anomaly_run_id=raw_id,
            passed=True,
            normal_origin_receipt_sha256=request["case_origin"]["origin"][
                "sha256"
            ],
            native_evidence=put("native-pilot-comparison", native),
            **metrics,
        ),
    )
    path.write_text(json.dumps(request), encoding="utf-8")
    code, receipt = result(path, capsys)
    assert code == (1 if damage else 0)
    assert receipt["expected_case_denominator"] == 45


def test_eligible_online_window_cannot_be_silently_omitted(tmp_path, capsys):
    path = make_request(tmp_path / "data")
    request = json.loads(path.read_text(encoding="utf-8"))
    ref = request["refs"]["online_manifest"]
    manifest_path = path.parent / ref["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays_path = path.parent / manifest["file"]["path"]
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key][:-1] for key in archive.files}
    np.savez(arrays_path, **arrays)
    manifest["file"]["sha256"] = pilot.sha(arrays_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ref["sha256"] = pilot.sha(manifest_path)
    path.write_text(json.dumps(request), encoding="utf-8")
    code, receipt = result(path, capsys)
    assert code == 1 and "selective omission" in receipt["reason"]


def test_historical77_exclusion_cannot_disappear(tmp_path, capsys):
    path = make_request(tmp_path / "data")
    old = pilot.json_file(pilot.HISTORICAL_EXCLUSIONS)["raw_attempt_ids"][0]
    edit(
        path,
        "registry",
        lambda d: d.update(
            entries=[
                r for r in d["entries"] if "run:" + old not in r["identities"]
            ]
        ),
    )
    code, receipt = result(path, capsys)
    assert code == 1 and "Development77+6" in receipt["reason"]


@pytest.mark.parametrize("passed", [True, False])
def test_path_connector_uses_actual_completion_and_never_repairs_false_receipts(
    tmp_path, capsys, passed
):
    source = make_request(tmp_path / "data")
    request = json.loads(source.read_text(encoding="utf-8"))
    if not passed:
        edit(
            source,
            "native_completion",
            lambda d: d.update(passed=False, allow_pilot45=False),
        )
        request = json.loads(source.read_text(encoding="utf-8"))
    bundle = dict(
        schema="drone-ai.pilot45.connector/1",
        task_id=pilot.TASK,
        phase="case",
        fixture_only=True,
        case_id=request["case_id"],
        paths={k: v["path"] for k, v in request["refs"].items()},
        case_origin={k: v["path"] for k, v in request["case_origin"].items()},
    )
    bundle_path = source.parent / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    output = source.parent / "prepared.json"
    if not passed:
        with pytest.raises(pilot.Pending, match="does not authorize"):
            prepare_request(bundle_path, output)
        assert not output.exists()
    else:
        report = prepare_request(bundle_path, output)
        assert (
            not report["actual_data_verified"]
            and report["status"] == "REQUEST_WRITTEN_NOT_CHECKED"
        )
        code, receipt = result(output, capsys)
        assert code == 0 and receipt["status"].startswith("SYNTHETIC_")


@pytest.mark.parametrize("damage", ["allow", "count", "source"])
def test_native_completion_cannot_be_replaced_by_a_passed_projection(
    tmp_path, capsys, damage
):
    path = make_request(tmp_path / "data")
    changes = {
        "allow": dict(allow_pilot45=None),
        "count": dict(actual_runs=0),
        "source": dict(source_sha256="0" * 64),
    }
    edit(path, "native_completion", lambda d: d.update(changes[damage]))
    code, receipt = result(path, capsys)
    assert code == (2 if damage == "allow" else 1)
    assert receipt["actual_data_verified"] is False


def test_native_pair_coverage_failure_is_not_hidden_by_zero_metrics(
    tmp_path, capsys
):
    path = make_request(tmp_path / "data")
    request = json.loads(path.read_text(encoding="utf-8"))
    development_path = (
        path.parent / request["refs"]["development_receipt"]["path"]
    )
    development = json.loads(development_path.read_text(encoding="utf-8"))
    ref = development["comparisons"][0]["native_evidence"]
    native_path = path.parent / ref["path"]
    native = json.loads(native_path.read_text(encoding="utf-8"))
    native.update(passed=False, reason="synthetic missing full480s coverage")
    native_path.write_text(json.dumps(native), encoding="utf-8")
    ref["sha256"] = pilot.sha(native_path)
    development_path.write_text(json.dumps(development), encoding="utf-8")
    request["refs"]["development_receipt"]["sha256"] = pilot.sha(
        development_path
    )
    path.write_text(json.dumps(request), encoding="utf-8")
    code, receipt = result(path, capsys)
    assert code == 2 and "coverage gates" in receipt["reason"]


@pytest.mark.parametrize(
    "damage", [None, "missing_actual", "short", "late", "policy_hash"]
)
def test_native_interval_preserves_actual_landing_tail_without_request_fallback(
    tmp_path, damage
):
    path = make_request(tmp_path / "data")
    request = json.loads(path.read_text(encoding="utf-8"))
    policy_path = path.parent / request["refs"]["paired_policy"]["path"]
    native_policy_path = path.parent / request["refs"]["native_policy"]["path"]
    native_policy_path.write_text(
        json.dumps(
            dict(
                schema="drone.joint-phase-policy/1",
                revision="joint-native-consumption-c2-v1",
                duration_us=480000000,
                minimum_start_us=60000000,
                maximum_landing_activation_delay_us=100000,
            )
        ),
        encoding="utf-8",
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy.update(
        alignment_mode="versioned_joint_phase",
        mission_source="evaluation/unit-native-receipt.json",
        native_policy_sha256=pilot.sha(native_policy_path),
    )
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    native_path = path.parent / policy["mission_source"]
    native_path.parent.mkdir()
    origin = json.loads(
        (path.parent / request["case_origin"]["origin"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    origin.update(
        actual_mission_start_us=60000000, actual_landing_start_us=540065000
    )
    if damage == "missing_actual":
        origin["actual_landing_start_us"] = None
        origin["requested_landing_us"] = 540000000
    elif damage == "short":
        origin["actual_landing_start_us"] = 539999999
    elif damage == "late":
        origin["actual_landing_start_us"] = 540100001
    elif damage == "policy_hash":
        native_policy_path.write_text(
            native_policy_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
    native_path.write_text(json.dumps(origin), encoding="utf-8")
    arguments = dict(
        policy_path=policy_path,
        native_policy_path=native_policy_path,
        run_path=path.parent / request["case_origin"]["run"]["path"],
    )
    if damage:
        with pytest.raises((ValueError, pilot.Pending)):
            native_mission_interval(native_path, **arguments)
    else:
        interval = native_mission_interval(native_path, **arguments)
        assert interval["start_us"] == 60000000
        assert (
            interval["end_us_exclusive"] == 540065000
            and interval["landing_tail_us"] == 65000
        )
        assert interval["source_sha256"] == pilot.sha(native_path)


def test_pilot_cannot_be_relabelled_for_fit_or_final():
    assert partition("normal_test", PILOT) == "evaluation:" + PILOT
    with pytest.raises(ValueError):
        partition("normal_train", PILOT)
    registry = dict(
        schema="drone-ai.global-protection/1",
        entries=[
            dict(
                identities=["run:pilot"],
                forbidden_uses=["fit", "calibration", "final_test"],
            )
        ],
    )
    with pytest.raises(ValueError):
        check_protection(
            [dict(run_id="pilot", lineage=[], role="normal_train")],
            registry,
            usage="fit",
        )
    with pytest.raises(ValueError):
        check_protection(
            [dict(run_id="renamed", lineage=[], cohort=PILOT)],
            registry,
            usage="final_test",
        )
