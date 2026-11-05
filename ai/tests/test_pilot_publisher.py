"""Synthetic native layouts only. Never real6, READY, flight, fit or forward."""

import json

import pytest

from learning.history19 import joint_native as native
from learning.history19.pilot import ROOT, TASK, json_file, sha
from learning.history19.pilot_connector import native_mission_interval, project_pair_receipt
from learning.history19.pilot_publish import publish


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def fixture(root):
    root.mkdir()
    policy = json_file(ROOT / "simulation/artifacts/paired-origin-20261107/recovery-01/policy.json")
    policy_path = write(root / "native-policy.json", policy)
    source_path = write(
        root / "source.json",
        dict(
            files={
                k: "a" * 64
                for k in (
                    "sim/src/capstone_sim/joint_phase.py",
                    "sim/src/capstone_sim/cli.py",
                    "profiles/collection/joint-phase-policy.json",
                )
            }
        ),
    )
    runtime_path = write(root / "runtime.json", dict(source_manifest_sha256=sha(source_path)))
    producer = dict(
        battery_policy="cmu-vah01-rpt16-only/1",
        battery_revision="cmu-vah01-baseline/1",
        catalog_sha256="a" * 64,
        propulsion_revision="x500-propulsion-binary-v1",
        propulsion_catalog_sha256="b" * 64,
        source_manifest_sha256=sha(source_path),
        runtime_sha256=sha(runtime_path),
    )
    producer_path = write(root / "producer.json", producer)
    reports, ids = [], []
    for i in range(3):
        paths, checks, starts = [], {}, {}
        for j, role in enumerate(("normal", "anomaly")):
            rid = f"synthetic-{i}-{role}"
            ids.append(rid)
            raw = root / rid
            paths.append(raw)
            first = 60000000 + j * 180000
            last = first + 480024000
            start = dict(
                schema="drone.joint-phase-native-receipt/1",
                revision=native.ADAPTER,
                kind="actual_reference_consumption_start",
                policy_sha256=sha(policy_path),
                requested_start_us=60000000,
                actual_start_us=first,
                actual_time_us=first,
                clock="px4-boot",
                consumer="PositionControl::setInputSetpoint",
            )
            landing = dict(
                start, kind="actual_auto_land_reference_consumption", actual_time_us=last
            )
            checked = dict(
                status="passed",
                actual_start_us=first,
                actual_landing_us=last,
                actual_duration_s=(last - first) / 1e6,
                terminal_hold_tail_us=24000,
                consumed_samples=120002,
                reference_rows_seen=4801,
                max_consumption_gap_us=4000,
                requested_and_actual_distinct=True,
            )
            scenario = dict(
                seed=2000 + i,
                paired_case_id=f"synthetic-pair-{i}",
                operation_variation=dict(path=dict(kind="random")),
                wind_profile=dict(environment="strong"),
                producer_lock_sha256=sha(runtime_path),
                health_eligibility="confirmed_healthy" if role == "normal" else "abnormal",
                joint_phase_policy=dict(
                    revision=native.ADAPTER,
                    policy_sha256=sha(policy_path),
                    reference_rows=4801,
                    reference_period_us=100000,
                    actual_start=start,
                    actual_landing=landing,
                ),
            )
            files = {
                "run.json": dict(
                    run_id=rid,
                    origin="synthetic_unit_fixture",
                    status="completed",
                    clock=dict(id="px4-boot", unit="us"),
                ),
                "evaluation/scenario.json": scenario,
                native.ORIGIN: start,
                native.LANDING: landing,
                native.PHYSICS: dict(
                    schema="drone.joint-physics-origin/1",
                    origin_us=first,
                    notice_us=first + 2000,
                    fault_onset_us=-1,
                ),
                native.CHECK: checked,
                "evaluation/execution.json": [
                    dict(state="mission", sim_time=first / 1e6),
                    dict(state="landing", sim_time=last / 1e6),
                ],
            }
            for name, value in files.items():
                write(raw / name, value)
            write(
                raw / "integrity.json",
                {
                    name: dict(sha256=sha(raw / name), size=(raw / name).stat().st_size)
                    for name in files
                },
            )
            checks[role], starts[role] = checked, first
        report = dict(
            schema="drone.joint-phase-pair-check/1",
            status="passed",
            errors=[],
            policy=dict(revision=native.ADAPTER, policy_sha256=sha(policy_path)),
            source_sha256=sha(source_path),
            runtime_sha256=sha(runtime_path),
            normal_run=str(paths[0].absolute()),
            anomaly_run=str(paths[1].absolute()),
            original_interval_us=[0, 480000000],
            interpolation_used=False,
            native_consumption=checks,
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
            origins=starts,
            wind_function_checks=[
                dict(status="passed", mission_samples=480001, max_abs_component_error_mps=0.0)
            ]
            * 2,
        )
        reports.append(write(root / f"pair-{i}.json", report))
    complete = write(
        root / "completion.json",
        dict(
            schema="unit.native.completion/1",
            task_id=TASK,
            passed=True,
            allow_pilot45=True,
            actual_runs=6,
            passed_runs=6,
            passed_pairs=3,
            actual_run_ids=ids,
            source_sha256=sha(source_path),
            runtime_sha256=sha(runtime_path),
            policy_sha256=sha(policy_path),
        ),
    )
    return dict(
        native_completion=complete,
        native_policy=policy_path,
        pair_checks=reports,
        producer_binding=producer_path,
        source_manifest=source_path,
        runtime_manifest=runtime_path,
        output_dir=root / "published",
        handoff=root / "handoff.json",
        fixture_only=True,
    )


def test_publisher_uses_actual_layout_and_keeps_fixture_out_of_ready(tmp_path):
    arguments = fixture(tmp_path / "unit")
    code, result = publish(**arguments)
    assert code == 0 and result["status"] == "SYNTHETIC_READY"
    assert result["native_development_verified"] is False
    assert result["model_forward_steps"] == result["optimizer_steps"] == result["GPU_jobs"] == 0
    policy_path = arguments["output_dir"] / "paired-policy.json"
    policy = json_file(policy_path)
    raw = tmp_path / "unit/synthetic-0-normal"
    interval = native_mission_interval(
        raw / native.ORIGIN,
        policy_path=policy_path,
        native_policy_path=arguments["native_policy"],
        run_path=raw / "run.json",
    )
    assert interval["start_us"] == 60000000 and interval["end_us_exclusive"] == 540024000
    assert (
        interval["landing_tail_us"] == 24000 and native.LANDING in interval["native_sources_sha256"]
    )
    projected = project_pair_receipt(arguments["pair_checks"][0], policy_path=policy_path)
    assert projected["origin_delta_us"] == 0
    assert sha(arguments["output_dir"] / "native-policy.json") == sha(arguments["native_policy"])
    assert (
        json_file(arguments["pair_checks"][0])["origins"]["normal"]
        != json_file(arguments["pair_checks"][0])["origins"]["anomaly"]
    )
    with pytest.raises(ValueError, match="fixture"):
        native.origin(raw / "run.json", policy, fixture=False)


@pytest.mark.parametrize(
    "damage",
    [
        "false_completion",
        "stopped_completion",
        "missing_report",
        "reuse_pair",
        "source",
        "policy",
        "coverage",
        "failed_pair",
        "integrity",
        "null_landing",
    ],
)
def test_publisher_fails_closed_and_never_repairs_or_filters_failed_evidence(tmp_path, damage):
    arguments = fixture(tmp_path / "unit")
    before = json_file(arguments["native_completion"])
    if damage == "false_completion":
        write(arguments["native_completion"], dict(before, passed=False, allow_pilot45=False))
    elif damage == "stopped_completion":
        write(
            arguments["native_completion"],
            dict(before, passed=False, allow_pilot45=False, status="FAILED_NATIVE_COVERAGE"),
        )
    elif damage == "missing_report":
        arguments["pair_checks"] = None
    elif damage == "reuse_pair":
        arguments["pair_checks"][2] = arguments["pair_checks"][0]
    elif damage == "source":
        value = json_file(arguments["runtime_manifest"])
        write(arguments["runtime_manifest"], dict(value, changed=True))
    elif damage == "policy":
        value = json_file(arguments["native_policy"])
        value["maximum_landing_activation_delay_us"] = 200000
        write(arguments["native_policy"], value)
    elif damage in {"coverage", "failed_pair"}:
        path = arguments["pair_checks"][0]
        value = json_file(path)
        if damage == "coverage":
            value["wind"]["missing_samples"] = 1
        else:
            value["status"] = "failed"
        write(path, value)
    elif damage == "integrity":
        path = tmp_path / "unit/synthetic-0-normal" / native.PHYSICS
        write(path, dict(json_file(path), notice_us=70000000))
    elif damage == "null_landing":
        raw = tmp_path / "unit/synthetic-0-normal"
        path = raw / native.LANDING
        write(path, dict(json_file(path), actual_time_us=None))
        integrity = json_file(raw / "integrity.json")
        integrity[native.LANDING] = dict(sha256=sha(path), size=path.stat().st_size)
        write(raw / "integrity.json", integrity)
    write(arguments["handoff"], dict(status="READY", old_scope="must_be_archived"))
    code, result = publish(**arguments)
    assert code in (1, 2) and result["status"] in {"WAITING", "FAILED"}
    assert json_file(arguments["handoff"])["status"] != "READY"
    assert list(arguments["handoff"].parent.glob("handoff.history-*.json"))
    if damage == "false_completion":
        assert json_file(arguments["native_completion"])["passed"] is False


def test_publication_never_overwrites_prior_snapshot(tmp_path):
    arguments = fixture(tmp_path / "unit")
    code, first = publish(**arguments)
    frozen = sha(arguments["output_dir"] / "publication-receipt.json")
    code, second = publish(**arguments)
    assert code == 1 and second["status"] == "FAILED"
    assert sha(arguments["output_dir"] / "publication-receipt.json") == frozen


def test_pair_projection_rejects_wrong_caller_pair(tmp_path):
    arguments = fixture(tmp_path / "unit")
    code, result = publish(**arguments)
    policy = json_file(arguments["output_dir"] / "paired-policy.json")
    with pytest.raises(ValueError, match="another physical pair"):
        native.pair(
            arguments["pair_checks"][0], policy, fixture=True, normal_run=tmp_path / "other"
        )


@pytest.mark.parametrize("collision", ["input", "snapshot"])
def test_handoff_cannot_overwrite_inputs_or_immutable_snapshots(tmp_path, collision):
    arguments = fixture(tmp_path / "unit")
    original = sha(arguments["producer_binding"])
    arguments["handoff"] = (
        arguments["producer_binding"]
        if collision == "input"
        else arguments["output_dir"] / "publication-receipt.json"
    )
    with pytest.raises(ValueError, match="overwrite|immutable"):
        publish(**arguments)
    assert sha(arguments["producer_binding"]) == original and not arguments["output_dir"].exists()
