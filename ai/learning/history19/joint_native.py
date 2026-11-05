"""Read the reviewed joint-native-consumption-c2-v1 layout without changing raw files.

Layout reviewed from Sim's recovery-01 offline-source candidate. Publication still
requires actual completed6/pair checks and the current stable source/runtime.
"""

import math
from pathlib import Path

from learning.admission19 import check_producer

from .duration_profiles import NATIVE_ADAPTERS, duration_us
from .pilot import Pending, json_file, regular, required, sha

ADAPTER = "joint-native-consumption-c2-v1"
ORIGIN = "evaluation/joint-origin.json"
LANDING = "evaluation/joint-landing.json"
PHYSICS = "evaluation/joint-physics-origin.json"
CHECK = "evaluation/joint-check.json"
WIND_CLOCK = "native_controller_consumption_relative_seconds_with_C2_initial_plateau"


def check_policy(native, *, duration_profile="standard480"):
    duration = duration_us(dict(duration_profile=duration_profile))
    expected = dict(
        minimum_start_us=60000000,
        duration_us=480000000,
        reference_period_us=100000,
        weather_initial_plateau_us=1000000,
        maximum_origin_notice_delay_us=500000,
        maximum_landing_activation_delay_us=100000,
    )
    gap_limit = native.get("acceptance", {}).get("native_consumption_max_gap_us")
    if type(gap_limit) is not int or gap_limit not in (40_000, 3_000_000):
        raise ValueError("Unapproved native consumption gap criterion")
    criteria = dict(
        wind_max_mps=2e-9,
        wind_rms_mps=2e-9,
        position_m=1e-8,
        velocity_mps=1e-8,
        circular_yaw_rad=1e-8,
        expected_reference_rows=4801,
        missing_reference_rows=0,
        expected_weather_ticks=480001,
        missing_weather_ticks=0,
        native_consumption_max_gap_us=gap_limit,
    )
    if (
        native.get("schema") != "drone.joint-phase-policy/1"
        or native.get("revision") not in NATIVE_ADAPTERS
        or any(native.get(k) != v for k, v in expected.items())
        or any(native.get("acceptance", {}).get(k) != v for k, v in criteria.items())
    ):
        raise ValueError("Joint native policy differs from reviewed revision/criteria")
    if native["revision"] == NATIVE_ADAPTERS[0] and duration != 480000000:
        raise ValueError("Native v1 is historical480 only")
    if native["revision"] == NATIVE_ADAPTERS[1] and (
        native.get("default_duration_us") != 480000000
        or native.get("supported_duration_us") != [180000000, 480000000]
    ):
        raise ValueError("Native v2 must explicitly preserve default480 and support180/480")


def origin(run_path, policy, *, fixture=False, external_audit=None):
    """Project actual native start, AUTO_LAND and physics notice, never request/ACK."""
    run_path = regular(run_path)
    folder, run = run_path.parent, json_file(run_path)
    if (
        run.get("origin") != ("synthetic_unit_fixture" if fixture else "px4_sitl")
        or run.get("status") != "completed"
    ):
        raise ValueError("Completed actual native raw required; fixture remains separate")
    if run["clock"].get("id") != "px4-boot" or run["clock"].get("unit") != "us":
        raise ValueError("Reviewed native PX4 boot clock required")
    integrity = json_file(folder / "integrity.json")
    duration = duration_us(policy)
    adapter = policy.get("native_adapter", ADAPTER)
    if adapter not in NATIVE_ADAPTERS:
        raise ValueError("Reviewed native adapter required")
    names = ("run.json", "evaluation/scenario.json", ORIGIN, LANDING, PHYSICS)
    if external_audit is None:
        names += (CHECK,)
    documents, hashes = {}, {}
    for name in names:
        path = regular(folder / name)
        hashes[name] = sha(path)
        if (
            integrity[name]["sha256"] != hashes[name]
            or integrity[name]["size"] != path.stat().st_size
        ):
            raise ValueError("Native raw evidence differs from closing integrity: " + name)
        documents[name] = json_file(path)
    scenario = documents["evaluation/scenario.json"]
    selected = required(scenario, "joint_phase_policy")
    if (
        scenario.get("producer_lock_sha256") != policy["producer"]["runtime_sha256"]
        or selected.get("revision") != adapter
        or selected.get("policy_sha256") != policy["native_policy_sha256"]
        or selected.get("reference_rows") != duration // 100000 + 1
        or selected.get("reference_period_us") != 100000
        or selected.get("duration_us", 480000000) != duration
    ):
        raise ValueError("Actual native scenario policy/source/reference binding differs")
    if adapter == NATIVE_ADAPTERS[1] and (
        scenario.get("requested_duration_s") != duration / 1000000
        or selected.get("expected_weather_ticks") != duration // 1000 + 1
    ):
        raise ValueError("Explicit per-run native v2 duration and weather count required")
    start, end = documents[ORIGIN], documents[LANDING]
    for receipt, kind in (
        (start, "actual_reference_consumption_start"),
        (end, "actual_auto_land_reference_consumption"),
    ):
        if (
            receipt.get("schema") != "drone.joint-phase-native-receipt/1"
            or receipt.get("revision") != adapter
            or receipt.get("kind") != kind
            or receipt.get("consumer") != "PositionControl::setInputSetpoint"
            or receipt.get("clock") != "px4-boot"
            or receipt.get("policy_sha256") != policy["native_policy_sha256"]
        ):
            raise ValueError("Actual controller receipt kind/clock/policy required")
        if adapter == NATIVE_ADAPTERS[1] and receipt.get("duration_us") != duration:
            raise ValueError("Native boundary receipt duration differs from selected profile")
        for key in ("actual_start_us", "actual_time_us"):
            if receipt.get(key) is None:
                raise Pending("Actual native receipt is unconfirmed: " + key)
            if type(receipt[key]) is not int:
                raise ValueError("Actual integer native timestamps required")
    first, last = start["actual_time_us"], end["actual_time_us"]
    if (
        start["actual_start_us"] != first
        or end["actual_start_us"] != first
        or first < 60000000
        or not first + duration <= last <= first + duration + 100000
        or selected.get("actual_start") != start
        or selected.get("actual_landing") != end
    ):
        raise ValueError("Native consumed interval/scenario receipt mismatch; no LAND tail crop")
    physics = documents[PHYSICS]
    external_provenance = None
    if external_audit is None:
        checked = documents[CHECK]
    else:
        from .native_audit import verified_check

        checked, external_provenance = verified_check(
            run_path,
            external_audit,
            generation=policy["producer"],
            policy_sha256=policy["native_policy_sha256"],
            duration_us=duration,
            fixture_only=fixture,
        )
    notice = required(physics, "notice_us")
    if (
        physics.get("schema") != "drone.joint-physics-origin/1"
        or physics.get("origin_us") != first
        or type(notice) is not int
        or not first <= notice <= first + 500000
        or checked.get("status") != "passed"
        or checked.get("actual_start_us") != first
        or checked.get("actual_landing_us") != last
        or checked.get("terminal_hold_tail_us") != last - first - duration
        or checked.get("reference_rows_seen") != duration // 100000 + 1
        or checked.get("native_consumption_max_gap_us", 40000) != policy.get("native_consumption_max_gap_us", 40000)
        or checked.get("consumed_samples", 0) <= 0
        or not 0 <= checked.get("max_consumption_gap_us", -1) <= policy.get("native_consumption_max_gap_us", 40000)
    ):
        raise ValueError("Native physics notice or full consumed reference check failed")
    if adapter == NATIVE_ADAPTERS[1] and (
        checked.get("duration_us") != duration
        or checked.get("backward_source_time_count") != 0
        or checked.get("conflicting_same_source_time_count") != 0
        or checked.get("consumed_samples")
        != checked.get("same_source_time_consumptions", -1)
        + checked.get("unique_source_time_samples", -1)
    ):
        raise ValueError(
            "Native v2 explicit duration and identical-time consumption counters required"
        )
    result = dict(
        run_id=run["run_id"],
        clock_id=run["clock"]["id"],
        policy_id=adapter,
        source_manifest_sha256=policy["producer"]["source_manifest_sha256"],
        runtime_sha256=policy["producer"]["runtime_sha256"],
        actual_mission_start_us=first,
        actual_landing_start_us=last,
        weather_origin_us=physics["origin_us"],
        path_origin_us=start["actual_start_us"],
        actual_source_ack_us=notice,
        source_acknowledged=True,
        native_sources_sha256=hashes,
        native_consumption_check=checked,
    )
    if external_provenance is not None:
        result["external_audit_evidence"] = external_provenance
    return result


def pair(
    report_path, policy, *, fixture=False, normal_run=None, anomaly_run=None, external_audits=None
):
    """Map the actual Sim pair report, including its failed coverage gates."""
    report_path = regular(report_path)
    report = json_file(report_path)
    duration = duration_us(policy)
    if report.get("schema") != "drone.joint-phase-pair-check/1":
        raise ValueError("Reviewed native pair report schema required")
    if report.get("status") != "passed" or report.get("errors") or report.get("passed") is False:
        raise Pending("Actual native pair check did not pass")
    check_producer(policy["producer"], policy["producer"])
    if (
        report.get("source_sha256") != policy["producer"]["source_manifest_sha256"]
        or report.get("runtime_sha256") != policy["producer"]["runtime_sha256"]
        or report.get("policy", {}).get("policy_sha256") != policy["native_policy_sha256"]
        or report.get("policy", {}).get("revision") != policy.get("native_adapter", ADAPTER)
        or report.get("original_interval_us") != [0, duration]
        or report.get("interpolation_used") is not False
    ):
        raise ValueError("Native pair source/policy/full interval differs")
    members = []
    for role, expected_path in (("normal", normal_run), ("anomaly", anomaly_run)):
        path = regular(Path(required(report, role + "_run")) / "run.json").parent
        if expected_path is not None and path != Path(expected_path).absolute():
            raise ValueError("Native verifier returned another physical pair")
        state = json_file(path / "run.json")
        projected = origin(
            path / "run.json",
            policy,
            fixture=fixture,
            external_audit=(external_audits or {}).get(state["run_id"]),
        )
        scenario = json_file(path / "evaluation/scenario.json")
        if scenario.get("health_eligibility") != (
            "confirmed_healthy" if role == "normal" else "abnormal"
        ):
            raise ValueError("Native pair health role mismatch")
        if report.get("native_consumption", {}).get(role) != projected["native_consumption_check"]:
            raise ValueError("Pair consumption summary differs from actual checked raw")
        if report.get("origins", {}).get(role) != projected["actual_mission_start_us"]:
            raise ValueError("Pair origin differs from actual native raw")
        members.append((path, projected, scenario))
    if members[0][1]["run_id"] == members[1][1]["run_id"]:
        raise ValueError("Independent normal and anomaly raw required")
    for key in ("seed", "operation_variation", "wind_profile", "paired_case_id"):
        if members[0][2].get(key) is None or members[0][2][key] != members[1][2].get(key):
            raise ValueError("Native pair forcing/identity differs: " + key)
    wind, reference = report.get("wind", {}), report.get("reference", {})
    if (
        wind.get("samples") != duration // 1000 + 1
        or wind.get("expected_samples") != duration // 1000 + 1
        or wind.get("missing_samples") != 0
        or reference.get("rows") != duration // 100000 + 1
        or reference.get("missing_rows") != 0
    ):
        raise ValueError("Full native wind/reference coverage required; no intersection-only pass")
    for container, key, limit in (
        (wind, "max_vector_difference_mps", 2e-9),
        (wind, "rms_vector_difference_mps", 2e-9),
        (reference, "max_position_difference_m", 1e-8),
        (reference, "max_velocity_difference_mps", 1e-8),
        (reference, "max_circular_yaw_difference_rad", 1e-8),
    ):
        value = required(container, key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError("Native pair exceeds reviewed criterion: " + key)
    oracle = required(report, "wind_function_checks")
    if len(oracle) != 2 or any(
        x.get("status") != "passed"
        or x.get("mission_samples") != duration // 1000 + 1
        or not 0 <= x.get("max_abs_component_error_mps", -1) <= 2e-9
        for x in oracle
    ):
        raise ValueError("Native full-run applied-wind oracle did not pass")
    return dict(
        passed=True,
        normal_run_id=members[0][1]["run_id"],
        anomaly_run_id=members[1][1]["run_id"],
        policy_id=policy.get("native_adapter", ADAPTER),
        source_manifest_sha256=policy["producer"]["source_manifest_sha256"],
        runtime_sha256=policy["producer"]["runtime_sha256"],
        origin_delta_us=max(
            abs(x[1]["weather_origin_us"] - x[1]["path_origin_us"]) for x in members
        ),
        wind_rms_mps=wind["rms_vector_difference_mps"],
        wind_peak_mps=wind["max_vector_difference_mps"],
        normal_origin_receipt_sha256=members[0][1]["native_sources_sha256"][ORIGIN],
        anomaly_origin_receipt_sha256=members[1][1]["native_sources_sha256"][ORIGIN],
    )
