"""AI19 intake metadata checks, without model imports or production execution.

This is an AI dataset gate, not a change to the producer's raw/C9 contract.
Data must independently bind these summaries to its audited raw evidence.
"""

import math
import re
from pathlib import PurePosixPath

BATTERY_POLICY = "cmu-vah01-rpt16-only/1"
PROPULSION_REVISION = "x500-propulsion-binary-v1"
MISSION_POLICY = "mission_only_actual_execution/1"
ONLINE_UNITS = "nameplate_transformed_common19_cellV_Crate"
RUN_UNITS = "physical_common19_pack_V_A"
PILOT = "pilot_development_reference"
SHORT = "short_cycle_development_reference"
FINAL16 = "final16_development_reference"
EVALUATION_COHORTS = {"preliminary", "development", "final", PILOT, SHORT, FINAL16}
FIT_ROLES = {"normal_train", "normal_validation", "normal_calibration"}
TEST_ROLES = {"normal_test", "anomaly_test"}
NORMAL_SCENARIOS = {
    "hover_normal",
    "altitude_normal",
    "horizontal_normal",
    "yaw_normal",
    "waypoint_normal",
    "low_soc_normal",
    "wind_normal",
    "cold_normal",
    "payload_normal",
}
PRODUCER_FIELDS = (
    "battery_policy",
    "battery_revision",
    "catalog_sha256",
    "propulsion_revision",
    "propulsion_catalog_sha256",
    "runtime_sha256",
    "source_manifest_sha256",
)


def partition(role, cohort=None):
    if role in FIT_ROLES:
        if cohort is not None and cohort != role:
            raise ValueError("Fit/calibration role cannot be renamed into an evaluation cohort")
        return role
    if role not in TEST_ROLES or cohort not in EVALUATION_COHORTS:
        raise ValueError("Evaluation role requires explicit preliminary/development/final cohort")
    return "evaluation:" + cohort


def validate_lineages(rows):
    owners, ids = {}, set()
    for row in rows:
        if row.get("purpose") == PILOT and row.get("cohort") != PILOT:
            raise ValueError("Pilot purpose cannot be relabelled into a fit or final cohort")
        if row.get("purpose") == SHORT and row.get("cohort") != SHORT:
            raise ValueError("Short-cycle purpose cannot be relabelled into fit/final")
        if row.get("purpose") == FINAL16 and row.get("cohort") != FINAL16:
            raise ValueError("FINAL16 purpose cannot be relabelled into fit/final")
        rid = row["run_id"]
        if not rid or rid in ids or not row["lineage"]:
            raise ValueError("Unique run IDs and explicit lineage required")
        ids.add(rid)
        scope = partition(row["role"], row.get("cohort"))
        for key in ("run:" + rid, *row["lineage"]):
            if not isinstance(key, str) or not key:
                raise ValueError("Nonempty lineage identifiers required")
            previous = owners.get(key)
            if previous is not None:
                old_scope, old_role = previous
                if old_scope != scope:
                    raise ValueError(
                        "Raw flight/retry/paired/group leakage across splits or cohorts"
                    )
                if old_role != row["role"] and not key.startswith(("pair:", "group:")):
                    raise ValueError("Only pair/group IDs may be shared by evaluation roles")
            owners[key] = scope, row["role"]
    return rows


def check_protection(rows, registry, *, usage):
    """Apply Data's frozen global identity exclusions, including unseen past cohorts."""
    if registry.get("schema") != "drone-ai.global-protection/1":
        raise ValueError("Explicit frozen global protection registry required")
    forbidden = {}
    for entry in registry["entries"]:
        if not entry.get("identities") or not isinstance(entry.get("forbidden_uses"), list):
            raise ValueError("Protection entries need identities and forbidden uses")
        if any(not isinstance(k, str) or not k for k in entry["identities"]) or any(
            not isinstance(u, str) or not u for u in entry["forbidden_uses"]
        ):
            raise ValueError("Typed global protection identities/uses required")
        for key in entry["identities"]:
            forbidden.setdefault(key, set()).update(entry["forbidden_uses"])
    for row in rows:
        if row.get("cohort") == PILOT or row.get("purpose") == PILOT:
            if usage != "pilot_input_check":
                raise ValueError("pilot_development_reference is globally no-fit/no-final")
        if row.get("cohort") == SHORT or row.get("purpose") == SHORT:
            if usage != "short_cycle_input_check":
                raise ValueError("Short-cycle input is globally no-fit/no-final")
        if row.get("cohort") == FINAL16 or row.get("purpose") == FINAL16:
            if usage != "final16_input_check":
                raise ValueError("FINAL16 input is globally no-fit/no-final")
        keys = list(row["lineage"])
        if row.get("run_id"):
            keys.append("run:" + row["run_id"])
        if row.get("case_id"):
            keys.append("case:" + row["case_id"])
        for key in keys:
            if usage in forbidden.get(key, set()):
                raise ValueError("Globally excluded identity reused: " + key)


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def check_producer(producer, expected):
    if expected is None:
        raise ValueError("An explicit frozen expected producer binding is required for actual data")
    if not isinstance(producer, dict) or not isinstance(expected, dict):
        raise ValueError("Explicit producer binding objects required")
    for key in PRODUCER_FIELDS:
        if key not in producer or producer[key] != expected.get(key):
            raise ValueError("Producer binding missing or changed: " + key)
        if key.endswith("sha256") and not digest(producer[key]):
            raise ValueError("Invalid producer hash: " + key)
    if (
        producer["battery_policy"] != BATTERY_POLICY
        or producer["battery_revision"] != "cmu-vah01-baseline/1"
        or producer["propulsion_revision"] != PROPULSION_REVISION
    ):
        raise ValueError("Current battery and joint binary propulsion policy required")


def source_evidence(row, producer):
    """Validate declared selection/transition summaries; never emit neural features."""
    partition(row["role"], row.get("cohort"))
    if row.get("origin") != "px4_sitl":
        raise ValueError("Actual intake requires px4_sitl origin")
    health = row.get("health_eligibility", row.get("health"))
    if health != ("abnormal" if row["role"] == "anomaly_test" else "confirmed_healthy"):
        raise ValueError("Explicit source health eligibility required for declared role")
    mission = row.get("mission_interval", {})
    start, end = mission.get("start_us"), mission.get("end_us_exclusive")
    member = mission.get("source", "")
    native_interval = (
        isinstance(member, str)
        and member.startswith("evaluation/")
        and member.endswith(".json")
        and ".." not in PurePosixPath(member).parts
        and mission.get("boundary_kind") == "actual_native_controller_consumption"
        and mission.get("native_policy_revision")
        in {"joint-native-consumption-c2-v1", "joint-native-consumption-c2-v2"}
    )
    if (
        mission.get("policy") != MISSION_POLICY
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end
        or not mission.get("clock_id")
        or (member != "evaluation/execution.json" and not native_interval)
        or not digest(mission.get("source_sha256"))
    ):
        raise ValueError("Actual mission/landing execution boundaries and source hash required")
    keys = row.get("lineage", [])
    if any(
        not any(k.startswith(prefix) for k in keys)
        for prefix in ("run:", "pair:", "group:", "retry:", "raw-sha256:")
    ):
        raise ValueError("Actual intake requires typed raw/pair/group/retry/hash lineage")
    if any(not digest(k.split(":", 1)[1]) for k in keys if k.startswith("raw-sha256:")):
        raise ValueError("Valid raw source hash lineage required")
    evidence = row.get("source_evidence", {})
    if evidence.get("producer_runtime_sha256") != producer["runtime_sha256"]:
        raise ValueError("Per-run producer runtime differs from frozen binding")
    scenario = evidence.get("scenario_name", "")
    battery, propulsion = (
        evidence.get(k, {}) for k in ("battery_selection", "propulsion_selection")
    )
    if (
        battery.get("catalog_revision") != producer["battery_revision"]
        or battery.get("catalog_sha256") != producer["catalog_sha256"]
        or battery.get("flight_policy", {}).get("revision") != BATTERY_POLICY
        or propulsion.get("catalog_revision") != PROPULSION_REVISION
        or propulsion.get("catalog_sha256") != producer["propulsion_catalog_sha256"]
    ):
        raise ValueError("Per-run battery/propulsion catalog binding mismatch")
    battery_scenario = scenario in {
        "battery_aging_existing",
        "battery_aging_step",
        "battery_aging_ramp",
    }
    propulsion_scenario = scenario in {
        "propulsion_loss_existing",
        "propulsion_loss_step",
        "propulsion_loss_ramp",
    }
    if (
        not scenario
        or not (battery_scenario or propulsion_scenario or scenario in NORMAL_SCENARIOS)
        or (not battery_scenario and not propulsion_scenario and row["role"] == "anomaly_test")
    ):
        raise ValueError(
            "Current joint-profile scenario required; old component faults unsupported"
        )
    initial_b, target_b = battery.get("initial_stage"), battery.get("target_stage")
    initial_p, target_p = propulsion.get("initial_state"), propulsion.get("target_state")
    if initial_b not in {"rpt_00", "rpt_16"} or target_b not in {"rpt_00", "rpt_16"}:
        raise ValueError("Only rpt_00/rpt_16 battery states allowed")
    if initial_p not in {"normal", "abnormal"} or target_p not in {"normal", "abnormal"}:
        raise ValueError("Only joint normal/abnormal propulsion states allowed")
    dynamic = scenario.endswith(("_step", "_ramp"))
    expected_p = (
        ("normal", "abnormal")
        if propulsion_scenario and dynamic
        else (("abnormal", "abnormal") if propulsion_scenario else ("normal", "normal"))
    )
    if (initial_p, target_p) != expected_p:
        raise ValueError("Propulsion initial/target states contradict scenario")
    if battery_scenario:
        expected_b = ("rpt_00", "rpt_16") if dynamic else ("rpt_16", "rpt_16")
        if (initial_b, target_b) != expected_b:
            raise ValueError("Battery initial/target states contradict scenario")
    elif initial_b != target_b:
        raise ValueError("Combined propulsion scenarios require a fixed battery stage")
    if row["role"] != "anomaly_test" and (
        initial_b != "rpt_00"
        or target_b != "rpt_00"
        or initial_p != "normal"
        or target_p != "normal"
    ):
        raise ValueError("Normal membership requires both systems healthy throughout")
    transitions = evidence.get("transition_evidence", {})
    for system, selected, active in (
        ("battery", battery, battery_scenario and dynamic),
        ("propulsion", propulsion, propulsion_scenario and dynamic),
    ):
        duration = selected.get("transition_s")
        if type(duration) not in (int, float) or not math.isfinite(duration):
            raise ValueError("Finite explicit transition duration required")
        if (
            duration < 0
            or selected.get("fixed_for_flight") is not (not active)
            or (duration > 0) != active
        ):
            raise ValueError("Fixed/dynamic transition selection mismatch")
        if active:
            record = transitions.get(system, {})
            onset, end = record.get("actual_onset_us"), record.get("actual_endpoint_us")
            request = record.get("requested_offset_s")
            if (
                type(onset) is not int
                or type(end) is not int
                or onset < 0
                or end < onset
                or type(request) not in (int, float)
                or not math.isfinite(request)
                or request < 0
                or record.get("clock_id") != mission["clock_id"]
            ):
                raise ValueError("Actual onset/endpoint and requested offset/clock required")
    # This derived metadata is for evaluator denominators, never model.forward.
    initial_anomaly = [
        s
        for s, abnormal in (
            ("battery", initial_b == "rpt_16"),
            ("propulsion", initial_p == "abnormal"),
        )
        if abnormal
    ]
    return dict(
        initial_anomaly_systems=initial_anomaly,
        pre_injection_is_normal=not initial_anomaly,
        metadata_only=True,
        raw_physical_audit_verified=False,
    )
