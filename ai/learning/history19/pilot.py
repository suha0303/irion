"""One-shot P27 admission. No model imports, fit, subprocess, waits or retries.

The policy/receipt schemas here are AI consumer projections, not a replacement
for the Sim raw contract. Native field mappings must be frozen by the owners.
"""

import hashlib
import json
import math
from pathlib import Path

from learning.admission19 import (
    MISSION_POLICY,
    PILOT,
    check_producer,
    check_protection,
    digest,
)
from learning.wind_metadata import KINDS

from .duration_profiles import NATIVE_ADAPTERS

ROOT = Path(__file__).absolute().parents[3]
TASK = "P27_PILOT45_20261107"
AUTH = ROOT / "ai/reports/p27-pilot45-20261107/attempt-01/authorization.json"
RECOVERY_AUTH = (
    ROOT / "ai/reports/p27-pilot45-20261107/recovery-01/authorization.json"
)
HISTORICAL_EXCLUSIONS = ROOT / (
    "ai/reports/p27-precollection-20261107/seal-review-05-"
    "bounded/development-reference-exclusions.json"
)
CRITICAL = (
    "ai/learning/admission19.py",
    "ai/learning/wind_metadata.py",
    "ai/learning/history19/pilot.py",
    "ai/learning/history19/pilot_views.py",
    "ai/learning/history19/pilot_connector.py",
    "ai/learning/history19/joint_native.py",
    "ai/learning/history19/native_audit.py",
    "ai/learning/history19/duration_profiles.py",
    "ai/learning/history19/short_cycle.py",
    "ai/learning/history19/quality_rejection.py",
    "ai/learning/history19/final16.py",
    "ai/learning/history19/pilot_publish.py",
    "ai/learning/history19/data.py",
    "ai/learning/history19/preprocessing.py",
    "ai/learning/history19/preprocessing.json",
    "ai/learning/history19/cli.py",
    "ai/learning/run_health19/data.py",
    "ai/learning/run_health19/cli.py",
    "ai/src/capstone_ai/mapping.py",
    "ai/src/capstone_ai/source.py",
)
ORIGIN_FIELDS = (
    "run_id",
    "clock_id",
    "source_manifest_sha256",
    "runtime_sha256",
    "policy_id",
    "actual_mission_start_us",
    "actual_landing_start_us",
    "weather_origin_us",
    "path_origin_us",
    "actual_source_ack_us",
    "source_acknowledged",
)
LIMITS = (
    "origin_delta_us",
    "wind_rms_mps",
    "wind_peak_mps",
    "receipt_min_offset_us",
    "receipt_max_offset_us",
)
COMPARISON_FIELDS = LIMITS[:3] + (
    "normal_run_id",
    "anomaly_run_id",
    "policy_id",
    "source_manifest_sha256",
    "runtime_sha256",
)


class Pending(ValueError):
    pass


class Unavailable(ValueError):
    pass


def required(obj, key):
    if obj.get(key) is None:
        raise Pending("Missing or unconfirmed: " + key)
    return obj[key]


def regular(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink/traversal inputs and outputs forbidden")
    if not path.is_file():
        raise Pending("File not available: " + str(path))
    return path


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def json_file(path):
    path = regular(path)
    if path.stat().st_size > 16 * (1 << 20):
        raise ValueError(
            "Admission JSON exceeds16MiB; use bounded metadata, not raw events"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class Evidence:
    """Resolve pilot request references while recording verified paths and
    hashes.
    """

    def __init__(self, request_path):
        self.path = regular(request_path)
        self.request = json_file(self.path)
        self.hashes = {"request": sha(self.path)}
        self.paths = {}
        self.checked = []

    def ref(self, name, ref=None):
        ref = (
            required(self.request.get("refs", {}), name) if ref is None else ref
        )
        path = required(ref, "path")
        expected = required(ref, "sha256")
        if not digest(expected):
            raise ValueError("Invalid evidence hash: " + name)
        path = Path(path)
        path = regular(path if path.is_absolute() else self.path.parent / path)
        actual = sha(path)
        if actual != expected:
            raise ValueError("Evidence changed: " + name)
        self.hashes[name], self.paths[name] = actual, path
        return path

    def doc(self, name, ref=None):
        return json_file(self.ref(name, ref))


def bind(document, expected):
    check_producer(required(document, "producer"), expected)


def policy_check(ctx, expected, *, short_cycle=False):
    policy = ctx.doc("paired_policy")
    if policy.get("schema") != "drone-ai.paired-source-clock-policy/1":
        raise ValueError("AI paired policy projection schema required")
    authority = (
        "approved_for_short_cycle" if short_cycle else "approved_for_pilot"
    )
    if policy.get("status") != "FROZEN" or policy.get(authority) is not True:
        raise Pending(
            "Paired policy/criteria are not actually frozen and approved"
        )
    bind(policy, expected)
    if (
        policy.get("alignment_mode")
        not in {"producer_source_clock", "versioned_joint_phase"}
        or policy.get("mission_policy") != MISSION_POLICY
        or policy.get("posthoc_adjustment") is not False
    ):
        raise ValueError(
            (
                "Producer source-clock alignment and unchanged mission-only "
                "policy required"
            )
        )
    for key in ("policy_id", "wind_model_clock", "native_receipt_schema"):
        if not isinstance(required(policy, key), str) or not policy[key]:
            raise ValueError("Explicit native clock/policy identity required")
    native = ctx.ref("native_policy")
    if policy.get("native_policy_sha256") != sha(native):
        raise ValueError(
            "Policy projection is not bound to actual native policy"
        )
    if policy["alignment_mode"] == "versioned_joint_phase":
        document = json_file(native)
        if (
            document.get("schema") != "drone.joint-phase-policy/1"
            or document.get("revision") not in NATIVE_ADAPTERS
        ):
            raise ValueError(
                "Reviewed explicit joint-phase native revision required"
            )
        required(policy, "mission_source")
    limits = required(policy, "limits")
    for key in LIMITS:
        value = required(limits, key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Finite actual frozen criterion required: " + key)
    if (
        any(limits[k] < 0 for k in LIMITS[:3])
        or limits["receipt_min_offset_us"] > limits["receipt_max_offset_us"]
    ):
        raise ValueError("Invalid origin/forcing/receipt criteria")
    if policy.get("native_adapter") in NATIVE_ADAPTERS:
        from .joint_native import check_policy

        if policy["native_adapter"] != json_file(native)["revision"]:
            raise ValueError(
                "Native policy/adapter revision cannot be relabelled"
            )
        check_policy(
            json_file(native),
            duration_profile=policy.get("duration_profile", "standard480"),
        )
        if policy.get("fixture_only", False) != (
            ctx.request.get("fixture_only") is True
        ):
            raise ValueError(
                "Native adapter fixture status cannot be relabelled"
            )
        ctx.checked.append("reviewed_joint_native_layout")
        return policy
    for key in ORIGIN_FIELDS:
        parts = required(policy.get("native_receipt_field_paths", {}), key)
        if (
            not isinstance(parts, list)
            or not parts
            or len(parts) > 12
            or any(not isinstance(p, (str, int)) for p in parts)
        ):
            raise ValueError("Frozen native receipt field mapping required")
    for key in COMPARISON_FIELDS:
        parts = required(policy.get("native_comparison_field_paths", {}), key)
        if not isinstance(parts, list) or not parts or len(parts) > 12:
            raise ValueError("Frozen native comparison field mapping required")
    ctx.checked.append("frozen_policy_and_native_mapping")
    return policy


def project_native(native, policy):
    if native.get("schema") != policy["native_receipt_schema"]:
        raise ValueError(
            "Actual native receipt schema differs from frozen mapping"
        )
    projection = {}
    for key, parts in policy["native_receipt_field_paths"].items():
        value = native
        for part in parts:
            value = value[part]
        if value is None:
            raise Pending(
                "Native receipt has unconfirmed/null actual field: " + key
            )
        projection[key] = value
    return projection


def origin_member(ctx, member, policy, expected, label):
    run = ctx.doc(label + ":run", required(member, "run"))
    execution = ctx.doc(label + ":execution", required(member, "execution"))
    native = ctx.doc(label + ":origin", required(member, "origin"))
    if policy.get("native_adapter") in NATIVE_ADAPTERS:
        from .joint_native import origin

        projection = origin(
            ctx.paths[label + ":run"],
            policy,
            fixture=ctx.request.get("fixture_only") is True,
            external_audit=ctx.request.get("external_audit")
            if label == "case"
            else None,
        )
        if (
            ctx.paths[label + ":origin"]
            != ctx.paths[label + ":run"].parent / policy["mission_source"]
        ):
            raise ValueError(
                "Actual native receipt must be the frozen raw member"
            )
        for name, value in projection["native_sources_sha256"].items():
            ctx.hashes[label + ":native:" + name] = value
        for name, value in projection.get(
            "external_audit_evidence", {}
        ).items():
            if name.endswith("sha256"):
                ctx.hashes[label + ":external:" + name] = value
    else:
        projection = project_native(native, policy)
    if projection["source_acknowledged"] is not True:
        raise Pending("Actual source acknowledgement is not recorded")
    if (
        projection["run_id"] != run["run_id"]
        or projection["policy_id"] != policy["policy_id"]
    ):
        raise ValueError("Receipt run/policy identity mismatch")
    for key in ("source_manifest_sha256", "runtime_sha256"):
        if projection[key] != expected[key]:
            raise ValueError("Receipt is from a different actual producer")
    fixture = ctx.request.get("fixture_only") is True
    if (
        run.get("origin")
        != ("synthetic_unit_fixture" if fixture else "px4_sitl")
        or run.get("status") != "completed"
    ):
        raise ValueError(
            "Completed actual run required; fixture cannot pass as SITL"
        )
    times = {}
    if policy["alignment_mode"] == "versioned_joint_phase":
        from .pilot_connector import native_mission_interval

        interval = native_mission_interval(
            ctx.paths[label + ":origin"],
            policy_path=ctx.paths["paired_policy"],
            native_policy_path=ctx.paths["native_policy"],
            run_path=ctx.paths[label + ":run"],
            external_audit=ctx.request.get("external_audit")
            if label == "case"
            else None,
        )
        times = dict(
            mission=interval["start_us"], landing=interval["end_us_exclusive"]
        )
    else:
        for state in ("mission", "landing"):
            events = [r for r in execution if r.get("state") == state]
            if len(events) != 1 or not isinstance(
                events[0].get("sim_time"), (int, float)
            ):
                raise ValueError(
                    "Actual mission/landing execution events required"
                )
            times[state] = round(events[0]["sim_time"] * 1e6)
    if not 0 <= times["mission"] < times["landing"]:
        raise ValueError("Ordered actual execution interval required")
    if (
        projection["actual_mission_start_us"] != times["mission"]
        or projection["actual_landing_start_us"] != times["landing"]
        or projection["clock_id"] != run["clock"]["id"]
    ):
        raise ValueError("Receipt cannot replace/shift actual mission or clock")
    for key in ("weather_origin_us", "path_origin_us", "actual_source_ack_us"):
        if type(projection[key]) is not int or projection[key] < 0:
            raise ValueError(
                "Actual integer native origin/receipt time required"
            )
    limits = policy["limits"]
    if any(
        abs(projection[k] - times["mission"]) > limits["origin_delta_us"]
        for k in ("weather_origin_us", "path_origin_us")
    ):
        raise ValueError("Actual weather/path origin violates frozen policy")
    offset = projection["actual_source_ack_us"] - times["mission"]
    if (
        not limits["receipt_min_offset_us"]
        <= offset
        <= limits["receipt_max_offset_us"]
    ):
        raise ValueError("Actual source receipt timing violates frozen policy")
    return projection


def pair_metrics(document, limits):
    if document.get("passed") is not True:
        raise Pending("Actual pair comparison is not passed")
    for key in LIMITS[:3]:
        value = required(document, key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= limits[key]
        ):
            raise ValueError(
                "Actual paired difference exceeds frozen criterion: " + key
            )


def comparison_evidence(ctx, pair, policy, label):
    native = ctx.doc(label, required(pair, "native_evidence"))
    if policy.get("native_adapter") in NATIVE_ADAPTERS:
        from .joint_native import pair as project_pair

        native = project_pair(
            ctx.paths[label],
            policy,
            fixture=ctx.request.get("fixture_only") is True,
            external_audits=ctx.request.get("external_audits"),
        )
    if native.get("passed") is not True:
        raise Pending(
            (
                "Actual native pair verification is not passed, including "
                "coverage gates"
            )
        )
    for key in COMPARISON_FIELDS:
        if policy.get("native_adapter") in NATIVE_ADAPTERS:
            value = native[key]
        else:
            value = native
            for part in required(
                policy.get("native_comparison_field_paths", {}), key
            ):
                value = value[part]
        if value is None:
            raise Pending("Native comparison metric is not confirmed: " + key)
        expected = (
            pair[key]
            if key in (*LIMITS[:3], "normal_run_id", "anomaly_run_id")
            else policy["policy_id"]
            if key == "policy_id"
            else policy["producer"][key]
        )
        if value != expected:
            raise ValueError(
                "Pair metrics differ from mapped actual native evidence: " + key
            )


def development_check(ctx, expected, policy):
    document = ctx.doc("development_receipt")
    if (
        document.get("schema") != "drone-ai.paired-development-evidence/1"
        or document.get("status") != "PASSED"
    ):
        raise Pending(
            "Final actual development6 receipt is not available/passed"
        )
    bind(document, expected)
    if document.get("paired_policy_sha256") != ctx.hashes["paired_policy"]:
        raise ValueError("Development receipt uses a different policy")
    members = required(document, "members")
    if len(members) != 6:
        raise ValueError("Exactly six actual development runs required")
    origins = [
        origin_member(ctx, m, policy, expected, "development:" + str(i))
        for i, m in enumerate(members)
    ]
    ids = {r["run_id"] for r in origins}
    if len(ids) != 6 or len(document.get("comparisons", [])) != 3:
        raise ValueError("Three independent development pairs required")
    completion = ctx.doc("native_completion")
    check_native_completion(
        completion, fixture=ctx.request.get("fixture_only") is True
    )
    if (
        set(completion["actual_run_ids"]) != ids
        or completion["source_sha256"] != expected["source_manifest_sha256"]
        or completion["runtime_sha256"] != expected["runtime_sha256"]
        or completion["policy_sha256"] != ctx.hashes["native_policy"]
    ):
        raise ValueError(
            "Development projection differs from actual native completion/seals"
        )
    used = []
    for pair in document["comparisons"]:
        pair_metrics(pair, policy["limits"])
        used.extend((pair["normal_run_id"], pair["anomaly_run_id"]))
        comparison_evidence(
            ctx, pair, policy, "development_pair_evidence:" + str(len(used))
        )
    if len(used) != len(set(used)) or set(used) != ids:
        raise ValueError(
            "Development pair membership differs from actual six receipts"
        )
    ctx.checked.append("actual_development6_receipts_and_pair_criteria")
    return ids


def check_native_completion(document, *, fixture=False):
    schema = (
        "unit.native.completion/1"
        if fixture
        else "drone.paired-origin-development-completion/1"
    )
    if document.get("schema") != schema or document.get("task_id") != TASK:
        raise ValueError(
            (
                "Known actual native completion schema/task required; never "
                "infer mappings"
            )
        )
    if (
        document.get("passed") is not True
        or document.get("allow_pilot45") is not True
    ):
        raise Pending("Actual native completion does not authorize pilot45")
    if (
        document.get("actual_runs") != 6
        or document.get("passed_runs") != 6
        or document.get("passed_pairs") != 3
        or len(set(required(document, "actual_run_ids"))) != 6
        or len(document["actual_run_ids"]) != 6
    ):
        raise ValueError(
            "Actual independent six-run/three-pair completion required"
        )


def plan_and_ledger(
    ctx,
    registry,
    *,
    task_id=TASK,
    cohort=PILOT,
    usage="pilot_input_check",
    quality_rejection_check=None,
):
    if quality_rejection_check is not None:
        from learning.admission19 import SHORT

        if (
            task_id != "SHORT_CYCLE_20261107"
            or cohort != SHORT
            or usage != "short_cycle_input_check"
        ):
            raise ValueError(
                (
                    "Quality rejection retries are restricted to the explicit "
                    "short cycle"
                )
            )
    manifest = ctx.doc("pilot_manifest")
    if (
        manifest.get("schema") != "drone-ai.pilot45.manifest/1"
        or manifest.get("task_id") != task_id
        or manifest.get("cohort") != cohort
        or manifest.get("purpose") != cohort
        or manifest.get("no_fit") is not True
        or manifest.get("status") != "FROZEN"
    ):
        raise Pending("Frozen pilot45 no-fit manifest is not ready")
    cases = required(manifest, "cases")
    case_ids = {c["case_id"] for c in cases}
    if len(cases) != len(case_ids) or len(cases) != 45:
        raise ValueError("Exactly45 distinct physical planned cases required")
    base = [c for c in cases if c["case_kind"] == "normal_base"]
    controls = [c for c in cases if c["case_kind"] == "paired_normal"]
    anomalies = [c for c in cases if c["case_kind"] == "anomaly"]
    if (len(base), len(controls), len(anomalies)) != (27, 9, 9):
        raise ValueError(
            "Pilot requires27 normal base+9 independent controls+9 anomalies"
        )
    if {(c["motion"], c["environment"]) for c in base} != {
        (k, e) for k in KINDS for e in ("calm", "low", "strong")
    }:
        raise ValueError("Pilot base must cover all27 declared conditions")
    for case in cases:
        if (
            type(case.get("seed")) is not int
            or case["seed"] < 0
            or not digest(case.get("operation_sha256"))
            or not case.get("pair_id")
            or not isinstance(case.get("lineage"), list)
            or not case["lineage"]
        ):
            raise ValueError(
                (
                    "Explicit frozen seed/operation/pair/lineage required for "
                    "each case"
                )
            )
        if case["role"] != (
            "anomaly_test" if case["case_kind"] == "anomaly" else "normal_test"
        ):
            raise ValueError(
                "Pilot has only36normal_test/9anomaly_test; never fit roles"
            )
        check_protection(
            [{**case, "cohort": cohort, "purpose": cohort}],
            registry,
            usage=usage,
        )
        protected = set().union(
            *(
                set(r["forbidden_uses"])
                for r in registry["entries"]
                if "case:" + case["case_id"] in r["identities"]
            )
        )
        if not {"fit", "calibration", "final_test"} <= protected:
            raise ValueError(
                (
                    "Every pilot case must be globally excluded from future "
                    "fit/calibration/final"
                )
            )
    for anomaly in anomalies:
        pair = [c for c in controls if c["pair_id"] == anomaly["pair_id"]]
        if len(pair) != 1 or any(
            pair[0][k] != anomaly[k]
            for k in ("seed", "motion", "environment", "operation_sha256")
        ):
            raise ValueError(
                (
                    "Every anomaly needs its own matching independent normal "
                    "control"
                )
            )
    if len({c["pair_id"] for c in anomalies}) != 9:
        raise ValueError(
            "A single control cannot stand in for nine independent pairs"
        )
    if {(c.get("health_family"), c.get("timing")) for c in anomalies} != {
        (f, t)
        for f in ("battery", "propulsion", "combined")
        for t in ("existing", "step", "ramp")
    }:
        raise ValueError(
            (
                "Pilot anomaly cases must preserve all nine family/timing "
                "combinations"
            )
        )
    ledger = ctx.doc("ledger")
    if (
        ledger.get("schema") != "drone-ai.pilot45.ledger/1"
        or ledger.get("task_id") != task_id
    ):
        raise ValueError("Explicit full-case ledger required")
    entries = ledger.get("cases", [])
    if len(entries) != 45 or {r["case_id"] for r in entries} != case_ids:
        raise ValueError(
            (
                "Failed/pending/unavailable cases cannot disappear from "
                "denominator45"
            )
        )
    by_id = {c["case_id"]: c for c in cases}
    raw_ids, failures, quality_rejected = set(), 0, 0
    for row in entries:
        case, attempts = by_id[row["case_id"]], row.get("attempts", [])
        states = {
            "pending",
            "running",
            "completed",
            "failed",
            "stopped",
            "unavailable",
        }
        if quality_rejection_check is not None:
            states.add("quality_rejected")
        if len(attempts) > 3 or row.get("status") not in states:
            raise ValueError("Bounded attempt/state ledger required")
        completed = 0
        for index, attempt in enumerate(attempts, 1):
            if (
                attempt["index"] != index
                or attempt.get("status")
                not in (
                    {
                        "running",
                        "completed",
                        "failed",
                        "stopped",
                        "quality_rejected",
                    }
                    if quality_rejection_check is not None
                    else {"running", "completed", "failed", "stopped"}
                )
                or not attempt.get("run_id")
                or attempt["run_id"] in raw_ids
                or attempt.get("seed") != case["seed"]
                or not set(case["lineage"]) <= set(attempt["lineage"])
            ):
                raise ValueError(
                    "Attempt ID/index/seed/group/retry lineage changed"
                )
            if attempt.get("parent_attempt_id") != (
                attempts[index - 2]["run_id"] if index > 1 else None
            ):
                raise ValueError("Failed parent/retry chain is missing")
            if completed:
                raise ValueError("A successful raw must never be regenerated")
            if quality_rejection_check is not None:
                quality_rejected += quality_rejection_check(case, attempt)
            elif attempt.get("quality_rejection") is not None:
                raise ValueError(
                    "Quality rejection evidence is short-cycle only"
                )
            if attempt["status"] in {"failed", "stopped"} and not attempt.get(
                "failure_reason"
            ):
                raise ValueError(
                    "Failed/stopped attempts must preserve their failure reason"
                )
            completed += attempt["status"] == "completed"
            failures += attempt["status"] == "failed"
            raw_ids.add(attempt["run_id"])
        if row["status"] in {"completed", "unavailable"} and (
            not attempts or attempts[-1]["status"] != "completed"
        ):
            raise ValueError(
                (
                    "Completed/unavailable case must preserve its completed "
                    "physical raw"
                )
            )
        if row["status"] == "quality_rejected" and (
            not attempts or attempts[-1]["status"] != "quality_rejected"
        ):
            raise ValueError(
                "Quality-rejected case must retain its rejected physical raw"
            )
    ctx.checked.append(
        "frozen45_global_protection_and_full_failure_denominator"
    )
    for rid in raw_ids:
        protected = set().union(
            *(
                set(r["forbidden_uses"])
                for r in registry["entries"]
                if "run:" + rid in r["identities"]
            )
        )
        if not {"fit", "calibration", "final_test"} <= protected:
            raise ValueError(
                (
                    "Every pilot attempt, including failures, needs permanent "
                    "raw-ID exclusion"
                )
            )
    counts = dict(
        planned_cases=45,
        physical_attempts=len(raw_ids),
        failed_attempts=failures,
        failed_cases=sum(r["status"] == "failed" for r in entries),
        stopped_cases=sum(r["status"] == "stopped" for r in entries),
        unavailable_cases=sum(r["status"] == "unavailable" for r in entries),
        pending_cases=sum(r["status"] == "pending" for r in entries),
    )
    if quality_rejection_check is not None:
        counts.update(
            physical_failed_attempts=failures,
            quality_rejected_attempts=quality_rejected,
            quality_rejected_cases=sum(
                r["status"] == "quality_rejected" for r in entries
            ),
        )
    return (
        by_id,
        {r["case_id"]: r for r in entries},
        counts,
    )


def prerequisites(ctx):
    auth = json_file(AUTH)
    if (
        auth.get("task_id") != TASK
        or auth.get("authorized", {}).get("pilot_runs") != 45
    ):
        raise Pending("Current finite pilot authorization is not available")
    ctx.hashes["authorization"] = sha(AUTH)
    recovery = json_file(RECOVERY_AUTH)
    if (
        recovery.get("task_id") != TASK
        or recovery.get("authorized", {}).get("pilot_runs") != 45
        or recovery.get("forbidden", {}).get(
            "automatic_training_fit_forward_scoring_gpu"
        )
        is not True
    ):
        raise Pending(
            "Current finite recovery/no-fit authorization is not available"
        )
    ctx.hashes["recovery_authorization"] = sha(RECOVERY_AUTH)
    expected = ctx.doc("expected_producer")
    check_producer(expected, expected)
    ctx.ref("source_manifest")
    runtime = ctx.doc("runtime_manifest")
    if (
        ctx.hashes["source_manifest"] != expected["source_manifest_sha256"]
        or ctx.hashes["runtime_manifest"] != expected["runtime_sha256"]
    ):
        raise ValueError("Actual final source/runtime file hash mismatch")
    if (
        runtime.get("source_manifest_sha256")
        != expected["source_manifest_sha256"]
    ):
        raise ValueError("Runtime does not bind the final actual source")
    policy = policy_check(ctx, expected)
    capsule = ctx.doc("capsule")
    if capsule.get("schema") != "drone-ai.consumer-capsule/1":
        raise ValueError("Frozen consumer capsule required")
    bind(capsule, expected)
    if capsule.get("paired_policy_sha256") != ctx.hashes["paired_policy"]:
        raise ValueError("Consumer capsule uses a different paired policy")
    files = required(capsule, "files")
    entrypoint = required(capsule, "Data_entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.startswith(
        "Data/generation/"
    ):
        raise ValueError("Explicit Data-owned pilot entrypoint source required")
    if not set(CRITICAL) | {entrypoint} <= set(files):
        raise ValueError(
            "Capsule must cover critical AI readers and the Data entrypoint"
        )
    for name, expected_hash in files.items():
        if not isinstance(name, str) or not name.startswith(
            ("ai/", "Data/generation/")
        ):
            raise ValueError(
                "Consumer capsule may contain only AI/Data source paths"
            )
        path = regular(ROOT / name)
        if path.suffix not in {
            ".py",
            ".json",
            ".sh",
        } or path.stat().st_size > 2 * (1 << 20):
            raise ValueError(
                (
                    "Capsule hashes small source/config files, never "
                    "model/data trees"
                )
            )
        if expected_hash != sha(path):
            raise ValueError(
                "Current consumer source differs from capsule: " + name
            )
    config = json_file(ROOT / "ai/learning/history19/preprocessing.json")
    if any(
        config.get(k) != v
        for k, v in dict(
            feature_profile="common19_ctx0",
            step_us=100000,
            history=100,
            future=10,
            emit_every_us=1000000,
            context_us=0,
            nominal_capacity_ah=12,
        ).items()
    ):
        raise ValueError(
            "Current fixed19/history/future/unit preprocessing contract changed"
        )
    ids = development_check(ctx, expected, policy)
    registry = ctx.doc("registry")
    check_protection([], registry, usage="pilot_input_check")
    historical = json_file(HISTORICAL_EXCLUSIONS)
    ctx.hashes["historical_reference_exclusions"] = sha(HISTORICAL_EXCLUSIONS)
    if (
        historical.get("count") != 77
        or len(set(historical["raw_attempt_ids"])) != 77
    ):
        raise ValueError("Preserved development77 inventory changed")
    for rid in ids | set(historical["raw_attempt_ids"]):
        protected = set().union(
            *(
                set(r["forbidden_uses"])
                for r in registry["entries"]
                if "run:" + rid in r["identities"]
            )
        )
        if (
            not {"pilot_input_check", "fit", "calibration", "final_test"}
            <= protected
        ):
            raise ValueError(
                (
                    "Development77+6 must be globally excluded from "
                    "pilot/fit/calibration/final"
                )
            )
    cases, ledger, counts = plan_and_ledger(ctx, registry)
    ctx.ledger_counts = counts
    ctx.checked.append("final_producer_and_consumer_capsule")
    return expected, policy, registry, cases, ledger, counts


def main(request_path, output=None):
    ctx = None
    try:
        ctx = Evidence(request_path)
        request = ctx.request
        if (
            request.get("schema") != "drone-ai.pilot45.check/1"
            or request.get("task_id") != TASK
            or request.get("purpose") != PILOT
        ):
            raise ValueError("Explicit P27 pilot-only check request required")
        expected, policy, registry, cases, ledger, counts = prerequisites(ctx)
        phase = request.get("phase")
        if phase == "case":
            from .pilot_views import check_case

            case_id = required(request, "case_id")
            result = check_case(
                ctx,
                expected,
                policy,
                registry,
                cases[case_id],
                ledger[case_id],
                cases,
                ledger,
            )
        elif phase == "prerequisites":
            result = dict(
                status="PREREQUISITES_CHECKED_NO_CASE_DATA",
                scope="prerequisites_only_not_future45_inputs",
            )
        else:
            raise ValueError("phase must be prerequisites or case")
        fixture = request.get("fixture_only") is True
        if fixture:
            result["status"] = "SYNTHETIC_" + result["status"]
        result.update(
            actual_data_verified=not fixture,
            usable_to_start_real_pilot=False,
            ai_prerequisites_passed=not fixture and phase == "prerequisites",
            pilot_start_requires_Data_and_coordinator_gates=True,
            fixture_only=fixture,
            ledger_counts=counts,
        )
        code = 0
    except (Pending, FileNotFoundError) as error:
        result, code = (
            dict(
                status="BLOCKED_PENDING_INPUT",
                reason=str(error),
                actual_data_verified=False,
                usable_to_start_real_pilot=False,
            ),
            2,
        )
    except Unavailable as error:
        result, code = (
            dict(
                status="INPUT_UNAVAILABLE",
                reason=str(error),
                actual_data_verified=False,
                usable_to_start_real_pilot=False,
            ),
            3,
        )
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        result, code = (
            dict(
                status="FAILED_INPUT_CHECK",
                reason=str(error),
                actual_data_verified=False,
                usable_to_start_real_pilot=False,
            ),
            1,
        )
    result.update(
        schema="drone-ai.pilot45.check-receipt/1",
        task_id=TASK,
        purpose=PILOT,
        eligible_for_fit=False,
        exit_code=code,
        checked_stages=ctx.checked if ctx else [],
        evidence_sha256=ctx.hashes if ctx else {},
        model_forward_steps=0,
        optimizer_steps=0,
        scaler_threshold_fits=0,
        GPU_jobs=0,
        phase=ctx.request.get("phase") if ctx else None,
        case_id=ctx.request.get("case_id") if ctx else None,
        expected_case_denominator=45,
    )
    if ctx and hasattr(ctx, "ledger_counts"):
        result["ledger_counts"] = ctx.ledger_counts
    if output is not None:
        output = Path(output).absolute()
        if (
            ".." in output.parts
            or any(p.is_symlink() for p in (output, *output.parents))
            or (ctx and output in ctx.paths.values())
        ):
            raise ValueError(
                "Receipt output must be a new regular path, separate from input"
            )
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    return code
