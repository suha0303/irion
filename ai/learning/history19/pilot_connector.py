"""One-shot path-to-request connector; no producer execution or native-field
guessing.
"""

import argparse
import json
from pathlib import Path

from learning.admission19 import MISSION_POLICY, PILOT

from .duration_profiles import NATIVE_ADAPTERS
from .pilot import (
    TASK,
    Pending,
    check_native_completion,
    json_file,
    project_native,
    regular,
    required,
    sha,
)

PREREQUISITE_REFS = (
    "expected_producer",
    "source_manifest",
    "runtime_manifest",
    "paired_policy",
    "native_policy",
    "native_completion",
    "capsule",
    "development_receipt",
    "registry",
    "pilot_manifest",
    "ledger",
)
CASE_REFS = (
    "scenario",
    "raw_integrity",
    "battery_selection",
    "propulsion_selection",
    "canonical_frames",
    "online_manifest",
    "run_manifest",
)


def project_pair_receipt(
    report_path,
    *,
    policy_path,
    normal_run=None,
    anomaly_run=None,
    external_audits=None,
):
    """Data calls this after the official native verifier; no Data field
    mapping.
    """
    from .joint_native import pair

    policy = json_file(policy_path)
    if (
        policy.get("native_adapter") not in NATIVE_ADAPTERS
        or policy.get("status") != "FROZEN"
    ):
        raise ValueError("Reviewed frozen joint-native adapter required")
    return pair(
        report_path,
        policy,
        fixture=policy.get("fixture_only") is True,
        normal_run=normal_run,
        anomaly_run=anomaly_run,
        external_audits=external_audits,
    )


def native_mission_interval(
    receipt_path,
    *,
    policy_path,
    native_policy_path,
    run_path,
    external_audit=None,
):
    """Return the complete actual controller-consumption interval, including
    LAND tail.

    The source member name and native field paths come from the
    AI-reviewed frozen
    projection, after Sim publishes its final receipt. No fallback to
    request/ACK.
    """
    receipt_path, run_path = regular(receipt_path), regular(run_path)
    policy, native_policy, run = (
        json_file(policy_path),
        json_file(native_policy_path),
        json_file(run_path),
    )
    from .duration_profiles import duration_us

    duration = duration_us(policy)
    short = (
        policy.get("scope_task") == "SHORT_CYCLE_20261107"
        and policy.get("approved_for_short_cycle") is True
        and policy.get("duration_profile") == "short180"
    )
    standard = (
        policy.get("approved_for_pilot") is True and duration == 480000000
    )
    final16 = False
    collection = False
    if policy.get("scope_task") == "MAIN_COLLECTION_CONNECTION_20261107":
        from learning.collection19.contract import validate_policy

        validate_policy(policy, native_policy_path)
        collection = True
    if policy.get("scope_task") == "FINAL16_20261107":
        from learning.admission19 import FINAL16

        from .final16 import AUTH_SHA256, authorization

        authorization(fixture_only=policy.get("fixture_only") is True)
        final16 = (
            policy.get("approved_for_final16") is True
            and policy.get("authorization_sha256") == AUTH_SHA256
            and policy.get("purpose") == FINAL16
            and policy.get("duration_profile") == "short180"
        )
    if (
        policy.get("status") != "FROZEN"
        or not (short or standard or final16 or collection)
        or policy.get("alignment_mode") != "versioned_joint_phase"
        or policy.get("posthoc_adjustment") is not False
        or policy.get("mission_policy") != MISSION_POLICY
        or native_policy.get("schema") != "drone.joint-phase-policy/1"
        or native_policy.get("revision") not in NATIVE_ADAPTERS
        or policy.get("native_policy_sha256")
        != sha(regular(native_policy_path))
    ):
        raise ValueError("Frozen reviewed joint-native policy binding required")
    source = required(policy, "mission_source")
    if (
        not isinstance(source, str)
        or not source.startswith("evaluation/")
        or not source.endswith(".json")
        or ".." in Path(source).parts
        or receipt_path != regular(run_path.parent / source)
    ):
        raise ValueError(
            "Native boundary source must be the exact frozen raw member"
        )
    if policy.get("native_adapter") in NATIVE_ADAPTERS:
        from .joint_native import check_policy, origin

        check_policy(
            native_policy,
            duration_profile=policy.get("duration_profile", "standard480"),
        )
        projected = origin(
            run_path,
            policy,
            fixture=policy.get("fixture_only") is True,
            external_audit=external_audit,
        )
    else:
        projected = project_native(json_file(receipt_path), policy)
    for key in ("actual_mission_start_us", "actual_landing_start_us"):
        if type(projected[key]) is not int:
            raise ValueError(
                "Actual integer controller-consumption boundary required"
            )
    if projected["source_acknowledged"] is not True:
        raise Pending("Actual native controller receipt is unconfirmed")
    if (
        projected["run_id"] != run["run_id"]
        or projected["clock_id"] != run["clock"]["id"]
    ):
        raise ValueError("Native boundary run/clock mismatch")
    if projected["policy_id"] != policy["policy_id"] or any(
        projected[key] != policy["producer"][key]
        for key in ("source_manifest_sha256", "runtime_sha256")
    ):
        raise ValueError("Native boundary source/policy mismatch")
    start, end = (
        projected["actual_mission_start_us"],
        projected["actual_landing_start_us"],
    )
    if (
        policy.get("native_adapter") not in NATIVE_ADAPTERS
        and required(native_policy, "duration_us") != duration
    ):
        raise ValueError(
            "Native policy and explicit AI duration profile differ"
        )
    tail = end - start - duration
    if start < required(
        native_policy, "minimum_start_us"
    ) or not 0 <= tail <= required(
        native_policy, "maximum_landing_activation_delay_us"
    ):
        raise ValueError(
            "Complete declared mission and retained native LAND tail required"
        )
    interval = dict(
        policy=MISSION_POLICY,
        start_us=start,
        end_us_exclusive=end,
        clock_id=projected["clock_id"],
        source=source,
        source_sha256=sha(receipt_path),
        boundary_kind="actual_native_controller_consumption",
        native_policy_revision=native_policy["revision"],
        landing_tail_us=tail,
        duration_profile=policy.get("duration_profile", "standard480"),
        mission_duration_us=duration,
    )
    if "native_sources_sha256" in projected:
        interval["native_sources_sha256"] = projected["native_sources_sha256"]
    if "external_audit_evidence" in projected:
        interval["external_audit_evidence"] = projected[
            "external_audit_evidence"
        ]
    return interval


def prepare_request(bundle_path, output):
    """Hash delivered paths once. AI owns paired_policy/development projections.

    A new Sim schema is deliberately rejected until its actual mapping is
    reviewed.
    This writes a request only; callers must then run the independent
    check-input.
    """
    bundle_path = regular(bundle_path)
    bundle = json_file(bundle_path)
    if (
        bundle.get("schema") != "drone-ai.pilot45.connector/1"
        or bundle.get("task_id") != TASK
    ):
        raise ValueError("Explicit pilot connector bundle required")
    paths = required(bundle, "paths")

    def ref(name, value=None):
        path = Path(required(paths, name) if value is None else value)
        path = regular(
            path if path.is_absolute() else bundle_path.parent / path
        )
        return dict(path=str(path), sha256=sha(path))

    native = ref("native_completion")
    fixture = bundle.get("fixture_only") is True
    check_native_completion(json_file(native["path"]), fixture=fixture)
    phase = required(bundle, "phase")
    if phase not in {"prerequisites", "case"}:
        raise ValueError("Connector phase must be prerequisites or case")
    request = dict(
        schema="drone-ai.pilot45.check/1",
        task_id=TASK,
        purpose=PILOT,
        phase=phase,
        refs={
            name: (native if name == "native_completion" else ref(name))
            for name in PREREQUISITE_REFS
        },
    )
    if fixture:
        request["fixture_only"] = True
    for key in ("external_audit", "external_audits"):
        if key in bundle:
            request[key] = bundle[key]
    if phase == "case":
        request["case_id"] = required(bundle, "case_id")
        request["case_origin"] = {
            name: ref(name, required(bundle.get("case_origin", {}), name))
            for name in ("run", "execution", "origin")
        }
        request["refs"].update({name: ref(name) for name in CASE_REFS})
        if paths.get("pair_receipt") is not None:
            request["refs"]["pair_receipt"] = ref("pair_receipt")
    output = Path(output).absolute()
    if ".." in output.parts or any(
        p.is_symlink() for p in (output, *output.parents)
    ):
        raise ValueError("New regular request output required")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(request, indent=2, allow_nan=False) + "\n")
    return dict(
        status="REQUEST_WRITTEN_NOT_CHECKED",
        request=str(output),
        request_sha256=sha(output),
        actual_data_verified=False,
        usable_to_start_real_pilot=False,
        fixture_only=fixture,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result, code = prepare_request(args.bundle, args.output), 0
    except (Pending, FileNotFoundError) as error:
        result, code = (
            dict(status="BLOCKED_PENDING_INPUT", reason=str(error)),
            2,
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        result, code = (
            dict(status="FAILED_REQUEST_PREPARATION", reason=str(error)),
            1,
        )
    result.update(
        exit_code=code, model_forward_steps=0, optimizer_steps=0, GPU_jobs=0
    )
    print(json.dumps(result, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
