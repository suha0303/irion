"""Explicit180s no-fit input gate. No dependency on old development6 or pilot READY."""

import argparse
import json
from pathlib import Path

from learning.admission19 import SHORT, check_producer, check_protection

from .duration_profiles import duration_us
from .pilot import (
    CRITICAL,
    ROOT,
    Evidence,
    Pending,
    Unavailable,
    json_file,
    plan_and_ledger,
    policy_check,
    regular,
    sha,
)

TASK = "SHORT_CYCLE_20261107"
AUTH = ROOT / "ai/reports/short-cycle-20261107/attempt-01/authorization.json"


def prepare_policy(
    native_policy_path, producer_binding_path, output, *, fixture_only=False
):
    """Write a new AI projection only after explicit short auth and native180 policy."""
    from .joint_native import ORIGIN, WIND_CLOCK, check_policy

    auth = json_file(AUTH)
    if (
        auth.get("task_id") != TASK
        or auth.get("scope", {}).get("proposed_mission_seconds") != 180
    ):
        raise ValueError("Explicit short-cycle180 authorization required")
    native = json_file(native_policy_path)
    check_policy(native, duration_profile="short180")
    producer = json_file(producer_binding_path)
    check_producer(producer, producer)
    value = dict(
        schema="drone-ai.paired-source-clock-policy/1",
        status="FROZEN",
        approved_for_short_cycle=True,
        approved_for_pilot=False,
        scope_task=TASK,
        purpose=SHORT,
        duration_profile="short180",
        policy_id=native["revision"],
        native_adapter=native["revision"],
        alignment_mode="versioned_joint_phase",
        mission_policy="mission_only_actual_execution/1",
        posthoc_adjustment=False,
        producer=producer,
        native_policy_sha256=sha(regular(native_policy_path)),
        native_receipt_schema="drone.joint-phase-native-receipt/1",
        mission_source=ORIGIN,
        wind_model_clock=WIND_CLOCK,
        fixture_only=fixture_only,
        limits=dict(
            origin_delta_us=0,
            wind_rms_mps=native["acceptance"]["wind_rms_mps"],
            wind_peak_mps=native["acceptance"]["wind_max_mps"],
            receipt_min_offset_us=0,
            receipt_max_offset_us=native["maximum_origin_notice_delay_us"],
        ),
    )
    output = Path(output).absolute()
    if ".." in output.parts or any(
        p.is_symlink() for p in (output, *output.parents)
    ):
        raise ValueError("New regular short policy output required")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    return value


def prepare_request(bundle_path, output):
    """Hash explicit short input paths, without any development6 dependency."""
    from .pilot_connector import CASE_REFS

    bundle_path = regular(bundle_path)
    bundle = json_file(bundle_path)
    if (
        bundle.get("schema") != "drone-ai.short-cycle.connector/1"
        or bundle.get("task_id") != TASK
    ):
        raise ValueError("Explicit short-cycle connector bundle required")

    def ref(path):
        path = Path(path)
        path = regular(
            path if path.is_absolute() else bundle_path.parent / path
        )
        return dict(path=str(path), sha256=sha(path))

    names = (
        "expected_producer",
        "source_manifest",
        "runtime_manifest",
        "paired_policy",
        "native_policy",
        "capsule",
        "registry",
        "pilot_manifest",
        "ledger",
    )
    paths = bundle.get("paths", {})
    request = dict(
        schema="drone-ai.short-cycle.check/1",
        task_id=TASK,
        purpose=SHORT,
        phase=bundle["phase"],
        refs={name: ref(paths[name]) for name in names},
        fixture_only=bundle.get("fixture_only") is True,
    )
    if bundle["phase"] == "case":
        request.update(
            case_id=bundle["case_id"],
            case_origin={
                key: ref(bundle["case_origin"][key])
                for key in ("run", "execution", "origin")
            },
        )
        request["refs"].update({key: ref(paths[key]) for key in CASE_REFS})
        if paths.get("pair_receipt") is not None:
            request["refs"]["pair_receipt"] = ref(paths["pair_receipt"])
    if bundle["phase"] == "final":
        request["case_receipts"] = [
            ref(path) for path in bundle["case_receipts"]
        ]
    for key in ("external_audit", "external_audits"):
        if key in bundle:
            request[key] = bundle[key]
    output = Path(output).absolute()
    if ".." in output.parts or any(
        p.is_symlink() for p in (output, *output.parents)
    ):
        raise ValueError("New separate request output required")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(request, indent=2) + "\n")
    return request


def consumer_inputs(
    ctx, *, scope_task=TASK, cohort=SHORT, usage="short_cycle_input_check"
):
    """Shared generation/config verification; no campaign membership or count changes."""
    from learning.admission19 import FINAL16

    if (scope_task, cohort, usage) not in {
        (TASK, SHORT, "short_cycle_input_check"),
        ("FINAL16_20261107", FINAL16, "final16_input_check"),
    }:
        raise ValueError("Only the two explicit180s input scopes are supported")
    if any(
        ctx.request.get("refs", {}).get(k) is not None
        for k in ("native_completion", "development_receipt")
    ):
        raise ValueError(
            "Short-cycle must not depend on historical development6 or pilot READY"
        )
    expected = ctx.doc("expected_producer")
    check_producer(expected, expected)
    ctx.ref("source_manifest")
    runtime = ctx.doc("runtime_manifest")
    if (
        ctx.hashes["source_manifest"] != expected["source_manifest_sha256"]
        or ctx.hashes["runtime_manifest"] != expected["runtime_sha256"]
        or runtime.get("source_manifest_sha256")
        != expected["source_manifest_sha256"]
    ):
        raise ValueError(
            "Short current generation source/runtime binding differs"
        )
    policy = policy_check(ctx, expected, short_cycle=True)
    if (
        policy.get("scope_task") != scope_task
        or policy.get("purpose") != cohort
        or policy.get("duration_profile") != "short180"
        or duration_us(policy) != 180000000
    ):
        raise ValueError(
            "A separate short180 policy is required, never old480 relabelling"
        )
    capsule = ctx.doc("capsule")
    check_producer(capsule["producer"], expected)
    if (
        capsule.get("schema") != "drone-ai.consumer-capsule/1"
        or capsule.get("paired_policy_sha256") != ctx.hashes["paired_policy"]
    ):
        raise ValueError("Frozen short consumer capsule required")
    files = capsule["files"]
    entry = capsule["Data_entrypoint"]
    if not entry.startswith("Data/generation/") or not set(CRITICAL) | {
        entry
    } <= set(files):
        raise ValueError(
            "Short capsule must include AI readers and actual Data entrypoint"
        )
    for name, digest in files.items():
        if not name.startswith(("ai/", "Data/generation/")):
            raise ValueError(
                "Only owned consumer source/config paths in capsule"
            )
        path = regular(ROOT / name)
        if (
            path.suffix not in {".py", ".json", ".sh"}
            or path.stat().st_size > 2 * (1 << 20)
            or sha(path) != digest
        ):
            raise ValueError("Short consumer source/config changed: " + name)
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
        raise ValueError("Fixed dual19/history100/future10 contract changed")
    registry = ctx.doc("registry")
    check_protection([], registry, usage=usage)
    return expected, policy, registry


def prerequisites(ctx):
    auth = json_file(AUTH)
    scope = auth.get("scope", {})
    if (
        auth.get("task_id") != TASK
        or scope.get("proposed_mission_seconds") != 180
        or scope.get("logical_cases_per_cycle") != 45
        or scope.get("training_fit_model_forward_gpu") is not False
    ):
        raise ValueError(
            "Explicit180s45case no-fit short-cycle authority required"
        )
    ctx.hashes["short_authorization"] = sha(AUTH)
    expected, policy, registry = consumer_inputs(ctx)
    from .quality_rejection import verify_attempt

    cases, ledger, counts = plan_and_ledger(
        ctx,
        registry,
        task_id=TASK,
        cohort=SHORT,
        usage="short_cycle_input_check",
        quality_rejection_check=lambda case, attempt: verify_attempt(
            ctx,
            case,
            attempt,
            expected_producer=expected,
            native_policy_sha256=policy["native_policy_sha256"],
        ),
    )
    ctx.ledger_counts = counts
    ctx.checked.append(
        "short180_static_generation_capsule_full45_no_development6"
    )
    return expected, policy, registry, cases, ledger, counts


def final_checks(ctx, cases, ledger):
    refs = ctx.request.get("case_receipts")
    if refs is None:
        raise Pending("All45 actual case input receipts are required")
    if len(refs) != 45:
        raise ValueError(
            "Exactly45 case input receipts; no failed case removal"
        )
    seen = set()
    pairs = {}
    for i, ref in enumerate(refs):
        r = ctx.doc("case_receipt:" + str(i), ref)
        cid = r["case_id"]
        if cid in seen or cid not in cases:
            raise ValueError("Unique full45 case membership required")
        seen.add(cid)
        c = cases[cid]
        entry = ledger[cid]
        if (
            r.get("task_id") != TASK
            or r.get("purpose") != SHORT
            or r.get("fixture_only")
            or r.get("actual_data_verified") is not True
            or r.get("exit_code") != 0
            or r.get("phase") != "case"
            or r.get("status") != "SHORT180_CASE_INPUTS_CHECKED_NO_FIT"
            or r.get("duration_profile") != "short180"
            or r.get("mission_duration_us") != 180000000
            or r.get("model_forward_steps") != 0
            or r.get("frames", 0) <= 100
            or r.get("online", {}).get("windows", 0) <= 0
        ):
            raise ValueError(
                "All45 actual nonempty no-fit case checks required"
            )
        if (
            entry["status"] != "completed"
            or r["run_id"] != entry["attempts"][-1]["run_id"]
        ):
            raise ValueError(
                "Unresolved/failed/unavailable case remains in full45 denominator"
            )
        for key in (
            "source_manifest",
            "runtime_manifest",
            "paired_policy",
            "capsule",
        ):
            if r["evidence_sha256"].get(key) != ctx.hashes[key]:
                raise ValueError("Case producer/policy/config changed")
        if c["case_kind"] != "normal_base":
            if r.get("pair_verified") is not True:
                raise ValueError("Actual full9pair verification required")
            evidence = r["evidence_sha256"]["pilot_pair_native_evidence"]
            pairs.setdefault(c["pair_id"], []).append(evidence)
    if (
        seen != set(cases)
        or len(pairs) != 9
        or any(len(v) != 2 or v[0] != v[1] for v in pairs.values())
        or len({v[0] for v in pairs.values()}) != 9
    ):
        raise ValueError(
            "Nine distinct native pair reports must bind all18 paired cases"
        )
    return dict(
        status="SHORT45_INPUT_AND_PAIR_CHECKS_PASSED",
        cases_checked=45,
        pairs_checked=9,
        scope="AI_input_only_Data_quality_and_actual_termination_still_required",
    )


def main(request_path, output=None):
    ctx = None
    try:
        ctx = Evidence(request_path)
        request = ctx.request
        if (
            request.get("schema") != "drone-ai.short-cycle.check/1"
            or request.get("task_id") != TASK
            or request.get("purpose") != SHORT
        ):
            raise ValueError("Explicit short-cycle request required")
        expected, policy, registry, cases, ledger, counts = prerequisites(ctx)
        phase = request.get("phase")
        if phase == "prerequisites":
            result = dict(status="SHORT180_STATIC_INPUTS_CHECKED_NO_FLIGHTS")
        elif phase == "case":
            from .pilot_views import check_case

            cid = request["case_id"]
            result = check_case(
                ctx,
                expected,
                policy,
                registry,
                cases[cid],
                ledger[cid],
                cases,
                ledger,
                cohort=SHORT,
                usage="short_cycle_input_check",
            )
            result["status"] = "SHORT180_CASE_INPUTS_CHECKED_NO_FIT"
            mission = ctx.doc("online_manifest")["run"]["mission_interval"]
            if (
                mission.get("duration_profile") != "short180"
                or mission.get("mission_duration_us") != 180000000
            ):
                raise ValueError(
                    "Short views require explicit short180 interval metadata"
                )
            result.update(
                duration_profile="short180", mission_duration_us=180000000
            )
        elif phase == "final":
            result = final_checks(ctx, cases, ledger)
        else:
            raise ValueError("phase must be prerequisites, case or final")
        fixture = request.get("fixture_only") is True
        if fixture:
            result["status"] = "SYNTHETIC_" + result["status"]
        result.update(
            actual_data_verified=not fixture,
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
            ),
            2,
        )
    except Unavailable as error:
        result, code = (
            dict(
                status="INPUT_UNAVAILABLE",
                reason=str(error),
                actual_data_verified=False,
            ),
            3,
        )
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        result, code = (
            dict(
                status="FAILED_INPUT_CHECK",
                reason=str(error),
                actual_data_verified=False,
            ),
            1,
        )
    result.update(
        schema="drone-ai.short-cycle.check-receipt/1",
        task_id=TASK,
        purpose=SHORT,
        phase=ctx.request.get("phase") if ctx else None,
        eligible_for_fit=False,
        usable_to_start_real_short_cycle=False,
        requires_Data_Sim_build_quality_termination_gates=True,
        expected_case_denominator=45,
        exit_code=code,
        checked_stages=ctx.checked if ctx else [],
        evidence_sha256=ctx.hashes if ctx else {},
        model_forward_steps=0,
        optimizer_steps=0,
        scaler_threshold_fits=0,
        GPU_jobs=0,
    )
    if output is not None:
        output = Path(output).absolute()
        if (
            ".." in output.parts
            or any(p.is_symlink() for p in (output, *output.parents))
            or (ctx and output in ctx.paths.values())
        ):
            raise ValueError("New separate receipt output required")
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    return code


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return main(args.request, args.output)


if __name__ == "__main__":
    raise SystemExit(cli())
