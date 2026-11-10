"""One-shot FINAL16 input checks. No flights, retries, model imports or
fitting.
"""

import argparse
import hashlib
import json
from pathlib import Path

from learning.admission19 import FINAL16, check_producer, check_protection
from learning.history19 import joint_native, pilot_connector

from . import pilot
from .pilot import ROOT, Evidence, Pending, Unavailable, json_file, regular, sha
from .short_cycle import consumer_inputs

CRITICAL = pilot.CRITICAL

TASK = "FINAL16_20261107"
AUTH = ROOT / "ai/reports/final16-20261107/attempt-01/authorization.json"
AUTH_SHA256 = "05017f2adc41caf14906ecab4ce82e02d9313b097da8be4609e18a3b9ca3041b"
EXPECTED = AUTH.parent / "expected-manifest-cases.json"
EXPECTED_SHA256 = (
    "f32ca8d740d8ebf5296cd0ea49568f34edacf53ea6d4617eefb4d4f767b01fb4"
)
SOURCE_EXPECTED = AUTH.parent / "expected-cases.json"
SOURCE_EXPECTED_SHA256 = (
    "1577246f5e72f473f880d54813112aef5690e5e8de66c62cba072b27a015108f"
)
USAGE = "final16_input_check"
PATH_KEYS = (
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


def authorization(*, fixture_only=False):
    if sha(regular(AUTH)) != AUTH_SHA256:
        raise ValueError("FINAL16 coordinator authorization changed")
    value = json_file(AUTH)
    fixed = dict(
        proposed_mission_seconds=180,
        logical_cases_per_cycle=16,
        normal_cases=8,
        anomaly_cases=8,
        native_pairs=8,
        source_pair_indices=list(range(1, 9)),
        case_prefix="final16",
        flight_workers=16,
        requested_rtf=2,
        aggregate_nonflight_workers=4,
        nonflight_when_flights_idle=True,
        physical_attempts_max_per_logical_case=1,
        physical_attempts_max_total=16,
        stop_on_first_physical_failure=True,
        automatic_reflight=False,
        training_fit_model_forward_gpu=False,
        LR_collection=False,
        main_collection=False,
        production_default_mission_seconds=480,
        prior45_reuse_as_new16_pass_forbidden=True,
    )
    scope = value.get("scope", {})
    if (
        value.get("schema") != "drone.coordinator.final16-authorization/1"
        or value.get("task_id") != TASK
        or (value.get("fixture_only") is True) is not fixture_only
        or any(scope.get(k) != v for k, v in fixed.items())
        or type(scope.get("seed_base")) is not int
        or not 0 <= scope["seed_base"] < 2**32
    ):
        raise ValueError(
            "Exact authorized16/8pair/one-attempt/no-fit scope required"
        )
    return value


def expected_plan(namespace, *, fixture_only=False):
    """Project the coordinator's independently frozen source definitions, not
    old flights.
    """
    auth = authorization(fixture_only=fixture_only)
    if sha(regular(EXPECTED)) != EXPECTED_SHA256:
        raise ValueError("Coordinator expected16 definitions changed")
    source = json_file(EXPECTED)
    if (
        source.get("schema")
        != "drone.coordinator.final16-expected-manifest-cases/1"
        or source.get("task_id") != TASK
        or source.get("authorization_sha256") != AUTH_SHA256
        or source.get("seed_base") != auth["scope"]["seed_base"]
        or len(source.get("cases", [])) != 16
        or not isinstance(namespace, str)
        or not namespace.startswith("final16")
        or any(
            c
            not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            )
            for c in namespace
        )
    ):
        raise ValueError("Frozen expected16 and new final16 namespace required")
    if (
        sha(regular(SOURCE_EXPECTED)) != SOURCE_EXPECTED_SHA256
        or source.get("source_definition")
        != dict(path=str(SOURCE_EXPECTED), sha256=SOURCE_EXPECTED_SHA256)
        or {c["case_id"]: c for c in source["cases"]}
        != {
            c["case_id"]: dict(c, cohort=FINAL16)
            for c in json_file(SOURCE_EXPECTED)["cases"]
        }
    ):
        raise ValueError(
            (
                "Only the no-fit cohort projection of original definitions is "
                "allowed"
            )
        )
    names = {
        f"final16-pair-{i:02d}-{role}"
        for i in range(1, 9)
        for role in ("normal", "anomaly")
    }
    if {c["case_id"] for c in source["cases"]} != names:
        raise ValueError(
            "Only the exact pair01..08 normal/anomaly definitions are allowed"
        )
    cases = []
    for c in source["cases"]:
        if c.get("attempts_max") != 1 or c.get("duration") != 180:
            raise ValueError(
                "Expected case must have exactly one180s physical attempt"
            )
        operation = json.dumps(
            c["operation_variation"], sort_keys=True, separators=(",", ":")
        )
        cases.append(
            dict(
                case_id=namespace + ":" + c["case_id"],
                case_kind=c["category"],
                role=c["role"],
                seed=c["seed"],
                motion=c["kind"],
                environment=c["environment"],
                operation_sha256=hashlib.sha256(operation.encode()).hexdigest(),
                pair_id=c["paired_case_id"],
                health_family=c["anomaly_family"],
                timing=c["anomaly_timing"],
                lineage=[
                    "pair:" + c["paired_case_id"],
                    "group:" + namespace + ":" + c["group_id"],
                    "retry:" + namespace + ":" + c["retry_family_id"],
                ],
            )
        )
    return dict(
        schema="drone-ai.final16.manifest/1",
        task_id=TASK,
        cohort=FINAL16,
        purpose=FINAL16,
        no_fit=True,
        status="FROZEN",
        cases=cases,
    )


def prepare_policy(
    native_policy_path, producer_binding_path, output, *, fixture_only=False
):
    auth = authorization(fixture_only=fixture_only)
    native = json_file(native_policy_path)
    joint_native.check_policy(native, duration_profile="short180")
    producer = json_file(producer_binding_path)
    check_producer(producer, producer)
    if (producer["source_manifest_sha256"], producer["runtime_sha256"]) != (
        auth["producer"]["source_sha256"],
        auth["producer"]["runtime_sha256"],
    ):
        raise ValueError(
            "FINAL16 producer must match the new scope authorization"
        )
    value = dict(
        schema="drone-ai.paired-source-clock-policy/1",
        status="FROZEN",
        approved_for_final16=True,
        approved_for_short_cycle=True,
        approved_for_pilot=False,
        scope_task=TASK,
        purpose=FINAL16,
        authorization_sha256=AUTH_SHA256,
        expected_cases_sha256=EXPECTED_SHA256,
        duration_profile="short180",
        policy_id=native["revision"],
        native_adapter=native["revision"],
        alignment_mode="versioned_joint_phase",
        mission_policy="mission_only_actual_execution/1",
        posthoc_adjustment=False,
        producer=producer,
        native_policy_sha256=sha(regular(native_policy_path)),
        native_receipt_schema="drone.joint-phase-native-receipt/1",
        mission_source=joint_native.ORIGIN,
        wind_model_clock=joint_native.WIND_CLOCK,
        fixture_only=fixture_only,
        limits=dict(
            origin_delta_us=0,
            wind_rms_mps=2e-9,
            wind_peak_mps=2e-9,
            receipt_min_offset_us=0,
            receipt_max_offset_us=500000,
        ),
    )
    _write(output, value)
    return value


def _write(output, value):
    output = Path(output).absolute()
    if ".." in output.parts or any(
        p.is_symlink() for p in (output, *output.parents)
    ):
        raise ValueError("New regular FINAL16 output required")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def prepare_request(bundle_path, output):
    bundle_path = regular(bundle_path)
    bundle = json_file(bundle_path)
    if (
        bundle.get("schema") != "drone-ai.final16.connector/1"
        or bundle.get("task_id") != TASK
    ):
        raise ValueError("Separate FINAL16 connector bundle required")

    def ref(path):
        path = Path(path)
        path = regular(
            path if path.is_absolute() else bundle_path.parent / path
        )
        return dict(path=str(path), sha256=sha(path))

    request = dict(
        schema="drone-ai.final16.check/1",
        task_id=TASK,
        purpose=FINAL16,
        phase=bundle["phase"],
        fixture_only=bundle.get("fixture_only") is True,
        refs={key: ref(bundle["paths"][key]) for key in PATH_KEYS},
    )
    if bundle["phase"] == "case":
        request.update(
            case_id=bundle["case_id"],
            case_origin={
                key: ref(bundle["case_origin"][key])
                for key in ("run", "execution", "origin")
            },
        )
        request["refs"].update(
            {
                key: ref(bundle["paths"][key])
                for key in pilot_connector.CASE_REFS
            }
        )
        request["refs"]["pair_receipt"] = ref(bundle["paths"]["pair_receipt"])
    if bundle["phase"] == "final":
        request["case_receipts"] = [
            ref(path) for path in bundle["case_receipts"]
        ]
    if any(
        key in bundle
        for key in ("external_audit", "external_audits", "quality_rejection")
    ):
        raise ValueError(
            (
                "Historical external audits and retry exceptions are outside "
                "FINAL16"
            )
        )
    _write(output, request)
    return request


def plan_and_ledger(ctx, registry):
    manifest = ctx.doc("pilot_manifest")
    cases = manifest.get("cases", [])
    if (
        not cases
        or not isinstance(cases[0].get("case_id"), str)
        or ":" not in cases[0]["case_id"]
    ):
        raise ValueError("New qualified FINAL16 case identities required")
    namespace = cases[0]["case_id"].rsplit(":", 1)[0]
    expected = expected_plan(
        namespace, fixture_only=ctx.request.get("fixture_only") is True
    )
    if any(manifest.get(k) != v for k, v in expected.items() if k != "cases"):
        raise ValueError(
            "Separate frozen FINAL16 manifest/no-fit cohort required"
        )
    by_id = {c["case_id"]: c for c in cases}
    if len(cases) != len(by_id) or by_id != {
        c["case_id"]: c for c in expected["cases"]
    }:
        raise ValueError(
            (
                "Exactly16 unchanged expected cases, fresh seeds/groups and "
                "eight pairs required"
            )
        )
    check_protection(cases, registry, usage=USAGE)
    # Prior entries must explicitly exclude this new scope too, not just old45
    # intake.
    for entry in registry["entries"]:
        if (
            entry.get("purpose") != FINAL16
            and USAGE not in entry["forbidden_uses"]
        ):
            raise ValueError(
                "Prior data must be globally excluded from FINAL16 input reuse"
            )

    def protected(identity):
        uses = set().union(
            *(
                set(r["forbidden_uses"])
                for r in registry["entries"]
                if identity in r["identities"]
            )
        )
        if not {"fit", "calibration", "final_test"} <= uses:
            raise ValueError(
                (
                    "Every FINAL16 case/attempt, including failures, needs "
                    "permanent fit exclusion"
                )
            )

    ledger = ctx.doc("ledger")
    rows = ledger.get("cases", [])
    if (
        ledger.get("schema") != "drone-ai.final16.ledger/1"
        or ledger.get("task_id") != TASK
    ):
        raise ValueError("Separate FINAL16 ledger required")
    if len(rows) != 16 or {r["case_id"] for r in rows} != set(by_id):
        raise ValueError(
            (
                "All16 cases, including failed/pending, must remain in the "
                "denominator"
            )
        )
    raw_ids, failures, completed, stopped = set(), 0, 0, 0
    for row in rows:
        case = by_id[row["case_id"]]
        protected("case:" + case["case_id"])
        attempts = row.get("attempts", [])
        if (
            row.get("status")
            not in {
                "pending",
                "running",
                "completed",
                "failed",
                "stopped",
                "unavailable",
            }
            or len(attempts) > 1
        ):
            raise ValueError(
                "FINAL16 permits only one attempt per case and no retries"
            )
        for a in attempts:
            if (
                type(a.get("index")) is not int
                or a.get("index") != 1
                or a.get("parent_attempt_id") is not None
                or a.get("status")
                not in {"running", "completed", "failed", "stopped"}
                or not a.get("run_id")
                or a["run_id"] in raw_ids
                or a.get("seed") != case["seed"]
                or not set(case["lineage"]) <= set(a.get("lineage", []))
                or a.get("quality_rejection") is not None
            ):
                raise ValueError(
                    (
                        "FINAL16 raw/index/seed/group/parent identity changed "
                        "or retry attempted"
                    )
                )
            if a["status"] in {"failed", "stopped"} and not a.get(
                "failure_reason"
            ):
                raise ValueError(
                    "Physical failure/stopped reason must be preserved"
                )
            raw_ids.add(a["run_id"])
            protected("run:" + a["run_id"])
            failures += a["status"] == "failed"
            stopped += a["status"] == "stopped"
            completed += a["status"] == "completed"
        if row["status"] in {"completed", "unavailable"} and (
            not attempts or attempts[0]["status"] != "completed"
        ):
            raise ValueError(
                "Completed case must retain its sole completed physical raw"
            )
        if row["status"] in {"failed", "stopped"} and (
            not attempts or attempts[0]["status"] != row["status"]
        ):
            raise ValueError("Failed/stopped physical attempt cannot disappear")
    counts = dict(
        planned_cases=16,
        physical_attempts=len(raw_ids),
        physical_failed_attempts=failures,
        failed_attempts=failures,
        physical_completed_attempts=completed,
        stopped_attempts=stopped,
        failed_cases=sum(r["status"] == "failed" for r in rows),
        pending_cases=sum(r["status"] == "pending" for r in rows),
        unavailable_cases=sum(r["status"] == "unavailable" for r in rows),
    )
    ctx.checked.append(
        "exact16_expected_membership_one_attempt_and_full_failure_denominator"
    )
    return by_id, {r["case_id"]: r for r in rows}, counts


def prerequisites(ctx):
    fixture = ctx.request.get("fixture_only") is True
    auth = authorization(fixture_only=fixture)
    expected, policy, registry = consumer_inputs(
        ctx, scope_task=TASK, cohort=FINAL16, usage=USAGE
    )
    if (
        policy.get("approved_for_final16") is not True
        or policy.get("authorization_sha256") != AUTH_SHA256
        or policy.get("expected_cases_sha256") != EXPECTED_SHA256
        or policy.get("fixture_only") is not fixture
        or expected["source_manifest_sha256"]
        != auth["producer"]["source_sha256"]
        or expected["runtime_sha256"] != auth["producer"]["runtime_sha256"]
        or any(
            k in ctx.request
            for k in ("external_audit", "external_audits", "quality_rejection")
        )
    ):
        raise ValueError(
            "FINAL16 authority/generation/independent-data policy differs"
        )
    ctx.hashes.update(
        final16_authorization=AUTH_SHA256,
        final16_expected_cases=EXPECTED_SHA256,
    )
    cases, ledger, counts = plan_and_ledger(ctx, registry)
    ctx.ledger_counts = counts
    return expected, policy, registry, cases, ledger, counts


def final_checks(ctx, cases, ledger):
    refs = ctx.request.get("case_receipts")
    if refs is None:
        raise Pending("All16 actual case receipts are required")
    if len(refs) != 16:
        raise ValueError(
            "Exactly16 actual case receipts required; no failed case removal"
        )
    seen, pairs = set(), {}
    for i, ref in enumerate(refs):
        r = ctx.doc("case_receipt:" + str(i), ref)
        cid = r["case_id"]
        if cid not in cases or cid in seen:
            raise ValueError("All16 unique FINAL16 case identities required")
        seen.add(cid)
        entry = ledger[cid]
        if (
            r.get("schema") != "drone-ai.final16.check-receipt/1"
            or r.get("task_id") != TASK
            or r.get("purpose") != FINAL16
            or r.get("phase") != "case"
            or r.get("status") != "FINAL16_CASE_INPUTS_CHECKED_NO_FIT"
            or r.get("actual_data_verified") is not True
            or r.get("fixture_only") is not False
            or r.get("exit_code") != 0
            or r.get("duration_profile") != "short180"
            or r.get("mission_duration_us") != 180000000
            or r.get("frames", 0) <= 100
            or r.get("online", {}).get("windows", 0) <= 0
            or r.get("pair_verified") is not True
            or any(
                r.get(k) != 0
                for k in (
                    "model_forward_steps",
                    "optimizer_steps",
                    "scaler_threshold_fits",
                    "GPU_jobs",
                )
            )
            or entry["status"] != "completed"
            or len(entry["attempts"]) != 1
            or r["run_id"] != entry["attempts"][0]["run_id"]
        ):
            raise ValueError(
                (
                    "All16 nonempty actual no-fit cases and sole successful "
                    "raws required"
                )
            )
        for key in (
            "source_manifest",
            "runtime_manifest",
            "paired_policy",
            "capsule",
            "final16_authorization",
            "final16_expected_cases",
        ):
            if r["evidence_sha256"].get(key) != ctx.hashes[key]:
                raise ValueError("FINAL16 case scope/producer/config changed")
        pairs.setdefault(cases[cid]["pair_id"], []).append(
            r["evidence_sha256"]["pilot_pair_native_evidence"]
        )
    if (
        seen != set(cases)
        or len(pairs) != 8
        or any(len(v) != 2 or v[0] != v[1] for v in pairs.values())
        or len({v[0] for v in pairs.values()}) != 8
    ):
        raise ValueError(
            "Eight distinct official native reports must bind all16 cases"
        )
    return dict(
        status="FINAL16_INPUT_AND_PAIR_CHECKS_PASSED",
        cases_checked=16,
        pairs_checked=8,
        scope=(
            "AI_input_only_Data_quality_and_actual_termination_still_required"
        ),
    )


def main(request_path, output=None):
    ctx = None
    try:
        ctx = Evidence(request_path)
        if (
            ctx.request.get("schema") != "drone-ai.final16.check/1"
            or ctx.request.get("task_id") != TASK
            or ctx.request.get("purpose") != FINAL16
        ):
            raise ValueError("Explicit FINAL16 request required")
        expected, policy, registry, cases, ledger, _ = prerequisites(ctx)
        phase = ctx.request.get("phase")
        if phase == "prerequisites":
            result = dict(status="FINAL16_STATIC_INPUTS_CHECKED_NO_FLIGHTS")
        elif phase == "case":
            from .pilot_views import check_case

            cid = ctx.request["case_id"]
            scenario = ctx.doc("scenario")
            operation = json.dumps(
                scenario.get("operation_variation"),
                sort_keys=True,
                separators=(",", ":"),
            )
            if (
                hashlib.sha256(operation.encode()).hexdigest()
                != cases[cid]["operation_sha256"]
            ):
                raise ValueError(
                    (
                        "Actual operation differs from the coordinator's "
                        "frozen definition"
                    )
                )
            result = check_case(
                ctx,
                expected,
                policy,
                registry,
                cases[cid],
                ledger[cid],
                cases,
                ledger,
                cohort=FINAL16,
                usage=USAGE,
            )
            mission = ctx.doc("online_manifest")["run"]["mission_interval"]
            if (
                mission.get("duration_profile") != "short180"
                or mission.get("mission_duration_us") != 180000000
            ):
                raise ValueError(
                    "FINAL16 retains the actual full180s mission interval"
                )
            result.update(
                status="FINAL16_CASE_INPUTS_CHECKED_NO_FIT",
                duration_profile="short180",
                mission_duration_us=180000000,
            )
        elif phase == "final":
            result = final_checks(ctx, cases, ledger)
        else:
            raise ValueError("phase must be prerequisites, case or final")
        fixture = ctx.request.get("fixture_only") is True
        if fixture:
            result["status"] = "SYNTHETIC_" + result["status"]
        result.update(actual_data_verified=not fixture, fixture_only=fixture)
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
        schema="drone-ai.final16.check-receipt/1",
        task_id=TASK,
        purpose=FINAL16,
        phase=ctx.request.get("phase") if ctx else None,
        expected_case_denominator=16,
        eligible_for_fit=False,
        usable_to_start_real_final16=False,
        requires_Data_quality_and_actual_termination=True,
        exit_code=code,
        checked_stages=ctx.checked if ctx else [],
        evidence_sha256=ctx.hashes if ctx else {},
        ledger_counts=getattr(ctx, "ledger_counts", {}) if ctx else {},
        model_forward_steps=0,
        optimizer_steps=0,
        scaler_threshold_fits=0,
        GPU_jobs=0,
    )
    if output is not None:
        _write(output, result)
    print(json.dumps(result))
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    raise SystemExit(main(args.request, args.output))
