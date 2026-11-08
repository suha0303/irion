"""No-fit canonical/NPZ comparison for one completed pilot case."""

import json
import zipfile

import numpy as np

from capstone_ai.mapping import FEATURES, TARGETS
from execution.prepare import npz_headers
from learning.admission19 import MISSION_POLICY, PILOT
from learning.run_health19.data import iter_manifests, physical_features
from learning.wind_metadata import describe

from .data import inspect_run
from .duration_profiles import NATIVE_ADAPTERS
from .pilot import (
    Pending,
    Unavailable,
    bind,
    comparison_evidence,
    origin_member,
    pair_metrics,
    required,
)

ATOL = RTOL = (
    1e-6  # Fixed before tests; physical/transformed FP32 export comparison.
)


def rows(path, key):
    headers = npz_headers(path)
    dtype, shape = np.dtype(headers[key]["descr"]), headers[key]["shape"]
    if dtype.kind != "f":
        raise ValueError("Floating input arrays required")
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        reader(stream)
        width = int(np.prod(shape[1:]))
        for start in range(0, shape[0], 64):
            count = min(64, shape[0] - start)
            body = stream.read(count * width * dtype.itemsize)
            if len(body) != count * width * dtype.itemsize:
                raise ValueError(
                    "Truncated input while comparing canonical values"
                )
            yield (
                start,
                np.frombuffer(body, dtype=dtype).reshape(count, *shape[1:]),
            )
        if stream.read(1):
            raise ValueError("Trailing input bytes")


def check_case(
    ctx,
    expected,
    policy,
    registry,
    case,
    ledger,
    all_cases,
    all_ledger,
    *,
    cohort=PILOT,
    usage="pilot_input_check",
):
    if (
        ledger["status"] not in {"completed", "unavailable"}
        or not ledger["attempts"]
    ):
        raise Pending("The actual case has no completed physical raw")
    current = ledger["attempts"][-1]["run_id"]
    projected = origin_member(
        ctx, ctx.request.get("case_origin", {}), policy, expected, "case"
    )
    if projected["run_id"] != current:
        raise ValueError(
            "Actual receipt differs from the ledger's completed run"
        )
    scenario = ctx.doc("scenario")
    integrity = ctx.doc("raw_integrity")
    battery = ctx.doc("battery_selection")
    propulsion = ctx.doc("propulsion_selection")
    raw_members = [
        ("case:run", "run.json"),
        ("case:execution", "evaluation/execution.json"),
        ("scenario", "evaluation/scenario.json"),
        ("battery_selection", "evaluation/battery-selection.json"),
        ("propulsion_selection", "evaluation/propulsion-selection.json"),
    ]
    if policy["alignment_mode"] == "versioned_joint_phase":
        raw_members.append(("case:origin", policy["mission_source"]))
    for alias, member in raw_members:
        if (
            integrity[member]["sha256"] != ctx.hashes[alias]
            or integrity[member]["size"] != ctx.paths[alias].stat().st_size
        ):
            raise ValueError(
                "Actual raw metadata differs from integrity: " + member
            )
    wind = describe(scenario, wind_clock=policy["wind_model_clock"])
    if (
        wind["environment"] != case["environment"]
        or wind["path_kind"] != case["motion"]
        or scenario.get("seed") != case["seed"]
    ):
        raise ValueError(
            "Actual operation differs from frozen case classification/seed"
        )
    if scenario.get("producer_lock_sha256") != expected["runtime_sha256"]:
        raise ValueError("Actual raw scenario uses another producer")
    if scenario.get("health_eligibility") != (
        "abnormal" if case["role"] == "anomaly_test" else "confirmed_healthy"
    ):
        raise ValueError(
            "Actual source health eligibility differs from the frozen pilot role"
        )
    if case["case_kind"] == "anomaly":
        battery_fault = "rpt_16" in (
            battery.get("initial_stage"),
            battery.get("target_stage"),
        )
        propulsion_fault = "abnormal" in (
            propulsion.get("initial_state"),
            propulsion.get("target_state"),
        )
        family = (
            "combined"
            if battery_fault and propulsion_fault
            else "battery"
            if battery_fault
            else "propulsion"
            if propulsion_fault
            else None
        )
        if (
            family != case["health_family"]
            or scenario["name"].rsplit("_", 1)[-1] != case["timing"]
        ):
            raise ValueError(
                "Actual selected systems/timing differ from frozen anomaly case"
            )
    if ledger["status"] == "unavailable":
        reasons = ledger.get("unavailable_reasons")
        if not reasons:
            raise ValueError(
                "Unavailable case needs explicit preserved reasons"
            )
        raise Unavailable(
            "Case retained with unavailable reasons: " + str(reasons)
        )
    online = ctx.doc("online_manifest")
    run_manifest = ctx.doc("run_manifest")
    bind(online, expected)
    bind(run_manifest, expected)
    if len(run_manifest["runs"]) != 1:
        raise ValueError(
            "One-shot checker accepts exactly one physical run, not a campaign"
        )
    row, other = online["run"], run_manifest["runs"][0]
    for item in (row, other):
        if (
            item.get("run_id") != current
            or item.get("case_id") != case["case_id"]
            or item.get("purpose") != cohort
            or item.get("cohort") != cohort
            or item.get("role") != case["role"]
            or not set(case["lineage"]) <= set(item["lineage"])
        ):
            raise ValueError(
                "Views must retain pilot purpose/cohort and frozen case/attempt identities"
            )
        if item.get("origin") != (
            "synthetic_unit_fixture"
            if ctx.request.get("fixture_only") is True
            else "px4_sitl"
        ):
            raise ValueError(
                "Synthetic view cannot bypass actual pilot admission"
            )
        evidence = item["source_evidence"]
        if (
            evidence["scenario_name"] != scenario["name"]
            or evidence["battery_selection"] != battery
            or evidence["propulsion_selection"] != propulsion
        ):
            raise ValueError(
                "View selection/scenario summary differs from actual raw metadata"
            )
    for key in ("lineage", "mission_interval", "source_evidence", "origin"):
        if row[key] != other[key]:
            raise ValueError(
                "The two views have different source/clock/mission/lineage"
            )
    mission = row["mission_interval"]
    if (
        mission["policy"] != MISSION_POLICY
        or mission["clock_id"] != projected["clock_id"]
        or mission["start_us"] != projected["actual_mission_start_us"]
        or mission["end_us_exclusive"] != projected["actual_landing_start_us"]
        or mission["source_sha256"]
        != ctx.hashes[
            "case:origin"
            if policy["alignment_mode"] == "versioned_joint_phase"
            else "case:execution"
        ]
        or mission["source"]
        != (
            policy["mission_source"]
            if policy["alignment_mode"] == "versioned_joint_phase"
            else "evaluation/execution.json"
        )
    ):
        raise ValueError(
            "Input mission boundary differs from actual receipt/execution"
        )
    if (
        policy.get("native_adapter") in NATIVE_ADAPTERS
        and mission.get("native_sources_sha256")
        != projected["native_sources_sha256"]
    ):
        raise ValueError(
            "Mission boundary must bind actual native start, LAND and physics receipts"
        )
    if projected.get("external_audit_evidence") != mission.get(
        "external_audit_evidence"
    ):
        raise ValueError(
            "Both views must preserve the explicit external full-audit provenance"
        )
    ctx.checked.append("actual_case_receipt_and_both_view_identity")
    canonical = ctx.ref("canonical_frames")
    frames, scope, count = {}, set(), 0
    with canonical.open(encoding="utf-8") as stream:
        for line in iter(lambda: stream.readline(1 << 20), ""):
            if not line.strip():
                continue
            if not line.endswith("\n"):
                raise ValueError("Partial or oversized canonical row")
            count += 1
            if count > 10000:
                raise ValueError(
                    "Canonical case exceeds finite10000-frame check limit"
                )
            frame = json.loads(line)
            if frame["run_id"] != current:
                raise ValueError("Canonical run identity mismatch")
            if (
                not mission["start_us"]
                <= frame["time_us"]
                < mission["end_us_exclusive"]
            ):
                continue
            if (
                frame.get("feature_order") != FEATURES
                or frame.get("feature_profile") != "common19_ctx0"
                or frame["clock_id"] != mission["clock_id"]
            ):
                raise ValueError("Canonical19 profile/clock mismatch")
            if frame.get("operating_state") != "airborne" or any(
                frame["quality"][n] != "valid" for n in FEATURES
            ):
                raise Unavailable(
                    "Mission contains missing/nonvalid features or non-airborne state; retained in denominator"
                )
            lineage = required(frame, "lineage")
            if not lineage:
                raise ValueError(
                    "Canonical sample/available provenance required"
                )
            if any(
                not 0
                <= r["sample_time_us"]
                <= r["available_time_us"]
                <= frame["time_us"]
                for r in lineage.values()
            ):
                raise ValueError(
                    "Canonical sample/available provenance uses a future observation"
                )
            key = (frame["segment_id"], frame["time_us"])
            if key in frames:
                raise ValueError("Duplicate canonical mission frame")
            vector = np.array(
                [frame["values"][n] for n in FEATURES], dtype=np.float64
            )
            if not np.isfinite(vector).all():
                raise Unavailable("Missing mission value cannot be zero-filled")
            frames[key] = vector
            scope.add(frame["segment_id"])
    if len(scope) != 1:
        raise Unavailable("Mission gap/reset/segment change cannot be joined")
    segment = next(iter(scope))
    first = ((mission["start_us"] + 99999) // 100000) * 100000
    last = ((mission["end_us_exclusive"] - 1) // 100000) * 100000
    if not 1 <= (last - first) // 100000 + 1 <= 10000:
        raise ValueError("Declared mission exceeds one-shot finite frame limit")
    times = list(range(first, last + 1, 100000))
    if len(times) <= 100 or set(frames) != {(segment, t) for t in times}:
        raise Unavailable(
            "Incomplete/short actual mission; no edge trimming or gap interpolation"
        )
    physical = np.stack([frames[(segment, t)] for t in times])
    actual_runs = list(
        iter_manifests(
            [ctx.paths["run_manifest"]],
            expected_producer=expected,
            protection=registry,
            usage=usage,
        )
    )
    if not np.allclose(actual_runs[0].values, physical, atol=ATOL, rtol=RTOL):
        raise ValueError(
            "Run NPZ must preserve physical pack V/A, not transformed or shifted data"
        )
    del actual_runs
    inspected = inspect_run(
        ctx.paths["online_manifest"],
        expected_producer=expected,
        protection=registry,
        usage=usage,
    )
    npz = ctx.paths["online_manifest"].parent / online["file"]["path"]
    with np.load(npz, allow_pickle=False) as archive:
        origins = archive["origin_us"].copy()
        segments = archive["segment_id"].copy()
    first_origin = ((first + 9900000 + 999999) // 1000000) * 1000000
    expected_origins = list(range(first_origin, last - 1000000 + 1, 1000000))
    if (
        not expected_origins
        or origins.tolist() != expected_origins
        or np.any(segments != segment)
    ):
        raise ValueError(
            "Online windows must cover every eligible mission origin; no selective omission"
        )
    target_columns = [FEATURES.index(n) for n in TARGETS]
    for name, offsets in (("X", range(-99, 1)), ("Y", range(1, 11))):
        for start, values in rows(npz, name):
            expected_values = []
            for index in range(start, start + len(values)):
                vectors = np.stack(
                    [
                        frames[
                            (
                                int(segments[index]),
                                int(origins[index]) + j * 100000,
                            )
                        ]
                        for j in offsets
                    ]
                )
                transformed = physical_features(vectors)
                expected_values.append(
                    transformed
                    if name == "X"
                    else transformed[:, target_columns]
                )
            if not np.allclose(
                values, np.stack(expected_values), atol=ATOL, rtol=RTOL
            ):
                raise ValueError(
                    "Online X/Y differs from causal canonical values or was transformed twice"
                )
    ctx.checked.append(
        "both19_views_match_full_mission_canonical_once_only_units"
    )
    if case["case_kind"] != "normal_base":
        pair = ctx.doc("pair_receipt")
        bind(pair, expected)
        if (
            pair.get("schema") != "drone-ai.pilot-pair-check/1"
            or pair.get("pair_id") != case["pair_id"]
            or pair.get("paired_policy_sha256") != ctx.hashes["paired_policy"]
            or current
            not in {pair.get("normal_run_id"), pair.get("anomaly_run_id")}
        ):
            raise ValueError(
                "Actual pilot pair receipt identity/policy mismatch"
            )
        partners = [
            c
            for c in all_cases.values()
            if c["case_kind"] != "normal_base"
            and c["pair_id"] == case["pair_id"]
        ]
        if len(partners) != 2:
            raise ValueError(
                "Pilot pair must retain two distinct frozen physical cases"
            )
        for partner in partners:
            entry = all_ledger[partner["case_id"]]
            if entry["status"] != "completed" or not entry["attempts"]:
                raise Pending(
                    "Pilot pair counterpart has not completed; reuse existing raw, do not regenerate"
                )
            role = (
                "normal"
                if partner["case_kind"] == "paired_normal"
                else "anomaly"
            )
            if pair.get(role + "_run_id") != entry["attempts"][-1]["run_id"]:
                raise ValueError(
                    "Pilot pair receipt differs from the complete case ledger"
                )
        role = "normal" if case["case_kind"] == "paired_normal" else "anomaly"
        if (
            pair.get(role + "_origin_receipt_sha256")
            != ctx.hashes["case:origin"]
        ):
            raise ValueError(
                "Pilot pair receipt is not bound to this actual source receipt"
            )
        pair_metrics(pair, policy["limits"])
        comparison_evidence(ctx, pair, policy, "pilot_pair_native_evidence")
    return dict(
        status="PILOT_CASE_INPUTS_CHECKED_NO_FIT",
        run_id=current,
        case_id=case["case_id"],
        pair_id=case["pair_id"],
        pair_verified=case["case_kind"] != "normal_base",
        external_audit_evidence=projected.get("external_audit_evidence"),
        frames=len(times),
        initial100=100,
        post_initial_frames=len(times) - 100,
        tail_frames=len(times) % 100,
        online=inspected,
        cross_view_tolerance=dict(atol=ATOL, rtol=RTOL),
        no_time_shift_or_interpolation=True,
    )
