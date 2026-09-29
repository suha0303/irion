"""Read-only post-run Collection assessment. Truth stays exclusively in evaluation."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import collection_config
from .recording import TOPICS
from .runtime_manifest import PLUGIN, PX4_BINARY, RUNTIME_MANIFEST, TRANSPORT

CRITICAL = set(TOPICS) - {"trajectory_setpoint", "vehicle_status", "vehicle_land_detected"}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def stream_contract(run, start, end):
    """Constant event memory; check every published watermark against future records."""
    errors = set()
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: dict(count=0, first=None, last=None, max_gap_us=0, invalid=0))
    count = 0
    published = -1
    last_mark = None
    with (run / "observations/watermarks.jsonl").open() as marks:
        point = json.loads(line) if (line := marks.readline()) else None

        def advance_marks():
            nonlocal point, published, last_mark
            while point is not None and point["recorded_events"] <= count:
                if point["recorded_events"] != count or point["through_available_time_us"] < published:
                    errors.add("watermark order")
                if last_mark and last_mark["closed"]:
                    errors.add("watermark after closed")
                published = max(published, point["through_available_time_us"])
                last_mark = point
                point = json.loads(line) if (line := marks.readline()) else None

        with (run / "observations/streams.jsonl").open() as stream:
            for line in stream:
                advance_marks()
                e = json.loads(line)
                if set(e) != {"schema", "run_id", "source", "topic", "instance", "seq",
                               "clock_id", "sample_time_us", "available_time_us", "fields",
                               "field_quality"}:
                    errors.add("event envelope/leakage")
                topic = e["topic"]
                if e["schema"] != "drone.raw.event/2.0" or e["run_id"] != run.name:
                    errors.add("identity/schema")
                if e["clock_id"] != "px4-boot" or any(
                    type(e[key]) is not int or e[key] < 0
                    for key in ("sample_time_us", "available_time_us", "seq", "instance")
                ):
                    errors.add("clock/integer timestamp")
                if e["available_time_us"] <= published or e["sample_time_us"] > e["available_time_us"]:
                    errors.add("causal order")
                if topic not in TOPICS:
                    errors.add("undeclared topic")
                    count += 1
                    continue
                s = stats[topic]
                if e["seq"] != s["count"] or e["source"] != "px4.uorb" or e["instance"] != 0:
                    errors.add("source/sequence")
                if set(e["fields"]) != set(TOPICS[topic][1]) or set(e["field_quality"]) != set(e["fields"]):
                    errors.add("fields/leakage")
                for key, value in e["fields"].items():
                    size = {"output": 4, "xyz": 3, "q": 4, "velocity": 3}.get(key)
                    if size and (not isinstance(value, list) or len(value) != size):
                        errors.add("array shape")
                    values = value if isinstance(value, list) else [value]
                    quality = e["field_quality"].get(key)
                    if quality not in ("valid", "unavailable", "invalid", "stale", "clipped"):
                        errors.add("quality enum")
                    if any(v is not None and not math.isfinite(v) for v in values):
                        errors.add("nonfinite observation")
                    if any(v is None for v in values) and quality == "valid":
                        errors.add("null marked valid")
                    if start <= e["sample_time_us"] <= end and quality != "valid":
                        s["invalid"] += 1
                t = e["sample_time_us"]
                if s["last"] is not None:
                    if t < s["last"]:
                        errors.add("sample time reversal")
                    if t >= start and s["last"] <= end:
                        s["max_gap_us"] = max(s["max_gap_us"], t-s["last"])
                s["first"] = t if s["first"] is None else s["first"]
                s["last"] = t
                s["count"] += 1
                count += 1
        advance_marks()
        if point is not None or not last_mark or not last_mark["closed"] or last_mark["recorded_events"] != count:
            errors.add("final watermark")
    if set(stats) != set(TOPICS):
        errors.add("missing topics")
    manifest = read(run / "integrity.json")
    required = {"run.json", "observations/streams.jsonl", "observations/metadata.json",
                "observations/watermarks.jsonl", "observations/diagnostics.jsonl",
                "evaluation/truth.jsonl", "evaluation/scenario.json", "evaluation/execution.json",
                "reference/source_lock.json", "reference/fc-configuration.json"}
    if not required <= set(manifest):
        errors.add("incomplete integrity manifest")
    for name, entry in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            errors.add("unsafe integrity path")
            continue
        p = run / relative
        if not p.is_file() or p.stat().st_size != entry["size"] or file_hash(p) != entry["sha256"]:
            errors.add("hash " + name)
    missing_ok = end > start and all(
        t in stats and stats[t]["first"] <= start and stats[t]["last"] >= end
        and stats[t]["max_gap_us"] <= 500000 and stats[t]["invalid"] == 0 for t in CRITICAL
    )
    return sorted(errors), dict(stats), missing_ok


def assess(run, instance, duration, profile_id):
    if not math.isfinite(duration) or duration <= 0 or not 0 <= instance <= 15:
        raise ValueError("invalid assessment duration/instance")
    run = Path(run)
    state = read(run / "run.json")
    scenario = read(run / "evaluation/scenario.json")
    timeline = read(run / "evaluation/execution.json")
    start = next((x["sim_time"] for x in timeline if x["state"] == "airborne"), None)
    end = next((x["sim_time"] for x in timeline if x["state"] == "landing"), None)
    elapsed = min(duration, max(0, end-start)) if start is not None and end is not None else 0
    start_us = int(start*1e6) if start is not None else 0
    end_us = start_us + int(elapsed*1e6)
    errors, stats, missing = stream_contract(run, start_us, end_us)
    if state.get("demo_only") or state.get("dataset_admission") == "excluded_gui_preview":
        errors.append("GUI preview is excluded from collection datasets")
    metadata = read(run / "observations/metadata.json")
    source = read(run / "reference/source_lock.json")
    legacy = state.get("configuration") == collection_config.LEGACY_NAME
    profile_path = collection_config.LEGACY_PROFILE_PATH if legacy else "profiles/collection/vehicle.yaml"
    expected_id = collection_config.LEGACY_BATTERY_ID if legacy else collection_config.battery()["profile_id"]
    if profile_id != expected_id or metadata["vehicle"].get("battery_profile_id") != expected_id:
        errors.append("profile identity")
    profile_digest = (
        collection_config.LEGACY_PROFILE_SHA256
        if legacy else file_hash(collection_config.ROOT / profile_path)
    )
    if source["local_sources"].get(profile_path) != profile_digest:
        errors.append("profile source hash")
    resolved_profile = run / "reference/resolved_vehicle.yaml"
    if resolved_profile.is_file() and file_hash(resolved_profile) != profile_digest:
        errors.append("resolved profile source hash")
    lock_name = scenario.get("producer_lock") or (
        "artifacts/round001-r3/producer-lock.json" if legacy else RUNTIME_MANIFEST
    )
    if lock_name not in {RUNTIME_MANIFEST, "artifacts/round001-r3/producer-lock.json",
                         "artifacts/round001-r3/producer-lock-diverse-v1.json",
                         "artifacts/round001-r3/producer-lock-cadence-v2.json",
                         "artifacts/round001-r3/producer-lock-speed3-v3.json",
                         "artifacts/round001-r3/producer-lock-balanced300-v4.json",
                         "artifacts/round001-r3/producer-lock-startup-v5.json",
                         "artifacts/round001-r3/producer-lock-metadata-v6.json",
                         "artifacts/round001-r3/producer-lock-layout-v7.json"}:
        raise ValueError("Unknown producer revision")
    lock_path = collection_config.ROOT / lock_name
    lock_digest = scenario.get("producer_lock_sha256")
    if lock_digest and (not lock_path.is_file() or file_hash(lock_path) != lock_digest):
        # Machine seals are archived by digest when explicitly rebuilt. This is
        # evidence lookup only; an old descriptor can never select today's runtime.
        archived = collection_config.ROOT / "artifacts/runtime-history" / (str(lock_digest) + ".json")
        if (len(str(lock_digest)) == 64 and all(c in "0123456789abcdef" for c in str(lock_digest))
                and archived.is_file() and file_hash(archived) == lock_digest):
            lock_path = archived
        else:
            errors.append("producer lock changed")
    producer = read(lock_path)
    for relative, expected in producer["files"].items():
        if relative.startswith(("sim/", "profiles/", "contracts/")):
            if source["local_sources"].get(relative) != expected:
                errors.append("frozen source " + relative)
        elif relative in (PX4_BINARY, PLUGIN, TRANSPORT,
                           "artifacts/round001-r3/build/libX500Powertrain.so",
                           "ai/runtime/gz-transport13-13.6.0-envcache-v1/lib/libgz-transport13.so.13.6.0"):
            if source["runtime_binary_sha256"].get(relative) != expected:
                errors.append("frozen binary " + relative)
    if metadata["vehicle"]["nominal_capacity_ah"] != 12 or metadata["vehicle"].get("nominal_parallel_cells") != 4:
        errors.append("nominal battery metadata")
    fc = read(run / "reference/fc-configuration.json")
    current_fc = collection_config.fc_parameters()
    if "fc_parameters" in producer:
        expected_fc = producer["fc_parameters"]
        policy_valid = (isinstance(expected_fc, dict)
                        and set(expected_fc) == set(current_fc)
                        and all(type(value) in (int, float) for value in expected_fc.values())
                        and collection_config.fc_verified(expected_fc))
    else:
        # Historical producer seals predate the touchdown setting. The hash-
        # checked seal, not the run's self-reported expected keys, selects this.
        expected_fc = {key: value for key, value in current_fc.items() if key != "MPC_LAND_SPEED"}
        policy_valid = True
    if not policy_valid or not fc.get("verified") or not collection_config.fc_verified(
        fc.get("observed", {}), expected=expected_fc
    ):
        errors.append("actual FC parameter gate")
    if scenario["requested_duration_s"] != duration or state["status"] != "completed":
        errors.append("requested duration or completion")
    if state.get("origin") != "px4_sitl":
        errors.append("not a PX4 SITL run")
    isolation = (state.get("instance") == instance and state.get("gz_partition") == "x500_"+run.name
                 and state.get("mavlink_source_checked") is True
                 and state.get("configuration") in (collection_config.NAME, collection_config.LEGACY_NAME))
    evidence: dict[str, Any] = dict(samples=0, min_voltage_v=None, min_soc=None, max_duty=None,
                    min_thrust_weight_ratio=None, max_thrust_weight_ratio=None, outside_samples=0)
    mass = collection_config.vehicle()["geometry"]["body_mass_kg"] + 4*collection_config.vehicle()["geometry"]["rotor_mass_kg_each"]
    mass += scenario["parameters"].get("payload", 0)
    finite = True
    with (run / "evaluation/truth.jsonl").open() as stream:
        for line in stream:
            e = json.loads(line)
            if not start_us <= e["time_us"] <= end_us:
                continue
            values = [e["voltage_v"], e["soc"], *e["command"], *e["thrust_n"], e["solver_residual_v"]]
            if not all(math.isfinite(v) for v in values) or not 0 <= e["soc"] <= 1:
                finite = False
                continue
            if abs(e["solver_residual_v"]) > 1e-6:
                finite = False
            evidence["samples"] += 1
            evidence["outside_samples"] += not e["model_valid"]
            ratio = sum(e["thrust_n"])/(mass*9.80665)
            for key, value, op in [("min_voltage_v", e["voltage_v"], min), ("min_soc", e["soc"], min),
                                   ("max_duty", max(e["command"]), max),
                                   ("min_thrust_weight_ratio", ratio, min),
                                   ("max_thrust_weight_ratio", ratio, max)]:
                evidence[key] = value if evidence[key] is None else op(evidence[key], value)
    anomaly = scenario["name"] in collection_config.ANOMALY
    if scenario["name"] not in collection_config.NORMAL | collection_config.ANOMALY:
        errors.append("unsupported Collection scenario")
    expected_revision = collection_config.LEGACY_REVISION if legacy else collection_config.vehicle()["revision"]
    if scenario.get("normal_range_revision") != expected_revision:
        errors.append("health policy revision")
    if not anomaly and scenario.get("health_eligibility") != "confirmed_healthy":
        errors.append("normal health eligibility")
    normal_gate = bool(evidence["samples"] and finite and evidence["outside_samples"] == 0
                       and evidence["min_voltage_v"] >= 12.4 and evidence["min_soc"] >= .2
                       and evidence["max_duty"] <= .9)
    physical = bool(evidence["samples"] and finite and (anomaly or normal_gate))
    return dict(schema="drone.run-assessment/1", run_id=run.name, instance=instance,
                gz_partition="x500_"+run.name, profile_id=expected_id,
                contract_passed=not errors, instance_isolation_passed=isolation,
                missing_data_passed=missing, physical_gate_passed=physical,
                observed_valid_seconds=elapsed if missing and not errors else 0.0,
                errors=errors, topic_quality=stats, physical_evidence=evidence,
                normal_endurance_gate_passed=normal_gate,
                physical_gate_scope="anomaly finite model/solver validity; normal endurance not an exclusion"
                if anomaly else "normal usable voltage/SOC/model envelope and duty <=0.9",
                thrust_note="actual thrust/weight and command duty; not measured maximum thrust reserve",
                missing_data_policy="common19 source topics: full mission coverage, valid fields, max gap0.5s",
                C9="not_reexecuted", scope="SITL run assessment, not hardware or AI detection performance")
