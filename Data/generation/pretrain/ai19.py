"""Current-battery raw admission and two19-signal exports; no flight or model fit."""

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .common import (
    ROOT,
    STORE,
    Budget,
    collector,
    external,
    producer_sources,
    read,
    sha,
    verify,
    write,
)

POLICY = "cmu-vah01-rpt16-only/1"
MISSION_POLICY = "mission_only_actual_execution/1"
ROLES = {
    "normal_train",
    "normal_validation",
    "normal_calibration",
    "normal_test",
    "anomaly_test",
}
CONSUMER = {
    "run.json",
    "observations/metadata.json",
    "observations/streams.jsonl",
    "observations/watermarks.jsonl",
    "observations/diagnostics.jsonl",
    "reference/source_lock.json",
}
# Historical synthetic fixture only. Actual admission binds the producer revision.
PROFILE_REVISION = "vtc6-4s4p-cmu-vah01-v1"


def no_links(path):
    path = Path(path).expanduser().absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink/traversal input or output forbidden")
    return path


def ai_modules():
    """Import the AI-owned causal engine, never the preserved ctx6 package."""
    for p in (ROOT / "ai/src", ROOT / "ai"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from capstone_ai import mapping, source
    from learning.history19 import data, preprocessing

    for module in (mapping, source, data, preprocessing):
        if (
            not Path(module.__file__)
            .resolve()
            .is_relative_to((ROOT / "ai").resolve())
        ):
            raise ValueError(
                "Active AI19 sources required; sealed ctx6 package already imported"
            )
    return mapping, source, data, preprocessing


def producer_binding():
    files = producer_sources()
    catalog = ROOT / "simulation/profiles/battery-aging/cmu-vah01.json"
    policy = read(catalog).get("flight_policy")
    if policy != dict(
        revision=POLICY, normal_stage="rpt_00", aged_stage="rpt_16"
    ):
        raise ValueError("Current rpt_00/rpt_16 producer policy required")
    sys.path.insert(0, str(ROOT / "simulation/sim/src"))
    from capstone_sim import collection_config, propulsion_health

    return dict(
        battery_policy=POLICY,
        battery_revision=read(catalog)["revision"],
        catalog_sha256=sha(catalog),
        propulsion_revision=propulsion_health.load_catalog()["revision"],
        propulsion_catalog_sha256=sha(
            ROOT / "simulation/profiles/propulsion-health.json"
        ),
        vehicle_profile_revision=collection_config.vehicle()["revision"],
        source_hashes=files,
        runtime_sha256=sha(ROOT / "simulation/artifacts/runtime-manifest.json"),
        source_manifest_sha256=sha(
            ROOT / "simulation/config/producer-sources.json"
        ),
    )


def lineage(row):
    required = (
        "run_id",
        "raw_flight_id",
        "paired_case_id",
        "group_id",
        "retry_family_id",
    )
    if row.get("role") not in ROLES or any(
        not isinstance(row.get(k), str) or not row[k] for k in required
    ):
        raise ValueError(
            "Explicit role/raw/pair/group/retry family identities required"
        )
    previous = row.get("previous_attempts", [])
    parent = row.get("parent_attempt_id")
    if not isinstance(previous, list) or any(
        not isinstance(p, str) or not p for p in previous
    ):
        raise ValueError(
            "Explicit previous attempt identifiers must be a list of nonempty strings"
        )
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise ValueError("Invalid parent attempt identity")
    result = [
        "run:" + row["run_id"],
        "run:" + row["raw_flight_id"],
        "pair:" + row["paired_case_id"],
        "group:" + row["group_id"],
        "retry:" + row["retry_family_id"],
    ]
    result += ["run:" + p for p in row.get("previous_attempts", [])]
    if row.get("parent_attempt_id"):
        result.append("run:" + row["parent_attempt_id"])
    if row.get("raw_sha256"):
        result.append("raw-sha256:" + row["raw_sha256"])
    return result


def validate_lineage(rows):
    """Connected families cannot cross partitions; both test roles share one partition."""
    owners, seen = {}, set()
    for row in rows:
        keys = lineage(row)
        if row["run_id"] in seen:
            raise ValueError("Duplicate raw run")
        seen.add(row["run_id"])
        split = (
            "test:" + row.get("cohort", "legacy_unspecified")
            if row["role"] in {"normal_test", "anomaly_test"}
            else row["role"]
        )
        for key in keys:
            if owners.setdefault(key, split) != split:
                raise ValueError("Raw/retry/paired/group split leakage")
    return rows


def admit(selection, binding, *, policy_bundle=None, assessment_path=None):
    run = no_links(selection["path"])
    integrity = read(run / "integrity.json")
    needed = CONSUMER | {
        "evaluation/scenario.json",
        "evaluation/battery-selection.json",
        "evaluation/battery-catalog.json",
        "evaluation/execution.json",
    }
    if not needed <= set(integrity):
        raise ValueError(
            "Current raw/producer/battery integrity entries required"
        )
    for name in needed:
        p = no_links(run / name)
        if (
            p.stat().st_size != integrity[name]["size"]
            or sha(p) != integrity[name]["sha256"]
        ):
            raise ValueError("Raw integrity changed: " + name)
    meta, scenario = (
        read(run / "run.json"),
        read(run / "evaluation/scenario.json"),
    )
    if (
        meta.get("status") != "completed"
        or meta.get("origin") != "px4_sitl"
        or meta.get("configuration") != "collection"
        or meta.get("demo_only")
    ):
        raise ValueError("Completed native collection run required")
    if (
        meta["run_id"] != run.name
        or selection.get("run_id", run.name) != run.name
    ):
        raise ValueError("Run identity mismatch")
    if scenario.get("producer_lock_sha256") != binding["runtime_sha256"]:
        raise ValueError(
            "Historical producer revision cannot enter new AI19 data"
        )
    collector().verify_run_source_snapshot(
        dict(
            protocol="DRONE_COLLECTION", source_hashes=binding["source_hashes"]
        ),
        read(run / "reference/source_lock.json"),
    )
    sys.path.insert(0, str(ROOT / "simulation/sim/src"))
    from capstone_sim import battery_aging, collection_config
    from capstone_sim.run_assessment import assess

    battery = read(run / "evaluation/battery-selection.json")
    if (
        battery.get("flight_policy", {}).get("revision") != POLICY
        or battery.get("catalog_sha256") != binding["catalog_sha256"]
    ):
        raise ValueError("Battery policy/catalog revision mismatch")
    battery_aging.verify_selection(
        battery,
        run / "evaluation/battery-catalog.json",
        collection_config.battery(),
        scenario["name"],
        binding["catalog_sha256"],
    )
    if selection.get("external_audit") is not None:
        ai_modules()
        from learning.history19.native_audit import reference as audit_reference
        from learning.history19.native_audit import verified_check

        verified_check(
            run / "run.json",
            selection["external_audit"],
            generation=binding,
            policy_sha256=sha(run / "evaluation/joint-phase-policy.json"),
            duration_us=round(scenario["requested_duration_s"] * 1_000_000),
        )
        native = read(audit_reference(selection["external_audit"]["receipt"]))
        assessment = read(audit_reference(native["full_gate_receipt"]))
    else:
        assessment = assess(
            run, meta["instance"], scenario["requested_duration_s"], "vtc6_4s4p"
        )
        if (
            scenario.get("joint_phase_policy") is not None
            and all(
                assessment.get(k) is True
                for k in (
                    "contract_passed",
                    "instance_isolation_passed",
                    "missing_data_passed",
                    "physical_gate_passed",
                )
            )
            and not assessment.get("errors")
        ):
            # assess() already ran the official native CSV/reference verifier.
            # Bind that pass to this closed raw and its saved producer check so
            # the following case audit need not traverse the same CSV again.
            assessment["source_binding"] = dict(
                schema="drone-data.native-assessment-binding/1",
                raw_integrity_sha256=sha(run / "integrity.json"),
                producer_source_manifest_sha256=binding[
                    "source_manifest_sha256"
                ],
                producer_runtime_sha256=binding["runtime_sha256"],
                native_policy_sha256=sha(
                    run / "evaluation/joint-phase-policy.json"
                ),
                native_check_sha256=sha(run / "evaluation/joint-check.json"),
                native_verifier_executed=True,
            )
    if assessment_path is not None:
        write(assessment_path, assessment)
    if any(
        assessment.get(k) is not True
        for k in (
            "contract_passed",
            "instance_isolation_passed",
            "missing_data_passed",
            "physical_gate_passed",
        )
    ):
        raise ValueError(
            "Producer assessment failed: " + json.dumps(assessment)
        )
    row = dict(
        selection,
        run_id=run.name,
        raw_flight_id=meta.get("raw_flight_id", run.name),
        raw_sha256=sha(run / "observations/streams.jsonl"),
        paired_case_id=scenario["paired_case_id"],
        parent_attempt_id=meta.get("parent_attempt_id"),
        health_eligibility=scenario["health_eligibility"],
        origin=meta["origin"],
        battery_policy=POLICY,
        producer_runtime_sha256=binding["runtime_sha256"],
    )
    for k in (
        "paired_case_id",
        "raw_flight_id",
        "parent_attempt_id",
        "health_eligibility",
    ):
        if k in selection and selection[k] != row[k]:
            raise ValueError("Selection/source identity mismatch: " + k)
    if row.get("role") in ROLES - {"anomaly_test"}:
        if (
            row["health_eligibility"] != "confirmed_healthy"
            or battery["initial_stage"] != "rpt_00"
            or battery["target_stage"] != "rpt_00"
        ):
            raise ValueError(
                "Normal membership requires source-confirmed rpt_00 health"
            )
    elif (
        row.get("role") == "anomaly_test"
        and row["health_eligibility"] != "abnormal"
    ):
        raise ValueError("Anomaly test requires source anomaly metadata")
    lineage(row)
    return row, dict(
        integrity_sha256=sha(run / "integrity.json"),
        assessment=assessment,
        battery_selection=battery,
        scenario_sha256=sha(run / "evaluation/scenario.json"),
        mission_interval=mission_interval(
            run,
            policy_bundle=policy_bundle,
            **(
                dict(external_audit=selection["external_audit"])
                if selection.get("external_audit") is not None
                else {}
            ),
        ),
    )


def mission_interval(run, *, policy_bundle=None, external_audit=None):
    """Use actual operational timestamps, never feature quality or fault truth."""
    if external_audit is not None and policy_bundle is None:
        raise ValueError(
            "External native evidence requires its explicit generation/policy bundle"
        )
    if policy_bundle is not None:
        ai_modules()
        from learning.history19.pilot_connector import native_mission_interval

        relative = Path(policy_bundle["origin_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Reviewed native origin member path required")
        origin = no_links(Path(run) / relative)
        expected = read(Path(run) / "integrity.json")[str(relative)]
        if (
            origin.stat().st_size != expected["size"]
            or sha(origin) != expected["sha256"]
        ):
            raise ValueError(
                "Native mission receipt differs from raw integrity"
            )
        return native_mission_interval(
            origin,
            policy_path=policy_bundle["paired_policy"],
            native_policy_path=policy_bundle["native_policy"],
            run_path=Path(run) / "run.json",
            **(
                dict(external_audit=external_audit)
                if external_audit is not None
                else {}
            ),
        )
    source = Path(run) / "evaluation/execution.json"
    records = read(source)
    bounds = []
    for state in ("mission", "landing"):
        events = [e for e in records if e.get("state") == state]
        if len(events) != 1:
            raise ValueError("One actual mission and landing boundary required")
        seconds = events[0].get("sim_time")
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError(
                "Finite nonnegative execution simulation timestamp required"
            )
        bounds.append(round(seconds * 1_000_000))
    if bounds[1] <= bounds[0]:
        raise ValueError("Landing must follow actual mission start")
    return dict(
        policy=MISSION_POLICY,
        clock_id="px4-boot",
        start_us=bounds[0],
        end_us_exclusive=bounds[1],
        source="evaluation/execution.json",
        source_sha256=sha(source),
    )


def replay(consumer, config, *, expected_profile_revision=None):
    mapping, source, _, module = ai_modules()
    metadata = read(consumer / "observations/metadata.json")
    vehicle = metadata["vehicle"]
    if expected_profile_revision is None:
        if (
            read(consumer / "run.json").get("origin")
            != "synthetic_unit_fixture"
        ):
            raise ValueError(
                "Actual replay requires a frozen expected profile revision"
            )
        expected_profile_revision = PROFILE_REVISION
    fixed = dict(
        profile_id="x500_reference",
        profile_revision=expected_profile_revision,
        rotor_count=4,
        nominal_series_cells=4,
        nominal_parallel_cells=4,
        nominal_capacity_ah=12,
        battery_profile_id="vtc6_4s4p",
    )
    if any(vehicle.get(k) != v for k, v in fixed.items()):
        raise ValueError("Current X5004S4P12Ah observation metadata required")
    validation = copy.deepcopy(metadata)
    validation["vehicle"]["battery_profile_id"] = "vtc6_4s4p_r3_long15"
    engine = module.Preprocessor(validation, config)
    engine.metadata = metadata
    tail = source.DirectoryTail(consumer, engine)
    started = time.monotonic()
    while not tail.closed:
        if time.monotonic() - started > 7200:
            raise TimeoutError("AI19 replay budget exceeded")
        if not tail.poll():
            break
    if not tail.closed:
        raise ValueError("Closed final watermark required")
    alias = dict(
        field="vehicle.battery_profile_id",
        original="vtc6_4s4p",
        validation_only="vtc6_4s4p_r3_long15",
        metadata_sha256=sha(consumer / "observations/metadata.json"),
        reason="AI shared mapper validates unchanged4S4P12Ah nameplate; current producer/catalog separately bound",
    )
    return engine, alias


def online_arrays(engine, mission=None):
    mapping, _, data, _ = ai_modules()
    if not engine.windows:
        return None, [dict(reason="no_valid_history_windows")]
    try:
        return data.arrived_samples(engine, mission)
    except ValueError as error:
        if str(error).startswith("No arrived valid target windows"):
            return None, [dict(reason="no_arrived_valid_future_targets")]
        raise


def run_array(engine, mission=None):
    """Retain every mission frame; canonical takeoff/landing evidence stays separate."""
    mapping, _, _, _ = ai_modules()
    frames = engine.frames
    if mission is None or mission.get("policy") != MISSION_POLICY:
        return None, dict(
            status="unavailable",
            reasons=["missing_mission_boundaries"],
            total_frames=len(frames),
        )
    start, end = mission["start_us"], mission["end_us_exclusive"]
    if type(start) is not int or type(end) is not int or end <= start:
        raise ValueError("Ordered integer mission boundaries required")
    indices = [i for i, f in enumerate(frames) if start <= f["time_us"] < end]
    if not indices:
        return None, dict(
            status="unavailable",
            reasons=["no_mission_frames"],
            total_frames=len(frames),
            mission_interval=mission,
            interval_policy=MISSION_POLICY,
        )
    selected = frames[indices[0] : indices[-1] + 1]
    reasons = []
    scopes = {(f["clock_id"], f["segment_id"]) for f in selected}
    if len(scopes) != 1 or any(
        clock != mission["clock_id"] for clock, _ in scopes
    ):
        reasons.append("multiple_clock_or_segment")
    times = np.asarray([f["time_us"] for f in selected], dtype=np.int64)
    expected_first = ((start + 99999) // 100000) * 100000
    expected_last = ((end - 1) // 100000) * 100000
    if times[0] != expected_first or times[-1] != expected_last:
        reasons.append("incomplete_mission_interval")
    if not np.all(np.diff(times) == 100000):
        reasons.append("gap_or_reset")
    valid = [
        all(
            f["quality"][n] == "valid" and f["values"][n] is not None
            for n in mapping.FEATURES
        )
        for f in selected
    ]
    if not all(valid):
        reasons.append("missing_or_nonvalid_feature")
    if any(f["operating_state"] != "airborne" for f in selected):
        reasons.append("unknown_or_ground_inside_airborne_interval")
    if len(selected) <= 100:
        reasons.append("no_post_initial100_points")
    report = dict(
        status="unavailable" if reasons else "available",
        reasons=reasons,
        total_frames=len(frames),
        interval_frames=len(selected),
        valid_frames=sum(valid),
        coverage=sum(valid) / len(selected),
        clocks_segments=[list(s) for s in sorted(scopes)],
        start_us=int(times[0]),
        end_us=int(times[-1]),
        history_warmup_frames=sum(f["warmup"] for f in selected),
        interval_policy=MISSION_POLICY,
        mission_interval=mission,
        excluded_airborne_before_mission=sum(
            f["operating_state"] == "airborne" and f["time_us"] < start
            for f in frames
        ),
        excluded_airborne_after_mission=sum(
            f["operating_state"] == "airborne" and f["time_us"] >= end
            for f in frames
        ),
        takeoff_landing_evidence="All original frames, quality and times remain in canonical evidence",
    )
    if reasons:
        return None, report
    values = np.asarray(
        [[f["values"][n] for n in mapping.FEATURES] for f in selected],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        report.update(status="unavailable", reasons=["nonfinite_feature"])
        return None, report
    return dict(values=values, time_us=times), report


def reference(path, base):
    return dict(path=str(path.relative_to(base)), sha256=sha(path))


def write_fit(output, records, engines):
    """Only unique valid airborne normal_train frames determine online statistics."""
    from capstone_ai.dataset import feature_transform

    mapping, _, data, _ = ai_modules()
    fit = [
        r for r in records if r["role"] in {"normal_train", "normal_validation"}
    ]
    if {r["role"] for r in fit} != {"normal_train", "normal_validation"} or any(
        not r["online"]["available"] for r in fit
    ):
        return dict(
            status="NOT_FIT_READY",
            reason="independent normal_train and normal_validation with arrived targets required",
        )
    files = {}
    for role in ("normal_train", "normal_validation"):
        values = []
        for r in fit:
            if r["role"] == role:
                with np.load(
                    output / r["online"]["file"]["path"], allow_pickle=False
                ) as arrays:
                    values.append({k: arrays[k].copy() for k in arrays.files})
        target = output / (role + ".npz")
        np.savez(
            target,
            **{k: np.concatenate([v[k] for v in values]) for k in data.ARRAYS},
        )
        files[role] = reference(target, output)
    frames = []
    for r in fit:
        if r["role"] == "normal_train":
            frames.extend(
                f
                for f in engines[r["run_id"]].frames
                if f["operating_state"] == "airborne"
                and all(f["quality"][n] == "valid" for n in mapping.FEATURES)
            )
    if not frames:
        raise ValueError("No unique valid normal_train frames for scaler")
    keys = [
        (f["run_id"], f["clock_id"], f["segment_id"], f["time_us"])
        for f in frames
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate train scaler frame")
    config = next(iter(engines.values())).config
    values = feature_transform(
        [[f["values"][n] for n in mapping.FEATURES] for f in frames], config
    )
    frame_path = output / "normal_train_frames.npz"
    np.savez(
        frame_path,
        values=values,
        **{
            k: np.asarray([f[k] for f in frames])
            for k in ("run_id", "clock_id", "segment_id", "time_us")
        },
    )
    files["normal_train_frames"] = reference(frame_path, output)
    mean, scale = values.mean(0), np.maximum(values.std(0), 1e-5)
    indices = [mapping.FEATURES.index(n) for n in mapping.TARGETS]
    scaler_path = output / "scaler.npz"
    np.savez(
        scaler_path,
        x_mean=mean,
        x_scale=scale,
        y_mean=mean[indices],
        y_scale=scale[indices],
    )
    files["scaler"] = reference(scaler_path, output)
    manifest = dict(
        schema=data.SCHEMA,
        fixture_only=all(
            r.get("origin") == "synthetic_unit_fixture" for r in fit
        ),
        feature_profile="common19_ctx0",
        feature_order=mapping.FEATURES,
        target_order=mapping.TARGETS,
        history=100,
        future=10,
        step_us=100000,
        emit_every_us=1000000,
        battery_policy=POLICY,
        scaler_scope="normal_train_unique_valid_frames",
        scaler_fit_run_ids=sorted(
            r["run_id"] for r in fit if r["role"] == "normal_train"
        ),
        runs=[
            {
                k: v
                for k, v in r.items()
                if k not in {"online", "run_health", "metadata_evidence"}
            }
            for r in fit
        ],
        files=files,
    )
    path = output / "history19-manifest.json"
    write(path, manifest)
    return dict(
        status="FIT_INPUT_EXPORTED_NOT_TRAINED",
        manifest=reference(path, output),
    )


def prepare(selection_path, output):
    mapping, _, _, _ = ai_modules()
    selection_path, output = no_links(selection_path), no_links(output)
    selection = read(selection_path)
    if selection.get(
        "schema"
    ) != "drone-data.ai19.selection/1" or not selection.get("runs"):
        raise ValueError("New explicit AI19 selection required")
    external(output)
    if output.exists():
        raise ValueError("New output required; existing results are preserved")
    for row in selection["runs"]:
        raw = no_links(row["path"])
        if output.is_relative_to(raw) or raw.is_relative_to(output):
            raise ValueError("Export cannot overlap raw source")
    binding = producer_binding()
    admitted = [admit(row, binding) for row in selection["runs"]]
    if len(admitted) > 1 and any(
        r["origin"] != "synthetic_unit_fixture" for r, _ in admitted
    ):
        raise ValueError(
            "Actual multi-run exports require the subprocess pilot45 path"
        )
    validate_lineage([r for r, evidence in admitted])
    output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "ai/learning/history19/preprocessing.json"
    config = read(config_path)
    consumer_sources = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            config_path,
            ROOT / "ai/learning/history19/preprocessing.py",
            ROOT / "ai/learning/history19/data.py",
            ROOT / "ai/src/capstone_ai/source.py",
            ROOT / "ai/src/capstone_ai/mapping.py",
            ROOT / "ai/src/capstone_ai/dataset.py",
        ]
    }
    write(
        output / "binding.json",
        dict(
            producer=binding,
            consumer_sources=consumer_sources,
            selection_sha256=sha(selection_path),
            training_started=False,
        ),
    )
    token = (
        output.name
        + "-"
        + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    )
    archive_root = external(STORE / "archives/ai19" / token)
    consumer_root = external(STORE / "consumer/ai19" / token)
    archive_root.mkdir(parents=True, exist_ok=False)
    consumer_root.mkdir(parents=True, exist_ok=False)
    c = collector()
    # The reused extractor imports the preserved contract registry, not its engine.
    from .common import package_imports

    # RAW_FILES registry imports are CPU-only. Keep the active AI engine in sys.modules.
    package_imports()
    from . import archive as archive_module

    records, engines, run_manifests = [], {}, {}
    try:
        for row, evidence in admitted:
            raw, rid = Path(row["path"]), row["run_id"]
            folder = output / "evidence" / rid
            folder.mkdir(parents=True)
            archived = c.archive_run(
                raw,
                archive_root / (rid + ".tar.gz"),
                folder,
                dict(
                    archive_wall_s=7200,
                    archive_verify_wall_s=7200,
                    byte_stall_s=600,
                ),
                lambda: None,
            )
            consumer = consumer_root / rid
            export = dict(
                path=archived["archive"],
                sha256=archived["sha256"],
                bytes=archived["bytes"],
                members=archived["members"],
            )
            archive_module.extract_consumer(
                dict(
                    archive=export,
                    path=str(consumer),
                    files={
                        n: archived["members"][n]["sha256"] for n in CONSUMER
                    },
                ),
                Budget(),
            )
            engine, alias = replay(
                consumer,
                config,
                expected_profile_revision=binding.get(
                    "vehicle_profile_revision"
                ),
            )
            engines[rid] = engine
            for name, rows in [
                ("frames", engine.frames),
                ("windows", engine.windows),
                ("exclusions", engine.exclusions),
            ]:
                with (folder / (name + ".jsonl")).open(
                    "x", encoding="utf-8"
                ) as f:
                    for item in rows:
                        f.write(json.dumps(item, allow_nan=False) + "\n")
            arrays, excluded = online_arrays(
                engine, evidence["mission_interval"]
            )
            npz = output / "online" / (rid + ".npz")
            npz.parent.mkdir(exist_ok=True)
            online = dict(
                available=arrays is not None,
                target_exclusions=excluded,
                issued_windows=len(engine.windows),
            )
            if arrays is not None:
                np.savez(npz, **arrays)
                online["file"] = reference(npz, output)
            physical, coverage = run_array(engine, evidence["mission_interval"])
            entry = dict(
                row,
                online=online,
                run_health=coverage,
                metadata_evidence=dict(
                    evidence,
                    validation_alias=alias,
                    archive=export,
                    consumer_path=str(consumer),
                ),
            )
            if physical is not None:
                path = output / "runs" / (rid + ".npz")
                path.parent.mkdir(exist_ok=True)
                np.savez(path, **physical)
                run_manifests.setdefault(row["role"], []).append(
                    dict(
                        run_id=rid,
                        role=row["role"],
                        lineage=lineage(row),
                        health=row["health_eligibility"],
                        origin=row["origin"],
                        mission_interval=evidence["mission_interval"],
                        **reference(path, output),
                    )
                )
            records.append(entry)
        manifests = {}
        for role, runs in run_manifests.items():
            value = dict(
                schema="drone-ai.run-health19.dataset/1",
                units="physical_common19_pack_V_A",
                step_us=100000,
                feature_order=mapping.FEATURES,
                vehicle_profile="x500_reference",
                battery_policy=POLICY,
                interval_policy=MISSION_POLICY,
                producer=binding,
                runs=runs,
            )
            path = output / ("run-health19-" + role + ".json")
            write(path, value)
            manifests[role] = reference(path, output)
        fit = write_fit(output, records, engines)
        verify(consumer_sources)
        verify(
            {
                str(ROOT / "simulation" / n): h
                for n, h in binding["source_hashes"].items()
            }
        )
        result = dict(
            schema="drone-data.ai19.export/1",
            status="EXPORTED_NOT_TRAINED",
            battery_policy=POLICY,
            producer=binding,
            feature_profile="common19_ctx0",
            feature_order=mapping.FEATURES,
            records=records,
            history19_fit=fit,
            run_health19_manifests=manifests,
            unavailable_runs=[
                dict(run_id=r["run_id"], role=r["role"], **r["run_health"])
                for r in records
                if r["run_health"]["status"] == "unavailable"
            ],
            run_manifest_policy="One manifest per role; sharedtestpairedlineage retained in index; never relabel or merge differing roles to bypass loader",
            no_training_or_scoring=True,
            legacy_ctx6_converted=False,
        )
        result["files"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.is_file()
        }
        write(output / "export.json", result)
        write(
            output / "seal.json",
            dict(export_sha256=sha(output / "export.json")),
        )
        return result
    except BaseException as e:
        write(
            output / "failure.json",
            dict(
                status="FAILED_OUTPUTS_PRESERVED",
                reason=f"{type(e).__name__}: {e}",
            ),
        )
        raise


def check_export(output):
    output = no_links(output)
    value = read(output / "export.json")
    if (
        sha(output / "export.json")
        != read(output / "seal.json")["export_sha256"]
    ):
        raise ValueError("Export seal mismatch")
    for name, digest in value["files"].items():
        p = no_links(output / name)
        if (
            Path(name).is_absolute()
            or not p.is_relative_to(output)
            or sha(p) != digest
        ):
            raise ValueError("Export member changed")
    validate_lineage(value["records"])
    _, _, data, _ = ai_modules()
    for row in value["records"]:
        online = row["online"]
        if online["available"]:
            with np.load(
                output / online["file"]["path"], allow_pickle=False
            ) as arrays:
                n = len(arrays["X"])
                if (
                    set(arrays.files) != data.ARRAYS
                    or arrays["X"].shape != (n, 100, 19)
                    or arrays["Y"].shape != (n, 10, 4)
                    or not n
                ):
                    raise ValueError("Online19 array contract mismatch")
                if (
                    not np.isfinite(arrays["X"]).all()
                    or not np.isfinite(arrays["Y"]).all()
                ):
                    raise ValueError("Nonfinite online observations/targets")
    for ref in value["run_health19_manifests"].values():
        manifest = read(output / ref["path"])
        if manifest.get("interval_policy") != MISSION_POLICY:
            raise ValueError(
                "Run view must declare the current mission-only interval policy"
            )
        for row in manifest["runs"]:
            with np.load(output / row["path"], allow_pickle=False) as arrays:
                v, t = arrays["values"], arrays["time_us"]
                if (
                    set(arrays.files) != {"values", "time_us"}
                    or v.shape != (len(t), 19)
                    or len(t) <= 100
                    or t.dtype != np.int64
                    or not np.all(np.diff(t) == 100000)
                    or not np.isfinite(v).all()
                ):
                    raise ValueError("Fullrun19 array contract mismatch")
    return dict(
        status="TWO_AI_INPUT_VIEWS_VERIFIED_NOT_TRAINED",
        runs=len(value["records"]),
        online_available=sum(
            r["online"]["available"] for r in value["records"]
        ),
        run_available=sum(
            r["run_health"]["status"] == "available" for r in value["records"]
        ),
        history19_fit=value["history19_fit"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Currentbattery raw -> history19 online + fullrun19 inputs; no flight/training"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    verify_parser = sub.add_parser("check")
    verify_parser.add_argument(
        "--export", dest="export_path", type=Path, required=True
    )
    sub.add_parser("preflight")
    a = parser.parse_args(argv)
    if a.action == "preflight":
        ai_modules()
        print(
            json.dumps(
                dict(
                    status="CURRENT_AI19_BINDING_AND_IMPORTS_CHECKED",
                    scope="producer hashes and AI19 imports only; actual two-view intake is separate",
                    actual_two_view_admission_done=False,
                    producer=producer_binding(),
                    flight_started=False,
                    training_started=False,
                )
            )
        )
        return 0
    if a.action == "check":
        print(json.dumps(check_export(a.export_path)))
        return 0
    result = prepare(a.selection, a.output)
    print(
        json.dumps(
            dict(
                status=result["status"],
                runs=len(result["records"]),
                unavailable=len(result["unavailable_runs"]),
                history19_fit=result["history19_fit"],
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
