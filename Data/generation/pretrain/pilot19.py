"""One pilot raw -> two no-fit19 views. The caller exits this process per run."""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from . import ai19
from .common import ROOT, STORE, Budget, collector, package_imports, read, sha, verify, write

COHORT = "pilot_development_reference"
SHORT_COHORT = "short_cycle_development_reference"
FINAL16_COHORT = "final16_development_reference"


def source_evidence(raw, binding, scenario):
    """Consume producer checks and recorded endpoints; truth never enters an array."""
    from capstone_sim import battery_aging, propulsion_health

    battery = read(raw / "evaluation/battery-selection.json")
    propulsion = read(raw / "evaluation/propulsion-selection.json")
    integrity = read(raw / "integrity.json")
    for name in (
        "evaluation/propulsion-selection.json",
        "evaluation/propulsion-catalog.json",
        "evaluation/truth.jsonl",
    ):
        path = ai19.no_links(raw / name)
        if path.stat().st_size != integrity[name]["size"] or sha(path) != integrity[name]["sha256"]:
            raise ValueError("Pilot source evidence changed: " + name)
    propulsion_health.verify_selection(
        propulsion,
        raw / "evaluation/propulsion-catalog.json",
        scenario["name"],
        binding["propulsion_catalog_sha256"],
    )
    propulsion_health.verify_truth(raw / "evaluation/truth.jsonl", propulsion)
    battery_aging.verify_truth(raw / "evaluation/truth.jsonl", battery)
    transitions = {}
    active = {
        system: selected
        for system, selected in (("battery", battery), ("propulsion", propulsion))
        if not selected["fixed_for_flight"]
    }
    endpoints = {}
    if active:
        with (raw / "evaluation/truth.jsonl").open() as stream:
            for line in stream:
                point = json.loads(line)
                for system in active:
                    field = (
                        "battery_stage_fraction"
                        if system == "battery"
                        else "propulsion_profile_fraction"
                    )
                    if system not in endpoints and point[field] >= 1 - 1e-9:
                        endpoints[system] = point["time_us"]
        schedule = scenario["fault_schedule"]
        for system in active:
            if system not in endpoints or not schedule.get("acknowledged"):
                raise ValueError("Actual dynamic endpoint/acknowledgement missing")
            transitions[system] = dict(
                actual_onset_us=schedule["applied_time_us"],
                actual_endpoint_us=endpoints[system],
                requested_offset_s=schedule["requested_offset_us"] / 1e6,
                clock_id="px4-boot",
            )
    return dict(
        producer_runtime_sha256=binding["runtime_sha256"],
        scenario_name=scenario["name"],
        battery_selection=battery,
        propulsion_selection=propulsion,
        transition_evidence=transitions,
    )


def helper_inventory(*, final16=False):
    paths = [
        Path(__file__),
        Path(ai19.__file__),
        ROOT / "ai/learning/wind_metadata.py",
        ROOT / "ai/learning/admission19.py",
        ROOT / "ai/learning/history19/data.py",
        ROOT / "ai/learning/history19/preprocessing.py",
        ROOT / "ai/learning/history19/preprocessing.json",
        ROOT / "ai/learning/history19/pilot.py",
        ROOT / "ai/learning/history19/pilot_views.py",
        ROOT / "ai/learning/history19/pilot_connector.py",
        ROOT / "ai/learning/history19/joint_native.py",
        ROOT / "ai/learning/history19/native_audit.py",
        ROOT / "ai/learning/history19/duration_profiles.py",
        ROOT / "ai/learning/history19/short_cycle.py",
        ROOT / "ai/learning/history19/quality_rejection.py",
        ROOT / "ai/learning/run_health19/data.py",
        ROOT / "ai/src/capstone_ai/mapping.py",
        ROOT / "ai/src/capstone_ai/source.py",
        ROOT / "ai/src/capstone_ai/dataset.py",
    ]
    if final16:
        paths.append(ROOT / "ai/learning/history19/final16.py")
    return {str(p): sha(p) for p in paths}


def prepare_one(
    selection,
    output,
    expected_path,
    archive_cache,
    protection_path,
    *,
    policy_bundle=None,
    short_cycle=False,
    final16=False,
):
    started = time.monotonic()
    mapping, _, _, _ = ai19.ai_modules()
    from learning.admission19 import check_producer
    from learning.admission19 import source_evidence as check_source
    from learning.history19.data import inspect_run
    from learning.run_health19.data import iter_manifests
    from learning.wind_metadata import describe

    if short_cycle and final16:
        raise ValueError("Separate no-fit scopes required")
    cohort = FINAL16_COHORT if final16 else SHORT_COHORT if short_cycle else COHORT
    usage = (
        "final16_input_check"
        if final16
        else "short_cycle_input_check"
        if short_cycle
        else "pilot_input_check"
    )
    if selection.get("cohort") != cohort or selection.get("role") not in {
        "normal_test",
        "anomaly_test",
    }:
        raise ValueError("Pilot no-fit cohort and evaluation roles required")
    output = ai19.no_links(output)
    ai19.external(output)
    if output.exists():
        raise ValueError("Preserve prior processing attempts; use a new attempt folder")
    binding = ai19.producer_binding()
    expected = read(expected_path)
    check_producer(binding, expected)
    row, evidence = (
        ai19.admit(selection, binding, policy_bundle=policy_bundle)
        if policy_bundle is not None
        else ai19.admit(selection, binding)
    )
    raw = Path(row["path"])
    scenario = read(raw / "evaluation/scenario.json")
    row.update(
        cohort=cohort,
        purpose=cohort,
        lineage=ai19.lineage(row),
        mission_interval=evidence["mission_interval"],
        source_evidence=source_evidence(raw, binding, scenario),
    )
    check_source(row, binding)
    wind = (
        describe(scenario, wind_clock=read(policy_bundle["paired_policy"])["wind_model_clock"])
        if policy_bundle
        else describe(scenario)
    )
    if wind["wind_mode"] != "bounded_profile":
        raise ValueError("Pilot requires explicit current bounded wind")
    helpers = helper_inventory(final16=True) if final16 else helper_inventory()
    output.mkdir(parents=True)
    protection = read(protection_path)
    if protection.get("schema") != "drone-ai.global-protection/1":
        raise ValueError("Frozen global protection registry required")
    protection["entries"].append(
        dict(
            identities=["run:" + row["run_id"], *row["lineage"]],
            forbidden_uses=[
                "fit",
                "input_check",
                "calibration",
                "final",
                "final_test",
                "final_input_check",
            ],
            purpose=cohort,
        )
    )
    write(output / "protection.json", protection)
    write(
        output / "binding.json",
        dict(
            producer=binding,
            expected=expected,
            expected_sha256=sha(expected_path),
            consumer_sources=helpers,
            cohort=cohort,
            no_fit=True,
            wind_metadata=wind,
            raw_integrity_sha256=sha(raw / "integrity.json"),
        ),
    )
    archive_cache = Path(archive_cache)
    token = archive_cache.parent.name + "-" + raw.name
    archive_path = (
        STORE / ("archives/final16" if final16 else "archives/pilot45") / (token + ".tar.gz")
    )
    if archive_cache.exists():
        cached = read(archive_cache)
        if cached["raw_integrity_sha256"] != sha(raw / "integrity.json") or cached["archive"][
            "sha256"
        ] != sha(Path(cached["archive"]["archive"])):
            raise ValueError("Preserved archive cache changed")
        archived = cached["archive"]
    else:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        (output / "archive-evidence").mkdir()
        archived = collector().archive_run(
            raw,
            archive_path,
            output / "archive-evidence",
            dict(archive_wall_s=14400, archive_verify_wall_s=14400, byte_stall_s=3600),
            lambda: None,
        )
        write(
            archive_cache, dict(raw_integrity_sha256=sha(raw / "integrity.json"), archive=archived)
        )
    package_imports()
    from .archive import extract_consumer

    consumer = STORE / ("consumer/final16" if final16 else "consumer/pilot45") / token / output.name
    consumer.parent.mkdir(parents=True, exist_ok=True)
    export = dict(
        path=archived["archive"],
        sha256=archived["sha256"],
        bytes=archived["bytes"],
        members=archived["members"],
    )
    extract_consumer(
        dict(
            archive=export,
            path=str(consumer),
            files={n: archived["members"][n]["sha256"] for n in ai19.CONSUMER},
        ),
        Budget(),
    )
    engine, alias = ai19.replay(
        consumer,
        read(ROOT / "ai/learning/history19/preprocessing.json"),
        expected_profile_revision=binding["vehicle_profile_revision"],
    )
    for name in ("frames", "windows", "exclusions"):
        with (output / (name + ".jsonl")).open("x") as stream:
            for point in getattr(engine, name):
                stream.write(json.dumps(point, allow_nan=False) + "\n")
    arrays, exclusions = ai19.online_arrays(engine, row["mission_interval"])
    physical, coverage = ai19.run_array(engine, row["mission_interval"])
    online = dict(
        available=arrays is not None,
        issued_windows=len(engine.windows),
        exclusions=exclusions,
        valid_windows=0,
        target_valid_seconds=0,
    )
    if arrays is not None:
        np.savez(output / "online.npz", **arrays)
        sidecar = dict(
            schema="drone-ai.history19.run/1",
            units="nameplate_transformed_common19_cellV_Crate",
            interval_policy=ai19.MISSION_POLICY,
            feature_order=mapping.FEATURES,
            target_order=mapping.TARGETS,
            step_us=100000,
            history=100,
            future=10,
            emit_every_us=1000000,
            producer=binding,
            run=row,
            file=ai19.reference(output / "online.npz", output),
        )
        write(output / "online.json", sidecar)
        inspect_run(
            output / "online.json",
            expected_producer=expected,
            protection=protection,
            usage=usage,
        )
        online.update(
            valid_windows=len(arrays["X"]),
            target_valid_seconds=len(arrays["X"]),
            manifest=str(output / "online.json"),
        )
    run_view = dict(coverage, manifest=None, post_initial100_seconds=0)
    if physical is not None:
        np.savez(output / "run.npz", **physical)
        item = dict(
            row, health=row["health_eligibility"], **ai19.reference(output / "run.npz", output)
        )
        manifest = dict(
            schema="drone-ai.run-health19.dataset/1",
            units="physical_common19_pack_V_A",
            step_us=100000,
            feature_order=mapping.FEATURES,
            vehicle_profile="x500_reference",
            battery_policy=ai19.POLICY,
            interval_policy=ai19.MISSION_POLICY,
            producer=binding,
            runs=[item],
        )
        write(output / "run.json", manifest)
        for checked in iter_manifests(
            [output / "run.json"],
            expected_producer=expected,
            protection=protection,
            usage=usage,
        ):
            del checked
        run_view.update(
            manifest=str(output / "run.json"),
            post_initial100_seconds=max(0, len(physical["time_us"]) - 100) / 10,
        )
    verify(helpers)
    verify({str(ROOT / "simulation" / p): h for p, h in binding["source_hashes"].items()})
    if "torch" in sys.modules:
        raise ValueError("Data preparation must not import torch")
    result = dict(
        schema="drone-data.pilot19.run/1",
        status="EXPORTED_NO_FIT",
        run=row,
        producer=binding,
        online=online,
        run_health=run_view,
        wind=wind,
        archive=export,
        consumer_path=str(consumer),
        validation_alias=alias,
        no_fit=True,
        training_started=False,
        inference_started=False,
        elapsed_s=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        output_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        consumer_bytes=sum(p.stat().st_size for p in consumer.rglob("*") if p.is_file()),
    )
    result["files"] = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}
    write(output / "receipt.json", result)
    write(output / "seal.json", dict(receipt_sha256=sha(output / "receipt.json")))
    return result


def audit_one(output, expected_path, *, short_cycle=False, final16=False):
    """Separate process: readers plus canonical/NPZ values, mission grid and units."""
    mapping, _, _, _ = ai19.ai_modules()
    from capstone_ai.dataset import feature_transform, target_transform
    from learning.history19.data import inspect_run
    from learning.run_health19.data import iter_manifests

    output = ai19.no_links(output)
    receipt = read(output / "receipt.json")
    if short_cycle and final16:
        raise ValueError("Separate no-fit scopes required")
    cohort = FINAL16_COHORT if final16 else SHORT_COHORT if short_cycle else COHORT
    usage = (
        "final16_input_check"
        if final16
        else "short_cycle_input_check"
        if short_cycle
        else "pilot_input_check"
    )
    if receipt["run"].get("cohort") != cohort or receipt["run"].get("purpose") != cohort:
        raise ValueError("Explicit matching no-fit purpose/cohort required for this audit")
    if sha(output / "receipt.json") != read(output / "seal.json")["receipt_sha256"]:
        raise ValueError("Per-run receipt changed")
    verify({str(output / p): h for p, h in receipt["files"].items()})
    expected = read(expected_path)
    protection = read(output / "protection.json")
    mission = receipt["run"]["mission_interval"]
    frames = {}
    with (output / "frames.jsonl").open() as stream:
        for line in stream:
            frame = json.loads(line)
            if mission["start_us"] <= frame["time_us"] < mission["end_us_exclusive"]:
                if frame["time_us"] in frames:
                    raise ValueError("Repeated mission frame")
                frames[frame["time_us"]] = frame
    if receipt["run_health"]["status"] == "available":
        for checked in iter_manifests(
            [output / "run.json"],
            expected_producer=expected,
            protection=protection,
            usage=usage,
        ):
            del checked
        with np.load(output / "run.npz", allow_pickle=False) as a:
            times = sorted(frames)
            if a["time_us"].tolist() != times:
                raise ValueError("Physical view lost mission/tail frames")
            values = [[frames[t]["values"][n] for n in mapping.FEATURES] for t in times]
            np.testing.assert_allclose(a["values"], values, rtol=0, atol=1e-12)
    if receipt["online"]["available"]:
        inspect_run(
            output / "online.json",
            expected_producer=expected,
            protection=protection,
            usage=usage,
        )
        config = read(ROOT / "ai/learning/history19/preprocessing.json")
        with np.load(output / "online.npz", allow_pickle=False) as a:
            origins, x, y = a["origin_us"], a["X"], a["Y"]
            for i, origin in enumerate(origins):
                history = [frames[int(origin) - 9900000 + j * 100000] for j in range(100)]
                future = [frames[int(origin) + j * 100000] for j in range(1, 11)]
                expected_x = feature_transform(
                    [[f["values"][n] for n in mapping.FEATURES] for f in history], config
                )
                expected_y = target_transform(
                    [[f["values"][n] for n in mapping.TARGETS] for f in future]
                )
                np.testing.assert_array_equal(x[i], expected_x)
                np.testing.assert_array_equal(y[i], expected_y)
    if "torch" in sys.modules or receipt.get("no_fit") is not True:
        raise ValueError("No-fit CPU input check only")
    return dict(
        status="INDEPENDENT_INPUT_QUALITY_CHECKED_NO_FIT",
        run_id=receipt["run"]["run_id"],
        cohort=cohort,
        online_available=receipt["online"]["available"],
        run_available=receipt["run_health"]["status"] == "available",
        mission_frames=len(frames),
        online_windows=receipt["online"]["valid_windows"],
        export_receipt_sha256=sha(output / "receipt.json"),
    )
