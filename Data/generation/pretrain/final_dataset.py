"""Fixed final holdout: collect and audit canonical data, never fit or score a model."""

import copy
import random
from collections import Counter, defaultdict
from pathlib import Path

from .common import (
    ADMISSION,
    HOME,
    PACKAGE,
    RESOURCES,
    ROOT,
    STORE,
    collector,
    external,
    load,
    package_imports,
    read,
    sha,
    verify,
    write,
)

PROFILE = "final324"
DESIGN_REVISION = "final324-diverse-v2"
COHORTS = {"standalone_normal": 108, "paired_control": 108, "anomaly": 108}
MOTIONS = ("hover", "vertical", "horizontal", "yaw", "composite", "random")
ENVIRONMENTS = ("reference", "middle_soc", "low_soc", "wind", "cold", "payload")
PAIR_ENVIRONMENTS = (
    ("reference", "low_soc", "cold"),
    ("reference", "middle_soc", "payload"),
    ("middle_soc", "wind", "payload"),
    ("reference", "wind", "cold"),
    ("middle_soc", "low_soc", "cold"),
    ("low_soc", "wind", "payload"),
)


def normal_scenario(motion, environment):
    if environment in {"low_soc", "wind", "cold", "payload"}:
        return environment + "_normal"
    return {
        "hover": "hover_normal",
        "vertical": "altitude_normal",
        "horizontal": "horizontal_normal",
        "yaw": "yaw_normal",
        "composite": "waypoint_normal",
        "random": "waypoint_normal",
    }[motion]


def operating_conditions(motion, environment, seed, stratum):
    """Freeze path and environment independently of fault/health parameters."""
    import math
    import sys

    from .common import SIMULATION

    simulation_source = str(SIMULATION / "sim/src")
    if simulation_source not in sys.path:
        sys.path.insert(0, simulation_source)
    from capstone_sim.normal_variation import sample, validate
    from capstone_sim.random_flight import build as random_path

    rng = random.Random(seed ^ 0xF324)

    # Three equal strata for independent normal replicates; pairs sample one stratum.
    def draw(low, high):
        return low + (high - low) * (stratum + rng.random()) / 3

    spec = sample("hover_normal", seed)
    spec.update(soc=0.9, temperature=298.15, payload=0.0, wind_e=0.0, wind_n=0.0, gust_sigma=0.0)
    if environment == "reference":
        spec["soc"] = draw(0.87, 0.95)
    elif environment == "middle_soc":
        spec["soc"] = draw(0.65, 0.75)
    elif environment == "low_soc":
        spec["soc"] = draw(0.4, 0.5)
    elif environment == "wind":
        speed, direction = draw(1.0, 2.0), rng.uniform(-math.pi, math.pi)
        spec.update(
            wind_e=speed * math.cos(direction),
            wind_n=speed * math.sin(direction),
            gust_sigma=draw(0.1, 0.3),
        )
    elif environment == "cold":
        spec["temperature"] = draw(283.15, 291.15)
    elif environment == "payload":
        spec["payload"] = draw(0.1, 0.2)
    else:
        raise ValueError("Unknown final operating condition")
    seconds = 180 if environment == "low_soc" else 480
    path = spec["path"]
    if motion == "random":
        spec["path"] = random_path(spec, seed, seconds)
    elif motion == "composite":
        path.update(kind="mixed", radius=min(path["radius"], 1.2), omega=min(path["omega"], 0.08))
        # Short low-SOC flights retain the existing >=80s component guard.
        kinds = (
            ("altitude", "line")
            if seconds == 180
            else ("hover", "altitude", "line", "yaw", "figure8", "ellipse")
        )
        components = [dict(path, kind=kind) for kind in kinds]
        path.update(
            duration_s=seconds,
            program=[
                dict(start_s=i * seconds / len(kinds), duration_s=seconds / len(kinds), path=p)
                for i, p in enumerate(components)
            ],
        )
    else:
        path["kind"] = {
            "hover": "hover",
            "vertical": "altitude",
            "horizontal": rng.choice(("line", "ellipse")),
            "yaw": "yaw",
        }[motion]
    return validate(spec)


def cases(seed_base):
    templates = read(RESOURCES / "case_manifest.json")["cases"]
    normal = {row["scenario"]: row for row in templates if row["role"] == "normal_train"}
    anomaly = {row["scenario"]: row for row in templates if row["role"] == "anomaly_dev"}
    if len(normal) != 9 or len(anomaly) != 6:
        raise ValueError("Final holdout requires the existing nine normal/six anomaly definitions")
    rows = []

    def add(template, cohort, seed, motion, environment, stratum, **extra):
        row = copy.deepcopy(template)
        row.pop("normal_variation", None)
        row.pop("fault_at_s", None)
        row.update(
            case_id=f"F324v2_{cohort}_{template['scenario']}_{seed}",
            role="anomaly_test" if cohort == "anomaly" else "normal_test",
            final_cohort=cohort,
            design_revision=DESIGN_REVISION,
            seed=seed,
            group_id=f"F324v2-normal-{seed}",
            mission_seconds=180 if environment == "low_soc" else 480,
            composition_seed=seed_base,
            motion=motion,
            environment=environment,
            condition_stratum=stratum,
            path_mode={"composite": "mixed", "random": "random"}.get(motion, "basic"),
            operation_variation=operating_conditions(motion, environment, seed, stratum),
            retries=0,
            attempts_max=3,
            parent_attempt_id=None,
        )
        row.update(extra)
        rows.append(row)

    for motion_index, motion in enumerate(MOTIONS):
        for environment_index, environment in enumerate(ENVIRONMENTS):
            for repeat in range(3):
                seed = seed_base + motion_index * 18 + environment_index * 3 + repeat
                add(
                    normal[normal_scenario(motion, environment)],
                    "standalone_normal",
                    seed,
                    motion,
                    environment,
                    repeat,
                )
    for kind, name in enumerate(sorted(anomaly)):
        long_index = 0
        for motion_index, motion in enumerate(MOTIONS):
            for repeat, environment in enumerate(PAIR_ENVIRONMENTS[motion_index]):
                seed = seed_base + 1000 + kind * 18 + motion_index * 3 + repeat
                group = f"F324v2-pair-{seed}"
                offset = (
                    100 if environment == "low_soc" else (180, 240, 300)[(long_index + kind) % 3]
                )
                if environment != "low_soc":
                    long_index += 1
                dynamic = name.endswith(("_step", "_ramp"))
                shared = dict(
                    group_id=group,
                    pair_id=group,
                    paired_anomaly=name,
                    pair_anchor_after_mission_s=offset if dynamic else None,
                )
                stratum = (kind + motion_index + repeat) % 3
                add(
                    normal[normal_scenario(motion, environment)],
                    "paired_control",
                    seed,
                    motion,
                    environment,
                    stratum,
                    **shared,
                )
                add(
                    anomaly[name],
                    "anomaly",
                    seed,
                    motion,
                    environment,
                    stratum,
                    fault_after_mission_s=offset if dynamic else None,
                    **shared,
                )
    random.Random(seed_base ^ 324).shuffle(rows)
    for index, row in enumerate(rows, 1):
        row["generation_index"] = index
    return rows


def validate(rows):
    if len(rows) != 324 or Counter(row.get("final_cohort") for row in rows) != COHORTS:
        raise ValueError("Final324 cohort membership mismatch")
    base = rows[0].get("composition_seed")
    if type(base) is not int or rows != cases(base):
        raise ValueError("Final324 conditions, pairing or length changed")


def protection_inputs(exclude_task=None):
    """Only manifests/planned membership, never old raw observations or model scores."""
    paths = [RESOURCES / "protected-inventory.json"]
    for storage in sorted({ROOT / "Data", Path(STORE)}):
        paths += sorted((storage / "processed").glob("*/prepared/frozen-manifest.json"))
        paths += sorted((storage / "processed/final-test").glob("*/sealed-manifest.json"))
    paths += sorted((HOME / "tasks").glob("*/planned-cases.json"))
    sources, seeds, groups, raw_ids, hashes = {}, set(), set(), set(), set()
    for path in paths:
        if exclude_task is not None and path.parent == Path(exclude_task):
            continue
        value = read(path)
        sources[str(path)] = sha(path)
        rows = value.get("runs", []) + value.get("cases", [])
        rows += [row for batch in value.get("sets", []) for row in batch]
        for row in rows:
            if type(row.get("seed")) is int:
                seeds.add(row["seed"])
            if row.get("group_id"):
                groups.add(row["group_id"])
            raw_ids.update(row[key] for key in ("run_id", "raw_flight_id") if row.get(key))
            if row.get("raw_sha256"):
                hashes.add(row["raw_sha256"])
    return dict(
        sources=sources,
        seeds=sorted(seeds),
        group_ids=sorted(groups),
        raw_ids=sorted(raw_ids),
        raw_sha256=sorted(hashes),
    )


def reject_overlap(rows, protected):
    if {row["seed"] for row in rows} & set(protected["seeds"]) or {
        row["group_id"] for row in rows
    } & set(protected["group_ids"]):
        raise ValueError(
            "Final holdout seed/group overlaps existing development or previous collection"
        )


def configure(task, config, rows):
    validate(rows)
    protected = protection_inputs(exclude_task=task)
    reject_overlap(rows, protected)
    protection = task / "final-protection.json"
    write(protection, protected)
    config["source_sha256"].update(protected["sources"])
    config["source_sha256"][str(protection)] = sha(protection)
    config.update(
        dataset_purpose="locked_final_test",
        design_revision=DESIGN_REVISION,
        final_protection=str(protection),
        final_test_scoring_authorized=False,
        training_authorized=False,
        scaler_fit_authorized=False,
        calibration_authorized=False,
        storage_profile="standard",
        pipeline_overlap=False,
        unit_attempts_max=3,
        continue_after_case_failure=True,
        scope="Final holdout324: collect, canonical preprocess, independent quality audit, seal; no scaler fit, training, calibration or scoring",
    )


def admit_rows(folder, planned, protected, destination):
    """Use the existing archive identity reader, with final-specific role guards."""
    package_imports()
    from preprocessing_tools.control import RAW_FILES

    module = load("final_archive_identity", ADMISSION / "freeze.py")
    ledger = read(folder / "execution/ledger.json")
    source_plan = read(folder / "plan.json")
    if ledger["status"] != "COLLECTED_PENDING_QUALITY_STOPPED" or ledger.get("not_started"):
        raise ValueError("Incomplete final collection cannot be sealed")
    if set(ledger["cases"]) != {row["case_id"] for row in planned}:
        raise ValueError("Final ledger membership differs")
    rows, seen_ids, seen_hashes = [], set(protected["raw_ids"]), set(protected["raw_sha256"])
    for case in planned:
        record = ledger["cases"][case["case_id"]]
        if record["status"] != "ARCHIVED_PENDING_QUALITY" or record["case"] != case:
            raise ValueError("Final case failed or changed")
        attempt = record.get("attempt", 1)
        limit = source_plan.get("unit_attempts_max", 1)
        history = record.get("previous_attempts", [])
        if (
            limit != 3
            or case.get("attempts_max") != 3
            or not isinstance(history, list)
            or type(attempt) is not int
            or not 1 <= attempt <= limit
            or record.get("attempts_max", limit) != limit
            or len(history) != attempt - 1
            or [item.get("attempt") for item in history] != list(range(1, attempt))
        ):
            raise ValueError("Final attempt identity/history mismatch")
        if any(
            record.get("assessment", {}).get(k) is not True
            for k in (
                "contract_passed",
                "instance_isolation_passed",
                "missing_data_passed",
                "physical_gate_passed",
            )
        ):
            raise ValueError("Final source quality assessment incomplete")
        export = record["export"]
        archive = dict(
            path=export["archive"],
            sha256=export["sha256"],
            bytes=export["bytes"],
            members=export["members"],
        )
        values = module.identity(archive)
        meta, scenario = values["run.json"], values["evaluation/scenario.json"]
        collector().verify_run_source_snapshot(source_plan, values["reference/source_lock.json"])
        if (
            meta.get("run_id"),
            meta.get("status"),
            meta.get("origin"),
            scenario.get("name"),
            scenario.get("seed"),
            scenario.get("requested_duration_s"),
            scenario.get("continuity"),
        ) != (
            record["run_id"],
            "completed",
            "px4_sitl",
            case["scenario"],
            case["seed"],
            case["mission_seconds"],
            "independent_reset",
        ):
            raise ValueError("Final archived identity mismatch")
        if any(
            value.get(key)
            for value in (meta, scenario)
            for key in ("parent_attempt_id", "previous_attempts", "continuation_id")
        ) or any(
            scenario.get(key) != case.get(key)
            for key in ("normal_variation", "operation_variation", "fault_after_mission_s")
        ):
            raise ValueError("Final continuation or conditions differ")
        onset = case.get("fault_after_mission_s")
        schedule = scenario.get("fault_schedule")
        if onset is not None:
            time_keys = (
                "mission_origin_time_us",
                "requested_offset_us",
                "scheduled_time_us",
                "applied_time_us",
            )
            if (
                not isinstance(schedule, dict)
                or schedule.get("acknowledged") is not True
                or schedule.get("clock") != "px4-boot"
                or schedule.get("origin") != "airborne_path_zero"
                or any(type(schedule.get(k)) is not int or schedule[k] < 0 for k in time_keys)
                or schedule["requested_offset_us"] != onset * 1_000_000
                or schedule["scheduled_time_us"]
                != schedule["mission_origin_time_us"] + schedule["requested_offset_us"]
                or not schedule["scheduled_time_us"]
                <= schedule["applied_time_us"]
                < schedule["mission_origin_time_us"] + case["mission_seconds"] * 1_000_000
            ):
                raise ValueError("Final actual mission fault schedule missing or inconsistent")
        elif schedule is not None:
            raise ValueError("Final unscheduled case contains a fault schedule")
        if (
            case["role"] == "normal_test"
            and scenario.get("health_eligibility") != "confirmed_healthy"
        ):
            raise ValueError("Final normal health eligibility missing")
        files = {name: export["members"][name]["sha256"] for name in RAW_FILES}
        if any(export["members"][name]["kind"] != "file" for name in RAW_FILES):
            raise ValueError("Final consumer inputs must be regular files")
        rid, raw_hash = record["run_id"], files["observations/streams.jsonl"]
        raw_id = meta.get("raw_flight_id", rid)
        if rid in seen_ids or raw_id in seen_ids or raw_hash in seen_hashes:
            raise ValueError("Final raw identity overlaps existing or another final run")
        seen_ids.add(rid)
        seen_ids.add(raw_id)
        seen_hashes.add(raw_hash)
        rows.append(
            dict(
                case,
                run_id=rid,
                raw_flight_id=raw_id,
                raw_sha256=raw_hash,
                path=str(STORE / "consumer/final-test" / destination / rid),
                files=files,
                archive=archive,
                fault_schedule=copy.deepcopy(schedule),
                collection_attempt=attempt,
                collection_attempts_max=limit,
                collection_previous_attempts=copy.deepcopy(history),
            )
        )
    return rows


def run(runner):
    config, task = runner.config, runner.task
    planned = read(config["planned_cases"])["sets"]
    if len(planned) != 1 or config["unit_attempts_max"] != 3 or config.get("pipeline_overlap"):
        raise ValueError("One final324 set, maximum3 attempts, isolated final path required")
    rows = planned[0]
    validate(rows)
    protected = read(config["final_protection"])
    verify(protected["sources"])
    reject_overlap(rows, protection_inputs(exclude_task=task))
    runner.update(planned_stage_count=5, final_test_scoring_authorized=False)
    folder = runner.flight("collection", config["flight_workers"], rows, "final-collection")
    runner.begin("final-admission", total=324, workers=1)
    admitted = admit_rows(folder, rows, protected, task.name)
    runner.finish("final-admission", dict(passed=True, completed=324))
    output = STORE / "processed/final-test" / task.name
    output.mkdir(parents=True, exist_ok=False)
    consumer = external(STORE / "consumer/final-test" / task.name)
    consumer.mkdir(parents=True, exist_ok=False)
    runner.batch("final-extract", "extract", admitted, config["io_workers"], task / "final-extract")
    runner.batch(
        "final-preprocess", "preprocess", admitted, config["workers"], output / "canonical"
    )
    runner.batch(
        "final-quality",
        "audit",
        admitted,
        config["audit_workers"],
        output / "quality",
        canonical=output / "canonical",
    )
    quality = {
        row["run_id"]: read(output / "quality" / row["run_id"] / "report.json") for row in admitted
    }
    canonical_hashes = {}
    for row in admitted:
        summary_path = output / "canonical" / row["run_id"] / "summary.json"
        summary = read(summary_path)
        inventory = {str(summary_path): sha(summary_path)}
        inventory.update(
            {
                str(summary_path.parent / (name + ".jsonl")): digest
                for name, digest in summary["output_hashes"].items()
            }
        )
        verify(inventory)
        canonical_hashes.update(inventory)
    pairs = defaultdict(list)
    for row in admitted:
        if row.get("pair_id"):
            pairs[row["pair_id"]].append(row["run_id"])
    sealed = dict(
        schema="drone.final-test-dataset/1",
        status="FINAL_DATASET_SEALED_STOPPED",
        task=task.name,
        collection_profile=PROFILE,
        design_revision=DESIGN_REVISION,
        condition_counts={
            cohort: {
                "motion": dict(
                    Counter(r["motion"] for r in admitted if r["final_cohort"] == cohort)
                ),
                "environment": dict(
                    Counter(r["environment"] for r in admitted if r["final_cohort"] == cohort)
                ),
                "planned_mission_seconds": sum(
                    r["mission_seconds"] for r in admitted if r["final_cohort"] == cohort
                ),
            }
            for cohort in COHORTS
        },
        cohorts=COHORTS,
        runs=admitted,
        pairs=dict(pairs),
        lineage_groups=len({row["group_id"] for row in admitted}),
        independence_basis="Distinct seeded resets between groups; each normal/anomaly pair remains correlated; no empirical independence claim",
        producer_binding=read(folder / "plan.json")["source_hashes"],
        protection=dict(path=config["final_protection"], sha256=sha(config["final_protection"])),
        cases_sha256=sha(Path(config["planned_cases"])),
        collection_ledger=dict(
            path=str(folder / "execution/ledger.json"), sha256=sha(folder / "execution/ledger.json")
        ),
        preprocessing_package_sha256=sha(PACKAGE / "PACKAGE.json"),
        canonical_sha256=canonical_hashes,
        quality_by_run=quality,
        scaler_fit_performed=False,
        normalization="Unscaled canonical inputs/targets; final inference must use existing training scaler",
        training_authorized=False,
        calibration_authorized=False,
        final_test_scoring_authorized=False,
        truth_policy="Truth remains evaluator-only in source archives; canonical preprocessing uses exact six consumer files",
        zero_target_runs=[rid for rid, q in quality.items() if not q["accepted_targets"]],
        exposure_note="Quality airborne seconds and accepted targets are not model-scored exposure",
    )
    write(output / "sealed-manifest.json", sealed)
    write(
        output / "seal.json",
        dict(manifest_sha256=sha(output / "sealed-manifest.json"), status=sealed["status"]),
    )
    runner.update(final_dataset_sealed=True, result=str(output / "sealed-manifest.json"))
