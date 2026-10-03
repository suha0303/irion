"""Use the existing collector, per-flight assessment and once-only claims."""

import copy
import hashlib
import json

from .common import (
    ADMISSION,
    COLLECTION_WAIT_BUDGETS,
    PRODUCER_MANIFEST,
    RESOURCES,
    ROOT,
    SIMULATION,
    STORE,
    collector,
    package_imports,
    producer_manifest,
    producer_sources,
    read,
    runs_root,
    sha,
    write,
)


def lanes(count):
    return [
        dict(
            instance=i,
            ports=[base + i for base in (14540, 14580, 18570, 14280, 13030)],
            gcs_remote_port=14600 + i,
        )
        for i in range(count)
    ]


def cases(seed_base, profile="legacy50"):
    if profile == "balanced300":
        from .balanced import cases as design
        from .composition import validate_cases

        rows = design(seed_base)
        validate_cases(rows, profile)
        return dict(
            schema="x500.finite-cases/2",
            revision="COLLECTION",
            cases=rows,
            collection_profile=profile,
            data_pool_id="collection/x500",
            normal_variation_revision="balanced300-random-continuous-v3",
        )
    record = read(RESOURCES / "case_manifest.json")
    seeds = sorted({r["seed"] for r in record["cases"]})
    mapping = {seed: seed_base + i for i, seed in enumerate(seeds)}
    for row in record["cases"]:
        previous = row["seed"]
        row.update(
            seed=mapping[previous],
            group_id=f"pretrain-group-{mapping[previous]}",
            case_id=f"pretrain_{row['scenario']}_{mapping[previous]}",
        )
    if profile in {"train104", "diverse104"}:
        train = [r for r in record["cases"] if r["role"] == "normal_train"]
        for index in range(6):
            for offset, template in enumerate(train):
                seed = seed_base + len(seeds) + 9 * index + offset
                row = copy.deepcopy(template)
                row.update(
                    seed=seed,
                    group_id=f"pretrain-group-{seed}",
                    case_id=f"pretrain_{row['scenario']}_{seed}",
                    generation_index=len(record["cases"]) + 1,
                )
                record["cases"].append(row)
    if profile == "diverse104":
        import sys

        sys.path.insert(0, str(SIMULATION / "sim/src"))
        from capstone_sim.normal_variation import sample

        for row in record["cases"]:
            if row["role"] != "anomaly_dev" and row["mission_seconds"] in {180, 900}:
                row["normal_variation"] = sample(row["scenario"], row["seed"])
        record["normal_variation_revision"] = "healthy-envelope-paths-v1"
    from .composition import validate_cases

    validate_cases(record["cases"], profile)
    record["collection_profile"] = profile
    record["data_pool_id"] = "collection/x500"
    return record


def preparation_assertions(preparation, profile_id):
    """Bind preparation to this machine's checked source and runtime seal."""
    active = producer_manifest()
    expected = dict(
        schema="x500.producer-runtime/1",
        revision="X500_CURRENT",
        configuration="collection",
        profile_id=profile_id,
        source_manifest_sha256=active["source_manifest_sha256"],
    )
    for key, value in expected.items():
        if preparation.get(key) != value:
            raise ValueError(
                f"Producer preparation {key} mismatch: "
                f"expected {value!r}, got {preparation.get(key)!r}"
            )
    return expected


def plan(task, phase, count, case_list, label, export_workers=1):
    c = collector()
    config = read(task / "config.json")
    from x500_storage import capture, task_root

    storage = task_root(config) if "external_storage" in config else STORE
    policy = config.get("storage_policy") or capture(storage)
    folder = task / label
    folder.mkdir()
    result = copy.deepcopy(read(RESOURCES / "collection/plan.json"))
    sources = producer_sources()
    project_sources = {"ai/x500_storage.py": sha(ROOT / "ai/x500_storage.py")}
    preparation = PRODUCER_MANIFEST
    project_sources[str(preparation.relative_to(ROOT))] = sha(preparation)
    isolation = dict(
        parallel_safe=True,
        uuid_partitioned=True,
        immutable_shared_inputs=True,
        lanes=lanes(count),
        source_hashes=sources,
        project_source_hashes=project_sources,
        basis="Reviewed private FC startup, unique UDP0..15, UUID Gazebo partitions, flock leases; actual capacity recorded separately",
    )
    write(folder / "isolation.json", isolation)
    result.update(
        root=str(SIMULATION),
        project_root=str(ROOT),
        phase=phase,
        campaign_id=f"pretrain_{task.name}_{label}",
        execution_dir=str(folder / "execution"),
        external_output=str(storage / "archives" / f"pretrain_{task.name}_{label}"),
        storage_policy=policy,
        runs_root=str(runs_root(ROOT)),
        storage_config=dict(path=str(task / "config.json"), sha256=sha(task / "config.json")),
        mission_seconds=120 if phase == "benchmark" else 900,
        concurrency=count,
        export_workers=min(export_workers, count),
        lanes=lanes(count),
        attempts_max=len(case_list),
        source_hashes=sources,
        project_source_hashes=project_sources,
        execution_authorized=True,
        gate_policy="optional_pretrain_cli_v1",
        pretrain_cli_approval=str(task / "config.json"),
        isolation_evidence=dict(
            path=str(folder / "isolation.json"), sha256=sha(folder / "isolation.json")
        ),
        producer_preparation=dict(path=str(preparation), sha256=sha(preparation)),
        preparation_assertions=preparation_assertions(read(preparation), result["profile_id"]),
        assessment_command=[
            str(SIMULATION / ".venv/bin/python"),
            "-B",
            str(SIMULATION / "sim/tools/assess_run.py"),
            "--run",
            "{run}",
            "--instance",
            "{instance}",
            "--duration",
            "{duration}",
            "--profile-id",
            "{profile_id}",
        ],
    )
    result.pop("gate_receipt", None)
    config = read(task / "config.json")
    if phase == "collection" and config.get("pipeline_overlap"):
        result["processing_stop_signal"] = str(task / "operator/processing-stop.json")
    if config.get("speed_test"):
        index = int(label.split("-")[-1]) - 1
        factor = config["speed_test"]["factors"][index]
        result["speed_comparison"] = dict(
            protocol=config["speed_test"].get("protocol", "information-v1"),
            index=index,
            factor=factor,
        )
        result["capture_interval_ms"] = 10.0 / factor
    result["continue_after_case_failure"] = config.get("continue_after_case_failure", False)
    result["unit_attempts_max"] = config.get("unit_attempts_max", 1)
    if config.get("use_available_ram"):
        if not (
            (config["mode"] == "test" and phase == "benchmark")
            or (config["mode"] == "collect" and phase in {"collection", "benchmark", "endurance"})
        ):
            raise ValueError("Expanded memory requires a finite pretrain task")
        memory_path = task / "operator/memory-policy.json"
        memory = read(memory_path)
        result["pretrain_memory_override"] = {
            "config": dict(path=str(task / "config.json"), sha256=sha(task / "config.json")),
            "memory": dict(path=str(memory_path), sha256=sha(memory_path)),
        }
        result["budgets"]["min_available_ram_bytes"] = 1_000_000_000
        result["budgets"]["memory_soft_cap_bytes"] = memory["max_bytes"]
    result["budgets"].update(COLLECTION_WAIT_BUDGETS)
    if phase == "collection":
        # Keep the old per-run allowance when the explicit finite set grows.
        for key in ("local_bytes_cap", "external_bytes_cap"):
            result["budgets"][key] = int(result["budgets"][key] * len(case_list) / 50)
    write(folder / "plan.json", result)
    write(folder / "manifest.json", dict(data_pool_id=result["data_pool_id"], cases=case_list))
    write(
        folder / "approval.json",
        dict(
            authorized=True,
            action="START_LONG_COLLECTION" if phase == "collection" else "START_COLLECTION_GATE",
            plan_sha256=sha(folder / "plan.json"),
            manifest_sha256=sha(folder / "manifest.json"),
            collector_sha256=sha(c.__file__),
        ),
    )
    return folder


class IncompleteCollection(RuntimeError):
    """All safe case attempts ended; incomplete splits must not enter preprocessing."""


def run(folder):
    c = collector()
    raw = read(folder / "plan.json")
    gate = raw["phase"] != "collection"
    plan, rows = c.bound(
        folder / "plan.json", folder / "manifest.json", folder / "approval.json", dry_run=gate
    )
    result = c.execute(plan, rows, c.helper(ROOT))
    if result["status"] == "COLLECTED_WITH_ERRORS_STOPPED":
        raise IncompleteCollection("Individual run failures preserved: " + str(folder))
    if result["status"] != "COLLECTED_PENDING_QUALITY_STOPPED" or result["not_started"]:
        raise RuntimeError("Collection/gate failed; evidence retained: " + str(folder))
    if gate and len(rows) >= plan["concurrency"]:
        from .common import overlap

        observed = overlap(
            [
                dict(started=r["flight_started_monotonic"], finished=r["flight_finished_monotonic"])
                for r in result["cases"].values()
            ]
        )
        if observed < plan["concurrency"]:
            raise RuntimeError("Requested flights did not overlap; not a parallel pass")
    return result


def freeze_benchmark(folder):
    """Admit fresh benchmark archives for processing tests, never training data."""
    package_imports()
    from preprocessing_tools.control import RAW_FILES

    from .common import load

    identity = load("pretrain_admission_identity", ADMISSION / "freeze.py").identity

    ledger = read(folder / "execution/ledger.json")
    cases = read(folder / "manifest.json")["cases"]
    if (
        ledger.get("phase") != "benchmark"
        or ledger.get("status") != "COLLECTED_PENDING_QUALITY_STOPPED"
        or ledger.get("not_started")
        or ledger.get("stop_reason")
        or set(ledger["cases"]) != {c["case_id"] for c in cases}
        or len(cases) < 3
    ):
        raise ValueError("Complete fresh benchmark ledger required")
    rows, seen = [], set()
    roles = ("normal_train", "normal_validation", "normal_calibration")
    for index, case in enumerate(cases):
        record = ledger["cases"][case["case_id"]]
        if (
            case["role"] != "benchmark"
            or record["case"] != case
            or record["status"] != "ARCHIVED_PENDING_QUALITY"
        ):
            raise ValueError("Benchmark case identity/status mismatch")
        assessment = record["assessment"]
        if assessment.get("run_id") != record["run_id"] or any(
            assessment.get(key) is not True
            for key in (
                "contract_passed",
                "instance_isolation_passed",
                "missing_data_passed",
                "physical_gate_passed",
            )
        ):
            raise ValueError("Benchmark assessment failed")
        export = record["export"]
        archive = dict(
            path=export["archive"],
            sha256=export["sha256"],
            bytes=export["bytes"],
            members=export["members"],
        )
        values = identity(archive)
        run, scenario = values["run.json"], values["evaluation/scenario.json"]
        if (
            run.get("run_id") != record["run_id"]
            or run.get("status") != "completed"
            or run.get("origin") != "px4_sitl"
            or run["run_id"] in seen
            or scenario.get("name") != case["scenario"]
            or scenario.get("seed") != case["seed"]
            or scenario.get("health_eligibility") != "confirmed_healthy"
            or scenario.get("requested_duration_s") != 120
            or scenario.get("normal_variation") != case.get("normal_variation")
            or scenario.get("continuity") != "independent_reset"
        ):
            raise ValueError("Fresh benchmark archive identity mismatch")
        seen.add(run["run_id"])
        files = {name: archive["members"][name]["sha256"] for name in RAW_FILES}
        if any(archive["members"][name]["kind"] != "file" for name in RAW_FILES):
            raise ValueError("Benchmark consumer members must be regular files")
        row = dict(
            case,
            run_id=run["run_id"],
            archive=archive,
            files=files,
            path=str(STORE / "consumer" / run["run_id"]),
            collection_role="benchmark",
            training_eligible=False,
        )
        # These are test-only processing partitions, not admitted training runs.
        row["role"] = roles[index % len(roles)]
        rows.append(row)
    result = dict(
        status="BENCHMARK_ONLY",
        training_eligible=False,
        runs=rows,
        ledger_sha256=sha(folder / "execution/ledger.json"),
    )
    write(folder / "benchmark-manifest.json", result)
    return result


def source_revision():
    source = read(SIMULATION / "artifacts/source_lock.json")
    source["local_sources"] = {
        str(p.relative_to(SIMULATION)): sha(p)
        for name in ("sim", "profiles", "contracts")
        for p in (SIMULATION / name).rglob("*")
        if p.is_file() and "__pycache__" not in str(p)
    }
    source["runtime_binary_sha256"] = {
        name: sha(SIMULATION / name)
        for name in (
            "vendor/PX4-Autopilot/build/px4_sitl_default/bin/px4",
            "build/current/libDronePowertrain.so",
        )
    }
    import sys

    sys.path.insert(0, str(SIMULATION / "sim/src"))
    from capstone_sim import collection_config

    library = collection_config.transport_library()
    if library is not None:
        source["runtime_binary_sha256"][str(library.relative_to(collection_config.ROOT))] = sha(library)
    # Path.rglob traversal differs across Python versions. Compare source content,
    # while each run's original file hash remains in its archive/member inventory.
    digest = hashlib.sha256(
        (json.dumps(source, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    ).hexdigest()
    producer_manifest()
    return dict(
        battery_profile="VTC6_4S4P",
        px4_binary_sha256=source["runtime_binary_sha256"][
            "vendor/PX4-Autopilot/build/px4_sitl_default/bin/px4"
        ],
        physics_plugin_sha256=source["runtime_binary_sha256"]["build/current/libDronePowertrain.so"],
        x500_profile_sha256=sha(SIMULATION / "profiles/collection/vehicle.yaml"),
        producer_lock_sha256=digest,
        producer_lock_hash_format="sorted-json-v1",
        preprocessing_sha256=sha(RESOURCES / "preprocessing/preprocessing.json"),
    )


def freeze_collection(task, folder, *, output=None, progress=None):
    output = task if output is None else output
    package_imports()
    # Load only the updated make_freeze entry; relative control remains the sealed package.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pretrain_admission_freeze", ADMISSION / "freeze.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = read(folder / "manifest.json")
    profile = read(task / "config.json").get("collection_profile", "legacy50")
    record["collection_profile"] = profile
    record["revision"] = "COLLECTION"
    write(output / "cases.json", record)
    model_plan = read(RESOURCES / "collection-plan.json")
    model_plan["storage_root"] = str(STORE)
    model_plan["cases_sha256"] = sha(output / "cases.json")
    write(output / "collection-plan.json", model_plan)
    write(output / "required-revision.json", source_revision())
    # Identity checks remain serial here; archive extraction/hash verification is parallel next.
    from .composition import validate_cases, validate_roles

    frozen = module.make_freeze(
        output / "collection-plan.json",
        output / "cases.json",
        folder / "execution/ledger.json",
        output / "required-revision.json",
        RESOURCES / "protected-inventory.json",
        validate_cases_fn=lambda rows: validate_cases(rows, profile),
        validate_roles_fn=lambda rows, cases, protected: validate_roles(
            rows, cases, protected, profile
        ),
        on_run=progress,
    )
    frozen["collection_profile"] = profile
    return frozen


def combine_frozen(task, parts):
    """Join independently admitted sets before fitting one global train scaler."""
    from .composition import validate_roles

    config = read(task / "config.json")
    if len(parts) != config.get("collection_sets", 1) or not parts:
        raise ValueError("All configured sets must finish before combined preparation")
    profile = config.get("collection_profile", "legacy50")
    merged = copy.deepcopy(parts[0][1])
    for key in ("ledger_sha256", "plan_sha256"):
        merged.pop(key, None)  # Per-set references are retained in source_freezes below.
    rows, cases, groups, references = [], [], [], []
    for output, frozen in parts:
        if frozen.get("status") != "DATA_FROZEN" or frozen.get("collection_profile") != profile:
            raise ValueError("Completed matching collection freeze required")
        for key in ("revision", "revision_lock", "protected_inventory_sha256"):
            if frozen.get(key) != merged.get(key):
                raise ValueError("Collection source/protection mismatch: " + key)
        record = read(output / "cases.json")
        cases.extend(record["cases"])
        groups.append([c["case_id"] for c in record["cases"]])
        rows.extend(frozen["runs"])
        write(output / "frozen-manifest.json", frozen)
        references.append(
            dict(
                path=str(output / "frozen-manifest.json"),
                sha256=sha(output / "frozen-manifest.json"),
                cases_sha256=sha(output / "cases.json"),
            )
        )
    validate_roles(
        rows, cases, read(RESOURCES / "protected-inventory.json")["runs"], profile, case_sets=groups
    )
    write(
        task / "cases.json",
        dict(
            revision="COLLECTION",
            collection_profile=profile,
            collection_set_case_ids=groups,
            cases=cases,
        ),
    )
    plan = read(RESOURCES / "collection-plan.json")
    plan["cases_sha256"] = sha(task / "cases.json")
    write(task / "collection-plan.json", plan)
    merged.update(
        runs=rows,
        cases_sha256=sha(task / "cases.json"),
        collection_sets=len(parts),
        collection_set_case_ids=groups,
        source_freezes=references,
        plan_sha256=sha(task / "collection-plan.json"),
        scaler_scope="all_selected_normal_train",
        training_authorized=False,
    )
    write(task / "frozen-manifest.json", merged)
    return merged
