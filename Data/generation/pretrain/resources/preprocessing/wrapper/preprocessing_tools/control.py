"""R3 has its own 50-case guard. R1's 49-case guard is never patched or bypassed."""

import importlib.util
import os
import stat
from collections import Counter
from pathlib import Path

from preprocessing_core.control import CONFIG, FIT_ROLES, MODELS, lineage, read, sha, write

REVISION = "R3_LONG15"
STORAGE = Path("/media/sa/X31/Data/X500/ROUND_001/R3_LONG15")
LEDGER = Path("/abr/coss43/.x500-round001-r3-long15")
COUNTS = dict(
    normal_train=9, normal_validation=9, normal_calibration=9, normal_dev_eval=11, anomaly_dev=12
)
STAGES = ("prepare", "main", "calibrate")
PROFILE = dict(
    config=CONFIG,
    models=list(MODELS),
    lstm_backend="aten_cuda",
    precision="float32",
    amp=False,
    stage_minutes=180,
    local_stage_minutes=120,
    calibration_minutes=120,
    cadence_seconds=1,
    union_false_alarms_per_hour=1,
    queue_minutes=15,
    retries=0,
    gpu_jobs_max=1,
    final_test_authorized=False,
    truth_join_authorized=False,
    data_generation_authorized=False,
)
RAW_FILES = {
    "run.json",
    "observations/metadata.json",
    "observations/streams.jsonl",
    "observations/watermarks.jsonl",
    "observations/diagnostics.jsonl",
    "reference/source_lock.json",
}
DATA_FILES = {
    "normal_train.npz",
    "normal_validation.npz",
    "normal_calibration.npz",
    "scaler.npz",
    "manifest.json",
}


def exact(record, expected, message):
    if any(type(record.get(k)) is not type(v) or record[k] != v for k, v in expected.items()):
        raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink input/output forbidden: " + str(path))
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("regular file required: " + str(path))
    return path


def verify_inventory(root, files):
    root = Path(root).resolve()
    if not files:
        raise ValueError("empty inventory")
    for name, digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe inventory path")
        path = regular(root / relative)
        if sha(path) != digest:
            raise ValueError("hash mismatch: " + name)


def package_check(root):
    root = Path(root).resolve()
    package = read(root / "PACKAGE.json")
    exact(
        package,
        {
            "round_id": "ROUND_001",
            "revision": REVISION,
            "profile": PROFILE,
            "execution_authorized": False,
        },
        "package profile mismatch",
    )
    verify_inventory(root, package["files"])
    validate_plan(read(root / "plan.yaml"))
    validate_cases(read(root / "cases.json")["cases"])
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(package["files"]) | {"PACKAGE.json"}:
        raise ValueError("unlisted package files")
    for module, prefix in (
        ("preprocessing_tools", "wrapper/preprocessing_tools"),
        ("preprocessing_core", "ai/preprocessing_core"),
        ("capstone_ai", "ai/src/capstone_ai"),
    ):
        source = Path(importlib.util.find_spec(module).origin).parent
        for p in source.glob("*.py"):
            if package["files"].get(prefix + "/" + p.name) != sha(p):
                raise ValueError("executing unverified source: " + str(p))
    if (
        sha(root / "plan.yaml") != package["plan_sha256"]
        or sha(root / "cases.json") != package["cases_sha256"]
    ):
        raise ValueError("plan binding mismatch")
    return package


def validate_plan(plan):
    exact(
        plan,
        {"round_id": "ROUND_001", "revision": REVISION, "storage_root": str(STORAGE)},
        "wrong R3 plan/storage",
    )
    exact(
        plan["model"],
        {
            "kind": "lstm",
            "seed": 17,
            "hidden": 32,
            "batch": 32,
            "precision": "float32",
            "backend": "aten_cuda",
            "max_epochs": 20,
            "patience": 5,
            "initialization": "fresh",
            "retries": 0,
            "baselines": list(MODELS[:2]),
        },
        "fixed model/baselines changed",
    )
    exact(
        plan["features"],
        {
            "rate_hz": 10,
            "history_steps": 100,
            "context_seconds": 60,
            "horizon_seconds": 1,
            "window_stride_seconds": 1,
        },
        "forecast contract changed",
    )
    exact(
        plan["gpu"],
        {
            "partition": "p02",
            "node": "DIS01",
            "nodes": 1,
            "tasks": 1,
            "gpus": 1,
            "jobs_max": 1,
            "main_wall_seconds": 10800,
            "queue_wait_seconds": 900,
            "modify_environment": False,
        },
        "GPU budget/resources changed",
    )
    exact(
        plan["budgets"],
        {
            "cpu_preparation_wall_seconds": 7200,
            "calibration_and_evaluation_shared_seconds": 7200,
            "external_additional_bytes_max": 80_000_000_000,
            "min_free_bytes": 50_000_000_000,
        },
        "finite CPU/storage budget changed",
    )
    exact(
        plan["calibration"],
        {
            "target_union_episodes_per_hour": 1,
            "quantiles": [0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999],
        },
        "normal-only finite calibration policy changed",
    )
    if plan["evaluation"].get("final_test_authorized") is not False:
        raise ValueError("final test forbidden")


def validate_cases(cases):
    if len(cases) != 50 or Counter(c["role"] for c in cases) != COUNTS:
        raise ValueError("R3 requires exactly 50 cases with roles9/9/9/11/12")
    normal = [c for c in cases if c["role"] != "anomaly_dev" and c["mission_seconds"] in {180, 900}]
    controls = [c for c in cases if c["role"] == "normal_dev_eval" and c["mission_seconds"] == 480]
    abnormal = [c for c in cases if c["role"] == "anomaly_dev"]
    if (len(normal), len(controls), len(abnormal)) != (36, 2, 12):
        raise ValueError("36 normal-condition/2 paired-control/12 anomaly cases required")
    if len({c["group_id"] for c in normal}) != 36 or len({c["seed"] for c in normal}) != 36:
        raise ValueError("normal-condition runs require36 independent groups/seeds")
    conditions = {
        "altitude_normal",
        "cold_normal",
        "horizontal_normal",
        "hover_normal",
        "low_soc_normal",
        "payload_normal",
        "waypoint_normal",
        "wind_normal",
        "yaw_normal",
    }
    for role in (*FIT_ROLES, "normal_dev_eval"):
        subset = [c for c in normal if c["role"] == role]
        names = {c.get("scenario") for c in subset}
        if len(subset) != 9 or len(names) != 9 or None in names or "low_soc_normal" not in names:
            raise ValueError("nine explicit independent normal conditions per split required")
        if conditions is not None and names != conditions:
            raise ValueError("normal-condition membership differs across splits")
        conditions = names
        if any(
            c["mission_seconds"] != (180 if c["scenario"] == "low_soc_normal" else 900)
            for c in subset
        ):
            raise ValueError("normal duration requires low_soc180/others900 seconds")
    if any(c["role"] != "normal_dev_eval" or c["mission_seconds"] != 480 for c in controls):
        raise ValueError("paired controls must be normal_dev_eval480s")
    families = Counter(c.get("scenario") for c in abnormal)
    if set(families) != {
        "battery_resistance_existing",
        "battery_resistance_ramp",
        "battery_resistance_step",
        "propulsion_loss_existing",
        "propulsion_loss_ramp",
        "propulsion_loss_step",
    } or set(families.values()) != {2}:
        raise ValueError("six fault families twice each required")
    for family in families:
        part = [c for c in abnormal if c["scenario"] == family]
        if len({c["seed"] for c in part}) != 2:
            raise ValueError("two distinct seeds per anomaly family required")
    if any(c["role"] != "anomaly_dev" or c["mission_seconds"] != 480 for c in abnormal):
        raise ValueError("anomaly480s required")


def validate_roles(rows, cases, protected):
    if len(rows) != 50 or len(cases) != 50 or Counter(r["role"] for r in rows) != COUNTS:
        raise ValueError("R3 requires exactly 9/9/9/11/12 cases, no test roles")
    validate_cases(cases)
    expected = {c["case_id"]: c for c in cases}
    if len(expected) != 50 or {r["case_id"] for r in rows} != set(expected):
        raise ValueError("case membership mismatch")
    for key in ("run_id", "raw_sha256", "raw_flight_id"):
        if len({r.get(key) for r in rows}) != 50 or any(not r.get(key) for r in rows):
            raise ValueError("missing/duplicate identity: " + key)
    forbidden = set().union(
        *(
            lineage(r) | ({"group:" + r["group_id"]} if r.get("group_id") else set())
            for r in protected
        )
    )
    seen = {}
    for row in rows:
        case = expected[row["case_id"]]
        for key in (
            "role",
            "seed",
            "scenario",
            "group_id",
            "mission_seconds",
        ):
            if key not in case or row.get(key) != case[key]:
                raise ValueError("planned case mismatch: " + key)
        exact(
            row,
            {
                "status": "completed",
                "origin": "px4_sitl",
                "continuity": "independent_reset",
                "previous_attempts": [],
            },
            "unfinished/continuation/retry/duration mismatch",
        )
        if row.get("parent_attempt_id") or row.get("continuation_id"):
            raise ValueError("retry or continuation forbidden")
        healthy = "abnormal" if row["role"] == "anomaly_dev" else "confirmed_healthy"
        if (
            row.get("health_eligibility") != healthy
            or not row.get("paired_case_id")
            or row.get("revision_lock", {}).get("battery_profile") != "VTC6_4S4P"
        ):
            raise ValueError("health/group identity missing")
        keys = lineage(row) | {"group:" + row["group_id"]}
        if keys & forbidden:
            raise ValueError("protected old/final lineage")
        partition = row["role"] if row["role"] in FIT_ROLES else "development"
        for key in keys:
            if key in seen and seen[key] != partition:
                raise ValueError("raw/paired/retry split leakage")
            seen[key] = partition
    controls = {r["paired_case_id"] for r in rows if r["role"] == "normal_dev_eval"}
    anomalies = Counter(r["paired_case_id"] for r in rows if r["role"] == "anomaly_dev")
    if len(anomalies) != 2 or set(anomalies.values()) != {6} or not set(anomalies) <= controls:
        raise ValueError("two paired normal controls and six anomalies per pair required")


def linked(record):
    path = regular(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError("linked manifest changed")
    return read(path)


def external(path, category):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("external path traversal forbidden")
    parent = STORAGE / category
    if not path.is_relative_to(parent) or path == parent:
        raise ValueError("persistent external category required: " + category)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink external path")
    if not os.path.ismount("/media/sa/X31") or not parent.is_dir():
        raise ValueError("external mount/category missing; no fallback creation")
    return path


def admission(root, approval_path, lock_path, stage, out):
    package = package_check(root)
    if Path(out).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("output must be outside immutable code package")
    if stage not in STAGES:
        raise ValueError("unknown stage; GPU gate not available")
    for path in (approval_path, lock_path):
        if Path(path).resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("external coordinator approval and freeze required")
    authorization, lock = read(regular(approval_path)), read(regular(lock_path))
    expected = dict(
        round_id="ROUND_001",
        revision=REVISION,
        stage=stage,
        execution_authorized=True,
        output=str(Path(out).resolve()),
        package_sha256=sha(Path(root) / "PACKAGE.json"),
        plan_sha256=package["plan_sha256"],
        profile=PROFILE,
    )
    exact(authorization, expected, "explicit stage approval mismatch")
    if not authorization.get("exact_message") or authorization.get("allowed_stages") != [stage]:
        raise ValueError("single stage START authorization required")
    exact(
        lock,
        {**expected, "status": "DATA_FROZEN", "approval_sha256": sha(approval_path)},
        "freeze/approval/output mismatch",
    )
    frozen, protected = linked(lock["frozen_manifest"]), linked(lock["protected_inventory"])
    exact(
        frozen,
        {"status": "DATA_FROZEN", "plan_sha256": package["plan_sha256"], "revision": REVISION},
        "wrong freeze revision",
    )
    if (
        protected.get("complete") is not True
        or not protected.get("runs")
        or not protected.get("deny_paths")
        or "R2_1GB" not in protected.get("includes_revisions", [])
    ):
        raise ValueError("complete protected identity inventory required")
    revision = read(Path(root) / "required-revision.json")
    if revision.get("fixture_only"):
        raise ValueError("synthetic fixture package cannot execute stages")
    if revision.get("battery_profile") != "VTC6_4S4P":
        raise ValueError("new VTC6_4S4P producer lock required")
    if frozen.get("revision_lock") != revision or any(
        r.get("revision_lock") != revision for r in frozen["runs"]
    ):
        raise ValueError("producer/firmware/profile/preprocessing lock mismatch")
    validate_roles(frozen["runs"], read(Path(root) / "cases.json")["cases"], protected["runs"])
    denied = [Path(p).resolve() for p in protected["deny_paths"]]
    for row in frozen["runs"]:
        path = Path(row["path"]).resolve()
        if not path.is_relative_to(STORAGE / "consumer") or any(
            path.is_relative_to(p) for p in denied
        ):
            raise ValueError("raw must be external and outside protected roots")
        if Path(row["run_id"]).name != row["run_id"] or row["run_id"] in {".", ".."}:
            raise ValueError("safe run ID required")
        if path != STORAGE / "consumer" / row["run_id"]:
            raise ValueError("explicit per-run external consumer destination required")
    expected_inputs = {"prepare": set(), "main": {"dataset"}, "calibrate": {"dataset", "trained"}}[
        stage
    ]
    if set(lock.get("inputs", {})) != expected_inputs:
        raise ValueError("stage input artifact set mismatch")
    if stage != "prepare":
        if "derived_audit" not in lock:
            raise ValueError("derived_audit required before main/calibration")
        quality = linked(lock["derived_audit"])
        exact(
            quality,
            {
                "status": "DERIVED_AUDIT_PASSED",
                "passed": True,
                "frozen_manifest_sha256": lock["frozen_manifest"]["sha256"],
                "prepared_artifact_sha256": lock["inputs"]["dataset"]["seal_sha256"],
                "train_only_scaler_verified": True,
                "lineage_passed": True,
                "memory_audit_passed": True,
            },
            "derived quality/scaler/window/RAM audit required",
        )
        if (
            quality.get("train_windows", 0) <= 0
            or quality.get("validation_windows", 0) <= 0
            or quality.get("calibration_windows", 0) <= 0
        ):
            raise ValueError("positive windows required in all fit roles")
    # Approval authorizes the exact freeze and artifact locations/hashes, not merely filenames.
    if authorization.get("lock_content") != {
        k: v for k, v in lock.items() if k != "approval_sha256"
    }:
        raise ValueError("approval does not bind exact freeze and input artifacts")
    if stage != "main":
        external(out, "processed" if stage == "prepare" else "calibration")
    elif not Path(out).resolve().is_relative_to(Path("/abr/coss43")):
        raise ValueError("approved user staging output required")
    return lock, frozen, denied


def claim(out, ledger, stage, binding):
    out, ledger = Path(out), Path(ledger)
    if stage not in STAGES:
        raise ValueError("unknown stage")
    if out.exists() or not out.parent.is_dir():
        raise ValueError("new output and existing parent required")
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger / (stage + ".claimed")).open("x") as f:
        import json

        json.dump({"stage": stage, "output": str(out.resolve()), **binding}, f)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(ledger, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    out.mkdir()


def seal(out, stage, binding, files):
    verify_inventory(out, files)
    result = dict(passed=True, stage=stage, revision=REVISION, files=files, **binding)
    write(Path(out) / "artifact.json", result)
    return result


def artifact(reference, stage, freeze_sha):
    path = Path(reference["path"]).resolve()
    regular(path / "artifact.json")
    if sha(path / "artifact.json") != reference["seal_sha256"]:
        raise ValueError("artifact seal changed")
    info = read(path / "artifact.json")
    exact(
        info,
        {
            "passed": True,
            "stage": stage,
            "revision": REVISION,
            "frozen_manifest_sha256": freeze_sha,
        },
        "artifact provenance mismatch",
    )
    verify_inventory(path, info["files"])
    return path, info


def ridge_memory(n, validation_n, dimension=1907, targets=40):
    if n <= 0 or validation_n <= 0 or dimension != 1907 or targets != 40:
        raise ValueError("positive 100x19+6+1 / 10x4 input required")
    return {
        "train_windows": n,
        "validation_windows": validation_n,
        "dtype": "float64",
        "flattened_dimensions": dimension,
        "target_dimensions": targets,
        "one_gram_bytes": 8 * n * n,
        "train_design_bytes": 8 * n * dimension,
        "validation_design_bytes": 8 * validation_n * dimension,
        "target_bytes": 8 * n * targets,
        "coefficient_bytes": 8 * dimension * targets,
        "four_gram_plus_arrays_planning_bytes": 32 * n * n
        + 8 * (n + validation_n) * dimension
        + 8 * n * targets
        + 8 * dimension * targets,
        "peak_rss_measured": False,
        "caveat": "Planning subtotal, not a peak bound: includes four Gram-sized arrays; LAPACK copies/workspace, normalized arrays and Python/runtime add memory. Solver unchanged.",
    }


def returned_model(reference, freeze_sha, package_sha):
    path, info = artifact(reference, "main", freeze_sha)
    external(path, "models")
    for name, key in (("release.json", "release_sha256"), ("exit.json", "exit_sha256")):
        if sha(regular(path / name)) != reference.get(key):
            raise ValueError("returned release/exit evidence hash required")
    release, exit_record = read(path / "release.json"), read(path / "exit.json")
    job = regular(path / "verified-job-id.txt").read_text().strip()
    if (
        release.get("passed") is not True
        or release.get("released") is not True
        or not job.isdigit()
        or release.get("job_id") != job
        or exit_record.get("exit_code") != 0
        or release.get("artifact_sha256") != sha(path / "artifact.json")
        or info["package_sha256"] != package_sha
    ):
        raise ValueError("verified returned/released same-package models required")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    checked = package_check(args.root)
    print("Verified code package files:", len(checked["files"]), "Execution not authorized.")
