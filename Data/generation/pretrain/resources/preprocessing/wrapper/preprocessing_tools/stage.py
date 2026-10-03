"""Authorized finite workers. Reuse the unchanged pipeline, never join event truth."""

import argparse
import os
import sys
from pathlib import Path

from preprocessing_core.control import MODELS, Budget, read, sha, write

from .control import (
    DATA_FILES,
    LEDGER,
    RAW_FILES,
    STORAGE,
    admission,
    artifact,
    external,
    linked,
    regular,
    returned_model,
    ridge_memory,
    seal,
    verify_inventory,
)


def read_guard(denied, raw_roots, stage, dataset=None):
    denied = [Path(p).resolve() for p in denied]
    raw_roots = [Path(p).resolve() for p in raw_roots]

    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name in {"test.npz", "events.jsonl"} or path.suffix in {
            "tar",
            ".tar",
            ".gz",
            ".xz",
            ".zip",
        }:
            raise PermissionError("test/truth/full archive access denied")
        if any(path.is_relative_to(p) for p in denied):
            raise PermissionError("protected old/final access denied")
        for raw in raw_roots:
            if path.is_relative_to(raw) and (
                stage != "prepare" or str(path.relative_to(raw)) not in RAW_FILES
            ):
                raise PermissionError("raw whitelist access denied")
        if stage == "main" and dataset is not None:
            if (
                path.is_relative_to(dataset / "canonical")
                or path == dataset / "normal_calibration.npz"
            ):
                raise PermissionError("training may only read train/validation arrays")

    sys.addaudithook(guard)


def tree_files(path):
    return {
        str(p.relative_to(path)): sha(regular(p))
        for p in sorted(Path(path).rglob("*"))
        if p.is_file()
    }


def prepare(frozen, out, preprocess, budget):
    for row in frozen["runs"]:
        budget.check()
        external(row["path"], "consumer")
        if set(row["files"]) != RAW_FILES:
            raise ValueError("six regular consumer files required; full tar/truth separate")
        verify_inventory(row["path"], row["files"])
        if row["files"]["observations/streams.jsonl"] != row["raw_sha256"]:
            raise ValueError("raw stream identity mismatch")
        run = read(Path(row["path"]) / "run.json")
        if any(run.get(k) != row[k] for k in ("run_id", "origin", "status")):
            raise ValueError("raw run identity mismatch")
    from preprocessing_core.pipeline import prepare as original_prepare

    dataset = original_prepare(frozen, out / "dataset", preprocess, budget)
    for row in frozen["runs"]:
        verify_inventory(row["path"], row["files"])
    canonical = tree_files(dataset / "canonical")
    write(dataset / "canonical-seal.json", canonical)
    from collections import Counter

    counts = Counter()
    for item in read(dataset / "manifest.json")["audit"]:
        if item["accepted_targets"] is not None:
            counts[item["role"]] += item["accepted_targets"]
    write(
        out / "ridge-memory.json", ridge_memory(counts["normal_train"], counts["normal_validation"])
    )
    return {
        "dataset/" + n: sha(dataset / n)
        for n in (DATA_FILES - {"normal_calibration.npz"}) | {"canonical-seal.json"}
    } | {"ridge-memory.json": sha(out / "ridge-memory.json")}


def train(dataset, out, preprocess, budget):
    from preprocessing_core.pipeline import bundles

    bundles(dataset, out / "models", preprocess, budget, lstm_backend="aten_cuda")
    # neural_fit already compares exported/restored predictions on the same CUDA device.
    result = read(out / "models/lstm/fit.json")
    if result["save_load_max_abs"] > 1e-6 or not result["history"]:
        raise ValueError("LSTM save/restore evidence missing")
    for name in MODELS:
        folder = out / "models" / name
        verify_inventory(folder, read(folder / "integrity.json"))
    return {"models/" + n: digest for n, digest in tree_files(out / "models").items()}


def calibrate(dataset, models, out, budget):
    from preprocessing_core.pipeline import calibrate_and_seal

    verify_inventory(dataset / "canonical", read(dataset / "canonical-seal.json"))
    calibrate_and_seal(dataset, models, out / "sealed", budget)
    seals = read(out / "sealed/all-forecast-seals.json")
    if len(seals) != 69:  # 3 models x (11 normal dev + 12 anomaly dev), none excluded.
        raise ValueError("all 69 model/run seals required, unavailable cases retained")
    verify_inventory(out / "sealed", seals)
    write(
        out / "evaluator-handoff.json",
        {
            "truth_join": "coordinator_only_not_executed",
            "models": list(MODELS),
            "expected_model_run_seals": 69,
            "sealed_root": str(out / "sealed"),
            "all_forecast_seals_sha256": sha(out / "sealed/all-forecast-seals.json"),
            "unavailable_cases": "retain in denominator; never convert missing scores to zero",
            "evaluation_minutes_cap": 120,
            "shared_calibration_evaluation_seconds": 7200,
            "evaluator_time_rule": "Coordinator must subtract calibration exit.json wall_including_queue_s from 7200; no fresh 120-minute allowance. Stop if no budget remains.",
            "final_test_authorized": False,
        },
    )
    return {"sealed/" + n: digest for n, digest in tree_files(out / "sealed").items()} | {
        "evaluator-handoff.json": sha(out / "evaluator-handoff.json")
    }


def run(args):
    root, out = args.root.resolve(), args.out.resolve()
    lock, frozen, denied = admission(root, args.approval, args.lock, args.stage, out)
    binding = {
        "package_sha256": sha(root / "PACKAGE.json"),
        "approval_sha256": sha(args.approval),
        "lock_sha256": sha(args.lock),
        "frozen_manifest_sha256": lock["frozen_manifest"]["sha256"],
    }
    ledger = LEDGER if args.stage == "main" else STORAGE / "metadata/ai-claims"
    claim = read(ledger / (args.stage + ".claimed"))
    if claim != {"stage": args.stage, "output": str(out), **binding}:
        raise ValueError("durable once-only stage claim mismatch")
    (out / "worker.started").open("x").close()
    dataset = None
    if args.stage != "prepare":
        prepared, info = artifact(
            lock["inputs"]["dataset"], "prepare", binding["frozen_manifest_sha256"]
        )
        if info["package_sha256"] != binding["package_sha256"]:
            raise ValueError("prepared artifact from different code package")
        dataset = prepared / "dataset"
        manifest = read(dataset / "manifest.json")
        if manifest["runs"] != frozen["runs"]:
            raise ValueError("prepared dataset membership differs from frozen raw runs")
        quality = linked(lock["derived_audit"])
        for role, field in (
            ("normal_train", "train_windows"),
            ("normal_validation", "validation_windows"),
            ("normal_calibration", "calibration_windows"),
        ):
            count = sum(row["accepted_targets"] for row in manifest["audit"] if row["role"] == role)
            if quality[field] != count:
                raise ValueError("derived audit window count differs from prepared artifact")
        if args.stage == "calibrate":
            external(prepared, "processed")
    budget = Budget(
        out if args.stage == "main" else STORAGE,
        (180 if args.stage == "main" else 120) * 60,
        20_000_000_000 if args.stage == "main" else 80_000_000_000,
    )
    preprocess = read(root / "preprocessing.json")
    if args.stage == "prepare":
        from .archive import extract_consumer

        evidence = {}
        for row in frozen["runs"]:
            evidence[row["run_id"]] = extract_consumer(row, budget)
        write(out / "consumer-extraction.json", evidence)
    read_guard(denied, [r["path"] for r in frozen["runs"]], args.stage, dataset)
    try:
        if args.stage == "prepare":
            files = prepare(frozen, out, preprocess, budget)
        elif args.stage == "main":
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
                raise ValueError("CUBLAS workspace profile mismatch")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from preprocessing_core.runtime import check

            environment = check(root, out, "main", job_prefix="x500-r3-long15-")
            from .memory import allocated_memory_gate

            memory = allocated_memory_gate(quality, environment)
            write(out / "actual-CPU-memory.json", memory)
            files = train(dataset, out, preprocess, budget)
            files["actual-CPU-memory.json"] = sha(out / "actual-CPU-memory.json")
            if os.environ.get("CUDA_VISIBLE_DEVICES") != environment["CUDA_VISIBLE_DEVICES"]:
                raise ValueError("Slurm CUDA_VISIBLE_DEVICES changed")
            files["environment.json"] = sha(out / "environment.json")
        else:
            trained = returned_model(
                lock["inputs"]["trained"],
                binding["frozen_manifest_sha256"],
                binding["package_sha256"],
            )
            files = calibrate(dataset, trained / "models", out, budget)
        budget.check()
        from .control import package_check

        package_check(root)
        if args.stage != "prepare":
            artifact(lock["inputs"]["dataset"], "prepare", binding["frozen_manifest_sha256"])
        if args.stage == "calibrate":
            artifact(lock["inputs"]["trained"], "main", binding["frozen_manifest_sha256"])
            verify_inventory(dataset / "canonical", read(dataset / "canonical-seal.json"))
        budget.check()
        seal(out, args.stage, binding, files)
    except Exception as error:
        write(out / "failure.json", {"status": "STOPPED", "error": str(error), "retry": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "out", "approval", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "main", "calibrate"), required=True)
    run(parser.parse_args())
