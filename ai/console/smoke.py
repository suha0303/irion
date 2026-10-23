"""Finite real-data save/restore gate; no main training or automatic retry.

The public worker is CUDA-only and verifies its existing Slurm allocation before
importing training code. CPU execution is used only by explicit synthetic tests.
"""

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from execution.common import Progress, child, read, sha, write
from execution.prepare import npz_headers

MODELS = ("lstm", "transformer")
PHASES = ("full", "partial", "resume")
ROLES = {
    "normal_train",
    "normal_validation",
    "normal_calibration",
    "normal_dev_eval",
    "anomaly_dev",
}
SAMPLE_WINDOWS = 4
BATCH_SIZE = 2


def _prefix(path, key, count, headers):
    """Read a bounded numeric/string prefix rather than decompressing a whole role."""
    header = headers[key]
    shape, dtype = header["shape"], np.dtype(header["descr"])
    if not shape or shape[0] < count or dtype.hasobject or dtype.kind not in "fiuUS":
        raise ValueError("Unsupported or insufficient smoke input: " + key)
    size = count * math.prod(shape[1:]) * dtype.itemsize
    if size > 4 * 1024**2:
        raise ValueError("Smoke prefix exceeds the bounded sample budget")
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }.get(version)
        if reader is None:
            raise ValueError("Unsupported smoke NPY version")
        actual_shape, fortran, actual_dtype = reader(stream)
        if actual_shape != shape or fortran or actual_dtype != dtype:
            raise ValueError("Smoke header changed")
        body = stream.read(size)
        if len(body) != size:
            raise ValueError("Truncated smoke prefix")
    return np.frombuffer(body, dtype=dtype).reshape(count, *shape[1:]).copy()


def _current_evidence(upstream, *, flat=False):
    """Keep full handoff/quality admission intact for current extracted packages."""
    dataset = child(upstream, "dataset")
    evidence = upstream if flat else child(upstream, "evidence")
    prefix = "" if flat else "evidence/"
    handoff = read(child(evidence, "handoff.json"))
    used = {}
    for name, field in (
        ("quality.json", "quality_sha256"),
        ("dataset-seal.json", "dataset_seal_sha256"),
        ("frozen-manifest.json", "frozen_manifest_sha256"),
    ):
        actual = sha(child(evidence, name))
        if actual != handoff.get(field):
            raise ValueError("Handoff evidence changed: " + name)
        used[prefix + name] = actual
    used[prefix + "handoff.json"] = sha(evidence / "handoff.json")
    quality, frozen, seal = [
        read(evidence / n) for n in ("quality.json", "frozen-manifest.json", "dataset-seal.json")
    ]
    if (
        handoff.get("scaler_scope") != "all_selected_normal_train"
        or quality.get("passed") is not True
        or quality.get("status") != "PRETRAIN_QUALITY_PASSED"
        or quality.get("final_test_used") is not False
        or quality.get("train_only_scaler_verified") is not True
    ):
        raise ValueError("Verified normal-only train-scaler evidence is required")
    benchmark = (
        handoff.get("status") == "BENCHMARK_ONLY_STOPPED"
        and frozen.get("status") == "BENCHMARK_ONLY"
    )
    if benchmark:
        _benchmark_membership(handoff, frozen, quality)
    elif handoff.get("training_eligible") is not True or frozen.get("status") != "DATA_FROZEN":
        raise ValueError("Verified normal-only train-scaler evidence is required")
    manifest_path = child(dataset, "manifest.json")
    if sha(manifest_path) != seal.get("manifest.json"):
        raise ValueError("Dataset manifest changed")
    return (
        seal,
        read(manifest_path),
        frozen.get("runs"),
        handoff.get("run_count"),
        used,
        dict(
            kind="fresh_three_run_benchmark_functionality_only"
            if benchmark
            else "full_handoff_evidence",
            quality_evidence_reverified=True,
            training_eligible=False,
            source_training_eligible=handoff.get("training_eligible"),
            main_training_admission=False,
            lineage_scope="benchmark case/group/seed/run IDs"
            if benchmark
            else "full split lineage",
            note=(
                "Fresh 120-second benchmark flights are test-only processing partitions. "
                "Independent numeric quality and train-only scaler evidence are checked, "
                "but these flights do not establish main-training admission, data diversity "
                "or detection performance. Original paired/raw-flight lineage fields are "
                "not fabricated; benchmark case/group/seed/run IDs are checked instead."
            )
            if benchmark
            else "Full handoff evidence is used only for a tiny correctness gate.",
        ),
    )


def _benchmark_membership(handoff, frozen, quality):
    """Admit only three fresh independent test partitions, never production data."""
    rows = frozen.get("runs", [])
    if (
        handoff.get("training_eligible") is not False
        or frozen.get("training_eligible") is not False
        or handoff.get("training_authorized") is not False
        or handoff.get("requires_new_training_package_admission") is not True
        or quality.get("training_admission") is not False
        or handoff.get("run_count") != 3
        or len(rows) != 3
        or Counter(row.get("role") for row in rows)
        != {"normal_train": 1, "normal_validation": 1, "normal_calibration": 1}
        or not re.fullmatch(r"[0-9a-f]{64}", str(frozen.get("ledger_sha256", "")))
    ):
        raise ValueError("Fresh three-run benchmark functionality-only evidence required")
    for key in ("seed", "group_id", "case_id", "run_id"):
        values = [row.get(key) for row in rows]
        if any(value is None or value == "" for value in values) or len(set(values)) != 3:
            raise ValueError("Benchmark identities must be independent: " + key)
    if any(
        row.get("collection_role") != "benchmark"
        or row.get("training_eligible") is not False
        or row.get("mission_seconds") != 120
        or type(row.get("seed")) is not int
        for row in rows
    ):
        raise ValueError("Benchmark run scope changed")


def _legacy_snapshot(upstream):
    """Functionality gate only for an already admitted, sealed 104-run snapshot.

    Its historical handoff path is never followed or rewritten. Missing original
    quality files are disclosed; this adapter cannot admit any main training.
    """
    dataset = child(upstream, "dataset")
    inventory_path, config_path = child(upstream, "inventory.json"), child(upstream, "config.json")
    inventory = read(inventory_path)
    if not isinstance(inventory, dict) or sha(config_path) != inventory.get("config.json"):
        raise ValueError("Legacy snapshot config inventory changed")
    cfg = read(config_path)
    if (
        cfg.get("revision") != "ROUND_001/R4_GPU104_FOUR_SEED17"
        or cfg.get("final_test_used") is not False
        or cfg.get("final_test_authorized") is not False
        or not isinstance(cfg.get("handoff"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(cfg.get("handoff_sha256", "")))
        or not isinstance(cfg.get("dataset_files"), dict)
    ):
        raise ValueError("Unsupported legacy functionality-test snapshot schema")
    seal = cfg["dataset_files"]
    used = {"inventory.json": sha(inventory_path), "config.json": sha(config_path)}
    for name in ("manifest.json", "normal_train.npz", "normal_validation.npz", "scaler.npz"):
        expected = seal.get(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Missing legacy dataset seal: " + name)
        if inventory.get("dataset/" + name) != expected:
            raise ValueError("Legacy inventory/config data seal mismatch: " + name)
    manifest_path = child(dataset, "manifest.json")
    if sha(manifest_path) != seal["manifest.json"]:
        raise ValueError("Legacy dataset manifest changed")
    manifest = read(manifest_path)
    expected_counts = dict(
        normal_train=63,
        normal_validation=9,
        normal_calibration=9,
        normal_dev_eval=11,
        anomaly_dev=12,
    )
    if (
        not isinstance(manifest.get("runs"), list)
        or Counter(row.get("role") for row in manifest["runs"]) != expected_counts
        or manifest.get("scaler_fit")
        != ("normal_train unique valid airborne frames; context once per train origin")
        or type(manifest.get("unique_train_frames")) is not int
        or manifest["unique_train_frames"] <= 0
    ):
        raise ValueError("Legacy 104-run membership/train-scaler policy changed")
    return (
        seal,
        manifest,
        manifest["runs"],
        104,
        used,
        dict(
            kind="legacy_admitted_104_snapshot_functionality_only",
            revision=cfg["revision"],
            original_handoff_sha256=cfg["handoff_sha256"],
            quality_evidence_reverified=False,
            training_eligible=False,
            note=(
                "Original admission handoff is recorded by hash but its full quality evidence "
                "is not present in this historical server snapshot. Only its inventory, config, "
                "split lineage, archive/scaler seals and tiny normal samples are checked. "
                "This does not establish current main-training admission or detection quality."
            ),
        ),
    )


def sample_input(upstream, destination):
    """Verify sealed normal-only input; persist four windows per fit role.

    This is a correctness gate with a deliberately tiny cached sample. It is not
    a training-throughput measurement, memory maximum, or detection evaluation.
    """
    upstream = Path(upstream).expanduser().absolute()
    if not upstream.is_dir() or any(p.is_symlink() for p in (upstream, *upstream.parents)):
        raise ValueError("A real extracted upstream directory is required")
    dataset, evidence = child(upstream, "dataset"), child(upstream, "evidence")
    if evidence.exists():
        seal, manifest, expected_rows, count, used, provenance = _current_evidence(upstream)
    elif child(upstream, "handoff.json").is_file():
        seal, manifest, expected_rows, count, used, provenance = _current_evidence(
            upstream, flat=True
        )
    else:
        seal, manifest, expected_rows, count, used, provenance = _legacy_snapshot(upstream)
    rows = manifest.get("runs", [])
    if (
        not rows
        or rows != expected_rows
        or len(rows) != count
        or len({r["run_id"] for r in rows}) != len(rows)
        or any(r.get("role") not in ROLES for r in rows)
    ):
        raise ValueError("Frozen membership/roles differ; final test is not admitted")
    partitions = {}
    for row in rows:
        part = (
            row["role"]
            if row["role"] in {"normal_train", "normal_validation", "normal_calibration"}
            else "development"
        )
        keys = ("seed", "group_id", "paired_case_id", "raw_flight_id")
        if provenance["kind"] == "fresh_three_run_benchmark_functionality_only":
            keys = ("seed", "group_id", "case_id", "run_id")
        for key in keys:
            value = row.get(key)
            if value is None:
                raise ValueError("Missing split lineage: " + key)
            token = key, str(value)
            if token in partitions and partitions[token] != part:
                raise ValueError("Cross-split lineage leakage")
            partitions[token] = part
    used["dataset/manifest.json"] = seal["manifest.json"]
    scaler_path = child(dataset, "scaler.npz")
    if sha(scaler_path) != seal.get("scaler.npz"):
        raise ValueError("Train-only scaler changed")
    used["dataset/scaler.npz"] = seal["scaler.npz"]
    with np.load(scaler_path, allow_pickle=False) as archive:
        scaler = {
            key + suffix: archive[key + suffix]
            for key in ("x", "c", "y")
            for suffix in ("_mean", "_scale")
        }
    for key, width in (("x", 19), ("c", 6), ("y", 4)):
        for suffix in ("_mean", "_scale"):
            value = scaler[key + suffix]
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError("Invalid scaler dimensions/values")
        if (scaler[key + "_scale"] <= 0).any():
            raise ValueError("Nonpositive scaler scale")
    sampled, details = {}, {}
    for prefix, role in (("train", "normal_train"), ("validation", "normal_validation")):
        path = child(dataset, role + ".npz")
        actual = sha(path)
        if actual != seal.get(path.name) or actual != manifest.get("hashes", {}).get(path.name):
            raise ValueError("Fit archive changed: " + role)
        used["dataset/" + path.name] = actual
        headers = npz_headers(path)
        n = headers["X"]["shape"][0]
        if any(
            headers[k]["shape"] != (n, *shape)
            for k, shape in (("X", (100, 19)), ("C", (6,)), ("Y", (10, 4)))
        ):
            raise ValueError("Forecast shape mismatch")
        run_ids = _prefix(path, "run_id", SAMPLE_WINDOWS, headers)
        allowed = {r["run_id"] for r in rows if r["role"] == role}
        if not allowed or any(str(run_id) not in allowed for run_id in run_ids):
            raise ValueError("Sample run membership mismatch")
        for key, scale in (("X", "x"), ("C", "c"), ("Y", "y")):
            if np.dtype(headers[key]["descr"]).kind != "f":
                raise ValueError("Floating forecast arrays required")
            block = _prefix(path, key, SAMPLE_WINDOWS, headers)
            normalized = (block - scaler[scale + "_mean"]) / scaler[scale + "_scale"]
            if not np.isfinite(normalized).all():
                raise ValueError("Nonfinite normalized fit sample")
            sampled[prefix + "_" + key] = normalized
        details[role] = dict(
            source_windows=n,
            sampled_windows=SAMPLE_WINDOWS,
            sampling="first four windows, bounded NPZ prefix stream",
            run_ids=sorted({str(value) for value in run_ids}),
        )
    np.savez(destination, **sampled)
    return dict(
        upstream=str(upstream),
        provenance=provenance,
        training_eligible=False,
        hashes=used,
        roles=details,
        final_test_used=False,
        truth_used=False,
        scaler_fitted=False,
        scaler_scope="all_selected_normal_train",
        throughput_measured=False,
        limitation="Tiny cached prefix is only a correctness gate; not representative performance",
    )


def _source_hashes():
    base = Path(__file__).resolve().parents[1]
    return {
        name: sha(child(base, name))
        for name in (
            "console/smoke.py",
            "console/profile.py",
            "execution/common.py",
            "remote/gpu_probe.py",
            "learning/models.py",
            "learning/training.py",
            "learning/telemetry.py",
            "pipeline/runtime.py",
            "src/capstone_ai/training.py",
        )
    }


def _verify_runtime(profile, output):
    # Allocation validation precedes the training module and any CUDA operation.
    from pipeline.runtime import check

    os.environ["DRONE_AI_PROFILE"] = str(Path(profile).absolute())
    return check(Path(__file__).resolve().parents[1], output, "benchmark", job_prefix="drone-ai-")


def _stop_signal(signum, frame):
    raise RuntimeError(f"Received signal {signum}; preserve smoke evidence and stop")


def _child(root, model, phase, profile=None):
    root = Path(root)
    cfg = read(root / "smoke.json")
    if model not in MODELS or phase not in PHASES or cfg.get("schema") != "drone-ai.smoke/1":
        raise ValueError("Unknown finite smoke phase")
    device = cfg["device"]
    if device == "cuda":
        if profile is None or sha(profile) != cfg["profile_sha256"]:
            raise ValueError("Smoke profile changed")
        _verify_runtime(profile, root)
    elif device != "cpu" or cfg.get("synthetic_cpu_fixture") is not True:
        raise ValueError("CPU smoke is restricted to explicit synthetic unit fixtures")
    if (
        sha(root / "inputs.npz") != cfg["inputs_sha256"]
        or _source_hashes() != cfg["binding"]["sources"]
    ):
        raise ValueError("Smoke input/source binding changed")
    import torch

    from learning.models import ModelSpec
    from learning.training import run_fit

    with np.load(root / "inputs.npz", allow_pickle=False) as archive:
        train = {k: archive["train_" + k] for k in ("X", "C", "Y")}
        validation = {k: archive["validation_" + k] for k in ("X", "C", "Y")}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_fit(
        ModelSpec(model, 32, 2),
        train,
        validation,
        root / f"{model}-{phase}",
        binding=cfg["binding"],
        device=device,
        gpu_authorized=device == "cuda",
        runtime_stage="benchmark",
        seed=0,
        batch_size=BATCH_SIZE,
        epochs=2,
        patience=3,
        learning_rate=0.001,
        stop_after_epoch=1 if phase == "partial" else None,
        resume=root / f"{model}-partial/last.pt" if phase == "resume" else None,
        progress=Progress(root / f"{model}-{phase}-progress.json", hours=0.15),
    )
    write(
        root / f"{model}-{phase}-memory.json",
        dict(
            device=device,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None,
            optimizer_steps=2 if phase != "full" else 4,
            discarded_smoke_weights=True,
            training_eligible=False,
        ),
    )


def restore_gate(root, cfg, progress, *, profile=None):
    """Six sequential fresh processes; compare exact continuation, not only loading."""
    root = Path(root)
    proofs = {}
    for index, model in enumerate(MODELS):
        for phase in PHASES:
            progress.check()
            progress.update(
                stage="save_restore_gate",
                model=model,
                phase=phase,
                done=index * 3 + PHASES.index(phase),
                total=6,
            )
            argv = [
                sys.executable,
                "-B",
                "-m",
                "console.smoke",
                "child",
                "--root",
                str(root),
                "--model",
                model,
                "--phase",
                phase,
            ]
            if profile is not None:
                argv += ["--profile", str(profile)]
            with (root / f"{model}-{phase}.log").open("x") as log:
                subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=480)
        import torch

        from learning.diagnostic import same

        full, resumed = [
            torch.load(root / f"{model}-{phase}/last.pt", map_location="cpu", weights_only=False)
            for phase in ("full", "resume")
        ]
        checks = {
            key: same(full[key], resumed[key])
            for key in (
                "model",
                "optimizer",
                "rng",
                "best",
                "best_epoch",
                "best_state",
                "next_epoch",
                "config",
                "binding",
                "environment",
            )
        }
        checks["loss_history"] = all(
            [row[key] for row in full["history"]] == [row[key] for row in resumed["history"]]
            for key in ("train_loss", "validation_loss")
        )
        checks["safetensors_export"] = sha(root / f"{model}-full/model.safetensors") == sha(
            root / f"{model}-resume/model.safetensors"
        )
        checks["completed_two_epochs"] = full["next_epoch"] == resumed["next_epoch"] == 2
        states = list(full["optimizer"]["state"].values())
        checks["optimizer_executed_four_steps"] = bool(states) and all(
            float(state["step"]) == 4 for state in states
        )
        checks["finite_losses"] = all(
            math.isfinite(row[key])
            for row in full["history"]
            for key in ("train_loss", "validation_loss")
        )
        proofs[model] = dict(
            checks=checks,
            optimizer_steps_executed=8,
            memory={phase: read(root / f"{model}-{phase}-memory.json") for phase in PHASES},
            full_checkpoint_sha256=sha(root / f"{model}-full/last.pt"),
            resumed_checkpoint_sha256=sha(root / f"{model}-resume/last.pt"),
        )
        write(root / "restore-proof.json", proofs)
        if not all(checks.values()):
            raise ValueError(
                model + " exact save/restore continuation mismatch; stop without retry"
            )
    return proofs


def run(profile, root):
    """CUDA worker; controller alone declares final success after Slurm release."""
    from console.profile import load

    profile = Path(profile).expanduser().absolute()
    settings = load(profile)
    root = Path(root).expanduser().absolute()
    if any(p.is_symlink() for p in (root, *root.parents)) or ".." in root.parts:
        raise ValueError("A new real output directory is required")
    upstream = Path(settings["dataset_root"])
    expected_project = Path(__file__).resolve().parents[2]
    if Path(settings["project_root"]) != expected_project:
        raise ValueError("Run smoke from the explicitly configured server clone")
    output = Path(settings["output_root"])
    if root == output or not root.is_relative_to(output):
        raise ValueError("Use a new attempt directory below the configured output_root")
    if root.is_relative_to(upstream) or upstream.is_relative_to(root):
        raise ValueError("Smoke output must be separate from immutable data")
    root.mkdir(parents=True, exist_ok=False)
    progress = Progress(root / "progress.json", hours=0.5)
    started = time.monotonic()
    previous_handlers = {
        number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)
    }
    signal.signal(signal.SIGTERM, _stop_signal)
    signal.signal(signal.SIGINT, _stop_signal)
    try:
        progress.update(
            stage="allocation_verification",
            training_eligible=False,
            production_training_authorized=False,
        )
        _verify_runtime(profile, root)
        progress.update(stage="sealed_normal_sample", done=0, total=1)
        source = sample_input(upstream, root / "inputs.npz")
        write(root / "sample-evidence.json", source)
        cfg = dict(
            schema="drone-ai.smoke/1",
            device="cuda",
            synthetic_cpu_fixture=False,
            profile_sha256=sha(profile),
            inputs_sha256=sha(root / "inputs.npz"),
            production_training_authorized=False,
            training_eligible=False,
            models=list(MODELS),
            hidden=32,
            layers=2,
            batch_size=BATCH_SIZE,
            seed=0,
            epochs=2,
            phases=list(PHASES),
            retries=0,
            precision="float32",
            amp=False,
            final_test_used=False,
            binding=dict(
                data=source["hashes"],
                scaler=source["hashes"]["dataset/scaler.npz"],
                sources=_source_hashes(),
            ),
        )
        write(root / "smoke.json", cfg)
        from learning.telemetry import Telemetry

        telemetry = Telemetry(root)
        telemetry.start()  # Allocation was checked before this PID creates any CUDA context.
        try:
            proofs = restore_gate(root, cfg, progress, profile=profile)
        finally:
            telemetry.stop()
        result = dict(
            status="SMOKE_COMPUTE_FINISHED_PENDING_RELEASE",
            passed=True,
            production_training_authorized=False,
            training_eligible=False,
            final_test_used=False,
            truth_used=False,
            automatic_retry=False,
            llm_calls=0,
            gpu_allocation_released="controller must confirm actual Slurm termination",
            elapsed_s=time.monotonic() - started,
            optimizer_steps_total=16,
            sample=source,
            proofs=proofs,
            scope="LSTM/Transformer hidden32 layers2 batch2 seed0; save and exact fresh-process restore only",
            excludes=[
                "main training",
                "calibration",
                "detection evaluation",
                "throughput benchmark",
            ],
        )
        write(root / "result.json", result)
        progress.update(status=result["status"], stage="await_allocation_release", done=6, total=6)
        return result
    except BaseException as exc:
        result = dict(
            status="FAILED_STOPPED",
            passed=False,
            reason=f"{type(exc).__name__}: {exc}",
            automatic_retry=False,
            production_training_authorized=False,
            training_eligible=False,
            final_test_used=False,
            llm_calls=0,
            elapsed_s=time.monotonic() - started,
        )
        write(root / "result.json", result)
        progress.update(**result)
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["run", "child"], default="run")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--phase", choices=PHASES)
    args = parser.parse_args()
    if args.command == "child":
        _child(args.root, args.model, args.phase, args.profile)
    elif args.profile is None:
        parser.error("--profile is required for an allocated CUDA smoke worker")
    else:
        run(args.profile, args.root)


if __name__ == "__main__":
    main()
