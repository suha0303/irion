"""Versioned 1200-run comparison contract and bounded host-side normalization."""

import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from execution.common import read, sha, write
from execution.prepare import npz_headers

REVISION = "ROUND_001/R6_1200_8MODEL_S0"
COUNTS = dict(
    normal_train=720,
    normal_validation=104,
    normal_calibration=104,
    normal_dev_eval=128,
    anomaly_dev=144,
)
VARIANTS = [
    dict(id=f"{name}_h{hidden}_l{layers}_s0", name=name, hidden=hidden, layers=layers, heads=heads)
    for hidden, layers, heads in ((64, 2, 4), (128, 2, 4), (128, 4, 4), (256, 4, 8))
    for name in ("lstm", "transformer")
]
MODELS = ["persistence", "ridge_forecast", *[v["id"] for v in VARIANTS]]


def is_comparison(cfg):
    from .study import REVISION as STUDY

    return cfg.get("revision") in {REVISION, STUDY}


def execution_scope(cfg):
    """Absent scope preserves sealed legacy comparisons; new scopes are explicit."""
    scope = cfg.get("execution_scope", "full_comparison")
    if scope not in {"full_comparison", "training_only"}:
        raise ValueError("Unknown comparison execution scope")
    if scope == "training_only":
        if not is_comparison(cfg):
            raise ValueError("Training-only scope requires the 1200-run comparison")
        if (
            cfg.get("calibration_authorized") is not False
            or cfg.get("evaluation_authorized") is not False
        ):
            raise ValueError(
                "Training-only scope must explicitly prohibit calibration and evaluation"
            )
    elif any(cfg.get(key) is False for key in ("calibration_authorized", "evaluation_authorized")):
        raise ValueError("Full comparison requires its post-training phases to be authorized")
    return scope


def config_check(cfg, *, require_approval=True):
    scope = execution_scope(cfg)
    fixed = dict(
        revision=REVISION,
        variants=VARIANTS,
        models=MODELS,
        seed=0,
        batch_size=256,
        epochs=60,
        early_stopping_patience=10,
        learning_rate=0.001,
        dropout=0.1,
        weight_decay=0.00001,
        model_revision="explicit_dropout_v1",
        history=100,
        future=10,
        step_us=100000,
        precision="float32",
        amp=False,
        retries=0,
        final_test_used=False,
        gate_epochs=4,
        gate_windows=512,
        memory_gib=64,
        cpus=4,
        training_stride_us=1000000,
        next_round_authorized=False,
        models_sequential=True,
        gpu_jobs_max=1,
        development_counts=dict(normal_dev_eval=128, anomaly_dev=144),
    )
    from . import study

    if cfg.get("revision") == study.REVISION:
        fixed = study.configure(fixed)
        if scope != "training_only" or "dropout" in cfg:
            raise ValueError("The new study is training-only and has no Dropout setting")
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("R6 comparison configuration mismatch")
    if type(cfg.get("production_training_authorized")) is not bool:
        raise ValueError("Explicit training authorization required")
    if require_approval and not cfg["production_training_authorized"]:
        raise ValueError("R6 production training has not been approved")
    for key in ("wall_hours", "queue_hours"):
        if type(cfg.get(key)) is not int or not 1 <= cfg[key] <= 168:
            raise ValueError("Explicit finite hours budget required: " + key)
    evaluation_hours = cfg.get("evaluation_hours")
    if type(evaluation_hours) is not int or (
        evaluation_hours != 0 if scope == "training_only" else not 1 <= evaluation_hours <= 168
    ):
        raise ValueError("Evaluation budget must be zero for training-only, otherwise within1..168")
    if cfg.get("memory", {}).get("estimated_host_peak_bytes", 2**63) + 1024**3 > 64 * 1024**3:
        raise MemoryError("Estimated peak plus reserve exceeds fixed64GiB request")


def admission(upstream):
    """Use restored sealed evidence only, never retired source/raw paths."""
    upstream = Path(upstream)
    h = read(upstream / "evidence/handoff.json")
    for name, key in (
        ("quality.json", "quality_sha256"),
        ("dataset-seal.json", "dataset_seal_sha256"),
        ("frozen-manifest.json", "frozen_manifest_sha256"),
    ):
        if sha(upstream / "evidence" / name) != h[key]:
            raise ValueError("Handoff evidence changed")
    q = read(upstream / "evidence/quality.json")
    if (
        h.get("training_eligible") is not True
        or h.get("run_count") != 1200
        or h.get("scaler_scope") != "all_selected_normal_train"
        or q.get("passed") is not True
        or q.get("final_test_used") is not False
        or q.get("train_only_scaler_verified") is not True
    ):
        raise ValueError("Verified1200-run common train-only scaler handoff required")
    dataset = upstream / "dataset"
    frozen = read(upstream / "evidence/frozen-manifest.json")
    seal = read(upstream / "evidence/dataset-seal.json")
    if sha(dataset / "manifest.json") != seal["manifest.json"]:
        raise ValueError("Dataset manifest changed")
    manifest = read(dataset / "manifest.json")
    rows = manifest["runs"]
    if (
        frozen.get("status") != "DATA_FROZEN"
        or rows != frozen["runs"]
        or Counter(r["role"] for r in rows) != COUNTS
        or len({r["run_id"] for r in rows}) != 1200
    ):
        raise ValueError("Frozen1200-run membership/split changed")
    partitions = {}
    for row in rows:
        part = (
            row["role"]
            if row["role"] in COUNTS
            and row["role"].startswith("normal_")
            and row["role"] != "normal_dev_eval"
            else "development"
        )
        for key in ("seed", "group_id", "paired_case_id", "raw_flight_id"):
            lineage = (key, row[key])
            if lineage in partitions and partitions[lineage] != part:
                raise ValueError("Cross-split lineage leakage")
            partitions[lineage] = part
    headers = {}
    for role in ("normal_train", "normal_validation", "normal_calibration"):
        path = dataset / (role + ".npz")
        if sha(path) != seal[path.name] or manifest["hashes"][path.name] != seal[path.name]:
            raise ValueError("Fit archive changed")
        headers[role] = npz_headers(path)
        n = headers[role]["X"]["shape"][0]
        if n < 512 or any(
            headers[role][k]["shape"] != (n, *shape)
            for k, shape in (("X", (100, 19)), ("C", (6,)), ("Y", (10, 4)))
        ):
            raise ValueError("Forecast shape/window count mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays["run_id"]) != {r["run_id"] for r in rows if r["role"] == role}:
                raise ValueError("Array run membership differs from frozen role")
    if sha(dataset / "scaler.npz") != seal["scaler.npz"]:
        raise ValueError("Common scaler changed")
    return rows, seal, headers


def memory_plan(headers):
    raw = sum(v["bytes"] for role in headers.values() for v in role.values())
    # mmap training avoids float64 normalization/full float32 duplicate copies.
    # Calibration still uses the historical float64 loader; bound all role arrays
    # conservatively, plus replay/evaluator/BLAS/Python reserves. No N-by-N ridge.
    d = 100 * 19 + 6 + 1
    estimate = max(12 * 1024**3, 4 * raw + 8 * d * d * 8 + 4 * 1024**3)
    return dict(
        array_bytes=raw,
        estimated_host_peak_bytes=estimate,
        requested_host_bytes=64 * 1024**3,
        primal_matrix_bytes=d * d * 8,
        old_dual_matrix_bytes=headers["normal_train"]["X"]["shape"][0] ** 2 * 8,
        kind="conservative estimate; runtime admission and Slurm enforce request",
        loader="chunk-normalized float32 mmap; no training windows copied wholesale",
    )


def normalized_mmap(dataset, role, scaler, cache, progress, chunk=512):
    """Stream numeric NPY members from NPZ, normalize in float64 then store float32.

    This preserves legacy normalization rounding before the model's float32 cast.
    A new owned cache is created once; partial caches never count as success.
    """
    if role not in {"normal_train", "normal_validation"}:
        raise ValueError("Mmap loader accepts normal train/validation only")
    dataset, cache = Path(dataset), Path(cache)
    path = dataset / (role + ".npz")
    expected = read(dataset / "manifest.json")["hashes"][path.name]
    if sha(path) != expected:
        raise ValueError("Fit archive changed")
    headers = npz_headers(path)
    cache.mkdir(parents=True, exist_ok=False)
    result = {}
    with zipfile.ZipFile(path) as archive:
        for key, scale in (("X", "x"), ("C", "c"), ("Y", "y")):
            shape = headers[key]["shape"]
            dtype = np.dtype(headers[key]["descr"])
            if dtype.kind != "f" or len(shape) < 2:
                raise ValueError("Floating forecast arrays required")
            width = math.prod(shape[1:])
            out = np.lib.format.open_memmap(
                cache / (key + ".npy"), mode="w+", dtype=np.float32, shape=shape
            )
            with archive.open(key + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                reader(stream)
                for offset in range(0, shape[0], chunk):
                    progress.check()
                    count = min(chunk, shape[0] - offset)
                    body = stream.read(count * width * dtype.itemsize)
                    if len(body) != count * width * dtype.itemsize:
                        raise ValueError("Truncated fit member")
                    block = np.frombuffer(body, dtype=dtype).reshape(count, *shape[1:])
                    normalized = (block - scaler[scale + "_mean"]) / scaler[scale + "_scale"]
                    if not np.isfinite(normalized).all():
                        raise ValueError("Nonfinite normalized fit input")
                    out[offset : offset + count] = normalized
                    progress.update(
                        force=offset == 0,
                        stage="normalize_mmap",
                        model=role,
                        array=key,
                        done=offset + count,
                        total=shape[0],
                    )
                if stream.read(1):
                    raise ValueError("Trailing fit member bytes")
            out.flush()
            # Keep array completion visible even if the last chunk was throttled.
            progress.update(done=shape[0], total=shape[0])
            result[key] = out
    write(cache / "ready.json", dict(source_sha256=expected, dtype="float32", chunk=chunk))
    return result
