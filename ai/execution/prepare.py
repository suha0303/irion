"""Local sealed-data admission, without collection, training or Slurm submission."""

import ast
import math
import shutil
import struct
import zipfile
from collections import Counter
from pathlib import Path

from .common import COUNTS, HANDOFF, HOME, MODELS, ROOT, child, read, sha, write


def npz_headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        if len({i.filename for i in archive.infolist()}) != len(archive.infolist()):
            raise ValueError("Duplicate NPZ member")
        for info in archive.infolist():
            if not info.filename.endswith(".npy") or Path(info.filename).name != info.filename:
                raise ValueError("Unexpected NPZ member")
            with archive.open(info) as stream:
                magic = stream.read(8)
                if magic[:6] != b"\x93NUMPY" or magic[6] not in (1, 2, 3):
                    raise ValueError("Unsupported NPY header")
                size = struct.unpack(
                    "<H" if magic[6] == 1 else "<I", stream.read(2 if magic[6] == 1 else 4)
                )[0]
                if size > 65536:
                    raise ValueError("Oversized NPY header")
                header = ast.literal_eval(stream.read(size).decode("latin1"))
                if "O" in header["descr"] or header["fortran_order"]:
                    raise ValueError("Object/Fortran arrays forbidden")
                result[info.filename[:-4]] = dict(header, bytes=info.file_size)
    return result


def memory_plan(headers):
    raw = sum(v["bytes"] for role in headers.values() for v in role.values())
    n = headers["normal_train"]["X"]["shape"][0]
    d = math.prod(headers["normal_train"]["X"]["shape"][1:]) + 6 + 1
    # Host arrays, decompression/normalization temporaries, ridge workspaces and replay Python objects.
    estimate = max(8 * 1024**3, 4 * raw + 8 * d * d * 8 + 2 * 1024**3)
    return dict(
        array_bytes=raw,
        estimated_host_peak_bytes=estimate,
        requested_host_bytes=16 * 1024**3,
        old_dual_matrix_bytes=n * n * 8,
        primal_matrix_bytes=d * d * 8,
        ridge_dimensions=d,
        kind="conservative estimate, not measured peak",
        gpu="Measured full batch for each architecture inside the single allocated job before fit",
    )


def admission(handoff=HANDOFF):
    import numpy as np

    handoff = Path(handoff)
    h = read(handoff)
    if (
        h.get("status") != "PRETRAIN_READY_PENDING_USER_REVIEW"
        or h.get("training_eligible") is not True
        or h.get("run_count") != 104
        or h.get("collection_sets") != 1
        or h.get("scaler_scope") != "all_selected_normal_train"
    ):
        raise ValueError("Complete104-run common-scaler handoff required")
    parent = handoff.parent
    for name, key in [
        ("quality.json", "quality_sha256"),
        ("dataset-seal.json", "dataset_seal_sha256"),
        ("frozen-manifest.json", "frozen_manifest_sha256"),
    ]:
        if sha(parent / name) != h[key]:
            raise ValueError("Changed handoff evidence: " + name)
    q = read(parent / "quality.json")
    if (
        q.get("status") != "PRETRAIN_QUALITY_PASSED"
        or q.get("passed") is not True
        or q.get("final_test_used") is not False
        or q.get("train_only_scaler_verified") is not True
    ):
        raise ValueError("Independent quality/train-only scaler proof required")
    dataset = Path(h["dataset"])
    seal = read(parent / "dataset-seal.json")
    if sha(dataset / "manifest.json") != seal["manifest.json"]:
        raise ValueError("Dataset manifest changed")
    manifest = read(dataset / "manifest.json")
    rows = manifest["runs"]
    frozen = read(parent / "frozen-manifest.json")
    if frozen.get("status") != "DATA_FROZEN" or rows != frozen["runs"]:
        raise ValueError("Frozen membership differs from prepared dataset")
    if Counter(r["role"] for r in rows) != COUNTS or len({r["run_id"] for r in rows}) != 104:
        raise ValueError("Expected63/9/9/11/12 split, unique104 runs, no final test")
    partitions = {}
    for row in rows:
        part = (
            row["role"]
            if row["role"] in {"normal_train", "normal_validation", "normal_calibration"}
            else "development"
        )
        for key in [
            ("seed", row["seed"]),
            ("group", row["group_id"]),
            ("pair", row["paired_case_id"]),
            ("run", row["raw_flight_id"]),
        ]:
            if key in partitions and partitions[key] != part:
                raise ValueError("Cross-split lineage leakage")
            partitions[key] = part
    headers = {}
    for role in ("normal_train", "normal_validation", "normal_calibration"):
        path = dataset / (role + ".npz")
        if sha(path) != seal[path.name] or manifest["hashes"][path.name] != seal[path.name]:
            raise ValueError("Fit archive changed")
        headers[role] = npz_headers(path)
        n = headers[role]["X"]["shape"][0]
        if (
            headers[role]["X"]["shape"] != (n, 100, 19)
            or headers[role]["C"]["shape"] != (n, 6)
            or headers[role]["Y"]["shape"] != (n, 10, 4)
            or n <= 0
        ):
            raise ValueError("Forecast dimensions mismatch")
        with np.load(path, allow_pickle=False) as data:
            if set(data["run_id"]) != {r["run_id"] for r in rows if r["role"] == role}:
                raise ValueError("Array run membership mismatch")
    if sha(dataset / "scaler.npz") != seal["scaler.npz"]:
        raise ValueError("Scaler changed")
    return h, rows, seal, headers


def create():
    h, rows, seal, headers = admission()
    HOME.mkdir(parents=True, exist_ok=True)
    config = dict(
        models=list(MODELS),
        seed=17,
        hidden=32,
        batch_size=32,
        epochs=20,
        learning_rate=0.001,
        early_stopping_patience=5,
        precision="float32",
        amp=False,
        retries=0,
        final_test_used=False,
        memory_gib=16,
        cpus=4,
        wall_hours=12,
        queue_hours=24,
        revision="ROUND_001/R4_GPU104_FOUR_SEED17",
        handoff=str(HANDOFF),
        handoff_sha256=sha(HANDOFF),
        dataset=h["dataset"],
        memory=memory_plan(headers),
        window_counts={k: v["X"]["shape"][0] for k, v in headers.items()},
        training_stride_us=1000000,
        history=100,
        future=10,
        step_us=100000,
        models_sequential=True,
        gpu_jobs_max=1,
        final_test_authorized=False,
    )
    source = {}
    for folder in ("execution", "src/capstone_ai", "pipeline"):
        for path in (ROOT / "ai" / folder).rglob("*.py"):
            source[str(path.relative_to(ROOT))] = sha(path)
    for path in [
        ROOT / "ai/remote/gpu_probe.py",
        ROOT
        / "Data/generation/pretrain/resources/preprocessing/preprocessing.json",
    ]:
        source[str(path.relative_to(ROOT))] = sha(path)
    config["source_sha256"] = source
    # Only normal fit arrays plus23 development canonical runs travel to the server.
    selected = {k: v for k, v in seal.items() if "/" not in k}
    dev = {r["run_id"] for r in rows if r["role"] in {"normal_dev_eval", "anomaly_dev"}}
    selected.update(
        {
            k: v
            for k, v in seal.items()
            if len(Path(k).parts) > 2
            and Path(k).parts[:1] == ("canonical",)
            and Path(k).parts[1] in dev
        }
    )
    config["dataset_files"] = selected
    path = HOME / "plan.json"
    if path.exists():
        raise FileExistsError("Plan already exists; preserve previous plan")
    write(path, config)
    (HOME / "plan.sha256").write_text(sha(path) + "\n")
    return path


def stage(config, destination, progress):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    total = len(config["source_sha256"]) + len(config["dataset_files"])
    done = 0
    inventory = {}
    for base, entries, prefix in [
        (ROOT, config["source_sha256"], ""),
        (Path(config["dataset"]), config["dataset_files"], "dataset/"),
    ]:
        for relative, expected in entries.items():
            progress.check()
            source = child(base, relative)
            name = prefix + relative
            target = child(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha(target) != expected:
                raise ValueError("Snapshot hash mismatch: " + name)
            inventory[name] = expected
            done += 1
            progress.update(stage="snapshot", done=done, total=total, model=None)
    package_preprocess = (
        "Data/generation/pretrain/resources/preprocessing/preprocessing.json"
    )
    shutil.copyfile(destination / package_preprocess, destination / "preprocessing.json")
    inventory["preprocessing.json"] = sha(destination / "preprocessing.json")
    write(destination / "config.json", config)
    inventory["config.json"] = sha(destination / "config.json")
    write(destination / "inventory.json", inventory)
    return destination
