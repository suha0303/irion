"""Read-only derived-data audit. No preprocessing, fitting, inference or truth reader."""

import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from capstone_ai.mapping import FEATURES, TARGETS

STORAGE = Path("/media/sa/X31/Data/X500/ROUND_001/R3_LONG15")
FIT = ("normal_train", "normal_validation", "normal_calibration")
ATOL, RTOL = 1e-10, 1e-10
RAW_DENY = STORAGE / "consumer"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("regular file without symlink required: " + str(path))
    return path.resolve()


def linked(ref):
    path = regular(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("linked input hash mismatch: " + str(path))
    return read(path)


def verify(root, inventory, allowed=None):
    root = Path(root).resolve()
    if not inventory:
        raise ValueError("empty input inventory")
    for name, digest in inventory.items():
        name_path = Path(name)
        if (
            name_path.is_absolute()
            or ".." in name_path.parts
            or (allowed is not None and name not in allowed)
        ):
            raise ValueError("unexpected inventory member: " + name)
        path = regular(root / name)
        if sha(path) != digest:
            raise ValueError("input hash mismatch: " + name)


def fence(denied):
    denied = [Path(p).resolve() for p in denied] + [RAW_DENY]

    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if (
                path.name in {"events.jsonl", "test.npz", "streams.jsonl"}
                or path.suffix in {".tar", ".gz", ".zip"}
                or any(path.is_relative_to(p) for p in denied)
            ):
                raise PermissionError("audit cannot read raw/truth/protected data: " + str(path))

    sys.addaudithook(guard)


def records(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def equal_array(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(label + ": shape/nonfinite mismatch")
    difference = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not np.allclose(a, b, rtol=RTOL, atol=ATOL):
        raise ValueError(label + ": value mismatch; max_abs=" + str(difference))
    return difference


def mean_scale(values):
    """Independent fsum two-pass population variance; not pipeline/scaler_from_train."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("nonempty finite scaler observations required")
    means = [math.fsum(column) / len(values) for column in values.T]
    scales = [
        max(math.sqrt(math.fsum((float(v) - m) ** 2 for v in column) / len(values)), 1e-5)
        for m, column in zip(means, values.T)
    ]
    return np.asarray(means), np.asarray(scales)


def frame_id(frame):
    return frame["clock_id"], frame["segment_id"], frame["time_us"]


def inspect_run(folder, row):
    frames, windows, exclusions = [
        records(folder / (name + ".jsonl")) for name in ("frames", "windows", "exclusions")
    ]
    summary = read(folder / "summary.json")
    identities = [frame_id(f) for f in frames]
    if len(set(identities)) != len(identities):
        # The original scaler would include every duplicated canonical row. Never silently dedup.
        raise ValueError("duplicate canonical frame identity: " + row["run_id"])
    lookup = dict(zip(identities, frames))
    quality, states, spans = {n: Counter() for n in FEATURES}, Counter(), defaultdict(list)
    train_rows, accepted, dropped, origins = [], [], Counter(), set()
    valid_frames = airborne_valid = warmup = airborne_warmup = 0
    previous = {}
    for f in frames:
        if (
            f["run_id"] != row["run_id"]
            or f["feature_order"] != FEATURES
            or set(f["quality"]) != set(FEATURES)
        ):
            raise ValueError("canonical identity/feature order mismatch")
        clock, segment, t = frame_id(f)
        scope = clock, segment
        if t % 100000 or t <= previous.get(scope, -1):
            raise ValueError("unordered/off-grid canonical frames")
        previous[scope] = t
        valid = all(f["quality"][n] == "valid" for n in FEATURES)
        valid_frames += valid
        warmup += bool(f["warmup"])
        states[f["operating_state"]] += 1
        for n in FEATURES:
            quality[n][f["quality"][n]] += 1
        if valid and not np.isfinite([f["values"][n] for n in FEATURES]).all():
            raise ValueError("valid canonical frame has nonfinite values")
        for trace in f.get("lineage", {}).values():
            if trace["sample_time_us"] > t or trace["available_time_us"] > t:
                raise ValueError("future observation in canonical frame")
        if f["operating_state"] == "airborne":
            spans[clock].append(t)
            airborne_warmup += bool(f["warmup"])
            if valid:
                airborne_valid += 1
                if row["role"] == "normal_train":
                    # Warmup rows and rows outside accepted histories remain included.
                    values = [f["values"][n] for n in FEATURES]
                    values[FEATURES.index("pack_voltage")] /= 4
                    values[FEATURES.index("pack_current")] /= 12
                    train_rows.append(values)
    for w in windows:
        key = w["clock_id"], w["segment_id"], w["origin_us"]
        if key in origins or w["run_id"] != row["run_id"] or w["feature_order"] != FEATURES:
            raise ValueError("duplicate or inconsistent window identity/order")
        origins.add(key)
        t = w["origin_us"]
        if w["history_times_us"] != list(range(t - 9900000, t + 1, 100000)) or w[
            "target_times_us"
        ] != list(range(t + 100000, t + 1000001, 100000)):
            raise ValueError("history/future grid mismatch")
        history = [lookup.get((*key[:2], timestamp)) for timestamp in w["history_times_us"]]
        if any(
            f is None
            or f["operating_state"] != "airborne"
            or any(f["quality"][n] != "valid" for n in FEATURES)
            for f in history
        ):
            raise ValueError("emitted window uses missing/invalid/nonairborne history")
        equal_array(w["X"], [[f["values"][n] for n in FEATURES] for f in history], "window history")
        current = lookup[key]
        if current["warmup"] or current["context"] is None or w["context_start_us"] != t - 60000000:
            raise ValueError("window context warmup/range mismatch")
        equal_array(w["C"], current["context"], "window context")
        if np.asarray(w["C"]).shape != (6,):
            raise ValueError("six context features required")
        target_frames = [lookup.get((*key[:2], timestamp)) for timestamp in w["target_times_us"]]
        missing = sum(f is None for f in target_frames)
        invalid = sum(
            f is not None and any(f["quality"][n] != "valid" for n in TARGETS)
            for f in target_frames
        )
        if missing or invalid:
            dropped["windows"] += 1
            dropped["missing_target_frames"] += missing
            dropped["nonvalid_target_frames"] += invalid
            dropped["windows_missing_target"] += bool(missing)
            dropped["windows_nonvalid_target"] += bool(invalid)
            continue
        x = np.asarray(w["X"], dtype=np.float64).copy()
        x[:, FEATURES.index("pack_voltage")] /= 4
        x[:, FEATURES.index("pack_current")] /= 12
        y = np.asarray([[f["values"][n] for n in TARGETS] for f in target_frames], dtype=np.float64)
        y[:, -1] /= 4
        accepted.append(
            (x, np.asarray(w["C"], dtype=np.float64), y, row["run_id"], t, w["segment_id"])
        )
    reasons = Counter(reason for item in exclusions for reason in item["reasons"])
    if any(
        summary.get(k) != v
        for k, v in {
            "run_id": row["run_id"],
            "closed": True,
            "frames": len(frames),
            "valid_frames": valid_frames,
            "windows": len(windows),
            "exclusions": dict(reasons),
            "feature_order": FEATURES,
        }.items()
    ):
        raise ValueError("canonical summary differs from recomputed counts")
    if row["role"] in FIT and not accepted:
        raise ValueError("required fit run has no arrived valid targets")
    return (
        {
            "run_id": row["run_id"],
            "role": row["role"],
            "frames": len(frames),
            "valid_frames": valid_frames,
            "windows": len(windows),
            "accepted_targets": len(accepted),
            "excluded_targets": dropped["windows"],
            "target_drop_detail": dict(dropped),
            "window_exclusion_rows": len(exclusions),
            "window_exclusion_reasons": dict(reasons),
            "quality_counts": {n: dict(counts) for n, counts in quality.items()},
            "operating_state_frames": dict(states),
            "warmup_frames": warmup,
            "airborne_warmup_frames": airborne_warmup,
            "valid_airborne_frames": airborne_valid,
            "observed_airborne_seconds": sum(len(ts) for ts in spans.values()) / 10,
            "valid_airborne_seconds": airborne_valid / 10,
            "airborne_envelope_seconds": sum(max(ts) - min(ts) + 100000 for ts in spans.values())
            / 1e6,
            "exposure_definition": "observed counts versus per-clock first-to-last airborne envelope including gaps; not model-scored exposure",
            "clocks": len({f["clock_id"] for f in frames}),
            "segments": len(previous),
            "duplicate_frame_ids": 0,
            "duplicate_window_ids": 0,
        },
        accepted,
        train_rows,
    )


def inspect_dataset(dataset, frozen):
    dataset = Path(dataset)
    manifest = read(dataset / "manifest.json")
    if manifest["runs"] != frozen["runs"]:
        raise ValueError("prepared membership differs from raw freeze")
    per_run, expected, scaler_frames = [], {role: [] for role in FIT}, []
    original_audit = {r["run_id"]: r for r in manifest["audit"]}
    if len(original_audit) != len(frozen["runs"]):
        raise ValueError("missing/duplicate prepared audit rows")
    for row in frozen["runs"]:
        if (
            not row["run_id"]
            or Path(row["run_id"]).name != row["run_id"]
            or row["run_id"] in {".", ".."}
        ):
            raise ValueError("unsafe run identity")
        report, samples, selected = inspect_run(dataset / "canonical" / row["run_id"], row)
        if row["role"] in FIT:
            expected[row["role"]].extend(samples)
            for key in ("accepted_targets", "excluded_targets"):
                if original_audit[row["run_id"]][key] != report[key]:
                    raise ValueError("prepared target accounting differs from canonical")
        scaler_frames.extend(selected)
        per_run.append(report)
    dimensions, max_error = {}, 0.0
    for role, samples in expected.items():
        path = dataset / (role + ".npz")
        if sha(path) != manifest["hashes"][path.name]:
            raise ValueError("NPZ hash mismatch")
        if not samples:
            raise ValueError("empty fit role")
        arrays = dict(zip(("X", "C", "Y", "run_id", "origin_us", "segment"), zip(*samples)))
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError("NPZ member set mismatch")
            for key in ("X", "C", "Y"):
                if saved[key].dtype != np.dtype("float64"):
                    raise ValueError("unexpected numeric NPZ dtype: " + role + "/" + key)
                max_error = max(max_error, equal_array(saved[key], arrays[key], role + "/" + key))
            for key in ("run_id", "origin_us", "segment"):
                if not np.array_equal(saved[key], np.asarray(arrays[key])):
                    raise ValueError("NPZ identity/order mismatch")
            dimensions[role] = {
                "windows": len(saved["X"]),
                "history": saved["X"].shape[1],
                "features": saved["X"].shape[2],
                "context": saved["C"].shape[1],
                "horizon": saved["Y"].shape[1],
                "targets": saved["Y"].shape[2],
                "stored_array_bytes": sum(saved[k].nbytes for k in saved.files),
            }
    x_mean, x_scale = mean_scale(scaler_frames)
    contexts = [sample[1] for sample in expected["normal_train"]]
    c_mean, c_scale = mean_scale(contexts)
    targets = [FEATURES.index(n) for n in TARGETS]
    recomputed = dict(
        x_mean=x_mean,
        x_scale=x_scale,
        c_mean=c_mean,
        c_scale=c_scale,
        y_mean=x_mean[targets],
        y_scale=x_scale[targets],
    )
    if sha(dataset / "scaler.npz") != manifest["hashes"]["scaler.npz"]:
        raise ValueError("scaler hash mismatch")
    if manifest["unique_train_frames"] != len(scaler_frames):
        raise ValueError("train scaler frame count mismatch")
    with np.load(dataset / "scaler.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(recomputed):
            raise ValueError("scaler member set mismatch")
        scaler_errors = {
            k: equal_array(saved[k], value, "scaler/" + k) for k, value in recomputed.items()
        }
    by_role = {}
    for role in sorted({r["role"] for r in per_run}):
        selected = [r for r in per_run if r["role"] == role]
        keys = (
            "frames",
            "valid_frames",
            "windows",
            "accepted_targets",
            "excluded_targets",
            "window_exclusion_rows",
            "warmup_frames",
            "airborne_warmup_frames",
            "valid_airborne_frames",
            "observed_airborne_seconds",
            "valid_airborne_seconds",
            "airborne_envelope_seconds",
        )
        by_role[role] = {"runs": len(selected), **{k: sum(r[k] for r in selected) for k in keys}}
        for field in ("target_drop_detail", "window_exclusion_reasons", "operating_state_frames"):
            totals = Counter()
            for r in selected:
                totals.update(r[field])
            by_role[role][field] = dict(totals)
        by_role[role]["quality_counts"] = {}
        for feature in FEATURES:
            totals = Counter()
            for r in selected:
                totals.update(r["quality_counts"][feature])
            by_role[role]["quality_counts"][feature] = dict(totals)
    return {
        "per_run": per_run,
        "by_role": by_role,
        "dimensions": dimensions,
        "numeric_atol": ATOL,
        "numeric_rtol": RTOL,
        "npz_max_abs_error": max_error,
        "scaler_max_abs_errors": scaler_errors,
        "train_only_scaler_verified": True,
        "scaler_observations": {
            "frame_rows": len(scaler_frames),
            "context_rows": len(contexts),
            "rule": "all valid airborne train canonical rows, including warmup and outside accepted histories; equal-valued rows retained; duplicate IDs rejected; context only accepted train windows",
            "variance": "population ddof0; floor1e-5; independent math.fsum two-pass",
            "targets": "x scaler body p/q/r and pack voltage columns, not Y statistics",
        },
    }


def memory_assessment(dimensions, evidence):
    train, val = dimensions["normal_train"], dimensions["normal_validation"]
    n, v = train["windows"], val["windows"]
    d = train["history"] * train["features"] + train["context"] + 1
    k = train["horizon"] * train["targets"]
    if n <= 0 or v <= 0 or d != 1907 or k != 40:
        raise ValueError("unexpected actual ridge dimensions")
    gram, design, targets, coefficients = 8 * n * n, 8 * (n + v) * d, 8 * (n + v) * k, 8 * d * k
    normalized = 8 * (n + v) * (d - 1 + k)
    # This is a declared admission estimate, not a change to the existing dual solver.
    working = 8 * gram + 2 * design + 2 * normalized + 3 * targets + 2 * coefficients
    result = {
        "N_train": n,
        "N_validation": v,
        "D": d,
        "K": k,
        "dtype": "float64",
        "one_gram_bytes": gram,
        "design_bytes": design,
        "normalized_numeric_bytes": normalized,
        "planning_working_bytes": working,
        "runtime_reserve_bytes": 2 * 1024**3,
        "formula": "8 Gram +2 design +2 normalized numeric +3 targets +2 coefficients +2GiB runtime reserve; retain >=25% declared usable RAM",
        "peak_rss_measured": False,
        "peak_bound_proven": False,
        "memory_audit_passed": False,
        "status": "needs_DIS01_RAM_evidence",
    }
    if evidence is None:
        return result
    if (
        evidence.get("hostname") != "DIS01"
        or evidence.get("kind") != "coordinator_verified_cpu_RAM_budget"
        or not evidence.get("source")
        or not evidence.get("observed_utc")
    ):
        raise ValueError("explicit DIS01 CPU RAM evidence required; GPU VRAM is not CPU RAM")
    budget, available = evidence["allocation_budget_bytes"], evidence["available_bytes"]
    if type(budget) is not int or type(available) is not int or min(budget, available) <= 0:
        raise ValueError("positive CPU allocation budget and observed available RAM required")
    usable = min(budget, available)
    required = working + result["runtime_reserve_bytes"]
    passed = required <= usable * 0.75
    result.update(
        memory_audit_passed=passed,
        status="fits_declared_RAM_plan" if passed else "insufficient_RAM_plan",
        usable_declared_bytes=usable,
        estimated_required_bytes=required,
        estimated_remaining_bytes=usable - required,
        evidence=evidence,
        limitation="planning audit only; coordinator must recheck memory availability/Slurm limit before main; LAPACK/runtime peak remains unmeasured",
    )
    return result
