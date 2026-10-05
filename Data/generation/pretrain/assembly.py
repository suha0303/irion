"""Disk-backed assembly and comparison of verified run outputs, never a trainer."""

import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from .common import ADMISSION, load, package_imports, read, sha, write

FIT = ("normal_train", "normal_validation", "normal_calibration")
KEYS = ("X", "C", "Y", "run_id", "origin_us", "segment")


class Moments:
    """Producer statistics combine bounded batches with population variance."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Invalid train scaler observations")
        n, mean, m2 = len(values), values.mean(0), ((values - values.mean(0)) ** 2).sum(0)
        if self.n == 0:
            self.n, self.mean, self.m2 = n, mean, m2
            return
        delta = mean - self.mean
        self.m2 += m2 + delta**2 * self.n * n / (self.n + n)
        self.mean += delta * n / (self.n + n)
        self.n += n

    def result(self):
        if not self.n:
            raise ValueError("Empty train scaler observations")
        return self.mean, np.maximum(np.sqrt(self.m2 / self.n), 1e-5)


def prepare(frozen, out, preprocess, budget, *, process_run):
    package_imports()
    from capstone_ai.dataset import feature_transform, target_transform
    from capstone_ai.mapping import FEATURES, TARGETS

    out = Path(out)
    out.mkdir(exist_ok=False)
    work = out.parent / "assembly-work"
    work.mkdir(exist_ok=False)
    reports, fragments = [], {role: [] for role in FIT}
    frames_stats, context_stats = Moments(), Moments()
    for index, row in enumerate(frozen["runs"]):
        budget.check()
        folder = out / "canonical" / row["run_id"]
        summary = process_run(row["path"], folder, preprocess)
        if not summary["closed"]:
            raise ValueError("Unclosed canonical run")
        item = dict(
            run_id=row["run_id"],
            role=row["role"],
            summary=summary,
            accepted_targets=None,
            excluded_targets=None,
        )
        if row["role"] in FIT:
            with (folder / "frames.jsonl").open() as stream:
                frames = [json.loads(line) for line in stream if line.strip()]
            lookup = {(f["clock_id"], f["segment_id"], f["time_us"]): f for f in frames}
            samples, windows = [], 0
            with (folder / "windows.jsonl").open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    w = json.loads(line)
                    windows += 1
                    if windows % 128 == 0:
                        budget.check()
                    targets = [
                        lookup.get((w["clock_id"], w["segment_id"], t))
                        for t in w["target_times_us"]
                    ]
                    if any(
                        f is None or any(f["quality"][n] != "valid" for n in TARGETS)
                        for f in targets
                    ):
                        continue
                    y = [[f["values"][n] for n in TARGETS] for f in targets]
                    samples.append(
                        (w["X"], w["C"], y, row["run_id"], w["origin_us"], w["segment_id"])
                    )
            if not samples or not summary["valid_frames"] or windows != summary["windows"]:
                raise ValueError("Fit run missing valid windows/frames")
            x, c, y, ids, origins, segments = zip(*samples)
            arrays = dict(
                X=feature_transform(x, preprocess),
                C=np.asarray(c),
                Y=target_transform(y),
                run_id=np.asarray(ids),
                origin_us=np.asarray(origins),
                segment=np.asarray(segments),
            )
            target = work / f"run-{index:06d}.npz"
            np.savez(target, **arrays)
            fragments[row["role"]].append((target, len(samples)))
            item.update(accepted_targets=len(samples), excluded_targets=windows - len(samples))
            if row["role"] == "normal_train":
                frames_stats.add(
                    feature_transform(
                        [
                            [f["values"][n] for n in FEATURES]
                            for f in frames
                            if f["operating_state"] == "airborne"
                            and all(f["quality"][n] == "valid" for n in FEATURES)
                        ],
                        preprocess,
                    )
                )
                context_stats.add(arrays["C"])
            del frames, lookup, samples, arrays, x, c, y
        reports.append(item)
    for role, parts in fragments.items():
        if not parts:
            raise ValueError("Empty fit role: " + role)
        total = sum(count for _, count in parts)
        # Widen identities once across all fragments; long IDs must never be truncated.
        dtypes, shapes = {}, {}
        for path, count in parts:
            with np.load(path, allow_pickle=False) as part:
                for key in KEYS:
                    a = part[key]
                    dtypes[key] = np.result_type(dtypes[key], a.dtype) if key in dtypes else a.dtype
                    shapes[key] = a.shape[1:]
        maps = {
            key: np.lib.format.open_memmap(
                work / f"{role}-{key}.npy",
                mode="w+",
                dtype=dtypes[key],
                shape=(total, *shapes[key]),
            )
            for key in KEYS
        }
        offset = 0
        for path, count in parts:
            budget.check()
            with np.load(path, allow_pickle=False) as part:
                for key in KEYS:
                    maps[key][offset : offset + count] = part[key]
            offset += count
        for value in maps.values():
            value.flush()
        np.savez_compressed(out / (role + ".npz"), **maps)
        del maps
        for key in KEYS:
            (work / f"{role}-{key}.npy").unlink()
        for path, _ in parts:
            path.unlink()
    xm, xs = frames_stats.result()
    cm, cs = context_stats.result()
    indices = [FEATURES.index(n) for n in TARGETS]
    np.savez(
        out / "scaler.npz",
        x_mean=xm,
        x_scale=xs,
        c_mean=cm,
        c_scale=cs,
        y_mean=xm[indices],
        y_scale=xs[indices],
    )
    write(
        out / "manifest.json",
        dict(
            runs=frozen["runs"],
            audit=reports,
            scaler_fit="normal_train unique valid airborne frames; context once per train origin",
            unique_train_frames=frames_stats.n,
            hashes={p.name: sha(p) for p in out.glob("*.npz")},
        ),
    )
    work.rmdir()
    budget.check()
    return out


def mmap_archive(path, target, budget):
    """Unzip numeric arrays to owned external scratch, then map without materializing X."""
    target.mkdir()
    result = {}
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(KEYS) or set(archive.namelist()) != {
            key + ".npy" for key in KEYS
        }:
            raise ValueError("Unexpected numeric archive members")
        for key in KEYS:
            destination = target / (key + ".npy")
            with archive.open(key + ".npy") as source, destination.open("xb") as output:
                while block := source.read(1024 * 1024):
                    budget.check()
                    output.write(block)
            result[key] = np.load(destination, mmap_mode="r", allow_pickle=False)
    return result


def inspect_dataset(dataset, frozen, budget, *, inspect_run_fn):
    """Compare against independently computed per-run audits; independent fsum scaler."""
    package_imports()
    from capstone_ai.mapping import FEATURES, TARGETS

    derived = load("pretrain_bounded_independent", ADMISSION / "derived.py")
    dataset = Path(dataset)
    manifest = read(dataset / "manifest.json")
    if manifest["runs"] != frozen["runs"]:
        raise ValueError("Prepared membership differs from raw freeze")
    original = {r["run_id"]: r for r in manifest["audit"]}
    if len(original) != len(frozen["runs"]):
        raise ValueError("Duplicate/missing prepared audit rows")
    work = dataset.parent / "quality-work"
    work.mkdir(exist_ok=False)
    arrays, dimensions, offsets = {}, {}, Counter()
    for role in FIT:
        path = dataset / (role + ".npz")
        if sha(path) != manifest["hashes"][path.name]:
            raise ValueError("NPZ hash mismatch")
        data = mmap_archive(path, work / role, budget)
        if any(data[k].dtype != np.dtype("float64") for k in ("X", "C", "Y")):
            raise ValueError("Unexpected numeric dtype")
        arrays[role] = data
        dimensions[role] = dict(
            windows=len(data["X"]),
            history=data["X"].shape[1],
            features=data["X"].shape[2],
            context=data["C"].shape[1],
            horizon=data["Y"].shape[1],
            targets=data["Y"].shape[2],
            stored_array_bytes=sum(a.nbytes for a in data.values()),
        )
    per_run, max_error, frame_count, context_count = [], 0.0, 0, 0
    with (
        (work / "train-frames.f64").open("xb") as frames_out,
        (work / "train-context.f64").open("xb") as contexts_out,
    ):
        for row in frozen["runs"]:
            budget.check()
            report, samples, selected = inspect_run_fn(dataset / "canonical" / row["run_id"], row)
            role = row["role"]
            if role in FIT:
                if not samples:
                    raise ValueError("Empty fit role run")
                for key in ("accepted_targets", "excluded_targets"):
                    if original[row["run_id"]][key] != report[key]:
                        raise ValueError("Target accounting differs from canonical")
                for key, values in zip(KEYS, zip(*samples)):
                    actual = arrays[role][key][offsets[role] : offsets[role] + len(samples)]
                    if key in ("X", "C", "Y"):
                        max_error = max(
                            max_error, derived.equal_array(actual, values, role + "/" + key)
                        )
                    elif not np.array_equal(actual, np.asarray(values)):
                        raise ValueError("Identity/order mismatch")
                offsets[role] += len(samples)
                if role == "normal_train":
                    contexts = np.asarray([s[1] for s in samples], dtype=np.float64)
                    contexts.tofile(contexts_out)
                    context_count += len(contexts)
            if len(selected):
                if role != "normal_train":
                    raise ValueError("Non-train scaler observations")
                values = np.asarray(selected, dtype=np.float64)
                values.tofile(frames_out)
                frame_count += len(values)
            per_run.append(report)
    if (
        any(offsets[role] != dimensions[role]["windows"] for role in FIT)
        or manifest["unique_train_frames"] != frame_count
    ):
        raise ValueError("Aggregate row count mismatch")
    if not frame_count or not context_count:
        raise ValueError("Missing train scaler observations")
    xm, xs = derived.mean_scale(
        np.memmap(
            work / "train-frames.f64",
            dtype=np.float64,
            mode="r",
            shape=(frame_count, len(FEATURES)),
        )
    )
    cm, cs = derived.mean_scale(
        np.memmap(work / "train-context.f64", dtype=np.float64, mode="r", shape=(context_count, 6))
    )
    indices = [FEATURES.index(n) for n in TARGETS]
    expected = dict(
        x_mean=xm, x_scale=xs, c_mean=cm, c_scale=cs, y_mean=xm[indices], y_scale=xs[indices]
    )
    if sha(dataset / "scaler.npz") != manifest["hashes"]["scaler.npz"]:
        raise ValueError("Scaler hash mismatch")
    with np.load(dataset / "scaler.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError("Scaler member mismatch")
        errors = {k: derived.equal_array(saved[k], v, "scaler/" + k) for k, v in expected.items()}
    by_role = {}
    fields = (
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
    for role in sorted({r["role"] for r in per_run}):
        reports = [r for r in per_run if r["role"] == role]
        totals = dict(runs=len(reports), **{key: sum(r[key] for r in reports) for key in fields})
        for key in ("target_drop_detail", "window_exclusion_reasons", "operating_state_frames"):
            counter = Counter()
            for report in reports:
                counter.update(report[key])
            totals[key] = dict(counter)
        totals["quality_counts"] = {}
        for feature in FEATURES:
            counter = Counter()
            for report in reports:
                counter.update(report["quality_counts"][feature])
            totals["quality_counts"][feature] = dict(counter)
        by_role[role] = totals
    del arrays, data, actual
    shutil.rmtree(work)  # Exact new scratch directory; never raw input or prior-task evidence.
    budget.check()
    return dict(
        per_run=per_run,
        by_role=by_role,
        dimensions=dimensions,
        numeric_atol=derived.ATOL,
        numeric_rtol=derived.RTOL,
        npz_max_abs_error=max_error,
        scaler_max_abs_errors=errors,
        train_only_scaler_verified=True,
        scaler_observations=dict(
            frame_rows=frame_count,
            context_rows=context_count,
            rule="all valid airborne train canonical rows, including warmup and outside accepted histories; equal-valued rows retained; duplicate IDs rejected; context only accepted train windows",
            variance="population ddof0; floor1e-5; independent math.fsum two-pass",
            targets="x scaler body p/q/r and pack voltage columns, not Y statistics",
        ),
    )
