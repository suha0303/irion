"""Run-bound physical feature admission and train-only normalization."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capstone_ai.mapping import FEATURES, TARGETS

TARGET_INDICES = tuple(FEATURES.index(name) for name in TARGETS)
STEP_US = 100_000
TRAIN_ORIGIN_INTERVAL_US = 1_000_000
ROLES = {
    "normal_train",
    "normal_validation",
    "normal_calibration",
    "normal_test",
    "anomaly_test",
}


def digest(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
        return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


@dataclass(frozen=True)
class Run:
    run_id: str
    values: np.ndarray
    time_us: np.ndarray
    segment: np.ndarray
    observed: np.ndarray  # Independent observed-target mask [T,4].


def load_run(path, run_id):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"values", "time_us", "segment", "observed"}:
            raise ValueError("Prepared data must have the new19 run schema")
        values, time, segment, observed = (
            archive[k] for k in ("values", "time_us", "segment", "observed")
        )
    n = len(time)
    if (
        n == 0
        or time.shape != (n,)
        or time.dtype != np.int64
        or np.any(np.diff(time) != STEP_US)
        or segment.shape != (n,)
        or segment.dtype != np.int64
        or np.any(segment < -1)
        or values.shape != (n, 19)
        or values.dtype not in (np.float32, np.float64)
        or observed.dtype != np.bool_
        or observed.shape != (n, 4)
        or np.any(observed & ~np.isfinite(values[:, TARGET_INDICES]))
        or np.any(observed[segment < 0])
    ):
        raise ValueError(
            f"Invalid observations, masks, or grid in run {run_id}"
        )
    return Run(run_id, values, time, segment, observed)


def load_split(manifest, roles):
    manifest = Path(manifest)
    data = read(manifest)
    if (
        data.get("schema") != "drone-ai.new19.dataset/1"
        or data.get("features") != FEATURES
    ):
        raise ValueError(
            "New19 dataset and exact common19 feature order required"
        )
    owners, ids, contents = {}, set(), set()
    calibration_groups = set()
    result = {role: [] for role in roles}
    for row in data["runs"]:
        role, ident = row["role"], row["run_id"]
        if (
            role not in ROLES
            or not isinstance(ident, str)
            or not ident
            or ident in ids
        ):
            raise ValueError("Roles must be explicit and run IDs unique")
        ids.add(ident)
        sha = row["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or sha in contents:
            raise ValueError("Duplicate or absent prepared hash")
        contents.add(sha)
        # Each lineage namespace is mandatory except paired cases, which may
        # have no partner. Replays of one raw flight stay in the same role.
        for key in ("raw_sha256", "group_id", "retry_group", "pair_id"):
            value = row[key]
            if key == "pair_id" and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing {key} in {ident}")
            if owners.setdefault((key, value), role) != role:
                raise ValueError(f"{key} crosses dataset roles: {value}")
            if role == "normal_calibration":
                if (key, value) in calibration_groups:
                    raise ValueError(
                        "Calibration requires independent run lineages"
                    )
                calibration_groups.add((key, value))
        if role not in result:
            continue
        path = manifest.parent / row["path"]
        root = manifest.parent.resolve()
        resolved = path.resolve(strict=True)
        if (
            not resolved.is_relative_to(root)
            or path.absolute() != resolved
            or not resolved.is_file()
            or digest(resolved) != sha
        ):
            raise ValueError(f"Prepared path or hash mismatch: {ident}")
        result[role].append(load_run(resolved, ident))
    if any(not rows for rows in result.values()):
        raise ValueError("Each requested role requires an independent run")
    return result


def forecast_origins(run):
    n = len(run.values)
    if n < 100:
        return np.empty(0, dtype=np.int64)
    input_valid = np.isfinite(run.values).all(axis=1) & (run.segment >= 0)
    window = np.lib.stride_tricks.sliding_window_view(input_valid, 100)
    ids = np.arange(99, n)
    breaks = np.r_[0, np.cumsum(run.segment[1:] != run.segment[:-1])]
    keep = (
        window.all(axis=-1)
        & run.observed[ids].all(axis=-1)
        & (breaks[ids - 99] == breaks[ids])
    )
    return ids[keep]


def origins(run, *, interval_us=STEP_US):
    if interval_us < STEP_US or interval_us % STEP_US != 0:
        raise ValueError(
            "Origin interval must be a positive multiple of the frame period"
        )
    n = len(run.values)
    ids = forecast_origins(run)
    ids = ids[ids + 10 < n]
    targets = ids[:, None] + np.arange(1, 11)
    breaks = np.r_[0, np.cumsum(run.segment[1:] != run.segment[:-1])]
    ids = ids[
        run.observed[targets].all(axis=(1, 2))
        & (breaks[ids] == breaks[ids + 10])
    ]
    if interval_us == STEP_US or not len(ids):
        return ids
    # Anchor the cadence to the first eligible origin in each segment. If a
    # later slot is unscorable, skip it and keep the cadence grid fixed.
    segment_starts = np.r_[True, breaks[ids[1:]] != breaks[ids[:-1]]]
    anchors = np.maximum.accumulate(np.where(segment_starts, ids, 0))
    return ids[(ids - anchors) % (interval_us // STEP_US) == 0]


def whole_admitted(run):
    v = run.values
    return (
        len(v) > 100
        and run.segment[0] >= 0
        and np.all(run.segment == run.segment[0])
        and np.isfinite(v[:100]).all()
        and np.isfinite(v[:, :17]).all()
        and np.isfinite(v[:, 18]).all()
        and run.observed[:100, 3].all()
        and run.observed[100:, 3].any()
    )


def physical_features(values):
    values = np.asarray(values, dtype=np.float64).copy()
    values[..., 17] /= 4
    values[..., 18] /= 12
    return values


class Moments:
    def __init__(self, width):
        self.n = 0
        self.mean = np.zeros(width)
        self.m2 = np.zeros(width)

    def add(self, values):
        n = len(values)
        if not n:
            return
        center = values.mean(axis=0)
        delta = center - self.mean
        self.m2 += ((values - center) ** 2).sum(axis=0) + delta**2 * (
            self.n * n / (self.n + n)
        )
        self.mean += delta * (n / (self.n + n))
        self.n += n

    def finish(self):
        if not self.n:
            raise ValueError("No valid normal training observations for scaler")
        return self.mean.tolist(), np.maximum(
            np.sqrt(self.m2 / self.n), 1e-5
        ).tolist()


def fit_scaler(runs, task, *, origin_interval_us=STEP_US):
    x, delta = Moments(19), Moments(4)
    for run in runs:
        v = physical_features(run.values)
        x.add(v[(run.segment >= 0) & np.isfinite(v).all(axis=1)])
        if task == "inflight":
            indices = origins(run, interval_us=origin_interval_us)
            four = v[:, TARGET_INDICES]
            for start in range(0, len(indices), 4096):
                i = indices[start : start + 4096]
                diff = four[i[:, None] + np.arange(1, 11)] - four[i, None]
                delta.add(diff.reshape(-1, 4))
    mean, scale = x.finish()
    scaler = {
        "mean": mean,
        "scale": scale,
        "fit_run_ids": [run.run_id for run in runs],
    }
    if task == "inflight":
        scaler["delta_mean"], scaler["delta_scale"] = delta.finish()
    return scaler


class Batches:
    def __init__(self, runs, scaler, task, *, origin_interval_us=STEP_US):
        self.runs = runs
        self.scaler = scaler
        self.task = task
        self.origin_interval_us = (
            origin_interval_us if task == "inflight" else None
        )
        self.features = []
        self.targets = []
        self.items = []
        for j, run in enumerate(runs):
            v = physical_features(run.values)
            x = ((v - scaler["mean"]) / scaler["scale"]).astype("f4")
            self.features.append(x)
            self.targets.append(v[:, TARGET_INDICES])
            if task == "inflight":
                self.items.extend(
                    (j, i) for i in origins(run, interval_us=origin_interval_us)
                )
            elif whole_admitted(run):
                self.items.append((j, None))
            else:
                raise ValueError(
                    f"Whole-run conditioning unavailable: {run.run_id}"
                )
        if not self.items:
            raise ValueError("No eligible items in role")

    def batch(self, selected):
        import torch

        items = [self.items[i] for i in selected]
        if self.task == "inflight":
            x = np.stack([self.features[j][i - 99 : i + 1] for j, i in items])
            delta = np.stack(
                [
                    self.targets[j][i + 1 : i + 11] - self.targets[j][i]
                    for j, i in items
                ]
            )
            y = (
                (delta - self.scaler["delta_mean"]) / self.scaler["delta_scale"]
            ).astype("f4")
            return torch.from_numpy(x), torch.from_numpy(y), None
        lengths = [len(self.features[j]) for j, _ in items]
        x = np.zeros((len(items), max(lengths), 19), dtype="f4")
        mask = np.zeros(x.shape[:2], dtype=np.bool_)
        for i, ((j, _), n) in enumerate(zip(items, lengths)):
            x[i, :n] = self.features[j]
            mask[i, 100:n] = self.runs[j].observed[100:, 3]
        return (
            torch.from_numpy(x),
            torch.tensor(lengths),
            torch.from_numpy(mask),
        )
