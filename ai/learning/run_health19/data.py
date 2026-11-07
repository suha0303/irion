"""Completed contiguous 10 Hz mission intervals, with normal-only fitting and lineage checks.

NPZ values use physical common19 units (pack volts/amperes). No old window dataset
is selected or converted. This consumer format does not change the raw contract.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capstone_ai.mapping import FEATURES
from execution.prepare import npz_headers
from learning.admission19 import (
    check_producer,
    check_protection,
    partition,
    source_evidence,
    validate_lineages,
)

SCHEMA = "drone-ai.run-health19.dataset/1"
INTERVAL_POLICY = "mission_only_actual_execution/1"
NORMAL = {
    "normal_train",
    "normal_validation",
    "normal_calibration",
    "normal_test",
}
ROLES = NORMAL | {"anomaly_test"}


@dataclass(frozen=True)
class Run:
    run_id: str
    role: str
    lineage: (
        tuple  # raw flight, retries, paired cases and collection group IDs.
    )
    health: str
    origin: str
    values: np.ndarray
    time_us: np.ndarray
    cohort: str | None = None

    def validate(self):
        partition(self.role, self.cohort)
        if not self.run_id or self.role not in ROLES or not self.lineage:
            raise ValueError("Run ID, role and explicit lineage required")
        if any(not isinstance(s, str) or not s for s in self.lineage):
            raise ValueError("Nonempty lineage identifiers required")
        if self.role in NORMAL and self.health != "confirmed_healthy":
            raise ValueError(
                "Normal roles require confirmed_healthy eligibility"
            )
        if self.role == "anomaly_test" and self.health != "abnormal":
            raise ValueError(
                "Anomaly truth is evaluator-only and must be explicit"
            )
        if self.origin not in {"px4_sitl", "synthetic_unit_fixture"}:
            raise ValueError("Explicit supported origin required")
        if (
            self.values.ndim != 2
            or self.values.shape[1] != 19
            or not len(self.values)
        ):
            raise ValueError("Nonempty full-run values[T,19] required")
        if (
            self.time_us.shape != (len(self.values),)
            or self.time_us.dtype != np.int64
        ):
            raise ValueError("One int64 timestamp per frame required")
        if not np.isfinite(self.values).all() or not np.all(
            np.diff(self.time_us) == 100000
        ):
            raise ValueError(
                "Gaps, clock resets and missing data require explicit abstention"
            )
        return self


def validate_splits(runs):
    if not runs:
        raise ValueError("Empty dataset")
    for run in runs:
        run.validate()
    validate_lineages(
        [
            dict(
                run_id=r.run_id, role=r.role, cohort=r.cohort, lineage=r.lineage
            )
            for r in runs
        ]
    )
    return runs


def physical_features(values):
    out = np.array(values, dtype=np.float64, copy=True)
    out[:, 17] /= 4  # Constant X500 nameplate, never true Q, SOC or health.
    out[:, 18] /= 12
    return out


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    scale: np.ndarray
    fit_run_ids: tuple

    @classmethod
    def fit(cls, runs):
        validate_splits(runs)
        if any(r.role != "normal_train" for r in runs):
            raise ValueError("Scaler may fit normal_train only")
        values = np.concatenate([physical_features(r.values) for r in runs])
        return cls(
            values.mean(0),
            np.maximum(values.std(0), 1e-5),
            tuple(r.run_id for r in runs),
        )

    def transform(self, run):
        run.validate()
        if self.mean.shape != (19,) or self.scale.shape != (19,):
            raise ValueError("Scaler dimension mismatch")
        if (
            not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or np.any(self.scale <= 0)
        ):
            raise ValueError("Invalid scaler")
        return (
            (physical_features(run.values) - self.mean) / self.scale
        ).astype(np.float32)

    def to_dict(self):
        return dict(
            mean=self.mean.tolist(),
            scale=self.scale.tolist(),
            fit_run_ids=list(self.fit_run_ids),
        )

    @classmethod
    def from_dict(cls, value):
        return cls(
            np.array(value["mean"]),
            np.array(value["scale"]),
            tuple(value["fit_run_ids"]),
        )


def batch(runs, scaler):
    import torch

    lengths = torch.tensor([len(r.values) for r in runs], dtype=torch.int64)
    values = torch.zeros(len(runs), int(lengths.max()), 19)
    for i, run in enumerate(runs):
        values[i, : lengths[i]] = torch.from_numpy(scaler.transform(run))
    return values, lengths


def fingerprint(runs):
    value = hashlib.sha256()
    for run in runs:
        value.update(
            json.dumps(
                [
                    run.run_id,
                    run.role,
                    run.lineage,
                    run.health,
                    run.origin,
                    run.cohort,
                ]
            ).encode()
        )
        value.update(np.asarray(run.values, dtype="<f8").tobytes())
        value.update(np.asarray(run.time_us, dtype="<i8").tobytes())
    return value.hexdigest()


def regular_path(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink input is unsupported")
    return path


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def manifest_rows(
    path, expected_producer, protection=None, usage="input_check"
):
    path = regular_path(path)
    value = json.loads(path.read_text())
    if value["schema"] != SCHEMA or value["feature_order"] != FEATURES:
        raise ValueError("New full-run schema/feature order required")
    if value.get("interval_policy") != INTERVAL_POLICY:
        raise ValueError(
            "Explicit mission-only operational interval policy required"
        )
    if (
        value.get("units") != "physical_common19_pack_V_A"
        or value.get("step_us") != 100000
    ):
        raise ValueError("Explicit physical units and 10 Hz required")
    if (
        value.get("vehicle_profile") != "x500_reference"
        or value.get("battery_policy") != "cmu-vah01-rpt16-only/1"
    ):
        raise ValueError(
            "This experiment is fixed to the current X500 battery policy"
        )
    if not value["runs"]:
        raise ValueError("Empty run manifest")
    actual = any(
        r.get("origin") != "synthetic_unit_fixture" for r in value["runs"]
    )
    if actual:
        check_producer(value.get("producer"), expected_producer)
        if protection is None:
            raise ValueError(
                "Actual intake requires the global protection registry"
            )
    for item in value["runs"]:
        partition(item["role"], item.get("cohort"))
        if protection is not None:
            check_protection([item], protection, usage=usage)
        if actual:
            source_evidence(item, value["producer"])
        yield path, item


def iter_manifests(
    paths,
    *,
    expected_producer=None,
    max_frames=10000,
    protection=None,
    usage="input_check",
):
    """Audit all small lineage records first, then decode/release one run at a time.

    Pass every role/cohort manifest for the campaign together. A single role file
    cannot establish independence from unseen files. No scaler/model is fitted.
    """
    records = [
        entry
        for path in paths
        for entry in manifest_rows(path, expected_producer, protection, usage)
    ]
    validate_lineages([item for _, item in records])
    for path, item in records:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Dataset file must stay inside its sealed root")
        source = regular_path(path.parent / relative)
        if file_sha(source) != item["sha256"]:
            raise ValueError("Run file hash mismatch")
        headers = npz_headers(source)
        if set(headers) != {"values", "time_us"}:
            raise ValueError(
                "Only values/time_us allowed; truth/context are not inputs"
            )
        shape = headers["values"]["shape"]
        if len(shape) != 2:
            raise ValueError(
                "Bounded float[T,19]/int64 times required before decoding"
            )
        n = shape[0]
        if (
            not 1 <= n <= max_frames
            or headers["values"]["shape"] != (n, 19)
            or np.dtype(headers["values"]["descr"]).kind != "f"
            or headers["time_us"]["shape"] != (n,)
            or np.dtype(headers["time_us"]["descr"]) != np.dtype("int64")
        ):
            raise ValueError(
                "Bounded float[T,19]/int64 times required before decoding"
            )
        with np.load(source, allow_pickle=False) as arrays:
            if set(arrays.files) != {"values", "time_us"}:
                raise ValueError(
                    "Only values/time_us allowed; truth/context are not inputs"
                )
            interval = item.get("mission_interval", {})
            start, end = (
                interval.get("start_us"),
                interval.get("end_us_exclusive"),
            )
            if (
                interval.get("policy") != INTERVAL_POLICY
                or type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
                or not len(arrays["time_us"])
                or arrays["time_us"][0] != ((start + 99999) // 100000) * 100000
                or arrays["time_us"][-1] != ((end - 1) // 100000) * 100000
            ):
                raise ValueError(
                    "Complete declared mission interval required; no quality trimming"
                )
            run = Run(
                item["run_id"],
                item["role"],
                tuple(item["lineage"]),
                item["health"],
                item["origin"],
                arrays["values"],
                arrays["time_us"],
                item.get("cohort"),
            )
        # Match the model's initial100 policy before reporting intake availability.
        if len(run.values) <= 100:
            raise ValueError("Run unavailable: no post-initial100 observations")
        yield run.validate()
        del run


def load_manifest(
    path, *, expected_producer=None, protection=None, usage="input_check"
):
    """Small development convenience; CLI uses the campaign-wide iterator instead."""
    rows = list(manifest_rows(path, expected_producer, protection, usage))
    if len(rows) > 16:
        raise ValueError(
            "Use iter_manifests for more than16 runs; never accumulate the campaign"
        )
    return list(
        iter_manifests(
            [path],
            expected_producer=expected_producer,
            protection=protection,
            usage=usage,
        )
    )
