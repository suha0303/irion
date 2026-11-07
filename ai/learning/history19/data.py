"""New dataset contract: X/Y and lineage only, without legacy dataset conversion."""

import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from capstone_ai.mapping import FEATURES, TARGETS
from execution.common import child, read, sha
from execution.prepare import npz_headers

from learning.admission19 import (
    MISSION_POLICY,
    ONLINE_UNITS,
    check_producer,
    check_protection,
    source_evidence,
    validate_lineages,
)

SCHEMA = "drone-ai.history19.dataset/1"
ARRAYS = {"X", "Y", "run_id", "clock_id", "segment_id", "origin_us"}


def inspect_run(
    manifest_path,
    *,
    expected_producer=None,
    protection=None,
    usage="input_check",
):
    """Read-only per-run intake for preliminary/development/final, without a scaler.

    Does not load the fit dataset, fit statistics, forecast, or evaluate detection.
    Campaign-wide lineage and unavailable-run inventory must also be audited.
    """
    path = Path(manifest_path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Dataset symlinks forbidden")
    value = read(path)
    fixed = dict(
        schema="drone-ai.history19.run/1",
        units=ONLINE_UNITS,
        interval_policy=MISSION_POLICY,
        feature_order=FEATURES,
        target_order=TARGETS,
        step_us=100000,
        history=100,
        future=10,
        emit_every_us=1000000,
    )
    if any(value.get(k) != v for k, v in fixed.items()):
        raise ValueError(
            "Explicit new per-run history19 units/order/mission policy required"
        )
    row = value["run"]
    validate_lineages([row])
    if row.get("origin") != "synthetic_unit_fixture":
        if protection is None:
            raise ValueError(
                "Actual intake requires the global protection registry"
            )
        check_producer(value.get("producer"), expected_producer)
        source_evidence(row, value["producer"])
    if protection is not None:
        check_protection([row], protection, usage=usage)
    source = child(path.parent, value["file"]["path"])
    if sha(source) != value["file"]["sha256"]:
        raise ValueError("Sealed online run changed")
    headers = npz_headers(source)
    if set(headers) != ARRAYS or len(headers["X"]["shape"]) != 3:
        raise ValueError("Only X/Y and window identities allowed")
    n = headers["X"]["shape"][0]
    if (
        not 1 <= n <= 10000
        or headers["X"]["shape"] != (n, 100, 19)
        or headers["Y"]["shape"] != (n, 10, 4)
    ):
        raise ValueError("Bounded per-run X[N,100,19]/Y[N,10,4] required")
    for key in ARRAYS - {"X", "Y"}:
        dtype = np.dtype(headers[key]["descr"])
        kinds = "US" if key in {"run_id", "clock_id"} else "iu"
        if (
            headers[key]["shape"] != (n,)
            or dtype.kind not in kinds
            or dtype.itemsize > 2048
        ):
            raise ValueError(
                "Bounded typed window identities required before decoding"
            )
    with np.load(source, allow_pickle=False) as archive:
        ids = {k: archive[k] for k in ARRAYS - {"X", "Y"}}
    if any(v.shape != (n,) for v in ids.values()):
        raise ValueError("One window identity per origin required")
    if any(
        ids[k].dtype.kind not in "US" for k in ("run_id", "clock_id")
    ) or any(
        ids[k].dtype.kind not in "iu" for k in ("segment_id", "origin_us")
    ):
        raise ValueError("Typed window identities required")
    mission = row["mission_interval"]
    origins = ids["origin_us"]
    if (
        set(ids["run_id"]) != {row["run_id"]}
        or set(ids["clock_id"]) != {mission["clock_id"]}
        or np.any(ids["segment_id"] < 0)
        or np.any(origins % 1000000)
        or np.any(np.diff(origins) <= 0)
        or np.any(origins - 9900000 < mission["start_us"])
        or np.any(origins + 1000000 >= mission["end_us_exclusive"])
    ):
        raise ValueError(
            "Origins/history/targets must stay in actual mission and clock"
        )
    with zipfile.ZipFile(source) as archive:
        for key in ("X", "Y"):
            dtype, shape = (
                np.dtype(headers[key]["descr"]),
                headers[key]["shape"],
            )
            if dtype.kind != "f":
                raise ValueError(
                    "Floating online inputs and arrived targets required"
                )
            with archive.open(key + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                reader(stream)
                width = int(np.prod(shape[1:]))
                for offset in range(0, n, 512):
                    count = min(512, n - offset)
                    body = stream.read(count * width * dtype.itemsize)
                    if (
                        len(body) != count * width * dtype.itemsize
                        or not np.isfinite(
                            np.frombuffer(body, dtype=dtype)
                        ).all()
                    ):
                        raise ValueError(
                            "Missing/nonfinite or truncated online input/arrived target"
                        )
                if stream.read(1):
                    raise ValueError("Trailing online bytes")
    return dict(
        status="INPUT_SCHEMA_VALIDATED_NOT_TRAINED",
        origin=row["origin"],
        run_id=row["run_id"],
        cohort=row.get("cohort"),
        windows=n,
        target_points=10 * n,
        scope="supplied_run_only_not_raw_coverage_or_independence",
        physical_eligibility_verified=False,
    )


def verify_train_frames(path, train_ids, scaler, progress=None, *, chunk=8192):
    """Recompute existing train statistics in bounded chunks, without fitting/saving a scaler."""
    fields = ("values", "run_id", "clock_id", "segment_id", "time_us")
    headers = npz_headers(path)
    if set(headers) != set(fields):
        raise ValueError(
            "Train scaler requires19-column frames and identity only"
        )
    n = headers["values"]["shape"][0]
    if not n or headers["values"]["shape"] != (n, 19):
        raise ValueError("Finite normal-train frames required")
    for field in fields:
        shape, dtype = (
            headers[field]["shape"],
            np.dtype(headers[field]["descr"]),
        )
        expected_kind = (
            "f"
            if field == "values"
            else "US"
            if field in {"run_id", "clock_id"}
            else "iu"
        )
        if (
            field != "values" and shape != (n,)
        ) or dtype.kind not in expected_kind:
            raise ValueError("Train frame dtype/identity mismatch")
    count, mean, m2, previous, seen = 0, np.zeros(19), np.zeros(19), {}, set()
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(path))
        streams = {
            k: stack.enter_context(archive.open(k + ".npy")) for k in fields
        }
        for stream in streams.values():
            version = np.lib.format.read_magic(stream)
            reader = (
                np.lib.format.read_array_header_1_0
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            reader(stream)
        for offset in range(0, n, chunk):
            if progress is not None:
                progress.check()
            size, arrays = min(chunk, n - offset), {}
            for field, stream in streams.items():
                dtype = np.dtype(headers[field]["descr"])
                width = 19 if field == "values" else 1
                body = stream.read(size * width * dtype.itemsize)
                if len(body) != size * width * dtype.itemsize:
                    raise ValueError("Truncated train frame array")
                arrays[field] = np.frombuffer(body, dtype=dtype).reshape(
                    (size, 19) if width == 19 else (size,)
                )
            values = arrays["values"].astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Finite normal-train frames required")
            for run, clock, segment, stamp in zip(
                *(arrays[k] for k in fields[1:])
            ):
                identity = str(run), str(clock), int(segment)
                if str(run) not in train_ids or not str(clock) or segment < 0:
                    raise ValueError("Train frame membership mismatch")
                if stamp <= previous.get(identity, -1):
                    raise ValueError("Duplicate/unordered train scaler frame")
                previous[identity] = int(stamp)
                seen.add(str(run))
            block_mean = values.mean(0)
            delta = block_mean - mean
            total = count + size
            m2 += ((values - block_mean) ** 2).sum(
                0
            ) + delta**2 * count * size / total
            mean += delta * size / total
            count = total
        if any(stream.read(1) for stream in streams.values()):
            raise ValueError("Trailing train frame bytes")
    scale = np.maximum(np.sqrt(m2 / count), 1e-5)
    if (
        seen != set(train_ids)
        or not np.allclose(scaler["x_mean"], mean, atol=1e-10, rtol=1e-10)
        or not np.allclose(scaler["x_scale"], scale, atol=1e-10, rtol=1e-10)
    ):
        raise ValueError("Scaler differs from unique normal_train frames")


def load(
    manifest_path,
    *,
    cache=None,
    progress=None,
    expected_producer=None,
    protection=None,
):
    path = Path(manifest_path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Dataset symlinks forbidden")
    manifest = read(path)
    fixed = dict(
        schema=SCHEMA,
        feature_profile="common19_ctx0",
        feature_order=FEATURES,
        target_order=TARGETS,
        history=100,
        future=10,
        step_us=100000,
        emit_every_us=1000000,
        scaler_scope="normal_train_unique_valid_frames",
    )
    if any(
        type(manifest.get(k)) is not type(v) or manifest[k] != v
        for k, v in fixed.items()
    ):
        raise ValueError(
            "New history19 dataset contract required; no legacy conversion"
        )
    runs = manifest["runs"]
    if not runs or len({r["run_id"] for r in runs}) != len(runs):
        raise ValueError("Unique nonempty run identities required")
    if {r["role"] for r in runs} != {"normal_train", "normal_validation"}:
        raise ValueError(
            "Fit manifest must contain normal train/validation only"
        )
    fixture = manifest.get("fixture_only") is True
    if fixture and any(r.get("origin") == "px4_sitl" for r in runs):
        raise ValueError(
            "Actual SITL data cannot bypass admission as a fixture"
        )
    if not fixture:
        if protection is None:
            raise ValueError(
                "Actual fit intake requires the global protection registry"
            )
        if (
            manifest.get("units") != ONLINE_UNITS
            or manifest.get("interval_policy") != MISSION_POLICY
        ):
            raise ValueError(
                "Explicit transformed online units and mission policy required"
            )
        check_producer(manifest.get("producer"), expected_producer)
    partitions = {}
    for row in runs:
        if row.get("health_eligibility") != "confirmed_healthy":
            raise ValueError("Confirmed normal fit runs required")
        ids = [
            ("run", row["run_id"]),
            ("run", row["raw_flight_id"]),
            ("pair", row["paired_case_id"]),
            ("group", row["group_id"]),
        ]
        ids += [("run", p) for p in row.get("previous_attempts", [])]
        if row.get("parent_attempt_id"):
            ids.append(("run", row["parent_attempt_id"]))
        if row.get("retry_family_id"):
            ids.append(("retry", row["retry_family_id"]))
        if row.get("raw_sha256"):
            ids.append(("raw-sha256", row["raw_sha256"]))
        if not fixture:
            protected_row = {**row, "lineage": [k + ":" + v for k, v in ids]}
            check_protection([protected_row], protection, usage="fit")
            source_evidence(protected_row, manifest["producer"])
        for identity in ids:
            if (
                not identity[1]
                or partitions.setdefault(identity, row["role"]) != row["role"]
            ):
                raise ValueError("Cross-split raw/paired/group/retry leakage")
    train_ids = sorted(r["run_id"] for r in runs if r["role"] == "normal_train")
    if sorted(manifest["scaler_fit_run_ids"]) != train_ids:
        raise ValueError("Scaler membership must match normal_train only")
    if set(manifest["files"]) != {
        "normal_train",
        "normal_validation",
        "scaler",
        "normal_train_frames",
    }:
        raise ValueError("Only normal fit arrays and their scaler may be read")
    paths = {}
    for role, ref in manifest["files"].items():
        source = child(path.parent, ref["path"])
        if source.is_symlink() or sha(source) != ref["sha256"]:
            raise ValueError("Sealed dataset member changed: " + role)
        paths[role] = source
    with np.load(paths["scaler"], allow_pickle=False) as source:
        scaler = {k: source[k].copy() for k in source.files}
    if set(scaler) != {"x_mean", "x_scale", "y_mean", "y_scale"}:
        raise ValueError("History19 scaler has X/Y statistics only; no C")
    for key, width in (
        ("x_mean", 19),
        ("x_scale", 19),
        ("y_mean", 4),
        ("y_scale", 4),
    ):
        if scaler[key].shape != (width,) or not np.isfinite(scaler[key]).all():
            raise ValueError("Invalid scaler shape/values")
        if key.endswith("scale") and np.any(scaler[key] < 1e-5):
            raise ValueError("Positive train scale floor1e-5 required")
    indices = [FEATURES.index(name) for name in TARGETS]
    for suffix in ("mean", "scale"):
        if not np.array_equal(
            scaler["y_" + suffix], scaler["x_" + suffix][indices]
        ):
            raise ValueError(
                "Target scaler must use the corresponding train X columns"
            )
    verify_train_frames(
        paths["normal_train_frames"], train_ids, scaler, progress
    )
    result = {}
    for role in ("normal_train", "normal_validation"):
        headers = npz_headers(paths[role])
        with np.load(paths[role], allow_pickle=False) as source:
            if set(source.files) != ARRAYS:
                raise ValueError(
                    "New NPZ requires X/Y and identity arrays only; C/truth forbidden"
                )
            arrays = {k: source[k].copy() for k in ARRAYS - {"X", "Y"}}
        n = headers["X"]["shape"][0]
        if (
            n == 0
            or headers["X"]["shape"] != (n, 100, 19)
            or headers["Y"]["shape"] != (n, 10, 4)
        ):
            raise ValueError("Expected positive X[N,100,19], Y[N,10,4]")
        for key in ARRAYS - {"X", "Y"}:
            if arrays[key].shape != (n,):
                raise ValueError("One identity per window required")
        if (
            arrays["origin_us"].dtype.kind not in "iu"
            or arrays["segment_id"].dtype.kind not in "iu"
        ):
            raise ValueError("Integer origin/segment required")
        if np.any(arrays["origin_us"] % 1000000):
            raise ValueError("One-second forecast origin grid required")
        if any(
            arrays[k].dtype.kind not in "US" for k in ("run_id", "clock_id")
        ):
            raise ValueError("String run/clock identities required")
        if set(arrays["run_id"]) != {
            r["run_id"] for r in runs if r["role"] == role
        }:
            raise ValueError("Fit windows differ from role membership")
        identities = list(
            zip(
                *(
                    arrays[k].tolist()
                    for k in ("run_id", "clock_id", "segment_id", "origin_us")
                )
            )
        )
        if len(set(identities)) != n:
            raise ValueError("Duplicate window identity")
        result[role] = {}
        destination = Path(cache) / role if cache is not None else None
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(paths[role]) as archive:
            for key, prefix in (("X", "x"), ("Y", "y")):
                shape, dtype = (
                    headers[key]["shape"],
                    np.dtype(headers[key]["descr"]),
                )
                if dtype.kind != "f":
                    raise ValueError("Floating X/Y required")
                values = (
                    np.lib.format.open_memmap(
                        destination / (key + ".npy"),
                        mode="w+",
                        dtype=np.float32,
                        shape=shape,
                    )
                    if destination is not None
                    else np.empty(shape, dtype=np.float32)
                )
                with archive.open(key + ".npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    reader(stream)
                    width = int(np.prod(shape[1:]))
                    for offset in range(0, n, 512):
                        if progress is not None:
                            progress.check()
                        count = min(512, n - offset)
                        body = stream.read(count * width * dtype.itemsize)
                        if len(body) != count * width * dtype.itemsize:
                            raise ValueError("Truncated fit array")
                        raw = np.frombuffer(body, dtype=dtype).reshape(
                            count, *shape[1:]
                        )
                        if not np.isfinite(raw).all():
                            raise ValueError(
                                "Missing/nonfinite inputs or arrived targets forbidden"
                            )
                        normalized = (raw - scaler[prefix + "_mean"]) / scaler[
                            prefix + "_scale"
                        ]
                        converted = normalized.astype(np.float32)
                        if not np.isfinite(converted).all():
                            raise ValueError("Nonfinite normalized input")
                        values[offset : offset + count] = converted
                    if stream.read(1):
                        raise ValueError("Trailing fit bytes")
                if destination is not None:
                    values.flush()
                result[role][key] = values
    binding = dict(
        data={
            k: v["sha256"]
            for k, v in manifest["files"].items()
            if k != "scaler"
        },
        scaler=manifest["files"]["scaler"]["sha256"],
        manifest_sha256=sha(path),
    )
    return result, scaler, binding


def arrived_samples(engine, mission=None):
    """Join future targets after issuing history-only windows; preserve exclusions."""
    from capstone_ai.dataset import feature_transform, target_transform

    lookup = {
        (f["run_id"], f["clock_id"], f["segment_id"], f["time_us"]): f
        for f in engine.frames
    }
    if len(lookup) != len(engine.frames):
        raise ValueError("Duplicate canonical frame identity")
    selected, excluded = [], []
    if mission is not None and (
        mission.get("policy") != MISSION_POLICY
        or type(mission.get("start_us")) is not int
        or type(mission.get("end_us_exclusive")) is not int
        or not 0 <= mission["start_us"] < mission["end_us_exclusive"]
        or not mission.get("clock_id")
    ):
        raise ValueError(
            "Explicit actual mission/landing interval and clock required"
        )
    for window in engine.windows:
        if mission is not None:
            reason = None
            if window["clock_id"] != mission["clock_id"]:
                reason = "mission_clock_mismatch"
            elif window["history_times_us"][0] < mission["start_us"]:
                reason = "history_outside_mission"
            elif window["target_times_us"][-1] >= mission["end_us_exclusive"]:
                reason = "target_outside_mission"
            if reason:
                excluded.append(
                    dict(
                        run_id=window["run_id"],
                        origin_us=window["origin_us"],
                        reason=reason,
                    )
                )
                continue
        scope = (window["run_id"], window["clock_id"], window["segment_id"])
        target = [lookup.get((*scope, t)) for t in window["target_times_us"]]
        if any(
            f is None or any(f["quality"][n] != "valid" for n in TARGETS)
            for f in target
        ):
            excluded.append(
                dict(
                    run_id=window["run_id"],
                    origin_us=window["origin_us"],
                    reason="missing_or_nonvalid_future_target",
                )
            )
            continue
        selected.append(
            (window, [[f["values"][n] for n in TARGETS] for f in target])
        )
    if not selected:
        raise ValueError(
            "No arrived valid target windows; never replace with zero-error targets"
        )
    windows, targets = zip(*selected)
    arrays = dict(
        X=feature_transform([w["X"] for w in windows], engine.config),
        Y=target_transform(targets),
    )
    for key in ("run_id", "clock_id", "segment_id", "origin_us"):
        arrays[key] = np.asarray([w[key] for w in windows])
    return arrays, excluded
