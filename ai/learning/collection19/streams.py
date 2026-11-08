"""Bounded canonical and NPZ IO; no raw collection, models or dataset
concatenation.
"""

import json
import zipfile
from pathlib import Path

import numpy as np

from capstone_ai.mapping import FEATURES
from execution.prepare import npz_headers
from learning.history19.pilot import regular


def grid(row):
    interval = row["mission_interval"]
    first = ((interval["start_us"] + 99999) // 100000) * 100000
    last = ((interval["end_us_exclusive"] - 1) // 100000) * 100000
    n = (last - first) // 100000 + 1
    if not 101 <= n <= 10000:
        raise ValueError(
            "Nonempty full mission after initial100 required; no duration crop"
        )
    return first, last, n


def canonical_blocks(path, row, *, chunk=256):
    if not 1 <= chunk <= 8192:
        raise ValueError("Bounded canonical block size required")
    first, last, n = grid(row)
    interval = row["mission_interval"]
    previous, segment, count = first - 100000, None, 0
    values, times = [], []
    with regular(path).open(encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 65536:
                raise ValueError("Canonical frame exceeds64KiB")
            f = json.loads(line)
            if (
                f.get("run_id") != row["run_id"]
                or type(f.get("time_us")) is not int
            ):
                raise ValueError("Canonical raw/time identity mismatch")
            if (
                not interval["start_us"]
                <= f["time_us"]
                < interval["end_us_exclusive"]
            ):
                continue
            sid = f.get("segment_id")
            if (
                f.get("feature_order") != FEATURES
                or f.get("feature_profile") != "common19_ctx0"
                or f.get("clock_id") != interval["clock_id"]
                or type(sid) is not int
                or sid < 0
                or (segment is not None and sid != segment)
                or f["time_us"] != previous + 100000
                or f.get("operating_state") != "airborne"
                or any(
                    f["quality"].get(k) != "valid" or f["values"].get(k) is None
                    for k in FEATURES
                )
                or not f.get("lineage")
                or any(
                    not 0
                    <= p["sample_time_us"]
                    <= p["available_time_us"]
                    <= f["time_us"]
                    for p in f["lineage"].values()
                )
            ):
                raise ValueError(
                    "Canonical mission clock/gap/quality/causality mismatch"
                )
            vector = [f["values"][k] for k in FEATURES]
            if not np.isfinite(vector).all():
                raise ValueError(
                    "Missing/nonfinite frame cannot enter statistics"
                )
            previous, segment = f["time_us"], sid
            count += 1
            values.append(vector)
            times.append(previous)
            if len(values) == chunk:
                yield (
                    np.asarray(values, dtype=np.float64),
                    np.asarray(times, dtype=np.int64),
                    segment,
                )
                values, times = [], []
    if count != n or previous != last:
        raise ValueError(
            "Canonical mission lost original start/LAND tail frames"
        )
    if values:
        yield (
            np.asarray(values, dtype=np.float64),
            np.asarray(times, dtype=np.int64),
            segment,
        )


def canonical_matrix(path, row):
    _, _, n = grid(row)
    values, times = np.empty((n, 19)), np.empty(n, dtype=np.int64)
    offset, segment = 0, None
    for block, stamps, segment in canonical_blocks(path, row):
        values[offset : offset + len(block)] = block
        times[offset : offset + len(block)] = stamps
        offset += len(block)
    return values, times, segment


def array_blocks(path, key, *, chunk=64):
    if not 1 <= chunk <= 8192:
        raise ValueError("Bounded array block size required")
    path = regular(path)
    headers = npz_headers(path)
    dtype, shape = np.dtype(headers[key]["descr"]), headers[key]["shape"]
    if dtype.kind not in "fiuUS" or not shape:
        raise ValueError("Bounded numeric/string array rows required")
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        actual_shape, fortran, actual_dtype = reader(stream)
        if (
            fortran
            or tuple(actual_shape) != tuple(shape)
            or actual_dtype != dtype
        ):
            raise ValueError("Canonical C-order NPY header required")
        width = int(np.prod(shape[1:]))
        for start in range(0, shape[0], chunk):
            count = min(chunk, shape[0] - start)
            body = stream.read(count * width * dtype.itemsize)
            if len(body) != count * width * dtype.itemsize:
                raise ValueError("Truncated array rows")
            yield (
                start,
                np.frombuffer(body, dtype=dtype).reshape(count, *shape[1:]),
            )
        if stream.read(1):
            raise ValueError("Trailing array bytes")


def write_npz(path, specs):
    """Write normal reader-compatible NPZ with bounded chunks and ZIP64, no
    scratch copy.
    """
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Preserve existing dataset members")
    with zipfile.ZipFile(
        path, "x", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for name, (shape, dtype, blocks) in specs.items():
            dtype = np.dtype(dtype)
            if dtype.hasobject:
                raise ValueError("Object arrays forbidden")
            count = 0
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array_header_2_0(
                    stream,
                    dict(
                        descr=np.lib.format.dtype_to_descr(dtype),
                        fortran_order=False,
                        shape=tuple(shape),
                    ),
                )
                for block in blocks:
                    a = np.asarray(block, dtype=dtype, order="C")
                    if a.shape[1:] != tuple(shape[1:]) or a.dtype.hasobject:
                        raise ValueError("Dataset block shape/dtype changed")
                    count += len(a)
                    if count > shape[0]:
                        raise ValueError("Dataset member contains excess rows")
                    stream.write(a.tobytes(order="C"))
            if count != shape[0]:
                raise ValueError("Dataset member row coverage mismatch")
    headers = npz_headers(path)
    if set(headers) != set(specs):
        raise ValueError("Written dataset member inventory mismatch")


class Moments:
    """Accumulate per-feature mean and variance without retaining all frames."""

    def __init__(self):
        self.count, self.mean, self.m2 = 0, np.zeros(19), np.zeros(19)

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 19
            or len(values) == 0
            or not np.isfinite(values).all()
        ):
            raise ValueError("Finite unique train frame blocks required")
        n = len(values)
        center = values.mean(0)
        delta, total = center - self.mean, self.count + n
        self.m2 += ((values - center) ** 2).sum(
            0
        ) + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def finish(self):
        if self.count <= 0:
            raise ValueError("No train frames; scaler cannot be produced")
        return self.mean.copy(), np.maximum(np.sqrt(self.m2 / self.count), 1e-5)
