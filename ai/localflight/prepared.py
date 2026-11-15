"""Strict boundary for new, separately audited real-flight common19 data.

This does not accept legacy simulator manifests or reinterpret onboard logs.
Unknown health is sufficient for prediction but never for normal adaptation.
"""

from pathlib import Path

import numpy as np

from capstone_ai.mapping import FEATURES
from execution.prepare import npz_headers

from .catalog import digest, read_json
from .profiles import eligibility

REVISION = "real-common19-segmented-v2"


def segment_ranges(metadata, length):
    segments = metadata.get("segments", [dict(start_index=0, end_index=length)])
    cursor = 0
    ranges = []
    for segment in segments:
        a, b = segment["start_index"], segment["end_index"]
        if (
            type(a) is not int
            or type(b) is not int
            or a != cursor
            or not a + 110 <= b <= length
        ):
            raise ValueError("Invalid contiguous segment boundaries")
        ranges.append((a, b))
        cursor = b
    if cursor != length:
        raise ValueError("Segment boundaries do not cover prepared frames")
    return ranges


def load(path, *, normal=False):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink input is not admitted")
    m = read_json(path)
    if (
        m.get("schema") != "drone.real-common19/1"
        or m.get("feature_order") != FEATURES
        or m.get("preprocessing_revision") != REVISION
        or m.get("units") != "physical_pack_V_A"
        or m.get("origin")
        not in {"px4_real", "ardupilot_real", "synthetic_unit_fixture"}
    ):
        raise ValueError(
            "A separately prepared real common19 manifest is required"
        )
    p = m["profile"]
    e = eligibility(p)
    if not e["profile_complete"] or not e["model_structure_compatible"]:
        raise ValueError(
            "Aircraft mapping and nameplate are incomplete or unsupported"
        )
    if m.get("mapping_verified") is not True or not m.get("mapping_evidence"):
        raise ValueError("An audited source-to-common19 mapping is required")
    if m.get("role") not in {
        "adaptation",
        "development",
        "evaluation",
        "calibration",
    }:
        raise ValueError("Explicit flight role required")
    if normal and (
        m.get("health") != "confirmed_normal" or not m.get("health_evidence")
    ):
        raise ValueError(
            (
                "Confirmed normal flight and its evidence required; "
                "probability is insufficient"
            )
        )
    if not m.get("source_sha256") or len(m["source_sha256"]) != 64:
        raise ValueError("Original flight identity required")
    if m["origin"] != "synthetic_unit_fixture":
        raw = path.parent / m["source_path"]
        if (
            not raw.is_relative_to(path.parent)
            or ".." in Path(m["source_path"]).parts
        ):
            raise ValueError("Raw path must stay inside its flight package")
        if (
            any(p.is_symlink() for p in (raw, *raw.parents))
            or digest(raw) != m["source_sha256"]
        ):
            raise ValueError("Original log changed")
    name = m["file"]["path"]
    data_path = path.parent / name
    if (
        Path(name).name != name
        or data_path.is_symlink()
        or digest(data_path) != m["file"]["sha256"]
    ):
        raise ValueError("Prepared array integrity mismatch")
    headers = npz_headers(data_path)
    if set(headers) != {"values", "time_us"}:
        raise ValueError("Only values/time_us arrays are admitted")
    shape = headers["values"]["shape"]
    if len(shape) != 2 or shape[1] != 19 or not 110 <= shape[0] <= 10000:
        raise ValueError("A contiguous interval of 110..10000 frames required")
    if headers["time_us"]["shape"] != (shape[0],):
        raise ValueError("Time axis length mismatch")
    if (
        np.dtype(headers["values"]["descr"]).str not in {"<f4", "<f8"}
        or np.dtype(headers["time_us"]["descr"]).str != "<i8"
    ):
        raise ValueError("Only float32/64 values and int64 timestamps admitted")
    with np.load(data_path, allow_pickle=False) as z:
        values, times = z["values"], z["time_us"]
    ranges = segment_ranges(m, len(times))
    if (
        not np.isfinite(values).all()
        or np.any(times < 0)
        or any(not np.all(np.diff(times[a:b]) == 100000) for a, b in ranges)
    ):
        raise ValueError(
            "Missing data/clock resets must be separated; never fill gaps"
        )
    if (
        np.any(values[:, :4] < 0)
        or np.any(values[:, :4] > 1)
        or np.any(values[:, 17] <= 0)
    ):
        raise ValueError("Invalid actuator requests or voltage")
    return m, values.astype("f8"), times


def windows(metadata, values, times, scaler):
    # Explicit nameplate conversion; the frozen training scaler itself is never
    # refit.
    physical = values.copy()
    physical[:, 17] /= metadata["profile"]["series_cells"]
    physical[:, 18] /= metadata["profile"]["capacity_ah"]
    indices = np.array(
        [
            i
            for a, b in segment_ranges(metadata, len(values))
            for i in range(a + 99, b - 10, 10)
        ]
    )
    if len(indices) == 0:
        raise ValueError("No complete causal history and arrived future target")
    x = np.stack([physical[i - 99 : i + 1] for i in indices])
    y = np.stack([physical[i + 1 : i + 11, [4, 5, 6, 17]] for i in indices])
    x = ((x - scaler["x_mean"]) / scaler["x_scale"]).astype("f4")
    y = ((y - scaler["y_mean"]) / scaler["y_scale"]).astype("f4")
    persistence = (
        (physical[indices][:, [4, 5, 6, 17]] - scaler["y_mean"])
        / scaler["y_scale"]
    ).astype("f4")
    return x, y, persistence, times[indices]


def distinct_flights(records):
    hashes = [m["source_sha256"] for m in records]
    if len(hashes) != len(set(hashes)):
        raise ValueError(
            "The same flight cannot appear in multiple roles or duplicate files"
        )
    profiles = [m["profile"] for m in records]
    if any(p != profiles[0] for p in profiles[1:]):
        raise ValueError(
            (
                "Adaptation and comparison require the same aircraft profile "
                "version"
            )
        )
    origins = {m["origin"] for m in records}
    if len(origins) > 1:
        raise ValueError(
            "Do not mix synthetic and real flights or different input sources"
        )
    policies = {m.get("timing", {}).get("operating_policy") for m in records}
    if len(policies) > 1:
        raise ValueError(
            "Do not mix different flight-state preparation policies"
        )
