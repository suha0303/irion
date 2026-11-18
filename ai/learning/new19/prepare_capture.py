"""Prepare one MAVLink capture in a disposable process and publish atomically."""

import json
import os
import sys
from pathlib import Path

import numpy as np

from .buffered import HOLD_US, REVISION, prepare
from .data import (
    STEP_US,
    digest,
    forecast_origins,
    load_run,
    origins,
    whole_admitted,
    write,
)


def scoreable_frames(run):
    """Count targets that the current forecast and scoring code can match."""
    count = len(run.time_us)
    predicted = np.zeros(count, dtype=bool)
    starts = forecast_origins(run)
    boundaries = np.r_[0, np.cumsum(run.segment[1:] != run.segment[:-1])]
    for lead in range(1, 11):
        origin = starts[starts + lead < count]
        target = origin + lead
        predicted[target[boundaries[target] == boundaries[origin]]] = True
    return int(np.count_nonzero(predicted & run.observed.all(axis=1)))


def stat_stamp(path):
    status = path.stat()
    return [
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    ]


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    job = json.load(sys.stdin)
    row = job["flight"]
    capture, execution = Path(row["wire"]), Path(row["execution"])
    profile = Path(job["profile"])
    candidate, final = Path(job["candidate"]), Path(job["final"])
    if (
        stat_stamp(capture) != row["wire_stat"]
        or digest(execution) != row["execution_sha256"]
    ):
        raise ValueError(
            "MAVLink capture or execution evidence changed before preparation"
        )
    if digest(profile) != job["profile_sha256"]:
        raise ValueError("Frozen aircraft profile changed")
    receipt = prepare(
        capture,
        profile,
        execution,
        candidate,
        source_system=row["source_system"],
        source_component=1,
        imu_id=job["imu_id"],
    )
    if (
        stat_stamp(capture) != row["wire_stat"]
        or receipt["profile_sha256"] != job["profile_sha256"]
        or receipt["execution_sha256"] != row["execution_sha256"]
        or receipt["preprocessing_revision"] != REVISION
        or receipt["hold_ms"] != HOLD_US // 1000
    ):
        raise ValueError("Input identity or preprocessing policy changed")
    run = load_run(candidate / "run.npz", row["run_id"])
    windows = len(origins(run))
    scoreable = scoreable_frames(run)
    whole = bool(whole_admitted(run))
    comparison = dict(
        plan_sha256=job["plan_sha256"],
        flight_id=row["flight_id"],
        run_id=row["run_id"],
        batch=row["batch"],
        seed=row["seed"],
        raw_sha256=receipt["raw_sha256"],
        npz_sha256=receipt["sha256"],
        receipt_sha256=digest(candidate / "receipt.json"),
        frames=receipt["frames"],
        valid_frames=receipt["valid_frames"],
        valid_takeoff_frames=receipt["valid_takeoff_frames"],
        valid_landing_frames=receipt["valid_landing_frames"],
        takeoff_missing_edge_frames=receipt["takeoff_missing_edge_frames"],
        landing_missing_edge_frames=receipt["landing_missing_edge_frames"],
        segments=receipt["segments"],
        forecast_windows=windows,
        scoreable_frames=scoreable,
        whole_run_usable=whole,
        frame_spacing_us=STEP_US,
    )
    write(candidate / "admission.json", comparison)
    for file in candidate.iterdir():
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
    sync_directory(candidate)
    if final.exists() or final.is_symlink():
        raise ValueError("An existing prepared flight cannot be overwritten")
    candidate.rename(final)
    sync_directory(final.parent)
    sync_directory(candidate.parent)


if __name__ == "__main__":
    main()
