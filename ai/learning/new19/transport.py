"""Decode the existing capture datagrams without changing producer files."""

from pathlib import Path
from struct import Struct

import numpy as np

from capstone_ai.mapping import TARGETS
from localflight.mavlink_input import Mapper
from localflight.prepare import CausalSampler

from .data import STEP_US, digest, read, write

MAGIC = b"MLUDP001"
HEADER = Struct("!QQ4sHH")


def datagrams(path):
    with Path(path).open("rb") as raw:
        if raw.read(len(MAGIC)) != MAGIC:
            raise ValueError("Expected original MLUDP001 capture")
        previous = -1
        while header := raw.read(HEADER.size):
            if len(header) != HEADER.size:
                raise ValueError("Truncated datagram header")
            receipt, _utc, address, port, size = HEADER.unpack(header)
            payload = raw.read(size)
            if len(payload) != size or receipt < previous:
                raise ValueError("Truncated datagram or reversed receipt clock")
            previous = receipt
            yield receipt, (address, port), payload


class DatagramMapper:
    def __init__(self, profile, source_system, source_component, imu_id):
        from pymavlink.dialects.v20 import common

        if (
            profile.get("autopilot"),
            profile.get("rotor_count"),
            profile.get("series_cells"),
            profile.get("capacity_ah"),
        ) != ("px4", 4, 4, 12):
            raise ValueError(
                "Explicit PX4 X500 four-motor, 4S, 12 Ah profile required"
            )
        if not all(
            type(value) is int and 1 <= value <= 255
            for value in (source_system, source_component)
        ):
            raise ValueError(
                "Explicit MAVLink system and component IDs required"
            )
        if type(imu_id) is not int or not 0 <= imu_id <= 255:
            raise ValueError("Explicit IMU ID required")
        self.mapper = Mapper(
            dict(
                profile=profile,
                source_system=source_system,
                source_component=source_component,
                imu_message="HIGHRES_IMU",
                imu_id=imu_id,
            )
        )
        self.decoder = common.MAVLink(None)
        self.decoder.robust_parsing = True
        self.recent = {}
        self.duplicates = 0
        self.bad = 0

    def received(self, receipt, peer, payload):
        """Yield source-bound, deduplicated messages with original receipt times."""
        for msg in self.decoder.parse_buffer(payload) or ():
            name = msg.get_type()
            if name == "BAD_DATA":
                self.bad += 1
                continue
            source = (msg.get_srcSystem(), msg.get_srcComponent())
            if source != (
                self.mapper.request["source_system"],
                self.mapper.request["source_component"],
            ):
                continue
            key = (*peer, *source, msg.get_seq())
            frame = bytes(msg.get_msgbuf())
            prior = self.recent.get(key)
            if (
                prior is not None
                and frame == prior[1]
                and 0 <= receipt - prior[0] <= 200_000_000
            ):
                self.duplicates += 1
                continue
            self.recent[key] = receipt, frame
            yield dict(
                source_system=source[0],
                source_component=source[1],
                receive_monotonic_ns=receipt,
                message_name=name,
                fields=msg.to_dict(),
            )

    def accept(self, receipt, peer, payload):
        updates = []
        for raw in self.received(receipt, peer, payload):
            updates.extend(self.mapper.accept(dict(type="mavlink", raw=raw)))
        return updates


def flight_boundaries(execution, phases):
    """Read ordered flight boundaries on the collector's monotonic clock."""
    events = read(execution)
    stamps = []
    for state in phases:
        times = [
            row["wall_time"] for row in events if row.get("state") == state
        ]
        if len(times) != 1:
            raise ValueError(f"Unique actual {state} boundary required")
        stamps.append(round(times[0] * 1e9))
    if any(end <= start for start, end in zip(stamps, stamps[1:])):
        raise ValueError("Reversed flight interval")
    return tuple(stamps)


def prepare(
    capture,
    profile,
    execution,
    output,
    *,
    source_system,
    source_component,
    imu_id,
):
    capture, output = Path(capture), Path(output)
    mapper = DatagramMapper(
        read(profile), source_system, source_component, imu_id
    )
    start_ns, end_ns = flight_boundaries(execution, ("airborne", "landing"))
    raw_sha = digest(capture)
    events = []
    for receipt, peer, payload in datagrams(capture):
        if receipt >= end_ns:
            break
        events.extend(mapper.accept(receipt, peer, payload))
    clock = mapper.mapper
    if (
        clock.base is None
        or not clock.base <= start_ns < clock.last_receipt < end_ns
        or end_ns - clock.last_receipt > STEP_US * 1000
    ):
        raise ValueError("Capture does not cover the actual mission interval")
    sampler = CausalSampler(require_airborne=True, fill_gaps=True)
    rows, stamps, exclusions, boundaries = sampler.sample(
        events, 0, (end_ns - clock.base + 999) // 1000
    )
    stamps = np.asarray(stamps, dtype=np.int64)
    values = np.asarray(rows, dtype=np.float32)
    selected = stamps * 1000 + clock.base >= start_ns
    valid = np.array([not reasons for reasons in exclusions])
    segments = np.full(len(stamps), -1, dtype=np.int64)
    segment, was_valid = -1, False
    for i, ready in enumerate(valid):
        if ready:
            if not was_valid or i in boundaries:
                segment += 1
            segments[i] = segment
        was_valid = ready
    observed = np.broadcast_to(valid[:, None], (len(valid), 4)).copy()
    for i, target in enumerate(TARGETS):
        observed[:, i] &= ~np.isin(
            stamps, sampler.filled_target_times.get(target, ())
        )
    if not selected.any() or digest(capture) != raw_sha:
        raise ValueError("Empty mission interval or changed capture")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "run.npz"
    np.savez_compressed(
        path,
        values=values[selected],
        time_us=stamps[selected],
        segment=segments[selected],
        observed=observed[selected],
    )
    result = dict(
        schema="drone-ai.new19.prepared/1",
        raw_sha256=raw_sha,
        profile_sha256=digest(profile),
        execution_sha256=digest(execution),
        sha256=digest(path),
        frames=int(selected.sum()),
        valid_frames=int((selected & valid).sum()),
        segments=len(set(segments[selected & valid])),
        duplicate_packets=mapper.duplicates,
        decoder_bad=mapper.bad,
        clock_resets=clock.resets,
        clock="receipt_monotonic",
        receipt_origin_ns=clock.base,
    )
    write(output / "receipt.json", result)
    return result
