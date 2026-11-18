"""Shared PX4 stream and replay preprocessing with a 10-second watermark."""

import heapq
from dataclasses import dataclass
from itertools import count
from pathlib import Path

import numpy as np

from capstone_ai.mapping import TARGETS
from localflight.mavlink_input import decode
from localflight.prepare import CausalSampler

from .data import STEP_US, digest, read, write
from .transport import DatagramMapper, datagrams, flight_boundaries

HOLD_US = 10_000_000
MISSION_REVISION = "px4-buffered-common19/1"
REVISION = "px4-buffered-common19/2"
BOOT_MS = {
    "ATTITUDE",
    "ATTITUDE_TARGET",
    "LOCAL_POSITION_NED",
    "POSITION_TARGET_LOCAL_NED",
}
POSSIBLE_BOOT_US = {"HIGHRES_IMU", "SERVO_OUTPUT_RAW"}


@dataclass(frozen=True)
class Frame:
    time_us: int
    values: list[float]
    segment: int
    observed: tuple[bool, ...]
    grounded: bool = False


class Px4Clock:
    """Map boot time to the receipt grid using a fixed per-epoch anchor."""

    def __init__(self):
        self.offset = None
        self.reference = None
        self.reset_candidate = None
        self.usec_boot = {}
        self.resets = 0
        self.receipt_fallbacks = 0

    def time(self, received_us, name, fields):
        if (
            name in BOOT_MS
            and (stamp := fields.get("time_boot_ms")) is not None
        ):
            boot = stamp * 1000
        elif name in POSSIBLE_BOOT_US:
            boot = fields.get("time_usec")
        else:
            boot = None

        reset = False
        if name == "LOCAL_POSITION_NED" and boot is not None:
            if self.reference is None:
                self.offset = received_us - boot
                self.reference = boot
            elif boot < self.reference - HOLD_US:
                prior = self.reset_candidate
                self.reset_candidate = (boot, received_us)
                if prior is None or not (
                    0 < boot - prior[0] <= STEP_US * 5
                    and 0 <= received_us - prior[1] <= STEP_US * 5
                ):
                    return received_us, False, False
                # Two successive increasing samples in the lower epoch distinguish
                # a reboot from one out-of-order position packet.
                self.offset = received_us - boot
                self.reference = boot
                self.reset_candidate = None
                self.usec_boot.clear()
                self.resets += 1
                reset = True
            else:
                self.reference = max(self.reference, boot)
                self.reset_candidate = None

        if boot is None or self.offset is None:
            self.receipt_fallbacks += 1
            return received_us, reset, True
        mapped = boot + self.offset
        if name in POSSIBLE_BOOT_US:
            if name not in self.usec_boot:
                self.usec_boot[name] = (
                    received_us - HOLD_US <= mapped <= received_us + STEP_US
                )
            if not self.usec_boot[name]:
                self.receipt_fallbacks += 1
                return received_us, reset, True
        if mapped > received_us + STEP_US:
            return received_us, reset, False
        return max(0, min(mapped, received_us)), reset, True


class BufferedPreprocessor:
    """Accept captured UDP payloads and publish immutable 100 ms grid frames."""

    def __init__(
        self,
        profile,
        source_system,
        source_component,
        imu_id,
        *,
        revision=REVISION,
    ):
        if revision not in {MISSION_REVISION, REVISION}:
            raise ValueError("Unsupported PX4 buffered preprocessing revision")
        self.revision = revision
        self.source = DatagramMapper(
            profile, source_system, source_component, imu_id
        )
        self.clock = Px4Clock()
        self.sampler = CausalSampler(
            require_airborne=True,
            fill_gaps=True,
            include_takeoff=revision == REVISION,
        )
        self.origin_ns = None
        self.last_receipt_ns = None
        self.watermark = -1
        self.segment = -1
        self.was_valid = False
        self.pending = []
        self.sequence = count()
        self.late = 0
        self.unaligned = 0
        self.reordered = 0
        self.latest_sample = -1

    def accept(self, receipt_ns, peer, payload):
        if (
            self.last_receipt_ns is not None
            and receipt_ns < self.last_receipt_ns
        ):
            raise ValueError("Reversed UDP receipt clock")
        self.last_receipt_ns = receipt_ns
        if self.origin_ns is None:
            self.origin_ns = receipt_ns
        received_us = (receipt_ns - self.origin_ns) // 1000
        request = self.source.mapper.request
        for item in self.source.received(receipt_ns, peer, payload):
            name, fields = item["message_name"], item["fields"]
            if name == "HEARTBEAT" and fields.get("autopilot") != 12:
                raise ValueError("PX4 HEARTBEAT required")
            updates = decode(name, fields, request["profile"], request)
            if not updates:
                continue
            sample, reset, accepted = self.clock.time(received_us, name, fields)
            if not accepted:
                self.unaligned += 1
                continue
            if sample <= max(self.watermark, received_us - HOLD_US):
                self.late += 1
                continue
            if reset:
                self.push(sample, {"__reset__": True})
            self.reordered += sample < self.latest_sample
            self.latest_sample = max(sample, self.latest_sample)
            self.push(sample, updates)
        return self.advance(receipt_ns)

    def push(self, sample, updates):
        heapq.heappush(self.pending, (sample, next(self.sequence), updates))

    def advance(self, receipt_ns):
        if self.origin_ns is None:
            return []
        watermark = (receipt_ns - self.origin_ns) // 1000 - HOLD_US
        if watermark <= self.watermark:
            return []
        self.watermark = watermark
        start = self.sampler.next_us or 0
        if start > watermark:
            return []
        events = []
        while self.pending and self.pending[0][0] <= watermark:
            t, _, updates = heapq.heappop(self.pending)
            events.append((t, t, updates))
        rows, stamps, invalid, boundaries = self.sampler.sample(
            events, start, watermark + 1
        )
        # An event newer than the last emitted grid is still needed by the next grid.
        for _, sample, updates in events[self.sampler.position :]:
            self.push(sample, updates)
        self.sampler.position = 0
        filled = {
            target: set(self.sampler.filled_target_times.pop(target, ()))
            for target in TARGETS
        }
        frames = []
        for index, (row, stamp, missing) in enumerate(
            zip(rows, stamps, invalid)
        ):
            valid = not missing
            if valid and (not self.was_valid or index in boundaries):
                self.segment += 1
            self.was_valid = valid
            observed = tuple(
                valid and stamp not in filled[target] for target in TARGETS
            )
            frames.append(
                Frame(
                    stamp,
                    row,
                    self.segment if valid else -1,
                    observed,
                    grounded="not_airborne" in missing,
                )
            )
        return frames


def flight_frames(frames, origin_ns, bounds):
    """Keep usable launch and descent observations and account for edge gaps."""
    start, airborne, descent, disarmed = (
        (stamp - origin_ns + 999) // 1000 for stamp in bounds
    )
    flight = [frame for frame in frames if start <= frame.time_us < disarmed]
    prelaunch = 0
    for frame in flight:
        if not frame.grounded or frame.time_us >= airborne:
            break
        prelaunch += 1
    flight = flight[prelaunch:]
    touchdown_tail = 0
    while flight and flight[-1].grounded and flight[-1].time_us >= descent:
        flight.pop()
        touchdown_tail += 1

    # Exclude only edge gaps within the two measured transition phases. Leave
    # internal gaps masked, and retain an entirely missing phase for admission.
    first = next(
        (i for i, frame in enumerate(flight) if frame.segment >= 0), None
    )
    launch_gap = 0
    if first is not None and flight[first].time_us < airborne:
        launch_gap = first
        flight = flight[first:]
    last = next(
        (i for i in range(len(flight) - 1, -1, -1) if flight[i].segment >= 0),
        None,
    )
    descent_gap = 0
    if last is not None and flight[last].time_us >= descent:
        descent_gap = len(flight) - last - 1
        flight = flight[: last + 1]

    launch = [frame for frame in flight if frame.time_us < airborne]
    landing = [frame for frame in flight if frame.time_us >= descent]
    return flight, dict(
        takeoff_frames=len(launch),
        valid_takeoff_frames=sum(frame.segment >= 0 for frame in launch),
        prelaunch_grounded_frames=prelaunch,
        takeoff_missing_edge_frames=launch_gap,
        landing_frames=len(landing),
        valid_landing_frames=sum(frame.segment >= 0 for frame in landing),
        landing_missing_edge_frames=descent_gap,
        grounded_tail_frames=touchdown_tail,
    )


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
    """Replay original receive times through the same incremental processor."""
    processor = BufferedPreprocessor(
        read(profile), source_system, source_component, imu_id
    )
    bounds = flight_boundaries(
        execution, ("armed", "airborne", "landing", "disarmed")
    )
    start_ns, _, _, end_ns = bounds
    raw_sha = digest(capture)
    frames = []
    drain_ns = end_ns + HOLD_US * 1000
    for receipt_ns, peer, payload in datagrams(capture):
        if receipt_ns > drain_ns:
            break
        frames.extend(processor.accept(receipt_ns, peer, payload))
    frames.extend(processor.advance(drain_ns))
    if (
        processor.origin_ns is None
        or processor.origin_ns > start_ns
        or processor.last_receipt_ns < end_ns - STEP_US * 1000
    ):
        raise ValueError("Capture does not cover the flight through disarm")
    flight, coverage = flight_frames(frames, processor.origin_ns, bounds)
    if not flight or digest(capture) != raw_sha:
        raise ValueError("Empty flight interval or changed capture")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "run.npz"
    np.savez_compressed(
        path,
        values=np.array([frame.values for frame in flight], dtype=np.float32),
        time_us=np.array([frame.time_us for frame in flight], dtype=np.int64),
        observed=np.array([frame.observed for frame in flight], dtype=bool),
        segment=np.array([frame.segment for frame in flight], dtype=np.int64),
    )
    valid = {frame.segment for frame in flight if frame.segment >= 0}
    result = dict(
        schema="drone-ai.new19.prepared/1",
        preprocessing_revision=REVISION,
        interval_policy="armed-takeoff-through-landing/1",
        hold_ms=HOLD_US // 1000,
        raw_sha256=raw_sha,
        profile_sha256=digest(profile),
        execution_sha256=digest(execution),
        sha256=digest(path),
        frames=len(flight),
        valid_frames=sum(frame.segment >= 0 for frame in flight),
        **coverage,
        segments=len(valid),
        decoder_bad=processor.source.bad,
        duplicate_packets=processor.source.duplicates,
        clock_resets=processor.clock.resets,
        receipt_fallbacks=processor.clock.receipt_fallbacks,
        reordered_samples=processor.reordered,
        late_samples=processor.late,
        unaligned_samples=processor.unaligned,
        receipt_origin_ns=processor.origin_ns,
    )
    write(output / "receipt.json", result)
    return result
