"""Explicit real-log adapters and bounded causal common19 preparation.

BIN units come from onboard FMTU/UNIT metadata, never from guessed
firmware units.
PX4 uses documented uORB SI/FRD/NED semantics. User-supplied mapping and
airborne
interval evidence are required; no ground/healthy/airframe inference is
performed.
"""

import contextlib
import io
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from capstone_ai import mapping
from capstone_ai.mapping import FEATURES, RATES
from learning.history19.preprocessing import LowPass
from localflight import bin_reader, run_summary

from .catalog import digest
from .logs import PX4_REQUIRED, inspect_log
from .prepared import REVISION
from .profiles import eligibility
from .timing import clock_maps, split_streams


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def bin_values(message, profile):
    kind, row = message.get_type(), message.to_dict()
    units = dict(zip(message.fmt.columns, message.fmt.units or []))

    def value(field, conversions):
        unit = units.get(field)
        if unit not in conversions:
            raise ValueError(f"Unverified onboard unit: {kind}.{field}={unit}")
        raw = _finite(row.get(field))
        return None if raw is None else raw * conversions[unit]

    if kind == "RATE":
        out = {}
        for axis, field in zip("pqr", ("R", "P", "Y")):
            out["body_rate_" + axis] = value(
                field, {"deg/s": math.pi / 180, "rad/s": 1}
            )
            out["rate_target_" + axis] = value(
                field + "Des", {"deg/s": math.pi / 180, "rad/s": 1}
            )
        return out
    if kind == "ATT":
        return {
            "roll": value("Roll", {"deg": math.pi / 180, "rad": 1}),
            "pitch": value("Pitch", {"deg": math.pi / 180, "rad": 1}),
        }
    if kind == "IMU" and row.get("I") == profile["imu_instance"]:
        return {
            "specific_force_" + axis: value(
                "Acc" + axis.upper(), {"m/s/s": 1, "m/s^2": 1}
            )
            if row.get("AH") == 1
            else None
            for axis in "xyz"
        }
    if kind == "BAT" and row.get("Instance") == profile["battery_instance"]:
        return dict(
            pack_voltage=value("Volt", {"V": 1}),
            pack_current=value("Curr", {"A": 1}),
        )
    if kind == "CTUN":
        return dict(
            climb_rate=value("CRt", {"cm/s": 0.01, "m/s": 1}),
            climb_target=value("DCRt", {"cm/s": 0.01, "m/s": 1}),
        )
    if kind == "RCOU":
        out = {}
        for position, channel in profile["motor_channels"].items():
            raw = value(f"C{channel}", {"us": 1, "µs": 1})
            out["motor_request_" + position] = (
                None
                if raw is None
                else (raw - profile["output_min"])
                / (profile["output_max"] - profile["output_min"])
            )
        return out
    return {}


def bin_events(path, profile, maps=None):
    warnings = io.StringIO()
    with (
        contextlib.redirect_stdout(warnings),
        contextlib.redirect_stderr(warnings),
    ):
        reader = bin_reader.open_bin(path)
    if warnings.getvalue():
        reader.close()
        raise ValueError(
            "Parser initialization warning has no reliable boundary: "
            + warnings.getvalue()[:1000]
        )
    streams = {}
    previous = {}
    boundary = None
    count = 0
    try:
        while True:
            offset = reader.offset
            try:
                with (
                    contextlib.redirect_stdout(warnings),
                    contextlib.redirect_stderr(warnings),
                ):
                    m = reader.recv_msg()
            except (TypeError, ValueError, IndexError) as error:
                # A corrupt BIN can make DFReader raise before emitting its
                # warning.
                # Retain only already decoded records, never resume beyond this
                # byte.
                boundary = dict(
                    reason="parser_exception",
                    byte_offset=offset if type(offset) is int else None,
                    detail=type(error).__name__ + ": " + str(error)[:1000],
                )
                break
            if warnings.getvalue():
                boundary = dict(
                    reason="parser_warning",
                    byte_offset=offset,
                    detail=warnings.getvalue()[:1000],
                )
                break
            if m is None:
                break
            if m.get_type() not in {
                "RATE",
                "ATT",
                "IMU",
                "BAT",
                "CTUN",
                "RCOU",
            }:
                continue
            stamp = getattr(m, "TimeUS", None)
            values = bin_values(m, profile)
            if not values:
                continue
            key = f"{m.get_type()}:{getattr(m, 'I', getattr(m, 'Instance', 0))}"
            if stamp is not None:
                if key in previous and stamp < previous[key]:
                    boundary = dict(
                        reason="clock_reset",
                        byte_offset=offset,
                        stream=key,
                        previous_us=int(previous[key]),
                        reset_us=int(stamp),
                    )
                    # Retain file prefix; never sort the next boot into it.
                    break
                previous[key] = stamp
            if stamp is None:
                continue
            streams.setdefault(key, []).append((int(stamp), int(stamp), values))
            count += 1
            if count > 500000:
                raise ValueError(
                    "More than 500000 source records; use a shorter log"
                )
    finally:
        reader.close()
    events, report = split_streams(streams, maps or {})
    return events, dict(streams=report, retained_prefix_boundary=boundary)


def px4_events(path, profile, maps=None):
    from pyulog import ULog

    ulog = ULog(str(path), message_name_filter_list=list(PX4_REQUIRED))
    streams = {}
    count = 0
    for data in ulog.data_list:
        topic = data.name
        instance = (
            profile["battery_instance"]
            if topic == "battery_status"
            else profile["actuator_instance"]
            if topic == "actuator_outputs"
            else profile["imu_instance"]
            if topic in {"vehicle_angular_velocity", "vehicle_acceleration"}
            else 0
        )
        if data.multi_id != instance:
            continue
        fields = data.data
        stamps = fields.get("timestamp", [])
        if len(stamps) and np.any(stamps[1:] < stamps[:-1]):
            raise ValueError(
                (
                    "ULog clock reset: topic arrays cannot identify shared "
                    "boot boundaries; supply a separately extracted boot log"
                )
            )
        for i, stamp in enumerate(stamps):
            sample = fields.get("timestamp_sample", stamps)[i]
            if sample > stamp or sample <= 0:
                raise ValueError("PX4 sample/availability time mismatch")
            raw = {}
            if topic == "actuator_outputs":
                out = {}
                counts = fields.get("noutputs")
                for pos, channel in profile["motor_channels"].items():
                    key = f"output[{channel - 1}]"
                    if key not in fields:
                        raise ValueError(f"Missing motor channel {key}")
                    raw_value = float(fields[key][i])
                    # Array padding is not an active output channel.
                    out["motor_request_" + pos] = (
                        (raw_value - profile["output_min"])
                        / (profile["output_max"] - profile["output_min"])
                        if counts is not None and channel <= counts[i] <= 16
                        else None
                    )
            elif topic == "vehicle_local_position":
                # Real uORB marks altitude and vertical velocity separately.
                # Keep the frozen simulation mapper's contract unchanged.
                velocity = _finite(fields["vz"][i])
                valid = fields.get("v_z_valid")
                out = dict(
                    climb_rate=-velocity
                    if velocity is not None
                    and valid is not None
                    and valid[i] == 1
                    else None
                )
            else:
                arrays = {
                    "vehicle_attitude": ("q", 4),
                    "vehicle_angular_velocity": ("xyz", 3),
                    "vehicle_acceleration": ("xyz", 3),
                }
                if topic in arrays:
                    key, n = arrays[topic]
                    raw[key] = [
                        float(fields[f"{key}[{j}]"][i]) for j in range(n)
                    ]
                else:
                    raw = {
                        k: float(a[i])
                        for k, a in fields.items()
                        if np.issubdtype(a.dtype, np.number)
                    }
                output = mapping.mapped(
                    dict(
                        topic=topic,
                        fields=raw,
                        field_quality={k: "valid" for k in raw},
                    ),
                    {},
                )
                out = {
                    k: v if q == "valid" else None
                    for k, (v, q) in output.items()
                }
                if topic == "battery_status" and out["pack_current"] == -1:
                    # uORB's documented unknown sentinel, not charging current.
                    out["pack_current"] = None
            if out:
                streams.setdefault(f"{topic}:{data.multi_id}", []).append(
                    (int(stamp), int(sample), out)
                )
                count += 1
            if count > 500000:
                raise ValueError(
                    "More than 500000 source records; use a shorter log"
                )
    events, report = split_streams(streams, maps or {})
    # Dropout metadata is a hard boundary even if held samples look sufficiently
    # fresh.
    for dropout in ulog.dropouts:
        events.append(
            (
                int(dropout.timestamp),
                int(dropout.timestamp),
                {"__reset__": True},
            )
        )
        end = int(dropout.timestamp) + int(dropout.duration) * 1000
        events.append((end, end, {"__reset__": True}))
        events = [e for e in events if not int(dropout.timestamp) < e[0] < end]
    return sorted(events, key=lambda e: e[0]), dict(
        streams=report,
        dropouts=len(ulog.dropouts),
        adapter_revision="px4-common19-units-v2",
        vertical_velocity_validity="vehicle_local_position.v_z_valid",
    )


class CausalSampler:
    """Retain causal filter state across chunks, using the offline frame kernel."""

    def __init__(self, fast_max_age_ms=200, battery_max_age_ms=350):
        self.fast_max_age_ms = fast_max_age_ms
        self.battery_max_age_ms = battery_max_age_ms
        self.latest, self.filters, self.previous_sample = {}, {}, {}
        self.position = 0
        self.next_us = None
        self.segment_start_us = None

    def sample(self, events, start_us, end_us):
        fast_max_age_ms, battery_max_age_ms = (
            self.fast_max_age_ms,
            self.battery_max_age_ms,
        )
        latest, filters, previous_sample = (
            self.latest,
            self.filters,
            self.previous_sample,
        )
        position = self.position
        if self.next_us is not None and start_us != self.next_us:
            raise ValueError(
                "Causal sampling chunks must continue without rewinding"
            )
        values, times, invalid, boundaries = [], [], [], set()
        for t in range(((start_us + 99999) // 100000) * 100000, end_us, 100000):
            while position < len(events) and events[position][0] <= t:
                available, sample, updates = events[position]
                if "__reset__" in updates:
                    latest.clear()
                    filters.clear()
                    previous_sample.clear()
                    boundaries.add(len(values))
                    position += 1
                    continue
                if sample > available or sample < 0:
                    raise ValueError("Invalid source clock")
                if any(
                    name in latest
                    and sample - latest[name][1]
                    > (
                        battery_max_age_ms
                        if name.startswith("pack_")
                        else fast_max_age_ms
                    )
                    * 1000
                    for name in updates
                ):
                    latest.clear()
                    filters.clear()
                    boundaries.add(len(values))
                for name, raw in updates.items():
                    if (
                        name in previous_sample
                        and sample < previous_sample[name]
                    ):
                        raise ValueError(
                            "Source clock reversal; split the log first"
                        )
                    previous_sample[name] = sample
                    value = _finite(raw)
                    if (
                        name.startswith("motor_request")
                        and value is not None
                        and not 0 <= value <= 1
                    ):
                        value = None
                    if (
                        name == "pack_voltage"
                        and value is not None
                        and value <= 0
                    ):
                        value = None
                    continuous = (
                        name in RATES
                        or name.startswith("specific_force_")
                        or name
                        in {"climb_rate", "pack_voltage", "pack_current"}
                    )
                    limit = (
                        battery_max_age_ms
                        if name.startswith("pack_")
                        else fast_max_age_ms
                    ) * 1000
                    old = latest.get(name)
                    if old is not None and sample - old[1] > limit:
                        boundaries.add(len(values))
                    if continuous and value is not None:
                        if (
                            name not in filters
                            or old is None
                            or old[0] is None
                            or sample - old[1] > limit
                        ):
                            filters[name] = LowPass(available, value, 0.08)
                        else:
                            filters[name].update(available, value)
                    latest[name] = (value, sample)
                position += 1

            def at(name, frame_time=t):
                if name not in latest:
                    return None
                v, sample = latest[name]
                limit = (
                    battery_max_age_ms
                    if name.startswith("pack_")
                    else fast_max_age_ms
                ) * 1000
                if v is None or frame_time - sample > limit:
                    return None
                return filters[name].at(frame_time)[1] if name in filters else v

            row = []
            for name in FEATURES:
                if "error" in name:
                    measured = (
                        "climb_rate"
                        if name == "climb_rate_error"
                        else "body_rate_" + name[-1]
                    )
                    target = (
                        "climb_target"
                        if name == "climb_rate_error"
                        else "rate_target_" + name[-1]
                    )
                    a, b = at(measured), at(target)
                    value = None if a is None or b is None else b - a
                else:
                    value = at(name)
                row.append(value)
            missing = [
                name
                for name, v in zip(FEATURES, row)
                if v is None or not math.isfinite(v)
            ]
            invalid.append(missing)
            values.append([np.nan if v is None else v for v in row])
            times.append(t)
            if missing:
                self.segment_start_us = None
            elif self.segment_start_us is None or len(values) - 1 in boundaries:
                self.segment_start_us = t
        self.position = position
        self.next_us = ((end_us + 99999) // 100000) * 100000
        return values, times, invalid, boundaries


def causal_frames(
    events, start_us, end_us, *, fast_max_age_ms=200, battery_max_age_ms=350
):
    if (
        type(start_us) is not int
        or type(end_us) is not int
        or start_us < 0
        or not 11_000_000 <= end_us - start_us <= 7_200_000_000
    ):
        raise ValueError(
            "Select an explicit inspection interval of 11..7200 seconds"
        )
    for age in (fast_max_age_ms, battery_max_age_ms):
        if type(age) is not int or not 20 <= age <= 2000:
            raise ValueError("Sample hold limits must be integer 20..2000 ms")
    values, times, invalid, boundaries = CausalSampler(
        fast_max_age_ms, battery_max_age_ms
    ).sample(events, start_us, end_us)
    spans, start = [], None
    for i in range(len(times) + 1):
        valid = i < len(times) and not invalid[i]
        if start is not None and (not valid or i in boundaries):
            spans.append(
                dict(
                    start_index=start,
                    end_index=i,
                    start_us=times[start],
                    end_us=times[i - 1] + 100000,
                    frames=i - start,
                    usable=i - start >= 110,
                )
            )
            start = None
        if valid and start is None:
            start = i

    reasons = Counter(name for names in invalid for name in names)
    excluded = []
    for t, missing in zip(times, invalid):
        if not missing:
            continue
        if excluded and excluded[-1]["end_us"] == t:
            excluded[-1]["end_us"] = t + 100000
            excluded[-1]["missing_features"] = sorted(
                set(excluded[-1]["missing_features"]) | set(missing)
            )
        else:
            excluded.append(
                dict(
                    start_us=t,
                    end_us=t + 100000,
                    reason="missing_or_stale",
                    missing_features=missing,
                )
            )
    report = dict(
        excluded_intervals=excluded,
        inspection_start_us=start_us,
        inspection_end_us=end_us,
        candidate_segments=spans,
        requested_frames=len(times),
        complete_frames=sum(not m for m in invalid),
        missing_or_stale_frames=sum(bool(m) for m in invalid),
        missing_by_feature=dict(reasons),
        boundary_count=len(boundaries),
        fast_max_age_ms=fast_max_age_ms,
        battery_max_age_ms=battery_max_age_ms,
        interpolation="causal bounded sample hold; no gap bridging",
        minimum_segment_frames=110,
    )
    return np.asarray(values, dtype="f8"), np.asarray(times, dtype="i8"), report


def prepare(
    path,
    profile,
    output=None,
    *,
    start_us,
    end_us,
    interval_evidence="",
    role="evaluation",
    health="unknown",
    health_evidence="",
    clock_corrections=None,
    fast_max_age_ms=200,
    battery_max_age_ms=350,
    selected_segments=None,
    preview=False,
    expected_source_sha256=None,
):
    e = eligibility(profile)
    if not e["model_structure_compatible"] or (
        not e["timeline_ready"] if preview else not e["profile_complete"]
    ):
        missing = (
            e["timeline_missing_fields"] if preview else e["log_missing_fields"]
        )
        raise ValueError("분석에 필요한 입력 설정: " + ", ".join(missing))
    if any(
        profile.get(k) is None
        for k in ("imu_instance", "battery_instance", "actuator_instance")
    ):
        raise ValueError(
            "Explicit IMU, battery and actuator instances required"
        )
    if role not in {
        "adaptation",
        "development",
        "evaluation",
        "calibration",
    } or health not in {
        "unknown",
        "confirmed_normal",
        "abnormal",
    }:
        raise ValueError("Invalid role or health state")
    if health == "confirmed_normal" and not health_evidence.strip():
        raise ValueError("Confirmed normal requires supporting evidence")
    if not preview and not interval_evidence.strip():
        raise ValueError("Reviewed airborne interval evidence required")
    audit = inspect_log(path)
    if (
        expected_source_sha256 is not None
        and audit["sha256"] != expected_source_sha256
    ):
        raise ValueError("Source log changed after the segment preview")
    if audit["autopilot"] != profile["autopilot"]:
        raise ValueError("Autopilot mismatch")
    if profile["autopilot"] == "px4" and audit["parser_warnings"]:
        raise ValueError(
            "ULog parser warning cannot be located safely: "
            + audit["parser_warnings"][:1000]
        )
    maps = clock_maps(clock_corrections)
    # A logged topic inventory alone is not the mapping check. Field units,
    # instances,
    # validity, chronology and the declared actuator map are checked while
    # decoding.
    messages = io.StringIO()
    with (
        contextlib.redirect_stdout(messages),
        contextlib.redirect_stderr(messages),
    ):
        events, decode = (
            bin_events if profile["autopilot"] == "ardupilot" else px4_events
        )(path, profile, maps)
    if messages.getvalue():
        raise ValueError(
            "Parser warnings during preparation: " + messages.getvalue()[:1000]
        )
    values, times, timing = causal_frames(
        events,
        start_us,
        end_us,
        fast_max_age_ms=fast_max_age_ms,
        battery_max_age_ms=battery_max_age_ms,
    )
    timing.update(
        clock_corrections=maps,
        decoding=decode,
        parser_warnings=audit["parser_warnings"],
        mode="offline_clock_reconstruction"
        if maps
        else "onboard_clock_causal_replay",
    )
    if digest(path) != audit["sha256"]:
        raise ValueError("Raw log changed")
    if preview:
        return dict(
            timing=timing,
            source_sha256=audit["sha256"],
            profile_id=profile["id"],
            health=health,
            interval_evidence=interval_evidence,
            imu_instance=profile.get("imu_instance"),
        )
    candidates = timing["candidate_segments"]
    chosen = (
        [i for i, s in enumerate(candidates) if s["usable"]]
        if selected_segments is None
        else selected_segments
    )
    if (
        not isinstance(chosen, list)
        or not chosen
        or any(
            type(i) is not int or i < 0 or i >= len(candidates) for i in chosen
        )
        or chosen != sorted(set(chosen))
        or any(not candidates[i]["usable"] for i in chosen)
    ):
        raise ValueError(
            "Select ordered, distinct preserved segments of at least 11 seconds"
        )
    arrays, time_arrays, segments = [], [], []
    cursor = 0
    for index in chosen:
        s = candidates[index]
        arrays.append(values[s["start_index"] : s["end_index"]])
        time_arrays.append(times[s["start_index"] : s["end_index"]])
        segments.append(
            dict(
                start_index=cursor,
                end_index=cursor + s["frames"],
                candidate_index=index,
            )
        )
        cursor += s["frames"]
    values, times = np.concatenate(arrays), np.concatenate(time_arrays)
    if len(times) > 10000:
        raise ValueError(
            (
                "Select at most 1000 seconds of retained segments per flight "
                "package"
            )
        )
    timing.update(
        selected_segments=chosen,
        retained_frames=len(times),
        coverage=len(times) / timing["requested_frames"],
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    source_name = "source" + Path(path).suffix.lower()
    shutil.copyfile(path, out / source_name)
    if digest(out / source_name) != audit["sha256"]:
        raise ValueError("Raw copy integrity mismatch")
    np.savez(out / "frames.npz", values=values, time_us=times)
    record = dict(
        schema="drone.real-common19/1",
        preprocessing_revision=REVISION,
        feature_order=FEATURES,
        units="physical_pack_V_A",
        profile=profile,
        origin="px4_real"
        if profile["autopilot"] == "px4"
        else "ardupilot_real",
        role=role,
        health=health,
        health_evidence=health_evidence,
        segments=segments,
        timing=timing,
        mapping_verified=True,
        mapping_evidence=profile.get("mapping_evidence")
        or "사용자가 저장한 채널/인스턴스 설정; 로그 단위와 인과 입력 검사 수행",
        mapping_scope=(
            "Declared channel/instance map, onboard units and causal input "
            "checks; not real-aircraft model validation"
        ),
        source_path=source_name,
        source_sha256=audit["sha256"],
        interval=dict(
            start_us=start_us, end_us=end_us, evidence=interval_evidence
        ),
        file=dict(path="frames.npz", sha256=digest(out / "frames.npz")),
    )

    record["run_context"] = run_summary.source_context(audit, start_us, end_us)
    (out / "manifest.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return dict(
        manifest=str(out / "manifest.json"),
        frames=len(times),
        source_sha256=audit["sha256"],
        role=role,
        health=health,
        timing=timing,
        real_aircraft_validated=False,
    )
