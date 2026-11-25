"""Display sampled observations from an inspected onboard log."""

import contextlib
import io
import math
from pathlib import Path

from . import bin_reader
from .catalog import digest

POSITION_KEYS = ("north_m", "east_m", "height_m", "speed_m_s")
ATTITUDE_KEYS = ("roll_deg", "pitch_deg", "yaw_deg")
MAX_POINTS = 6000


def number(value):
    """Convert finite numeric parser values and their missing sentinels."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def position(north, east, down, vn, ve, vd):
    speed = math.hypot(vn, ve, vd) if None not in (vn, ve, vd) else None
    return dict(
        zip(
            POSITION_KEYS,
            (north, east, -down if down is not None else None, speed),
        )
    )


def battery(volts, amps, percent):
    return dict(
        battery_voltage_v=volts if volts is not None and volts > 0 else None,
        battery_current_a=amps,
        battery_soc_pct=percent
        if percent is not None and 0 <= percent <= 100
        else None,
    )


def attitude(w, x, y, z):
    if None in (w, x, y, z) or not 0.98 < math.hypot(w, x, y, z) < 1.02:
        return dict.fromkeys(ATTITUDE_KEYS)
    pitch = max(-1, min(1, 2 * (w * y - z * x)))
    angles = (
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(pitch),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    )
    return dict(zip(ATTITUDE_KEYS, map(math.degrees, angles)))


class Timeline:
    """Bound the screen series while preserving missing observations and cuts."""

    def __init__(self, clocks):
        self.buckets = {}
        self.breaks = set()
        self.last = {}
        self.cutoff = None
        sample_range = list(clocks.values())
        span = (
            max(c["last"] for c in sample_range)
            - min(c["first"] for c in sample_range)
            if sample_range
            else 0
        )
        self.step = max(
            100_000, math.ceil(span / (MAX_POINTS - 1) / 100_000) * 100_000
        )

    def add(self, stamp, stream, values):
        if stamp < 0:
            return
        previous = self.last.get(stream, stamp)
        if stamp < previous:
            if previous - stamp > 1_000_000:
                self.cutoff = min(
                    previous,
                    self.cutoff if self.cutoff is not None else previous,
                )
            return
        self.last[stream] = stamp
        if self.cutoff is not None and stamp > self.cutoff:
            return
        bucket = self.buckets.setdefault(stamp // self.step, {})
        if len(self.buckets) > MAX_POINTS * 2:
            raise ValueError(
                "로그 표시 범위가 너무 넓습니다. 파일을 나누어 가져오세요."
            )
        for key, value in values.items():
            if key not in bucket or stamp >= bucket[key][0]:
                bucket[key] = (stamp, value)

    def result(self, warning):
        latest, points = {}, []
        cuts = iter(sorted(self.breaks))
        cut = next(cuts, None)
        epoch = 0
        previous = 0
        origin = 0
        for row in (self.buckets[key] for key in sorted(self.buckets)):
            stamp = max(time for time, _ in row.values())
            if self.cutoff is not None and stamp > self.cutoff:
                break
            changed = False
            while cut is not None and cut <= stamp:
                changed = True
                cut = next(cuts, None)
            if points and (
                changed or stamp - previous > max(1_000_000, self.step * 2)
            ):
                epoch += 1
                latest.clear()
                points.append(
                    dict(
                        t_ms=(previous - origin + 1) / 1000,
                        epoch=epoch,
                        values={},
                    )
                )
            latest.update(row)
            values = {
                key: value
                for key, (time, value) in latest.items()
                if stamp - time
                <= (1_000_000 if key.startswith("battery_") else 300_000)
            }
            if not points:
                origin = stamp
            points.append(
                dict(t_ms=(stamp - origin) / 1000, epoch=epoch, values=values)
            )
            previous = stamp
        if not points or not any(
            any(v is not None for v in p["values"].values()) for p in points
        ):
            raise ValueError("로그에 표시할 배터리·자세·위치 관측이 없습니다.")
        if self.cutoff is not None:
            warning = "시계가 되돌아가기 전까지의 관측을 표시합니다."
        return dict(
            points=points,
            duration_ms=points[-1]["t_ms"],
            step_ms=self.step / 1000,
            start_us=origin,
            warning=warning,
        )


def read_bin(path, timeline, battery_instance, cores):
    reader = bin_reader.open_bin(path)
    primary = cores[0] if len(cores) == 1 else None
    try:
        while message := reader.recv_match(type={"BAT", "ATT", "XKF1", "XKF4"}):
            kind = message.get_type()
            row = message.to_dict()
            stamp = row.get("TimeUS")
            if type(stamp) is not int:
                continue
            units = dict(zip(message.fmt.columns, message.fmt.units or []))

            def field(name, unit):
                return (
                    number(row.get(name)) if units.get(name) == unit else None
                )

            if kind == "XKF4":
                current = row.get("PI")
                if (
                    isinstance(current, int)
                    and current >= 0
                    and current != primary
                ):
                    timeline.breaks.add(stamp)
                    primary = current
                continue
            if kind == "ATT":
                values = dict(
                    zip(
                        ATTITUDE_KEYS,
                        (
                            field("Roll", "deg"),
                            field("Pitch", "deg"),
                            field("Yaw", "degheading"),
                        ),
                    )
                )
            elif (
                kind == "XKF1"
                and primary is not None
                and row.get("C") == primary
            ):
                values = position(
                    *(field(name, "m") for name in ("PN", "PE", "PD")),
                    *(field(name, "m/s") for name in ("VN", "VE", "VD")),
                )
            elif kind == "BAT" and row.get("Instance", 0) == battery_instance:
                values = battery(
                    field("Volt", "V"), field("Curr", "A"), field("RemPct", "%")
                )
            else:
                continue
            timeline.add(stamp, kind, values)
    finally:
        reader.close()


def read_px4(path, timeline, battery_instance):
    from pyulog import ULog

    topics = ["vehicle_attitude", "vehicle_local_position", "battery_status"]
    log = ULog(str(path), message_name_filter_list=topics)
    for topic in log.data_list:
        if topic.multi_id != (
            battery_instance if topic.name == "battery_status" else 0
        ):
            continue
        data = topic.data
        previous = None
        old_counters = None
        for i, raw_stamp in enumerate(data.get("timestamp", [])):
            stamp = int(raw_stamp)
            if previous is not None and stamp < previous:
                timeline.cutoff = min(
                    previous,
                    timeline.cutoff
                    if timeline.cutoff is not None
                    else previous,
                )
                break

            def field(name):
                array = data.get(name)
                return number(array[i]) if array is not None else None

            if topic.name == "vehicle_attitude":
                values = attitude(*(field(f"q[{j}]") for j in range(4)))
            elif topic.name == "vehicle_local_position":
                counters = tuple(
                    field(name)
                    for name in ("xy_reset_counter", "z_reset_counter")
                )
                if i and counters != old_counters:
                    timeline.breaks.add(stamp)
                old_counters = counters
                xy, z = field("xy_valid") == 1, field("z_valid") == 1
                vxy, vz = field("v_xy_valid") == 1, field("v_z_valid") == 1
                values = position(
                    field("x") if xy else None,
                    field("y") if xy else None,
                    field("z") if z else None,
                    field("vx") if vxy else None,
                    field("vy") if vxy else None,
                    field("vz") if vz else None,
                )
            else:
                soc = field("remaining")
                current = field("current_a")
                values = battery(
                    field("voltage_v"),
                    current if current != -1 else None,
                    soc * 100 if soc is not None else None,
                )
            timeline.add(stamp, topic.name, values)
            previous = stamp
    timeline.breaks.update(int(drop.timestamp) for drop in log.dropouts)


def view(request):
    """Read the inspected file without changing source or analysis artifacts."""
    path = Path(request["path"])
    if (
        any(item.is_symlink() for item in (path, *path.parents))
        or digest(path) != request["sha256"]
    ):
        raise ValueError("검사 이후 원본 로그가 바뀌었습니다.")
    timeline = Timeline(request["clocks"])
    instances = request["battery_instances"]
    instance = int(next(iter(instances))) if len(instances) == 1 else 0
    messages = io.StringIO()
    with (
        contextlib.redirect_stdout(messages),
        contextlib.redirect_stderr(messages),
    ):
        if path.suffix.lower() == ".bin":
            read_bin(path, timeline, instance, request["primary_cores"])
        elif path.suffix.lower() == ".ulg":
            read_px4(path, timeline, instance)
        else:
            raise ValueError("PX4 ULog 또는 ArduPilot BIN을 선택하세요.")
    if digest(path) != request["sha256"]:
        raise ValueError("관측을 읽는 동안 원본 로그가 바뀌었습니다.")
    warning = (
        "파서 경고가 있어 읽을 수 있었던 관측을 표시합니다."
        if messages.getvalue()
        else ""
    )
    result = timeline.result(warning)
    result["battery_instance"] = instance
    return result
