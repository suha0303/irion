"""Map source-bound MAVLink observations without loading AI weights."""

import math

from . import operating


def number(value):
    return (
        float(value)
        if type(value) in (int, float) and math.isfinite(value)
        else None
    )


def decode(name, f, p, request):
    """common.xml SI/FRD/NED semantics, explicit channels/instances and
    masks.
    """
    if name == "HEARTBEAT":
        mode = f.get("base_mode")
        return dict(
            arming_state=(2 if mode & 128 else 1)
            if type(mode) is int and 0 <= mode <= 255
            else None
        )
    if name == "EXTENDED_SYS_STATE":
        return dict(landed_state=number(f.get("landed_state")))
    if name == "ATTITUDE":
        return dict(
            zip(
                ("roll", "pitch", "body_rate_p", "body_rate_q", "body_rate_r"),
                (
                    number(f.get(k))
                    for k in (
                        "roll",
                        "pitch",
                        "rollspeed",
                        "pitchspeed",
                        "yawspeed",
                    )
                ),
            )
        )
    if name == "ATTITUDE_TARGET":
        mask = f.get("type_mask", 255)
        return {
            "rate_target_" + a: None if mask & (1 << i) else number(f.get(k))
            for i, (a, k) in enumerate(
                zip(
                    "pqr",
                    ("body_roll_rate", "body_pitch_rate", "body_yaw_rate"),
                )
            )
        }
    if name == request["imu_message"]:
        if name == "HIGHRES_IMU" and f.get("id", 0) != request["imu_id"]:
            return {}
        scale = 1 if name == "HIGHRES_IMU" else 9.80665 / 1000
        return {
            "specific_force_" + a: (
                number(f.get(a + "acc")) * scale
                if number(f.get(a + "acc")) is not None
                else None
            )
            for i, a in enumerate("xyz")
            if name != "HIGHRES_IMU" or f.get("fields_updated", 0) & (1 << i)
        }
    if name == "LOCAL_POSITION_NED":
        v = number(f.get("vz"))
        return dict(climb_rate=-v if v is not None else None)
    if name == "POSITION_TARGET_LOCAL_NED":
        v = number(f.get("vz"))
        # Only LOCAL_NED. Body/local-offset frames require a separate
        # conversion.
        return dict(
            climb_target=-v
            if v is not None
            and f.get("coordinate_frame") == 1
            and not f.get("type_mask", 65535) & 32
            else None
        )
    if name == "SERVO_OUTPUT_RAW" and f.get("port") == p["actuator_instance"]:
        return {
            "motor_request_" + pos: (
                (number(f.get(f"servo{channel}_raw")) - p["output_min"])
                / (p["output_max"] - p["output_min"])
                if number(f.get(f"servo{channel}_raw")) is not None
                else None
            )
            for pos, channel in p["motor_channels"].items()
        }
    if name == "BATTERY_STATUS" and f.get("id") == p["battery_instance"]:
        # MAVLink uses summed mV cells, UINT16_MAX for unused primary slots,
        # and 0 for unused extension slots. Capacity is never inferred here.
        cells = f.get("voltages", [])
        valid = [number(v) for v in cells if v != 65535]
        extension = [
            number(v) for v in f.get("voltages_ext", []) if v not in (0, 65535)
        ]
        voltage = (
            sum(valid + extension) / 1000
            if valid and all(v is not None and v > 0 for v in valid + extension)
            else None
        )
        current = number(f.get("current_battery"))
        return dict(
            pack_voltage=voltage,
            pack_current=None
            if current is None or current == -1
            else current / 100,
        )
    return {}


class DisarmGuard:
    """Require an observed arming and three fresh disarmed heartbeats over 2
    s.
    """

    def __init__(self):
        self.seen_armed = False
        self.first = self.last = None
        self.count = 0

    def accept(self, armed, timestamp):
        if armed is True:
            self.seen_armed = True
        if armed is not False or not self.seen_armed:
            self.first = self.last = None
            self.count = 0
            return
        if self.last is not None and timestamp <= self.last:
            return
        if self.last is None or timestamp - self.last > 3_000_000:
            self.first, self.count = timestamp, 0
        self.last = timestamp
        self.count += 1

    @property
    def ready(self):
        return self.count >= 3 and self.last - self.first >= 2_000_000

    def landing_ready(self, now_us, last_measurement_us):
        """Require positive disarm evidence and ten seconds of measurement silence."""
        return (
            self.ready
            and last_measurement_us is not None
            and now_us - max(self.first, last_measurement_us)
            >= operating.LANDING_SILENCE_US
        )


class Mapper:
    """Convert received MAVLink observations into ordered profile-bound
    events.
    """

    def __init__(self, request):
        self.request = request
        self.profile = request["profile"]
        self.base = None
        self.previous = {}
        self.heartbeat = None
        self.last_us = 0
        self.last_receipt = None
        self.last_measurement_us = None
        self.messages = {}
        self.resets = 0
        self.duplicates = 0
        self.armed = None
        self.disarm = DisarmGuard()

    def accept(self, item):
        if item.get("type") != "mavlink":
            return []
        raw = item["raw"]
        if (raw["source_system"], raw["source_component"]) != (
            self.request["source_system"],
            self.request["source_component"],
        ):
            return []
        ns = raw["receive_monotonic_ns"]
        if type(ns) is not int or ns < 0:
            raise ValueError("Invalid receipt timestamp")
        if self.base is None:
            self.base = ns
        # A monotonic-clock reset is a hard stop, never fabricated elapsed time.
        if self.last_receipt is not None and ns < self.last_receipt:
            raise ValueError("수신 시계 역행: 새 세션이 필요합니다.")
        self.last_receipt = ns
        t = (ns - self.base) // 1000
        self.last_us = t
        name, fields = raw["message_name"], raw["fields"]
        self.messages[name] = self.messages.get(name, 0) + 1
        if name == "HEARTBEAT":
            expected = 12 if self.profile["autopilot"] == "px4" else 3
            if fields.get("autopilot") != expected:
                raise ValueError(
                    "HEARTBEAT autopilot과 기체 프로파일이 다릅니다."
                )
            self.heartbeat = t
            mode = fields.get("base_mode")
            self.armed = (
                bool(mode & 128)
                if type(mode) is int and 0 <= mode <= 255
                else None
            )
            self.disarm.accept(self.armed, t)
        events = []
        updates = decode(name, fields, self.profile, self.request)
        clock = fields.get("time_boot_ms", fields.get("time_usec"))
        key = (name, fields.get("id"), fields.get("port"))
        old = self.previous.get(key)
        if item.get("boundary") or (
            old is not None and clock is not None and clock < old
        ):
            events.append((t, t, {"__reset__": True}))
            self.previous.clear()
            self.disarm = DisarmGuard()
            self.last_measurement_us = None
            if name == "HEARTBEAT":
                self.disarm.accept(self.armed, t)
            self.resets += 1
            old = None
        if clock is not None:
            self.previous[key] = clock
        if clock is not None and old == clock:
            self.duplicates += 1
            return events
        if updates:
            if any(
                value is not None
                for name, value in updates.items()
                if name not in operating.FIELDS
            ):
                self.last_measurement_us = t
            events.append((t, t, updates))
        return events
