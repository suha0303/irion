"""Read-only, causal MAVLink decoding shared by replay and future stream adapters.

Never use simulator labels as model output. Never merge samples across an epoch.
All wall/source nanoseconds stay Python integers; UI receives relative milliseconds.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import median


MAX_AGE = {"BATTERY_STATUS": 600, "ESC_STATUS": 250, "LOCAL_POSITION_NED": 300,
           "ATTITUDE": 300, "HEARTBEAT": 1500, "EXTENDED_SYS_STATE": 600,
           "POSITION_TARGET_LOCAL_NED": 300, "ATTITUDE_TARGET": 300,
           "ACTUATOR_OUTPUT_STATUS": 250, "ESC_INFO": 600, "SERVO_OUTPUT_RAW": 300}
RATE_TARGETS = {"ESC_STATUS": 20, "BATTERY_STATUS": 5, "ATTITUDE": 10,
                "LOCAL_POSITION_NED": 10, "HEARTBEAT": 2, "EXTENDED_SYS_STATE": 5,
                "POSITION_TARGET_LOCAL_NED": 10, "ATTITUDE_TARGET": 10,
                "ACTUATOR_OUTPUT_STATUS": 20}
LABELS = {"battery_voltage_v": ("팩 전압", "V", "BATTERY_STATUS"),
          "battery_current_a": ("팩 전류", "A", "BATTERY_STATUS"),
          "battery_soc_pct": ("충전 잔량", "%", "BATTERY_STATUS"),
          "battery_consumed_mah": ("누적 소비", "mAh", "BATTERY_STATUS"),
          "north_m": ("북쪽 위치", "m", "LOCAL_POSITION_NED"),
          "east_m": ("동쪽 위치", "m", "LOCAL_POSITION_NED"),
          "height_m": ("epoch 시작점 대비 높이", "m", "LOCAL_POSITION_NED"),
          "speed_m_s": ("속력", "m/s", "LOCAL_POSITION_NED"),
          "roll_deg": ("Roll", "°", "ATTITUDE"), "pitch_deg": ("Pitch", "°", "ATTITUDE"),
          "yaw_deg": ("Yaw", "°", "ATTITUDE")}
for i in range(1, 5):
    LABELS[f"motor_{i}_rpm"] = (f"모터 {i} RPM", "rpm", "ESC_STATUS")
    LABELS[f"motor_{i}_current_a"] = (f"모터 {i} 전류", "A", "ESC_STATUS")


def finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass
class Sample:
    t_ms: float
    epoch: int
    name: str
    fields: dict
    receive_ns: int
    line: int
    profile: str


class TelemetryRun:
    def __init__(self, path: Path, name: str | None = None):
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.run_id = "run-" + self.sha256[:16]
        self.name = name or path.name
        self.initialize()
        with path.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as error:
                    self.total_lines += 1
                    self.issues.append({"line": number, "reason": str(error), "severity": "excluded"})
                    continue
                self.ingest(record, number)
        if not self.samples:
            raise ValueError("재생할 수 있는 유효한 MAVLink 레코드가 없습니다.")

    def initialize(self):
        self.samples = []
        self.issues = []
        self.boundaries = []
        self.by_stream = defaultdict(list)
        self.stream_times = defaultdict(list)
        self.sample_times = []
        self.epoch_starts = [0.0]
        self.base_z = {}
        self.source = None
        self.total_lines = 0
        self.previous_ns = None
        self.previous_source_times = {}
        self.duration_ms = 0.0
        self._epoch = 0
        self._series = None
        self.max_age = dict(MAX_AGE)
        self.rates = dict(RATE_TARGETS)
        self.gap_ms = 1000
        self.voltage_encoding = None
        self.missing_current_raw = None

    def ingest(self, record, number=None, force_boundary=False):
        self.total_lines += 1
        number = number or self.total_lines
        try:
            if not isinstance(record, dict):
                raise ValueError("JSON 객체가 아닙니다.")
            for key in ("receive_monotonic_ns", "source_system", "source_component", "message_name", "fields"):
                if key not in record:
                    raise ValueError(f"필수 필드 누락: {key}")
            ns = record["receive_monotonic_ns"]
            if type(ns) is not int or not 0 <= ns <= 2**63 - 1:
                raise ValueError("수신 시각은 음수가 아닌 int64 ns여야 합니다.")
            if any(type(record[k]) is not int or not 0 <= record[k] <= 255 for k in ("source_system", "source_component")):
                raise ValueError("잘못된 source system/component")
            fields, message = record["fields"], record["message_name"]
            if not isinstance(fields, dict) or not isinstance(message, str):
                raise ValueError("메시지 이름 또는 fields 형식 오류")
            source = (record["source_system"], record["source_component"])
            if self.source is not None and source != self.source:
                raise ValueError(f"다른 source 제외: {source}")
            if message not in MAX_AGE:
                raise ValueError(f"미지원 메시지: {message}")
            if message == "BATTERY_STATUS" and fields.get("id") != 0:
                raise ValueError("배터리 id 0 외 인스턴스 제외")
            if message == "ESC_STATUS" and fields.get("index") != 0:
                raise ValueError("ESC index 0 외 bank 제외")
        except (ValueError, TypeError) as error:
            self.issues.append({"line": number, "reason": str(error), "severity": "excluded"})
            return False
        self.source = source
        reason = "수집 연결 재개" if force_boundary and self.samples else None
        source_time = fields.get("time_boot_ms")
        if source_time is None and message in ("ESC_STATUS", "ACTUATOR_OUTPUT_STATUS"):
            source_time = fields.get("time_usec")
            source_time = source_time / 1000 if finite(source_time) else None
        if self.previous_ns is not None:
            delta_ms = (ns - self.previous_ns) / 1e6
            if delta_ms < 0:
                reason = "수신 clock 역행"
                self.duration_ms += 100
            else:
                self.duration_ms += delta_ms
                if delta_ms > self.gap_ms:
                    reason = "수신 공백"
        if finite(source_time) and source_time > 0:
            if message in self.previous_source_times and source_time < self.previous_source_times[message]:
                reason = reason or "동일 메시지 source clock 역행"
        if reason:
            self._epoch += 1
            self.epoch_starts.append(self.duration_ms)
            self.boundaries.append({"t_ms": self.duration_ms, "epoch": self._epoch, "reason": reason, "line": number})
            self.previous_source_times.clear()
        if finite(source_time) and source_time > 0:
            self.previous_source_times[message] = source_time
        quality = record.get("quality")
        profile = quality.get("profile", "unknown") if isinstance(quality, dict) else "unknown"
        sample = Sample(self.duration_ms, self._epoch, message, fields, ns, number, profile)
        self.samples.append(sample)
        self.sample_times.append(sample.t_ms)
        self.by_stream[message].append(sample)
        self.stream_times[message].append(sample.t_ms)
        if message == "LOCAL_POSITION_NED" and finite(fields.get("z")):
            self.base_z.setdefault(self._epoch, fields["z"])
        self.previous_ns = ns
        self._series = None
        return True

    def epoch_at(self, cursor):
        return max(0, bisect_right(self.epoch_starts, cursor) - 1)

    def latest(self, name, cursor):
        index = bisect_right(self.stream_times.get(name, []), cursor) - 1
        if index < 0:
            return None
        sample = self.by_stream[name][index]
        return sample if sample.epoch == self.epoch_at(cursor) else None

    def frame(self, cursor):
        cursor = max(0.0, min(float(cursor), self.duration_ms))
        epoch = self.epoch_at(cursor)
        latest = {name: self.latest(name, cursor) for name in MAX_AGE}
        values = {}

        def put(key, value, valid=True, reason=""):
            label, unit, message = LABELS[key]
            sample = latest[message]
            age = cursor - sample.t_ms if sample else None
            status = "missing" if not sample else "stale" if age > self.max_age[message] else "valid" if valid and finite(value) else "invalid"
            values[key] = {"value": value if status == "valid" else None, "unit": unit, "status": status,
                           "age_ms": round(age, 3) if age is not None else None, "label": label,
                           "message": message, "reason": reason if status == "invalid" else ""}

        battery = latest["BATTERY_STATUS"]
        fields = battery.fields if battery else {}
        cells = fields.get("voltages", [])
        cells = cells if isinstance(cells, list) else []
        arion_profile = self.voltage_encoding == "arion_4s" or bool(self.voltage_encoding is None and battery and battery.profile == "arion_cmu_v1")
        cell_valid = arion_profile and len(cells) >= 4 and all(finite(c) and 0 <= c < 65535 for c in cells[:4])
        put("battery_voltage_v", sum(cells[:4]) / 1000 if cell_valid else None, cell_valid,
            "유효한 4S 셀 전압과 arion_cmu_v1 생산자 계약이 필요합니다.")
        if self.voltage_encoding == "mavlink" or (battery and battery.profile == "mavlink_common_v1"):
            extended = fields.get("voltages_ext", [])
            extended = extended if isinstance(extended,list) else []
            valid_cells = [v for v in cells + extended if finite(v) and 0 < v < 65535]
            put("battery_voltage_v", sum(valid_cells) / 1000 if valid_cells else None, bool(valid_cells), "미제공 전압")
        current = fields.get("current_battery")
        sentinels = self.missing_current_raw if self.missing_current_raw is not None else [-1, -100] if arion_profile else [-1]
        current_valid = finite(current) and current not in sentinels
        put("battery_current_a", current / 100 if current_valid else None, current_valid, "미제공 전류 sentinel")
        soc = fields.get("battery_remaining")
        put("battery_soc_pct", soc, finite(soc) and 0 <= soc <= 100, "SOC 미제공 또는 범위 밖")
        consumed = fields.get("current_consumed")
        put("battery_consumed_mah", consumed, finite(consumed) and consumed >= 0)
        esc = latest["ESC_STATUS"]
        for i in range(4):
            for field, suffix in (("rpm", "rpm"), ("current", "current_a")):
                array = esc.fields.get(field, []) if esc else []
                value = array[i] if isinstance(array, list) and len(array) > i else None
                put(f"motor_{i + 1}_{suffix}", value)
        local = latest["LOCAL_POSITION_NED"]
        motion = local.fields if local else {}
        put("north_m", motion.get("x"))
        put("east_m", motion.get("y"))
        z = motion.get("z")
        put("height_m", self.base_z[epoch] - z if epoch in self.base_z and finite(z) else None)
        velocity = [motion.get(key) for key in ("vx", "vy", "vz")]
        put("speed_m_s", math.sqrt(sum(v * v for v in velocity)) if all(finite(v) for v in velocity) else None)
        attitude = latest["ATTITUDE"]
        for key in ("roll", "pitch", "yaw"):
            value = attitude.fields.get(key) if attitude else None
            put(key + "_deg", math.degrees(value) if finite(value) else None)

        def fresh_fields(name):
            s = latest[name]
            return s.fields if s and cursor - s.t_ms <= self.max_age[name] else {}

        hb = fresh_fields("HEARTBEAT")
        mode = hb.get("custom_mode") if hb.get("autopilot") == 12 else None
        base = hb.get("base_mode")
        main_mode = (mode >> 16) & 0xFF if type(mode) is int else None
        submode = (mode >> 24) & 0xFF if type(mode) is int else None
        mode_name = {1: "MANUAL", 2: "ALTCTL", 3: "POSCTL", 4: "AUTO", 5: "ACRO", 6: "OFFBOARD", 7: "STABILIZED"}.get(main_mode, "미확인")
        if main_mode == 4:
            mode_name = {2: "AUTO TAKEOFF", 3: "AUTO HOLD", 4: "AUTO MISSION", 5: "AUTO RTL", 6: "AUTO LAND"}.get(submode, "AUTO")
        landed = fresh_fields("EXTENDED_SYS_STATE").get("landed_state")
        flight_state = {1: "지상", 2: "비행 중", 3: "이륙 중", 4: "착륙 중"}.get(landed, "미확인")
        streams = []
        for name in self.rates:
            sample = latest[name]
            age = cursor - sample.t_ms if sample else None
            streams.append({"name": name, "age_ms": round(age, 1) if age is not None else None,
                            "last_t_ms": sample.t_ms if sample else None,
                            "status": "missing" if not sample else "stale" if age > self.max_age[name] else "valid"})
        return {"schema_version": "arion-telemetry-v1", "run_id": self.run_id, "epoch": epoch,
                "t_ms": cursor, "values": values, "streams": streams,
                "armed": bool(base & 128) if type(base) is int else None, "mode": mode_name,
                "flight_state": flight_state}

    def manifest(self):
        statistics = []
        for name, samples in self.by_stream.items():
            intervals = [b.t_ms - a.t_ms for a, b in zip(samples, samples[1:]) if b.epoch == a.epoch and b.t_ms > a.t_ms]
            statistics.append({"name": name, "count": len(samples), "median_hz": 1000 / median(intervals) if intervals else None,
                               "max_interval_ms": max(intervals) if intervals else None, "target_hz": self.rates.get(name)})
        return {"schema_version": "arion-run-v1", "run_id": self.run_id, "name": self.name,
                "input_sha256": self.sha256, "source": "SITL 기록" if all(s.profile == "arion_cmu_v1" for s in self.samples) else "MAVLink 기록",
                "source_system": self.source[0] if self.source else None, "source_component": self.source[1] if self.source else None,
                "duration_ms": self.duration_ms, "total_records": self.total_lines, "accepted_records": len(self.samples),
                "excluded_records": len(self.issues), "issues": self.issues, "boundaries": self.boundaries,
                "epochs": len(self.epoch_starts), "streams": statistics, "supported_signals": list(LABELS),
                "source_time_note": "수신 monotonic 기준. 높이는 epoch의 첫 유효 위치 대비입니다."}

    def series(self):
        if self._series is None:
            step = max(100, math.ceil(self.duration_ms / 10000 / 100) * 100)
            ticks = set(range(0, math.ceil(self.duration_ms), step)) | {self.duration_ms}
            for boundary in self.boundaries:
                ticks.update([max(0, boundary["t_ms"] - 0.001), boundary["t_ms"]])
            points = []
            previous_epoch = None
            for tick in sorted(ticks):
                f = self.frame(tick)
                if previous_epoch is not None and f["epoch"] != previous_epoch:
                    points.append({"t_ms": tick, "epoch": f["epoch"], "values": {k: None for k in LABELS}})
                points.append({"t_ms": tick, "epoch": f["epoch"], "values": {k: v["value"] for k, v in f["values"].items()}})
                previous_epoch = f["epoch"]
            self._series = {"step_ms": step, "points": points}
        return self._series
