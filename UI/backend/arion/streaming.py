"""Append-only session journals and passive MAVLink transports."""

import asyncio
import hashlib
import json
import math
import os
import signal
import socket
import sys
import time
import uuid
from bisect import bisect_right
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from statistics import median
from threading import RLock

from arion import npz_archive

from .contracts import (
    InputProfile,
    NormalizedInput,
    ProcessedFrame,
    SourceConfig,
)
from .file_io import hash_file
from .live_journal import LiveJournal
from .npz_archive import NpzArchive
from .telemetry import TelemetryRun, is_display_filter
from .transport_reader import UDP_MAGIC, UDP_RECORD

BUFFER_MS = 300_000
MAX_BUFFER_RECORDS = 60_000
UDP_STORAGE = "udp-wire/1"


def encode(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def safe(value):
    if isinstance(value, (bytes, bytearray)):
        return {"$bytes": bytes(value).hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": str(value)}
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    return value


def mavlink_decoder():
    from pymavlink.dialects.v20 import common

    from .esc_status import MAVLink_esc_status_message

    common.mavlink_map.setdefault(291, MAVLink_esc_status_message)
    decoder = common.MAVLink(None)
    decoder.robust_parsing = True
    return decoder


def mavlink_record(message, monotonic_ns, utc_ns):
    fields = message.to_dict()
    fields.pop("mavpackettype", None)
    return dict(
        receive_monotonic_ns=monotonic_ns,
        receive_utc_ns=utc_ns,
        source_system=message.get_srcSystem(),
        source_component=message.get_srcComponent(),
        message_name=message.get_type(),
        fields=fields,
    )


class SessionRun:
    """Journal live input and reconstruct replay state across producer
    epochs.
    """

    def __init__(
        self,
        path,
        config=None,
        profile=None,
        session_id=None,
        *,
        compact=False,
        raw_only=False,
    ):
        self.path = Path(path)
        self.live = config is not None
        self.entries = []
        self.times = []
        self.issues = deque(maxlen=200)
        self.input_issues = deque(maxlen=200)
        self.boundaries = []
        self.epoch_starts = [0.0]
        self.total_lines = 0
        self.accepted = 0
        self.excluded = 0
        self.input_errors = 0
        self.dropped_sequences = 0
        self.stream_stats = defaultdict(
            lambda: {"count": 0, "last": None, "intervals": deque(maxlen=2048)}
        )
        self.last_receipt = time.perf_counter()
        self.receipt_intervals = deque(maxlen=2048)
        self.duration_ms = 0.0
        self.last_external_sequence = None
        self.external_epoch = None
        self.force_boundary = False
        self.state = "waiting"
        self.error = ""
        self.core = TelemetryRun.__new__(TelemetryRun)
        self.core.initialize()
        self.sha256 = None
        self.lock = RLock()
        self.archive = None
        self.raw_only = raw_only
        self.recent_inputs = {}
        self.last_ingest = {}
        if self.live:
            self.run_id = session_id or "live-" + uuid.uuid4().hex[:16]
            self.config, self.profile = config, profile
            self.name = config.name
            if self.raw_only and config.kind != "udp":
                raise ValueError("원문 UDP 저장은 UDP 세션에서만 지원합니다.")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = dict(
                schema_version="arion-journal-v2",
                session_id=self.run_id,
                config=config.model_dump(),
                profile=profile.model_dump(),
            )
            if self.raw_only:
                with self.wire_path.open("xb") as wire:
                    wire.write(UDP_MAGIC)
                    wire.flush()
                    os.fsync(wire.fileno())
                header["storage"] = UDP_STORAGE
            self.writer = self.path.open("xb")
            self.writer.write(encode(header) + b"\n")
            self.writer.flush()
            os.fsync(self.writer.fileno())
            if config.kind != "normalized":
                self.writer = LiveJournal(self.writer)
            if config.kind == "normalized":
                self.archive = NpzArchive(
                    self.path.parent / self.run_id / "npz",
                    self.path,
                    profile,
                    self.run_id,
                )
                self.archive.data["metadata"]["source"] = config.model_dump()
        else:
            with self.path.open("rb") as f:
                header = json.loads(f.readline())
                self.run_id = header["session_id"]
                self.config = SourceConfig.model_validate(header["config"])
                self.profile = InputProfile.model_validate(header["profile"])
                self.name = self.config.name
                self.raw_only = header.get("storage") == UDP_STORAGE
                if self.raw_only:
                    offsets = set()
                    for line in f:
                        try:
                            item = json.loads(line)
                        except ValueError:
                            self.issue("중단된 연결 메타데이터 · 원문 보존")
                            continue
                        if item.get("type") == "udp_reconnect":
                            offsets.add(item["offset"])
                else:
                    self._replay_journal(f, compact)
        for spec in self.profile.signals.values():
            if spec.message in self.core.max_age:
                self.core.max_age[spec.message] = spec.ttl_ms
        self.core.rates = dict(
            sorted(self.profile.expected_rates_hz.items())
            if self.raw_only
            else self.profile.expected_rates_hz
        )
        self.core.gap_ms = self.profile.gap_ms
        self.core.reset_step_ms = self.profile.sample_period_ms or 100
        self.core.voltage_encoding = self.profile.voltage_encoding
        self.core.missing_current_raw = self.profile.missing_current_raw
        self.core.run_id = self.run_id
        self.source = getattr(
            self,
            "source",
            (self.config.source_system, self.config.source_component),
        )
        if not self.live:
            if self.raw_only:
                self._replay_wire(offsets)
            self.state = "finished"
            self.sha256 = hash_file(self.source_path)

    def _replay_journal(self, stream, compact):
        last_sample = None
        for line in stream:
            self.total_lines += 1
            try:
                item = json.loads(line)
            except ValueError:
                self._add_issue(
                    dict(
                        line=self.total_lines,
                        reason="중단된 수집의 손상 행 · 원문 보존",
                        severity="excluded",
                    )
                )
                continue
            if item["type"] == "sample":
                last_sample = item
                self._remember(
                    item, retain=not compact or self._display_tick(item)
                )
            elif item["type"] == "issue":
                self._add_issue(item["issue"])
        if last_sample is not None and last_sample is not self.entries[-1]:
            self.entries.append(last_sample)
            self.times.append(last_sample["t_ms"])

    def _display_tick(self, item):
        return (
            not self.entries
            or item["epoch"] != self.entries[-1]["epoch"]
            or item["t_ms"] - self.times[-1] >= 100
        )

    def _replay_wire(self, offsets):
        decoder = mavlink_decoder()
        with self.wire_path.open("rb") as wire:
            if wire.read(len(UDP_MAGIC)) != UDP_MAGIC:
                raise ValueError("지원하지 않는 MAVLink UDP 원문")
            while True:
                offset = wire.tell()
                header = wire.read(UDP_RECORD.size)
                if not header:
                    break
                if len(header) != UDP_RECORD.size:
                    self.issue("끝의 불완전한 UDP 헤더 · 원문 보존")
                    break
                stamp, utc, _, _, size = UDP_RECORD.unpack(header)
                payload = wire.read(size)
                if len(payload) != size:
                    self.issue("끝의 불완전한 UDP 패킷 · 원문 보존")
                    break
                if offset in offsets:
                    decoder = mavlink_decoder()
                    self.force_boundary = bool(self.entries)
                for message in decoder.parse_buffer(payload) or []:
                    if message.get_type() == "BAD_DATA":
                        self.issue("MAVLink 프레임 해석 실패")
                        continue
                    self._ingest_raw(mavlink_record(message, stamp, utc))
        if self.accepted and self.entries[-1]["sequence"] != self.accepted:
            item = self._sample_item(
                self._raw_frame(),
                self.core.samples[-1].name,
                {},
                self.accepted,
            )
            self.entries.append(item)
            self.times.append(item["t_ms"])

    @property
    def now_ms(self):
        return (
            self.duration_ms + (time.perf_counter() - self.last_receipt) * 1000
            if self.live and self.entries
            else self.duration_ms
        )

    def _remember(self, item, *, retain=True):
        self.accepted += 1
        if retain:
            self.entries.append(item)
            self.times.append(item["t_ms"])
        self.duration_ms = item["t_ms"]
        epoch = item["epoch"]
        if epoch >= len(self.epoch_starts):
            self.epoch_starts.append(item["t_ms"])
            self.boundaries.append(
                {
                    "t_ms": item["t_ms"],
                    "epoch": epoch,
                    "reason": "연결·시각·수신 경계",
                    "line": item["sequence"],
                }
            )
        if item.get("raw", {}).get("source_system") is not None:
            self.source = (
                item["raw"]["source_system"],
                item["raw"]["source_component"],
            )
        name = item.get("message", "NORMALIZED")
        stat = self.stream_stats[name]
        stat["count"] += 1
        if (
            stat["last"]
            and stat["last"][1] == epoch
            and item["t_ms"] > stat["last"][0]
        ):
            stat["intervals"].append(item["t_ms"] - stat["last"][0])
        stat["last"] = (item["t_ms"], epoch)
        if self.live:
            cut = max(
                bisect_right(self.times, self.duration_ms - BUFFER_MS) - 1,
                len(self.entries) - MAX_BUFFER_RECORDS,
                0,
            )
            if cut:
                del self.entries[:cut]
                del self.times[:cut]
        if self.live or self.raw_only:
            # The decoding cache is not an archive. Retain only the display
            # window.
            for key in list(self.core.by_stream):
                times = self.core.stream_times[key]
                n = max(
                    0,
                    bisect_right(times, self.duration_ms - BUFFER_MS) - 1,
                    len(times) - MAX_BUFFER_RECORDS,
                )
                if n:
                    del times[:n]
                    del self.core.by_stream[key][:n]
            n = max(
                0,
                bisect_right(
                    self.core.sample_times, self.duration_ms - BUFFER_MS
                )
                - 1,
                len(self.core.samples) - MAX_BUFFER_RECORDS,
            )
            if n:
                del self.core.samples[:n]
                del self.core.sample_times[:n]

    def _add_issue(self, issue):
        self.excluded += 1
        self.issues.append(issue)
        if not is_display_filter(issue["reason"]):
            self.input_errors += 1
            self.input_issues.append(issue)

    def issue(self, reason, raw=None):
        self.total_lines += 1
        issue = {
            "line": self.total_lines,
            "reason": str(reason)[:500],
            "severity": "excluded",
        }
        self._add_issue(issue)
        if not self.raw_only:
            self._write({"type": "issue", "issue": issue, "raw": safe(raw)})

    def _write(self, item):
        self.writer.write(encode(item) + b"\n")
        if not isinstance(self.writer, LiveJournal):
            self.writer.flush()
            os.fsync(self.writer.fileno())

    def record_message_rate_request(
        self, target, command, message_id, interval_us
    ):
        """Journal an already-sent stream configuration request."""
        with self.lock:
            if not self.live:
                raise ValueError("종료된 세션입니다.")
            self._write(
                dict(
                    type="message_rate_request",
                    target=list(target),
                    command=command,
                    message_id=message_id,
                    interval_us=interval_us,
                    at_monotonic_ns=time.monotonic_ns(),
                )
            )

    def append_raw(self, record):
        if not self.live:
            raise ValueError("종료된 세션입니다.")
        if self.config.kind == "normalized":
            raise ValueError("입력 형식 불일치")
        return self._ingest_raw(record)

    def _ingest_raw(self, record):
        if (
            self.config.source_system is not None
            and record["source_system"] != self.config.source_system
        ):
            self.issue("다른 system 제외", record)
            return False
        if (
            self.config.source_component is not None
            and record["source_component"] != self.config.source_component
        ):
            self.issue("다른 component 제외", record)
            return False
        # Preserve the full passive transport for the independent common19
        # adapter.
        # Display-only decoding may exclude IMU messages or nonzero battery
        # instances.
        if not self.raw_only:
            self._write(
                {
                    "type": "mavlink",
                    "raw": safe(record),
                    "boundary": self.force_boundary,
                }
            )
        record = dict(record, quality={"profile": self.profile.profile_id})
        ok = self.core.ingest(record, force_boundary=self.force_boundary)
        if not ok:
            self.issue(self.core.issues[-1]["reason"], record)
            self.core.issues = self.core.issues[-200:]
            return False
        self.force_boundary = False
        self.source = self.core.source
        message = record["message_name"]
        if not self.live:
            sample = self.core.samples[-1]
            item = dict(
                t_ms=sample.t_ms,
                epoch=sample.epoch,
                sequence=self.accepted + 1,
                message=message,
            )
            if not self._display_tick(item):
                self.total_lines += 1
                self._remember(item, retain=False)
                return True
        return self._append(self._raw_frame(), message, safe(record))

    def _raw_frame(self):
        frame = self.core.frame(self.core.duration_ms)
        frame["values"] = {
            k: v
            for k, v in frame["values"].items()
            if k in self.profile.signals
        }
        return frame

    def append_normalized(self, payload):
        with self.lock:
            return self._append_processed(payload)

    def _append_processed(self, payload):
        if (
            not self.live
            or self.config.kind != "normalized"
            or self.state in ("disconnected", "error")
        ):
            raise ValueError("활성 전처리 세션이 아닙니다.")
        try:
            is_v1 = payload.get("schema_version") == "arion-processed-v1"
            p = (
                ProcessedFrame.model_validate(payload)
                if is_v1
                else NormalizedInput.model_validate(payload)
            )
            if p.session_id != self.run_id:
                raise ValueError("session_id 불일치")
            if is_v1 and p.profile_id != self.profile.profile_id:
                raise ValueError("프로필 버전 불일치")
            if p.preprocess_version != self.profile.preprocess_version:
                raise ValueError("전처리 버전 불일치")
            if p.sample_period_ms != self.profile.sample_period_ms:
                raise ValueError(
                    "전처리 주기 변경: 새 프로필·세션이 필요합니다."
                )
            identity = f"{p.epoch}:{p.sequence}"
            digest = hashlib.sha256(encode(p.model_dump())).hexdigest()
            known = self.recent_inputs.get(identity)
            if (
                known is None
                and self.last_external_sequence is not None
                and p.sequence <= self.last_external_sequence
            ):
                for _, _, item in npz_archive.journal_items(self.path):
                    if item.get("input_identity") == identity:
                        known = (
                            item["input_sha256"],
                            item["sequence"],
                            item.get("row_id"),
                        )
                        break
            if known:
                if known[0] != digest:
                    raise ValueError("같은 프레임 식별자의 값이 다릅니다.")
                self.last_ingest = {
                    "accepted": True,
                    "duplicate": True,
                    "sequence": known[1],
                    "row_id": known[2],
                }
                return True
            if (
                self.external_epoch is not None
                and p.epoch < self.external_epoch
            ):
                raise ValueError("오래된 producer epoch")
            if (
                self.external_epoch == p.epoch
                and self.last_external_sequence is not None
                and p.sequence <= self.last_external_sequence
            ):
                raise ValueError("중복 또는 순서가 역전된 sequence")
            for key, value in p.values.items():
                if key not in self.profile.signals:
                    raise ValueError(f"미등록 신호: {key}")
                if value.unit != self.profile.signals[key].unit:
                    raise ValueError(f"단위 불일치: {key}")
            boundary = self.force_boundary or (
                self.external_epoch is not None
                and self.external_epoch != p.epoch
            )
            if p.source_time_ns < getattr(
                self, "previous_processed_source_ns", p.source_time_ns
            ):
                boundary = True
            if (
                self.external_epoch == p.epoch
                and self.last_external_sequence is not None
            ):
                self.dropped_sequences += max(
                    0, p.sequence - self.last_external_sequence - 1
                )
            clock = p.relative_time_ns if is_v1 else p.source_time_ns
            self.core.ingest(
                {
                    "receive_monotonic_ns": clock,
                    "source_system": self.config.source_system or 1,
                    "source_component": self.config.source_component or 1,
                    "message_name": "HEARTBEAT",
                    "fields": {},
                },
                force_boundary=boundary,
            )
            t = self.core.duration_ms
            epoch = self.core.epoch_at(t)
            values = {}
            processed_values = {}
            for key, spec in self.profile.signals.items():
                v = p.values.get(key)
                age_ns = (
                    (v.age_ns if is_v1 else round(v.age_ms * 1e6))
                    if v
                    else None
                )
                valid = (v.mask if is_v1 else v.valid) if v else False
                status = (
                    "missing"
                    if v is None
                    else v.status
                    if is_v1
                    else "invalid"
                    if not valid or v.value is None
                    else "valid"
                )
                if status == "valid" and age_ns > spec.ttl_ms * 1e6:
                    status = "stale"
                processed_values[key] = {
                    "value": v.value if v else None,
                    "unit": spec.unit,
                    "mask": status == "valid",
                    "status": status,
                    "age_ns": age_ns,
                }
                values[key] = {
                    "value": v.value if status == "valid" else None,
                    "status": status,
                    "unit": spec.unit,
                    "label": spec.label,
                    "message": spec.message,
                    "age_ms": age_ns / 1e6 if age_ns is not None else None,
                    "reason": "mask=false"
                    if v and not valid and status != "missing"
                    else "",
                }
                if spec.display_key:
                    values[spec.display_key] = dict(values[key])
            frame = {
                "epoch": epoch,
                "t_ms": t,
                "values": values,
                "streams": [],
                "armed": p.armed,
                "mode": p.mode,
                "flight_state": p.flight_state,
            }
            processed = {
                "schema_version": "arion-processed-v1",
                "session_id": self.run_id,
                "profile_id": self.profile.profile_id,
                "sequence": p.sequence,
                "epoch": epoch,
                "source_time_ns": p.source_time_ns,
                "relative_time_ns": round(t * 1e6),
                "preprocess_version": p.preprocess_version,
                "sample_period_ms": p.sample_period_ms,
                "values": processed_values,
                "armed": p.armed,
                "mode": p.mode,
                "flight_state": p.flight_state,
            }
            row_id = hashlib.sha256(
                encode([self.run_id, p.epoch, p.sequence])
            ).hexdigest()
            frame.update(
                row_id=row_id,
                producer_sequence=p.sequence,
                producer_epoch=p.epoch,
            )
            row = {
                "processed": processed,
                "row_id": row_id,
                "input_identity": identity,
                "input_sha256": digest,
                "tick_ns": round(t * 1e6),
            }
            row["row_sha256"] = hashlib.sha256(encode(processed)).hexdigest()
            frame["row_sha256"] = row["row_sha256"]
            result = self._append(frame, "NORMALIZED", p.model_dump(), row)
            self.external_epoch = p.epoch
            self.last_external_sequence = p.sequence
            self.force_boundary = False
            self.previous_processed_source_ns = p.source_time_ns
            self.recent_inputs[identity] = (digest, self.accepted, row_id)
            while len(self.recent_inputs) > len(self.entries):
                self.recent_inputs.pop(next(iter(self.recent_inputs)))
            self.last_ingest = {
                "accepted": True,
                "duplicate": False,
                "sequence": self.accepted,
                "row_id": row_id,
            }
            return result
        except (ValueError, TypeError) as error:
            self.issue(str(error), payload)
            raise ValueError(str(error)) from error

    def _sample_item(self, frame, message, raw, sequence):
        frame.update(
            schema_version="arion-telemetry-v2",
            run_id=self.run_id,
            sequence=sequence,
            profile_id=self.profile.profile_id,
            preprocess_version=self.profile.preprocess_version,
            sample_period_ms=self.profile.sample_period_ms,
        )
        return {
            "type": "sample",
            "sequence": sequence,
            "t_ms": frame["t_ms"],
            "epoch": frame["epoch"],
            "message": message,
            "raw": raw,
            "frame": frame,
        }

    def _append(self, frame, message, raw, extra=None):
        self.total_lines += 1
        item = self._sample_item(frame, message, raw, self.accepted + 1)
        if extra:
            item.update(extra)
        if not self.raw_only:
            offset = self.writer.tell()
            self._write(item)
        self._remember(item)
        if self.archive:
            self.archive.observe(item, offset, self.writer.tell())
        if self.live and self.accepted > 1:
            self.receipt_intervals.append(
                (time.perf_counter() - self.last_receipt) * 1000
            )
        self.last_receipt = time.perf_counter()
        self.state = "connected"
        self.error = ""
        return True

    def epoch_at(self, cursor):
        return max(0, bisect_right(self.epoch_starts, cursor) - 1)

    def frame(self, cursor):
        cursor = max(0, float(cursor))
        i = bisect_right(self.times, cursor) - 1
        if i < 0:
            return {
                "schema_version": "arion-telemetry-v2",
                "run_id": self.run_id,
                "epoch": self.epoch_at(cursor),
                "t_ms": cursor,
                "sequence": 0,
                "values": {},
                "streams": [],
                "armed": None,
                "mode": "미확인",
                "flight_state": "미확인",
            }
        f = deepcopy(self.entries[i]["frame"])
        delta = cursor - f["t_ms"]
        f["t_ms"] = cursor
        for key, v in f["values"].items():
            if v["age_ms"] is not None:
                v["age_ms"] = round(v["age_ms"] + delta, 3)
            spec = self.profile.signals.get(key) or next(
                (
                    s
                    for s in self.profile.signals.values()
                    if s.display_key == key
                ),
                None,
            )
            if (
                v["status"] == "valid"
                and v["age_ms"] is not None
                and v["age_ms"] > spec.ttl_ms
            ):
                v.update(value=None, status="stale")
        if delta > self.profile.gap_ms:
            f.update(armed=None, mode="미확인", flight_state="미확인")
        if self.config.kind == "normalized":
            f["streams"] = [
                {
                    "name": "NORMALIZED",
                    "last_t_ms": self.entries[i]["t_ms"],
                    "age_ms": delta,
                    "status": "valid"
                    if delta <= self.profile.gap_ms
                    else "stale",
                }
            ]
        else:
            for stream in f["streams"]:
                if stream["age_ms"] is not None:
                    stream["age_ms"] += delta
                    if stream["age_ms"] > self.core.max_age[stream["name"]]:
                        stream["status"] = "stale"
        return f

    def manifest(self):
        streams = []
        for name, s in self.stream_stats.items():
            xs = list(s["intervals"])
            streams.append(
                {
                    "name": name,
                    "count": s["count"],
                    "median_hz": 1000 / median(xs) if xs else None,
                    "target_hz": self.profile.expected_rates_hz.get(name),
                    "max_interval_ms": max(xs) if xs else None,
                }
            )
        intervals = list(self.receipt_intervals)
        received_hz = (
            1000 / median(intervals)
            if intervals and median(intervals) > 0
            else None
        )
        output_hz = next(
            (s["median_hz"] for s in streams if s["name"] == "NORMALIZED"), None
        )
        return {
            "schema_version": "arion-run-v2",
            "run_id": self.run_id,
            "name": self.name,
            "input_sha256": self.sha256,
            "source": f"{self.config.kind.upper()} "
            + ("실시간" if self.live else "수집 기록"),
            "source_kind": "live" if self.live else "replay",
            "storage": UDP_STORAGE if self.raw_only else None,
            "source_system": self.source[0],
            "source_component": self.source[1],
            "duration_ms": self.now_ms,
            "total_records": self.total_lines,
            "accepted_records": self.accepted,
            "excluded_records": self.excluded,
            "input_error_records": self.input_errors,
            "input_issues": list(self.input_issues),
            "issues": list(self.issues),
            "boundaries": self.boundaries[-200:],
            "epochs": len(self.epoch_starts),
            "streams": streams,
            "supported_signals": list(self.profile.signals),
            "profile": self.profile.model_dump(),
            "session": self.session_status(),
            "npz": self.archive.status() if self.archive else None,
            "rates": {
                "raw_receive_hz": received_hz
                if self.config.kind != "normalized"
                else None,
                "processed_receipt_hz": received_hz
                if self.config.kind == "normalized"
                else None,
                "processed_output_hz": output_hz,
                "display_max_hz": 10,
                "model_inference_period_ms": None,
            },
        }

    def session_status(self):
        state = self.state
        if (
            self.live
            and state == "connected"
            and self.now_ms - self.duration_ms > self.profile.gap_ms
        ):
            state = "stale"
        return {
            "session_id": self.run_id,
            "state": state,
            "message": self.error,
            "kind": self.config.kind,
            "buffer_start_ms": self.times[0] if self.times else 0,
            "recording": self.live,
            "last_sequence": self.accepted,
            "dropped_sequences": self.dropped_sequences,
            "journal": self.writer.status()
            if self.live and isinstance(self.writer, LiveJournal)
            else None,
            "config": self.config.model_dump(),
        }

    def series(self):
        points = []
        last = -1
        epoch = None
        for item in self.entries:
            f = item["frame"]
            if f["t_ms"] - last < 100 and f["epoch"] == epoch:
                continue
            if epoch is not None and (
                f["epoch"] != epoch or f["t_ms"] - last > self.profile.gap_ms
            ):
                points.append(
                    {
                        "t_ms": f["t_ms"] - 0.001,
                        "epoch": f["epoch"],
                        "values": {k: None for k in self.profile.signals},
                    }
                )
            points.append(
                {
                    "t_ms": f["t_ms"],
                    "epoch": f["epoch"],
                    "values": {k: v["value"] for k, v in f["values"].items()},
                }
            )
            last = f["t_ms"]
            epoch = f["epoch"]
        return {"step_ms": 100, "points": points}

    def segment(self, start, end):
        items = [x for x in self.entries if start <= x["sequence"] <= end]
        if (
            not items
            or items[0]["sequence"] != start
            or items[-1]["sequence"] != end
            or len(items) != end - start + 1
        ):
            raise ValueError(
                "입력 구간이 표시 버퍼에 없거나 sequence가 누락됐습니다."
            )
        if len({x["epoch"] for x in items}) != 1:
            raise ValueError("입력 구간이 epoch 경계를 넘습니다.")
        digest = hashlib.sha256(
            b"\n".join(encode(x.get("processed", x["frame"])) for x in items)
        ).hexdigest()
        return {
            "kind": "stream",
            "session_id": self.run_id,
            "start_sequence": start,
            "end_sequence": end,
            "segment_sha256": digest,
            "preprocess_version": self.profile.preprocess_version,
        }, items

    def finish(self):
        if self.live:
            if self.archive:
                self.archive.close()
            self.writer.close()
            self.live = False
            self.state = "finished"
            self.sha256 = hash_file(self.source_path)
        return self.sha256

    @property
    def wire_path(self):
        return self.path.with_suffix(".mavlink.bin")

    @property
    def source_path(self):
        return self.wire_path if self.raw_only else self.path


class LiveSources:
    """Own live transport tasks, handles and telemetry-rate negotiations."""

    def __init__(self, store):
        from .message_rates import MessageRates

        self.store = store
        self.tasks = {}
        self.handles = {}
        self.receivers = {}
        self.stopping = set()
        self.rates = MessageRates(self)

    async def connect(self, run):
        await self.disconnect(run)
        if not run.live:
            raise ValueError("종료된 세션은 다시 연결할 수 없습니다.")
        c = run.config
        if c.kind == "normalized":
            run.state = "waiting"
            return
        try:
            if c.kind == "udp":
                handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    handle.bind((c.host, c.port))
                    handle.setblocking(False)
                except Exception:
                    handle.close()
                    raise
            else:
                import serial

                if not c.device.startswith("/dev/"):
                    raise ValueError("명시적인 /dev/ 시리얼 장치가 필요합니다.")
                handle = serial.Serial(
                    c.device,
                    c.baud,
                    timeout=0,
                    write_timeout=0.2,
                    exclusive=True,
                )
            self.handles[run.run_id] = handle
            wire_offset = None
            wire_arg = []
            if run.raw_only:
                wire_offset = run.wire_path.stat().st_size
                if wire_offset > len(UDP_MAGIC):
                    run._write(dict(type="udp_reconnect", offset=wire_offset))
                wire_arg = [str(run.wire_path)]
            receiver = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "arion.transport_reader",
                str(handle.fileno()),
                c.kind,
                str(os.getpid()),
                *wire_arg,
                pass_fds=(handle.fileno(),),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=(
                    asyncio.subprocess.DEVNULL
                    if run.raw_only
                    else asyncio.subprocess.PIPE
                ),
                stderr=asyncio.subprocess.PIPE,
                limit=262144,
            )
            self.receivers[run.run_id] = receiver
            self.tasks[run.run_id] = asyncio.create_task(
                self._read(run, handle, receiver, wire_offset)
            )
            run.state = "waiting"
            run.error = ""
        except (OSError, ValueError) as e:
            handle = self.handles.pop(run.run_id, None)
            if handle:
                handle.close()
            run.state = "error"
            run.error = str(e)
            raise ValueError(str(e)) from e

    async def disconnect(self, run):
        await self.rates.disconnect(run.run_id)
        task = self.tasks.pop(run.run_id, None)
        receiver = self.receivers.pop(run.run_id, None)
        self.stopping.add(run.run_id)
        if receiver and receiver.returncode is None:
            receiver.terminate()
        if task:
            try:
                await asyncio.wait_for(asyncio.shield(task), 5)
            except asyncio.TimeoutError:
                if receiver and receiver.returncode is None:
                    receiver.kill()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                run.state = "error"
                run.error = "수신 프로세스 종료 시간 초과 · 기록을 확인하세요."
        if receiver:
            await receiver.wait()
        self.stopping.discard(run.run_id)
        handle = self.handles.pop(run.run_id, None)
        if handle:
            handle.close()
        if run.live and isinstance(run.writer, LiveJournal):
            await asyncio.to_thread(run.writer.flush)
        run.force_boundary = bool(run.entries)
        if run.live and run.state != "error":
            run.state = "disconnected"

    def _decode_packet(self, run, decoder, data, peer, stamp, utc):
        for msg in decoder.parse_buffer(data) or ():
            if msg.get_type() == "BAD_DATA":
                run.issue("MAVLink 프레임 해석 실패", msg.to_dict())
                continue
            self.rates.observe(run, msg, peer)
            run.append_raw(mavlink_record(msg, stamp, utc))

    async def _check_exit(self, run, receiver):
        code = await receiver.wait()
        if run.run_id not in self.stopping or code not in (0, -signal.SIGTERM):
            lines = (
                (await receiver.stderr.read())
                .decode(errors="replace")
                .splitlines()
            )
            detail = lines[-1] if lines else "오류 로그 없음"
            raise OSError(f"수신 프로세스 종료 ({code}): {detail}")

    async def _tail_udp(self, run, receiver, decoder, start):
        with run.wire_path.open("rb") as wire:
            wire.seek(start)
            while True:
                run.writer.check_error()
                offset = wire.tell()
                header = wire.read(UDP_RECORD.size)
                if len(header) == UDP_RECORD.size:
                    stamp, utc, address, port, size = UDP_RECORD.unpack(header)
                    data = wire.read(size)
                    if len(data) == size:
                        peer = (socket.inet_ntoa(address), port)
                        self._decode_packet(
                            run, decoder, data, peer, stamp, utc
                        )
                        await asyncio.sleep(0)
                        continue
                if receiver.returncode is not None:
                    await self._check_exit(run, receiver)
                    if wire.tell() != offset:
                        raise OSError("끝의 불완전한 UDP 패킷 · 원문 보존")
                    return
                wire.seek(offset)
                await asyncio.sleep(0.02)

    async def _read(self, run, handle, receiver, wire_offset):
        decoder = mavlink_decoder()
        wire = None
        try:
            if wire_offset is not None:
                await self._tail_udp(run, receiver, decoder, wire_offset)
                return
            if run.config.kind == "udp":
                wire = run.wire_path.open("ab", buffering=0)
                if not wire.tell():
                    wire.write(UDP_MAGIC)
            while True:
                run.writer.check_error()
                try:
                    header = await asyncio.wait_for(
                        receiver.stdout.readline(), 0.2
                    )
                except asyncio.TimeoutError:
                    continue
                if not header:
                    await self._check_exit(run, receiver)
                    break
                packet = json.loads(header)
                data = await receiver.stdout.readexactly(packet["size"])
                peer = tuple(packet["peer"]) if packet["peer"] else None
                monotonic_ns = packet["receive_monotonic_ns"]
                utc_ns = packet["receive_utc_ns"]
                if wire is not None:
                    record = (
                        UDP_RECORD.pack(
                            monotonic_ns,
                            utc_ns,
                            socket.inet_aton(peer[0]),
                            peer[1],
                            len(data),
                        )
                        + data
                    )
                    if wire.write(record) != len(record):
                        raise OSError("Incomplete MAVLink UDP recording")
                self._decode_packet(
                    run, decoder, data, peer, monotonic_ns, utc_ns
                )
                # Busy sockets must still yield to HTTP, websocket and rate tasks.
                await asyncio.sleep(0)
        except Exception as e:
            run.state = "error"
            run.error = f"{type(e).__name__}: {e}"[:300]
        finally:
            if wire is not None:
                try:
                    os.fsync(wire.fileno())
                except OSError as error:
                    run.state = "error"
                    run.error = f"MAVLink UDP 기록 저장 실패: {error}"
                finally:
                    wire.close()
            handle.close()
            if receiver.returncode is None:
                receiver.terminate()
                try:
                    await asyncio.wait_for(receiver.wait(), 3)
                except asyncio.TimeoutError:
                    receiver.kill()
                    await receiver.wait()

    async def close(self):
        for run in list(self.store.active.values()):
            await self.disconnect(run)
            self.store.finish_session(run.run_id)
