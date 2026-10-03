import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def shared_strings(record):
    """Share repeated JSON labels without changing any value or duplicate comparison."""
    return {
        sys.intern(key): sys.intern(value) if isinstance(value, str) else value
        for key, value in record.items()
    }


class Lines:
    def __init__(self, path):
        self.path = Path(path)
        self.offset, self.buffer, self.identity, self.rows = 0, b"", None, 0
        self.timing = {}

    def poll(self, limit):
        begin = time.monotonic_ns()
        if not self.path.exists():
            return []
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
            raise ValueError("source replaced/truncated; refusing unsafe resume")
        self.identity = identity
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read(limit)
        read_done = time.monotonic_ns()
        self.offset += len(data)
        parts = (self.buffer + data).split(b"\n")
        self.buffer = parts.pop()
        result = [json.loads(p, object_hook=shared_strings) for p in parts if p.strip()]
        self.rows += len(result)
        self.timing = {
            "start_ns": begin,
            "read_done_ns": read_done,
            "parsed_ns": time.monotonic_ns(),
            "offset": self.offset,
            "complete_offset": self.offset - len(self.buffer),
            "bytes": len(data),
            "rows": self.rows,
            "new_rows": len(result),
        }
        return result


class DirectoryTail:
    def __init__(self, run, engine, chunk_bytes=1024 * 1024):
        self.run, self.engine = Path(run).resolve(), engine
        self.stream = Lines(self.run / "observations/streams.jsonl")
        self.watermarks = Lines(self.run / "observations/watermarks.jsonl")
        self.events: deque[dict[str, Any]] = deque()
        self.marks: deque[dict[str, Any]] = deque()
        self.ingested, self.closed, self.last_mark = 0, False, None
        self.chunk_bytes = chunk_bytes
        self.trace = None
        self.frame_ready = {}
        self.mark_seen = {}

    def poll(self):
        if self.closed:
            return False
        begin = time.monotonic_ns()
        rows = self.stream.poll(self.chunk_bytes)
        marks = self.watermarks.poll(self.chunk_bytes)
        self.events.extend(rows)
        self.marks.extend(marks)
        for mark in marks:
            key = (mark["clock_id"], mark["recorded_events"], mark["closed"])
            self.mark_seen.setdefault(key, deque()).append(self.watermarks.timing["read_done_ns"])
        progress = bool(rows or marks)
        while self.marks:
            w = self.marks[0]
            if (
                w["schema"] != "drone.raw.watermark/2.0"
                or w["run_id"] != self.engine.metadata["run_id"]
            ):
                raise ValueError("watermark identity/schema")
            count = w["recorded_events"]
            if count < self.ingested:
                raise ValueError("recorded_events regressed")
            if self.stream.rows < count:
                break  # Watermark may be visible before the stream on a filesystem.
            self.marks.popleft()
            commit_start = time.monotonic_ns()
            frame_start = len(self.engine.frames)
            while self.ingested < count:
                self.engine.ingest(self.events.popleft())
                self.ingested += 1
            self.engine.advance(w["clock_id"], w["through_available_time_us"])
            ready = time.monotonic_ns()
            seen_key = (w["clock_id"], count, w["closed"])
            watermark_seen = self.mark_seen[seen_key].popleft()
            if not self.mark_seen[seen_key]:
                del self.mark_seen[seen_key]
            timing = {
                "kind": "watermark_commit",
                "poll_start_ns": begin,
                "clock_id": w["clock_id"],
                "recorded_events": count,
                "through_available_time_us": w["through_available_time_us"],
                "watermark_seen_ns": watermark_seen,
                "commit_start_ns": commit_start,
                "frames_ready_ns": ready,
                "stream_read": self.stream.timing,
                "watermark_read": self.watermarks.timing,
            }
            if self.trace:
                self.trace(timing)
                for frame in self.engine.frames[frame_start:]:
                    self.frame_ready[(frame["clock_id"], frame["segment_id"], frame["time_us"])] = (
                        timing
                    )
            self.last_mark, self.closed = w, w["closed"]
            if self.closed:
                if any(
                    item[-1]["available_time_us"] <= w["through_available_time_us"]
                    and item[0] != w["clock_id"]
                    for item in self.engine.pending
                ):
                    raise ValueError("uncommitted previous clock events")
                break
        if self.trace:
            self.trace(
                {
                    "kind": "tail_poll",
                    "start_ns": begin,
                    "end_ns": time.monotonic_ns(),
                    "stream": self.stream.timing,
                    "watermarks": self.watermarks.timing,
                    "pending_watermarks": len(self.marks),
                    "progress": progress,
                }
            )
        return progress

    def checkpoint(self):
        # Recovery replays the verified prefix to reconstruct filters, context and sequence state.
        return {
            "run": str(self.run),
            "run_id": self.engine.metadata["run_id"],
            "config": self.engine.config,
            "last_watermark": self.last_mark,
            "files": {
                str(x.path): {
                    "offset": x.offset,
                    "identity": x.identity,
                    "prefix_sha256": self._prefix(x.path, x.offset),
                }
                for x in [self.stream, self.watermarks]
            },
        }

    @staticmethod
    def _prefix(path, offset):
        with Path(path).open("rb") as f:
            return hashlib.sha256(f.read(offset)).hexdigest()

    def verify_checkpoint(self, checkpoint):
        if checkpoint["run"] != str(self.run) or checkpoint["config"] != self.engine.config:
            raise ValueError("checkpoint source/config mismatch")
        for path, record in checkpoint["files"].items():
            stat = Path(path).stat()
            if (
                (stat.st_dev, stat.st_ino) != tuple(record["identity"])
                or stat.st_size < record["offset"]
                or self._prefix(path, record["offset"]) != record["prefix_sha256"]
            ):
                raise ValueError("checkpoint identity or prefix changed")


def resolve_profile(path):
    from .mapping import BINDINGS

    profile = read_json(path)
    if (
        profile["schema"] != "drone.source.profile/2.0"
        or profile["kind"] != "drone_raw_directory_tail"
    ):
        raise ValueError("unsupported source profile")
    required = {
        "streams_file": "observations/streams.jsonl",
        "watermarks_file": "observations/watermarks.jsonl",
        "diagnostics_file": "observations/diagnostics.jsonl",
        "metadata_file": "observations/metadata.json",
        "poll_interval_ms": 20,
    }
    for key, value in required.items():
        if profile[key] != value:
            raise ValueError("unsupported source option: " + key)
    mapping = read_json(profile["mapping_profile"])
    expected = {k: {"topic": t, "units": u} for k, (t, u) in BINDINGS.items()}
    if mapping["roles"] != expected:
        raise ValueError("mapping profile differs from implemented PX4 mapping")
    return Path(profile["run_directory"]).resolve()
