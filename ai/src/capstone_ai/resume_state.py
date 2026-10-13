"""Local, hash-bound runtime snapshots with a restricted non-executable class allowlist."""

import gc
import io
import os
import pickle
from collections import Counter, deque
from pathlib import Path

from capstone_ai.preprocessing import LowPass, Preprocessor
from capstone_ai.source import DirectoryTail, sha


class StateReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("capstone_ai.preprocessing", "Preprocessor"): Preprocessor,
            ("capstone_ai.preprocessing", "LowPass"): LowPass,
            ("collections", "Counter"): Counter,
            ("collections", "deque"): deque,
        }
        if (module, name) not in allowed:
            raise ValueError("unsupported runtime snapshot class")
        return allowed[(module, name)]


def adopt_shadow(tail, config, metadata, checkpoint):
    """Accept only a same-source common engine already beyond the saved prefix."""
    if tail.engine.config != config or tail.engine.metadata != metadata:
        raise ValueError("shadow config/metadata mismatch")
    tail.verify_checkpoint(checkpoint)
    if tail.ingested < checkpoint["last_watermark"]["recorded_events"]:
        raise ValueError("shadow has not ingested the checkpoint prefix")
    for reader in [tail.stream, tail.watermarks]:
        if reader.offset < checkpoint["files"][str(reader.path)]["offset"]:
            raise ValueError("shadow has not read the checkpoint prefix")
    return tail.engine, tail


def save_handover_state(tail, path):
    """Persist causal state; full seen/archive state remains bound to the raw prefix."""
    state = {
        "version": 1,
        "recovery": "same_engine_live_shadow_with_verified_raw_prefix",
        "engine": {
            key: value
            for key, value in vars(tail.engine).items()
            if key not in {"seen", "frames", "windows", "exclusions"}
        },
        "ingested": tail.ingested,
        "last_watermark": tail.last_mark,
        "queued_events": tail.events,
        "queued_watermarks": tail.marks,
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        pickle.dump(state, file, protocol=5)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)
    return sha(path)


def save_runtime(tail, path):
    state = {
        "version": 1,
        "engine": tail.engine,
        "tail": {
            k: getattr(tail, k)
            for k in [
                "ingested",
                "closed",
                "last_mark",
                "events",
                "marks",
                "mark_seen",
                "frame_ready",
            ]
        },
        "lines": {
            name: {
                k: getattr(getattr(tail, name), k)
                for k in ["offset", "buffer", "identity", "rows", "timing"]
            }
            for name in ["stream", "watermarks"]
        },
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as f:
        pickle.dump(state, f, protocol=5)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    return sha(path)


def restore_runtime(run, path, expected_sha256, config, metadata, checkpoint):
    if sha(path) != expected_sha256:
        raise ValueError("runtime snapshot hash mismatch")
    # Rebuilding tens of thousands of acyclic event records otherwise triggers
    # repeated collections mid-load; refcounts still manage temporary buffers.
    collecting = gc.isenabled()
    gc.disable()
    try:
        state = StateReader(io.BytesIO(Path(path).read_bytes())).load()
    finally:
        if collecting:
            gc.enable()
    if state["version"] != 1 or type(state["engine"]) is not Preprocessor:
        raise ValueError("runtime snapshot version/type mismatch")
    engine = state["engine"]
    if engine.config != config or engine.metadata != metadata:
        raise ValueError("runtime snapshot config/metadata mismatch")
    tail = DirectoryTail(run, engine)
    tail.verify_checkpoint(checkpoint)
    for key, value in state["tail"].items():
        if key not in {
            "ingested",
            "closed",
            "last_mark",
            "events",
            "marks",
            "mark_seen",
            "frame_ready",
        }:
            raise ValueError("unsupported runtime tail key")
        setattr(tail, key, value)
    for name in ["stream", "watermarks"]:
        reader = getattr(tail, name)
        for key, value in state["lines"][name].items():
            if key not in {"offset", "buffer", "identity", "rows", "timing"}:
                raise ValueError("unsupported runtime reader key")
            setattr(reader, key, value)
        if reader.offset != checkpoint["files"][str(reader.path)]["offset"]:
            raise ValueError("runtime snapshot offset mismatch")
    if tail.last_mark != checkpoint["last_watermark"]:
        raise ValueError("runtime snapshot watermark mismatch")
    return engine, tail
