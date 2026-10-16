import copy
import json
import pickle
import time

import pytest
from test_preprocessing import CONFIG, META, events

from capstone_ai.file_wait import FileWaiter
from capstone_ai.preprocessing import Preprocessor
from capstone_ai.resume_state import restore_runtime, save_runtime
from capstone_ai.source import DirectoryTail, sha


def test_fast_runtime_restore_partial_rows_queued_marks_and_duplicate_semantics(tmp_path):
    obs = tmp_path / "observations"
    obs.mkdir()
    es = list(events(63))
    cut = 6030
    original = b"".join((json.dumps(e) + "\n").encode() for e in es)
    prefix = b"".join((json.dumps(e) + "\n").encode() for e in es[:cut])
    stream = obs / "streams.jsonl"
    stream.write_bytes(prefix + original[len(prefix) : len(prefix) + 51])
    marks = []
    for count, through, closed in [(cut, 60200000, False), (len(es), 63000000, True)]:
        marks.append(
            {
                "schema": "drone.raw.watermark/2.0",
                "run_id": META["run_id"],
                "clock_id": "px4-boot",
                "through_available_time_us": through,
                "recorded_events": count,
                "closed": closed,
            }
        )
    (obs / "watermarks.jsonl").write_text("".join(json.dumps(m) + "\n" for m in marks))
    engine = Preprocessor(META, CONFIG)
    tail = DirectoryTail(tmp_path, engine, len(original) + 1)
    tail.poll()
    assert tail.stream.buffer and tail.marks
    checkpoint = tail.checkpoint()
    digest = save_runtime(tail, tmp_path / "runtime.bin")
    restored, rtail = restore_runtime(
        tmp_path, tmp_path / "runtime.bin", digest, CONFIG, META, checkpoint
    )
    assert restored.frames == engine.frames and restored.windows == engine.windows
    with stream.open("ab") as f:
        f.write(original[len(prefix) + 51 :])
    tail.poll()
    rtail.poll()
    assert tail.closed and rtail.closed
    assert restored.frames == engine.frames and restored.windows == engine.windows
    assert restored.exclusions == engine.exclusions
    restored.ingest(es[0])  # Historical duplicate remains recognized after fast recovery.
    bad = copy.deepcopy(es[0])
    bad["fields"]["output"][0] = 999
    with pytest.raises(ValueError, match="conflicting"):
        restored.ingest(bad)
    (tmp_path / "runtime.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        restore_runtime(tmp_path, tmp_path / "runtime.bin", digest, CONFIG, META, checkpoint)


def test_snapshot_rejects_executable_pickle_class(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(pickle.dumps(eval))
    with pytest.raises(ValueError, match="unsupported runtime snapshot class"):
        restore_runtime(tmp_path, path, sha(path), CONFIG, META, {})


def test_file_write_wakes_wait_without_timeout(tmp_path):
    obs = tmp_path / "observations"
    obs.mkdir()
    for name in ["streams.jsonl", "watermarks.jsonl"]:
        (obs / name).touch()
    waiter = FileWaiter(tmp_path)
    try:
        assert not waiter.wait(0)
        (obs / "streams.jsonl").write_bytes(b"partial row")
        # Event is already pending: no timing-sensitive thread scheduling assertion.
        start = time.monotonic()
        assert waiter.wait(1)
        assert time.monotonic() - start < 0.5
        assert not waiter.wait(0)
    finally:
        waiter.close()


def test_json_storage_shares_labels_without_changing_values(tmp_path):
    from capstone_ai.source import Lines

    first = list(events(1))[0]
    first["fields"]["output"][0] = 0
    second = copy.deepcopy(first)
    second["seq"] += 1
    second["fields"]["output"][0] = 0.0  # Numeric Python equality remains unchanged.
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    decoded = Lines(path).poll(10000)
    assert decoded == [first, second]
    assert decoded[0]["topic"] is decoded[1]["topic"]
    assert decoded[0]["field_quality"]["output"] is decoded[1]["field_quality"]["output"]
    engine = Preprocessor(META, CONFIG)
    engine.ingest(decoded[0])
    equal_duplicate = copy.deepcopy(decoded[0])
    equal_duplicate["fields"]["output"][0] = 0.0
    engine.ingest(equal_duplicate)
    assert engine.diagnostics["duplicates"] == 1


def test_consumed_watermark_tracking_is_released_with_repeated_marks(tmp_path):
    obs = tmp_path / "observations"
    obs.mkdir()
    (obs / "streams.jsonl").touch()
    mark = {
        "schema": "drone.raw.watermark/2.0",
        "run_id": META["run_id"],
        "clock_id": "px4-boot",
        "through_available_time_us": 0,
        "recorded_events": 0,
        "closed": False,
    }
    (obs / "watermarks.jsonl").write_text((json.dumps(mark) + "\n") * 3)
    tail = DirectoryTail(tmp_path, Preprocessor(META, CONFIG))
    tail.poll()
    assert tail.mark_seen == {}
    assert tail.last_mark == mark


def test_live_gc_deferral_restores_prior_policy_even_on_failure():
    import gc

    from capstone_ai.live_inference import live_gc_scope

    was_enabled = gc.isenabled()
    try:
        gc.enable()
        with pytest.raises(ValueError):
            with live_gc_scope(True):
                assert not gc.isenabled()
                raise ValueError("bad checkpoint")
        assert gc.isenabled()
        gc.disable()
        with live_gc_scope(True):
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        if was_enabled:
            gc.enable()


def test_shadow_prefix_acceptance_preserves_future_causality_and_rejects_lag(tmp_path):
    from capstone_ai.resume_state import adopt_shadow

    obs = tmp_path / "observations"
    obs.mkdir()
    source = obs / "streams.jsonl"
    marks = obs / "watermarks.jsonl"
    es = list(events(64))

    def append(chunk, through, count):
        with source.open("a") as file:
            file.write("".join(json.dumps(e) + "\n" for e in chunk))
        with marks.open("a") as file:
            file.write(
                json.dumps(
                    {
                        "schema": "drone.raw.watermark/2.0",
                        "run_id": META["run_id"],
                        "clock_id": "px4-boot",
                        "through_available_time_us": through,
                        "recorded_events": count,
                        "closed": False,
                    }
                )
                + "\n"
            )

    cut = sum(e["available_time_us"] <= 62000000 for e in es)
    primary = DirectoryTail(tmp_path, Preprocessor(META, CONFIG), 10**8)
    shadow = DirectoryTail(tmp_path, Preprocessor(META, CONFIG), 10**8)
    append(es[:cut], 62000000, cut)
    primary.poll()
    checkpoint = primary.checkpoint()
    with pytest.raises(ValueError, match="shadow has not ingested"):
        adopt_shadow(shadow, CONFIG, META, checkpoint)
    shadow.poll()
    past = copy.deepcopy(primary.engine.windows)
    # Shadow may advance farther; issued windows at earlier origins must be unchanged.
    append(es[cut:], 64000000, len(es))
    shadow.poll()
    engine, _ = adopt_shadow(shadow, CONFIG, META, checkpoint)
    assert engine.windows[: len(past)] == past
    primary.poll()
    assert engine.frames == primary.engine.frames and engine.windows == primary.engine.windows
    bad = copy.deepcopy(CONFIG)
    bad["history"] += 1
    with pytest.raises(ValueError, match="shadow config"):
        adopt_shadow(shadow, bad, META, checkpoint)
    original = source.read_bytes()
    source.write_bytes(b" " + original[1:])
    with pytest.raises(ValueError, match="prefix changed"):
        adopt_shadow(shadow, CONFIG, META, checkpoint)


def test_handover_persists_causal_state_without_mutating_event_archive(tmp_path):
    import io

    from test_preprocessing import run

    from capstone_ai.resume_state import StateReader, save_handover_state

    engine = run(list(events(64)))
    tail = DirectoryTail(tmp_path, engine)
    count = len(engine.seen)
    frames = copy.deepcopy(engine.frames)
    path = tmp_path / "handover.bin"
    digest = save_handover_state(tail, path)
    saved = StateReader(io.BytesIO(path.read_bytes())).load()
    assert digest == sha(path)
    assert list(saved["engine"]["history"]) == list(engine.history)
    assert list(saved["engine"]["context"]) == list(engine.context)
    for key in [
        "charge",
        "energy",
        "integral_coverage_s",
        "watermark",
        "next_t",
        "signals",
        "pending",
    ]:
        assert saved["engine"][key] == getattr(engine, key)
    for key, value in engine.filters.items():
        assert vars(saved["engine"]["filters"][key]) == vars(value)
    assert len(engine.seen) == count and engine.frames == frames
    assert "seen" not in saved["engine"]  # Recreated by shadow's verified full raw prefix.
