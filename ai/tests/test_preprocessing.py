import copy
import json
import math
from pathlib import Path

import pytest

from capstone_ai.mapping import mapped
from capstone_ai.preprocessing import LowPass, Preprocessor
from capstone_ai.source import DirectoryTail, Lines, dump, read_json

ROOT = Path(__file__).resolve().parents[2]
META = read_json(Path(__file__).parent / "fixtures/synthetic_metadata.json")
CONFIG = read_json(ROOT / "ai/configs/preprocessing.json")


def events(seconds=72):
    values = {
        "actuator_outputs": {"output": [100, 200, 300, 400]},
        "vehicle_angular_velocity": {"xyz": [1.0, 2.0, 3.0]},
        "vehicle_rates_setpoint": {"roll": 1.1, "pitch": 2.2, "yaw": 3.3},
        "vehicle_attitude": {"q": [1.0, 0.0, 0.0, 0.0]},
        "vehicle_acceleration": {"xyz": [0.0, 0.0, -9.81]},
        "vehicle_local_position": {"x": 0.0, "y": 0.0, "z": -2.0, "vz": -0.5, "z_valid": 1},
        "vehicle_local_position_setpoint": {"vx": 0.0, "vy": 0.0, "vz": -0.6},
        "battery_status": {"voltage_v": 14.8, "current_a": 6.0},
        "vehicle_status": {"arming_state": 2, "nav_state": 14, "hil_state": 0},
        "vehicle_land_detected": {"landed": 0},
    }
    for j in range(seconds * 10 + 1):
        for topic, fields in values.items():
            yield {
                "schema": "drone.raw.event/2.0",
                "run_id": META["run_id"],
                "source": "px4.uorb",
                "topic": topic,
                "instance": 0,
                "seq": j,
                "sample_time_us": j * 100000,
                "available_time_us": j * 100000,
                "clock_id": "px4-boot",
                "fields": copy.deepcopy(fields),
                "field_quality": dict.fromkeys(fields, "valid"),
            }


def run(es, chunk=317):
    p = Preprocessor(META, CONFIG)
    for start in range(0, len(es), chunk):
        part = es[start : start + chunk]
        for e in part:
            p.ingest(e)
        # Finalize strictly below the last timestamp, allowing ties in the next batch.
        p.advance(part[-1]["clock_id"], part[-1]["available_time_us"] - 1)
    p.advance(es[-1]["clock_id"], es[-1]["available_time_us"])
    return p


def test_known_mapping_context_and_window():
    p = run(list(events()))
    f = p.frames[600]
    assert f["time_us"] == 60000000
    assert f["values"]["motor_request_front_left"] == 0.3
    assert f["values"]["motor_request_rear_left"] == 0.2
    assert f["values"]["climb_rate"] == 0.5
    assert f["values"]["specific_force_z"] == -9.81
    assert f["values"]["body_rate_error_p"] == pytest.approx(0.1)
    assert f["context"] == pytest.approx([3.7, 0.0, 1.0, 0.0, 1 / 60, 1 / 60])
    assert p.frames[599]["warmup"] is True
    assert p.windows[0]["origin_us"] == 60000000
    assert len(p.windows[0]["X"]) == 100
    assert len(p.windows[0]["X"][0]) == 19
    assert p.windows[0]["history_times_us"][0] == 50100000
    assert p.windows[0]["target_times_us"] == list(range(60100000, 61100000, 100000))


def test_filter_analytic_step_and_frequency():
    f = LowPass(0, 0.0, 0.08)
    f.update(0, 1.0)
    t = 0.16
    assert f.at(int(t * 1e6))[1] == pytest.approx(1 - (1 + t / 0.08) * math.exp(-t / 0.08))
    f.update(70000, 1.0)
    assert f.at(160000)[1] == pytest.approx(1 - 3 * math.exp(-2))

    # Analytic continuous cascade gain; verify numerical driven response at 1 and 20 Hz.
    def gain(hz):
        lp = LowPass(0, 0.0, 0.08)
        outputs = []
        for i in range(1, 50001):
            t = i * 100
            lp.update(t, math.sin(2 * math.pi * hz * t / 1e6))
            if t > 3000000:
                outputs.append(lp.y2)
        return (max(outputs) - min(outputs)) / 2

    assert gain(1) == pytest.approx(1 / (1 + (2 * math.pi * 0.08) ** 2), rel=0.01)
    assert gain(20) < 0.011


def test_future_mutation_does_not_change_prefix_or_windows():
    es = list(events())
    a = run(es)
    altered = copy.deepcopy(es)
    for e in altered:
        if e["sample_time_us"] > 65000000 and e["topic"] == "battery_status":
            e["fields"]["voltage_v"] = 1000.0
            e["fields"]["current_a"] = 200.0
    b = run(altered, chunk=503)
    assert [f for f in a.frames if f["time_us"] <= 65000000] == [
        f for f in b.frames if f["time_us"] <= 65000000
    ]
    assert [w for w in a.windows if w["origin_us"] <= 65000000] == [
        w for w in b.windows if w["origin_us"] <= 65000000
    ]


def test_late_available_and_old_sample():
    es = list(events(2))
    delayed = next(copy.deepcopy(e) for e in es if e["topic"] == "battery_status")
    delayed["seq"] = 1000
    delayed["available_time_us"] = 1500000
    delayed["fields"]["voltage_v"] = 999.0
    es.append(delayed)
    es.sort(key=lambda e: e["available_time_us"])
    p = run(es)
    assert all(f["values"]["pack_voltage"] == pytest.approx(14.8) for f in p.frames)
    assert p.diagnostics["old_sample_discarded"] == 2


def test_duplicates_reversal_and_snapshot():
    es = list(events(2))
    p = Preprocessor(META, CONFIG)
    for e in reversed(es):
        p.ingest(e)
    p.ingest(es[0])
    p.advance("px4-boot", 1000000)
    q = p.snapshot()
    p.advance("px4-boot", 2000000)
    q.advance("px4-boot", 2000000)
    assert p.frames == q.frames
    assert p.diagnostics["duplicates"] == 1
    assert p.frames == run(es).frames
    bad = copy.deepcopy(es[0])
    bad["fields"]["output"][0] = 999
    with pytest.raises(ValueError, match="conflicting"):
        p.ingest(bad)


def test_gap_clock_instance_reset_and_quality():
    es = [e for e in events() if not (10000000 < e["available_time_us"] < 13000000)]
    p = run(es)
    assert p.diagnostics["reset_signal_gap"] >= 1
    assert any("stale" in f["quality"].values() for f in p.frames)
    assert not p.windows  # Only 59s after gap, cannot satisfy 60s context.
    for e in events(1):
        e["clock_id"] = "boot-2"
        p.ingest(e)
    p.advance("boot-2", 1000000)
    assert p.diagnostics["reset_clock_change"] == 1
    assert p.frames[-1]["warmup"]
    p2 = Preprocessor(META, CONFIG)
    for e in events(2):
        if e["topic"] == "vehicle_angular_velocity" and e["sample_time_us"] >= 1000000:
            e["instance"] = 1
        p2.ingest(e)
    p2.advance("px4-boot", 2000000)
    assert p2.diagnostics["reset_instance_change"] == 1
    with pytest.raises(ValueError, match="clock reset"):
        p2.advance("px4-boot", 0)


def test_partial_lines_visibility_and_restart(tmp_path):
    obs = tmp_path / "observations"
    obs.mkdir()
    stream = obs / "streams.jsonl"
    water = obs / "watermarks.jsonl"
    rows = list(events(1))
    blob = "".join(json.dumps(e) + "\n" for e in rows).encode()
    stream.write_bytes(blob[:123])
    mark = {
        "schema": "drone.raw.watermark/2.0",
        "run_id": META["run_id"],
        "clock_id": "px4-boot",
        "through_available_time_us": 1000000,
        "recorded_events": len(rows),
        "closed": True,
    }
    water.write_text(json.dumps(mark) + "\n")
    p = Preprocessor(META, CONFIG)
    tail = DirectoryTail(tmp_path, p, 999999)
    tail.poll()
    assert not p.frames and not tail.closed
    with stream.open("ab") as f:
        f.write(blob[123:])
    tail.poll()
    assert tail.closed and len(p.frames) == 11
    checkpoint = tail.checkpoint()
    dump(tmp_path / "checkpoint.json", checkpoint)
    q = Preprocessor(META, CONFIG)
    restored = DirectoryTail(tmp_path, q)
    restored.verify_checkpoint(read_json(tmp_path / "checkpoint.json"))
    restored.poll()
    assert q.frames == p.frames
    with stream.open("r+b") as f:
        f.write(b"!")
    with pytest.raises(ValueError, match="prefix changed"):
        restored.verify_checkpoint(read_json(tmp_path / "checkpoint.json"))


def test_quality_null_clipped_and_quaternion():
    e = next(e for e in events(1) if e["topic"] == "vehicle_attitude")
    e["fields"]["q"] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]
    assert mapped(e, META)["roll"][0] == pytest.approx(math.pi / 2)
    e["fields"]["q"] = [0, 0, 0, 0]
    assert mapped(e, META)["roll"] == (None, "invalid")
    e = next(e for e in events(1) if e["topic"] == "battery_status")
    e["field_quality"]["current_a"] = "clipped"
    assert mapped(e, META)["pack_current"] == (None, "clipped")
    e["fields"]["voltage_v"] = None
    assert mapped(e, META)["pack_voltage"] == (None, "unavailable")


def test_file_replacement(tmp_path):
    path = tmp_path / "x"
    path.write_text("{}\n")
    reader = Lines(path)
    assert reader.poll(100) == [{}]
    path.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        reader.poll(100)
