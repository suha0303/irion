"""Check causal live inputs, stale data, reset handling and flight
completion.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from localflight import live
from localflight.calibration import admit_completed
from localflight.live import DisarmGuard, Mapper, completed, decode, validate
from localflight.prepare import causal_frames
from localflight.prepared import load


def request():
    return dict(
        profile=dict(
            schema="drone.real-aircraft/1",
            id="live-fixture",
            name="Fixture",
            autopilot="px4",
            firmware="fixture",
            rotor_count=4,
            series_cells=4,
            capacity_ah=12,
            battery_id="fixture",
            motor_channels=dict(
                front_right=1, front_left=2, rear_right=3, rear_left=4
            ),
            output_min=1000,
            output_max=2000,
            mapping_evidence="synthetic fixture",
            imu_instance=0,
            battery_instance=0,
            actuator_instance=0,
        ),
        source_system=1,
        source_component=1,
        scope="simulation_test",
        imu_message="HIGHRES_IMU",
        imu_id=0,
        forecast_model="forecast_lstm_h64_l2",
        post_model="run_health_lstm_h64_l2",
    )


def samples(seconds=13, start_ns=1000000000):
    for t in range(seconds * 10):
        ms = t * 100
        rows = [
            ("HEARTBEAT", dict(autopilot=12, base_mode=0)),
            (
                "ATTITUDE",
                dict(
                    time_boot_ms=ms,
                    roll=0,
                    pitch=0,
                    rollspeed=0.01,
                    pitchspeed=0.02,
                    yawspeed=0.03,
                ),
            ),
            (
                "ATTITUDE_TARGET",
                dict(
                    time_boot_ms=ms,
                    type_mask=0,
                    body_roll_rate=0,
                    body_pitch_rate=0,
                    body_yaw_rate=0,
                ),
            ),
            (
                "HIGHRES_IMU",
                dict(
                    time_usec=ms * 1000,
                    id=0,
                    fields_updated=7,
                    xacc=0,
                    yacc=0,
                    zacc=-9.81,
                ),
            ),
            ("LOCAL_POSITION_NED", dict(time_boot_ms=ms, vz=-1)),
            (
                "POSITION_TARGET_LOCAL_NED",
                dict(time_boot_ms=ms, coordinate_frame=1, type_mask=0, vz=-1),
            ),
            (
                "SERVO_OUTPUT_RAW",
                dict(
                    time_usec=ms * 1000,
                    port=0,
                    **{f"servo{i}_raw": 1400 for i in range(1, 5)},
                ),
            ),
            (
                "BATTERY_STATUS",
                dict(
                    id=0, voltages=[4000] * 4 + [65535] * 6, current_battery=800
                ),
            ),
        ]
        for name, fields in rows:
            yield dict(
                type="mavlink",
                raw=dict(
                    receive_monotonic_ns=start_ns + ms * 1000000,
                    source_system=1,
                    source_component=1,
                    message_name=name,
                    fields=fields,
                ),
            )


def test_live_alarm_history_survives_completion(tmp_path, monkeypatch):
    """Replay a deterministic clock through the real CPU worker and recorder."""
    clock = dict(now=1.0)
    accept = live.Mapper.accept

    def advance(mapper, item):
        clock["now"] = item["raw"]["receive_monotonic_ns"] / 1e9
        return accept(mapper, item)

    monkeypatch.setattr(live.Mapper, "accept", advance)
    monkeypatch.setattr(
        live,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock["now"],
            monotonic_ns=lambda: int(clock["now"] * 1e9),
            time=lambda: clock["now"],
        ),
    )
    # A test-only threshold exercises the alarm path; no production calibration
    # or model artifact is changed and this is not detection-performance proof.
    monkeypatch.setattr(
        live.localflight_calibration,
        "active",
        lambda *_: dict(
            id="synthetic-test",
            source_sha256=[],
            threshold=0,
            unit="fixture",
            validation_status="synthetic_test",
        ),
    )
    journal = tmp_path / "journal.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in samples(16)))
    output = tmp_path / "live"
    output.mkdir()
    (output / "finish").touch()
    live.run(
        dict(
            request(),
            journal=str(journal),
            output=str(output),
            data_root=str(tmp_path),
        )
    )
    rows = [
        json.loads(line)
        for line in (output / "alarms.jsonl").read_text().splitlines()
    ]
    assert rows
    times = [row["decision"]["time_us"] for row in rows]
    assert times == sorted(set(times))
    for row in rows:
        assert row["model_id"] == request()["forecast_model"]
        assert row["decision"]["status"] == "suspect"
        assert max(row["decision"]["target_scores"]) == pytest.approx(
            row["decision"]["scores"][0]
        )
    final = json.loads((output / "state.json").read_text())
    assert final["status"] == "finished"
    assert final["result"]["forecast_analyses"]


def test_units_masks_instances_and_unknowns():
    r = request()
    p = r["profile"]
    assert validate(r)["origin"] == "synthetic_unit_fixture"
    assert (
        decode(
            "POSITION_TARGET_LOCAL_NED",
            dict(vz=2, type_mask=32, coordinate_frame=1),
            p,
            r,
        )["climb_target"]
        is None
    )
    assert (
        decode(
            "POSITION_TARGET_LOCAL_NED",
            dict(vz=2, type_mask=0, coordinate_frame=8),
            p,
            r,
        )["climb_target"]
        is None
    )
    assert decode("BATTERY_STATUS", dict(id=1), p, r) == {}
    assert decode(
        "BATTERY_STATUS",
        dict(id=0, voltages=[16000] + [65535] * 9, current_battery=-1),
        p,
        r,
    ) == dict(pack_voltage=16, pack_current=None)
    assert decode(
        "HIGHRES_IMU", dict(id=0, fields_updated=2, yacc=2), p, r
    ) == dict(specific_force_y=2)
    r["imu_message"] = "SCALED_IMU2"
    assert (
        decode("SCALED_IMU2", dict(xacc=0, yacc=0, zacc=-1000), p, r)[
            "specific_force_z"
        ]
        == -9.80665
    )
    r["profile"]["autopilot"] = "ardupilot"
    assert validate(r)["valid"]
    with pytest.raises(ValueError, match="autopilot"):
        Mapper(r).accept(next(samples()))


def test_clock_source_isolation_and_gaps():
    r = request()
    mapper = Mapper(r)
    row = next(samples())
    row["raw"]["source_system"] = 2
    assert mapper.accept(row) == [] and mapper.base is None
    events = [e for row in samples(13) for e in mapper.accept(row)]
    values, _, report = causal_frames(events, 0, 13000000)
    assert report["candidate_segments"][0]["usable"]
    assert np.isfinite(values).all() and np.all(values[:, :4] == 0.4)
    assert np.all(values[:, 17] == 16)
    # Duplicate source stamps cannot refresh an old sample.
    row = list(samples(1))[-2]
    row["raw"]["receive_monotonic_ns"] = 14000000000
    row["raw"]["fields"]["time_usec"] = 12900000
    assert mapper.accept(row) == []
    assert mapper.duplicates == 1
    row["raw"]["receive_monotonic_ns"] += 1000000
    row["raw"]["fields"]["time_usec"] = 1
    assert mapper.accept(row)[0][2] == {"__reset__": True}
    with pytest.raises(ValueError, match="역행"):
        mapper.accept(next(samples()))
    # Disconnect/long gap must not construct one history across two pieces.
    mapper = Mapper(r)
    rows = list(samples(6)) + list(samples(6, start_ns=21000000000))
    events = [e for row in rows for e in mapper.accept(row)]
    _, _, report = causal_frames(events, 0, 26000000)
    assert not any(s["usable"] for s in report["candidate_segments"])


def test_end_report_runs_real_cpu_without_health_claim(tmp_path):
    r = request()
    mapper = Mapper(r)
    rows = list(samples())
    events = [e for row in rows for e in mapper.accept(row)]
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    result = completed(
        events, journal, tmp_path / "out", r, "synthetic_unit_fixture"
    )
    assert result["status"] == "complete" and result["health_verdict"] is None
    assert (
        result["forecast_analyses"][0]["analysis"]["model_id"]
        == r["forecast_model"]
    )
    assert {item["model_id"] for item in result["run_summary"]["models"]} == {
        r["forecast_model"],
        r["post_model"],
    }
    analysis = result["analyses"][0]["analysis"]
    assert analysis["metrics"]["frames"] >= 110
    assert not analysis["real_aircraft_validated"]
    manifest = result["analyses"][0]["manifest"]
    with pytest.raises(ValueError, match="Confirmed normal"):
        load(manifest, normal=True)

    with pytest.raises(ValueError, match="정상 확인"):
        admit_completed(
            manifest,
            tmp_path / "rejected",
            role="calibration",
            health="unknown",
            evidence="unknown",
        )
    admitted = admit_completed(
        manifest,
        tmp_path / "declared",
        role="calibration",
        health="confirmed_normal",
        evidence="Synthetic fixture only",
    )
    new_metadata, _, _ = load(admitted["manifest"], normal=True)
    old_metadata, _, _ = load(manifest)
    assert old_metadata["health"] == "unknown"
    assert new_metadata["source_sha256"] == old_metadata["source_sha256"]
    assert new_metadata["live_input"]["imu_message"] == "HIGHRES_IMU"


def test_disarm_requires_armed_and_fresh_debounced_heartbeats():
    guard = DisarmGuard()
    for t in range(5):
        guard.accept(False, t * 1_000_000)
    assert not guard.ready
    guard.accept(True, 5_000_000)
    guard.accept(False, 6_000_000)
    for _ in range(10):
        guard.accept(False, 6_000_000)
    assert not guard.ready
    guard.accept(False, 10_000_000)  # gap restarts confirmation
    guard.accept(False, 11_000_000)
    assert not guard.ready
    guard.accept(False, 12_000_000)
    assert guard.ready
    guard.accept(True, 13_000_000)
    assert not guard.ready
    guard.accept(None, 14_000_000)
    assert not guard.ready
    mapper = Mapper(request())
    for row in samples(1):
        mapper.accept(row)
    assert not mapper.disarm.ready
    row = next(samples())
    row["raw"]["source_system"] = 2
    row["raw"]["fields"]["base_mode"] = 128
    mapper.accept(row)
    assert not mapper.disarm.seen_armed
