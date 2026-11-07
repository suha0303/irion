import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_localflight_adapt import flight

from capstone_ai.mapping import FEATURES
from localflight.catalog import digest
from localflight.prepare import bin_events, bin_values, causal_frames, prepare
from localflight.prepared import distinct_flights, load, windows
from localflight.runtime import Runtime
from localflight.timing import ClockAudit, clock_maps, split_streams


def events(seconds=40, gap=None):
    rows = []
    for t in range(0, seconds * 1000000, 100000):
        if gap and gap[0] <= t < gap[1]:
            continue
        values = dict.fromkeys(FEATURES, 0.0)
        values.update({k: 0.4 for k in FEATURES[:4]})
        values.update(
            pack_voltage=16.0,
            pack_current=8.0,
            specific_force_z=-9.81,
            rate_target_p=0.0,
            rate_target_q=0.0,
            rate_target_r=0.0,
            climb_target=0.0,
        )
        rows.append((t, t, values))
    return rows


def test_gap_retains_two_segments_and_does_not_bridge():
    v, t, r = causal_frames(events(gap=(15000000, 20000000)), 0, 40000000)
    usable = [s for s in r["candidate_segments"] if s["usable"]]
    assert len(usable) == 2
    assert usable[0]["end_us"] <= 15300000 and usable[1]["start_us"] == 20000000
    assert r["missing_or_stale_frames"] > 40
    assert np.isnan(v[(t > 16000000) & (t < 20000000)]).all()


def test_future_measurements_do_not_change_past_frames():
    source = events()
    before = causal_frames(source, 0, 40000000)[0]
    changed = [
        (a, s, {**v, "pack_voltage": 30.0}) if a > 20000000 else (a, s, v)
        for a, s, v in source
    ]
    after = causal_frames(changed, 0, 40000000)[0]
    np.testing.assert_array_equal(before[:201], after[:201])
    assert not np.array_equal(before[220:], after[220:])


def test_clock_audit_is_instance_local():
    a = ClockAudit()
    for t in [1000000, 1100000, 1100000, 3000000, 100000]:
        a.add("IMU:0", t)
    for t in [900000, 1000000]:
        a.add("IMU:1", t)
    s = a.result()["streams"]
    assert s["IMU:0"]["reversals"] == 1 and s["IMU:0"]["duplicates"] == 1
    assert s["IMU:0"]["max_gap_us"] == 1900000 and len(s["IMU:0"]["spans"]) == 3
    assert s["IMU:1"]["reversals"] == 0


def test_affine_clock_correction_and_anchor_support():
    maps = clock_maps(
        {
            "BAT:0": dict(
                anchors_us=[[1000000, 1100000], [21000000, 21102000]],
                evidence="paired sync pulses",
            )
        }
    )
    streams = {
        "BAT:0": [(0, 0, {}), (1000000, 1000000, {}), (21000000, 21000000, {})]
    }
    out, r = split_streams(streams, maps)
    assert [e[0] for e in out] == [1100000, 21102000]
    assert r["BAT:0"]["outside_anchor_range"] == 1
    assert abs(maps["BAT:0"]["drift_ppm"] - 100) < 1e-6
    with pytest.raises(ValueError, match="evidence"):
        clock_maps({"BAT:0": dict(anchors_us=[[0, 0], [1, 1]])})
    with pytest.raises(ValueError, match="increasing"):
        clock_maps({"BAT:0": dict(anchors_us=[[0, 1], [1, 0]], evidence="bad")})
    with pytest.raises(ValueError, match="Unknown"):
        split_streams(streams, {"missing": maps["BAT:0"]})


def test_reversal_is_not_silently_sorted_or_aligned_by_count():
    with pytest.raises(ValueError, match="epoch boundary"):
        split_streams({"RATE:0": [(10, 10, {}), (20, 20, {}), (1, 1, {})]}, {})
    out, report = split_streams(
        {
            "RATE:0": [
                (10, 10, {"x": 1}),
                (10, 10, {"x": 2}),
                (20, 20, {"x": 3}),
            ]
        },
        {},
    )
    assert (
        out[0][2]["x"] == 1
        and report["RATE:0"]["duplicate_records_removed"] == 1
    )


def test_onboard_units_cm_per_second_are_respected():
    message = SimpleNamespace(
        get_type=lambda: "CTUN",
        to_dict=lambda: dict(CRt=123.0, DCRt=250.0),
        fmt=SimpleNamespace(columns=["CRt", "DCRt"], units=["cm/s", "cm/s"]),
    )
    assert bin_values(message, {}) == dict(climb_rate=1.23, climb_target=2.5)
    message.fmt.units = ["unknown", "cm/s"]
    with pytest.raises(ValueError, match="Unverified onboard unit"):
        bin_values(message, {})


def test_prepared_segments_feed_only_contained_windows(tmp_path, monkeypatch):
    fixture = flight(tmp_path, "fixture", "evaluation")
    profile = json.loads(fixture.read_text())["profile"]
    profile.update(imu_instance=0, battery_instance=0, actuator_instance=0)
    raw = tmp_path / "log.ulg"
    raw.write_bytes(b"synthetic raw used only through test decoder")
    monkeypatch.setattr(
        "localflight.prepare.inspect_log",
        lambda p: dict(autopilot="px4", sha256=digest(p), parser_warnings=""),
    )
    monkeypatch.setattr(
        "localflight.prepare.px4_events",
        lambda *args: (events(gap=(15000000, 20000000)), {}),
    )
    preview = prepare(raw, profile, preview=True, start_us=0, end_us=40000000)
    assert len(preview["timing"]["candidate_segments"]) == 2
    incomplete = {**profile, "series_cells": None, "capacity_ah": None}
    for key in ("firmware", "battery_id", "mapping_evidence"):
        incomplete.pop(key, None)
    assert (
        prepare(raw, incomplete, preview=True, start_us=0, end_us=40000000)[
            "timing"
        ]["complete_frames"]
        > 0
    )
    with pytest.raises(ValueError, match="series_cells"):
        prepare(
            raw,
            incomplete,
            tmp_path / "not-admitted",
            start_us=0,
            end_us=40000000,
        )
    r = prepare(
        raw,
        profile,
        tmp_path / "prepared",
        start_us=0,
        end_us=40000000,
        interval_evidence="synthetic airborne fixture",
        role="evaluation",
    )
    m, v, t = load(r["manifest"])
    assert len(m["segments"]) == 2 and r["timing"]["coverage"] < 1
    x, y, _, origins = windows(m, v, t, Runtime("forecast_lstm_h64_l2").scaler)
    assert not np.any((origins > 14500000) & (origins < 29900000))
    assert len(x) == len(y) and len(x) > 0
    with pytest.raises(ValueError, match="same flight"):
        distinct_flights([m, m])
    # A manually merged gap must not pass the loader.
    m["segments"] = [dict(start_index=0, end_index=len(t))]
    path = tmp_path / "prepared/manifest.json"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Missing data"):
        load(path)


def test_short_segments_and_sample_hold_are_reported():
    _, _, report = causal_frames(
        events(seconds=20, gap=(5000000, 15000000)), 0, 20000000
    )
    assert not any(s["usable"] for s in report["candidate_segments"])
    with pytest.raises(ValueError, match="20..2000"):
        causal_frames(events(), 0, 40000000, fast_max_age_ms=2001)


def test_dropout_resets_state_even_without_a_missing_grid_frame():
    source = events()
    source.append((15050000, 15050000, {"__reset__": True}))
    _, _, report = causal_frames(
        sorted(source, key=lambda e: e[0]), 0, 40000000
    )
    assert len(report["candidate_segments"]) == 2


def test_large_finite_sensor_excursions_are_retained():
    source = events()
    source = [
        (a, s, {**v, "body_rate_p": 5000.0, "specific_force_x": -2000.0})
        for a, s, v in source
    ]
    values, _, report = causal_frames(source, 0, 40000000)
    assert report["missing_or_stale_frames"] == 0
    assert values[-1, 4] == pytest.approx(5000.0)
    assert values[-1, 12] == pytest.approx(-2000.0)


@pytest.mark.parametrize("failure", ["reset", "corrupt", "exception"])
def test_bin_keeps_only_file_prefix_at_first_reset_or_corruption(
    monkeypatch, failure
):
    import localflight.bin_reader as reader_module

    class Reader:
        offset = 0
        closed = False

        def recv_msg(self):
            self.offset += 10
            stamp = {10: 1000000, 20: 2000000, 30: 100000, 40: 3000000}.get(
                self.offset
            )
            if stamp is None:
                return None
            if self.offset == 30 and failure == "exception":
                raise TypeError("malformed parser offset")
            if self.offset == 30 and failure == "corrupt":
                print("bad msg at offset 30")
            return SimpleNamespace(
                TimeUS=stamp,
                get_type=lambda: "CTUN",
                to_dict=lambda: dict(CRt=100.0, DCRt=100.0),
                fmt=SimpleNamespace(
                    columns=["CRt", "DCRt"], units=["cm/s", "cm/s"]
                ),
            )

        def close(self):
            self.closed = True

    reader = Reader()
    monkeypatch.setattr(reader_module, "open_bin", lambda p: reader)
    rows, report = bin_events("fixture.BIN", {})
    assert reader.closed and [r[0] for r in rows] == [1000000, 2000000]
    assert (
        report["retained_prefix_boundary"]["reason"]
        == (
            {
                "reset": "clock_reset",
                "corrupt": "parser_warning",
                "exception": "parser_exception",
            }[failure]
        )
    )


def test_segmented_health_does_not_carry_context_between_segments(tmp_path):
    from localflight.analysis import analyze
    from localflight.catalog import DEFAULT_BUNDLE

    path = flight(tmp_path, "segmented", "evaluation")
    m = json.loads(path.read_text())
    data = path.parent / "frames.npz"
    with np.load(data) as z:
        v, t = z["values"], z["time_us"]
    t[125:] += 10000000
    m["segments"] = [
        dict(start_index=0, end_index=125),
        dict(start_index=125, end_index=250),
    ]
    np.savez(data, values=v, time_us=t)
    m["file"]["sha256"] = digest(data)
    path.write_text(json.dumps(m))
    first = analyze("run_health_lstm_h64_l2", path, bundle=DEFAULT_BUNDLE)
    v[125:, 17] += 3
    np.savez(data, values=v, time_us=t)
    m["file"]["sha256"] = digest(data)
    path.write_text(json.dumps(m))
    second = analyze("run_health_lstm_h64_l2", path, bundle=DEFAULT_BUNDLE)
    assert first["metrics"]["segments"][0] == second["metrics"]["segments"][0]
    assert first["metrics"]["segments"][1] != second["metrics"]["segments"][1]
    assert (
        first["health_verdict"] is None
        and not first["metrics"]["whole_flight_verdict"]
    )
