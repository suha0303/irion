"""Exercise short receipt gaps, measured targets, and landing confirmation."""

import numpy as np
import pytest
from test_localflight_live import request, samples
from test_localflight_timing import events

from capstone_ai.mapping import FEATURES
from localflight.live import DisarmGuard, Mapper, arrived_decision, completed
from localflight.prepare import CausalSampler, causal_frames
from localflight.prepared import observed_targets, windows


def battery_gap():
    source = events(seconds=22)
    for stamp, _, values in source:
        if 10_800_000 <= stamp <= 11_100_000:
            values.pop("pack_voltage")
            values.pop("pack_current")
    return source


def test_short_gap_preserves_history_without_fabricating_targets():
    source = battery_gap()
    values, times, timing = causal_frames(source, 0, 22_000_000, fill_gaps=True)
    assert timing["boundary_count"] == 0
    assert timing["missing_or_stale_frames"] == 0
    assert timing["filled_target_times_us"] == {"pack_voltage": [11_100_000]}
    assert len(timing["candidate_segments"]) == 1
    strict = causal_frames(source, 0, 22_000_000)[2]
    assert len(strict["candidate_segments"]) == 2
    chunked = CausalSampler(fill_gaps=True)
    first = chunked.sample(source, 0, 11_200_000)
    second = chunked.sample(source, 11_200_000, 22_000_000)
    np.testing.assert_array_equal(first[0] + second[0], values)
    np.testing.assert_array_equal(first[1] + second[1], times)
    assert chunked.filled_target_times == timing["filled_target_times_us"]

    metadata = dict(timing=timing, profile=dict(series_cells=4, capacity_ah=12))
    scaler = dict(
        x_mean=np.zeros(19),
        x_scale=np.ones(19),
        y_mean=np.zeros(4),
        y_scale=np.ones(4),
    )
    _, _, _, origins = windows(metadata, values, times, scaler)
    assert 10_900_000 not in origins
    assert 11_900_000 in origins
    pending = [
        dict(origin=10_900_000, due=11_900_000, reset=0, scaled=[0, 0, 0, 4])
    ]
    future, decision = arrived_decision(
        pending,
        values,
        times,
        0,
        0,
        4,
        scaler,
        None,
        measured=observed_targets(metadata, times),
    )
    assert future == [] and decision is None


@pytest.mark.parametrize("feature", ["body_rate_p", "pack_voltage"])
def test_shared_one_second_cap_keeps_boundary_sample_then_splits(feature):
    source = events(seconds=22)
    for _, _, row in source[108:119]:
        row.pop(feature)
    values, times, report = causal_frames(source, 0, 22_000_000, fill_gaps=True)
    index = FEATURES.index(feature)
    assert np.isfinite(values[times == 11_700_000, index]).all()
    assert np.isnan(values[times == 11_800_000, index]).all()
    assert report["filled_target_times_us"][feature][-1] == 11_700_000
    assert report["missing_or_stale_frames"] == 1
    assert len(report["candidate_segments"]) == 2


def test_explicit_bad_values_still_split_history():
    source = events(seconds=22)
    source[110][2]["pack_voltage"] = None
    _, _, report = causal_frames(source, 0, 22_000_000, fill_gaps=True)
    assert report["missing_or_stale_frames"] > 0
    assert not report["filled_target_times_us"]


def test_descent_is_retained_after_airborne_but_ground_is_excluded():
    source = events(seconds=22)
    for t, _, updates in source:
        updates["landed_state"] = (
            1 if t < 1e6 or t >= 18e6 else 4 if t >= 15e6 else 2
        )
    _, _, timing = causal_frames(
        source, 0, 22_000_000, require_airborne=True, fill_gaps=True
    )
    assert [
        (s["start_us"], s["end_us"]) for s in timing["candidate_segments"]
    ] == [(1_000_000, 18_000_000)]


def test_automatic_finish_needs_disarm_and_ten_seconds_of_data_silence():
    guard = DisarmGuard()
    guard.accept(True, 0)
    assert not guard.landing_ready(60_000_000, 0)
    for t in (1_000_000, 2_000_000, 3_000_000):
        guard.accept(False, t)
    assert not guard.landing_ready(10_999_999, 0)
    assert guard.landing_ready(11_000_000, 0)
    assert not guard.landing_ready(11_000_000, 3_000_000)
    assert guard.landing_ready(13_000_000, 3_000_000)
    guard.accept(True, 13_000_000)
    assert not guard.landing_ready(99_000_000, 0)

    mapper = Mapper(request())
    rows = list(samples(1))
    for row in rows:
        mapper.accept(row)
    assert mapper.last_measurement_us == 900_000
    hb = next(samples(start_ns=4_000_000_000, armed=False))
    mapper.accept(hb)
    assert mapper.last_measurement_us == 900_000


def test_saved_analysis_excludes_held_voltage_from_scoring(tmp_path):
    journal = tmp_path / "source.jsonl"
    journal.write_text("synthetic fixture\n")
    result = completed(
        battery_gap(),
        journal,
        tmp_path / "completed",
        request(),
        "synthetic_unit_fixture",
        end_us=22_000_000,
    )
    assert result["status"] == "complete"
    assert len(result["analyses"]) == 1
    health = result["analyses"][0]["analysis"]
    forecast = result["forecast_analyses"][0]["analysis"]
    assert health["metrics"]["segments"][0]["components"]["samples"] == 119
    assert forecast["metrics"]["windows"] == 11
    assert (
        health["calibration_binding"]["fill_policy"]
        == result["timing"]["fill_policy"]
    )
    assert result["timing"]["filled_frames"] == 1


def test_worker_finalizes_only_after_the_silence_window(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from localflight import live

    output = tmp_path / "live"
    output.mkdir()
    marker = output / "auto-finish.json"
    clock = [1_000_000_000]
    accept = Mapper.accept

    def advance(mapper, row):
        clock[0] = row["raw"]["receive_monotonic_ns"]
        return accept(mapper, row)

    def sleep(_seconds):
        if clock[0] == 19_000_000_000:
            assert not marker.exists()
            clock[0] = 26_999_999_000
        elif clock[0] == 26_999_999_000:
            assert not marker.exists()
            clock[0] = 27_000_000_000
        else:
            assert (
                json.loads(marker.read_text())["confirmed_at_us"] == 26_000_000
            )
            (output / "finish").touch()

    monkeypatch.setattr(Mapper, "accept", advance)
    monkeypatch.setattr(
        live,
        "time",
        SimpleNamespace(
            monotonic_ns=lambda: clock[0],
            monotonic=lambda: clock[0] / 1e9,
            time=lambda: clock[0] / 1e9,
            sleep=sleep,
        ),
    )
    rows = list(samples(15))
    for seconds in (17, 18, 19):
        rows.append(
            next(samples(start_ns=seconds * 1_000_000_000, armed=False))
        )
    journal = tmp_path / "journal.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    live.run(
        dict(
            request(),
            auto_finish_disarm=True,
            output=str(output),
            data_root=str(tmp_path),
            journal=str(journal),
        )
    )
    final = json.loads((output / "state.json").read_text())
    assert final["status"] == "finished"
    assert final["finish_reason"] == "confirmed_disarm"
    assert final["result"]["run_summary"]["duration_us"] == 26_000_000
    assert final["result"]["run_summary"]["excluded_us"] >= 10_000_000
