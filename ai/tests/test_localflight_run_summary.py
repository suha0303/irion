import json

import numpy as np
import pytest
from test_localflight_adapt import flight
from test_localflight_live import request
from test_localflight_timing import events

from localflight.analysis import analyze, analyze_run
from localflight.catalog import DEFAULT_BUNDLE, digest
from localflight.live import completed
from localflight.prepare import causal_frames
from localflight.run_summary import aggregate, partition, source_context


def fixture(root, name="summary"):
    manifest = flight(root, name, "evaluation")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    x, t, r = causal_frames(
        events(seconds=40, gap=(15_000_000, 20_000_000)), 0, 40_000_000
    )
    chosen = [s for s in r["candidate_segments"] if s["usable"]]
    m["segments"] = []
    arrays = []
    stamps = []
    cursor = 0
    for s in chosen:
        a, b = s["start_index"], s["end_index"]
        arrays.append(x[a:b])
        stamps.append(t[a:b])
        m["segments"].append(dict(start_index=cursor, end_index=cursor + b - a))
        cursor += b - a
    np.savez(
        manifest.parent / "frames.npz",
        values=np.concatenate(arrays),
        time_us=np.concatenate(stamps),
    )
    m["file"]["sha256"] = digest(manifest.parent / "frames.npz")
    m["timing"] = r
    m["run_context"] = dict(
        start_us=0,
        end_us=50_000_000,
        extent_known=True,
        clock="onboard_boot",
        notes=[],
    )
    m["interval"] = dict(start_us=0, end_us=40_000_000)
    manifest.write_text(json.dumps(m), encoding="utf-8")
    return manifest


@pytest.mark.parametrize(
    "model", ["forecast_lstm_h64_l2", "run_health_lstm_h64_l2"]
)
def test_run_summary_covers_gaps_and_source_tail_and_never_invents_normal(
    tmp_path, model
):
    manifest = fixture(tmp_path)
    result = analyze_run([model], [str(manifest)], bundle=DEFAULT_BUNDLE)
    s = result["run_summary"]
    a = result["analyses"][0]
    assert s["status"] == "unavailable" and s["partial"]
    assert len(a["segment_results"]) == 2
    assert {r["reason"] for r in s["timeline"]} == {
        "analyzed",
        "missing_or_stale",
        "outside_selection",
    }
    assert s["analyzed_us"] + s["excluded_us"] == 50_000_000
    assert s["coverage"] == s["analyzed_us"] / 50_000_000
    assert sum(r["end_us"] - r["start_us"] for r in s["timeline"]) == 50_000_000
    assert all(
        x["end_us"] == y["start_us"]
        for x, y in zip(s["timeline"], s["timeline"][1:])
    )
    if model.startswith("forecast"):
        np.testing.assert_allclose(s["models"][0]["rmse"], a["metrics"]["rmse"])
        assert all(v["metrics"]["windows"] > 0 for v in a["segment_results"])
    else:
        assert s["models"][0]["samples"] == sum(
            v["metrics"]["samples"] for v in a["segment_results"]
        )
        assert s["models"][0]["rmse"][0] > 0


def test_ab_does_not_double_coverage_and_duplicate_packages_rejected(tmp_path):
    path = fixture(tmp_path)
    result = analyze_run(
        ["forecast_lstm_h64_l2", "forecast_transformer_h64_l2"],
        [str(path)],
        bundle=DEFAULT_BUNDLE,
    )
    assert len(result["run_summary"]["models"]) == 2
    assert result["run_summary"]["analyzed_us"] < 40_000_000
    with pytest.raises(ValueError, match="distinct"):
        analyze_run(
            ["forecast_lstm_h64_l2"],
            [str(path), str(path)],
            bundle=DEFAULT_BUNDLE,
        )
    clone = path.with_name("clone.json")
    clone.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="overlap"):
        analyze_run(
            ["forecast_lstm_h64_l2"],
            [str(path), str(clone)],
            bundle=DEFAULT_BUNDLE,
        )
    other = fixture(tmp_path, "other")
    with pytest.raises(ValueError, match="One source"):
        analyze_run(
            ["forecast_lstm_h64_l2"],
            [str(path), str(other)],
            bundle=DEFAULT_BUNDLE,
        )


def test_summary_decision_preserves_partial_and_unavailable(tmp_path):
    a = analyze(
        "forecast_lstm_h64_l2", fixture(tmp_path), bundle=DEFAULT_BUNDLE
    )
    for segment in a["segment_results"]:
        segment["decision"] = dict(
            status="normal", observations=5, exceedances=0
        )
    assert aggregate([a])["status"] == "incomplete"
    a["segment_results"][0]["decision"] = dict(
        status="suspect", observations=5, exceedances=1
    )
    s = aggregate([a])
    assert s["status"] == "suspect" and s["partial"]
    assert (
        s["models"][0]["observations"] == 10
        and s["models"][0]["exceedances"] == 1
    )
    a["segment_results"][0]["decision"] = dict(
        status="unavailable", reason="missing calibration"
    )
    assert aggregate([a])["status"] == "unavailable"


def test_live_no_usable_data_has_complete_exclusion_summary(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text("", encoding="utf-8")
    for length, reason in [
        (5_000_000, "too_short"),
        (30_000_000, "missing_or_stale"),
    ]:
        result = completed(
            [],
            journal,
            tmp_path / str(length),
            request(),
            "synthetic_unit_fixture",
            end_us=length,
        )
        s = result["run_summary"]
        assert (
            result["status"] == "insufficient_data"
            and s["status"] == "unavailable"
        )
        assert (
            s["analyzed_us"] == 0
            and s["excluded_us"] == length
            and s["coverage"] == 0
        )
        assert len(s["timeline"]) == 1 and s["timeline"][0]["reason"] == reason


def test_legacy_and_clock_reset_never_claim_whole_run_coverage(tmp_path):
    m = flight(tmp_path, "old", "evaluation")
    result = analyze_run(
        ["forecast_lstm_h64_l2"], [str(m)], bundle=DEFAULT_BUNDLE
    )
    assert result["run_summary"]["coverage"] is None
    context = source_context(
        {
            "timing": {
                "streams": {
                    "IMU": dict(
                        reversals=1, spans=[dict(start_us=0, end_us=100)]
                    )
                }
            }
        },
        0,
        20_000_000,
    )
    assert context["extent_known"] is False


def test_partition_short_unselected_and_bounded_durations():
    c = dict(start_us=0, end_us=100)
    rows = partition(
        c,
        [dict(start_us=0, end_us=20)],
        [
            dict(start_us=20, end_us=30, reason="too_short"),
            dict(start_us=30, end_us=50, reason="not_selected"),
        ],
    )
    assert [(r["start_us"], r["end_us"], r["reason"]) for r in rows] == [
        (0, 20, "analyzed"),
        (20, 30, "too_short"),
        (30, 50, "not_selected"),
        (50, 100, "outside_selection"),
    ]


def test_separate_packages_match_one_segmented_package(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from localflight import analysis
    from localflight.prepared import load, segment_ranges

    original = fixture(tmp_path)
    m, values, times = load(original)
    paths = []
    for i, (a, b) in enumerate(segment_ranges(m, len(values))):
        folder = tmp_path / f"part{i}"
        folder.mkdir()
        np.savez(folder / "frames.npz", values=values[a:b], time_us=times[a:b])
        part = json.loads(json.dumps(m))
        part["segments"] = [dict(start_index=0, end_index=b - a)]
        part["file"]["sha256"] = digest(folder / "frames.npz")
        p = folder / "manifest.json"
        p.write_text(json.dumps(part), encoding="utf-8")
        paths.append(str(p))
    models = ["forecast_lstm_h64_l2", "run_health_lstm_h64_l2"]
    one = analyze_run(models, [str(original)], bundle=DEFAULT_BUNDLE)[
        "run_summary"
    ]
    checked_load = Mock(wraps=analysis.load)
    checked_runtime = Mock(wraps=analysis.load_runtime)
    monkeypatch.setattr(analysis, "load", checked_load)
    monkeypatch.setattr(analysis, "load_runtime", checked_runtime)
    grouped = analyze_run(models, list(reversed(paths)), bundle=DEFAULT_BUNDLE)
    assert checked_load.call_count == len(paths)
    assert checked_runtime.call_count == len(models)
    starts = [a["segment_results"][0]["start_us"] for a in grouped["analyses"]]
    assert starts == sorted(starts)
    joined = grouped["run_summary"]
    assert (
        one["analyzed_us"] == joined["analyzed_us"]
        and one["timeline"] == joined["timeline"]
    )
    for a, b in zip(one["models"], joined["models"]):
        np.testing.assert_allclose(a["rmse"], b["rmse"], rtol=1e-6)
    p = tmp_path / "part1/manifest.json"
    part = json.loads(p.read_text(encoding="utf-8"))
    part["timing"]["clock_corrections"] = {"test": "different"}
    p.write_text(json.dumps(part), encoding="utf-8")
    with pytest.raises(ValueError, match="same clock"):
        analyze_run(models, paths, bundle=DEFAULT_BUNDLE)


def test_inspection_outside_recorded_time_does_not_expand_whole_run(tmp_path):
    audit = {
        "timing": {
            "streams": {
                "IMU": dict(
                    reversals=0,
                    spans=[dict(start_us=2_000_000, end_us=20_000_000)],
                )
            }
        }
    }
    context = source_context(audit, 0, 100_000_000)
    assert context["start_us"] == 2_000_000 and context["end_us"] == 20_100_000
    with pytest.raises(ValueError, match="distinct models"):
        analyze_run(
            ["forecast_lstm_h64_l2"] * 2,
            [str(fixture(tmp_path))],
            bundle=DEFAULT_BUNDLE,
        )
