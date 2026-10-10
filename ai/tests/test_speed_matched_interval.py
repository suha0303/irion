"""Different startup lengths must not alter fixed mission comparison; real gaps must fail."""

import copy
import json

import pytest

from pretrain import common, speed_metrics, speed_test


def fixture(tmp_path, monkeypatch, name, start_seconds, anchor_seconds, missing_raw=False):
    monkeypatch.setattr(speed_metrics, "ROOT", tmp_path)
    rid = name
    raw = tmp_path / "runs" / rid
    anchor = round(anchor_seconds * 1e6)
    common.write(
        raw / "evaluation/execution.json",
        [
            dict(state="mission", sim_time=anchor_seconds),
            dict(state="landing", sim_time=anchor_seconds + 118),
        ],
    )
    stream = raw / "observations/streams.jsonl"
    stream.parent.mkdir(parents=True)
    with stream.open("w") as f:
        for t in range(start_seconds * 1_000_000, 152_000_001, 100_000):
            if missing_raw and anchor + 30_000_000 <= t < anchor + 30_100_000:
                continue
            f.write(
                json.dumps(
                    dict(
                        run_id=rid,
                        clock_id="px4-boot",
                        sample_time_us=t,
                        available_time_us=t,
                        topic="battery_status",
                        field_quality=dict(voltage="valid"),
                    )
                )
                + "\n"
            )
    canonical = tmp_path / "canonical" / rid
    canonical.mkdir(parents=True)
    names = ["body_rate_p", "body_rate_q", "body_rate_r", "pack_voltage"]
    frames = [
        dict(
            run_id=rid,
            clock_id="px4-boot",
            segment_id=1,
            time_us=t,
            operating_state="airborne",
            quality=dict.fromkeys(names, "valid"),
            values=dict.fromkeys(names, 1.0),
        )
        for t in range(start_seconds * 1_000_000, 152_000_001, 100_000)
    ]
    (canonical / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    windows = []
    for t in range((start_seconds + 60) * 1_000_000, 151_000_000, 1_000_000):
        windows.append(
            dict(
                clock_id="px4-boot",
                segment_id=1,
                origin_us=t,
                context_start_us=t - 60_000_000,
                history_times_us=list(range(t - 9_900_000, t + 1, 100_000)),
                target_times_us=list(range(t + 100_000, t + 1_000_001, 100_000)),
            )
        )
    (canonical / "windows.jsonl").write_text("".join(json.dumps(w) + "\n" for w in windows))
    members = {
        n: dict(sha256=common.sha(raw / n))
        for n in ["evaluation/execution.json", "observations/streams.jsonl"]
    }
    common.write(
        canonical / "summary.json",
        dict(
            run_id=rid,
            closed=True,
            feature_order=names,
            frames=len(frames),
            windows=len(windows),
            input_hashes={
                "observations/streams.jsonl": members["observations/streams.jsonl"]["sha256"]
            },
            config=dict(context_us=60_000_000),
            output_hashes={
                n: common.sha(canonical / (n + ".jsonl")) for n in ["frames", "windows"]
            },
        ),
    )
    record = dict(
        run_id=rid,
        export=dict(members=members),
        assessment=dict(topic_quality=dict(battery_status={}), effective_rtf=2),
    )
    return speed_metrics.metrics(canonical, record)


def test_startup_and_anchor_phase_do_not_change_mission_window_count(tmp_path, monkeypatch):
    a = fixture(tmp_path, monkeypatch, "a", 1, 10.020)
    b = fixture(tmp_path, monkeypatch, "b", 7, 12.028)
    assert a["whole_run_diagnostic"]["windows"] != b["whole_run_diagnostic"]["windows"]
    assert a["windows"] == b["windows"] == 50
    assert (
        a["topic_quality"]["battery_status"]["count"]
        == b["topic_quality"]["battery_status"]["count"]
        == 900
    )
    assert speed_test.compare(a, a, b)["passed"]


def test_real_gap_inside_fixed_interval_is_not_hidden(tmp_path, monkeypatch):
    a = fixture(tmp_path, monkeypatch, "a", 1, 10.0)
    b = fixture(tmp_path, monkeypatch, "b", 7, 12.0, missing_raw=True)
    result = speed_test.compare(a, a, b)
    assert not result["passed"]
    assert not result["topic_quality"]["battery_status"]["gap_passed"]
    changed = copy.deepcopy(b)
    changed["windows"] = 46
    assert not speed_test.compare(a, a, changed)["windows_passed"]


def test_incomplete_topic_or_mixed_protocol_never_passes(tmp_path, monkeypatch):
    a = fixture(tmp_path, monkeypatch, "a", 1, 10.0)
    b = copy.deepcopy(a)
    b["topic_quality"]["battery_status"]["covered"] = False
    assert not speed_test.compare(a, a, b)["passed"]
    b["protocol"] = "information-v1"
    with pytest.raises(ValueError, match="contract changed"):
        speed_test.compare(a, a, b)
