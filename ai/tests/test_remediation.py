"""Historical dataset and asset fixtures; never starts collection or training."""

import copy
import json

import pytest

from capstone_ai.dataset import validate_split, verify_input
from capstone_ai.evaluation import flight_bootstrap, match_event
from capstone_ai.source import dump, sha


def test_actual_new_selection_is_group_independent():
    rows = json.load(open("ai/configs/remediation/dataset.json"))["runs"]
    assert len(validate_split(rows)) == 28
    assert {r["split"] for r in rows if r["seed"] == 601} == {"test"}
    assert {r["scenario"] for r in rows if r["split"] == "train"} == {
        r["scenario"] for r in rows if r["normal_diversity_member"]
    }
    bad = copy.deepcopy(rows)
    bad[1]["previous_attempts"] = [bad[0]["run_id"]]
    with pytest.raises(ValueError, match="leakage"):
        validate_split(bad)


def test_integrity_tamper_and_escape(tmp_path):
    names = [
        "run.json",
        "observations/metadata.json",
        "observations/streams.jsonl",
        "observations/watermarks.jsonl",
        "evaluation/scenario.json",
    ]
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    integrity = {n: {"size": 2, "sha256": sha(tmp_path / n)} for n in names}
    dump(tmp_path / "integrity.json", integrity)
    verify_input(tmp_path, {})
    (tmp_path / "run.json").write_text("[]")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_input(tmp_path, {})
    dump(tmp_path / "integrity.json", {"../outside": {"size": 0, "sha256": ""}})
    with pytest.raises(ValueError, match="escapes"):
        verify_input(tmp_path, {})


def decision(t, active, start=False):
    return {
        "scores": [1, 1],
        "decision_time_us": t,
        "state": "anomalous" if active else "ready",
        "alarm_active": active,
        "episode_start": start,
    }


def test_event_matching_preexisting_pre_alarm_and_end():
    event = {"event_id": "e", "onset_kind": "pre_existing", "physical_onset_time_us": None}
    result = match_event(event, [decision(60000000, True, True)], 5000000)
    assert result["physical_onset_delay_s"] is None
    assert result["recognition_from_observation_start_s"] == 55
    assert result["recognition_from_first_valid_s"] == 0
    event.update(onset_kind="in_flight_step", physical_onset_time_us=2000000, end_time_us=5000000)
    decisions = [
        decision(1000000, True, True),
        decision(2000000, True),
        decision(3000000, False),
        decision(4000000, True, True),
    ]
    result = match_event(event, decisions, 0)
    assert result["alarm_active_before_onset"]
    assert result["detection_time_us"] == 4000000
    event["end_time_us"] = 3500000
    assert match_event(event, decisions, 0)["detection_time_us"] is None


def test_bootstrap_identical_exposure():
    assert flight_bootstrap([1, 1], [1.0, 1.0]) == [1.0, 1.0]
    assert flight_bootstrap([0], [1.0]) is None


def test_asset_replacement_and_missing_conditions():
    from capstone_ai.trend import comparison_gate

    records = json.load(open("simulation/artifacts/remediation-20261014/asset-fixtures/asset-registry.json"))[
        "records"
    ]
    a = comparison_gate(records[0], records[1])
    b = comparison_gate(records[1], records[2])
    assert not a["comparable"] and not b["comparable"]
    assert {"asset_replacement_or_identity_change", "independent_reset"} <= set(a["reasons"])
    assert "insufficient_comparable_conditions" in b["reasons"]
    assert "independent_reset" in b["reasons"]
    continuous = copy.deepcopy(records[0])
    continuous.update(
        continuity="same_asset_future",
        bundle_sha256="b",
        start_time_utc="2026-10-01T01:00:00+00:00",
        end_time_utc="2026-10-01T02:00:00+00:00",
    )
    after = {**continuous, "run_id": "next", "start_time_utc": "2026-10-02T01:00:00+00:00"}
    assert comparison_gate(continuous, after)["comparable"]
    after["bundle_sha256"] = "changed"
    assert "bundle_revision_changed" in comparison_gate(continuous, after)["reasons"]


def test_deadline_bracket_strict_boundary():
    from capstone_ai.clock_probe import deadline_bounds

    a = {"run_id": "r", "host_receive_ns": 10, "sim_time_us": 90, "nonce": 1}
    b = {"run_id": "r", "host_send_ns": 30, "sim_time_us": 99, "nonce": 2}
    assert deadline_bounds(a, b, 20, 100)["status"] == "on_time"
    b["sim_time_us"] = 100
    assert deadline_bounds(a, b, 20, 100)["status"] == "unverified"
    a["sim_time_us"] = 100
    assert deadline_bounds(a, b, 20, 100)["status"] == "late"
    b["host_send_ns"] = 19
    with pytest.raises(ValueError, match="brackets"):
        deadline_bounds(a, b, 20, 100)
    assert deadline_bounds(None, None, 20, 100)["status"] == "unverified"


def test_live_jitter_keeps_origin_cadence_and_unverified_no_score():
    import numpy as np

    from capstone_ai.inference import PendingForecasts

    p = PendingForecasts(np.ones((10, 4)), [1, 1])
    for i, jitter in enumerate([100, 700, 1000]):
        origin = i * 1000000
        k = p.issue("r", "c", 0, origin, origin, np.zeros((10, 4)))
        for j in range(10):
            t = origin + (j + 1) * 100000
            p.observe("r", "c", 0, t, t, [2, 2, 2, 2])
        p.advance(origin + 1000000 + jitter)
    assert p.decisions[k]["episode_start"]
    k = p.issue("r", "c", 0, 3000000, None, np.zeros((10, 4)))
    p.advance(5000000, closed=True)
    assert (
        p.decisions[k]["scores"] is None and p.decisions[k]["state"] == "forecast_timing_unverified"
    )
