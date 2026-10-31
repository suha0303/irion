import copy

import numpy as np
import pytest

from capstone_ai.detection import DetectionLedger, DomainAlarms, residual_evidence
from capstone_ai.detection_protocol import (
    choose_threshold,
    evaluation_metrics,
    event_result,
    fit_calibration,
    grouped_rate_interval,
    point_metrics,
)


def observed(ledger, origin, delay=0, values=None):
    for j in range(1, 11):
        t = origin + j * 100000
        ledger.observe("r", "c", 0, t, t + delay, [2, 2, 2, 2] if values is None else values)


def test_analytic_contributions_and_scale_unit_invariance():
    a = residual_evidence(np.zeros((10, 4)), np.tile([1, 2, 3, -4], (10, 1)), np.ones((10, 4)))
    assert a["scores"] == pytest.approx([14 / 3, 16])
    assert a["signed_residual"][0] == [1, 2, 3, -4]
    assert a["absolute_error"][0] == [1, 2, 3, 4]
    units = np.array([.1, .2, .3, 4])
    b = residual_evidence(np.zeros((10, 4)), np.tile([1, 2, 3, -4], (10, 1)) * units,
                          np.ones((10, 4)) * units**2)
    assert a["scores"] == pytest.approx(b["scores"], abs=1e-12)


def test_origin_order_cannot_be_overtaken_by_later_arrivals():
    ledger = DetectionLedger(np.ones((10, 4)), [1, 1])
    first = ledger.issue("r", "c", 0, 0, 1, np.zeros((10, 4)))
    second = ledger.issue("r", "c", 0, 1000000, 1000001, np.zeros((10, 4)))
    observed(ledger, 0, 2000000)
    observed(ledger, 1000000)
    ledger.advance(2000000)
    assert not ledger.decisions
    snapshot = ledger.snapshot()
    ledger.advance(3000000)
    snapshot.advance(3000000)
    assert list(ledger.decisions) == [first, second]
    assert ledger.decisions == snapshot.decisions
    assert all(d["decision_time_us"] == 3000000 for d in ledger.decisions.values())


def test_invalid_missing_late_and_unarrived_coverage_is_not_normal():
    for values, expected in [([np.nan, 2, 2, 2], "nonfinite_or_malformed_target"),
                             ([2, 2], "nonfinite_or_malformed_target")]:
        ledger = DetectionLedger(np.ones((10, 4)), [1, 1])
        key = ledger.issue("r", "c", 0, 0, 1, np.zeros((10, 4)))
        observed(ledger, 0, values=values)
        ledger.advance(1000000)
        assert ledger.decisions[key]["scores"] is None
        assert expected in ledger.decisions[key]["unscored_reasons"]
    ledger = DetectionLedger(np.ones((10, 4)), [1, 1])
    key = ledger.issue("r", "c", 0, 0, 100000, np.zeros((10, 4)))
    observed(ledger, 0, 2000000)
    ledger.advance(1000000, closed=True)
    result = ledger.decisions[key]
    assert result["scores"] is None and result["coverage"] == 0
    assert result["state"] == "forecast_late"
    assert result["last_target_available_us"] is None
    before = copy.deepcopy(result)
    ledger.advance(4000000)
    assert before == ledger.decisions[key]


def test_domain_union_gap_hysteresis_and_segment_reset():
    alarms = DomainAlarms([1, 1])
    for i in range(3):
        result = alarms.update([2, 0], i * 1000000)
    assert result["episode_start"] and result["domains"]["propulsion"]["alarm_active"]
    assert not result["domains"]["battery"]["alarm_active"]
    gap = alarms.update(None, 3000000)
    assert gap["alarm_active"] and gap["state"] == "insufficient_data"
    for i in range(9):
        assert alarms.update([0, 0], (4 + i) * 1000000)["alarm_active"]
    assert alarms.update([0, 0], 13000000)["episode_end"]
    ledger = DetectionLedger(np.ones((10, 4)), [1, 1])
    ledger.issue("r", "c", 0, 0, 1, np.zeros((10, 4)))
    with pytest.raises(ValueError, match="close pending"):
        ledger.issue("r", "c", 1, 1000000, 1000001, np.zeros((10, 4)))
    ledger.advance(1000000, closed=True)
    ledger.issue("r", "c", 1, 1000000, 1000001, np.zeros((10, 4)))
    assert not any(a.active for a in ledger.domain_alarm.alarms)


def decision(t, active=False, start=False, valid=True):
    return {"run_id": "r", "clock_id": "c", "segment_id": 0, "origin_time_us": t - 1000000,
            "decision_time_us": t, "scores": [2, 0] if valid else None,
            "alarm_active": active, "episode_start": start,
            "unscored_reasons": [] if valid else ["targets_missing_or_not_yet_available"],
            "domains": {n: {"alarm_active": active, "episode_start": start}
                        for n in ["propulsion", "battery"]}}


def test_events_preexisting_alarm_new_recall_and_zero_opportunity():
    ds = [decision(1000000, True, True), decision(2000000, True), decision(3000000, True)]
    event = {"event_id": "a", "onset_kind": "in_flight_step", "physical_onset_time_us": 1500000}
    e = event_result(event, ds, 0)
    assert e["preexisting_alarm"] and e["detected"] and not e["new_episode_detected"]
    assert e["onset_delay_s"] == .5
    event.update(onset_kind="pre_existing", physical_onset_time_us=0)
    e = event_result(event, ds, 0)
    assert e["onset_delay_s"] is None and e["recognition_from_first_valid_s"] == 0
    missing = event_result({"event_id": "b", "onset_kind": "pre_existing"}, [], 0)
    metrics = evaluation_metrics([e, missing], ds, 10)
    assert metrics["recall_all"] == .5 and metrics["recall_conditional"] == 1
    assert metrics["events_insufficient_data"] == 1 and metrics["coverage"] == .3
    assert metrics["episodes"]["union"][0]["censored"]


def test_ramp_boundary_end_and_group_bootstrap():
    e = event_result({"event_id": "ramp", "onset_kind": "in_flight_ramp",
                      "applied_time_us": 1000000, "degradation_boundary_time_us": 3000000,
                      "end_time_us": 5000000}, [decision(2000000, True, True),
                                               decision(4000000, True)], 0)
    assert e["onset_delay_s"] == 1 and e["boundary_delay_s"] == 1
    no = event_result({"event_id": "short", "end_time_us": 1000000}, [decision(2000000, True)], 0)
    assert no["insufficient_data"] and not no["detected"]
    interval = grouped_rate_interval([{"group": "paired", "episodes": 1, "hours": 1},
                                      {"group": "paired", "episodes": 0, "hours": 1}])
    assert interval["groups"] == 1 and interval["interval"] is None


def test_normal_only_calibration_floor_and_zero_exposure_rejection():
    result = fit_calibration(np.zeros((4, 10, 4)), np.zeros((8, 10, 4)),
                             ["a"] * 4 + ["b"] * 4, [0] * 8, list(range(0, 4000000, 1000000)) * 2,
                             [1, 1, 1, 4], {"a": "g1", "b": "g2"})
    assert result["chosen"]["q"] == .95
    assert result["status"] == "provisional"
    assert result["error_scale_physical"][0] == [1e-8, 1e-8, 1e-8, 16e-8]
    assert result["poisson_zero_event_95_upper_per_hour"] > 1000
    with pytest.raises(ValueError, match="nonempty"):
        fit_calibration(np.zeros((0, 10, 4)), np.zeros((8, 10, 4)), [], [], [], [], {})


def test_nonfinite_forecast_scale_and_duplicate_exposure_rejected():
    with pytest.raises(ValueError, match="positive"):
        DetectionLedger(np.zeros((10, 4)), [1, 1])
    ledger = DetectionLedger(np.ones((10, 4)), [1, 1])
    with pytest.raises(ValueError, match="nonfinite"):
        ledger.issue("r", "c", 0, 0, 0, np.full((10, 4), np.inf))
    d = decision(1000000)
    with pytest.raises(ValueError, match="duplicate"):
        evaluation_metrics([], [d, d], 10)
    with pytest.raises(ValueError, match="overlapping"):
        evaluation_metrics([], [d, {**d, "origin_time_us": 500000}], 10)


def test_holdout_rejects_old_raw_retry_pair_and_allows_new_pair():
    from capstone_ai.detection_replay import require_new_holdout

    prior = [{"run_id": "old", "paired_case_id": "p", "previous_attempts": ["retry"],
              "raw_sha256": "digest"}]
    for fields in [{"parent_attempt_id": "retry"}, {"paired_case_id": "p"},
                   {"raw_sha256": "digest"}, {"previous_attempts": ["old"]}]:
        with pytest.raises(ValueError, match="previously exposed"):
            require_new_holdout([{"run_id": "new", **fields}], prior)
    require_new_holdout([{"run_id": "a", "paired_case_id": "fresh"},
                         {"run_id": "b", "paired_case_id": "fresh"}], prior)


def test_truth_join_preserves_sealed_forecasts_and_refuses_tamper(tmp_path):
    import json

    from capstone_ai.detection_replay import evaluate_sealed
    from capstone_ai.source import dump, sha

    forecast = tmp_path / "forecasts.jsonl"
    forecast.write_text('{"fixture": "immutable prediction"}\n')
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(decision(1000000, True, True)) + "\n")
    dump(tmp_path / "seal.json", {"files": {p.name: sha(p) for p in [forecast, decisions]}})
    dump(tmp_path / "summary.json", {"operational_seconds": 10})
    events = tmp_path / "truth.jsonl"
    events.write_text('{"event_id":"a","onset_kind":"pre_existing"}\n')
    original = sha(forecast)
    a = evaluate_sealed(tmp_path, events, 0)
    events.write_text('{"event_id":"b","onset_kind":"in_flight_step","applied_time_us":2000000}\n')
    b = evaluate_sealed(tmp_path, events, 0)
    assert a["metrics"]["events_detected"] == 1 and b["metrics"]["events_detected"] == 0
    assert sha(forecast) == original
    decisions.write_text(decisions.read_text() + "\n")
    with pytest.raises(ValueError, match="sealed"):
        evaluate_sealed(tmp_path, events, 0)


def test_raw_point_metrics_no_adjustment_and_clipped_exposure():
    events = [{"onset_kind": "in_flight_step", "applied_time_us": 2000000, "end_time_us": 4000000}]
    ds = [decision(1000000, True, True), decision(2000000), decision(3000000, True),
          decision(4000000, valid=False)]
    result = point_metrics(events, ds, 0)
    assert [result[n] for n in ["tp", "fp", "fn", "unavailable"]] == [1, 1, 1, 1]
    assert result["f1"] == .5 and not result["point_adjustment"]
    clipped = evaluation_metrics([], ds, 3.5, {"c": [.5e6, 4e6]})
    assert clipped["scored_seconds"] == 2.5


def test_most_sensitive_threshold_or_best_unmet_is_not_infinite():
    candidates = [{"q": .95, "false_alarms_per_hour": 3},
                  {"q": .99, "false_alarms_per_hour": 2},
                  {"q": .999, "false_alarms_per_hour": 2.5}]
    assert choose_threshold(candidates)["q"] == .99
    candidates[1]["false_alarms_per_hour"] = 1
    candidates[2]["false_alarms_per_hour"] = 0
    assert choose_threshold(candidates)["q"] == .99


def test_deadline_override_and_backdated_decision_rejected():
    ledger = DetectionLedger(np.ones((10, 4)), [1, 1])
    with pytest.raises(ValueError, match="strict deadline"):
        ledger.issue("r", "c", 0, 0, 100000, np.zeros((10, 4)), "awaiting_targets")
    ledger.issue("r", "c", 0, 0, 1, np.zeros((10, 4)))
    with pytest.raises(ValueError, match="precedes forecast"):
        ledger.advance(0, closed=True)
