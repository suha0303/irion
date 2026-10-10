import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "campaign_accounting", Path(__file__).resolve().parents[2] / "Data/generation/tools/summarize_independent_campaign.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_failed_attempt_and_retry_both_remain_in_denominator():
    lock = {"cases": [{"case_id": "one", "scenario": "battery_resistance_step"},
                       {"case_id": "two", "scenario": "propulsion_loss_step"}]}
    attempts = [{"case_id": "one", "run_id": "failed"}, {"case_id": "one", "run_id": "retry"}]
    event = {"detected": True, "insufficient_data": False, "new_episode_detected": True,
             "preexisting_alarm": False, "onset_kind": "in_flight_step", "event_id": "fault",
             "pre_onset_valid_decisions": 20}
    comparison = {"lstm": {"per_run": [{"run_id": "retry", "events": [event],
                                       "metrics": {"scored_seconds": 60, "operational_seconds": 100}}]}}
    result = module.account(lock, attempts, comparison)["models"]["lstm"]
    assert result["intended_abnormal_attempt_slots"] == 3
    assert result["unavailable_abnormal_slots"] == 2
    assert result["detection_rate_all_attempt_slots"] == 1 / 3
    assert result["inflight_pre_onset_requirement_failures"] == ["fault"]
    assert len(result["ledger"]) == 3
    assert result["normal_poisson_zero_event_95_upper_per_hour"] is None


def test_miss_is_not_unavailable_and_duplicates_rejected():
    lock = {"cases": [{"case_id": "one", "scenario": "propulsion_loss_existing"}]}
    attempts = [{"case_id": "one", "run_id": "miss"}]
    event = {"detected": False, "insufficient_data": False, "new_episode_detected": False,
             "preexisting_alarm": False, "onset_kind": "pre_existing"}
    comparison = {"lstm": {"per_run": [{"run_id": "miss", "events": [event],
                                       "metrics": {"scored_seconds": 60, "operational_seconds": 100}}]}}
    result = module.account(lock, attempts, comparison)["models"]["lstm"]
    assert result["missed_with_opportunity"] == 1
    assert result["unavailable_abnormal_slots"] == 0
    with pytest.raises(ValueError, match="duplicate attempt"):
        module.account(lock, attempts + attempts, comparison)
    with pytest.raises(ValueError, match="unregistered"):
        module.account(lock, [], comparison)
