"""Saved-prediction sensitivity checks; no model/GPU execution."""

import copy

import numpy as np
import pytest

from learning.sensitivity_metrics import assess_sensitivity, validate_policy


def policy(**changes):
    return {
        "schema": "drone-ai.sensitivity-policy/1",
        "batch_size": 256,
        "require_same_batch_exact": True,
        "max_scaled_abs": 1e-4,
        "physical_max": [1e-5, 1e-5, 1e-5, 1e-4],
        "max_score_delta_over_high": 0.01,
        "max_ewma_delta_over_high": 0.01,
        "max_changed_alarm_states": 0,
        **changes,
    }


def inputs(count=24):
    return dict(
        reference_batch1=np.zeros((count, 10, 4), dtype=np.float64),
        batch256=np.zeros((count, 10, 4), dtype=np.float64),
        repeat256=np.zeros((count, 10, 4), dtype=np.float64),
        actual_scaled=np.zeros((count, 10, 4), dtype=np.float64),
        identities={
            "run_id": np.array(["normal"] * count),
            "segment": np.zeros(count, dtype=int),
            "origin_us": 60_000_000 + np.arange(count, dtype=np.int64) * 1_000_000,
        },
        diagnostic_calibration={
            "reference_error_scale_scaled": np.ones((10, 4)).tolist(),
            "high": [1.0, 1.0],
            "source": "separate normal-only fixture",
        },
        physical_scale4=[1.0, 1.0, 1.0, 4.0],
        policy=policy(),
    )


def test_known_alarm_entry_and_exit_trajectories_are_preserved():
    data = inputs()
    data["actual_scaled"][:3, :, :3] = 2.0
    snapshot = copy.deepcopy(data)
    report, arrays = assess_sensitivity(**data)
    assert report["status"] == "PASSED"
    assert report["reference_union_episodes"] == report["batch_union_episodes"] == 1
    assert np.flatnonzero(arrays["reference_union_start"]).tolist() == [2]
    # EWMA falls below0.7 at index7; ten consecutive low values close at16.
    assert np.flatnonzero(arrays["reference_union_end"]).tolist() == [16]
    assert np.flatnonzero(arrays["reference_union_active"]).tolist() == list(range(2, 16))
    assert np.flatnonzero(arrays["reference_domain_end"][:, 0]).tolist() == [16]
    np.testing.assert_array_equal(arrays["reference_domain_active"][:, 1], False)
    np.testing.assert_array_equal(arrays["reference_scores"][:3], [[4.0, 0.0]] * 3)
    assert report["changed_alarm_origins"] == 0
    assert report["threshold_changed_between_paths"] is False
    assert report["event_truth_used"] is False
    assert len(report["policy_sha256"]) == len(report["diagnostic_calibration_sha256"]) == 64
    for key in ("reference_batch1", "batch256", "repeat256", "actual_scaled"):
        np.testing.assert_array_equal(data[key], snapshot[key])
    assert data["policy"] == snapshot["policy"]


def test_tiny_prediction_delta_can_fail_alarm_equivalence_with_all_numeric_budgets_passing():
    data = inputs(5)
    data["actual_scaled"][:, :, :3] = 1.0
    data["batch256"][:, :, :3] = -1e-6
    data["repeat256"] = data["batch256"].copy()
    report, arrays = assess_sensitivity(**data)
    assert report["status"] == "FAILED_STOPPED"
    assert report["constraints"]["scaled_absolute"]
    assert report["constraints"]["physical_absolute"]
    assert report["constraints"]["score_delta"]
    assert report["constraints"]["ewma_delta"]
    assert not report["constraints"]["alarm_states"]
    assert report["changed_alarm_origins"] == 3
    assert report["changed_alarm_fields"]["union_start"] == 1
    assert not arrays["reference_union_active"].any()
    assert arrays["batch_union_start"].tolist() == [False, False, True, False, False]


def test_each_scope_resets_without_inventing_a_cross_run_episode_end():
    data = inputs(6)
    data["actual_scaled"][:3, :, :3] = 2.0
    data["identities"]["run_id"] = np.array(["first"] * 3 + ["second"] * 3)
    data["identities"]["origin_us"] = np.array([1, 2, 3, 1, 2, 3]) * 1_000_000
    report, arrays = assess_sensitivity(**data)
    assert report["status"] == "PASSED"
    assert arrays["reference_union_active"].tolist() == [False, False, True, False, False, False]
    assert not arrays["reference_union_end"].any()
    assert not arrays["reference_domain_end"].any()


@pytest.mark.parametrize("change", ["value", "dtype", "negative_zero"])
def test_identical_batch_requires_exact_repeat_bits_and_type(change):
    data = inputs()
    if change == "value":
        data["repeat256"][0, 0, 0] = 1e-12
    elif change == "dtype":
        data["repeat256"] = data["repeat256"].astype(np.float32)
    else:
        data["repeat256"][0, 0, 0] = -0.0
    report, _ = assess_sensitivity(**data)
    assert report["status"] == "FAILED_STOPPED"
    assert not report["constraints"]["same_batch_exact"]


@pytest.mark.parametrize("key", ["reference_batch1", "batch256", "repeat256"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_prediction_returns_failure_and_exact_evidence(key, value):
    data = inputs()
    data[key][0, 0, 0] = value
    report, arrays = assess_sensitivity(**data)
    assert report["status"] == "FAILED_STOPPED"
    assert not report["constraints"]["finite_predictions"]
    assert "reference_scores" not in arrays
    assert not np.isfinite(arrays[key][0, 0, 0])


def test_physical_cap_is_applied_in_pack_volts_without_reapplying_cell_factor():
    data = inputs()
    data["batch256"][:, :, 3] = 2.6e-5
    data["repeat256"] = data["batch256"].copy()
    report, _ = assess_sensitivity(**data)
    assert report["constraints"]["scaled_absolute"]
    assert not report["constraints"]["physical_absolute"]
    assert report["per_target_max_physical"][-1] == pytest.approx(0.000104)
    assert report["status"] == "FAILED_STOPPED"


def test_score_delta_is_relative_to_fixed_threshold_not_reference_score():
    data = inputs()
    data["actual_scaled"][:] = 10.0
    data["batch256"][:] = 1e-6
    data["repeat256"] = data["batch256"].copy()
    data["diagnostic_calibration"]["reference_error_scale_scaled"] = np.full((10, 4), 1e-8).tolist()
    data["diagnostic_calibration"]["high"] = [100.0, 200.0]
    report, arrays = assess_sensitivity(**data)
    expected = abs(100.0 - (10.0 - 1e-6) ** 2) / 1e-8
    assert report["per_domain_max_score_delta_over_high"] == pytest.approx(
        [expected / 100, expected / 200]
    )
    assert not report["constraints"]["score_delta"]
    assert not report["constraints"]["ewma_delta"]
    np.testing.assert_array_equal(arrays["diagnostic_high"], [100.0, 200.0])


def test_overflow_in_derived_scores_is_failure_without_alarm_execution():
    data = inputs()
    data["reference_batch1"][:] = 1e308
    report, arrays = assess_sensitivity(**data)
    assert report["status"] == "FAILED_STOPPED"
    assert not report["constraints"]["finite_derived_scores"]
    assert "reference_scores" not in arrays


@pytest.mark.parametrize(
    "change",
    [
        "gap",
        "duplicate",
        "reverse",
        "float_origin",
        "negative_segment",
        "reappearing_scope",
        "empty_id",
    ],
)
def test_identity_integrity_blocks_discontinuous_or_ambiguous_sequences(change):
    data = inputs(6)
    ids = data["identities"]
    if change == "gap":
        ids["origin_us"][3:] += 1_000_000
    elif change == "duplicate":
        ids["origin_us"][3] = ids["origin_us"][2]
    elif change == "reverse":
        ids["origin_us"] = ids["origin_us"][::-1]
    elif change == "float_origin":
        ids["origin_us"] = ids["origin_us"].astype(float)
    elif change == "negative_segment":
        ids["segment"][0] = -1
    elif change == "reappearing_scope":
        ids["run_id"] = np.array(["a", "a", "b", "b", "a", "a"])
    else:
        ids["run_id"][0] = ""
    with pytest.raises(ValueError):
        assess_sensitivity(**data)


@pytest.mark.parametrize(
    "change",
    [
        {"model_overrides": {}},
        {"batch_size": 1},
        {"batch_size": True},
        {"require_same_batch_exact": False},
        {"max_changed_alarm_states": 1},
        {"max_changed_alarm_states": False},
        {"max_scaled_abs": float("nan")},
        {"max_score_delta_over_high": float("inf")},
        {"max_ewma_delta_over_high": -1},
        {"physical_max": [1.0, 1.0, 1.0]},
        {"physical_max": [True, 1.0, 1.0, 1.0]},
    ],
)
def test_common_policy_rejects_implicit_or_model_specific_exceptions(change):
    with pytest.raises(ValueError):
        validate_policy(policy(**change))


@pytest.mark.parametrize(
    "change",
    ["high_zero", "scale_zero", "scale_nan", "physical_negative", "actual_nan", "shape", "complex"],
)
def test_malformed_calibration_or_data_raises_before_any_admission(change):
    data = inputs()
    if change == "high_zero":
        data["diagnostic_calibration"]["high"][0] = 0
    elif change == "scale_zero":
        data["diagnostic_calibration"]["reference_error_scale_scaled"][0][0] = 0
    elif change == "scale_nan":
        data["diagnostic_calibration"]["reference_error_scale_scaled"][0][0] = float("nan")
    elif change == "physical_negative":
        data["physical_scale4"][0] = -1
    elif change == "actual_nan":
        data["actual_scaled"][0, 0, 0] = float("nan")
    elif change == "shape":
        data["batch256"] = data["batch256"][:-1]
    else:
        data["reference_batch1"] = data["reference_batch1"].astype(complex)
    with pytest.raises(ValueError):
        assess_sensitivity(**data)
