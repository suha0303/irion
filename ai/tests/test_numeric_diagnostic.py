from types import SimpleNamespace

import numpy as np
import pytest

from execution.common import read
from learning.replay_validation import assess_diagnostic, diagnose_prediction_batch


def normal_sample(count=300):
    return {
        "X": np.zeros((count, 100, 19), dtype=np.float32),
        "C": np.zeros((count, 6), dtype=np.float32),
        "Y": np.ones((count, 10, 4), dtype=np.float32),
        "run_id": np.asarray([f"run-{i // 30}" for i in range(count)]),
        "segment": np.arange(count, dtype=np.int64) // 30,
        "origin_us": np.arange(count, dtype=np.int64) * 100_000,
    }


def fake_predictor(perturb=0.0, drift=0.0):
    calls = []

    def predict(x, c):
        calls.append(len(x))
        value = np.zeros((len(x), 10, 4), dtype=np.float64)
        if len(x) > 1:
            value[:, :, -1] = perturb + drift * sum(n > 1 for n in calls)
        return value

    return SimpleNamespace(predict_scaled=predict, scaler={"y_scale": np.array([1., 2., 3., 5.])}), calls


def policy(**changes):
    return {
        "schema": "drone-ai.numeric-policy/1", "batch_size": 256, "windows": 257,
        "rtol": 1e-5, "atol": 1e-6, "require_same_batch_exact": True, **changes,
    }


def test_diagnostic_saves_untuned_normal_evidence_and_physical_units(tmp_path):
    predictor, calls = fake_predictor(perturb=6.8e-6)
    output = tmp_path / "diagnostic.json"
    report = diagnose_prediction_batch(predictor, normal_sample(), output)
    assert report["status"] == "DIAGNOSTIC_COMPLETE"
    assert report["acceptance_policy_applied"] is False
    assert report["same_batch_repeated_exactly"] is True
    assert calls == [1] * 257 + [256, 1, 256, 1]
    assert report["batch1_vs_batch"]["legacy_check"]["failed_elements"] == 2560
    assert report["batch1_vs_batch"]["per_target_max_physical"] == [0., 0., 0., 6.8e-6 * 20]
    assert report["physical_target_scale"] == [1., 2., 3., 20.]
    assert report["weights_changed"] is False
    assert report["calibration_changed"] is False
    assert report["event_truth_used"] is False
    with np.load(tmp_path / report["arrays"]["file"], allow_pickle=False) as evidence:
        selected = np.linspace(0, 299, 257, dtype=np.int64)
        np.testing.assert_array_equal(evidence["selected_indices"], selected)
        np.testing.assert_array_equal(evidence["sample_y"], normal_sample()["Y"][selected])
        np.testing.assert_array_equal(evidence["run_id"], normal_sample()["run_id"][selected])
        np.testing.assert_array_equal(evidence["origin_us"], normal_sample()["origin_us"][selected])
        np.testing.assert_array_equal(evidence["batch_first"], evidence["batch_repeat"])
    with pytest.raises(FileExistsError, match="already exists"):
        diagnose_prediction_batch(predictor, normal_sample(), output)


def test_diagnostic_exposes_repeat_instability_without_tuning_criteria(tmp_path):
    predictor, _ = fake_predictor(drift=1e-5)
    output = tmp_path / "diagnostic.json"
    report = diagnose_prediction_batch(predictor, normal_sample(), output)
    assert report["status"] == "DIAGNOSTIC_COMPLETE"
    assert report["same_batch_repeated_exactly"] is False
    assessed = tmp_path / "assessment.json"
    with pytest.raises(ValueError, match="Common numerical policy failed"):
        assess_diagnostic(output, policy(atol=0.1), assessed)
    assert read(assessed)["constraints"]["same_batch_repeat"] is False


@pytest.mark.parametrize("perturb", [float("nan"), float("inf")])
def test_nonfinite_prediction_preserves_arrays_and_failure_report(tmp_path, perturb):
    predictor, _ = fake_predictor(perturb=perturb)
    output = tmp_path / "diagnostic.json"
    with pytest.raises(ValueError, match="Nonfinite or malformed"):
        diagnose_prediction_batch(predictor, normal_sample(), output)
    assert read(output)["status"] == "FAILED_STOPPED"
    assert read(output)["finite"] is False
    assert output.with_suffix(".npz").is_file()


def test_frozen_common_policy_acceptance_and_failure_are_recorded(tmp_path):
    predictor, _ = fake_predictor(perturb=6.8e-6)
    output = tmp_path / "diagnostic.json"
    diagnose_prediction_batch(predictor, normal_sample(), output)
    strict = tmp_path / "strict.json"
    with pytest.raises(ValueError, match="Common numerical policy failed"):
        assess_diagnostic(output, policy(), strict)
    assert read(strict)["status"] == "FAILED_STOPPED"
    assert read(strict)["policy"] == policy()
    explicit = policy(atol=1e-5, physical_atol=[1e-4, 1e-4, 1e-4, 2e-4])
    result = assess_diagnostic(output, explicit, tmp_path / "explicit.json")
    assert result["status"] == "PASSED"
    assert result["policy"] == explicit
    assert len(result["policy_sha256"]) == 64
    with pytest.raises(FileExistsError):
        assess_diagnostic(output, explicit, strict)


def test_physical_limit_is_independent_of_scaled_tolerance(tmp_path):
    predictor, _ = fake_predictor(perturb=6.8e-6)
    output = tmp_path / "diagnostic.json"
    diagnose_prediction_batch(predictor, normal_sample(), output)
    assessed = tmp_path / "physical-failure.json"
    with pytest.raises(ValueError, match="Common numerical policy failed"):
        assess_diagnostic(output, policy(atol=1e-3, physical_atol=[1e-4] * 4), assessed)
    assert read(assessed)["constraints"]["batch1_vs_batch"] is True
    assert read(assessed)["constraints"]["physical_max_absolute"] is False


def test_symmetric_common_tolerance_and_absolute_cap(tmp_path):
    predictor, _ = fake_predictor()
    predictor.predict_scaled = lambda x, c: np.full((len(x), 10, 4), 2. if len(x) == 1 else 1.)
    output = tmp_path / "diagnostic.json"
    diagnose_prediction_batch(predictor, normal_sample(), output)
    # Symmetric reference magnitude2 gives tolerance1.2. A batched-only magnitude
    # would incorrectly give0.6 for these exact saved predictions.
    explicit = policy(atol=0., rtol=0.6, max_absolute_difference=1.)
    result = assess_diagnostic(output, explicit)
    assert result["constraints"]["batch1_vs_batch"] is True
    assert result["constraints"]["normalized_max_absolute"] is True
    assessed = tmp_path / "absolute-cap.json"
    with pytest.raises(ValueError, match="Common numerical policy failed"):
        assess_diagnostic(output, policy(**{**explicit, "max_absolute_difference": 0.9}), assessed)
    assert read(assessed)["constraints"]["batch1_vs_batch"] is True
    assert read(assessed)["constraints"]["normalized_max_absolute"] is False


@pytest.mark.parametrize("changes", [
    {"model_overrides": {}}, {"rtol": float("nan")}, {"atol": True},
    {"physical_atol": [1, 2, 3]}, {"physical_atol": [True] * 4},
    {"windows": 256}, {"require_same_batch_exact": "yes"},
    {"max_absolute_difference": float("nan")}, {"max_absolute_difference": -1},
])
def test_policy_rejects_implicit_or_model_specific_criteria(tmp_path, changes):
    with pytest.raises(ValueError):
        assess_diagnostic(tmp_path / "not-read.json", policy(**changes))


def test_saved_arrays_are_hash_verified(tmp_path):
    predictor, _ = fake_predictor()
    output = tmp_path / "diagnostic.json"
    diagnose_prediction_batch(predictor, normal_sample(), output)
    with output.with_suffix(".npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash changed"):
        assess_diagnostic(output, policy())


def test_input_signature_matches_legacy_and_identity_changes_are_detectable(tmp_path):
    from learning.replay_validation import check_prediction_batch

    predictor, _ = fake_predictor()
    normal = normal_sample()
    legacy = check_prediction_batch(predictor, normal, tmp_path / "legacy.json")
    diagnostic = diagnose_prediction_batch(predictor, normal, tmp_path / "diagnostic.json")
    assert legacy["input_sha256"] == diagnostic["input_sha256"]
    normal["origin_us"] += 1
    changed = diagnose_prediction_batch(predictor, normal, tmp_path / "identity-changed.json")
    assert changed["input_sha256"] == diagnostic["input_sha256"]
    assert changed["sample_sha256"] != diagnostic["sample_sha256"]


def test_without_arrays_is_diagnostic_only(tmp_path):
    predictor, _ = fake_predictor()
    output = tmp_path / "diagnostic.json"
    result = diagnose_prediction_batch(predictor, normal_sample(), output, save_arrays=False)
    assert result["arrays"] is None
    assert not output.with_suffix(".npz").exists()
    with pytest.raises(ValueError, match="saved exact"):
        assess_diagnostic(output, policy())
