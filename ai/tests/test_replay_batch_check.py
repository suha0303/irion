from types import SimpleNamespace

import numpy as np
import pytest

from execution.common import read
from learning.replay_validation import check_prediction_batch


@pytest.mark.parametrize("perturb", [0.0, 0.01, float("nan")])
def test_batch_check_saves_normal_only_result_and_stops_on_difference(tmp_path, perturb):
    calls = []

    def predict(x, c):
        calls.append(len(x))
        values = np.repeat(x[:, :1, :4], 10, axis=1)
        return values + (perturb if len(x) > 1 else 0)

    normal = {"X": np.ones((300, 100, 19)), "C": np.zeros((300, 6))}
    output = tmp_path / "check.json"
    predictor = SimpleNamespace(predict_scaled=predict)
    if perturb == 0:
        result = check_prediction_batch(predictor, normal, output)
        assert result["status"] == "PASSED"
    else:
        with pytest.raises(ValueError, match="numerical check failed"):
            check_prediction_batch(predictor, normal, output)
        assert read(output)["status"] == "FAILED_STOPPED"
    result = read(output)
    assert calls == [1] * 257 + [256, 1]
    assert result["weights_changed"] is False
    assert result["event_truth_used"] is False
    assert result["windows"] == 257
    assert len(result["input_sha256"]) == 64


def test_batch_check_refuses_small_sample(tmp_path):
    with pytest.raises(ValueError, match="full inference batch"):
        check_prediction_batch(None, {"X": np.zeros((256, 100, 19))}, tmp_path / "check.json")
