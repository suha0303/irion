"""Verify diagnostic units, bias decomposition and causal target alignment."""

import json

import numpy as np
import pytest
from test_localflight_adapt import flight

from localflight.prepared import load
from localflight.voltage_diagnostics import (
    _package,
    diagnose,
    error_metrics,
    inputs,
)


def scaler():
    return dict(
        x_mean=np.zeros(19),
        x_scale=np.ones(19),
        y_mean=np.zeros(4),
        y_scale=np.ones(4),
    )


def test_nameplate_conversion_is_once_and_preserves_raw_inputs():
    values = np.zeros((3, 19))
    values[:, 17], values[:, 18] = 48, 22
    before = values.copy()
    physical, normalized, stats = inputs(
        values, dict(series_cells=12, capacity_ah=44), scaler()
    )
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(physical[:, 17], [4, 4, 4])
    np.testing.assert_array_equal(normalized[:, 18], [0.5, 0.5, 0.5])
    assert stats[17]["median"] == 4
    assert stats[18]["abs_z_p95"] == 0.5


def test_bias_and_centered_error_reconstruct_total_without_correction():
    result = error_metrics(np.array([-3.0, 1.0]), np.zeros(2), 12)
    assert result["bias_predicted_minus_observed_pack_v"] == -12
    assert result["centered_rmse_pack_v"] == 24
    assert result["rmse_pack_v"] ** 2 == pytest.approx(720)


class ConstantCellPredictor:
    entry = dict(task="forecast")

    def __init__(self):
        self.scaler = scaler()

    def predict_scaled(self, values):
        prediction = np.zeros((len(values), 10, 4), dtype="f4")
        prediction[:, :, 3] = 4
        return prediction


def test_complete_target_window_metrics_match_independent_physical_values(
    tmp_path,
):
    path = flight(tmp_path, "voltage", "evaluation", "unknown")
    metadata, values, times = load(path)
    before = values.copy()
    result = _package(ConstantCellPredictor(), metadata, values, times)
    # Fixture voltage is 16 - frame_index * 0.0001 V. The first target is
    # frame 100 (not origin 99), and every emitted window has ten targets.
    target_indices = np.array(
        [list(range(i + 1, i + 11)) for i in range(99, 240, 10)]
    )
    expected_error = target_indices * 0.0001
    assert result["full_error"]["samples"] == expected_error.size
    assert result["full_error"]["rmse_pack_v"] == pytest.approx(
        np.sqrt(np.mean(expected_error**2)), abs=1e-6
    )
    assert result["target_reconstruction_max_abs_cell_v"] < 1e-6
    assert result["baseline"]["rmse_pack_v"] == pytest.approx(
        np.sqrt(np.mean((np.arange(1, 11) * 0.0001) ** 2)), abs=1e-6
    )
    np.testing.assert_array_equal(values, before)
    assert json.loads(path.read_text(encoding="utf-8"))["health"] == "unknown"


def test_diagnostics_reject_duplicate_packages_before_model_execution(
    tmp_path,
):
    path = flight(tmp_path, "duplicate", "evaluation")
    with pytest.raises(ValueError, match="Duplicate"):
        diagnose([path, path])
