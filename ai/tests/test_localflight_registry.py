"""Check versioned model registration, activation and aircraft binding."""

import json
import zipfile

import numpy as np
import pytest
from test_localflight_adapt import flight

from localflight import registry
from localflight.adapt import fit
from localflight.analysis import analyze, compare
from localflight.catalog import DEFAULT_BUNDLE


@pytest.mark.parametrize(
    "model_id", ["forecast_lstm_h64_l2", "run_health_lstm_h64_l2"]
)
def test_model_backup_restore_select_and_rollback(tmp_path, model_id):
    train = flight(tmp_path, "train", "adaptation")
    evaluation = flight(tmp_path, "evaluation", "evaluation")
    fitted = tmp_path / "fitted"
    fit(model_id, [train], [], fitted, bundle=DEFAULT_BUNDLE, epochs=1)
    model = registry.register(
        tmp_path / "registry", fitted, bundle=DEFAULT_BUNDLE
    )
    assert (
        registry.register(tmp_path / "registry", fitted, bundle=DEFAULT_BUNDLE)[
            "id"
        ]
        == model["id"]
    )
    archive = tmp_path / "model.zip"
    registry.export_model(
        tmp_path / "registry", model["id"], archive, bundle=DEFAULT_BUNDLE
    )
    imported = registry.import_model(
        tmp_path / "restored", archive, bundle=DEFAULT_BUNDLE
    )
    assert imported == model
    assert (
        registry.list_models(tmp_path / "restored", bundle=DEFAULT_BUNDLE)[
            "selection"
        ]["active"]
        == {}
    )
    folder = registry.model_path(tmp_path / "restored", model["id"])
    receipt = compare(folder, evaluation, bundle=DEFAULT_BUNDLE)
    with pytest.raises(ValueError, match="independent"):
        registry.select_model(
            tmp_path / "restored", model["id"], {}, bundle=DEFAULT_BUNDLE
        )
    event = registry.select_model(
        tmp_path / "restored", model["id"], receipt, bundle=DEFAULT_BUNDLE
    )
    assert (
        registry.active_folder(
            tmp_path / "restored",
            model["profile"],
            model["base_model_id"],
            bundle=DEFAULT_BUNDLE,
        )
        == folder
    )
    registry.rollback(tmp_path / "restored", event["id"], bundle=DEFAULT_BUNDLE)
    assert (
        registry.active_folder(
            tmp_path / "restored",
            model["profile"],
            model["base_model_id"],
            bundle=DEFAULT_BUNDLE,
        )
        is None
    )
    with pytest.raises(ValueError):
        registry.rollback(
            tmp_path / "restored", event["id"], bundle=DEFAULT_BUNDLE
        )
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {*registry.FILES, "package.json"}
        assert "output" not in json.loads(z.read("result.json"))


def test_import_rejects_traversal_and_tamper(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../escape", "x")
    with pytest.raises(ValueError, match="entries"):
        registry.import_model(tmp_path / "models", path, bundle=DEFAULT_BUNDLE)
    assert not (tmp_path / "escape").exists()


def test_forecast_series_has_correct_units_horizon_and_residual(tmp_path):
    path = flight(tmp_path, "evaluation", "evaluation")
    result = analyze("forecast_lstm_h64_l2", path, bundle=DEFAULT_BUNDLE)
    series = result["metrics"]["series"][0]
    assert np.all(
        np.asarray(series["time_us"]) - np.asarray(series["origin_us"])
        == 1000000
    )
    np.testing.assert_allclose(
        np.asarray(series["observed"]) - np.asarray(series["predicted"]),
        series["residual"],
        rtol=1e-4,
        atol=2e-6,
    )
    assert (
        min(np.asarray(series["observed"])[:, 3]) > 15
    )  # Pack volts, not cell volts.
    assert result["elapsed_seconds"] > 0 and result["normal_alarm_rate"] is None
