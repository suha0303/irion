"""Check model catalog integrity, CPU inputs and aircraft profile
eligibility.
"""

import copy

import numpy as np
import pytest

from localflight.catalog import MODEL_IDS, Catalog
from localflight.profiles import eligibility, list_profiles, save_profile
from localflight.runtime import Runtime


def profile():
    return dict(
        schema="drone.real-aircraft/1",
        id="unknown-quad",
        name="기체 미확정",
        autopilot="ardupilot",
        rotor_count=None,
        series_cells=None,
        capacity_ah=None,
        motor_channels={},
        notes="정상 여부 모름",
    )


def test_unknown_profile_cannot_become_eligible(tmp_path):
    p = profile()
    result = save_profile(tmp_path, p)
    assert not result["eligibility"]["profile_complete"]
    assert not result["eligibility"]["real_aircraft_validated"]
    save_profile(tmp_path, p)
    assert len(list_profiles(tmp_path)) == 1
    p["series_cells"] = 12
    revised = save_profile(tmp_path, p)
    assert revised["version"] != result["version"]
    assert len(list_profiles(tmp_path)) == 2


def test_profile_rejects_duplicate_motors_and_nonfinite():
    p = profile()
    p["motor_channels"] = {"front_right": 1, "front_left": 1}
    with pytest.raises(ValueError, match="distinct"):
        eligibility(p)
    p = profile()
    p["capacity_ah"] = float("nan")
    with pytest.raises(ValueError, match="capacity"):
        eligibility(p)


def test_catalog_exact_set_and_tamper_detection():
    catalog = Catalog()
    assert {r["id"] for r in catalog.list()} == set(MODEL_IDS)
    with pytest.raises(ValueError, match="Unknown"):
        catalog.entry("../../model")
    catalog.manifest = copy.deepcopy(catalog.manifest)
    catalog.manifest["files"][
        "models/forecast_lstm_h64_l2/model.safetensors"
    ] = "0" * 64
    with pytest.raises(ValueError, match="integrity"):
        catalog.entry("forecast_lstm_h64_l2")


@pytest.mark.parametrize(
    "model_id", ["forecast_lstm_h64_l2", "forecast_transformer_h64_l2"]
)
def test_runtime_cpu_restore_and_input_gate(model_id):
    runtime = Runtime(model_id)
    a = np.zeros((1, 100, 19), dtype="f4")
    out = runtime.predict_scaled(a)
    restored = Runtime(model_id).predict_scaled(a)
    np.testing.assert_array_equal(out, restored)
    assert out.shape == (1, 10, 4)
    assert all(p.device.type == "cpu" for p in runtime.model.parameters())
    a[0, 1, 2] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        runtime.predict_scaled(a)


def test_run_health_requires_complete_unpadded_run():
    runtime = Runtime("run_health_lstm_h64_l2")
    a = np.zeros((1, 101, 19), dtype="f4")
    assert runtime.predict_scaled(a, np.array([101], dtype="i8")).shape == (
        1,
        101,
    )
    with pytest.raises(ValueError, match="Exact"):
        runtime.predict_scaled(a, np.array([100], dtype="i8"))
