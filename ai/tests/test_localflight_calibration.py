import json

import numpy as np
import pytest
from test_localflight_adapt import flight

from localflight import calibration
from localflight.adapt import fit as adapt
from localflight.analysis import analyze
from localflight.catalog import DEFAULT_BUNDLE, digest


def make_flight(root, name, role, *, clock=None):
    path = flight(root, name, role)
    m = json.loads(path.read_text())
    with np.load(path.parent / "frames.npz") as data:
        values = np.tile(data["values"], (2, 1))
    np.savez(path.parent / "frames.npz", values=values, time_us=np.arange(500, dtype="i8") * 100000)
    m["file"]["sha256"] = digest(path.parent / "frames.npz")
    if clock:
        m["timing"] = dict(clock=clock)
    path.write_text(json.dumps(m))
    return path


@pytest.mark.parametrize("model", ["forecast_lstm_h64_l2", "run_health_transformer_h64_l2"])
def test_calibration_roundtrip_decision_binding_and_independent_rates(tmp_path, model):
    a = make_flight(tmp_path, "cal-a", "calibration")
    b = make_flight(tmp_path, "cal-b", "calibration")
    evaluation = make_flight(tmp_path, "evaluation", "evaluation")
    record = calibration.fit(tmp_path, model, [a, b], bundle=DEFAULT_BUNDLE)
    assert not record["active"] and record["normal_flights"] == 2
    assert not record["false_alarm_guarantee"]
    calibration.select(tmp_path, record["id"])
    result = analyze(model, evaluation, bundle=DEFAULT_BUNDLE, data_root=tmp_path)
    assert result["decision"]["status"] == "normal"
    assert result["normal_alarm_rate"] == 0 and not result["real_aircraft_validated"]
    bound = calibration.active(tmp_path, result["calibration_binding"])
    t = record["threshold"]
    assert calibration.decide(bound, [t])["status"] == "normal"
    assert calibration.decide(bound, [t + 1])["status"] == "suspect"
    assert calibration.decide(bound, [float("nan")])["status"] == "unavailable"
    assert (
        analyze(model, a, bundle=DEFAULT_BUNDLE, data_root=tmp_path)["decision"]["status"]
        == "unavailable"
    )
    changed = dict(result["calibration_binding"], weight_sha256="0" * 64)
    assert calibration.active(tmp_path, changed) is None
    live = dict(result["calibration_binding"], clock="receipt_monotonic")
    assert calibration.active(tmp_path, live) is None
    calibration.select(tmp_path, record["id"], False)
    assert (
        analyze(model, evaluation, bundle=DEFAULT_BUNDLE, data_root=tmp_path)["health_verdict"]
        is None
    )
    p = tmp_path / "calibrations" / (record["id"] + ".json")
    value = json.loads(p.read_text())
    value["threshold"] = 1e20
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity"):
        calibration.select(tmp_path, record["id"])


def test_calibration_rejects_leakage_unknown_roles_and_changed_clock(tmp_path):
    a = make_flight(tmp_path, "a", "calibration")
    b = make_flight(tmp_path, "b", "calibration")
    with pytest.raises(ValueError, match="same flight"):
        calibration.fit(tmp_path, "forecast_lstm_h64_l2", [a, a], bundle=DEFAULT_BUNDLE)
    m = json.loads(b.read_text())
    m["health"] = "unknown"
    b.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Confirmed normal"):
        calibration.fit(tmp_path, "forecast_lstm_h64_l2", [a, b], bundle=DEFAULT_BUNDLE)
    m["health"] = "confirmed_normal"
    m["timing"] = {"clock": "receipt_monotonic"}
    b.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="identical clock"):
        calibration.fit(tmp_path, "forecast_lstm_h64_l2", [a, b], bundle=DEFAULT_BUNDLE)
    train = make_flight(tmp_path, "train", "adaptation")
    adapt("forecast_lstm_h64_l2", [train], [], tmp_path / "fitted", bundle=DEFAULT_BUNDLE, epochs=1)
    m = json.loads(train.read_text())
    m["role"] = "calibration"
    train.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="재사용"):
        calibration.fit(
            tmp_path,
            "forecast_lstm_h64_l2",
            [train, a],
            bundle=DEFAULT_BUNDLE,
            adapted=tmp_path / "fitted",
        )


def test_live_decision_waits_for_actual_target_and_rejects_gap_reset():
    from localflight.live import arrived_decision

    pending = [dict(origin=10000000, due=11000000, reset=0, scaled=[0, 0, 0, 4])]
    values = np.zeros((111, 19))
    values[:, 17] = 16
    times = np.arange(111) * 100000
    scaler = dict(y_mean=np.zeros(4), y_scale=np.ones(4))
    record = dict(
        id="fixture",
        source_sha256=[],
        threshold=0.5,
        unit="fixture",
        validation_status="experimental_normal_only",
    )
    future, decision = arrived_decision(pending, values[:-1], times[:-1], 0, 0, 4, scaler, record)
    assert future == pending and decision is None
    future, decision = arrived_decision(pending, values, times, 0, 0, 4, scaler, record)
    assert not future and decision["status"] == "normal"
    values[-1, 4] = 1
    assert (
        arrived_decision(pending, values, times, 0, 0, 4, scaler, record)[1]["status"] == "suspect"
    )
    assert arrived_decision(pending, values, times, 10100000, 0, 4, scaler, record)[1] is None
    assert arrived_decision(pending, values, times, 0, 1, 4, scaler, record)[1] is None
    assert arrived_decision(pending, values, times + 1, 0, 0, 4, scaler, record)[1] is None
