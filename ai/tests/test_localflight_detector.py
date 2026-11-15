"""Frozen-score parity, streaming filters, source splits and event accounting."""

import numpy as np
import pytest
from test_localflight_adapt import flight
from test_localflight_calibration import make_flight
from test_localflight_live import request, samples

from capstone_ai import detection_protocol
from localflight import calibration, detector, normal_plan
from localflight.adapt_review import _event_detection
from localflight.analysis import analyze
from localflight.catalog import DEFAULT_BUNDLE
from localflight.live import Mapper, arrived_decision
from localflight.prepare import CausalSampler, causal_frames


def evidence():
    residual = np.zeros((30, 10, 4))
    residual[3:20, :, 3] = 3
    return dict(
        residual_scaled=residual,
        origin_us=np.arange(30) * 1000000,
        segment_id=[0] * 30,
        time_us=(np.arange(30) + 1) * 1000000,
    )


def parameters():
    return dict(
        reference_error_scale_scaled=np.ones((10, 4)).tolist(), high=[1, 1]
    )


def test_full_lead_score_and_stream_match_frozen_postprocessor():
    data, params = evidence(), parameters()
    scores = detector.forecast_scores(data, params)
    expected = detection_protocol.score_sequence(
        scores,
        ["flight"] * 30,
        data["segment_id"],
        data["origin_us"],
        params["high"],
    )
    offline = detector.decide("forecast", data, params)
    stream = detector.LiveDetector(dict(parameters=params, id="fixture"))
    live = [
        stream.update(residual, int(t), 0)
        for residual, t in zip(data["residual_scaled"], data["origin_us"])
    ]
    assert offline["alarms"] == [row["alarm_active"] for row in expected]
    assert [r["alarms"][0] for r in live] == offline["alarms"]
    np.testing.assert_allclose(
        [r["scores"][0] for r in live], offline["scores"], rtol=0, atol=0
    )
    # An early lead alone must contribute; the legacy one-second-only score misses it.
    residual = np.zeros((1, 10, 4))
    residual[0, 0, 3] = 10
    assert (
        detector.forecast_scores(dict(residual_scaled=residual), params)[0, 1]
        == 10
    )


def test_gaps_and_scope_resets_match_frozen_alarm():
    data, params = evidence(), parameters()
    data["origin_us"][15:] += 3000000
    data["segment_id"][22:] = [1] * 8
    expected = detector.decide("forecast", data, params)
    stream = detector.LiveDetector(dict(parameters=params, id="fixture"))
    live = [
        stream.update(r, int(t), s)
        for r, t, s in zip(
            data["residual_scaled"], data["origin_us"], data["segment_id"]
        )
    ]
    assert expected["alarms"] == [r["alarms"][0] for r in live]
    with pytest.raises(ValueError, match="duplicate"):
        stream.update(
            data["residual_scaled"][-1], int(data["origin_us"][-1]), 1
        )


def test_live_float32_and_saved_json_residuals_have_identical_scores():
    residual = np.linspace(-0.7, 0.9, 1200, dtype="f4").reshape(30, 10, 4)
    live = detector.forecast_scores(dict(residual_scaled=residual), parameters())
    saved = detector.forecast_scores(
        dict(residual_scaled=residual.tolist()), parameters()
    )
    np.testing.assert_array_equal(live, saved)


def test_preexisting_alarm_is_not_a_new_fault_detection():
    data = evidence()
    result = dict(model_id="forecast_fixture", scoring=data)
    record = dict(
        schema=detector.POLICY, task="forecast", parameters=parameters()
    )
    event = dict(start_us=12000000, end_us=20000000, target="pack_voltage")
    assert not _event_detection(result, record, event, 1)["detected"]
    event["start_us"] = 3000000
    assert _event_detection(result, record, event, 1)["detected"]
    event["target"] = "propulsion"
    assert not _event_detection(result, record, event, 1)["detected"]


def test_chunked_filter_is_exactly_offline_with_gaps_and_reset():
    mapper = Mapper(request())
    events = []
    for item in samples(40):
        events.extend(mapper.accept(item))
    events = [e for e in events if not 14000000 < e[0] < 15000000]
    events.append((25000000, 25000000, {"__reset__": True}))
    events.sort(key=lambda e: e[0])
    expected, times, _ = causal_frames(events, 0, 40000000)
    sampler = CausalSampler()
    values, stamps = [], []
    for end in [11345678, 14999999, 20100321, 25000001, 40000000]:
        chunk, ts, _, _ = sampler.sample(events, sampler.next_us or 0, end)
        values.extend(chunk)
        stamps.extend(ts)
    np.testing.assert_array_equal(times, stamps)
    np.testing.assert_allclose(expected, values, rtol=0, atol=0, equal_nan=True)


def test_stream_waits_for_all_ten_targets_and_uses_same_scores():
    times = np.arange(111) * 100000
    values = np.zeros((111, 19))
    values[:, 17] = 16
    pending = [
        dict(origin=9900000, due=10900000, reset=0, scaled=np.zeros((10, 4)))
    ]
    state = detector.LiveDetector(dict(parameters=parameters(), id="fixture"))
    scaler = dict(y_mean=np.zeros(4), y_scale=np.ones(4))
    _, result = arrived_decision(
        pending, values[:109], times[:109], 0, 0, 4, scaler, None, state
    )
    assert result is None
    _, result = arrived_decision(
        pending, values, times, 0, 0, 4, scaler, None, state
    )
    assert result["domain_scores"] == [0, 16]


@pytest.mark.parametrize(
    "model", ["forecast_transformer_h64_l4", "run_health_transformer_h64_l2"]
)
def test_protocol_binding_rejects_reference_reuse_and_retains_legacy(
    tmp_path, model
):
    reference = flight(tmp_path, "ref", "development")
    cal = [make_flight(tmp_path, f"cal{i}", "calibration") for i in range(2)]
    evaluation = flight(tmp_path, "evaluation", "evaluation")
    old = calibration.fit(tmp_path, model, cal, bundle=DEFAULT_BUNDLE)
    new = calibration.fit_protocol(
        tmp_path, model, [reference], cal, bundle=DEFAULT_BUNDLE
    )
    calibration.select(tmp_path, old["id"])
    calibration.select(tmp_path, new["id"])
    result = analyze(
        model, evaluation, bundle=DEFAULT_BUNDLE, data_root=tmp_path
    )
    assert result["decision"]["detector_policy"] == detector.POLICY
    assert (
        analyze(model, reference, bundle=DEFAULT_BUNDLE, data_root=tmp_path)[
            "decision"
        ]["status"]
        == "unavailable"
    )
    assert calibration.read(tmp_path, old["id"])["schema"] == calibration.POLICY
    calibration.select(tmp_path, old["id"])
    assert calibration.active(tmp_path, result["calibration_binding"])["id"] == old["id"]
    if model.startswith("run_health"):
        assert not new["parameters"]["finite_sample_rank_available"]


def test_normal_pool_is_source_disjoint_and_preserves_original(tmp_path):
    paths = [flight(tmp_path, f"normal{i}", "evaluation") for i in range(20)]
    before = [p.read_bytes() for p in paths]
    plan = normal_plan.create(paths, tmp_path / "plan")
    assert plan["counts"] == dict(
        adaptation=3, development=3, calibration=8, evaluation=6
    )
    assert len({r["source_sha256"] for r in plan["flights"]}) == 20
    assert before == [p.read_bytes() for p in paths]
    assert normal_plan.counts(3) == dict(
        adaptation=1, development=1, calibration=1, evaluation=0
    )
    assert sum(normal_plan.counts(50).values()) == 50
    assert max(normal_plan.counts(50).values()) <= 20
    with pytest.raises(ValueError, match="same flight"):
        normal_plan.create([paths[0], paths[0]], tmp_path / "duplicate")
