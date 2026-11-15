"""Regress ground-history leakage and slow status resetting signal filters."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_localflight_timing import events

from capstone_ai.mapping import FEATURES
from localflight import calibration, operating
from localflight.prepare import CausalSampler, causal_frames
from localflight.prepared import distinct_flights, windows


def observations():
    rows = []
    for available, sample, update in events(seconds=40):
        update = {k: v for k, v in update.items() if k not in operating.FIELDS}
        # A smooth physical change exposes unwanted repeated LPF resets.
        update["pack_voltage"] = 16 - available / 1e8
        rows.append((available, sample, update))
        if available % 1_000_000 == 0:
            state = 1 if available < 5_000_000 else 2
            if 25_000_000 <= available < 28_000_000:
                state = 4
            rows.append(
                (available, sample, dict(arming_state=2, landed_state=state))
            )
    return rows


def test_status_cadence_preserves_physical_filter_values():
    source = observations()
    reference = [e for e in source if not operating.FIELDS.intersection(e[2])]
    actual, times, report = causal_frames(source, 0, 40_000_000)
    expected, _, _ = causal_frames(reference, 0, 40_000_000)
    np.testing.assert_array_equal(actual, expected)
    assert len(times) == 400 and report["boundary_count"] == 0


def test_ground_and_landing_never_enter_history_or_target():
    values, times, report = causal_frames(
        observations(), 0, 40_000_000, require_airborne=True
    )
    spans = report["candidate_segments"]
    assert [(s["start_us"], s["end_us"]) for s in spans] == [
        (5_000_000, 25_000_000),
        (28_000_000, 40_000_000),
    ]
    assert {s["reason"] for s in report["excluded_intervals"]} == {
        "not_airborne",
        "takeoff_or_landing",
    }
    assert report["operating_excluded_frames"] == 80
    assert report["missing_or_stale_frames"] == 0
    assert np.isfinite(values).all()  # No fabricated or clipped signal values.
    for span in spans:
        a, b = span["start_index"], span["end_index"]
        meta = dict(profile=dict(series_cells=4, capacity_ah=12))
        scaler = dict(
            x_mean=np.zeros(19),
            x_scale=np.ones(19),
            y_mean=np.zeros(4),
            y_scale=np.ones(4),
        )
        _, _, _, origins = windows(meta, values[a:b], times[a:b], scaler)
        assert origins[0] == span["start_us"] + 9_900_000
        assert origins[-1] + 1_000_000 < span["end_us"]


def test_chunked_sampler_matches_offline_with_stale_state_and_reset():
    source = observations()
    source = [
        e
        for e in source
        if not (
            12_000_000 <= e[0] < 17_000_000
            and operating.FIELDS.intersection(e[2])
        )
    ]
    source.append((30_050_000, 30_050_000, {"__reset__": True}))
    source.sort(key=lambda e: e[0])
    values, times, report = causal_frames(
        source, 0, 40_000_000, require_airborne=True
    )
    assert any(
        r["reason"] == "flight_state_unknown"
        for r in report["excluded_intervals"]
    )
    sampler = CausalSampler(require_airborne=True)
    chunks, stamps = [], []
    for start, end in [
        (0, 13_000_000),
        (13_000_000, 30_500_000),
        (30_500_000, 40_000_000),
    ]:
        v, t, invalid, _ = sampler.sample(source, start, end)
        chunks.extend(v)
        stamps.extend(t)
        if end == 30_500_000:
            assert sampler.segment_start_us is None
            assert "flight_state_unknown" in invalid[-1]
    np.testing.assert_array_equal(chunks, values)
    np.testing.assert_array_equal(stamps, times)


def test_old_calibration_binding_and_mixed_roles_rejected(tmp_path):
    scaler = tmp_path / "scaler.npz"
    scaler.write_bytes(b"test-only scaler hash")
    runtime = SimpleNamespace(
        entry=dict(
            id="fixture", weight_sha256="a" * 64, files={"scaler.npz": scaler}
        )
    )
    old = dict(
        profile={}, origin="synthetic_unit_fixture", source_sha256="a" * 64
    )
    new = dict(
        old,
        source_sha256="b" * 64,
        timing=dict(operating_policy=operating.POLICY),
    )
    previous, current = [calibration.binding(runtime, m) for m in (old, new)]
    assert "operating_policy" not in previous
    assert current == dict(previous, operating_policy=operating.POLICY)
    with pytest.raises(ValueError, match="flight-state"):
        distinct_flights([old, new])


def test_unknown_is_not_inferred_from_valid_features():
    source = [
        e for e in observations() if not operating.FIELDS.intersection(e[2])
    ]
    _, _, report = causal_frames(source, 0, 40_000_000, require_airborne=True)
    assert report["complete_frames"] == 0 and not report["candidate_segments"]
    assert report["excluded_intervals"][0]["reason"] == "flight_state_unknown"
    assert not set(report["missing_by_feature"]) - set(FEATURES)


def test_airborne_large_residual_inputs_are_retained():
    source = observations()
    changed = [
        (a, s, dict(v, pack_voltage=10.0, body_rate_p=4.0))
        if 18_000_000 <= a < 22_000_000 and "pack_voltage" in v
        else (a, s, v)
        for a, s, v in source
    ]
    _, _, before = causal_frames(source, 0, 40_000_000, require_airborne=True)
    values, times, after = causal_frames(
        changed, 0, 40_000_000, require_airborne=True
    )
    assert before["candidate_segments"] == after["candidate_segments"]
    assert before["excluded_intervals"] == after["excluded_intervals"]
    row = values[np.searchsorted(times, 20_000_000)]
    assert row[FEATURES.index("pack_voltage")] < 11
    assert row[FEATURES.index("body_rate_p")] > 3.9
