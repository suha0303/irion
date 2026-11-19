import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from capstone_ai.mapping import FEATURES
from learning.new19.buffered import (
    HOLD_US,
    BufferedPreprocessor,
    Frame,
    Px4Clock,
    flight_frames,
)
from learning.new19.data import (
    STEP_US,
    TRAIN_ORIGIN_INTERVAL_US,
    Run,
    digest,
    forecast_origins,
    load_split,
    origins,
    write,
)
from learning.new19.inference import FlightSession, replay_forecasts
from learning.new19.models import VARIANTS, make_model
from learning.new19.scoring import (
    FlightAlarm,
    FlightReference,
    RunReference,
    run_components,
)
from learning.new19.training import training_batches


def run_fixture(n=125):
    values = np.ones((n, 19), dtype=np.float32)
    values[:, 17] = 15
    return Run(
        "run",
        values,
        np.arange(n) * STEP_US,
        np.zeros(n, dtype=np.int64),
        np.ones((n, 4), dtype=bool),
    )


def test_training_origin_stride_uses_segment_grid_and_real_targets():
    run = run_fixture(270)
    run.segment[140:] = 1
    run.observed[[118, 239]] = False
    train = training_batches([run], "inflight")
    assert train.origin_interval_us == TRAIN_ORIGIN_INTERVAL_US
    assert [i for _, i in train.items] == [99, 119, 129, 240, 250]
    assert 109 in forecast_origins(run) and 109 not in origins(run)
    assert 100 in origins(run)
    x, y, _ = train.batch(range(len(train.items)))
    assert x.shape == (5, 100, 19) and y.shape == (5, 10, 4)


def test_buffer_publishes_sorted_frames_at_ten_seconds_and_masks_held_targets():
    profile = dict(
        autopilot="px4", rotor_count=4, series_cells=4, capacity_ah=12
    )
    processor = BufferedPreprocessor(profile, 1, 1, 0)
    processor.origin_ns = 1_000_000_000
    observations = dict.fromkeys(FEATURES, 0.25)
    observations.update(
        arming_state=2,
        landed_state=2,
        pack_voltage=15,
        climb_target=0,
        rate_target_p=0,
        rate_target_q=0,
        rate_target_r=0,
    )
    processor.push(50_000, {"roll": 0.5})
    processor.push(0, observations)

    def advance(logical_us):
        return processor.advance(processor.origin_ns + logical_us * 1000)

    assert advance(HOLD_US - 1) == []
    frames = advance(HOLD_US + 400_000)
    assert [frame.time_us for frame in frames] == [
        0,
        100_000,
        200_000,
        300_000,
        400_000,
    ]
    assert [frame.values[FEATURES.index("roll")] for frame in frames] == [
        0.25,
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    assert frames[0].observed == (True,) * 4
    assert frames[-1].observed == (False,) * 4
    assert advance(HOLD_US + 1_100_000)[-1].segment == -1


def test_px4_clock_accepts_reordered_boot_time_and_preserves_receipt_fallback():
    clock = Px4Clock()
    assert clock.time(1_000, "LOCAL_POSITION_NED", {"time_boot_ms": 4000}) == (
        1_000,
        False,
        True,
    )
    assert clock.time(150_000, "ATTITUDE", {"time_boot_ms": 4100}) == (
        101_000,
        False,
        True,
    )
    assert clock.time(180_000, "ATTITUDE", {"time_boot_ms": 4050}) == (
        51_000,
        False,
        True,
    )
    assert clock.time(180_000, "BATTERY_STATUS", {}) == (180_000, False, True)
    assert clock.time(
        180_000, "HIGHRES_IMU", {"time_usec": 1_791_234_567_000_000}
    ) == (180_000, False, True)


def test_buffer_rejects_late_packet_after_silence_with_or_without_wall_ticks(
    monkeypatch,
):
    profile = dict(
        autopilot="px4", rotor_count=4, series_cells=4, capacity_ah=12
    )
    for tick_advanced in (False, True):
        processor = BufferedPreprocessor(profile, 1, 1, 0)
        processor.origin_ns = 0
        processor.clock.time(0, "LOCAL_POSITION_NED", {"time_boot_ms": 4000})
        raw = {
            "message_name": "ATTITUDE",
            "fields": {"time_boot_ms": 4100, "roll": 0.75},
        }
        monkeypatch.setattr(
            processor.source, "received", lambda *_: iter((raw,))
        )
        if tick_advanced:
            processor.advance((HOLD_US + 500_000) * 1000)
        frames = processor.accept(
            (HOLD_US + 600_000) * 1000, (b"test", 1), b"payload"
        )
        assert processor.late == 1
        assert all(
            np.isnan(frame.values[FEATURES.index("roll")]) for frame in frames
        )


def test_buffered_flight_includes_takeoff_and_descent_but_stops_at_touchdown():
    profile = dict(
        autopilot="px4", rotor_count=4, series_cells=4, capacity_ah=12
    )
    processor = BufferedPreprocessor(profile, 1, 1, 0)
    processor.origin_ns = 0
    observations = dict.fromkeys(FEATURES, 0.25)
    observations.update(
        arming_state=1,
        landed_state=1,
        pack_voltage=15,
        climb_target=0,
        rate_target_p=0,
        rate_target_q=0,
        rate_target_r=0,
    )
    processor.push(0, observations)
    processor.push(STEP_US, {"arming_state": 2})
    for index, state in enumerate((3, 2, 4, 1, 1), 2):
        processor.push(index * STEP_US, {"landed_state": state})
    frames = processor.advance((HOLD_US + 6 * STEP_US) * 1000)
    assert [frame.segment for frame in frames] == [-1, 0, 0, 0, 0, -1, -1]
    assert frames[-2].grounded and frames[-1].grounded


def test_flight_bounds_keep_interior_gaps_and_all_valid_edge_frames():
    frames = [
        Frame(i * STEP_US, [], segment, (), grounded=i == 7)
        for i, segment in enumerate((-1, 0, 0, -1, 1, 1, -1, -1))
    ]
    chosen, coverage = flight_frames(
        frames, 0, tuple(i * STEP_US * 1000 for i in (0, 2, 5, 8))
    )
    assert chosen == frames[1:6]
    assert (
        coverage["valid_takeoff_frames"]
        == coverage["valid_landing_frames"]
        == 1
    )
    assert coverage["takeoff_missing_edge_frames"] == 1
    assert coverage["landing_missing_edge_frames"] == 1
    assert coverage["grounded_tail_frames"] == 1


@pytest.mark.parametrize("parallel", (False, True))
def test_buffered_live_journal_matches_prepared_flight_edges(
    tmp_path, monkeypatch, parallel
):
    from learning.new19 import app_live

    frames = [
        Frame(i * STEP_US, [], segment, ())
        for i, segment in enumerate((-1, 0, 0, -1, 1, 1, -1, -1))
    ]
    selected, _ = flight_frames(
        frames, 0, tuple(i * STEP_US * 1000 for i in (0, 2, 5, 8))
    )
    inflight = [
        name for name, variant in VARIANTS.items() if variant.task == "inflight"
    ]
    binding = {"frame_policy": app_live.FRAME_POLICY}
    if parallel:
        binding["models"] = {name: {} for name in inflight}
    else:
        binding["model"] = "test"
    source = SimpleNamespace(
        accept=lambda *_: frames[:5],
        advance=lambda *_: frames[5:],
        last_receipt_ns=0,
        late=0,
    )

    def commit(frame):
        row = dict(
            origin_us=frame.time_us, forecast=None, decision={"alarm": None}
        )
        return row, None

    monkeypatch.setattr(app_live, "validate", lambda _: binding)
    monkeypatch.setattr(app_live, "processor", lambda _: source)
    monkeypatch.setattr(app_live, "datagrams", lambda _: [(0, None, b"")])
    monkeypatch.setattr(
        app_live, "Detector", lambda *_: SimpleNamespace(commit=commit)
    )
    request = dict(
        wire=tmp_path / "wire", binding=binding, scope="simulation_test"
    )
    if parallel:
        request.update(inflight_root=str(tmp_path))
    else:
        request.update(
            new19_bundle=str(tmp_path), new19_calibration=str(tmp_path)
        )
    result = app_live.analyze(request, tmp_path, replay=True)
    journal = [
        json.loads(row)
        for row in (tmp_path / "buffered-results.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["origin_us"] for row in journal] == [
        frame.time_us for frame in selected
    ]
    assert result["frames"] == len(selected)
    if parallel:
        assert list(result["model_summaries"]) == inflight
        assert all(list(row["models"]) == inflight for row in journal)


def test_eight_exact_models_and_whole_voltage_blindness():
    torch.set_num_threads(2)
    assert len(VARIANTS) == 8
    x = torch.randn(2, 181, 19, generator=torch.Generator().manual_seed(5))
    lengths = torch.tensor([151, 181])
    for name, variant in VARIANTS.items():
        model = make_model(name).eval()
        assert (
            len(
                (
                    model.encoder if variant.task == "inflight" else model.local
                ).layers
            )
            == variant.layers
        )
        with torch.no_grad():
            if variant.task == "inflight":
                assert model(x[:, :100]).shape == (2, 10, 4)
                continue
            baseline = model(x, lengths)
            assert baseline.shape == (2, 181, 1)
            modified = x.clone()
            modified[:, 100:, 17] = float("nan")
            modified[0, 151:] = float("nan")
            torch.testing.assert_close(
                model(modified, lengths), baseline, rtol=0, atol=0
            )
            torch.testing.assert_close(
                model(x[:1, :151], lengths[:1]),
                baseline[:1, :151],
                rtol=1e-4,
                atol=1e-6,
            )
            assert torch.count_nonzero(baseline[0, 151:]) == 0


def test_different_input_and_target_admission():
    run = run_fixture()
    assert origins(run).tolist() == list(range(99, 115))
    run.observed[105] = False
    # A missing future target rules out supervised origins, not publication.
    assert 99 in forecast_origins(run)
    assert 99 not in origins(run)
    run.segment[115:] = 1
    assert forecast_origins(run).max() == 114
    assert len(origins(run)) == 0


class FixedForecast:
    def forecast(self, x):
        return np.broadcast_to(x[:, -1:, [4, 5, 6, 17]], (len(x), 10, 4)).copy()


def test_deadline_overlap_gap_and_no_future_replay():
    clock = [0]
    run = run_fixture()
    session = FlightSession(FixedForecast(), clock_us=lambda: clock[0])
    for index, origin in enumerate(run.time_us[:99]):
        assert (
            session.step(
                origin,
                0,
                run.values[index],
                run.observed[index],
                last_input_available_us=origin,
            )[0]
            is None
        )
    t = run.time_us[99]
    clock[0] = t + STEP_US
    assert (
        session.step(
            t, 0, run.values[99], run.observed[99], last_input_available_us=t
        )[2]
        == "deadline_missed"
    )
    t += STEP_US
    clock[0] = t + 1
    packet, residual, status = session.step(
        t, 0, run.values[100], run.observed[100], last_input_available_us=t
    )
    assert (
        packet.issued_us == t + 1 and residual is None and status == "published"
    )
    with pytest.raises(ValueError, match="Late observations"):
        session.step(
            t + STEP_US,
            0,
            run.values[101],
            run.observed[101],
            last_input_available_us=t + STEP_US + 1,
        )
    clock[0] += STEP_US
    _, residual, _ = session.step(
        t + STEP_US,
        0,
        run.values[101],
        run.observed[101],
        last_input_available_us=t + STEP_US,
    )
    np.testing.assert_array_equal(residual, np.zeros(4))
    clock[0] += STEP_US
    assert (
        session.step(
            t + 2 * STEP_US,
            1,
            run.values[102],
            run.observed[102],
            last_input_available_us=t + 2 * STEP_US,
        )[1]
        is None
    )

    a, counts, _ = replay_forecasts(FixedForecast(), run)
    assert counts[100] == 1 and counts[109] == 10 and counts[-1] == 10
    run.values[110:, 17] += 10
    b, _, _ = replay_forecasts(FixedForecast(), run)
    np.testing.assert_equal(a[:111], b[:111])


def test_scores_and_unscored_target_behavior():
    reference = FlightReference.fit(
        np.random.default_rng(5).normal(size=(30, 4))
    )
    high = reference.score([0, 0, 0, 1])
    low = reference.score([0, 0, 0, -1])
    assert low[1] == pytest.approx(high[1] * 5)
    alarm = FlightAlarm([1, 1])
    for i in range(30):
        status = alarm.update(i * STEP_US, [5, 0])
        assert status["alarm"] == (i == 29)
    assert alarm.update(30 * STEP_US, None)["alarm"] is None

    run = run_fixture(222)
    run.observed[104, 3] = False
    run.values[104, 17] = 5
    components, coverage = run_components(run, np.full(len(run.values), 3.5))
    np.testing.assert_allclose(components, [0.0, 0.25])
    assert coverage == pytest.approx(121 / 122)
    calibrated = RunReference.fit([[1, 1]], [[1, 1]] * 19)
    result = calibrated.decide((np.array([2.0, 2.0]), 0.9))
    assert result["score"] == pytest.approx(np.sqrt(8)) and result["alarm"]


def test_split_blocks_raw_lineage_even_when_other_role_is_not_opened(tmp_path):
    names = []
    for role in ("normal_train", "anomaly_test"):
        path = tmp_path / f"{role}.npz"
        run = run_fixture()
        np.savez(
            path,
            values=run.values,
            time_us=run.time_us,
            segment=run.segment,
            observed=run.observed,
        )
        names.append(
            dict(
                path=path.name,
                sha256=digest(path) if not names else "a" * 64,
                run_id=role,
                role=role,
                raw_sha256="same-origin",
                retry_group=role,
                group_id=role,
                pair_id=None,
            )
        )
    manifest = tmp_path / "manifest.json"
    write(
        manifest,
        dict(schema="drone-ai.new19.dataset/1", features=FEATURES, runs=names),
    )
    with pytest.raises(ValueError, match="raw_sha256 crosses"):
        load_split(manifest, ("normal_train",))
