"""Compare optimized replay with the prior sequential causal ledger."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from capstone_ai.dataset import feature_transform
from capstone_ai.detection import DetectionLedger
from capstone_ai.detection_replay import records, replay
from capstone_ai.mapping import FEATURES, TARGETS
from capstone_ai.source import dump, read_json, sha
from learning.models import ModelSpec, make_model

spec = importlib.util.spec_from_file_location(
    "replay_reference", Path(__file__).parent / "fixtures/replay_reference.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class FixturePredictor:
    def __init__(self, folder, model=None):
        self.folder = folder
        self.model = model
        self.calls = []
        self.scaler = dict(
            x_mean=np.zeros(19), x_scale=np.ones(19), c_mean=np.zeros(6), c_scale=np.ones(6)
        )

    def predict_scaled(self, x, c):
        self.calls.append(len(x))
        if self.model is None:
            values = x[:, -1, :4] * 0.01 + c[:, :4] * 0.01
            return np.repeat(values[:, None, :], 10, axis=1)
        with torch.no_grad():
            return self.model(
                torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(c, dtype=torch.float32)
            ).numpy()

    def physical(self, value):
        return value


def case(tmp_path, windows_per_segment=8):
    canonical, bundle = tmp_path / "canonical", tmp_path / "bundle"
    canonical.mkdir()
    bundle.mkdir()
    dump(bundle / "preprocessing.json", {})
    dump(bundle / "integrity.json", {})
    rng = np.random.default_rng(8)
    frames, windows = [], []
    for segment in range(2):
        base = segment * 20_000_000
        for i in range(windows_per_segment * 10 + 1):
            # Missing sample blocks, invalid targets and a truncated final
            # forecast exercise close/reset and unavailable decisions.
            if segment == 0 and i == 15:
                continue
            t = base + i * 100_000
            frames.append(
                dict(
                    run_id="run",
                    clock_id="clock",
                    segment_id=segment,
                    time_us=t,
                    values=dict.fromkeys(TARGETS, 2.0),
                    quality=dict.fromkeys(TARGETS, "invalid" if i == 25 else "valid"),
                    operating_state="airborne",
                    warmup=False,
                )
            )
            if i % 10 == 0:
                windows.append(
                    dict(
                        clock_id="clock",
                        segment_id=segment,
                        origin_us=t,
                        X=rng.normal(size=(100, len(FEATURES))).tolist(),
                        C=rng.normal(size=6).tolist(),
                    )
                )
    # Canonical window storage order need not equal issuance order.
    rng.shuffle(windows)
    for name, values in (("frames.jsonl", frames), ("windows.jsonl", windows)):
        (canonical / name).write_text("".join(json.dumps(v) + "\n" for v in values))
    dump(canonical / "summary.json", dict(config={}, exclusions={}, output_hashes={}))
    return canonical, bundle, frames, windows


def sequential_reference(predictor, frames, windows):
    windows = {(w["clock_id"], w["segment_id"], w["origin_us"]): w for w in windows}
    ledger = reference.DetectionLedger(np.ones((10, 4)), [1, 1])
    previous, last = None, None
    for frame in frames:
        scope, t = (frame["clock_id"], frame["segment_id"]), frame["time_us"]
        if previous is not None and previous != scope:
            ledger.advance(last, closed=True)
        previous, last = scope, t
        values = (
            [frame["values"][n] for n in TARGETS]
            if all(frame["quality"][n] == "valid" for n in TARGETS)
            else None
        )
        ledger.observe("run", *scope, t, t, values)
        ledger.advance(t)
        w = windows.get((*scope, t))
        if w is not None:
            prediction = predictor.predict_scaled(
                feature_transform([w["X"]], {}), np.asarray([w["C"]])
            )[0]
            ledger.issue("run", *scope, t, t, prediction)
    ledger.advance(last, closed=True)
    return ledger


def test_sequential_replay_matches_prior_causal_decisions_and_syncs_only_at_seal(
    tmp_path, monkeypatch
):
    canonical, bundle, frames, windows = case(tmp_path)
    expected = sequential_reference(FixturePredictor(bundle), frames, windows)
    synced = []
    monkeypatch.setattr("capstone_ai.detection_replay.os.fsync", lambda fd: synced.append(fd))
    predictor = FixturePredictor(bundle)
    out = tmp_path / "result"
    replay(
        canonical,
        bundle,
        {"error_scale_physical": np.ones((10, 4)), "high": [1, 1]},
        out,
        predictor=predictor,
    )
    assert records(out / "decisions.jsonl") == list(expected.decisions.values())
    forecast_fields = set(next(iter(expected.forecasts.values())))
    assert [{k: f[k] for k in forecast_fields} for f in records(out / "forecasts.jsonl")] == list(
        expected.forecasts.values()
    )
    assert len(synced) == 5  # two streams, summary, seal, directory; independent of origin count
    assert predictor.calls == [1] * len(windows)
    seal = read_json(out / "seal.json")
    assert seal["files"]["summary.json"] == sha(out / "summary.json")


@pytest.mark.parametrize("name", [None, "lstm", "transformer"])
@pytest.mark.parametrize("batch_size", [4, 256])
def test_batched_inputs_keep_forecasts_and_discrete_decisions(tmp_path, name, batch_size):
    torch.manual_seed(1)
    torch.set_num_threads(1)
    model = None if name is None else make_model(ModelSpec(name, hidden=64, layers=2)).eval()
    canonical, bundle, frames, windows = case(tmp_path)
    expected = sequential_reference(FixturePredictor(bundle, model), frames, windows)
    predictor = FixturePredictor(bundle, model)
    out = tmp_path / "result"
    summary = replay(
        canonical,
        bundle,
        {"error_scale_physical": np.ones((10, 4)), "high": [1, 1]},
        out,
        predictor=predictor,
        prediction_batch_size=batch_size,
    )
    assert predictor.calls == [
        min(batch_size, len(windows) - i) for i in range(0, len(windows), batch_size)
    ]
    assert summary["compute_seconds_p50_p95_p99"] is None
    assert summary["prediction_batches"] == len(predictor.calls)
    actual = records(out / "decisions.jsonl")
    for new, old in zip(actual, expected.decisions.values()):
        for field in (
            "forecast_id",
            "decision_time_us",
            "target_times_us",
            "matched_target_count",
            "coverage",
            "unscored_reasons",
            "state",
            "alarm_active",
            "episode_start",
            "episode_end",
        ):
            assert new[field] == old[field]
        for domain in ("propulsion", "battery"):
            for field in ("state", "alarm_active", "episode_start"):
                assert new["domains"][domain][field] == old["domains"][domain][field]
        if new["scores"] is not None:
            np.testing.assert_allclose(new["scores"], old["scores"], rtol=2e-5, atol=2e-6)
    for new, old in zip(records(out / "forecasts.jsonl"), expected.forecasts.values()):
        np.testing.assert_allclose(new["prediction"], old["prediction"], rtol=2e-5, atol=2e-6)
        assert new["issued_time_us"] < new["target_times_us"][0]
        assert new["compute_seconds_scope"] == "batch_forward_amortized_per_forecast"


def test_pending_queue_preserves_duplicate_snapshot_and_does_not_scan_finished(tmp_path):
    ledger = DetectionLedger(np.ones((10, 4)), [1, 1])
    key = ledger.issue("run", "clock", 0, 0, 0, np.zeros((10, 4)))
    ledger.issue("run", "clock", 0, 0, 0, np.zeros((10, 4)))
    assert list(ledger.pending) == [key]
    snapshot = ledger.snapshot()
    assert snapshot.advance(1_000_000, closed=True) == ledger.advance(1_000_000, closed=True)
    assert ledger.advance(2_000_000, closed=True) == []
    ledger.issue("run", "clock", 0, 0, 0, np.zeros((10, 4)))
    assert not ledger.pending

    class NoScanDict(dict):
        def items(self):
            raise AssertionError("must not traverse completed forecasts")

    ledger.forecasts = NoScanDict(ledger.forecasts)
    assert ledger.advance(3_000_000) == []


def test_failed_durable_write_does_not_seal_partial_run(tmp_path, monkeypatch):
    canonical, bundle, _, _ = case(tmp_path)

    def fail(fd):
        raise OSError("fixture disk failure")

    monkeypatch.setattr("capstone_ai.detection_replay.os.fsync", fail)
    out = tmp_path / "result"
    with pytest.raises(OSError, match="disk failure"):
        replay(
            canonical,
            bundle,
            {"error_scale_physical": np.ones((10, 4)), "high": [1, 1]},
            out,
            predictor=FixturePredictor(bundle),
            prediction_batch_size=256,
        )
    assert not (out / "seal.json").exists()


@pytest.mark.parametrize("batch", [0, -1, True, 1.5, 4097])
def test_invalid_batch_is_rejected_without_output(tmp_path, batch):
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="prediction_batch_size"):
        replay(tmp_path, tmp_path, {}, out, prediction_batch_size=batch)
    assert not out.exists()
