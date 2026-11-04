"""Small CPU probe regressions and pure budget admission; no GPU or server work."""

import copy

import numpy as np
import pytest
import torch

from execution.common import read
from learning import benchmark
from learning.models import ModelSpec


def arrays(n):
    rng = np.random.default_rng(0)
    return {
        "X": rng.normal(size=(n, 100, 19)).astype("float32"),
        "C": rng.normal(size=(n, 6)).astype("float32"),
        "Y": rng.normal(size=(n, 10, 4)).astype("float32"),
    }


def test_validation_is_a_separate_full_pass_without_optimizer_steps(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    step = benchmark.safe_step
    calls = []

    def recorded(*args):
        calls.append(True)
        return step(*args)

    monkeypatch.setattr(benchmark, "safe_step", recorded)
    row = benchmark.measure(
        ModelSpec("lstm", 8),
        arrays(7),
        [3],
        tmp_path / "probe",
        warmup=1,
        steps=3,
        seed=0,
        validation=arrays(5),
    )[0]
    assert len(calls) == 4
    assert row["steps"] == 3 and row["actual_samples"] == 7
    assert row["validation"]["windows"] == 5 and row["validation"]["steps"] == 2
    assert row["validation"]["full_validation"] is True
    assert row["validation"]["seconds"] > 0
    assert row["validation"]["prepare_seconds"] >= 0
    assert row["seed"] == 0 and row["precision"] == "float32"
    assert "warms cache" in row["input_mode"]
    assert not list((tmp_path / "probe").glob("*.pt"))


def test_invalid_validation_stops_and_preserves_failure(tmp_path):
    validation = arrays(5)
    validation["X"][4, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        benchmark.measure(
            ModelSpec("lstm", 8),
            arrays(7),
            [3],
            tmp_path / "failed",
            warmup=0,
            steps=1,
            validation=validation,
        )
    assert read(tmp_path / "failed/failure.json")["automatic_retry"] is False


def row():
    return dict(
        passed=True,
        device="cuda",
        batch=256,
        steps=500,
        warmup_steps=50,
        representative_windows=1000,
        step_seconds=0.5,
        seed=0,
        model_spec=dict(name="lstm", hidden=128, layers=2, heads=4),
        validation=dict(windows=300, full_validation=True, seconds=1.5),
    )


def estimate(rows, hours=2):
    return benchmark.estimate_fit_budget(
        rows,
        train_windows=1000,
        validation_windows=300,
        epochs=60,
        elapsed_seconds=100,
        wall_hours=hours,
    )


def test_budget_accounts_for_full_epochs_validation_elapsed_and_reserves():
    result = estimate([row()])
    assert result["models"][0]["train_seconds"] == 120
    assert result["models"][0]["validation_seconds"] == 90
    assert result["projected_total_with_reserves_seconds"] == 100 + 210 * 1.25 + 3600
    assert result["passed"] is True
    assert estimate([row()], hours=1)["passed"] is False
    assert result["automatic_budget_extension"] is False
    assert "development evaluation" in result["excluded_stages"]


@pytest.mark.parametrize(
    "patch",
    [
        dict(steps=20),
        dict(warmup_steps=10),
        dict(device="cpu"),
        dict(representative_windows=64),
        dict(passed=False),
        dict(step_seconds=float("nan")),
        dict(validation=dict(windows=4, full_validation=True, seconds=1.5)),
        dict(validation=None),
    ],
)
def test_budget_rejects_unrepresentative_or_incomplete_evidence(patch):
    value = copy.deepcopy(row())
    value.update(patch)
    with pytest.raises(ValueError):
        estimate([value])
