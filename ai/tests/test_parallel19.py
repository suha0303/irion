"""Check fixed training plans, resource admission and bounded execution."""

import numpy as np
import pytest
import torch

from learning.parallel19.fit import Data, ranking, trial


@pytest.mark.parametrize("task", ["forecast", "run_health"])
def test_staged_continuation_equals_full_and_initial_is_shared(tmp_path, task):
    torch.set_num_threads(1)
    entry = dict(task=task, name="lstm", hidden=64, layers=2, batch=2)
    rng = np.random.default_rng(21)
    for role in ("normal_train", "normal_validation"):
        p = tmp_path / "data" / task / role
        p.mkdir(parents=True)
        np.save(
            p / "X.npy",
            rng.normal(size=(4, 100 if task == "forecast" else 201, 19)).astype(
                "f4"
            ),
        )
        if task == "forecast":
            np.save(p / "Y.npy", rng.normal(size=(4, 10, 4)).astype("f4"))
        else:
            np.save(
                p / "lengths.npy", np.array([201, 199, 201, 199], dtype="i8")
            )
    a = Data(tmp_path / "data", entry)
    b = Data(tmp_path / "data", entry)
    trial(entry, a, tmp_path / "staged", 0.01, 2, {"test": 1}, device="cpu")
    staged = trial(
        entry, a, tmp_path / "staged", 0.01, 4, {"test": 1}, device="cpu"
    )
    full = trial(
        entry, b, tmp_path / "full", 0.01, 4, {"test": 1}, device="cpu"
    )
    assert staged["initial"] == full["initial"]
    assert staged["validation_loss"] == full["validation_loss"]
    changed = trial(
        entry, b, tmp_path / "other_rate", 0.1, 4, {"test": 1}, device="cpu"
    )
    assert changed["initial"] == full["initial"]
    with pytest.raises(ValueError, match="identity"):
        trial(entry, a, tmp_path / "staged", 0.01, 5, {"test": 2}, device="cpu")


def test_balanced_probe_preserves_all_runs(tmp_path):
    for role in ("normal_train", "normal_validation"):
        p = tmp_path / "forecast" / role
        p.mkdir(parents=True)
        n = 9000
        np.save(p / "X.npy", np.zeros((n, 100, 19), dtype="f4"))
        np.save(p / "Y.npy", np.zeros((n, 10, 4), dtype="f4"))
        np.save(p / "run_id.npy", np.repeat(["a", "b", "c"], 3000))
    d = Data(tmp_path, dict(task="forecast", batch=256), probe=True)
    for role, n in [("normal_train", 8192), ("normal_validation", 4096)]:
        ids = d.ids[role]
        assert len(ids) == n and len(set(ids)) == n
        assert np.ptp(np.bincount(ids // 3000)) <= 1


def test_no_incomplete_or_divergent_selection():
    rows = [
        dict(status="DIVERGED", validation_loss=None, rate=0.1),
        dict(status="PASS", validation_loss=1.0, rate=0.01),
    ]
    assert ranking(rows, 1)[0]["rate"] == 0.01
    with pytest.raises(RuntimeError):
        ranking(rows, 2)


def test_completed_receipt_waits_for_nfs_visibility(monkeypatch):
    from learning.parallel19 import control

    calls = []

    def read(path):
        calls.append(path)
        if len(calls) < 3:
            raise FileNotFoundError(path)
        return {"status": "GATE_PASSED"}

    monkeypatch.setattr(control, "read", read)
    monkeypatch.setattr(control.time, "sleep", lambda _: None)
    assert control.completed_result("receipt")["status"] == "GATE_PASSED"
    assert len(calls) == 3

    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(control, "read", missing)
    with pytest.raises(FileNotFoundError):
        control.completed_result("missing", timeout=0)
