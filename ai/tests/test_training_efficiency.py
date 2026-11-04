"""Numerical equivalence and failure guards for the measured training hot paths."""

import copy

import numpy as np
import pytest
import torch

from capstone_ai.training import loss, seed_all
from execution import common
from learning import comparison, study
from learning.diagnostic import same
from learning.models import make_model
from learning.optimization import make_optimizer, policy_for
from learning.training import backend, clip_gradients, safe_step


def reference_step(model, optimizer, x, c, y):
    """Pre-optimization implementation, kept here as an independent oracle."""
    optimizer.zero_grad(set_to_none=True)
    value = loss(model(x, c), y)
    if not bool(torch.isfinite(value)):
        raise ValueError("Nonfinite loss")
    value.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise ValueError("Missing gradients")
    flags = [torch.isfinite(g).all() for g in gradients]
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    flags.append(torch.isfinite(norm))
    if not bool(torch.stack(flags).all()):
        raise ValueError("Nonfinite gradients/norm")
    optimizer.step()
    return value.detach()


@pytest.mark.parametrize("entry", study.VARIANTS, ids=lambda entry: entry["id"])
def test_three_steps_match_previous_weights_gradients_optimizer_and_rng(entry):
    seed_all(0)
    old = make_model(study.spec_for(entry))
    new = copy.deepcopy(old)
    policy = policy_for(entry["optimizer"])
    left = make_optimizer(old.parameters(), .001, policy)
    right = make_optimizer(new.parameters(), .001, policy)
    inputs = (torch.randn(2, 100, 19), torch.randn(2, 6), torch.randn(2, 10, 4) * 10)
    with backend("cpu"):
        for _ in range(3):
            rng = torch.get_rng_state()
            expected = reference_step(old, left, *inputs)
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            actual = safe_step(new, right, *inputs)
            assert same(expected, actual)
            assert same(old.state_dict(), new.state_dict())
            assert same(left.state_dict(), right.state_dict())
            assert all(same(p.grad, q.grad) for p, q in zip(old.parameters(), new.parameters()))
            assert torch.equal(expected_rng, torch.get_rng_state())


@pytest.mark.parametrize("value", [0., 1e-30, 1., 1e10, 3e38])
def test_clipping_matches_previous_reduction_including_finite_overflow(value):
    model = torch.nn.Linear(3, 2)
    for p in model.parameters():
        p.grad = torch.full_like(p, value)
    previous = copy.deepcopy(model)
    # Module deepcopy does not copy .grad.
    for p, q in zip(model.parameters(), previous.parameters()):
        q.grad = p.grad.clone()
    norm = torch.nn.utils.clip_grad_norm_(previous.parameters(), 1.0, foreach=True)
    if not bool(torch.isfinite(norm)):
        with pytest.raises(ValueError, match="Nonfinite gradients/norm"):
            clip_gradients(model)
    else:
        clip_gradients(model)
        assert all(torch.equal(p.grad, q.grad) for p, q in zip(model.parameters(), previous.parameters()))


@pytest.mark.parametrize("kind", ["sgd", "adam"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 3e38])
def test_bad_gradient_or_overflow_cannot_change_weights_or_optimizer(kind, bad):
    seed_all(0)
    model = make_model(study.spec_for(study.VARIANTS[0]))
    optimizer = make_optimizer(model.parameters(), .001, policy_for(kind))
    inputs = (torch.randn(2, 100, 19), torch.randn(2, 6), torch.randn(2, 10, 4))
    with backend("cpu"):
        # Populate optimizer state first; a failed step must preserve existing moments too.
        safe_step(model, optimizer, *inputs)
        weights = copy.deepcopy(model.state_dict())
        state = copy.deepcopy(optimizer.state_dict())
        handle = next(model.parameters()).register_hook(lambda g: torch.full_like(g, bad))
        try:
            with pytest.raises(ValueError, match="Nonfinite gradients/norm"):
                safe_step(model, optimizer, *inputs)
        finally:
            handle.remove()
    assert same(model.state_dict(), weights)
    assert same(optimizer.state_dict(), state)


def test_no_gradients_and_nonfinite_loss_still_stop_before_update():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="Missing gradients"):
        clip_gradients(model)
    seed_all(0)
    model = make_model(study.spec_for(study.VARIANTS[0]))
    optimizer = make_optimizer(model.parameters(), .001, policy_for("adam"))
    weights = copy.deepcopy(model.state_dict())
    inputs = (torch.zeros(2, 100, 19), torch.zeros(2, 6), torch.full((2, 10, 4), float("nan")))
    with backend("cpu"), pytest.raises(ValueError, match="Nonfinite loss"):
        safe_step(model, optimizer, *inputs)
    assert same(model.state_dict(), weights)
    assert not optimizer.state


def test_normalization_throttles_writes_but_preserves_checks_values_and_final_status(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    arrays = {"X": rng.normal(size=(23, 100, 19)), "C": rng.normal(size=(23, 6)),
              "Y": rng.normal(size=(23, 10, 4))}
    scaler = {k + suffix: np.full(n, value) for k, n in (("x", 19), ("c", 6), ("y", 4))
              for suffix, value in (("_mean", .13), ("_scale", 1.7))}
    path = tmp_path / "normal_train.npz"
    np.savez_compressed(path, **arrays)
    common.write(tmp_path / "manifest.json", {"hashes": {path.name: common.sha(path)}})
    # A fast interval should perform boundary writes only. All memory checks remain.
    monkeypatch.setattr(common.time, "monotonic", lambda: 100.)
    checks, writes = [], []
    monkeypatch.setattr(common, "memory_snapshot", lambda: checks.append(1) or {"available_bytes": 2**32})
    original_write = common.write

    def record(path, value):
        if path.name == "progress.json":
            writes.append(copy.deepcopy(value))
        original_write(path, value)

    monkeypatch.setattr(common, "write", record)
    progress = common.Progress(tmp_path / "progress.json")
    result = comparison.normalized_mmap(tmp_path, "normal_train", scaler, tmp_path / "cache", progress, chunk=3)
    assert len(checks) == 24  # No safety-check throttling.
    assert len(writes) == 6  # Was24 forced writes; first and completion per array remain.
    for key, scale in (("X", "x"), ("C", "c"), ("Y", "y")):
        expected = ((arrays[key] - scaler[scale + "_mean"]) / scaler[scale + "_scale"]).astype(np.float32)
        np.testing.assert_array_equal(result[key], expected)
    final = common.read(tmp_path / "progress.json")
    assert final["array"] == "Y" and final["done"] == final["total"] == 23
    assert [w["array"] for w in writes] == ["X", "X", "C", "C", "Y", "Y"]


def test_progress_throttling_keeps_periodic_updates(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(common.time, "monotonic", lambda: clock[0])
    progress = common.Progress(tmp_path / "progress.json")
    progress.update(done=0)
    clock[0] += 1
    progress.update(force=False, done=1)
    assert common.read(progress.path)["done"] == 0
    clock[0] += 1
    progress.update(force=False, done=2)
    assert common.read(progress.path)["done"] == 2
    progress.update(done=3)
    assert common.read(progress.path)["done"] == 3
