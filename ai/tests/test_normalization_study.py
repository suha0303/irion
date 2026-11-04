"""CPU-only model, initialization and optimizer/schedule resume regressions."""

import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from learning import models
from learning.diagnostic import same
from learning.models import PostDecoderLayer, PostEncoderLayer, StudySpec, make_model
from learning.optimization import (
    WarmupCosine,
    learning_rate,
    make_optimizer,
    policy_for,
    warmup_steps,
)
from learning.training import backend


@pytest.mark.parametrize("name", ["lstm", "transformer"])
@pytest.mark.parametrize("hidden,layers,heads", [(64, 2, 4), (128, 2, 4), (128, 4, 4), (256, 4, 8)])
def test_new_model_has_no_dropout_and_finite_gradients(name, hidden, layers, heads):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = make_model(StudySpec(name, hidden, layers, heads)).train()
    assert not any(isinstance(m, (nn.Dropout, nn.Identity)) for m in model.modules())
    assert all(m.dropout == 0 for m in model.modules() if isinstance(m, nn.MultiheadAttention))
    with backend("cpu"):
        y = model(torch.randn(2, 100, 19), torch.randn(2, 6))
        assert y.shape == (2, 10, 4) and torch.isfinite(y).all()
        y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if name == "lstm":
        assert len(model.encoder.norms) == len(model.decoder.norms) == layers - 1


def test_new_source_does_not_construct_or_remove_dropout():
    source = inspect.getsource(models)
    assert "nn.Dropout" not in source
    assert "remove_dropout" not in source
    assert "nn.Identity" not in source


def test_lstm_gate_initialization():
    torch.manual_seed(0)
    model = make_model(StudySpec("lstm", 16, 2))
    for stack in (model.encoder, model.decoder):
        for layer in stack.layers:
            for gate in layer.weight_hh_l0.chunk(4, 0):
                torch.testing.assert_close(gate.T @ gate, torch.eye(16), atol=1e-6, rtol=1e-5)
            effective = layer.bias_ih_l0 + layer.bias_hh_l0
            expected = torch.zeros(64)
            expected[16:32] = 1
            torch.testing.assert_close(effective, expected, rtol=0, atol=0)
            for gate in layer.weight_ih_l0.chunk(4, 0):
                limit = math.sqrt(6 / sum(gate.shape))
                assert gate.abs().max() <= limit
    for norm in (model.encoder.norms[0], model.decoder.norms[0]):
        assert torch.equal(norm.weight, torch.ones(16))
        assert torch.equal(norm.bias, torch.zeros(16))


@pytest.mark.parametrize("decoder", [False, True])
def test_post_ln_layer_matches_native_equations_and_backward(decoder):
    torch.manual_seed(0)
    layer = PostDecoderLayer(16, 4) if decoder else PostEncoderLayer(16, 4)
    klass = nn.TransformerDecoderLayer if decoder else nn.TransformerEncoderLayer
    reference = klass(16, 4, 32, dropout=0, activation="gelu", batch_first=True)
    state = {}
    for key, value in layer.state_dict().items():
        key = key.replace("self_attention.", "self_attn.")
        key = key.replace("cross_attention.", "multihead_attn.")
        key = key.replace("attention.", "self_attn.")
        key = key.replace("feedforward.0.", "linear1.").replace("feedforward.2.", "linear2.")
        state[key] = value
    reference.load_state_dict(state)
    x = torch.randn(2, 10, 16, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    memory = torch.randn(2, 20, 16)
    with backend("cpu"):
        actual = layer(x, memory) if decoder else layer(x)
        expected = reference(other, memory) if decoder else reference(other)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
    torch.testing.assert_close(x.grad, other.grad, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name", ["sgd", "adam"])
def test_schedule_endpoints_and_optimizer_settings(name):
    policy = policy_for(name)
    rates = [learning_rate(i, 1000, 0.001, policy) for i in range(1000)]
    warmup = warmup_steps(1000, policy)
    assert warmup == 50
    assert rates[0] == pytest.approx(0.0001)
    assert rates[warmup - 1] == pytest.approx(0.001)
    assert rates[-1] == pytest.approx(0.00001)
    assert all(a <= b for a, b in zip(rates[:warmup - 1], rates[1:warmup]))
    assert all(a >= b for a, b in zip(rates[warmup - 1:-1], rates[warmup:]))
    optimizer = make_optimizer([nn.Parameter(torch.ones(2))], 0.001, policy)
    assert type(optimizer) is (torch.optim.SGD if name == "sgd" else torch.optim.Adam)
    assert optimizer.param_groups[0]["weight_decay"] == 0
    if name == "sgd":
        assert optimizer.param_groups[0]["momentum"] == 0
        assert optimizer.param_groups[0]["nesterov"] is False
    with pytest.raises(ValueError):
        make_optimizer([nn.Parameter(torch.ones(2))], 0.001, {**policy, "momentum": 0.9})


@pytest.mark.parametrize("bad_optimizer", [None, [], {}, True, "adamw"])
def test_invalid_optimizer_is_rejected_consistently(bad_optimizer):
    with pytest.raises(ValueError):
        policy_for(bad_optimizer)


def test_cached_schedule_keeps_exact_values_and_freezes_policy():
    policy = policy_for("adam")
    schedule = WarmupCosine(1000, .001, policy)
    for step in range(1000):
        assert schedule(step) == learning_rate(step, 1000, .001, policy)
    policy["minimum_ratio"] = .5
    assert schedule(999) == pytest.approx(.00001)
    for step in (-1, 1000, True, .5):
        with pytest.raises(ValueError):
            schedule(step)


@pytest.mark.parametrize("dimensions", [dict(hidden=64.0), dict(layers=True), dict(heads=4.0)])
def test_model_dimensions_reject_nonintegers(dimensions):
    with pytest.raises(ValueError, match="integers"):
        StudySpec("lstm", **dimensions).validate()


@pytest.mark.parametrize("name", ["lstm", "transformer"])
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_new_schedule_resume_in_separate_processes(tmp_path, name, optimizer):
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "data.npz", X=rng.normal(size=(4, 100, 19)),
             C=rng.normal(size=(4, 6)), Y=rng.normal(size=(4, 10, 4)))
    runner = tmp_path / "run.py"
    runner.write_text('''import sys,numpy as np
from pathlib import Path
from learning.models import StudySpec
from learning.optimization import policy_for
from learning.training import run_fit
phase,name,optimizer,root=sys.argv[1:]
root=Path(root); arrays=dict(np.load(root/'data.npz'))
run_fit(StudySpec(name,8),arrays,arrays,root/phase,
    binding={'data':'fixture','scaler':'fixture','sources':'fixture'},
    seed=0,batch_size=2,epochs=4,patience=10,learning_rate=.001,
    optimization=policy_for(optimizer),
    stop_after_epoch=2 if phase=='partial' else None,
    resume=root/'partial/last.pt' if phase=='resume' else None)
''')
    for phase in ("full", "partial", "resume"):
        subprocess.run([sys.executable, "-B", str(runner), phase, name, optimizer, str(tmp_path)],
                       check=True, capture_output=True, env=dict(os.environ), timeout=40)
    full = torch.load(tmp_path / "full/last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(tmp_path / "resume/last.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "best_state", "best_epoch", "rng",
                "completed_optimizer_steps", "early_stop_bad_epochs", "initial_state_sha256"):
        assert same(full[key], resumed[key]), key
    for key in ("train_loss", "validation_loss", "learning_rate_last", "optimizer_steps"):
        assert [h[key] for h in full["history"]] == [h[key] for h in resumed["history"]]
    assert (tmp_path / "full/model.safetensors").read_bytes() == (
        tmp_path / "resume/model.safetensors").read_bytes()
    initial = json.loads((tmp_path / "full/initialization.json").read_text())
    assert initial["state_sha256"] == full["initial_state_sha256"]
    # The model was trained/exported in another process. Restore through the public reader.
    from capstone_ai.mapping import FEATURES, TARGETS
    from capstone_ai.training import Predictor
    from execution.common import sha, write

    bundle = tmp_path / "full"
    write(bundle / "model.json", dict(name=name, hidden=8, task="forecast",
          feature_order=FEATURES, target_order=TARGETS, model_revision=models.STUDY_REVISION,
          config=dict(batch_size=4)))
    np.savez(bundle / "scaler.npz", x_mean=np.zeros(19), x_scale=np.ones(19),
             c_mean=np.zeros(6), c_scale=np.ones(6), y_mean=np.zeros(4), y_scale=np.ones(4))
    write(bundle / "integrity.json", {n: sha(bundle / n) for n in
          ("model.json", "model-spec.json", "scaler.npz", "model.safetensors")})
    predictor = Predictor(bundle, device="cpu")
    reference = make_model(StudySpec(name, 8)).eval()
    reference.load_state_dict(full["best_state"])
    with np.load(tmp_path / "data.npz") as data, backend("cpu"), torch.no_grad():
        actual = predictor.predict_scaled(data["X"], data["C"])
        expected = reference(torch.as_tensor(data["X"], dtype=torch.float32),
                             torch.as_tensor(data["C"], dtype=torch.float32)).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_study_contract_is_sixteen_post_ln_variants():
    from test_model_comparison import config

    from learning import comparison, lifecycle, study

    cfg = study.configure(dict(config(), execution_scope="training_only",
                               calibration_authorized=False, evaluation_authorized=False,
                               evaluation_hours=0))
    comparison.config_check(cfg)
    assert len(cfg["models"]) == len(set(cfg["models"])) == 16
    assert {v["optimizer"] for v in cfg["variants"]} == {"sgd", "adam"}
    assert "dropout" not in cfg and cfg["learning_rate"] is None
    assert all(lifecycle.spec_for(cfg, name).revision == models.STUDY_REVISION
               for name in cfg["models"])
    for change in ({"dropout": 0}, {"weight_decay": 1e-5}, {"models": cfg["models"][:-1]},
                   {"learning_rate": .001}):
        with pytest.raises(ValueError):
            comparison.config_check({**cfg, **change})


def test_configuring_a_study_does_not_mutate_its_contract():
    from learning import study

    cfg = study.configure({})
    cfg["variants"][0]["optimizer"] = "adamw"
    cfg["models"].clear()
    fresh = study.configure({})
    assert fresh["variants"][0]["optimizer"] == "sgd"
    assert len(fresh["models"]) == 16


def test_learning_rate_selection_is_finite_normal_only_and_bound(tmp_path, monkeypatch):
    from execution.common import read, write
    from learning import study, training

    cfg = dict(variants=study.VARIANTS, binding=dict(data="normal", scaler="common", sources="frozen"),
               seed=0, batch_size=256, early_stopping_patience=10)
    train = {k: np.zeros(shape) for k, shape in
             (("X", (4, 100, 19)), ("C", (4, 6)), ("Y", (4, 10, 4)))}
    val = {k: np.ones_like(v) for k, v in train.items()}
    calls = []

    def fit(spec, train, validation, folder, **kw):
        assert np.all(train["X"] == 0) and np.all(validation["X"] == 1)
        assert kw["epochs"] == 3 and kw["seed"] == 0
        assert kw["optimization"]["optimizer"] in {"sgd", "adam"}
        calls.append(kw)
        history = [{"epoch": i + 1, "validation_loss": (kw["learning_rate"] - .001)**2}
                   for i in range(3)]
        write(folder / "history.json", history)
        write(folder / "initialization.json", dict(
            state_sha256=hashlib.sha256(str(spec).encode()).hexdigest(), model_spec=spec.to_dict(),
            seed=kw["seed"], weights_from_previous_training=False,
        ))
        return history

    monkeypatch.setattr(training, "run_fit", fit)
    progress = type("Progress", (), {"check": lambda self: None,
                                    "update": lambda self, **kw: None})()
    study.select_rates(cfg, tmp_path, train, val, progress)
    assert len(calls) == 48
    result = read(tmp_path / "learning-rate-selection.json")
    assert result["event_truth_used"] is result["development_data_used"] is False
    assert result["probe_weights_reused_for_main"] is False
    for name in study.MODELS:
        assert study.selected_settings(cfg, name, tmp_path)["learning_rate"] == .001
    original = copy.deepcopy(result)
    for key in ("event_truth_used", "development_data_used", "final_test_used",
                "probe_weights_reused_for_main"):
        changed = copy.deepcopy(original)
        changed[key] = True
        write(tmp_path / "learning-rate-selection.json", changed)
        with pytest.raises(ValueError, match="protocol"):
            study.selected_settings(cfg, study.MODELS[0], tmp_path)
    changed = copy.deepcopy(original)
    changed["protocol"]["epochs"] = 99
    write(tmp_path / "learning-rate-selection.json", changed)
    with pytest.raises(ValueError, match="protocol"):
        study.selected_settings(cfg, study.MODELS[0], tmp_path)
    result = original
    result["models"][study.MODELS[0]]["peak_learning_rate"] = .03
    write(tmp_path / "learning-rate-selection.json", result)
    with pytest.raises(ValueError, match="selection rule"):
        study.selected_settings(cfg, study.MODELS[0], tmp_path)


def test_early_stopping_patience_starts_after_warmup(tmp_path, monkeypatch):
    from learning import training

    monkeypatch.setattr(training, "safe_step", lambda *a: torch.tensor(1.0))
    monkeypatch.setattr(training, "loss", lambda *a: torch.tensor(1.0))
    a = dict(X=np.zeros((2, 100, 19)), C=np.zeros((2, 6)), Y=np.zeros((2, 10, 4)))
    history = training.run_fit(
        StudySpec("lstm", 8), a, a, tmp_path / "fit", epochs=20, patience=2, batch_size=2,
        binding=dict(data="normal", scaler="common", sources="frozen"),
        optimization=policy_for("sgd"),
    )
    assert len(history) == 4
    assert [h["early_stop_bad_epochs"] for h in history] == [0, 0, 1, 2]
