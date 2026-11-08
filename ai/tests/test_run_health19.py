"""Run-model software tests. All inputs are synthetic, never detection evidence."""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from capstone_ai.mapping import FEATURES
from learning.history19.training import backend
from learning.run_health19 import data
from learning.run_health19.data import Scaler, batch, validate_splits
from learning.run_health19.fixtures import normal_runs
from learning.run_health19.models import (
    ModelSpec,
    make_model,
    normal_loss,
    scoring_mask,
)
from learning.run_health19.scoring import Calibration, components
from learning.run_health19.training import fit_cpu, restore


@pytest.fixture(autouse=True)
def fixed_runtime():
    torch.set_num_threads(2)
    torch.manual_seed(0)


@pytest.mark.parametrize("name", ["lstm", "transformer"])
@pytest.mark.parametrize(
    "hidden,layers", [(64, 2), (64, 4), (128, 2), (128, 4)]
)
def test_hierarchical_forward_backward_and_no_later_voltage_copy(
    name, hidden, layers
):
    model = make_model(ModelSpec(name, hidden, layers))
    x = torch.randn(2, 237, 19)
    lengths = torch.tensor([205, 237])
    with backend("cpu"):
        y = model(x, lengths)
        assert y.shape == (2, 237)
        loss = normal_loss(y, x[:, :, 17], lengths)
        loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    assert not any(
        isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()
    )
    assert (
        len(
            model.run_encoder.stack.layers
            if name == "lstm"
            else model.run_encoder.layers
        )
        == layers
    )
    assert (
        len(model.local.stack.layers if name == "lstm" else model.local.layers)
        == layers
    )
    changed = x.clone()
    changed[:, 100:, 17] = 1e5
    with torch.no_grad(), backend("cpu"):
        before, after = model(x, lengths), model(changed, lengths)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert not torch.equal(x[:, 100:, 17], changed[:, 100:, 17])
    changed[:, :100, 17] += 5
    with torch.no_grad(), backend("cpu"):
        assert not torch.equal(
            before[:, 100:], model(changed, lengths)[:, 100:]
        )


@pytest.mark.parametrize("name", ["lstm", "transformer"])
@pytest.mark.parametrize(
    "hidden,layers", [(64, 2), (64, 4), (128, 2), (128, 4)]
)
def test_padding_partial_tail_and_later_block_load_independence(
    name, hidden, layers
):
    model = make_model(ModelSpec(name, hidden, layers)).eval()
    x = torch.randn(2, 421, 19)
    lengths = torch.tensor([205, 421])
    padded = torch.nn.functional.pad(x, (0, 0, 0, 117), value=float("nan"))
    padded[0, 205:] = float("nan")
    changed = x.clone()
    changed[:, 300:, 18] += 100
    with torch.no_grad(), backend("cpu"):
        first = model(x, lengths)
        extra_padding = model(padded, lengths)
        latter = model(changed, lengths)
        single = model(x[0:1, :205], lengths[0:1])
    torch.testing.assert_close(first[0, :205], single[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        first, extra_padding[:, :421], atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(first[:, :300], latter[:, :300], atol=0, rtol=0)
    assert (first[0, 205:] == 0).all()
    assert scoring_mask(lengths, 421).sum(1).tolist() == [105, 321]
    changed = x.clone()
    changed[0, 204, 18] += 5
    with torch.no_grad(), backend("cpu"):
        assert model(changed, lengths)[0, 204] != first[0, 204]


def test_loss_equal_per_run_initial_padding_excluded_and_short_run_unavailable():
    prediction = torch.zeros(2, 205)
    observed = torch.zeros_like(prediction)
    observed[0, 100:105] = 1
    observed[1, 100:] = 3
    observed[:, :100] = float("nan")
    observed[0, 105:] = float("nan")
    assert (
        normal_loss(prediction, observed, torch.tensor([105, 205])).item() == 5
    )
    with pytest.raises(ValueError, match="initial ten seconds"):
        normal_loss(prediction[:1], prediction[:1], torch.tensor([100]))
    assert (
        components(np.zeros(100), np.zeros(100), cell_voltage_scale=1)["status"]
        == "unavailable"
    )


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_reject_changed_architecture_and_bad_inputs(name):
    for field in (
        dict(hidden=256),
        dict(layers=3),
        dict(heads=8),
        dict(hidden=True),
    ):
        with pytest.raises(ValueError):
            ModelSpec(name, **field).validate()
    model = make_model(ModelSpec(name))
    with pytest.raises(ValueError, match="19"):
        model(torch.zeros(1, 101, 25), torch.tensor([101]))
    bad = torch.zeros(1, 101, 19)
    bad[0, 100, 0] = float("nan")
    with pytest.raises(ValueError, match="Missing"):
        model(bad, torch.tensor([101]))


def test_train_only_scaler_preserves_voltage_level_and_guards_lineage():
    train = normal_runs("normal_train", 2)
    validation = normal_runs("normal_validation", 1, seed=1)
    scaler = Scaler.fit(train)
    with pytest.raises(ValueError, match="normal_train only"):
        Scaler.fit([*train, *validation])
    expected = np.concatenate([data.physical_features(r.values) for r in train])
    np.testing.assert_allclose(scaler.mean, expected.mean(0), atol=0, rtol=0)
    altered = dataclasses.replace(
        validation[0], values=validation[0].values.copy()
    )
    altered.values[:, 17] += 4
    difference = (
        scaler.transform(altered)[:, 17]
        - scaler.transform(validation[0])[:, 17]
    )
    np.testing.assert_allclose(difference, 1 / scaler.scale[17], rtol=1e-6)
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(
            [
                train[0],
                dataclasses.replace(validation[0], lineage=train[0].lineage),
            ]
        )
    with pytest.raises(ValueError, match="eligibility"):
        dataclasses.replace(train[0], health="abnormal").validate()
    with pytest.raises(ValueError, match="clock resets"):
        dataclasses.replace(train[0], time_us=train[0].time_us * 2).validate()


def test_run_components_hand_calculation_and_missing_not_zero_error():
    observed = np.r_[np.zeros(100), np.tile([-1.0, 3.0], 50), np.full(5, 2.0)]
    result = components(np.zeros(205), observed, cell_voltage_scale=1)
    assert result["samples"] == 105
    assert result["short_cell_v"] == pytest.approx(np.sqrt(400 / 105))
    assert result["long_cell_v"] == pytest.approx(np.sqrt(120 / 105))
    observed[-1] = np.nan
    assert (
        components(np.zeros(205), observed, cell_voltage_scale=1)["status"]
        == "unavailable"
    )


def test_normal_calibration_independence_small_sample_test_data_rejected():
    validation = normal_runs("normal_validation", 2, seed=1)
    calibration = normal_runs("normal_calibration", 19, seed=2)
    record = dict(status="available", short_cell_v=1.0, long_cell_v=2.0)
    cal = Calibration.fit(
        [(r, record) for r in validation], [(r, record) for r in calibration]
    )
    assert cal.threshold == 1
    assert cal.decide(record)["alarm"] is False
    assert cal.decide({**record, "short_cell_v": 2})["alarm"] is True
    assert (
        cal.decide(dict(status="unavailable", reason="missing"))["alarm"]
        is None
    )
    with pytest.raises(ValueError, match="Insufficient"):
        Calibration.fit(
            [(r, record) for r in validation],
            [(r, record) for r in calibration[:2]],
        )
    wrong = dataclasses.replace(
        calibration[0], role="normal_test", cohort="development"
    )
    with pytest.raises(ValueError, match="may not use"):
        Calibration.fit([(r, record) for r in validation], [(wrong, record)])
    same_group = dataclasses.replace(
        calibration[1], lineage=calibration[0].lineage
    )
    with pytest.raises(ValueError, match="independent lineage"):
        Calibration.fit(
            [(r, record) for r in validation],
            [(calibration[0], record), (same_group, record)],
        )


def test_manifest_fixed_order_hash_truth_rejection_and_no_old_dataset_conversion(
    tmp_path,
):
    run = normal_runs("normal_train", 1)[0]
    source = tmp_path / "run.npz"
    np.savez(source, values=run.values, time_us=run.time_us)
    item = dict(
        run_id=run.run_id,
        role=run.role,
        lineage=run.lineage,
        health=run.health,
        origin=run.origin,
        mission_interval=dict(
            policy=data.INTERVAL_POLICY, start_us=0, end_us_exclusive=20500000
        ),
        path=source.name,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    manifest = dict(
        schema=data.SCHEMA,
        feature_order=FEATURES,
        units="physical_common19_pack_V_A",
        step_us=100000,
        vehicle_profile="x500_reference",
        battery_policy="cmu-vah01-rpt16-only/1",
        interval_policy=data.INTERVAL_POLICY,
        runs=[item],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert len(data.load_manifest(path)[0].values) == 205
    manifest.pop("interval_policy")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mission-only"):
        data.load_manifest(path)
    manifest["interval_policy"] = data.INTERVAL_POLICY
    item["mission_interval"]["start_us"] = 100000
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Complete declared mission"):
        data.load_manifest(path)
    item["mission_interval"]["start_us"] = 0
    path.write_text(json.dumps(manifest), encoding="utf-8")
    source.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.load_manifest(path)
    np.savez(
        source, values=run.values, time_us=run.time_us, truth=np.zeros(205)
    )
    item["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="truth/context"):
        data.load_manifest(path)


@pytest.mark.parametrize("name", ["lstm", "transformer"])
@pytest.mark.parametrize(
    "hidden,layers", [(64, 2), (64, 4), (128, 2), (128, 4)]
)
def test_cpu_normal_fit_validation_fresh_process_resume_export(
    name, hidden, layers, tmp_path
):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    def call(directory, *arguments):
        command = [
            sys.executable,
            "-B",
            "-m",
            "learning.run_health19.cli",
            "smoke",
            "--model",
            name,
            "--hidden",
            str(hidden),
            "--layers",
            str(layers),
            "--output",
            str(tmp_path / directory),
            *arguments,
        ]
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    full = call("full")
    call("partial", "--stop-after-epoch", "2")
    resumed = call("resumed", "--resume", str(tmp_path / "partial/last.pt"))
    assert full == resumed and full["optimizer_steps"] == 8
    assert full["origins"] == ["synthetic_unit_fixture"]
    assert full["detection_performance_verified"] is False
    state = torch.load(tmp_path / "full/last.pt", weights_only=True)
    other = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert state["plan"]["optimization"]["optimizer"] == "sgd"
    assert state["plan"]["model"] == ModelSpec(name, hidden, layers).to_dict()
    for key, value in state["model"].items():
        torch.testing.assert_close(value, other["model"][key], atol=0, rtol=0)
    torch.testing.assert_close(state["rng"], other["rng"], atol=0, rtol=0)
    saved = load_file(tmp_path / "full/model.safetensors")
    for key, value in saved.items():
        torch.testing.assert_close(
            value, state["best_model"][key], atol=0, rtol=0
        )
        torch.testing.assert_close(
            value,
            load_file(tmp_path / "resumed/model.safetensors")[key],
            atol=0,
            rtol=0,
        )
    model, scaler = restore(tmp_path / "full")
    run = normal_runs("normal_validation", 1, seed=1)
    x, lengths = batch(run, scaler)
    with torch.no_grad(), backend("cpu"):
        assert torch.isfinite(model(x, lengths)).all()
    train = normal_runs("normal_train", 4)
    train[0].values[0, 0] += 1
    with pytest.raises(ValueError, match="Resume"):
        fit_cpu(
            ModelSpec(name, hidden, layers),
            train,
            normal_runs("normal_validation", 2, seed=1),
            tmp_path / "bad",
            resume=tmp_path / "partial/last.pt",
        )


def test_config_eight_run_models_match_online_sizes_and_optimizer():
    from execution.common import ROOT

    cfg = json.loads(
        (ROOT / "ai/configs/run-health19.json").read_text(encoding="utf-8")
    )
    expected = {
        (name, hidden, layers)
        for name in ("lstm", "transformer")
        for hidden, layers in ((64, 2), (64, 4), (128, 2), (128, 4))
    }
    assert {
        (v["name"], v["hidden"], v["layers"]) for v in cfg["variants"]
    } == expected
    assert len(cfg["variants"]) == 8
    assert all(v["heads"] == 4 for v in cfg["variants"])
    assert (
        cfg["optimizer"] == "sgd" and cfg["production"]["authorized"] is False
    )
    assert cfg["gpu"]["authorized"] is False
    online = json.loads(
        (ROOT / "ai/configs/history19-sgd.json").read_text(encoding="utf-8")
    )
    assert len(online["variants"]) == 8
    assert {
        (v["name"], v["hidden"], v["layers"]) for v in online["variants"]
    } == expected
    assert all(v["optimizer"] == cfg["optimizer"] for v in online["variants"])
