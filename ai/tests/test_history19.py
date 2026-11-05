"""Fresh19-only SGD implementation: synthetic CPU data only, no GPU/data collection."""

import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from capstone_ai.mapping import FEATURES, TARGETS
from execution.common import ROOT, sha
from learning.history19 import data, runner
from learning.history19.models import ModelSpec, make_model
from learning.history19.predict import Predictor, seal_export
from learning.history19.preprocessing import Preprocessor
from learning.history19.training import Batches, backend, run_fit
from learning.optimization import policy_for


def fixture_dataset(path):
    path.mkdir()
    rng = np.random.default_rng(0)
    runs, files = [], {}
    for role in ("normal_train", "normal_validation"):
        run = role + "-fixture"
        arrays = dict(
            X=np.tile(np.where(np.arange(100) % 2, 1.0, -1.0)[None, :, None], (4, 1, 19)),
            Y=rng.normal(size=(4, 10, 4)),
            run_id=np.array([run] * 4),
            clock_id=np.array(["boot"] * 4),
            segment_id=np.zeros(4, dtype=np.int64),
            origin_us=np.arange(10, 14, dtype=np.int64) * 1_000_000,
        )
        if role == "normal_validation":
            arrays["X"] += 10
        np.savez(path / (role + ".npz"), **arrays)
        runs.append(
            dict(
                run_id=run,
                role=role,
                raw_flight_id=run,
                paired_case_id=run,
                group_id=run,
                health_eligibility="confirmed_healthy",
            )
        )
        files[role] = dict(path=role + ".npz", sha256=sha(path / (role + ".npz")))
    np.savez(
        path / "scaler.npz",
        x_mean=np.zeros(19),
        x_scale=np.ones(19),
        y_mean=np.zeros(4),
        y_scale=np.ones(4),
    )
    files["scaler"] = dict(path="scaler.npz", sha256=sha(path / "scaler.npz"))
    np.savez(
        path / "train-frames.npz",
        values=np.tile(np.where(np.arange(140) % 2, 1.0, -1.0)[:, None], (1, 19)),
        run_id=np.array([runs[0]["run_id"]] * 140),
        clock_id=np.array(["boot"] * 140),
        segment_id=np.zeros(140, dtype=np.int64),
        time_us=np.arange(140, dtype=np.int64) * 100000,
    )
    files["normal_train_frames"] = dict(
        path="train-frames.npz", sha256=sha(path / "train-frames.npz")
    )
    manifest = dict(
        schema=data.SCHEMA,
        feature_profile="common19_ctx0",
        feature_order=FEATURES,
        target_order=TARGETS,
        history=100,
        future=10,
        step_us=100000,
        emit_every_us=1000000,
        scaler_scope="normal_train_unique_valid_frames",
        scaler_fit_run_ids=[runs[0]["run_id"]],
        runs=runs,
        files=files,
        fixture_only=True,
    )
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path / "manifest.json"


def change_array(manifest_path, key, transform):
    manifest = json.loads(manifest_path.read_text())
    role = "normal_train"
    source = manifest_path.parent / (role + ".npz")
    with np.load(source) as archive:
        values = dict(archive)
    transform(values)
    np.savez(source, **values)
    manifest["files"][role]["sha256"] = sha(source)
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("name", ["lstm", "transformer"])
@pytest.mark.parametrize("hidden,layers,heads", [(64, 2, 4), (64, 4, 4), (128, 2, 4), (128, 4, 4)])
def test_only19_features_same_sizes_norm_initialization(name, hidden, layers, heads):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = make_model(ModelSpec(name, hidden, layers, heads))
    assert list(inspect.signature(model.forward).parameters) == ["history"]
    assert not any("context" in k for k in model.state_dict())
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    x = torch.randn(2, 100, 19)
    with backend("cpu"):
        output = model(x)
        assert output.shape == (2, 10, 4) and torch.isfinite(output).all()
        output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(TypeError):
        model(x, torch.zeros(2, 6))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 100, 25))
    if name == "lstm":
        assert model.fuse[0].in_features == hidden
        assert len(model.encoder.norms) == layers - 1
        for stack in (model.encoder, model.decoder):
            for layer in stack.layers:
                expected = torch.zeros(4 * hidden)
                expected[hidden : 2 * hidden] = 1
                torch.testing.assert_close(
                    layer.bias_ih_l0 + layer.bias_hh_l0, expected, atol=0, rtol=0
                )
                for gate in layer.weight_hh_l0.chunk(4):
                    torch.testing.assert_close(
                        gate.T @ gate, torch.eye(hidden), atol=1e-6, rtol=1e-5
                    )


def test_old_conditions_preserved_sgd_only():
    old = json.loads((ROOT / "ai/configs/normalization-study.json").read_text())
    new = runner.config()
    for key in (
        "normalization",
        "initialization",
        "optimization",
        "learning_rate_probe",
        "main_fit",
        "gpu",
    ):
        assert new[key] == old[key]
    assert new["learning_rate_candidates"] == {"sgd": old["learning_rate_candidates"]["sgd"]}
    assert len(new["variants"]) == 8
    assert {(e["hidden"], e["layers"]) for e in new["variants"]} == {
        (64, 2),
        (64, 4),
        (128, 2),
        (128, 4),
    }
    assert all(e["optimizer"] == "sgd" for e in new["variants"])
    assert new["production_training_authorized"] is False
    old_preprocess = json.loads(
        (ROOT / "Data/generation/pretrain/resources/preprocessing/preprocessing.json").read_text()
    )
    new_preprocess = json.loads((ROOT / "ai/learning/history19/preprocessing.json").read_text())
    assert new_preprocess == {**old_preprocess, "feature_profile": "common19_ctx0", "context_us": 0}


def new_engine(seconds=20, change_future=False, chunk=317):
    from test_preprocessing import CONFIG, META, events

    cfg = {**CONFIG, "feature_profile": "common19_ctx0", "context_us": 0}
    engine = Preprocessor(META, cfg)
    rows = list(events(seconds))
    if change_future:
        for row in rows:
            if row["sample_time_us"] > 15_000_000 and row["topic"] == "battery_status":
                row["fields"]["voltage_v"] = 10
    for offset in range(0, len(rows), chunk):
        part = rows[offset : offset + chunk]
        for row in part:
            engine.ingest(row)
        engine.advance(part[-1]["clock_id"], part[-1]["available_time_us"] - 1)
    engine.advance(rows[-1]["clock_id"], rows[-1]["available_time_us"])
    return engine


def test_no60s_warmup_no_context_or_integrals_and_future_causality():
    first, changed = new_engine(), new_engine(change_future=True)
    assert first.windows[0]["origin_us"] == 10_000_000
    assert first.windows[0]["history_times_us"] == list(range(100000, 10000001, 100000))
    assert len(first.windows) == 11
    assert not hasattr(first, "charge") and not hasattr(first, "context")
    for window in first.windows:
        assert len(window["X"]) == 100 and len(window["X"][0]) == 19
        assert "C" not in window and "context_start_us" not in window
        assert window["target_times_us"] == list(
            range(window["origin_us"] + 100000, window["origin_us"] + 1000001, 100000)
        )
    assert [w for w in first.windows if w["origin_us"] <= 15_000_000] == [
        w for w in changed.windows if w["origin_us"] <= 15_000_000
    ]
    assert all("context_warmup" not in e["reasons"] for e in first.exclusions)


def test_chunked_streaming_uses_identical_engine_and_missing_targets_excluded():
    first, second = new_engine(chunk=317), new_engine(chunk=701)
    assert first.frames == second.frames and first.windows == second.windows
    arrays, excluded = data.arrived_samples(first)
    assert arrays["X"].shape == (10, 100, 19) and arrays["Y"].shape == (10, 10, 4)
    assert "C" not in arrays and len(excluded) == 1 and excluded[0]["origin_us"] == 20_000_000


def test_new_dataset_loader_has_no_C_and_keeps_train_scaler(tmp_path):
    manifest = fixture_dataset(tmp_path / "data")
    before = sha(manifest.parent / "normal_train.npz")
    arrays, scaler, binding = data.load(manifest, cache=tmp_path / "cache")
    assert set(arrays["normal_train"]) == {"X", "Y"}
    assert isinstance(arrays["normal_train"]["X"], np.memmap)
    assert np.mean(arrays["normal_train"]["X"]) == 0
    assert np.mean(arrays["normal_validation"]["X"]) == 10
    assert set(scaler) == {"x_mean", "x_scale", "y_mean", "y_scale"}
    assert sha(manifest.parent / "normal_train.npz") == before


@pytest.mark.parametrize(
    "bad", ["C", "width", "target_nan", "duplicate", "extra_truth", "float_time"]
)
def test_bad_or_legacy_dataset_rejected_even_after_rehash(tmp_path, bad):
    manifest = fixture_dataset(tmp_path / "data")

    def mutate(values):
        if bad == "C":
            values["C"] = np.zeros((4, 6))
        elif bad == "width":
            values["X"] = np.zeros((4, 100, 25))
        elif bad == "target_nan":
            values["Y"][0, 0, 0] = np.nan
        elif bad == "duplicate":
            values["origin_us"][1] = values["origin_us"][0]
        elif bad == "extra_truth":
            values["truth"] = np.zeros(4)
        else:
            values["origin_us"] = values["origin_us"].astype(float)

    change_array(manifest, bad, mutate)
    with pytest.raises(ValueError):
        data.load(manifest)


@pytest.mark.parametrize("bad", ["group", "pair", "raw", "retry", "scaler", "features"])
def test_split_and_future_input_rejection(tmp_path, bad):
    path = fixture_dataset(tmp_path / "data")
    manifest = json.loads(path.read_text())
    field = {"group": "group_id", "pair": "paired_case_id", "raw": "raw_flight_id"}.get(bad)
    if field:
        manifest["runs"][1][field] = manifest["runs"][0][field]
    elif bad == "retry":
        manifest["runs"][1]["previous_attempts"] = [manifest["runs"][0]["run_id"]]
    elif bad == "scaler":
        manifest["scaler_fit_run_ids"].append(manifest["runs"][1]["run_id"])
    else:
        manifest["feature_order"] = FEATURES + ["future_voltage"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        data.load(path)


@pytest.mark.parametrize("bad", ["validation_scaler", "duplicate_frame", "foreign_frame"])
def test_scaler_recomputed_from_unique_train_frames(tmp_path, bad):
    path = fixture_dataset(tmp_path / "data")
    manifest = json.loads(path.read_text())
    if bad == "validation_scaler":
        source = path.parent / "scaler.npz"
        with np.load(source) as saved:
            values = dict(saved)
        values["x_mean"] += 10
        values["y_mean"] += 10
        np.savez(source, **values)
        manifest["files"]["scaler"]["sha256"] = sha(source)
    else:
        source = path.parent / "train-frames.npz"
        with np.load(source) as saved:
            values = dict(saved)
        if bad == "duplicate_frame":
            values["time_us"][1] = values["time_us"][0]
        else:
            values["run_id"][0] = manifest["runs"][1]["run_id"]
        np.savez(source, **values)
        manifest["files"]["normal_train_frames"]["sha256"] = sha(source)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        data.load(path)


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_SGD_separate_process_train_resume_export(tmp_path, name):
    torch.set_num_threads(2)
    manifest = fixture_dataset(tmp_path / "dataset")
    script = tmp_path / "run.py"
    script.write_text("""import sys
from pathlib import Path
from learning.history19.data import load
from learning.history19.models import ModelSpec
from learning.history19.training import run_fit
from learning.optimization import policy_for
phase,name,manifest,out,resume=sys.argv[1:]
a,scaler,binding=load(manifest); binding['sources']='synthetic-history19'
run_fit(ModelSpec(name,8),a['normal_train'],a['normal_validation'],out,binding=binding,
seed=0,batch_size=2,epochs=4,patience=5,learning_rate=.01,optimization=policy_for('sgd'),
resume=None if resume=='-' else resume,stop_after_epoch=2 if phase=='partial' else None)
""")
    for phase in ("full", "partial", "resume"):
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(script),
                phase,
                name,
                str(manifest),
                str(tmp_path / phase),
                str(tmp_path / "partial/last.pt") if phase == "resume" else "-",
            ],
            check=True,
            timeout=45,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )
    full = torch.load(tmp_path / "full/last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(tmp_path / "resume/last.pt", map_location="cpu", weights_only=False)
    assert full["config"]["optimization"]["optimizer"] == "sgd"
    assert full["completed_optimizer_steps"] == resumed["completed_optimizer_steps"] == 8
    for key, value in full["model"].items():
        assert torch.equal(value, resumed["model"][key])
    assert torch.equal(full["rng"]["cpu"], resumed["rng"]["cpu"])
    for key in ("train_loss", "validation_loss"):
        assert [r[key] for r in full["history"]] == [r[key] for r in resumed["history"]]
    assert (tmp_path / "full/model.safetensors").read_bytes() == (
        tmp_path / "resume/model.safetensors"
    ).read_bytes()
    arrays, scaler, _ = data.load(manifest)
    seal_export(tmp_path / "full", scaler)
    restored = Predictor(tmp_path / "full")
    expected = make_model(ModelSpec(name, 8)).eval()
    expected.load_state_dict(full["best_state"])
    with backend("cpu"), torch.no_grad():
        reference = expected(torch.as_tensor(arrays["normal_validation"]["X"])).numpy()
    np.testing.assert_allclose(
        restored.predict_scaled(arrays["normal_validation"]["X"]), reference, atol=1e-6, rtol=1e-6
    )


def test_no_Adam_no_context_no_silent_GPU(tmp_path):
    a = {"X": np.zeros((4, 100, 19)), "Y": np.zeros((4, 10, 4))}
    with pytest.raises(ValueError, match="SGD only"):
        run_fit(
            ModelSpec("lstm", 8),
            a,
            a,
            tmp_path / "adam",
            binding=dict(data="fixture", scaler="fixture", sources="fixture"),
            optimization=policy_for("adam"),
        )
    with pytest.raises(ValueError, match="context"):
        Batches({**a, "C": np.zeros((4, 6))}, 2, "cpu")
    with pytest.raises(ValueError, match="explicit GPU"):
        run_fit(ModelSpec("lstm", 8), a, a, tmp_path / "gpu", device="cuda")
    with pytest.raises(ValueError, match="approve-training"):
        runner.fit(tmp_path / "absent.json", tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_new_CLI_help_without_profile():
    result = subprocess.run(
        [str(ROOT / "ai/drone-ai"), "train", "history19", "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "validate" in result.stdout and "preprocess" in result.stdout


def test_finite_three_rate_selection_is_SGD_only(tmp_path, monkeypatch):
    cfg = runner.config()
    calls = []

    def fixture_fit(spec, train, validation, folder, **kwargs):
        assert set(train) == set(validation) == {"X", "Y"}
        assert kwargs["optimization"] == policy_for("sgd") and kwargs["epochs"] == 3
        calls.append((spec.name, kwargs["learning_rate"]))
        folder.mkdir(parents=True)
        (folder / "initialization.json").write_text(
            json.dumps({"state_sha256": spec.name + str(spec.hidden) + str(spec.layers)})
        )
        rows = [{"validation_loss": abs(kwargs["learning_rate"] - 0.01)}] * 3
        (folder / "history.json").write_text(json.dumps(rows))
        return rows

    monkeypatch.setattr(runner, "run_fit", fixture_fit)
    arrays = {"X": np.zeros((4, 100, 19)), "Y": np.zeros((4, 10, 4))}
    record = runner.select_rates(cfg, tmp_path, arrays, arrays, {"data": "fixture"}, None)
    assert len(calls) == 24 and len(record["models"]) == 8
    assert all(v["rate"] == 0.01 for v in record["models"].values())
    assert not record["development_data_used"] and not record["probe_weights_reused_for_main"]


def test_bounded_CPU_benchmark_uses_new_X_only_model(tmp_path):
    from learning.history19.benchmark import measure

    arrays = {"X": np.zeros((4, 100, 19)), "Y": np.zeros((4, 10, 4))}
    result = measure(
        ModelSpec("lstm", 8),
        arrays,
        [2],
        tmp_path / "benchmark",
        device="cpu",
        warmup=1,
        steps=3,
        optimization=policy_for("sgd"),
    )
    assert result[0]["passed"]


def test_raw_file_replay_is_synthetic_and_source_is_unchanged(tmp_path):
    from test_preprocessing import CONFIG, META, events

    from learning.history19.preprocessing import replay

    run = tmp_path / "synthetic-raw"
    (run / "observations").mkdir(parents=True)
    rows = list(events(20))
    (run / "run.json").write_text(
        json.dumps(dict(run_id=META["run_id"], origin="synthetic_fixture", status="completed"))
    )
    (run / "observations/metadata.json").write_text(json.dumps(META))
    stream = run / "observations/streams.jsonl"
    stream.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "observations/watermarks.jsonl").write_text(
        json.dumps(
            dict(
                schema="drone.raw.watermark/2.0",
                run_id=META["run_id"],
                clock_id="px4-boot",
                recorded_events=len(rows),
                through_available_time_us=20_000_000,
                closed=True,
            )
        )
        + "\n"
    )
    before = sha(stream)
    report = replay(
        run,
        tmp_path / "preprocessed",
        {**CONFIG, "feature_profile": "common19_ctx0", "context_us": 0},
    )
    assert report["closed"] and report["windows"] == 11 and not report["live_verified"]
    assert sha(stream) == before
