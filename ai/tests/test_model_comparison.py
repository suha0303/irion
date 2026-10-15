"""Synthetic CPU/data and mocked scheduler checks; never submits a real GPU job."""

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from execution.common import read, sha, write
from learning import comparison, lifecycle, operator, prepare_compare
from learning.training import Batches


class Progress:
    def check(self):
        pass

    def update(self, **kw):
        pass


def server_profile(root, monkeypatch):
    profile = root / "server-profile.json"
    write(
        profile,
        dict(
            schema="drone-ai.profile/1",
            name="fixture",
            project_root=str(root),
            python=sys.executable,
            runtime_profile="torch21",
            dataset_root=str(root / "dataset"),
            output_root=str(root / "results"),
            ssh=None,
            slurm=dict(
                partition="p02", node="DIS01", nodes=1, ntasks=1, gpus=1, cpus=4, memory_gib=16
            ),
        ),
    )
    write(root / "prepared.json", dict(server_profile_sha256=sha(profile)))
    monkeypatch.setenv("DRONE_AI_PROFILE", str(profile))
    return profile


def config(approved=True):
    return dict(
        revision=comparison.REVISION,
        variants=copy.deepcopy(comparison.VARIANTS),
        models=list(comparison.MODELS),
        seed=0,
        batch_size=256,
        epochs=60,
        early_stopping_patience=10,
        learning_rate=0.001,
        dropout=0.1,
        weight_decay=0.00001,
        model_revision="explicit_dropout_v1",
        history=100,
        future=10,
        step_us=100000,
        precision="float32",
        amp=False,
        retries=0,
        final_test_used=False,
        gate_epochs=4,
        gate_windows=512,
        memory_gib=64,
        cpus=4,
        training_stride_us=1000000,
        next_round_authorized=False,
        models_sequential=True,
        gpu_jobs_max=1,
        development_counts=dict(normal_dev_eval=128, anomaly_dev=144),
        production_training_authorized=approved,
        memory=dict(estimated_host_peak_bytes=12 * 1024**3),
        wall_hours=48,
        evaluation_hours=12,
        queue_hours=24,
    )


def test_fixed_eight_variants_seed0_and_approval():
    cfg = config()
    lifecycle.config_check(cfg)
    assert len(lifecycle.neural_ids(cfg)) == 8 and len(lifecycle.model_ids(cfg)) == 10
    for v in cfg["variants"]:
        spec = lifecycle.spec_for(cfg, v["id"])
        assert (spec.name, spec.hidden, spec.layers, spec.heads) == (
            v["name"],
            v["hidden"],
            v["layers"],
            v["heads"],
        )
    for key, value in (
        ("seed", 17),
        ("seed", False),
        ("epochs", 61),
        ("retries", 1),
        ("final_test_used", True),
        ("models", comparison.MODELS[:-1]),
        ("memory_gib", 16),
        ("variants", []),
    ):
        changed = {**cfg, key: value}
        with pytest.raises(ValueError):
            lifecycle.config_check(changed)
    with pytest.raises(ValueError, match="approved"):
        comparison.config_check(config(False))
    comparison.config_check(config(False), require_approval=False)
    with pytest.raises(MemoryError):
        comparison.config_check({**cfg, "memory": dict(estimated_host_peak_bytes=64 * 1024**3)})


def test_r5_still_rejects_seed0():
    cfg = dict(
        revision="ROUND_001/R5_H128_B256_S17",
        models=["persistence", "ridge_forecast", "lstm", "transformer"],
        seed=17,
        hidden=128,
        layers=2,
        heads=4,
        batch_size=256,
        epochs=60,
        early_stopping_patience=10,
        learning_rate=0.001,
        precision="float32",
        amp=False,
        retries=0,
        final_test_used=False,
        gate_epochs=4,
        gate_windows=512,
        production_training_authorized=True,
    )
    lifecycle.config_check(cfg)
    with pytest.raises(ValueError, match="R5"):
        lifecycle.config_check({**cfg, "seed": 0})


def test_mmap_normalization_matches_legacy_and_avoids_whole_copy(tmp_path):
    rng = np.random.default_rng(3)
    arrays = dict(
        X=rng.normal(size=(7, 100, 19)), C=rng.normal(size=(7, 6)), Y=rng.normal(size=(7, 10, 4))
    )
    scaler = {
        k + suffix: np.arange(n) * 0.1 + (1 if suffix == "_scale" else 0)
        for k, n in (("x", 19), ("c", 6), ("y", 4))
        for suffix in ("_mean", "_scale")
    }
    path = tmp_path / "normal_train.npz"
    np.savez_compressed(path, **arrays)
    write(tmp_path / "manifest.json", dict(hashes={path.name: sha(path)}))
    result = comparison.normalized_mmap(
        tmp_path, "normal_train", scaler, tmp_path / "cache", Progress(), chunk=3
    )
    batches = Batches(result, 3, "cpu")
    for key, scale in (("X", "x"), ("C", "c"), ("Y", "y")):
        expected = ((arrays[key] - scaler[scale + "_mean"]) / scaler[scale + "_scale"]).astype(
            "float32"
        )
        np.testing.assert_array_equal(result[key], expected)
        assert np.shares_memory(result[key], batches.data[key].numpy())
    for actual, key in zip(batches.get([6, 0, 2]), ("X", "C", "Y")):
        np.testing.assert_array_equal(actual.numpy(), result[key][[6, 0, 2]])
    with pytest.raises(FileExistsError):
        comparison.normalized_mmap(tmp_path, "normal_train", scaler, tmp_path / "cache", Progress())
    with pytest.raises(ValueError):
        comparison.normalized_mmap(tmp_path, "anomaly_dev", scaler, tmp_path / "bad", Progress())
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        comparison.normalized_mmap(tmp_path, "normal_train", scaler, tmp_path / "bad", Progress())


def test_batch_finite_check_includes_later_chunks():
    arrays = dict(
        X=np.zeros((514, 100, 19), dtype="float32"),
        C=np.zeros((514, 6), dtype="float32"),
        Y=np.zeros((514, 10, 4), dtype="float32"),
    )
    arrays["Y"][-1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        Batches(arrays, 256, "cpu")


def test_exact_eight_gates_no_extra_model_or_seed(tmp_path, monkeypatch):
    cfg = config()
    measured, children = [], []

    def measure(spec, arrays, batches, folder, **kw):
        measured.append((spec, kw["seed"], batches))
        return [dict(passed=True)]

    def child(argv, **kw):
        name = argv[argv.index("--model") + 1]
        phase = argv[argv.index("--phase") + 1]
        children.append((name, phase))
        target = tmp_path / "gates" / (name + "-" + phase)
        target.mkdir()
        torch.save(
            dict(
                model={},
                optimizer={},
                rng={},
                best=1,
                best_epoch=0,
                best_state={},
                history=[dict(train_loss=1, validation_loss=1)],
            ),
            target / "last.pt",
        )
        (target / "model.safetensors").write_bytes(name.encode())

    monkeypatch.setattr(lifecycle, "measure", measure)
    monkeypatch.setattr(lifecycle.subprocess, "run", child)
    arrays = dict(X=np.zeros((512, 100, 19)), C=np.zeros((512, 6)), Y=np.zeros((512, 10, 4)))
    lifecycle.gates(tmp_path, cfg, tmp_path, arrays, arrays, Progress())
    assert len(measured) == 8 and all(
        seed == 0 and batches == [256] for _, seed, batches in measured
    )
    assert children == [
        (v["id"], p) for v in comparison.VARIANTS for p in ("full", "partial", "resume")
    ]
    assert set(read(tmp_path / "gates/passed.json")["proofs"]) == set(lifecycle.neural_ids(cfg))


def test_r6_submission_budget_and_no_submission_before_approval(tmp_path, monkeypatch):
    (tmp_path / "input").mkdir()
    write(tmp_path / "input/inventory.json", {})
    server_profile(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "LOCK", tmp_path / "project.lock")
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    calls = []
    monkeypatch.setattr(operator, "command", lambda argv: calls.append(argv) or "123")
    with pytest.raises(ValueError, match="approved"):
        operator.submit(tmp_path, "fit", config(False))
    assert not calls and not (tmp_path / "fit-submitted.claim").exists()
    operator.submit(tmp_path, "fit", config())
    assert all(
        v in calls[0]
        for v in (
            "--mem=64G",
            "--time=48:00:00",
            "--partition=p02",
            "--nodelist=DIS01",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
        )
    )
    with pytest.raises(FileExistsError):
        operator.submit(tmp_path, "fit", config())
    assert len(calls) == 1


def test_authorization_is_explicit_one_time_no_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_compare, "external_root", lambda p: p.resolve())
    write(tmp_path / "input/config.json", config(False))
    write(tmp_path / "input/inventory.json", {"config.json": sha(tmp_path / "input/config.json")})
    write(tmp_path / "prepared.json", {})
    with pytest.raises(ValueError):
        prepare_compare.authorize(tmp_path, False)
    prepare_compare.authorize(tmp_path, True)
    assert read(tmp_path / "input/config.json")["production_training_authorized"]
    assert not (tmp_path / "controller.claim").exists()
    with pytest.raises(ValueError):
        prepare_compare.authorize(tmp_path, True)


def test_remote_controller_requires_tmux_before_claim(tmp_path, monkeypatch):
    server_profile(tmp_path, monkeypatch)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    write(tmp_path / "input/config.json", config())
    with pytest.raises(ValueError, match="tmux"):
        operator.control(tmp_path)
    assert not (tmp_path / "controller.claim").exists()


def test_revision_display_and_remote_scope(tmp_path):
    write(tmp_path / "input/config.json", config(False))
    shown = operator.render(tmp_path)
    assert "seed0" in shown and "1200run" in shown and "10종" in shown
    for path in (
        "/",
        "relative/attempt",
        "/srv/drone-ai/../other",
        "/srv/drone-ai/bad$name",
    ):
        with pytest.raises(ValueError):
            prepare_compare.remote_path(path)
    assert prepare_compare.remote_path("/srv/drone-ai/comparison").name == "comparison"
    assert prepare_compare.remote_path("/tmp/another-attempt").name == "another-attempt"


@pytest.mark.parametrize("bad", [dict(seed=17), dict(config_sha256="changed")])
def test_output_seed_config_binding_checked_before_truth_join(tmp_path, monkeypatch, bad):
    cfg = {**config(), "upstream": str(tmp_path / "upstream")}
    write(tmp_path / "input/config.json", cfg)
    write(tmp_path / "input/inventory.json", {})
    result = dict(seed=0, config_sha256=sha(tmp_path / "input/config.json"))
    write(tmp_path / "compute/result-seal.json", {**result, **bad})
    monkeypatch.setattr(lifecycle, "check", lambda *a, **kw: {})
    monkeypatch.setattr(lifecycle.signal, "signal", lambda *a: None)
    with pytest.raises(ValueError, match="binding mismatch before truth"):
        lifecycle.evaluate_worker(tmp_path)
    assert not (tmp_path / "evaluation").exists()
    assert read(tmp_path / "evaluation-progress.json")["status"] == "FAILED_STOPPED"


def test_invalid_budget_stops_before_restore(tmp_path, monkeypatch):
    profile = server_profile(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="hours"):
        prepare_compare.prepare(SimpleNamespace(wall_hours=0, evaluation_hours=12, profile=profile))


def test_model_ids_do_not_reuse_existing_weights(tmp_path):
    cfg = config()
    ids = lifecycle.neural_ids(cfg)
    assert len(set(ids)) == 8
    for name in ids:
        spec = lifecycle.spec_for(cfg, name)
        model = lifecycle.make_model(spec)
        assert sum(p.numel() for p in model.parameters()) > 100_000


@pytest.fixture
def admitted(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows = []
    for role, count in comparison.COUNTS.items():
        for i in range(count):
            rid = role + str(i)
            rows.append(
                dict(
                    run_id=rid,
                    role=role,
                    seed=len(rows),
                    group_id=rid,
                    paired_case_id=rid,
                    raw_flight_id=rid,
                )
            )
    for role in ("normal_train", "normal_validation", "normal_calibration"):
        ids = [r["run_id"] for r in rows if r["role"] == role]
        n = max(512, len(ids))
        np.savez_compressed(
            dataset / (role + ".npz"),
            X=np.zeros((n, 100, 19)),
            C=np.zeros((n, 6)),
            Y=np.zeros((n, 10, 4)),
            run_id=np.array([ids[i % len(ids)] for i in range(n)]),
        )
    np.savez(dataset / "scaler.npz", x_mean=np.zeros(19))
    write(
        dataset / "manifest.json",
        dict(runs=rows, hashes={p.name: sha(p) for p in dataset.glob("*.npz")}),
    )
    write(tmp_path / "evidence/frozen-manifest.json", dict(status="DATA_FROZEN", runs=rows))
    write(
        tmp_path / "evidence/quality.json",
        dict(passed=True, final_test_used=False, train_only_scaler_verified=True),
    )
    write(tmp_path / "evidence/dataset-seal.json", {p.name: sha(p) for p in dataset.iterdir()})
    h = dict(training_eligible=True, run_count=1200, scaler_scope="all_selected_normal_train")
    for name, key in (
        ("quality.json", "quality_sha256"),
        ("dataset-seal.json", "dataset_seal_sha256"),
        ("frozen-manifest.json", "frozen_manifest_sha256"),
    ):
        h[key] = sha(tmp_path / "evidence" / name)
    write(tmp_path / "evidence/handoff.json", h)
    return tmp_path


def test_portable_admission_never_uses_retired_paths(admitted):
    rows, seal, headers = comparison.admission(admitted)
    assert len(rows) == 1200 and len(headers) == 3
    memory = comparison.memory_plan(headers)
    assert memory["primal_matrix_bytes"] == 1907**2 * 8
    assert memory["requested_host_bytes"] == 64 * 1024**3
    (admitted / "dataset/scaler.npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="scaler"):
        comparison.admission(admitted)


def test_cross_split_lineage_rejected_even_with_resealed_manifest(admitted):
    manifest = read(admitted / "dataset/manifest.json")
    manifest["runs"][720]["seed"] = manifest["runs"][0]["seed"]
    write(admitted / "dataset/manifest.json", manifest)
    write(
        admitted / "evidence/frozen-manifest.json",
        dict(status="DATA_FROZEN", runs=manifest["runs"]),
    )
    seal = read(admitted / "evidence/dataset-seal.json")
    seal["manifest.json"] = sha(admitted / "dataset/manifest.json")
    write(admitted / "evidence/dataset-seal.json", seal)
    h = read(admitted / "evidence/handoff.json")
    h.update(
        dataset_seal_sha256=sha(admitted / "evidence/dataset-seal.json"),
        frozen_manifest_sha256=sha(admitted / "evidence/frozen-manifest.json"),
    )
    write(admitted / "evidence/handoff.json", h)
    with pytest.raises(ValueError, match="lineage"):
        comparison.admission(admitted)


def test_seed0_continuous_and_resume_match_with_variant_progress(tmp_path):
    from learning.diagnostic import same
    from learning.training import run_fit

    rng = np.random.default_rng(7)
    arrays = dict(
        X=rng.normal(size=(3, 100, 19)), C=rng.normal(size=(3, 6)), Y=rng.normal(size=(3, 10, 4))
    )
    spec = lifecycle.spec_for(config(), "lstm_h64_l2_s0")
    seen = []
    progress = SimpleNamespace(check=lambda: None, update=lambda **kw: seen.append(kw))
    kwargs = dict(
        binding=dict(data="synthetic", scaler="synthetic", sources="synthetic"),
        seed=0,
        batch_size=2,
        epochs=2,
        patience=10,
        progress=progress,
        model_id="lstm_h64_l2_s0",
    )
    run_fit(spec, arrays, arrays, tmp_path / "full", **kwargs)
    run_fit(spec, arrays, arrays, tmp_path / "partial", stop_after_epoch=1, **kwargs)
    run_fit(
        spec, arrays, arrays, tmp_path / "resumed", resume=tmp_path / "partial/last.pt", **kwargs
    )
    full = torch.load(tmp_path / "full/last.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert full["config"]["seed"] == 0
    for key in ("model", "optimizer", "rng", "best", "best_epoch", "best_state"):
        assert same(full[key], resumed[key])
    assert sha(tmp_path / "full/model.safetensors") == sha(tmp_path / "resumed/model.safetensors")
    assert any(row.get("model") == "lstm_h64_l2_s0" for row in seen)


def test_complete_local_prepare_restores_only_fixture_and_never_submits(
    admitted, tmp_path, monkeypatch
):
    from execution.common import ROOT
    from pretrain.storage import pack_files

    frozen = read(admitted / "evidence/frozen-manifest.json")
    frozen["revision_lock"] = dict(
        preprocessing_sha256=sha(
            ROOT / "Data/generation/pretrain/resources/preprocessing/preprocessing.json"
        )
    )
    for row in frozen["runs"]:
        if row["role"] in {"normal_dev_eval", "anomaly_dev"}:
            import hashlib

            row["archive"] = dict(
                members={
                    "evaluation/events.jsonl": dict(bytes=0, sha256=hashlib.sha256(b"").hexdigest())
                }
            )
    write(admitted / "evidence/frozen-manifest.json", frozen)
    manifest = read(admitted / "dataset/manifest.json")
    manifest["runs"] = frozen["runs"]
    write(admitted / "dataset/manifest.json", manifest)
    seal = read(admitted / "evidence/dataset-seal.json")
    seal["manifest.json"] = sha(admitted / "dataset/manifest.json")
    write(admitted / "evidence/dataset-seal.json", seal)
    h = read(admitted / "evidence/handoff.json")
    h.update(
        dataset_seal_sha256=sha(admitted / "evidence/dataset-seal.json"),
        frozen_manifest_sha256=sha(admitted / "evidence/frozen-manifest.json"),
    )
    write(admitted / "evidence/handoff.json", h)
    bundle = tmp_path / "portable"
    bundle.mkdir()
    members = {
        str(p.relative_to(admitted)): p
        for folder in ("dataset", "evidence")
        for p in (admitted / folder).rglob("*")
        if p.is_file()
    }
    data_ref = pack_files(members, bundle / "dataset.tar.xz", Progress())
    empty = tmp_path / "empty-events.jsonl"
    empty.write_bytes(b"")
    truth_ref = pack_files(
        {
            r["run_id"] + ".jsonl": empty
            for r in frozen["runs"]
            if r["role"] in {"normal_dev_eval", "anomaly_dev"}
        },
        bundle / "truth.tar.xz",
        Progress(),
    )
    for ref in (data_ref, truth_ref):
        ref["path"] = ref["path"].split("/")[-1]
    write(
        bundle / "receipt.json",
        dict(
            status="PORTABLE_DATASET_VERIFIED",
            final_test_included=False,
            dataset=data_ref,
            evaluator_truth=truth_ref,
            uncompressed_dataset_bytes=sum(v["bytes"] for v in data_ref["members"].values()),
        ),
    )
    fake_home = tmp_path / "home"
    write(
        fake_home / "comparison-plan.json",
        dict(
            dataset=dict(
                receipt_sha256=sha(bundle / "receipt.json"), archive_sha256=data_ref["sha256"]
            )
        ),
    )
    monkeypatch.setattr(prepare_compare, "HOME", fake_home)
    monkeypatch.setattr(prepare_compare, "external_root", lambda p: p.resolve())
    monkeypatch.setattr(
        prepare_compare.shutil, "disk_usage", lambda p: SimpleNamespace(free=1024**4)
    )
    monkeypatch.setattr(
        prepare_compare.subprocess,
        "run",
        lambda *a, **k: pytest.fail("no external process during prepare"),
    )
    root = tmp_path / "new-preparation"
    profile = server_profile(tmp_path, monkeypatch)
    args = SimpleNamespace(
        profile=profile,
        root=root,
        bundle=bundle,
        remote_root="/srv/drone-ai/comparison-fixture",
        wall_hours=48,
        evaluation_hours=12,
    )
    prepare_compare.prepare(args)
    cfg = read(root / "input/config.json")
    comparison.config_check(cfg, require_approval=False)
    assert cfg["seed"] == 0 and len(cfg["models"]) == 10
    assert not cfg["production_training_authorized"]
    assert (
        sha(root / "server-profile.json") == read(root / "prepared.json")["server_profile_sha256"]
    )
    assert (root / "input/ai/console/profile.py").is_file()
    assert len(read(root / "truth/index.json")) == 272
    assert read(root / "prepared.json")["submitted_jobs"] == []
    assert not (root / "controller.claim").exists()
    assert sha(root / "upstream/dataset/manifest.json") == seal["manifest.json"]
    with pytest.raises(FileExistsError):
        prepare_compare.prepare(args)
