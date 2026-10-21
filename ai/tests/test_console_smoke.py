"""Synthetic CPU and sealed-sample checks; never submit a GPU job or train real data."""

import copy
from pathlib import Path

import numpy as np
import pytest
from console import smoke
from execution.common import Progress, read, sha, write


def fixture(upstream):
    dataset, evidence = upstream / "dataset", upstream / "evidence"
    dataset.mkdir(parents=True)
    evidence.mkdir()
    rng = np.random.default_rng(37)
    arrays, rows = {}, []
    for index, role in enumerate(("normal_train", "normal_validation")):
        run_id = role + "-fixture"
        values = dict(
            X=rng.normal(size=(7, 100, 19)),
            C=rng.normal(size=(7, 6)),
            Y=rng.normal(size=(7, 10, 4)),
            run_id=np.array([run_id] * 7),
        )
        arrays[role] = values
        np.savez_compressed(dataset / (role + ".npz"), **values)
        rows.append(
            dict(
                run_id=run_id,
                role=role,
                seed=index,
                group_id=str(index),
                paired_case_id=str(index),
                raw_flight_id=str(index),
            )
        )
    scaler = {
        key + suffix: np.full(width, 2 if suffix == "_scale" else 0.3)
        for key, width in (("x", 19), ("c", 6), ("y", 4))
        for suffix in ("_mean", "_scale")
    }
    np.savez(dataset / "scaler.npz", **scaler)
    write(
        dataset / "manifest.json",
        dict(
            runs=rows,
            hashes={
                name: sha(dataset / name) for name in ("normal_train.npz", "normal_validation.npz")
            },
        ),
    )
    write(
        evidence / "quality.json",
        dict(
            status="PRETRAIN_QUALITY_PASSED",
            passed=True,
            final_test_used=False,
            train_only_scaler_verified=True,
        ),
    )
    write(evidence / "frozen-manifest.json", dict(status="DATA_FROZEN", runs=rows))
    write(evidence / "dataset-seal.json", {p.name: sha(p) for p in dataset.iterdir()})
    handoff = dict(training_eligible=True, scaler_scope="all_selected_normal_train", run_count=2)
    for name, key in (
        ("quality.json", "quality_sha256"),
        ("frozen-manifest.json", "frozen_manifest_sha256"),
        ("dataset-seal.json", "dataset_seal_sha256"),
    ):
        handoff[key] = sha(evidence / name)
    write(evidence / "handoff.json", handoff)
    return arrays


def reseal(upstream):
    dataset, evidence = upstream / "dataset", upstream / "evidence"
    write(evidence / "dataset-seal.json", {p.name: sha(p) for p in dataset.iterdir()})
    handoff = read(evidence / "handoff.json")
    for name, key in (
        ("quality.json", "quality_sha256"),
        ("frozen-manifest.json", "frozen_manifest_sha256"),
        ("dataset-seal.json", "dataset_seal_sha256"),
    ):
        handoff[key] = sha(evidence / name)
    write(evidence / "handoff.json", handoff)


def cpu_input(root):
    root.mkdir()
    rng = np.random.default_rng(19)
    arrays = {
        prefix + "_" + key: rng.normal(size=(4, *shape))
        for prefix in ("train", "validation")
        for key, shape in (("X", (100, 19)), ("C", (6,)), ("Y", (10, 4)))
    }
    np.savez(root / "inputs.npz", **arrays)
    cfg = dict(
        schema="drone-ai.smoke/1",
        device="cpu",
        synthetic_cpu_fixture=True,
        inputs_sha256=sha(root / "inputs.npz"),
        binding=dict(
            data="synthetic_cpu_fixture",
            scaler="synthetic_cpu_fixture",
            sources=smoke._source_hashes(),
        ),
    )
    write(root / "smoke.json", cfg)
    return cfg


def test_bounded_sealed_normal_sample_without_full_role_load(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    arrays = fixture(upstream)
    before = {str(p.relative_to(upstream)): sha(p) for p in upstream.rglob("*") if p.is_file()}
    original = smoke.np.load

    def load(path, **kwargs):
        assert Path(path).name == "scaler.npz", "whole-role np.load is forbidden in smoke sampler"
        return original(path, **kwargs)

    monkeypatch.setattr(smoke.np, "load", load)
    report = smoke.sample_input(upstream, tmp_path / "sample.npz")
    assert report["final_test_used"] is report["truth_used"] is report["scaler_fitted"] is False
    assert report["throughput_measured"] is False
    assert report["roles"]["normal_train"]["sampled_windows"] == 4
    assert report["roles"]["normal_train"]["source_windows"] == 7
    with original(tmp_path / "sample.npz", allow_pickle=False) as sample:
        for prefix, role in (("train", "normal_train"), ("validation", "normal_validation")):
            for key in ("X", "C", "Y"):
                np.testing.assert_array_equal(
                    sample[prefix + "_" + key], (arrays[role][key][:4] - 0.3) / 2
                )
    assert before == {
        str(p.relative_to(upstream)): sha(p) for p in upstream.rglob("*") if p.is_file()
    }


def test_sample_rejects_changed_archive_and_preserves_it(tmp_path):
    upstream = tmp_path / "upstream"
    fixture(upstream)
    path = upstream / "dataset/normal_train.npz"
    with path.open("ab") as out:
        out.write(b"modified")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="archive changed"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")
    assert path.read_bytes() == before
    assert not (tmp_path / "sample.npz").exists()


def test_sample_rejects_cross_split_lineage(tmp_path):
    upstream = tmp_path / "upstream"
    fixture(upstream)
    manifest = read(upstream / "dataset/manifest.json")
    manifest["runs"][1]["seed"] = manifest["runs"][0]["seed"]
    write(upstream / "dataset/manifest.json", manifest)
    write(
        upstream / "evidence/frozen-manifest.json",
        dict(status="DATA_FROZEN", runs=manifest["runs"]),
    )
    reseal(upstream)
    with pytest.raises(ValueError, match="lineage leakage"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")


@pytest.mark.parametrize(
    "field,value",
    [("final_test_used", True), ("train_only_scaler_verified", False), ("passed", False)],
)
def test_sample_rejects_unapproved_quality(tmp_path, field, value):
    upstream = tmp_path / "upstream"
    fixture(upstream)
    quality = read(upstream / "evidence/quality.json")
    quality[field] = value
    write(upstream / "evidence/quality.json", quality)
    reseal(upstream)
    with pytest.raises(ValueError, match="normal-only"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")


def test_sample_rejects_symlinked_archive(tmp_path):
    upstream = tmp_path / "upstream"
    fixture(upstream)
    path = upstream / "dataset/normal_train.npz"
    moved = tmp_path / "moved.npz"
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError, match="Symlink"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")


def test_sample_rejects_zero_scale(tmp_path):
    upstream = tmp_path / "upstream"
    fixture(upstream)
    path = upstream / "dataset/scaler.npz"
    with np.load(path, allow_pickle=False) as archive:
        scaler = dict(archive)
    scaler["x_scale"][0] = 0
    np.savez(path, **scaler)
    reseal(upstream)
    with pytest.raises(ValueError, match="Nonpositive"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")


def test_public_child_has_no_implicit_cpu_or_gpu_approval(tmp_path):
    root = tmp_path / "trial"
    cfg = cpu_input(root)
    cfg["synthetic_cpu_fixture"] = False
    write(root / "smoke.json", cfg)
    with pytest.raises(ValueError, match="synthetic"):
        smoke._child(root, "lstm", "full")
    cfg["device"] = "cuda"
    write(root / "smoke.json", cfg)
    with pytest.raises(ValueError, match="profile changed"):
        smoke._child(root, "lstm", "full")
    assert not (root / "lstm-full").exists()


def test_changed_sources_stop_before_optimizer(tmp_path):
    root = tmp_path / "trial"
    cfg = cpu_input(root)
    cfg["binding"]["sources"] = {"wrong": "wrong"}
    write(root / "smoke.json", cfg)
    with pytest.raises(ValueError, match="binding changed"):
        smoke._child(root, "lstm", "full")
    assert not (root / "lstm-full").exists()


def test_cpu_fresh_process_exact_save_restore_both_models(tmp_path):
    root = tmp_path / "trial"
    cfg = cpu_input(root)
    proofs = smoke.restore_gate(root, cfg, Progress(root / "progress.json", hours=0.1))
    assert list(proofs) == ["lstm", "transformer"]
    assert all(all(row["checks"].values()) for row in proofs.values())
    assert sum(row["optimizer_steps_executed"] for row in proofs.values()) == 16
    for name in smoke.MODELS:
        assert (root / f"{name}-full/model.safetensors").is_file()
        memory = read(root / f"{name}-resume-memory.json")
        assert memory["training_eligible"] is False
        assert memory["peak_allocated_bytes"] is None
    with pytest.raises(FileExistsError):
        smoke.restore_gate(root, cfg, Progress(root / "second.json"))


def test_bad_child_scope_and_phase_preserves_prior_files(tmp_path):
    root = tmp_path / "trial"
    cfg = cpu_input(root)
    before = copy.deepcopy(read(root / "smoke.json"))
    for model, phase in (("ridge", "full"), ("lstm", "retry")):
        with pytest.raises(ValueError, match="Unknown finite"):
            smoke._child(root, model, phase)
    assert cfg == before == read(root / "smoke.json")


def test_worker_allocation_failure_preserved_before_data_or_model(tmp_path, monkeypatch):
    from console import profile

    output = tmp_path / "output"
    root = output / "attempt/compute"
    profile_file = tmp_path / "profile.json"
    profile_file.write_text("profile fixture")
    monkeypatch.setattr(
        profile,
        "load",
        lambda path: dict(
            project_root=str(Path(smoke.__file__).resolve().parents[2]),
            dataset_root=str(tmp_path / "immutable-input"),
            output_root=str(output),
        ),
    )
    calls = []

    def reject(*args):
        calls.append("allocation")
        raise ValueError("allocation not verified")

    monkeypatch.setattr(smoke, "_verify_runtime", reject)
    monkeypatch.setattr(smoke, "sample_input", lambda *args: calls.append("data"))
    with pytest.raises(ValueError, match="allocation not verified"):
        smoke.run(profile_file, root)
    assert calls == ["allocation"]
    result = read(root / "result.json")
    assert result["status"] == "FAILED_STOPPED"
    assert result["automatic_retry"] is result["training_eligible"] is False
    assert not (root / "inputs.npz").exists()
    assert not (root / "lstm-full").exists()


def legacy_fixture(upstream):
    import shutil

    fixture(upstream)
    shutil.rmtree(upstream / "evidence")
    manifest_path = upstream / "dataset/manifest.json"
    manifest = read(manifest_path)
    counts = dict(
        normal_train=63,
        normal_validation=9,
        normal_calibration=9,
        normal_dev_eval=11,
        anomaly_dev=12,
    )
    rows = []
    for role, count in counts.items():
        for index in range(count):
            identity = f"{role}-{index}"
            run_id = role + "-fixture" if index == 0 else identity
            rows.append(
                dict(
                    run_id=run_id,
                    role=role,
                    seed=len(rows),
                    group_id=identity,
                    raw_flight_id=identity,
                    paired_case_id=identity,
                )
            )
    manifest.update(
        runs=rows,
        unique_train_frames=10000,
        scaler_fit="normal_train unique valid airborne frames; context once per train origin",
    )
    write(manifest_path, manifest)
    seal = {p.name: sha(p) for p in (upstream / "dataset").iterdir()}
    cfg = dict(
        revision="ROUND_001/R4_GPU104_FOUR_SEED17",
        dataset_files=seal,
        handoff="/path/never/read/handoff.json",
        handoff_sha256="a" * 64,
        final_test_used=False,
        final_test_authorized=False,
    )
    write(upstream / "config.json", cfg)
    write(
        upstream / "inventory.json",
        {
            "config.json": sha(upstream / "config.json"),
            **{"dataset/" + name: digest for name, digest in seal.items()},
        },
    )


def test_legacy_snapshot_only_admits_tiny_functional_sample(tmp_path):
    upstream = tmp_path / "historical-input"
    legacy_fixture(upstream)
    before = {str(p.relative_to(upstream)): sha(p) for p in upstream.rglob("*") if p.is_file()}
    result = smoke.sample_input(upstream, tmp_path / "sample.npz")
    provenance = result["provenance"]
    assert provenance["kind"] == "legacy_admitted_104_snapshot_functionality_only"
    assert provenance["quality_evidence_reverified"] is False
    assert provenance["training_eligible"] is result["training_eligible"] is False
    assert "full quality evidence" in provenance["note"]
    assert provenance["original_handoff_sha256"] == "a" * 64
    assert before == {
        str(p.relative_to(upstream)): sha(p) for p in upstream.rglob("*") if p.is_file()
    }


def test_legacy_snapshot_requires_cross_bound_inventory(tmp_path):
    upstream = tmp_path / "historical-input"
    legacy_fixture(upstream)
    path = upstream / "inventory.json"
    inventory = read(path)
    inventory["dataset/scaler.npz"] = "b" * 64
    write(path, inventory)
    with pytest.raises(ValueError, match="inventory/config data seal mismatch"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")


def test_legacy_snapshot_rejects_unknown_revision(tmp_path):
    upstream = tmp_path / "historical-input"
    legacy_fixture(upstream)
    path = upstream / "config.json"
    cfg = read(path)
    cfg["revision"] = "unknown-history"
    write(path, cfg)
    inventory = read(upstream / "inventory.json")
    inventory["config.json"] = sha(path)
    write(upstream / "inventory.json", inventory)
    with pytest.raises(ValueError, match="snapshot schema"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")


def test_broken_full_evidence_never_falls_back_to_legacy(tmp_path):
    upstream = tmp_path / "historical-input"
    legacy_fixture(upstream)
    evidence = upstream / "evidence"
    evidence.mkdir()
    write(evidence / "handoff.json", {})
    with pytest.raises(FileNotFoundError):
        smoke.sample_input(upstream, tmp_path / "sample.npz")


def test_gpu_telemetry_closes_on_gate_failure_without_declaring_success(tmp_path, monkeypatch):
    from console import profile
    from learning import telemetry

    output = tmp_path / "output"
    root = output / "attempt/compute"
    profile_file = tmp_path / "profile.json"
    profile_file.write_text("profile fixture")
    monkeypatch.setattr(
        profile,
        "load",
        lambda path: dict(
            project_root=str(Path(smoke.__file__).resolve().parents[2]),
            dataset_root=str(tmp_path / "immutable-input"),
            output_root=str(output),
        ),
    )
    calls = []
    monkeypatch.setattr(smoke, "_verify_runtime", lambda *args: calls.append("allocation"))

    def sample(upstream, destination):
        destination.write_bytes(b"mock sample, no GPU data")
        return {"hashes": {"dataset/scaler.npz": "mock scaler"}}

    class MockTelemetry:
        def __init__(self, output):
            pass

        def start(self):
            calls.append("telemetry start")

        def stop(self):
            calls.append("telemetry stop")

    def fail(*args, **kwargs):
        calls.append("gate")
        raise ValueError("mock restore mismatch")

    monkeypatch.setattr(smoke, "sample_input", sample)
    monkeypatch.setattr(smoke, "restore_gate", fail)
    monkeypatch.setattr(telemetry, "Telemetry", MockTelemetry)
    with pytest.raises(ValueError, match="restore mismatch"):
        smoke.run(profile_file, root)
    assert calls == ["allocation", "telemetry start", "gate", "telemetry stop"]
    assert read(root / "result.json")["status"] == "FAILED_STOPPED"
    assert not (root / "restore-proof.json").exists()


def benchmark_fixture(upstream):
    """Match the Data runner's flat, test-only three-partition handoff layout."""
    fixture(upstream)
    dataset = upstream / "dataset"
    for path in (upstream / "evidence").iterdir():
        path.rename(upstream / path.name)
    (upstream / "evidence").rmdir()
    manifest = read(dataset / "manifest.json")
    rows = manifest["runs"]
    rng = np.random.default_rng(71)
    role = "normal_calibration"
    run_id = role + "-fixture"
    np.savez_compressed(
        dataset / (role + ".npz"),
        X=rng.normal(size=(7, 100, 19)),
        C=rng.normal(size=(7, 6)),
        Y=rng.normal(size=(7, 10, 4)),
        run_id=np.array([run_id] * 7),
    )
    rows.append(dict(run_id=run_id, role=role, seed=2, group_id="2"))
    for index, row in enumerate(rows):
        row.pop("paired_case_id", None)
        row.pop("raw_flight_id", None)
        row.update(
            case_id=f"probe-1-{index}",
            collection_role="benchmark",
            training_eligible=False,
            mission_seconds=120,
        )
    manifest["hashes"] = {p.name: sha(p) for p in dataset.glob("*.npz")}
    write(dataset / "manifest.json", manifest)
    write(
        upstream / "frozen-manifest.json",
        dict(status="BENCHMARK_ONLY", training_eligible=False, runs=rows, ledger_sha256="c" * 64),
    )
    quality = read(upstream / "quality.json")
    quality["training_admission"] = False
    write(upstream / "quality.json", quality)
    handoff = read(upstream / "handoff.json")
    handoff.update(
        status="BENCHMARK_ONLY_STOPPED",
        training_eligible=False,
        training_authorized=False,
        requires_new_training_package_admission=True,
        run_count=3,
    )
    write(upstream / "handoff.json", handoff)
    reseal_flat(upstream)


def reseal_flat(upstream):
    write(
        upstream / "dataset-seal.json", {p.name: sha(p) for p in (upstream / "dataset").iterdir()}
    )
    handoff = read(upstream / "handoff.json")
    for name, key in (
        ("quality.json", "quality_sha256"),
        ("frozen-manifest.json", "frozen_manifest_sha256"),
        ("dataset-seal.json", "dataset_seal_sha256"),
    ):
        handoff[key] = sha(upstream / name)
    write(upstream / "handoff.json", handoff)


def test_fresh_three_run_flat_benchmark_handoff_is_functionality_only(tmp_path):
    upstream = tmp_path / "new-flight-test"
    benchmark_fixture(upstream)
    before = {str(p.relative_to(upstream)): sha(p) for p in upstream.rglob("*") if p.is_file()}
    report = smoke.sample_input(upstream, tmp_path / "sample.npz")
    provenance = report["provenance"]
    assert provenance["kind"] == "fresh_three_run_benchmark_functionality_only"
    assert provenance["quality_evidence_reverified"] is True
    assert provenance["source_training_eligible"] is False
    assert provenance["main_training_admission"] is False
    assert provenance["lineage_scope"] == "benchmark case/group/seed/run IDs"
    assert "not fabricated" in provenance["note"]
    assert report["training_eligible"] is report["final_test_used"] is False
    assert report["roles"]["normal_train"]["sampled_windows"] == 4
    assert "handoff.json" in report["hashes"]
    assert before == {
        str(p.relative_to(upstream)): sha(p) for p in upstream.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("field", ["seed", "group_id", "case_id", "run_id"])
def test_benchmark_duplicate_identity_is_rejected(tmp_path, field):
    upstream = tmp_path / "new-flight-test"
    benchmark_fixture(upstream)
    frozen = read(upstream / "frozen-manifest.json")
    frozen["runs"][1][field] = frozen["runs"][0][field]
    write(upstream / "frozen-manifest.json", frozen)
    manifest = read(upstream / "dataset/manifest.json")
    manifest["runs"] = frozen["runs"]
    write(upstream / "dataset/manifest.json", manifest)
    reseal_flat(upstream)
    with pytest.raises(ValueError, match="identities must be independent"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")


def test_benchmark_incomplete_authorization_markers_rejected(tmp_path):
    upstream = tmp_path / "new-flight-test"
    benchmark_fixture(upstream)
    handoff = read(upstream / "handoff.json")
    handoff["training_authorized"] = True
    write(upstream / "handoff.json", handoff)
    with pytest.raises(ValueError, match="functionality-only"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")


def test_normal_full_handoff_does_not_inherit_benchmark_lineage_rules(tmp_path):
    upstream = tmp_path / "new-flight-test"
    benchmark_fixture(upstream)
    frozen = read(upstream / "frozen-manifest.json")
    frozen.update(status="DATA_FROZEN", training_eligible=True)
    write(upstream / "frozen-manifest.json", frozen)
    handoff = read(upstream / "handoff.json")
    handoff.update(status="PRETRAIN_READY_PENDING_USER_REVIEW", training_eligible=True)
    write(upstream / "handoff.json", handoff)
    reseal_flat(upstream)
    with pytest.raises(ValueError, match="Missing split lineage"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")


def test_plain_ineligible_handoff_is_not_promoted_to_smoke_benchmark(tmp_path):
    upstream = tmp_path / "normal-upstream"
    fixture(upstream)
    handoff = read(upstream / "evidence/handoff.json")
    handoff["training_eligible"] = False
    write(upstream / "evidence/handoff.json", handoff)
    with pytest.raises(ValueError, match="normal-only train-scaler"):
        smoke.sample_input(upstream, tmp_path / "sample.npz")
