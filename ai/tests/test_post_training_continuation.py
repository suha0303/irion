"""Stopped calibration evidence is reusable; development inference is always fresh."""

import copy
import os

import pytest

from execution.common import read, sha, write
from learning import operator
from learning import post_training as post


@pytest.fixture
def prior(tmp_path, monkeypatch):
    root = tmp_path / "previous"
    name = "lstm_h128_l2_s0"
    cfg = dict(
        source_training="/trained",
        source_training_config_sha256="training-cfg",
        source_training_seal_sha256="training-seal",
        upstream="/upstream",
        source_dataset_sha256="dataset",
        dataset_files={"manifest.json": "manifest"},
        model_files={"models/fixture": "model"},
        models=[name],
        seed=0,
        receipt_sha256="receipt",
        truth_archive={"path": "truth.tar.xz"},
        evaluation_plan_sha256="plan",
        protocol={"detector": {"ewma_alpha": 0.3}, "implementation_sha256": {"replay": "old"}},
    )
    release = dict(job_id="123", released=True, slurm_state="CANCELLED", exit_code="0:15", queue=[])
    for path in (
        "input/config.json",
        "input/inventory.json",
        "prepared.json",
        "server-profile.json",
        "authorization.json",
    ):
        write(root / path, {})
    write(root / "controller.json", {"status": "FAILED_STOPPED"})
    write(
        root / "calibrate-submission.json", dict(job_id="123", phase="calibrate", uid=os.getuid())
    )
    write(root / "release.json", release)
    model = root / "compute/models" / name
    write(model / "integrity.json", {"model.json": "frozen-model"})
    folder = root / "compute/calibration" / name
    write(
        folder / "calibration.json",
        {"high": [1.0, 2.0], "bundle_integrity_sha256": sha(model / "integrity.json")},
    )
    (folder / "normal-residuals.npz").write_bytes(b"exact normal residual fixture")
    write(
        folder / "normal-residuals.json",
        dict(
            schema="drone-ai.normal-residual-evidence/1",
            model=name,
            file="normal-residuals.npz",
            roles=["normal_validation", "normal_calibration"],
            truth_used=False,
            sha256=sha(folder / "normal-residuals.npz"),
            calibration_sha256=sha(folder / "calibration.json"),
            bundle_integrity_sha256=sha(model / "integrity.json"),
            dataset_manifest_sha256="manifest",
        ),
    )
    write(folder / "completed-old-run/seal.json", {"old": "development result must not copy"})
    write(folder / "partial-old-run/forecasts.jsonl", {"old": "partial remains untouched"})
    monkeypatch.setattr(post, "checked", lambda path: (path, cfg))
    monkeypatch.setattr(operator, "snapshot", lambda record: copy.deepcopy(release))
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    return root, cfg, name


def test_only_normal_calibration_copied_and_old_attempt_unchanged(prior, tmp_path):
    previous, cfg, name = prior
    before = {str(p.relative_to(previous)): sha(p) for p in previous.rglob("*") if p.is_file()}
    new_cfg = copy.deepcopy(cfg)
    new_cfg["protocol"]["implementation_sha256"]["replay"] = "optimized"
    proof = post.calibration_continuation(previous, new_cfg)
    output = tmp_path / "new"
    post.copy_normal_calibration(previous, output, proof)
    assert proof["all_model_replay_fresh"] is True
    assert proof["development_inference_reused"] is False
    folder = output / "compute/calibration" / name
    assert {p.name for p in folder.iterdir()} == {
        "calibration.json",
        "normal-residuals.json",
        "normal-residuals.npz",
    }
    assert read(folder / "calibration.json")["high"] == [1.0, 2.0]
    for relative, digest in proof["source_calibration_files"].items():
        assert sha(output / "compute" / relative) == digest
    assert before == {
        str(p.relative_to(previous)): sha(p) for p in previous.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "path",
    [
        "evaluate-submitted.claim",
        "evaluate-submission-intent.json",
        "evaluate-submission.json",
        "evaluation-submission.json",
        "evaluation",
        "truth",
        "completion.json",
    ],
)
def test_prior_truth_or_evaluation_use_is_rejected(prior, path):
    root, cfg, _ = prior
    write(root / path, {})
    with pytest.raises(ValueError, match="truth exposure"):
        post.calibration_continuation(root, cfg)


@pytest.mark.parametrize("kind", ["running", "other_owner", "unknown_accounting", "queue"])
def test_requires_actual_released_own_job(prior, monkeypatch, kind):
    root, cfg, _ = prior
    if kind == "running":
        write(root / "controller.json", {"status": "RUNNING"})
    elif kind == "other_owner":
        record = read(root / "calibrate-submission.json")
        record["uid"] += 1
        write(root / "calibrate-submission.json", record)
    elif kind == "unknown_accounting":
        monkeypatch.setattr(operator, "snapshot", lambda record: {"released": False})
    else:
        monkeypatch.setattr(operator, "own_queue", lambda: "other-pending-job")
    with pytest.raises(ValueError, match="released own|allocation release"):
        post.calibration_continuation(root, cfg)


@pytest.mark.parametrize(
    "key", ["source_training_seal_sha256", "source_dataset_sha256", "seed", "protocol"]
)
def test_frozen_semantics_cannot_change(prior, key):
    root, cfg, _ = prior
    new_cfg = copy.deepcopy(cfg)
    new_cfg[key] = {"detector": {"ewma_alpha": 0.9}} if key == "protocol" else "changed"
    with pytest.raises(ValueError, match="frozen input|original detection"):
        post.calibration_continuation(root, new_cfg)


@pytest.mark.parametrize("filename", ["calibration.json", "normal-residuals.npz"])
def test_corrupted_complete_normal_evidence_is_not_refitted(prior, filename):
    root, cfg, name = prior
    (root / "compute/calibration" / name / filename).write_text("{}")
    with pytest.raises(ValueError, match="binding changed"):
        post.calibration_continuation(root, cfg)


def test_partial_normal_evidence_remains_preserved_and_is_not_reused(prior):
    root, cfg, name = prior
    (root / "compute/calibration" / name / "normal-residuals.json").unlink()
    proof = post.calibration_continuation(root, cfg)
    assert proof["reuse_calibration"] == {}
    assert proof["incomplete_calibration_preserved"] == [name]
    assert (root / "compute/calibration" / name / "calibration.json").exists()


def test_symlink_evidence_is_rejected(prior, tmp_path):
    root, cfg, name = prior
    path = root / "compute/calibration" / name / "normal-residuals.npz"
    original = tmp_path / "residuals.npz"
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(ValueError, match="Symlink"):
        post.calibration_continuation(root, cfg)
