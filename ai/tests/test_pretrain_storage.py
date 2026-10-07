"""Portable byte equality, truth separation and tamper rejection, no real data."""

import io
import tarfile

import pytest

from pretrain import storage
from pretrain.common import read, sha, write


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "external", lambda p: p)
    root = tmp_path / "prepared"
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    raw = tmp_path / "raw.tar.gz"
    body = b'{"event":"fixture"}\n'
    with tarfile.open(raw, "w:gz") as tar:
        info = tarfile.TarInfo("evaluation/events.jsonl")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
        info = tarfile.TarInfo("observations/streams.jsonl")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"raw"))
    import hashlib

    row = dict(
        run_id="run1",
        role="anomaly_dev",
        archive=dict(
            path=str(raw),
            bytes=raw.stat().st_size,
            sha256=sha(raw),
            members={
                "evaluation/events.jsonl": dict(
                    kind="file", bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
                )
            },
        ),
    )
    write(dataset / "manifest.json", dict(runs=[row]))
    (dataset / "normal_train.npz").write_bytes(b"fixture array bytes" * 100)
    write(root / "dataset-seal.json", {p.name: sha(p) for p in dataset.iterdir()})
    write(root / "quality.json", dict(passed=True, train_only_scaler_verified=True))
    write(root / "frozen-manifest.json", dict(runs=[row]))
    write(
        root / "handoff.json",
        dict(
            training_eligible=True,
            dataset=str(dataset),
            **{
                k: sha(root / n)
                for k, n in [
                    ("dataset_seal_sha256", "dataset-seal.json"),
                    ("quality_sha256", "quality.json"),
                    ("frozen_manifest_sha256", "frozen-manifest.json"),
                ]
            },
        ),
    )
    return root


def test_lossless_restore_and_truth_separation(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    original = (root / "handoff.json").read_bytes()
    out = tmp_path / "portable"
    receipt = storage.pack(root, out)
    assert receipt["raw_deleted"] is False
    assert all(
        "streams" not in name and "events" not in name for name in receipt["dataset"]["members"]
    )
    restored = storage.restore(out, tmp_path / "restored")
    assert (restored / "dataset/normal_train.npz").read_bytes() == (
        root / "dataset/normal_train.npz"
    ).read_bytes()
    assert read(restored / "evidence/handoff.json")["dataset"] == str(restored / "dataset")
    truth = storage.restore(out, tmp_path / "evaluator", truth=True)
    assert (truth / "run1.jsonl").read_bytes() == b'{"event":"fixture"}\n'
    row = read(root / "dataset/manifest.json")["runs"][0]
    target = tmp_path / "joined.jsonl"
    storage.truth_provider(out)(row, target, storage.Budget())
    assert target.read_bytes() == (truth / "run1.jsonl").read_bytes()
    row["role"] = "normal_train"
    with pytest.raises(ValueError, match="development evaluator"):
        storage.truth_provider(out)(row, tmp_path / "leak.jsonl", storage.Budget())
    assert (root / "handoff.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        storage.pack(root, out)


def test_bad_quality_and_archive_tamper_fail_closed(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    out = tmp_path / "portable"
    storage.pack(root, out)
    with (out / "dataset.tar.xz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="archive changed"):
        storage.restore(out, tmp_path / "restored")
    quality = read(root / "quality.json")
    quality["passed"] = False
    write(root / "quality.json", quality)
    with pytest.raises(ValueError, match="verified dataset"):
        storage.pack(root, tmp_path / "bad")


@pytest.mark.parametrize("name", ["../raw", "/absolute", "a/../../raw", "a\\b"])
def test_unsafe_names_rejected(name):
    with pytest.raises(ValueError):
        storage.safe_name(name)
