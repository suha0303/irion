"""Packaging failures must be rejected before importing or training a model."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("handoff_runner", "ai/handoff/run_smoke.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def package(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"locked-input")
    value = {"files": {"data.bin": {"sha256": runner.digest(tmp_path / "data.bin"), "bytes": 12}}}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(value))
    (tmp_path / "SHA256SUMS").write_text("".join(
        runner.digest(tmp_path / name) + "  " + name + "\n"
        for name in ["data.bin", "MANIFEST.json"]
    ))
    return tmp_path


def test_manifest_pass_and_content_tamper(tmp_path):
    root = package(tmp_path)
    assert runner.verify(root)["verified_files"] == 1
    (root / "data.bin").write_bytes(b"bad-input!!!")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.verify(root)


def test_missing_payload_rejected(tmp_path):
    root = package(tmp_path)
    (root / "data.bin").unlink()
    with pytest.raises(ValueError, match="missing"):
        runner.verify(root)


def test_unlisted_source_rejected(tmp_path):
    root = package(tmp_path)
    (root / "extra.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="unlisted"):
        runner.verify(root)


@pytest.mark.parametrize("name", ["../external", "/tmp/external"])
def test_manifest_escape_rejected(tmp_path, name):
    root = package(tmp_path)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    manifest["files"][name] = {"sha256": "irrelevant", "bytes": 1}
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid manifest path"):
        runner.verify(root)


def test_checksum_list_change_rejected(tmp_path):
    root = package(tmp_path)
    sums = root / "SHA256SUMS"
    sums.write_text(sums.read_text().replace("data.bin", "wrong.bin"))
    with pytest.raises(ValueError, match="membership"):
        runner.verify(root)


def test_packaged_source_matches_historical_v5():
    from capstone_ai.source import read_json, sha

    readiness = Path("ai/artifacts/gpu-readiness-v5/readiness.json")
    root = Path("ai/artifacts/gpu-handoff-v6/release/x500-gpu-smoke-v1")
    old = read_json(readiness)
    # The sealed payload retains V5 sources while the working tree evolves.
    manifest = read_json(root / "MANIFEST.json")
    assert manifest["source_readiness_sha256"] == sha(readiness)
    assert runner.verify(root)["verified_files"] == len(manifest["files"])
    assert all(sha(root / name) == digest for name, digest in old["source_hashes"].items())
