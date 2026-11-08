import hashlib
import json

import pytest

from execution.delivery.poweroff_after_fetch import ready


def fixture(tmp_path):
    names = [
        f"{t}_{a}_h{h}_l{layers}"
        for t in ("forecast", "run_health")
        for a in ("lstm", "transformer")
        for h, layers in ((64, 2), (64, 4), (128, 2), (128, 4))
    ]
    for n in names:
        p = tmp_path / "models" / n / "model.safetensors"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"fixture")
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {"models": names, "files": {"example": "hash"}, "source": "remote"}
        ),
        encoding="utf-8",
    )
    config = {"destination": str(tmp_path), "remote_base": "remote"}
    receipt = {
        "status": "LOCAL_FINAL_READY",
        "destination": str(tmp_path),
        "models": 16,
        "files": 1,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    return config, receipt, manifest


def test_transfer_or_hash_validation_in_progress_cannot_shutdown(tmp_path):
    for status in [
        "PREPARING_FINAL_EXPORT",
        "DOWNLOADING",
        "VERIFYING_LOCAL_FILES",
        "DELIVERY_FAILED_STOPPED",
    ]:
        assert not ready({"destination": str(tmp_path)}, {"status": status})


def test_only_verified_matching_complete_bundle_is_ready(tmp_path):
    config, receipt, manifest = fixture(tmp_path)
    assert ready(config, receipt)
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest"):
        ready(config, receipt)


def test_missing_model_prevents_shutdown(tmp_path):
    config, receipt, _ = fixture(tmp_path)
    next((tmp_path / "models").rglob("model.safetensors")).unlink()
    with pytest.raises(ValueError, match="Missing"):
        ready(config, receipt)
