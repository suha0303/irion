"""Historical evidence cannot select or replace the installed producer."""

import pytest
from capstone_sim import collection_config


def test_current_producer_path_is_stable_even_with_retired_revision_files(tmp_path, monkeypatch):
    monkeypatch.setattr(collection_config, "ROOT", tmp_path)
    archive = tmp_path / "artifacts/round001-r3"
    archive.mkdir(parents=True)
    (archive / "producer-lock-layout-v7.json").write_text('{"forged":"old selector"}')
    assert collection_config.producer_lock_path() == tmp_path / "artifacts/runtime-manifest.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        collection_config.assert_frozen()


def test_installed_runtime_is_bound_to_reviewed_source_and_binary():
    from capstone_sim.runtime_manifest import SOURCE_MANIFEST, sha, verify_runtime

    installed = verify_runtime(collection_config.ROOT)
    assert installed["source_manifest_sha256"] == sha(collection_config.ROOT / SOURCE_MANIFEST)
    assert installed["px4_binary_sha256"] == installed["files"][
        "vendor/PX4-Autopilot/build/px4_sitl_default/bin/px4"
    ]
