"""Portable runtime seals use relative identities and reject stale or missing inputs."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from capstone_sim import collection_config
from capstone_sim.runtime_manifest import (
    PLUGIN,
    PROFILE,
    PX4_BINARY,
    RUNTIME_MANIFEST,
    SOURCE_MANIFEST,
    TRANSPORT,
    checked_files,
    sha,
    verify_runtime,
    verify_sources,
)


class PortableRuntimeTest(unittest.TestCase):
    def fixture(self, root, *, include_fc_policy=True):
        source_files = {PROFILE: "profile", "sim/src/example.py": "input"}
        vendor_files = {"vendor/PX4-Autopilot/patch.cpp": "patched upstream"}
        for name, body in (
            source_files
            | vendor_files
            | {
                PX4_BINARY: "px4",
                PLUGIN: "physics",
                TRANSPORT: "transport",
                "artifacts/source_lock.json": "recorded native environment",
            }
        ).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        (root / PX4_BINARY).chmod(0o755)
        (root / TRANSPORT).with_name("libgz-transport13.so.13").symlink_to(Path(TRANSPORT).name)
        source = dict(
            schema="x500.producer-sources/1",
            files={name: sha(root / name) for name in source_files},
            vendor_files={name: sha(root / name) for name in vendor_files},
        )
        if include_fc_policy:
            source["fc_parameters"] = collection_config.fc_parameters()
        path = root / SOURCE_MANIFEST
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(source))
        files = source["files"] | source["vendor_files"]
        for name in (SOURCE_MANIFEST, PX4_BINARY, PLUGIN, TRANSPORT, "artifacts/source_lock.json"):
            files[name] = sha(root / name)
        runtime = dict(
            schema="x500.producer-runtime/1",
            revision="X500_CURRENT",
            files=files,
            source_manifest=SOURCE_MANIFEST,
            source_manifest_sha256=sha(path),
            px4_binary_sha256=files[PX4_BINARY],
            physics_plugin_sha256=files[PLUGIN],
            x500_profile_sha256=files[PROFILE],
            source_revision_sha256=files["artifacts/source_lock.json"],
        )
        if include_fc_policy:
            runtime["fc_parameters"] = dict(source["fc_parameters"])
        path = root / RUNTIME_MANIFEST
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(runtime))
        return runtime

    def test_same_relative_manifest_verifies_in_different_checkout_with_spaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "original"
            self.fixture(first)
            second = Path(temporary) / "another checkout"
            shutil.copytree(first, second, symlinks=True)
            self.assertEqual(verify_runtime(first), verify_runtime(second))

    def test_current_runtime_requires_the_reviewed_fc_policy(self):
        for changed in (None, {}, {"MPC_LAND_SPEED": 0.3},
                        collection_config.fc_parameters() | {"MPC_LAND_SPEED": 0.7}):
            with self.subTest(policy=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                runtime = self.fixture(root)
                runtime["fc_parameters"] = changed
                (root / RUNTIME_MANIFEST).write_text(json.dumps(runtime))
                with self.assertRaisesRegex(ValueError, "runtime FC policy"):
                    verify_runtime(root)

    def test_resigning_source_cannot_make_incomplete_or_wrong_fc_policy_valid(self):
        for changed in ({}, {"MPC_LAND_SPEED": 0.3},
                        collection_config.fc_parameters() | {"MPC_LAND_SPEED": 0.7},
                        collection_config.fc_parameters() | {"MPC_LAND_SPEED": "0.3"}):
            with self.subTest(policy=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                runtime = self.fixture(root)
                source = json.loads((root / SOURCE_MANIFEST).read_text())
                source["fc_parameters"] = changed
                (root / SOURCE_MANIFEST).write_text(json.dumps(source))
                digest = sha(root / SOURCE_MANIFEST)
                runtime["source_manifest_sha256"] = digest
                runtime["files"][SOURCE_MANIFEST] = digest
                runtime["fc_parameters"] = changed
                (root / RUNTIME_MANIFEST).write_text(json.dumps(runtime))
                with self.assertRaisesRegex(ValueError, "runtime FC policy"):
                    verify_runtime(root)

    def test_historical_seal_without_fc_policy_keeps_its_original_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self.fixture(root, include_fc_policy=False)
            self.assertEqual(verify_runtime(root), runtime)

    def test_changed_source_and_binary_are_not_resealed_implicitly(self):
        for name in (PROFILE, PX4_BINARY, PLUGIN, TRANSPORT):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.fixture(root)
                (root / name).write_text("changed")
                with self.assertRaisesRegex(ValueError, "frozen inputs changed"):
                    verify_runtime(root)

    def test_runtime_cannot_remove_trusted_source_or_required_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self.fixture(root)
            del runtime["files"][PLUGIN]
            (root / RUNTIME_MANIFEST).write_text(json.dumps(runtime))
            with self.assertRaisesRegex(ValueError, "missing required"):
                verify_runtime(root)

    def test_machine_runtime_cannot_select_another_source_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self.fixture(root)
            runtime["source_manifest"] = "other.json"
            (root / RUNTIME_MANIFEST).write_text(json.dumps(runtime))
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_runtime(root)

    def test_old_descriptor_presence_does_not_select_a_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / RUNTIME_MANIFEST).unlink()
            old = root / "artifacts/round001-r3/producer-lock-layout-v7.json"
            old.parent.mkdir(parents=True)
            old.write_text('{"files": {}}')
            verify_sources(root)
            with self.assertRaisesRegex(ValueError, "not prepared"):
                verify_runtime(root)
            from unittest.mock import patch

            with patch.object(collection_config, "ROOT", root):
                self.assertEqual(collection_config.producer_lock_path(), root / RUNTIME_MANIFEST)

    def test_unsafe_paths_and_symlink_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkout"
            root.mkdir()
            outside = Path(temporary) / "outside"
            outside.write_text("outside")
            (root / "escape").symlink_to(outside)
            for name in ("../outside", str(outside), "escape", "windows\\escape"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    checked_files(root, {name: sha(outside)})


if __name__ == "__main__":
    unittest.main()
