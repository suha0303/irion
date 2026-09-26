"""Pinned pristine PX4 inputs must reproduce every reviewed vendor byte."""

import hashlib
import json
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path

NATIVE = Path(__file__).resolve().parents[2]
PATCHER = NATIVE / "sim/tools/patch_px4.py"
FIXTURE = Path(__file__).parent / "fixtures/px4_patch/pristine.json"
ESC = "vendor/PX4-Autopilot/src/modules/simulation/gz_bridge/GZMixingInterfaceESC.cpp"


def digest(content):
    return hashlib.sha256(content).hexdigest()


class PX4PatchTest(unittest.TestCase):
    def fixture(self, root):
        recorded = json.loads(FIXTURE.read_text())
        manifest = json.loads((NATIVE / "config/producer-sources.json").read_text())
        upstream = manifest["upstream"]["px4"]
        self.assertEqual(recorded["px4_commit"], upstream["commit"])
        for name, entry in recorded["files"].items():
            expected_commit = (
                upstream["submodule_commits"]["Tools/simulation/gz"]
                if "/Tools/simulation/gz/" in name
                else upstream["commit"]
            )
            self.assertEqual(entry["upstream_commit"], expected_commit)
            content = entry["text"].encode()
            self.assertEqual(digest(content), entry["sha256"])
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        script = root / "sim/tools/patch_px4.py"
        script.parent.mkdir(parents=True)
        shutil.copyfile(PATCHER, script)
        for name in ("RawCapture.hpp", "CaptureTiming.hpp"):
            adapter = root / "sim/adapters" / name
            adapter.parent.mkdir(exist_ok=True)
            shutil.copyfile(NATIVE / "sim/adapters" / name, adapter)
        return script, manifest["vendor_files"]

    def assert_reviewed(self, root, expected):
        actual = {
            name: digest((root / name).read_bytes())
            for name in expected
        }
        self.assertEqual(actual, expected)
        return actual

    def test_pristine_checkout_reproduces_all_vendor_hashes_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script, expected = self.fixture(root)
            runpy.run_path(str(script))
            first = self.assert_reviewed(root, expected)
            runpy.run_path(str(script))
            self.assertEqual(self.assert_reviewed(root, expected), first)

    def test_exact_previous_installer_indentation_variant_is_repaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script, expected = self.fixture(root)
            namespace = runpy.run_path(str(script))
            capture_header = namespace["capture_header"]
            target = root / ESC
            source = target.read_text()
            self.assertEqual(source.count(capture_header), 1)
            target.write_text(
                source.replace(
                    capture_header,
                    "\n".join("    " + line for line in capture_header.splitlines()),
                    1,
                )
            )
            self.assertNotEqual(digest(target.read_bytes()), expected[ESC])
            runpy.run_path(str(script))
            self.assert_reviewed(root, expected)

    def test_unknown_capture_change_is_rejected_instead_of_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script, expected = self.fixture(root)
            runpy.run_path(str(script))
            self.assert_reviewed(root, expected)
            target = root / ESC
            altered = target.read_text().replace(
                '"post_mapping_count_0_1000"', '"unreviewed_semantics"', 1
            )
            target.write_text(altered)
            with self.assertRaisesRegex(RuntimeError, "patch context not found"):
                runpy.run_path(str(script))
            self.assertEqual(target.read_text(), altered)


if __name__ == "__main__":
    unittest.main()
