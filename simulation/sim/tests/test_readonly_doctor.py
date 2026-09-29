"""Diagnostics and blocked builds must not change shared producer inputs."""

import fcntl
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from capstone_sim import cli, operations


class ReadonlyDoctorTest(unittest.TestCase):
    def fixture(self, root):
        native = root / "simulation"
        (native / "artifacts").mkdir(parents=True)
        specs = root / "docs/specifications/CODEX_REBUILD_PACKAGE_V4_2"
        specs.mkdir(parents=True)
        for name in (
            "00_SCOPE_AND_REVIEW.md",
            "00_SKILL_LINKS.md",
            "00_SOURCE_NOTES.md",
            "00_SHARED_DATA_CONTRACT.md",
            "01_SIMULATION_CODEX_SPEC.md",
        ):
            (specs / name).write_text("immutable spec fixture")
        return native

    def test_diagnostic_does_not_rewrite_source_lock_or_request_git_pip_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary))
            lock = root / "artifacts/source_lock.json"
            lock.write_text('{"preserved":"source identity"}\n')
            with (
                patch.object(cli, "ROOT", root),
                patch.object(cli.platform, "platform", return_value="isolated fixture host"),
                patch.object(
                    cli.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stdout="tool version", stderr=""),
                ),
                patch.object(
                    cli.subprocess,
                    "check_output",
                    side_effect=AssertionError("No identity rewrite"),
                ),
            ):
                result = cli.doctor(root / "artifacts/environment.json")
            self.assertFalse(result["source_lock_modified"])
            self.assertEqual(lock.read_text(), '{"preserved":"source identity"}\n')
            self.assertEqual(
                len(json.loads((root / "artifacts/environment.json").read_text())["tools"]), 6
            )

    def test_diagnostic_cannot_overwrite_runtime_or_source_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary))
            for name in (
                "artifacts/source_lock.json",
                "artifacts/runtime-manifest.json",
                "artifacts/../config/producer-sources.json",
            ):
                with self.subTest(name=name), patch.object(cli, "ROOT", root):
                    with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                        cli.doctor(root / name)

    def test_native_build_routes_to_locking_installer(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(operations, "ROOT", Path(temporary)),
        ):
            with patch.object(
                operations.subprocess, "run", return_value=SimpleNamespace(returncode=3)
            ) as run:
                result = operations.native_build(5)
            self.assertEqual(result["status"], "FAILED_STOPPED")
            self.assertIn("sim/tools/install.py", run.call_args.args[0][2])
            self.assertEqual(run.call_args.args[0][-2:], ["--jobs", "5"])

    def test_active_installer_lock_blocks_build_without_replacing_its_status(self):
        path = Path(__file__).resolve().parents[1] / "tools/install.py"
        spec = importlib.util.spec_from_file_location("isolated_installer_test", path)
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary))
            lock = root / ".cache/setup.lock"
            lock.parent.mkdir()
            status = root / "artifacts/setup-status.json"
            status.write_text('{"status":"RUNNING","owner":"existing"}')
            with lock.open("a") as guard:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (
                    patch.object(installer, "ROOT", root),
                    patch.object(installer, "verify_sources", return_value={}),
                    patch.object(installer, "prerequisites", return_value=dict(passed=True)),
                    patch.object(
                        installer.Setup, "python", side_effect=AssertionError("Must not build")
                    ),
                    redirect_stderr(StringIO()),
                ):
                    self.assertEqual(installer.main([]), 3)
            self.assertEqual(json.loads(status.read_text())["owner"], "existing")


if __name__ == "__main__":
    unittest.main()
