"""A CLI retry retains the original raw parent link without editing raw metadata."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_vehicle_runner as fixtures
from capstone_sim import cli


class ParentAttemptCliTest(unittest.TestCase):
    def test_cli_forwards_existing_parent_field_and_native_preparation_records_it(self):
        parent = "a" * 32
        args = ["capstone-sim", "run", "--duration", "480", "--parent-attempt-id", parent]
        with patch("sys.argv", args), patch.object(cli, "run_flight", return_value={"status": "completed"}) as launch:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(launch.call_args.kwargs["parent"], parent)
        with tempfile.TemporaryDirectory() as folder:
            result = fixtures.VehicleRunnerTest().run_mocked(Path(folder), custom=False,
                configuration="collection", flight_options={"parent": parent})
            state = json.loads((result["run"] / "run.json").read_text())
            self.assertEqual(state["parent_attempt_id"], parent)
