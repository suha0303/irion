"""New wind settings reach the native environment without changing observation inputs."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_vehicle_runner as fixtures
from capstone_sim import cli


class BoundedWindCliTest(unittest.TestCase):
    def test_cli_and_native_preparation_forward_explicit_environment(self):
        args = ["capstone-sim", "run", "--configuration", "collection", "--duration", "480",
                "--wind-environment", "strong", "--wind-direction-deg", "90", "--wind-variation", "1.5"]
        with patch("sys.argv", args), patch.object(cli, "run_flight", return_value={"status": "completed"}) as launch:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(launch.call_args.kwargs["wind_environment"], "strong")
        self.assertEqual(launch.call_args.kwargs["wind_direction_deg"], 90.)
        self.assertEqual(launch.call_args.kwargs["wind_variation"], 1.5)
        with tempfile.TemporaryDirectory() as folder:
            prepared = fixtures.VehicleRunnerTest().run_mocked(Path(folder), custom=False,
                configuration="collection", flight_options=dict(wind_environment="strong", wind_direction_deg=90., wind_variation=1.5))
            env = prepared["environment"]
            self.assertEqual(env["X500_WIND_ENVIRONMENT"], "strong")
            self.assertEqual(env["X500_WIND_DIRECTION_DEG"], "90.0")
            scenario = json.loads((prepared["run"] / "evaluation/scenario.json").read_text())
            self.assertEqual(scenario["wind_model"]["revision"], "bounded-latent-ou-c2-enu-v1")
            self.assertEqual(scenario["parameters"]["gust_sigma"], 0.)

    def test_unselected_gaussian_wind_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "Gaussian OU"):
            fixtures.VehicleRunnerTest().run_mocked(Path(folder), custom=False,
                configuration="collection", scenario="wind_normal")
