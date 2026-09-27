import contextlib
import io
import unittest
from unittest.mock import patch

from capstone_sim import cli
from capstone_sim.runner import SCENARIOS, scenario_parameters


class FaultOnsetTest(unittest.TestCase):
    def test_override_keeps_physical_magnitude_ramp_and_global_defaults(self):
        for name in [
            "battery_resistance_step",
            "battery_resistance_ramp",
            "propulsion_loss_step",
            "propulsion_loss_ramp",
        ]:
            original = dict(SCENARIOS[name])
            resolved = scenario_parameters(name, 110)
            self.assertEqual(resolved, original | {"fault_at": 110})
            self.assertEqual(SCENARIOS[name], original)
            self.assertEqual(scenario_parameters(name)["fault_at"], 60)

    def test_rejects_nonphysical_or_invalid_override(self):
        for name in ["hover_normal", "battery_resistance_existing", "sensor_dropout_quality"]:
            with self.assertRaises(ValueError):
                scenario_parameters(name, 110)
        for value in [0, -1, 1801, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                scenario_parameters("battery_resistance_step", value)

    def test_cli_passes_boot_time_without_changing_recording_interval(self):
        args = [
            "capstone-sim",
            "run",
            "--scenario",
            "propulsion_loss_ramp",
            "--duration",
            "180",
            "--fault-at-s",
            "110",
        ]
        with patch("sys.argv", args), patch.object(cli, "run_flight") as run:
            run.return_value = {"status": "completed"}
            with contextlib.redirect_stdout(io.StringIO()):
                result = cli.main()
        self.assertEqual(result, 0)
        self.assertEqual(run.call_args.kwargs["fault_at_s"], 110)
        self.assertEqual(run.call_args.kwargs["capture_interval_ms"], 5)
