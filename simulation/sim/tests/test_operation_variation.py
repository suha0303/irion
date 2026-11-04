"""Final collection request and actual native scheduling boundaries; no flights."""

import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from capstone_sim import cli, collection_config, runner
from capstone_sim import control as controls
from capstone_sim.fault_schedule import MissionFaultRequest, complete_evidence, schedule_evidence
from capstone_sim.flight_program import build
from capstone_sim.normal_variation import sample


class OperationVariationTest(unittest.TestCase):
    def test_new_duration_retains_existing_path_bounds(self):
        for mode in ("mixed", "random"):
            for seconds in (180, 480, 900):
                spec = build(sample("hover_normal", 41), mode, 41, seconds, 0)
                collection_config.validate_request("hover_normal", seconds, None, operation_variation=spec)
                self.assertEqual(spec["path"]["duration_s"], seconds)
                with self.assertRaises(ValueError):
                    collection_config.validate_request("hover_normal", seconds - 1, None, operation_variation=spec)

    def test_all_fault_kinds_accept_health_independent_low_soc_motion(self):
        spec = sample("hover_normal", 44)
        spec["soc"] = .45
        spec["path"]["kind"] = "altitude"
        original = copy.deepcopy(runner.SCENARIOS)
        for scenario in collection_config.ANOMALY:
            onset = 100 if scenario.endswith(("step", "ramp")) else None
            collection_config.validate_request(
                scenario, 180, None, operation_variation=spec, fault_after_mission_s=onset
            )
            with self.assertRaises(ValueError):
                collection_config.validate_request(
                    scenario, 480, None, operation_variation=spec, fault_after_mission_s=onset
                )
        self.assertEqual(original, runner.SCENARIOS)
        for key in ("r0", "friction", "fault_value", "fault_at", "capacity"):
            bad = dict(spec, **{key: 1})
            with self.assertRaises(ValueError):
                collection_config.validate_request("battery_resistance_existing", 180, None, operation_variation=bad)

    def test_new_onset_does_not_weaken_old_boot_policy(self):
        spec = sample("hover_normal", 41)
        for value in (0, -1, 480, 460, True, .0000001, math.inf, math.nan):
            with self.assertRaises(ValueError):
                collection_config.validate_request("battery_resistance_ramp", 480, None, operation_variation=spec, fault_after_mission_s=value)
        with self.assertRaises(ValueError):
            collection_config.validate_request("battery_resistance_step", 480, 240)
        collection_config.validate_request("battery_resistance_step", 480, 180)
        with self.assertRaises(ValueError):
            collection_config.validate_request("battery_resistance_step", 480, 180, operation_variation=spec, fault_after_mission_s=240)
        for scenario in ("hover_normal", "battery_resistance_existing"):
            with self.assertRaises(ValueError):
                collection_config.validate_request(scenario, 480, None, operation_variation=spec, fault_after_mission_s=240)
        with self.assertRaises(ValueError):
            runner.run_flight("hover_normal", operation_variation=spec)
        with self.assertRaises(ValueError):
            runner.run_flight("hover_normal", configuration="collection", operation_variation=spec, normal_variation=spec)

    def test_cli_preserves_new_request_clock_and_conditions(self):
        spec = sample("hover_normal", 41)
        args = ["capstone-sim", "run", "--configuration", "collection", "--scenario", "propulsion_loss_ramp", "--duration", "480", "--operation-variation", json.dumps(spec), "--fault-after-mission-s", "240"]
        with patch("sys.argv", args), patch.object(cli, "run_flight", return_value={"status": "completed"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(run.call_args.kwargs["operation_variation"], spec)
        self.assertEqual(run.call_args.kwargs["fault_after_mission_s"], 240)
        self.assertIsNone(run.call_args.kwargs["fault_at_s"])
        self.assertIsNone(run.call_args.kwargs["normal_variation"])

    def test_reset_preserves_relative_schedule_and_operating_conditions(self):
        spec = sample("hover_normal", 41)
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / ("a" * 32)
            (run / "evaluation").mkdir(parents=True)
            (run / "run.json").write_text(json.dumps(dict(run_id=run.name, status="completed", configuration="collection", instance=3)))
            (run / "evaluation/scenario.json").write_text(json.dumps(dict(
                name="battery_resistance_ramp", seed=41, requested_duration_s=480,
                parameters={"fault_at": 287.123}, operation_variation=spec,
                fault_after_mission_s=240, normal_variation=None,
            )))
            with patch.object(controls, "run_flight", return_value={}) as launch:
                controls.control(run, "reset")
            options = launch.call_args.kwargs
            self.assertEqual(options["operation_variation"], spec)
            self.assertEqual(options["fault_after_mission_s"], 240)
            self.assertNotIn("fault_at_s", options)


class MissionScheduleTest(unittest.TestCase):
    env = {"GZ_PARTITION": "x500_" + "a" * 32, "GZ_IP": "127.0.0.1"}

    def test_ack_matches_observed_origin_not_fixed_boot_guess(self):
        child = Mock()
        child.poll.side_effect = [None, 0, 0]
        child.communicate.return_value = ("data: 0\ndata: 287123000\n", "")
        with patch("capstone_sim.fault_schedule.subprocess.Popen", return_value=child) as launch:
            request = MissionFaultRequest(47.123, 240, self.env)
            self.assertIsNone(request.poll())
            child.communicate.assert_not_called()
            result = request.poll()
            self.assertTrue(result["acknowledged"])
            self.assertEqual(result["mission_origin_time_us"], 47_123_000)
            self.assertEqual(result["scheduled_time_us"], 287_123_000)
            self.assertEqual(result["requested_offset_us"], 240_000_000)
            self.assertIsNone(result["applied_time_us"])
            self.assertIn("data: 1 data: 47123000", launch.call_args.args[0])
            self.assertIsNone(request.poll())
            request.close()
        child.terminate.assert_not_called()

    def test_rejection_mismatch_and_timeout_fail_closed(self):
        for reply in ("data: -1\n", "data: 0\ndata: 240000000\n", "data: nan\n"):
            child = Mock()
            child.poll.return_value = 0
            child.communicate.return_value = (reply, "")
            with patch("capstone_sim.fault_schedule.subprocess.Popen", return_value=child):
                request = MissionFaultRequest(47, 240, self.env)
                with self.assertRaises(RuntimeError):
                    request.poll()
        child = Mock()
        child.poll.return_value = None
        with patch("capstone_sim.fault_schedule.subprocess.Popen", return_value=child), patch("capstone_sim.fault_schedule.time.monotonic", side_effect=[0, 16]):
            request = MissionFaultRequest(47, 240, self.env)
            with self.assertRaises(TimeoutError):
                request.poll()
            request.close()
            child.terminate.assert_called_once()

    def test_actual_truth_is_required_and_must_be_within_mission(self):
        schedule = schedule_evidence(47.123, 100)
        with self.assertRaises(ValueError):
            complete_evidence(schedule, 147_123_000, 180)
        schedule["acknowledged"] = True
        result = complete_evidence(schedule, 147_124_000, 180)
        self.assertEqual(result["applied_time_us"], 147_124_000)
        self.assertEqual(result["application_delay_us"], 1000)
        for applied in (None, False, 147_122_999, 227_123_000):
            with self.assertRaises(ValueError):
                complete_evidence(schedule, applied, 180)
        with self.assertRaises(ValueError):
            complete_evidence(None, 147_124_000, 180)
        with self.assertRaises(ValueError):
            complete_evidence(dict(schedule, scheduled_time_us=147_124_000), 147_124_000, 180)

    def test_invalid_origin_and_partition_never_launch(self):
        for origin, delay in ((-1, 240), (math.nan, 100), (10, 0), (10, math.inf), (1, .0000001)):
            with self.assertRaises(ValueError):
                schedule_evidence(origin, delay)
        with patch("capstone_sim.fault_schedule.subprocess.Popen") as launch:
            with self.assertRaises(ValueError):
                MissionFaultRequest(47, 240, {})
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
