"""Synthetic launch-boundary checks; never start PX4, Gazebo or a socket."""

import contextlib
import fcntl
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from capstone_sim import cli, runner


class CollectionIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for relative in [
            "profiles/cmu_vah07_4s2p.json",
            "profiles/x500_reference.yaml",
            "artifacts/propulsion-comparison.json",
        ]:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((runner.ROOT / relative).read_bytes())
        (self.root / "artifacts/source_lock.json").write_text("{}")
        (self.root / "build").mkdir()
        self.binary = self.root / "build/px4"
        self.binary.write_text("synthetic non-executable binary placeholder")
        (self.root / "build/libDronePowertrain.so").write_text("synthetic plugin placeholder")

    def launch_boundary(self, instance, seed, duration):
        child = Mock(pid=12345, returncode=0)
        child.poll.return_value = 0
        with (
            patch.object(runner, "ROOT", self.root),
            patch.object(runner, "BINARY", self.binary),
            patch.object(runner.subprocess, "Popen", return_value=child) as launch,
            patch.object(
                runner.mavutil,
                "mavlink_connection",
                side_effect=RuntimeError("synthetic stop before connection"),
            ) as connection,
            patch.object(runner.os, "killpg") as kill,
            patch.dict(os.environ, {"X500_CAPTURE_TIMING_PATH": "inherited", "X500_RUN_ID": "old"}),
        ):
            result = runner.run_flight(
                "hover_normal", seed=seed, duration=duration, instance=instance
            )
        kill.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(launch.call_args_list), 2)
        return Path(result["run"]), launch.call_args_list, connection.call_args

    def test_twenty_and_thirty_minutes_reach_launch_boundary(self):
        for duration in (1200, 1800):
            run, calls, _ = self.launch_boundary(0, 123, duration)
            scenario = json.loads((run / "evaluation/scenario.json").read_text())
            self.assertEqual(scenario["requested_duration_s"], duration)
            self.assertEqual(calls[0].args[0][:4], ["gz", "sim", "-r", "-s"])
            self.assertEqual(calls[0].kwargs["env"]["PX4_SIM_SPEED_FACTOR"], "1")
            world = (run / "reference/world.sdf").read_text()
            self.assertIn("<max_step_size>0.001</max_step_size>", world)
            self.assertIn("<real_time_factor>2</real_time_factor>", world)

    def test_run_files_partition_seed_ports_and_process_groups_are_separate(self):
        a, ca, ma = self.launch_boundary(0, 123, 1200)
        b, cb, mb = self.launch_boundary(1, 124, 1200)
        self.assertNotEqual(a, b)
        for run, calls, connection, instance, seed in (
            (a, ca, ma, 0, 123), (b, cb, mb, 1, 124)
        ):
            env = calls[0].kwargs["env"]
            state = json.loads((run / "run.json").read_text())
            self.assertEqual(state["instance"], instance)
            self.assertEqual(state["gz_partition"], "x500_" + run.name)
            self.assertEqual(calls[1].kwargs["env"], env)
            self.assertEqual(env["GZ_PARTITION"], "x500_" + run.name)
            self.assertEqual(env["X500_SEED"], str(seed))
            self.assertEqual(env["X500_CAPTURE_PATH"], str(run / "native_logs/capture.jsonl"))
            self.assertEqual(env["X500_TRUTH_PATH"], str(run / "evaluation/truth.jsonl"))
            self.assertEqual(env["GZ_HOMEDIR"], str(run / "gz"))
            self.assertEqual(connection.args[0], f"udpin:127.0.0.1:{14540 + instance}")
            self.assertEqual(calls[1].args[0][-4:], ["-i", str(instance), "-w", str(run / "px4")])
            self.assertTrue(all(c.kwargs["start_new_session"] for c in calls))
            self.assertNotIn("X500_CAPTURE_TIMING_PATH", env)
            self.assertNotIn("X500_RUN_ID", env)
            self.assertFalse((run / "native_logs/timing-config.json").exists())
            self.assertFalse((run / "native_logs/producer-timing.jsonl").exists())

    def test_reference_nonzero_instance_is_recorded_before_process_start(self):
        child = Mock(pid=12345, returncode=0)
        child.poll.return_value = 0
        seen = []

        def launch(_args, **kwargs):
            run = Path(kwargs["env"]["GZ_HOMEDIR"]).parent
            state = json.loads((run / "run.json").read_text())
            seen.append((state["instance"], state["gz_partition"]))
            return child

        with (
            patch.object(runner, "ROOT", self.root),
            patch.object(runner, "BINARY", self.binary),
            patch.object(runner.subprocess, "Popen", side_effect=launch),
            patch.object(runner.mavutil, "mavlink_connection", side_effect=RuntimeError("synthetic stop")),
        ):
            result = runner.run_flight("hover_normal", instance=3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(seen, [(3, "x500_" + Path(result["run"]).name)] * 2)

    def test_held_instance_fails_before_creating_run_or_launching(self):
        lock = self.root / ".cache/instances/2.lock"
        lock.parent.mkdir(parents=True)
        with lock.open("a") as lease, patch.object(runner, "ROOT", self.root):
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(runner.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "already owned"):
                    runner.run_flight("hover_normal", instance=2, duration=1200)
                launch.assert_not_called()
        self.assertFalse((self.root / "runs").exists())

    def test_invalid_limits_fail_without_creating_run(self):
        with patch.object(runner, "ROOT", self.root):
            for duration in (0, -1, 1801, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    runner.run_flight("hover_normal", duration=duration)
            for instance in (-1, 10):
                with self.assertRaises(ValueError):
                    runner.run_flight("hover_normal", instance=instance)
        self.assertFalse((self.root / "runs").exists())

    def test_cli_keeps_timing_opt_in_for_long_runs(self):
        for duration in (1200, 1800):
            with (
                patch("sys.argv", ["capstone-sim", "run", "--duration", str(duration), "--instance", "1"]),
                patch.object(cli, "run_flight", return_value={"status": "completed"}) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(cli.main(), 0)
            self.assertEqual(run.call_args.args[2:4], (duration, 1))
            self.assertFalse(run.call_args.kwargs["timing"])
            self.assertEqual(run.call_args.kwargs["capture_interval_ms"], 5)
