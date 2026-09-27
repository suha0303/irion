"""No SITL: test shutdown escalation and a real child taking nine seconds to exit."""

import signal
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from capstone_sim.collection_config import FCParameterGate
from capstone_sim.runner import COLLECTION_PHASE_WAIT_S, stop_owned_process


class ShutdownChecks(unittest.TestCase):
    def service_process(self):
        process = Mock(pid=123, returncode=None)
        process.poll.side_effect = lambda: process.returncode
        process.wait.side_effect = lambda **kw: setattr(process, "returncode", 0)
        return process

    def service_args(self):
        run = Path("/tmp") / ("a" * 32)
        return dict(run=run, env={"GZ_PARTITION": "x500_" + run.name, "GZ_IP": "127.0.0.1"})

    def test_service_ack_and_actual_exit_need_no_signal(self):
        process = self.service_process()
        response = Mock(returncode=0, stdout="data: true\n", stderr="")
        with (
            patch("capstone_sim.runner.subprocess.run", return_value=response) as request,
            patch("capstone_sim.runner.os.killpg") as kill,
        ):
            record = stop_owned_process(process, "gazebo", **self.service_args())
        self.assertEqual(request.call_args.kwargs["env"], self.service_args()["env"])
        self.assertIn("/server_control", request.call_args.args[0])
        self.assertEqual(request.call_args.kwargs["timeout"], 15)
        process.wait.assert_called_once_with(timeout=600)
        kill.assert_not_called()
        self.assertTrue(record["service_stop"]["acknowledged"])
        self.assertFalse(record["service_stop"]["signal_fallback"])
        self.assertEqual(record["exit_code"], 0)

    def test_wrong_partition_cannot_send_stop_to_another_instance(self):
        process = self.service_process()
        args = self.service_args()
        args["env"]["GZ_PARTITION"] = "unrelated"
        with (
            patch("capstone_sim.runner.subprocess.run") as request,
            patch("capstone_sim.runner.os.killpg") as kill,
        ):
            record = stop_owned_process(process, "gazebo", **args)
        request.assert_not_called()
        kill.assert_called_once_with(123, signal.SIGTERM)
        self.assertIn("mismatch", record["service_stop"]["error"])

    def test_failed_or_missing_reply_falls_back_to_owned_process_only(self):
        for response in (
            Mock(returncode=0, stdout="data: false\n", stderr=""),
            Mock(returncode=1, stdout="data: true\n", stderr="error"),
            Mock(returncode=0, stdout="", stderr=""),
            subprocess.TimeoutExpired("gz service", 15),
            FileNotFoundError("gz"),
        ):
            with self.subTest(response=response):
                process = self.service_process()
                with (
                    patch("capstone_sim.runner.subprocess.run") as request,
                    patch("capstone_sim.runner.os.killpg") as kill,
                ):
                    if isinstance(response, Exception):
                        request.side_effect = response
                    else:
                        request.return_value = response
                    record = stop_owned_process(process, "gazebo", **self.service_args())
                self.assertFalse(record["service_stop"]["acknowledged"])
                self.assertEqual(record["signals"], ["SIGTERM"])
                kill.assert_called_once_with(123, signal.SIGTERM)

    def test_ack_is_not_exit_and_timeout_keeps_signal_escalation(self):
        process = self.service_process()

        def wait(**kw):
            if kw["timeout"] == 600:
                raise subprocess.TimeoutExpired("gazebo", 600)
            process.returncode = -9

        process.wait.side_effect = wait
        with (
            patch(
                "capstone_sim.runner.subprocess.run",
                return_value=Mock(returncode=0, stdout="data: true\n", stderr=""),
            ),
            patch("capstone_sim.runner.os.killpg"),
        ):
            record = stop_owned_process(process, "gazebo", **self.service_args())
        self.assertTrue(record["service_stop"]["exit_wait_expired"])
        self.assertEqual(record["signals"], ["SIGTERM", "SIGKILL"])
        self.assertEqual(record["exit_code"], -9)

    def test_ack_does_not_hide_nonzero_exit(self):
        process = self.service_process()
        process.wait.side_effect = lambda **kw: setattr(process, "returncode", 4)
        with (
            patch(
                "capstone_sim.runner.subprocess.run",
                return_value=Mock(returncode=0, stdout="data: true\n", stderr=""),
            ),
            patch("capstone_sim.runner.os.killpg") as kill,
        ):
            record = stop_owned_process(process, "gazebo", **self.service_args())
        kill.assert_not_called()
        self.assertEqual(record["exit_code"], 4)

    def test_gazebo_gets_ten_minutes_and_px4_one_minute(self):
        for name, seconds in (("gazebo", 600), ("px4", 60)):
            process = Mock(pid=123, returncode=0)
            process.poll.return_value = None
            with patch("capstone_sim.runner.os.killpg") as kill:
                record = stop_owned_process(process, name)
            process.wait.assert_called_once_with(timeout=seconds)
            kill.assert_called_once_with(123, signal.SIGTERM)
            self.assertFalse(record["grace_expired"])

    def test_timeout_records_sigkill_and_nonzero_exit(self):
        process = Mock(pid=123, returncode=-9)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 600), -9]
        with patch("capstone_sim.runner.os.killpg"):
            record = stop_owned_process(process, "gazebo")
        self.assertEqual(record["signals"], ["SIGTERM", "SIGKILL"])
        self.assertTrue(record["grace_expired"])
        self.assertEqual(record["exit_code"], -9)

    def test_fc_wait_and_phase_limits_remain_finite(self):
        gate = FCParameterGate(timeout_wall_s=300)
        self.assertFalse(gate.step(0, lambda _: None))
        self.assertFalse(gate.step(60, lambda _: None))
        self.assertFalse(gate.step(299, lambda _: None))
        with self.assertRaises(TimeoutError):
            gate.step(300, lambda _: None)
        self.assertEqual(gate.evidence(300, False, "fixture")["timeout_wall_s"], 300)
        self.assertEqual(COLLECTION_PHASE_WAIT_S, dict(streaming=300, takeoff=180, landing=300))

    def test_preexisting_failure_preserved(self):
        process = Mock(pid=123, returncode=-9)
        process.poll.return_value = -9
        with patch("capstone_sim.runner.os.killpg") as kill:
            record = stop_owned_process(process, "gazebo")
        kill.assert_not_called()
        self.assertEqual(record["exit_code"], -9)
        self.assertEqual(record["signals"], [])

    def test_real_slow_graceful_exit_survives_old_eight_second_limit(self):
        code = """import signal,time,sys
def stop(*args):
    time.sleep(9)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
print("ready", flush=True)
while True: signal.pause()
"""
        process = subprocess.Popen(
            [sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE, text=True
        )
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            record = stop_owned_process(process, "gazebo")
            self.assertEqual(record["exit_code"], 0)
            self.assertEqual(record["signals"], ["SIGTERM"])
            self.assertGreaterEqual(record["elapsed_seconds"], 9)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
