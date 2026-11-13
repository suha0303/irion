"""A real TERM unwinds cleanup and restores the process signal handler."""

import os
import signal
import subprocess
import sys
import unittest

from capstone_sim.runner import termination_guard


class RunnerTerminationTest(unittest.TestCase):
    def test_handler_is_restored_after_normal_exit(self):
        previous = signal.getsignal(signal.SIGTERM)
        with termination_guard():
            self.assertNotEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_term_reaps_only_owned_child(self):
        code = '''
import subprocess, sys, time
from capstone_sim.runner import termination_guard, stop_owned_process
try:
    with termination_guard():
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                 start_new_session=True, stdin=subprocess.DEVNULL)
        try:
            print(child.pid, flush=True)
            time.sleep(60)
        finally:
            record = stop_owned_process(child, 'px4')
            assert record['exit_code'] is not None, record
except RuntimeError as error:
    print(str(error), flush=True)
'''
        process = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        child_pid = int(process.stdout.readline())
        try:
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn("Runner received SIGTERM", stdout)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
