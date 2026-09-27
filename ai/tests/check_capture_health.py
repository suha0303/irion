"""No SITL: reproduce captured stream stall against the live PX4 time."""

import unittest
from types import SimpleNamespace

from capstone_sim.runner import capture_health


class CaptureHealthChecks(unittest.TestCase):
    def test_recorded_stall(self):
        result = capture_health(SimpleNamespace(through=14070000, count=3866), 19.584, 12.018)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["lag_sim_seconds"], 5.514)

    def test_sim_clock_boundaries(self):
        self.assertTrue(
            capture_health(SimpleNamespace(through=199600000, count=10000), 199.616, 12)["passed"]
        )
        self.assertTrue(
            capture_health(SimpleNamespace(through=15000000, count=10), 20, 12)["passed"]
        )
        self.assertFalse(
            capture_health(SimpleNamespace(through=15000000, count=10), 20.001, 12)["passed"]
        )

    def test_missing_capture_startup_grace(self):
        recorder = SimpleNamespace(through=-1, count=0)
        self.assertTrue(capture_health(recorder, 16, 12)["passed"])
        self.assertFalse(capture_health(recorder, 18, 12)["passed"])


if __name__ == "__main__":
    unittest.main()
