"""Offline frozen-clock replay and child-only native-runtime selection."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from capstone_sim import collection_config
from capstone_sim.progress_guard import (
    SimulationClockGuard,
    SimulationClockStalled,
    record_secondary_failure,
)


class ProgressGuardTests(unittest.TestCase):
    def test_real_frozen_clock_trace_is_bounded_by_wall_time(self):
        guard = SimulationClockGuard(58102.986447581)
        guard.observe(58113.01309104, 14.046, "flying")
        guard.observe(58413.0, 14.046, "flying")
        with self.assertRaises(SimulationClockStalled) as failure:
            guard.observe(58413.014, 14.046, "flying")
        self.assertEqual(failure.exception.evidence["px4_boot_s"], 14.046)
        self.assertEqual(failure.exception.evidence["limit_wall_s"], 300)

    def test_slow_progress_and_long_flight_are_allowed(self):
        guard = SimulationClockGuard(0)
        for wall in range(9001):
            guard.observe(wall, (wall // 299) * .001, "flying")

    def test_no_position_and_invalid_clock(self):
        guard = SimulationClockGuard(0)
        guard.observe(299, 0, "waiting")
        with self.assertRaises(SimulationClockStalled):
            guard.observe(300, 0, "waiting")
        for value in (-1, math.nan, math.inf):
            with self.assertRaises(RuntimeError):
                SimulationClockGuard(0).observe(1, value, "flying")

    def test_finalization_keeps_primary_crash(self):
        state = dict(termination_reason="owned process exited: [-11, None]",
                     primary_failure=dict(stage="waiting", reason="owned process exited: [-11, None]"))
        record_secondary_failure(state, "finalization_failure", FileNotFoundError("truth.jsonl"))
        self.assertEqual(state["termination_reason"], "owned process exited: [-11, None]")
        self.assertEqual(state["secondary_failures"][0]["stage"], "finalization_failure")
        self.assertEqual(state["status"], "failed")

    def test_finalization_error_without_primary_still_fails(self):
        state = dict(termination_reason="mission_landed_disarmed")
        record_secondary_failure(state, "recording_failure", OSError("disk"))
        self.assertEqual(state["termination_reason"], "recording_failure: disk")
        self.assertEqual(state["status"], "failed")

    def test_private_library_only_changes_child_environment(self):
        original = dict(LD_LIBRARY_PATH="/another/approved/lib", GZ_PARTITION="test")
        with patch.object(collection_config, "transport_library", return_value=Path("/private/lib/libgz-transport13.so.13.6.0")):
            child = collection_config.transport_environment(original)
        self.assertEqual(child["LD_LIBRARY_PATH"], "/private/lib:/another/approved/lib")
        self.assertEqual(original["LD_LIBRARY_PATH"], "/another/approved/lib")
        self.assertEqual(child["GZ_PARTITION"], "test")
        with patch.object(collection_config, "transport_library", return_value=None):
            self.assertEqual(collection_config.transport_environment(original), original)

    def test_missing_private_library_cannot_silently_fall_back(self):
        with patch.object(collection_config, "producer_lock_path", return_value=Path("producer-lock-startup-v5.json")), \
                patch.object(collection_config, "ROOT", Path("/nonexistent/x500-test")):
            with self.assertRaisesRegex(ValueError, "missing or replaced"):
                collection_config.transport_library()
