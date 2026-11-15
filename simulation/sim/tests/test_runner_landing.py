"""A native LAND request permits the transition, never certifies completion."""
import unittest
from unittest.mock import patch

from capstone_sim.runner import native_land_requested


class RunnerLandingTest(unittest.TestCase):
    def test_publication_allows_waiting_despite_stale_position_time(self):
        request = dict(kind="native_land_publication_not_activation",
                       actual_start_us=60_004_000, duration_us=480_000_000,
                       actual_time_us=540_004_000)
        # The failed real run's latest position was only 539.756 seconds.
        with patch("capstone_sim.joint_phase.receipt", return_value=request) as read:
            self.assertTrue(native_land_requested("owned-run", "joint-native-consumption-c2-v2"))
        read.assert_called_once_with("owned-run", "land-request")

    def test_missing_early_or_wrong_receipt_does_not_hide_mode_loss(self):
        records = [None,
                   dict(kind="native_land_publication_not_activation",
                        actual_start_us=60_000_000, duration_us=480_000_000,
                        actual_time_us=539_000_000),
                   dict(kind="unrelated", actual_start_us=60_000_000,
                        duration_us=480_000_000, actual_time_us=540_000_000)]
        for record in records:
            with self.subTest(record=record), patch("capstone_sim.joint_phase.receipt", return_value=record):
                self.assertFalse(native_land_requested("owned-run", "joint-native-consumption-c2-v2"))
        with patch("capstone_sim.joint_phase.receipt") as read:
            self.assertFalse(native_land_requested("owned-run", None))
            read.assert_not_called()

    def test_invalid_receipt_still_fails(self):
        with patch("capstone_sim.joint_phase.receipt", side_effect=ValueError("bad identity")):
            with self.assertRaises(ValueError):
                native_land_requested("owned-run", "joint-native-consumption-c2-v2")
