"""Mission qualification must not erase start/stop violations or admit mission violations."""

import json
import tempfile
import unittest
from pathlib import Path

from capstone_sim import model_envelope


class ModelEnvelopeTest(unittest.TestCase):
    def report(self, rows, timeline=None):
        if timeline is None:
            timeline = [dict(state="airborne", sim_time=1), dict(state="landing", sim_time=3)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "truth.jsonl"
            path.write_text("".join(json.dumps(dict(time_us=t, model_valid=valid,
                inplane_advance_ratio_defined=defined, max_inplane_advance_ratio=ratio)) + "\n"
                for t, valid, defined, ratio in rows))
            return model_envelope.check(path, timeline)

    def test_unpowered_start_and_spin_down_remain_visible_outside_mission(self):
        report = self.report([(0, False, False, 0), (1_000_000, True, True, .06),
                              (3_000_000, True, True, .07), (4_000_000, False, True, .85)])
        self.assertTrue(report["normal_eligibility_passed"])
        self.assertEqual(report["samples"], 4)
        self.assertEqual(report["outside_samples"], 2)
        self.assertEqual(report["segments"]["mission"]["samples"], 2)
        self.assertEqual(report["segments"]["before_mission"]["undefined_flow_samples"], 1)
        self.assertEqual(report["segments"]["after_mission"]["max_inplane_advance_ratio"], .85)

    def test_violation_at_either_inclusive_boundary_blocks_normal(self):
        for stamp in (1_000_000, 2_000_000, 3_000_000):
            report = self.report([(stamp, False, True, .11)])
            self.assertFalse(report["normal_eligibility_passed"])
            self.assertEqual(report["segments"]["mission"]["outside_samples"], 1)

    def test_missing_empty_reversed_or_nonfinite_mission_fails_closed(self):
        timelines = ([], [dict(state="airborne", sim_time=1)],
                     [dict(state="airborne", sim_time=3), dict(state="landing", sim_time=1)],
                     [dict(state="airborne", sim_time=float("nan")), dict(state="landing", sim_time=3)])
        for timeline in timelines:
            self.assertFalse(self.report([(2_000_000, True, True, .06)], timeline)["normal_eligibility_passed"])
        self.assertFalse(self.report([])["normal_eligibility_passed"])
