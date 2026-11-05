import csv
import json
import tempfile
import unittest
from pathlib import Path

from capstone_sim import joint_phase
from capstone_sim.bounded_wind import definition, native_header, profile
from capstone_sim.normal_variation import sample


class JointPhaseTest(unittest.TestCase):
    def test_explicit_scope_and_identity_required(self):
        spec = sample("hover_normal", 3)
        spec.update(wind_e=0., wind_n=0., gust_sigma=0., wind_profile=profile("strong"))
        for duration in (180,480):
            joint_phase.validate(joint_phase.REVISION, "collection", duration, spec, False, "pair_1")
        for revision, configuration, duration, paired_id in [(None, "collection", 480, "pair_1"),
                (joint_phase.REVISION, "collection", 900, "pair_1"),
                (joint_phase.REVISION, None, 480, "pair_1"),
                (joint_phase.REVISION, "collection", 480, None)]:
            with self.assertRaises(ValueError):
                joint_phase.validate(revision, configuration, duration, spec, False, paired_id)

    def test_preload_native_float_rows_and_missing_receipt_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "evaluation").mkdir()
            info = joint_phase.prepare(run, sample("hover_normal", 9))
            with (run / "evaluation/joint-reference.csv").open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual((len(rows), rows[0][0], rows[-1][0]), (4801, "0", "480000000"))
            self.assertEqual(info["reference_rows"], 4801)
            with self.assertRaisesRegex(ValueError, "Missing actual"):
                joint_phase.verify(run)

    def test_legacy_header_unchanged_and_new_clock_explicit(self):
        value = profile("strong")
        old = definition(3, value)
        new = definition(3, value, joint_phase.REVISION)
        self.assertEqual(native_header(old)["schema"], "drone.bounded-wind-trace/1")
        self.assertNotIn("phase_revision", old)
        self.assertEqual(native_header(new)["schema"], "drone.bounded-wind-trace/2")
        self.assertEqual(new["phase_revision"], joint_phase.REVISION)

    def test_ack_or_metadata_cannot_substitute_native_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "evaluation").mkdir()
            (run / "evaluation/joint-origin.json").write_text(json.dumps(dict(actual_time_us=60000000, acknowledged=True)))
            with self.assertRaisesRegex(ValueError, "Invalid actual"):
                joint_phase.receipt(run, "origin")

    def test_only_explicit_supported_durations(self):
        for invalid in (True, 0, 179, 180.1, 479, 900, float("nan")):
            with self.assertRaises(ValueError):
                joint_phase.duration_us(invalid)

    def test_short_reference_is_same_function_prefix_without_time_rescaling(self):
        spec = sample("hover_normal", 9)
        short = list(joint_phase.reference_rows(spec, 180))
        full = list(joint_phase.reference_rows(spec))
        self.assertEqual(short, full[:1801])
        self.assertEqual(len(full),4801)
        self.assertEqual(short[-1][0],180000000)
