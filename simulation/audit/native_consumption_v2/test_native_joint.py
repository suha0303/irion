import csv
import json
import tempfile
import unittest
from pathlib import Path

from capstone_sim import joint_phase as generation
from capstone_sim.normal_variation import sample

from . import native_joint as audit


class ConsumptionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.folder = self.run / "evaluation"
        self.folder.mkdir()
        spec = sample("hover_normal", 9)
        spec["path"]["kind"] = "hover"
        info = generation.prepare(self.run, spec)
        (self.folder / "scenario.json").write_text(json.dumps(dict(operation_variation=spec, joint_phase_policy=info)))
        base = dict(schema="drone.joint-phase-native-receipt/1", revision=generation.REVISION,
            policy_sha256=info["policy_sha256"], actual_start_us=60_000_000,
            consumer="PositionControl::setInputSetpoint", clock="px4-boot")
        (self.folder / "joint-origin.json").write_text(json.dumps(dict(base, actual_time_us=60_000_000)))
        (self.folder / "joint-landing.json").write_text(json.dumps(dict(base, actual_time_us=540_010_000)))
        with (self.folder / "joint-reference.csv").open() as stream:
            self.reference = list(csv.reader(stream))
        self.rows = [[index+1, 60_000_000+index*10_000, index*10_000,
                      index//10, *self.reference[index//10][1:]] for index in range(48_001)]

    def write(self):
        with (self.folder / "joint-consumption.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow("sequence,source_us,phase_us,index,px,py,pz,vx,vy,vz,ax,ay,az,yaw".split(","))
            writer.writerows(self.rows)

    def duplicate(self):
        self.rows.insert(10_001, list(self.rows[10_000]))
        for index, row in enumerate(self.rows):
            row[0] = index+1

    def test_original_strict_checker_and_fixed_checker_agree_without_repeated_time(self):
        self.write()
        old, new = generation.verify(self.run), audit.verify(self.run)
        for key in old:
            self.assertEqual(old[key], new[key])
        self.assertEqual(new["same_source_time_consumptions"], 0)

    def test_exact_same_time_payload_keeps_every_row_and_counts_repeat(self):
        self.duplicate()
        self.write()
        with self.assertRaisesRegex(ValueError, "source time reversal"):
            generation.verify(self.run)
        result = audit.verify(self.run)
        self.assertEqual(result["consumed_samples"], 48_002)
        self.assertEqual(result["same_source_time_consumptions"], 1)
        self.assertEqual(result["unique_source_time_samples"], 48_001)
        self.assertEqual(result["reference_rows_seen"], 4801)
        self.assertEqual(result["max_consumption_gap_us"], 10_000)

    def test_same_time_conflict_rejected_even_inside_reference_numeric_tolerance(self):
        self.duplicate()
        self.rows[10_001][7] = "1e-10"
        self.write()
        with self.assertRaisesRegex(ValueError, "Conflicting consumption"):
            audit.verify(self.run)

    def test_actual_backwards_time_rejected(self):
        row = self.rows[10_001]
        row[1] = 159_999_000
        row[2] = 99_999_000
        row[3] = 999
        row[4:] = self.reference[999][1:]
        self.write()
        with self.assertRaisesRegex(ValueError, "source time reversal"):
            audit.verify(self.run)

    def test_sequence_phase_and_index_failures_remain_rejected(self):
        for column in (0, 2, 3):
            original = self.rows[100][column]
            self.rows[100][column] = int(original)+1
            self.write()
            with self.assertRaisesRegex(ValueError, "sequence/phase/index"):
                audit.verify(self.run)
            self.rows[100][column] = original

    def test_missing_final_coverage_is_not_filled_by_repeated_rows(self):
        self.rows.pop()
        self.duplicate()
        self.write()
        with self.assertRaisesRegex(ValueError, "Incomplete actual"):
            audit.verify(self.run)

    def test_long_gap_is_not_hidden_by_repeated_time(self):
        del self.rows[10_001:10_006]
        for index, row in enumerate(self.rows):
            row[0] = index+1
        self.write()
        with self.assertRaisesRegex(ValueError, "Incomplete actual"):
            audit.verify(self.run)
