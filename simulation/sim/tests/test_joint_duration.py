import csv
import json
import tempfile
import unittest
from pathlib import Path

from capstone_sim import joint_phase as generation
from capstone_sim.normal_variation import sample

audit = generation


class Consumption180Test(unittest.TestCase):
    duration_s = 180
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.folder = self.run / "evaluation"
        self.folder.mkdir()
        spec = sample("hover_normal", 9)
        spec["path"]["kind"] = "hover"
        info = generation.prepare(self.run, spec, self.duration_s)
        (self.folder / "scenario.json").write_text(json.dumps(dict(operation_variation=spec, joint_phase_policy=info, requested_duration_s=self.duration_s)))
        base = dict(schema="drone.joint-phase-native-receipt/1", revision=generation.REVISION,
            policy_sha256=info["policy_sha256"], actual_start_us=60_000_000,
            consumer="PositionControl::setInputSetpoint", clock="px4-boot", duration_us=self.duration_s*1_000_000)
        (self.folder / "joint-origin.json").write_text(json.dumps(dict(base, actual_time_us=60_000_000)))
        (self.folder / "joint-landing.json").write_text(json.dumps(dict(base, actual_time_us=60_010_000+self.duration_s*1_000_000)))
        with (self.folder / "joint-reference.csv").open() as stream:
            self.reference = list(csv.reader(stream))
        self.rows = [[index+1, 60_000_000+index*10_000, index*10_000,
                      index//10, *self.reference[index//10][1:]] for index in range(self.duration_s*100+1)]

    def write(self):
        with (self.folder / "joint-consumption.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow("sequence,source_us,phase_us,index,px,py,pz,vx,vy,vz,ax,ay,az,yaw".split(","))
            writer.writerows(self.rows)

    def duplicate(self):
        self.rows.insert(10_001, list(self.rows[10_000]))
        for index, row in enumerate(self.rows):
            row[0] = index+1

    def test_complete_declared_interval_and_native_tail(self):
        self.write()
        result = audit.verify(self.run)
        self.assertEqual(result["duration_us"], self.duration_s*1_000_000)
        self.assertEqual(result["reference_rows_seen"], self.duration_s*10+1)
        self.assertEqual(result["terminal_hold_tail_us"], 10_000)
        self.assertEqual(result["same_source_time_consumptions"], 0)

    def test_exact_same_time_payload_keeps_every_row_and_counts_repeat(self):
        self.duplicate()
        self.write()
        result = audit.verify(self.run)
        self.assertEqual(result["consumed_samples"], self.duration_s*100+2)
        self.assertEqual(result["same_source_time_consumptions"], 1)
        self.assertEqual(result["unique_source_time_samples"], self.duration_s*100+1)
        self.assertEqual(result["reference_rows_seen"], self.duration_s*10+1)
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
        self.duplicate()
        self.write()
        with self.assertRaisesRegex(ValueError, "Incomplete actual"):
            audit.verify(self.run)

    def test_declared_duration_and_native_receipt_must_agree(self):
        self.write()
        path = self.folder / "joint-origin.json"
        value = json.loads(path.read_text())
        value["duration_us"] = 480_000_000 if self.duration_s == 180 else 180_000_000
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Invalid actual"):
            audit.verify(self.run)

    def test_landing_budget_cannot_be_shortened_or_extended(self):
        self.write()
        path = self.folder / "joint-landing.json"
        value = json.loads(path.read_text())
        for offset in (-1, 100_001):
            value["actual_time_us"] = 60_000_000+self.duration_s*1_000_000+offset
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Actual joint phase duration"):
                audit.verify(self.run)


class Consumption480Test(Consumption180Test):
    duration_s = 480
