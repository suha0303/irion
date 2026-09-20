import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema
from capstone_sim.progress_guard import SimulationClockGuard, SimulationClockStalled
from test_fixtures import RETAINED, ROOT, retained_records


class RunSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((ROOT / "contracts/run.schema.json").read_text())
        cls.validator = jsonschema.Draft202012Validator(cls.schema)
        row = json.loads((RETAINED / "manifest.json").read_text())["runs"][0]
        cls.record = json.loads((RETAINED / "runs" / row["run_id"] / "run.json").read_text())

    def test_legacy_required_contract_unchanged(self):
        required = {"run_id", "parent_attempt_id", "origin", "status", "termination_reason",
                    "start_utc", "end_utc", "clock", "observation_metadata", "source_lock",
                    "native_logs"}
        self.assertEqual(set(self.schema["required"]), required)
        self.validator.validate({key: self.record[key] for key in required})
        for key in required:
            broken = copy.deepcopy(self.record)
            del broken[key]
            self.assertFalse(self.validator.is_valid(broken), key)

    def test_failure_and_stall_diagnostics(self):
        record = copy.deepcopy(self.record)
        record.update(status="failed", primary_failure=dict(stage="flying", reason=""),
                      secondary_failures=[dict(stage="finalization_failure", reason="missing")],
                      capture_health_failure=dict(passed=False))
        guard = SimulationClockGuard(10)
        with self.assertRaises(SimulationClockStalled) as failure:
            guard.observe(310, 0, "waiting")
        record["clock_stall_evidence"] = failure.exception.evidence
        self.validator.validate(record)
        record["clock_stall_evidence"]["limit_wall_s"] = 0
        self.assertFalse(self.validator.is_valid(record))

    def test_unknown_fields_and_bad_diagnostic_types_rejected(self):
        for key, value in {
            "unregistered_field": True, "instance": -1, "configuration": 1,
            "mavlink_source_checked": "true", "gz_partition": "", "wait_policy": [],
            "progress": [], "process_shutdown": ["not an object"],
            "px4_socket_cleanup": False, "capture_health_failure": [],
            "primary_failure": {"stage": "flying"}, "secondary_failures": ["error"],
            "clock_stall_evidence": {"phase": "waiting"},
        }.items():
            with self.subTest(key=key):
                self.assertFalse(self.validator.is_valid(dict(self.record, **{key: value})))

    def test_generated_schemas_match_committed_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(ROOT / "sim/src"))
            subprocess.run([sys.executable, "-B", str(ROOT / "sim/tools/generate_schemas.py"),
                            "--output-dir", directory], env=env, check=True, timeout=10)
            for path in (ROOT / "contracts").glob("*.schema.json"):
                self.assertEqual(path.read_bytes(), (Path(directory) / path.name).read_bytes(),
                                 path.name)


class RetainedFixtureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.manifest = json.loads((RETAINED / "manifest.json").read_text())
        self.manifest["runs"] = self.manifest["runs"][:1]
        self.row = self.manifest["runs"][0]
        shutil.copytree(RETAINED / "runs" / self.row["run_id"],
                        self.folder / "runs" / self.row["run_id"])
        self.write_manifest()

    def write_manifest(self):
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest))

    def test_snapshot_is_independent_of_retired_paths(self):
        # This temporary root contains only retained records, no campaign or raw runs.
        self.assertEqual(len(list(retained_records(self.folder))), 1)

    def test_modified_file_rejected(self):
        (self.folder / "runs" / self.row["run_id"] / "run.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            list(retained_records(self.folder))

    def test_missing_file_fails_instead_of_skipping(self):
        (self.folder / "runs" / self.row["run_id"] / "evaluation/events.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            list(retained_records(self.folder))

    def test_final_test_role_and_duplicates_rejected(self):
        self.row["role"] = "final_test"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "identity/role"):
            list(retained_records(self.folder))
        self.row["role"] = "anomaly_dev"
        self.manifest["runs"].append(self.row)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "identity/role"):
            list(retained_records(self.folder))
