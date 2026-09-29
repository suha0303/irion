import hashlib
import json
import unittest
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
RETAINED = ROOT / "sim/tests/fixtures/metadata"


def retained_records(folder):
    """Validate preserved development records, without requiring retired raw runs."""
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema"] != "x500.retained-schema-fixture/1" or not manifest["runs"]:
        raise ValueError("Missing retained development schema fixture")
    seen = set()
    for row in manifest["runs"]:
        run_id = row["run_id"]
        if (len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id)
                or run_id in seen or row["role"] not in {"normal_dev_eval", "anomaly_dev"}):
            raise ValueError("Invalid development fixture identity/role")
        seen.add(run_id)
        records = {}
        for name in ("run.json", "observations/metadata.json", "evaluation/events.jsonl"):
            body = (folder / "runs" / run_id / name).read_bytes()
            expected = row["files"][name]
            if (len(body) != expected["bytes"]
                    or hashlib.sha256(body).hexdigest() != expected["sha256"]):
                raise ValueError("Retained fixture hash mismatch: " + run_id + "/" + name)
            records[name] = body
        yield run_id, records


class FixtureTest(unittest.TestCase):
    def test_focused_contract_vectors(self):
        base = ROOT / "artifacts/fixtures"
        validator = jsonschema.Draft202012Validator(
            json.loads((ROOT / "contracts/raw-event.schema.json").read_text())
        )
        for name in [
            "delay_reordering",
            "missing_long_gap",
            "clock_reset_instance",
            "separate_truth",
        ]:
            p = base / name
            self.assertEqual(
                json.loads((p / "fixture.json").read_text())["origin"], "synthetic_contract_fixture"
            )
            for line in (p / "streams.jsonl").read_text().splitlines():
                validator.validate(json.loads(line))
        late = [
            json.loads(s)
            for s in (base / "delay_reordering/streams.jsonl").read_text().splitlines()
        ]
        self.assertGreater(late[0]["available_time_us"], late[1]["available_time_us"])
        self.assertLess(late[0]["sample_time_us"], late[1]["sample_time_us"])
        reset = [
            json.loads(s)
            for s in (base / "clock_reset_instance/streams.jsonl").read_text().splitlines()
        ]
        self.assertNotEqual(reset[0]["clock_id"], reset[1]["clock_id"])
        self.assertNotEqual(reset[0]["instance"], reset[1]["instance"])
        self.assertFalse((base / "partial_end/streams.jsonl").read_bytes().endswith(b"\n"))

    def test_retained_actual_evaluation_schema(self):
        # Historical campaign-final.json is evidence, not a live dataset locator.
        # These byte-identical metadata/events survive raw-data retirement.
        schema = json.loads((ROOT / "contracts/evaluation-event.schema.json").read_text())
        validator = jsonschema.Draft202012Validator(schema)
        count = 0
        for run_id, records in retained_records(RETAINED):
            count += 1
            for schema_name, filename in [
                ("run", "run.json"),
                ("metadata", "observations/metadata.json"),
            ]:
                record = json.loads(records[filename])
                self.assertEqual(record["run_id"], run_id)
                self.assertEqual(record["origin"], "px4_sitl")
                contract = json.loads(
                    (ROOT / "contracts" / (schema_name + ".schema.json")).read_text()
                )
                jsonschema.Draft202012Validator(contract).validate(record)
            for line in records["evaluation/events.jsonl"].splitlines():
                e = json.loads(line)
                self.assertEqual(e["run_id"], run_id)
                validator.validate(e)
                if e["onset_kind"] == "pre_existing":
                    self.assertIsNone(e["physical_onset_time_us"])
                    self.assertEqual(e["applied_time_us"], 0)
                else:
                    self.assertGreaterEqual(e["applied_time_us"], e["requested_time_us"])
                    if e["onset_kind"] == "in_flight_ramp":
                        self.assertGreater(e["degradation_boundary_time_us"], e["applied_time_us"])
        self.assertEqual(count, 136)
