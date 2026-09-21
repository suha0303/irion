import json
import tempfile
import unittest
from pathlib import Path

import jsonschema
from capstone_sim.recording import TOPICS, Recorder, dump, integrity, verify

ROOT = Path(__file__).resolve().parents[2]


class ContractTest(unittest.TestCase):
    def make_run(self, root):
        run = root / "contract_fixture"
        for part in ["observations", "native_logs", "evaluation", "reference"]:
            (run / part).mkdir(parents=True)
        dump(
            run / "run.json",
            {
                "run_id": run.name,
                "origin": "synthetic_contract_fixture",
                "status": "completed",
            },
        )
        return run

    def test_ninety_seconds_all_roles_and_tied_watermark(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temp:
            run = self.make_run(Path(temp))
            rec = Recorder(run, "synthetic_contract_fixture")
            native = (run / "native_logs/capture.jsonl").open("w")
            for t in range(0, 90000000, 100000):
                for topic, (_, units, _) in TOPICS.items():
                    fields = {k: 0.0 for k in units}
                    if topic == "actuator_outputs":
                        fields = {"output": [500.0] * 4}
                    if topic in ["vehicle_angular_velocity", "vehicle_acceleration"]:
                        fields = {"xyz": [0.0, 0.0, 0.0]}
                    if topic == "vehicle_attitude":
                        fields = {"q": [1.0, 0.0, 0.0, 0.0]}
                    if topic == "trajectory_setpoint":
                        fields = {"velocity": [None, None, None]}
                    native.write(
                        json.dumps(
                            {
                                "topic": topic,
                                "sample_time_us": t,
                                "available_time_us": t + 5000,
                                "fields": fields,
                            }
                        )
                        + "\n"
                    )
                if t % 1000000 == 0:
                    native.flush()
                    rec.consume()
            native.close()
            rec.close()
            integrity(run)
            self.assertEqual(verify(run)["status"], "passed")
            schema = json.loads((ROOT / "contracts/raw-event.schema.json").read_text())
            validator = jsonschema.Draft202012Validator(schema)
            for line in (run / "observations/streams.jsonl").read_text().splitlines():
                validator.validate(json.loads(line))

    def test_partial_line_waits_then_recovers(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temp:
            run = self.make_run(Path(temp))
            rec = Recorder(run, "synthetic_contract_fixture")
            f = run / "native_logs/capture.jsonl"
            e = json.dumps(
                {
                    "topic": "battery_status",
                    "sample_time_us": 10,
                    "available_time_us": 12,
                    "fields": {"voltage_v": 15.0, "current_a": 2.0},
                }
            )
            f.write_text(e[:30])
            rec.consume()
            self.assertEqual(rec.count, 0)
            with f.open("a") as out:
                out.write(e[30:] + "\n" + e[:20])
            rec.consume()
            self.assertEqual(rec.count, 1)
            rec.close()
            self.assertIn(
                "incomplete_final_line",
                (run / "observations/diagnostics.jsonl").read_text(),
            )

    def test_reject_truth_leak_and_future_sample(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temp:
            run = self.make_run(Path(temp))
            rec = Recorder(run, "synthetic_contract_fixture")
            (run / "native_logs/capture.jsonl").write_text(
                json.dumps(
                    {
                        "topic": "battery_status",
                        "sample_time_us": 20,
                        "available_time_us": 10,
                        "fields": {"voltage_v": 15.0, "current_a": 2.0, "soc": 0.9},
                    }
                )
                + "\n"
            )
            rec.close()
            integrity(run)
            result = verify(run)
            self.assertIn("topic fields/leakage", result["errors"])
            self.assertIn("sample after available", result["errors"])


if __name__ == "__main__":
    unittest.main()
