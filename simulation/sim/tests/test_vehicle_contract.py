"""Synthetic generic recording preserves the separate, fixed X500 AI contract."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema
from capstone_sim.recording import TOPICS, Recorder, dump, integrity, sha, verify

ROOT = Path(__file__).resolve().parents[2]


class VehicleContractTest(unittest.TestCase):
    def make_run(self, folder):
        run = Path(folder) / "synthetic_vehicle"
        for part in ("observations", "native_logs", "evaluation", "reference"):
            (run / part).mkdir(parents=True)
        dump(
            run / "run.json",
            dict(run_id=run.name, status="completed", origin="synthetic_contract_fixture"),
        )
        return run

    def profile(self, count):
        return dict(
            id="synthetic_vertical_multirotor",
            revision="synthetic-physical-fixture",
            rotor_count=count,
            motor_order=[f"rotor_{i}" for i in range(count)],
            battery=dict(ns=6, np=2, capacity_ah=10.0, profile_id="synthetic_pack"),
            ai_compatible=False,
            simulation_profile_manifest="reference/vehicle-profile.json",
        )

    def bind(self, run, profile):
        dump(
            run / "reference/vehicle-profile.json",
            dict(
                schema="drone.simulation.profile/1",
                profile_id=profile["id"],
                observation_vehicle=profile,
            ),
        )

    def record(self, run, count, *, profile=None):
        if profile is not None:
            self.bind(run, profile)
        recorder = Recorder(run, "synthetic_contract_fixture", vehicle_profile=profile)
        events = []
        for topic, (_, units, _) in TOPICS.items():
            fields = dict.fromkeys(units, 0.0)
            for field in fields:
                length = {"output": count, "xyz": 3, "q": 4, "velocity": 3}.get(field)
                if length:
                    fields[field] = [500.0] * length if field == "output" else [0.0] * length
            if "q" in fields:
                fields["q"][0] = 1.0
            events.append(dict(topic=topic, sample_time_us=10, available_time_us=20, fields=fields))
        (run / "native_logs/capture.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events)
        )
        recorder.close()
        integrity(run)
        return json.loads((run / "observations/metadata.json").read_text())

    def validator(self, name):
        return jsonschema.Draft202012Validator(json.loads((ROOT / "contracts" / name).read_text()))

    def test_four_six_eight_rotors_have_separate_raw_contract(self):
        for count in (4, 6, 8):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp:
                run = self.make_run(temp)
                metadata = self.record(run, count, profile=self.profile(count))
                self.validator("generic-metadata.schema.json").validate(metadata)
                self.assertFalse(metadata["ai_compatible"])
                self.assertEqual(metadata["vehicle"]["rotor_count"], count)
                self.assertEqual(metadata["vehicle"]["nominal_series_cells"], 6)
                self.assertEqual(
                    metadata["simulation_profile_manifest"]["sha256"],
                    sha(run / "reference/vehicle-profile.json"),
                )
                for line in (run / "observations/streams.jsonl").read_text().splitlines():
                    event = json.loads(line)
                    self.validator("generic-raw-event.schema.json").validate(event)
                    self.assertEqual(event["schema"], "drone.raw.event/3.0")
                    with self.assertRaises(jsonschema.ValidationError):
                        self.validator("raw-event.schema.json").validate(event)
                result = verify(run)
                self.assertEqual(result["status"], "passed", result)
                self.assertEqual(result["C9"], "not_supported_generic_simulator_only")
                self.assertFalse(result["ai_compatible"])

    def test_legacy_x500_recording_remains_four_motor_schema_two(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(temp)
            metadata = self.record(run, 4)
            self.validator("metadata.schema.json").validate(metadata)
            self.assertEqual(metadata["vehicle"]["profile_id"], "x500_reference")
            self.assertNotIn("ai_compatible", metadata)
            self.assertNotIn("simulation_profile_manifest", metadata)
            self.assertEqual(verify(run)["status"], "passed")
            for line in (run / "observations/streams.jsonl").read_text().splitlines():
                self.validator("raw-event.schema.json").validate(json.loads(line))

    def test_output_count_must_match_metadata_even_within_generic_schema_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(temp)
            self.record(run, 6, profile=self.profile(8))
            result = verify(run)
            self.assertEqual(result["status"], "failed")
            self.assertIn("array length", result["errors"])

    def test_legacy_contract_does_not_silently_accept_six_rotor_data(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(temp)
            self.record(run, 6)
            self.assertIn("array length", verify(run)["errors"])

    def test_profile_manifest_changes_are_rejected_even_after_integrity_refresh(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(temp)
            self.record(run, 8, profile=self.profile(8))
            path = run / "reference/vehicle-profile.json"
            record = json.loads(path.read_text())
            record["observation_vehicle"]["battery"]["capacity_ah"] = 99
            dump(path, record)
            integrity(run)
            result = verify(run)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("digest mismatch" in e for e in result["errors"]))

    def test_manifest_and_observation_identity_must_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(temp)
            self.record(run, 8, profile=self.profile(8))
            path = run / "observations/metadata.json"
            metadata = json.loads(path.read_text())
            metadata["vehicle"]["battery_profile_id"] = "another_pack"
            dump(path, metadata)
            integrity(run)
            result = verify(run)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("differs from its profile" in e for e in result["errors"]))

    def test_generic_metadata_cannot_be_relabelled_as_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(temp)
            self.record(run, 4, profile=self.profile(4))
            path = run / "observations/metadata.json"
            metadata = json.loads(path.read_text())
            metadata["schema"] = "drone.observation.metadata/2.0"
            dump(path, metadata)
            integrity(run)
            self.assertIn("metadata contract/schema", verify(run)["errors"])

    def test_reserved_ai_profile_or_wrong_motor_order_rejected_before_recording(self):
        for damage in ("reserved", "duplicate", "missing", "unsupported", "ai", "capacity", "path"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                run = self.make_run(temp)
                profile = self.profile(8)
                if damage == "reserved":
                    profile["id"] = "x500_reference"
                elif damage == "duplicate":
                    profile["motor_order"][1] = profile["motor_order"][0]
                elif damage == "missing":
                    profile["motor_order"].pop()
                elif damage == "unsupported":
                    profile["rotor_count"] = 9
                elif damage == "ai":
                    profile["ai_compatible"] = True
                elif damage == "capacity":
                    profile["battery"]["capacity_ah"] = float("inf")
                else:
                    profile["simulation_profile_manifest"] = "../outside.json"
                if damage == "capacity":
                    # The source file stays finite; validation must reject the active input first.
                    self.bind(run, self.profile(8))
                else:
                    self.bind(run, profile)
                with self.assertRaises((ValueError, jsonschema.ValidationError)):
                    Recorder(run, "synthetic_contract_fixture", vehicle_profile=profile)
                self.assertFalse((run / "observations/streams.jsonl").exists())

    def test_existing_x500_ai_consumer_rejects_generic_simulator_profile(self):
        sys.path.insert(0, str(ROOT.parent / "ai/src"))
        from capstone_ai.mapping import validate_metadata

        config = json.loads((ROOT.parent / "ai/configs/preprocessing.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(temp)
            metadata = self.record(run, 8, profile=self.profile(8))
            with self.assertRaisesRegex(ValueError, "metadata schema.*X500 motor mapping"):
                validate_metadata(metadata, config)
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(temp)
            metadata = self.record(run, 4)
            legacy = copy.deepcopy(config)
            legacy.pop("battery_profile_id", None)
            legacy["nominal_capacity_ah"] = 6
            validate_metadata(metadata, legacy)


if __name__ == "__main__":
    unittest.main()
