"""Generic recovery is an observation fragment; resets retain approved vehicle identity."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_vehicle_contract as fixtures
from capstone_sim import collection_config
from capstone_sim import control as controls
from capstone_sim.recording import dump, integrity, sha, verify_generic_metadata
from capstone_sim.recovery import recover


class VehicleRecoveryControlTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.fixture = fixtures.VehicleContractTest()
        self.run = self.fixture.make_run(self.folder)

    def custom_run(self):
        profile = self.fixture.profile(8)
        metadata = self.fixture.record(self.run, 8, profile=profile)
        path = self.run / "reference/vehicle-profile.json"
        manifest = json.loads(path.read_text())
        manifest.update(source_profile=str(self.folder / "original.yaml"), recipe_sha256="a" * 64)
        dump(path, manifest)
        metadata["simulation_profile_manifest"]["sha256"] = sha(path)
        dump(self.run / "observations/metadata.json", metadata)
        dump(
            self.run / "run.json",
            dict(
                run_id=self.run.name,
                status="completed",
                configuration="vehicle_profile:" + profile["id"],
                instance=6,
            ),
        )
        dump(
            self.run / "evaluation/scenario.json",
            dict(
                name="battery_resistance_step",
                seed=81,
                requested_duration_s=240,
                requested_real_time_factor=4,
                parameters=dict(fault_at=187),
            ),
        )
        integrity(self.run)
        return metadata, manifest

    def test_recovery_preserves_bound_profile_paths_without_claiming_a_complete_run(self):
        metadata, _ = self.custom_run()
        stream = self.run / "observations/streams.jsonl"
        with stream.open("ab") as target:
            target.write(b'{"schema":')
        before = {
            p.relative_to(self.run): p.read_bytes() for p in self.run.rglob("*") if p.is_file()
        }
        output = self.folder / "fragment"
        report = recover(self.run, output)
        self.assertEqual(
            before,
            {p.relative_to(self.run): p.read_bytes() for p in self.run.rglob("*") if p.is_file()},
        )
        self.assertEqual(report["artifact_kind"], "observation_fragment")
        for field in (
            "full_run",
            "closed_watermark_created",
            "training_eligible",
            "new_flight",
            "ai_compatible",
        ):
            self.assertFalse(report[field])
        self.assertEqual(report["discarded_incomplete_bytes"], 10)
        self.assertEqual(report["C9"], "not_verified_recovery_fragment")
        self.assertEqual(
            (output / "metadata.json").read_bytes(),
            (self.run / "observations/metadata.json").read_bytes(),
        )
        verify_generic_metadata(output, metadata)
        self.assertTrue((output / metadata["simulation_profile_manifest"]["path"]).is_file())
        self.assertFalse((output / "vehicle-profile.json").exists())
        self.assertFalse((output / "run.json").exists())
        self.assertFalse((output / "observations/watermarks.jsonl").exists())
        self.assertEqual((output / "streams.jsonl").read_bytes(), stream.read_bytes()[:-10])

    def test_recovery_rejects_profile_tampering_before_creating_output(self):
        self.custom_run()
        path = self.run / "reference/vehicle-profile.json"
        record = json.loads(path.read_text())
        record["profile_id"] = "a_different_vehicle"
        dump(path, record)
        output = self.folder / "fragment"
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            recover(self.run, output)
        self.assertFalse(output.exists())

    def test_recovery_refuses_another_run_or_clock_in_a_generic_fragment(self):
        self.custom_run()
        path = self.run / "observations/streams.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for field in ("run_id", "clock_id"):
            with self.subTest(field=field):
                other = copy.deepcopy(rows)
                other[0][field] = "another_identity"
                path.write_text("".join(json.dumps(row) + "\n" for row in other))
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    recover(self.run, self.folder / ("fragment-" + field))

    def test_custom_reset_retains_profile_instance_seed_duration_speed_and_onset(self):
        _, manifest = self.custom_run()
        with (
            patch(
                "capstone_sim.vehicle_runtime.read_prepared", return_value=(self.folder, manifest)
            ),
            patch.object(controls, "run_flight", return_value={"not_executed": True}) as flight,
        ):
            result = controls.control(self.run, "reset")
        self.assertEqual(result, {"not_executed": True})
        args, options = flight.call_args
        self.assertEqual(args, ("battery_resistance_step", 81, 240, 6, self.run.name))
        self.assertEqual(options["vehicle_profile"], manifest["source_profile"])
        self.assertEqual(options["fault_at_s"], 187)
        self.assertEqual(options["real_time_factor"], 4)
        self.assertNotIn("configuration", options)

    def test_changed_recipe_or_state_vehicle_cannot_be_reset(self):
        _, manifest = self.custom_run()
        changed = dict(manifest, recipe_sha256="b" * 64)
        with (
            patch(
                "capstone_sim.vehicle_runtime.read_prepared", return_value=(self.folder, changed)
            ),
            patch.object(controls, "run_flight") as flight,
        ):
            with self.assertRaisesRegex(ValueError, "profile changed"):
                controls.control(self.run, "reset")
            flight.assert_not_called()
        state_path = self.run / "run.json"
        state = json.loads(state_path.read_text())
        state["configuration"] = "vehicle_profile:another_vehicle"
        dump(state_path, state)
        with patch.object(controls, "run_flight") as flight:
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                controls.control(self.run, "reset")
            flight.assert_not_called()

    def collection_run(self, scenario, *, variation=None):
        dump(
            self.run / "run.json",
            dict(run_id=self.run.name, status="completed", configuration="collection", instance=7),
        )
        old = dict(
            name=scenario,
            seed=44,
            requested_duration_s=240,
            requested_real_time_factor=2,
            parameters=dict(fault_at=180) if scenario.endswith(("step", "ramp")) else {},
            normal_variation=variation,
        )
        dump(self.run / "evaluation/scenario.json", old)
        return old

    def test_collection_reset_retains_required_fault_onset_and_original_variation(self):
        self.collection_run("battery_resistance_step")
        with patch.object(controls, "run_flight", return_value={}) as flight:
            controls.control(self.run, "reset")
        args, options = flight.call_args
        self.assertEqual(args, ("battery_resistance_step", 44, 240, 7, self.run.name))
        self.assertEqual(options["configuration"], "collection")
        self.assertEqual(options["fault_at_s"], 180)
        collection_config.validate_request(args[0], args[2], options["fault_at_s"])
        old = self.collection_run("hover_normal", variation=dict(soc=0.95, path="frozen_fixture"))
        with patch.object(controls, "run_flight", return_value={}) as flight:
            controls.control(self.run, "reset")
        self.assertEqual(flight.call_args.kwargs["normal_variation"], old["normal_variation"])

    def test_collection_scenario_change_uses_new_onset_without_the_old_normal_variation(self):
        self.collection_run("hover_normal", variation=dict(soc=0.95, path="frozen_fixture"))
        with patch.object(controls, "run_flight", return_value={}) as flight:
            controls.control(self.run, "set_scenario", scenario="battery_resistance_ramp")
        args, options = flight.call_args
        self.assertEqual(args[0], "battery_resistance_ramp")
        self.assertEqual(options["fault_at_s"], 180)
        self.assertIsNone(options["normal_variation"])
        collection_config.validate_request(args[0], args[2], options["fault_at_s"])

    def test_reset_never_replaces_an_active_run(self):
        self.custom_run()
        path = self.run / "run.json"
        state = json.loads(path.read_text())
        state["status"] = "flying"
        dump(path, state)
        with patch.object(controls, "run_flight") as flight:
            with self.assertRaisesRegex(ValueError, "finalize the current run"):
                controls.control(self.run, "reset")
            flight.assert_not_called()

    def reference_run(self, **changes):
        state = dict(run_id=self.run.name, status="completed", instance=3)
        state.update(changes)
        dump(self.run / "run.json", state)
        dump(self.run / "evaluation/scenario.json", dict(
            name="hover_normal", seed=18, requested_duration_s=120,
            requested_real_time_factor=2, parameters={},
        ))

    def test_reference_reset_and_scenario_change_keep_nonzero_instance(self):
        self.reference_run()
        for operation, scenario in (("reset", None), ("set_scenario", "yaw_normal")):
            with patch.object(controls, "run_flight", return_value={}) as flight:
                controls.control(self.run, operation, scenario=scenario)
            self.assertEqual(flight.call_args.args, (scenario or "hover_normal", 18, 120, 3, self.run.name))
            self.assertNotIn("configuration", flight.call_args.kwargs)

    def test_legacy_missing_instance_is_preserved_and_never_assumed_zero(self):
        self.reference_run()
        path = self.run / "run.json"
        state = json.loads(path.read_text())
        del state["instance"]
        dump(path, state)
        original = path.read_bytes()
        for operation in ("reset", "set_scenario"):
            with patch.object(controls, "run_flight") as flight:
                with self.assertRaisesRegex(ValueError, "cannot assume instance 0"):
                    controls.control(self.run, operation, scenario="yaw_normal")
                flight.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_reference_instance_or_unknown_configuration_is_rejected(self):
        for instance in (True, "3", 3.0, -1, 10):
            self.reference_run(instance=instance)
            with patch.object(controls, "run_flight") as flight:
                with self.assertRaisesRegex(ValueError, "instance is missing or invalid"):
                    controls.control(self.run, "reset")
                flight.assert_not_called()
        self.reference_run(configuration="unknown")
        with patch.object(controls, "run_flight") as flight:
            with self.assertRaisesRegex(ValueError, "configuration is not supported"):
                controls.control(self.run, "reset")
            flight.assert_not_called()

    def test_legacy_collection_reset_emits_generic_configuration(self):
        self.collection_run("hover_normal")
        path = self.run / "run.json"
        state = json.loads(path.read_text())
        state["configuration"] = collection_config.LEGACY_NAME
        dump(path, state)
        with patch.object(controls, "run_flight", return_value={}) as flight:
            controls.control(self.run, "reset")
        self.assertEqual(flight.call_args.args[3], 7)
        self.assertEqual(flight.call_args.kwargs["configuration"], "collection")


if __name__ == "__main__":
    unittest.main()
