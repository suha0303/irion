"""Preparation binds all three profiles and never launches a simulator or flight."""

import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from capstone_sim import vehicle_runtime as runtime
from capstone_sim.recording import dump


class VehicleRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        shutil.copytree(runtime.ROOT / "profiles/examples/hex_example", self.inputs)
        self.path = self.inputs / "vehicle.yaml"
        self.build = self.root / "prepared"
        self.native = self.root / "native.json"
        self.native.write_text("fixed native runtime")
        for target, value in (("BUILD", self.build),):
            guard = patch.object(runtime, target, value)
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch.object(runtime.collection_config, "assert_frozen")
        guard.start()
        self.addCleanup(guard.stop)
        guard = patch.object(runtime.collection_config, "producer_lock_path", return_value=self.native)
        guard.start()
        self.addCleanup(guard.stop)
        self.commands = []

    def compile_fixture(self, argv, **kwargs):
        self.commands.append(argv)
        if argv[:2] == ["cmake", "--build"]:
            build = Path(argv[2])
            build.mkdir(parents=True)
            (build / "libDronePowertrain.so").write_bytes(b"unit-test compiled output")
        return subprocess.CompletedProcess(argv, 0)

    def prepare(self):
        from capstone_sim import vehicle_assets

        item = runtime.recipe(self.path)
        with patch.object(runtime.subprocess, "run", side_effect=self.compile_fixture):
            result = runtime._prepare(item, runtime.location(item), 2, False, vehicle_assets)
        return result, runtime.read_prepared(self.path)

    def test_inspect_has_no_process_and_binds_pack_and_propulsion(self):
        with patch.object(runtime.subprocess, "run", side_effect=AssertionError("no subprocess")):
            result = runtime.inspection(self.path)
        self.assertEqual(result["rotor_count"], 6)
        self.assertEqual(len(result["inputs"]), 3)
        self.assertFalse(result["actual_flight_started"])
        self.assertFalse(result["training_started"])
        self.assertGreater(result["hover_estimate"]["loaded_pack_voltage_v"], 20)

    def test_prepare_only_builds_numeric_test_and_freezes_input_snapshot(self):
        result, (folder, record) = self.prepare()
        self.assertEqual(result["status"], "PREPARED_NO_FLIGHT")
        self.assertEqual(set(record["files"]), runtime.PREPARED_FILES)
        self.assertEqual(record["vehicle"]["rotor_count"], 6)
        self.assertEqual(record["battery"]["ns"], 6)
        self.assertEqual(
            record["observation_vehicle"]["motor_order"], record["vehicle"]["motor_order"]
        )
        self.assertEqual(len(self.commands), 3)
        self.assertEqual(self.commands[0][0], "cmake")
        self.assertEqual(self.commands[1][:2], ["cmake", "--build"])
        self.assertEqual(self.commands[2], [str(folder / "build/profile_physics_test")])
        self.assertFalse(record["ai_compatible"])
        self.assertFalse(record["flight_verified"])

    def test_manifest_cannot_change_geometry_pack_or_remove_required_files(self):
        _, (folder, original) = self.prepare()
        changes = (
            lambda r: r["vehicle"]["geometry"].update(body_mass_kg=999),
            lambda r: r["battery"].update(ns=4),
            lambda r: r["observation_vehicle"].update(ai_compatible=True),
            lambda r: r.update(files={}),
            lambda r: r["files"].pop("libDronePowertrain.so"),
            lambda r: r.update(hover_estimate={}),
        )
        for change in changes:
            with self.subTest(change=change):
                record = copy.deepcopy(original)
                change(record)
                dump(folder / "manifest.json", record)
                with self.assertRaises(ValueError):
                    runtime.read_prepared(self.path)
        dump(folder / "manifest.json", original)
        self.assertEqual(runtime.read_prepared(self.path)[1], original)

    def test_changed_inputs_binary_and_runtime_require_explicit_new_preparation(self):
        _, (folder, _) = self.prepare()
        binary = folder / "libDronePowertrain.so"
        binary.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "file changed"):
            runtime.read_prepared(self.path)
        self.native.write_text("changed global runtime")
        with self.assertRaisesRegex(ValueError, "runtime changed"):
            runtime.read_prepared(self.path)
        pack = json.loads((self.inputs / "battery.json").read_text())
        pack["r0_ohm"] *= 1.01
        dump(self.inputs / "battery.json", pack)
        with self.assertRaisesRegex(ValueError, "not prepared"):
            runtime.read_prepared(self.path)

    def test_resigned_runtime_text_cannot_disagree_with_frozen_physics(self):
        _, (folder, record) = self.prepare()
        source = folder / "battery_runtime.txt"
        source.write_text(source.read_text().replace("0.06742070443911984", "0.5"))
        record["files"][source.name] = runtime.sha(source)
        dump(folder / "manifest.json", record)
        with self.assertRaisesRegex(ValueError, "native inputs disagree"):
            runtime.read_prepared(self.path)

    def test_failed_build_is_preserved_without_automatic_retry(self):
        from capstone_sim import vehicle_assets

        item = runtime.recipe(self.path)
        folder = runtime.location(item)
        with patch.object(
            runtime.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "cmake")
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                runtime._prepare(item, folder, 2, False, vehicle_assets)
        state = json.loads((folder / "prepare-state.json").read_text())
        self.assertEqual(state["status"], "FAILED_PRESERVED")
        self.assertFalse(state["retry"])
        before = (folder / "vehicle.yaml").read_bytes()
        with self.assertRaisesRegex(ValueError, "--retry"):
            runtime._prepare(item, folder, 2, False, vehicle_assets)
        self.assertEqual((folder / "vehicle.yaml").read_bytes(), before)

    def test_payload_changes_hover_load_and_infeasible_pack_is_rejected(self):
        item = runtime.recipe(self.path)
        heavier = runtime.hover_estimate(item["profile"], item["battery"], payload=0.2)
        self.assertAlmostEqual(heavier["mass_kg"] - item["hover"]["mass_kg"], 0.2)
        self.assertGreater(heavier["pack_current_a"], item["hover"]["pack_current_a"])
        with self.assertRaises(ValueError):
            runtime.hover_estimate(item["profile"], item["battery"], payload=100)


if __name__ == "__main__":
    unittest.main()
