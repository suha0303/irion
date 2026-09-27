import ast
import hashlib
import inspect
import json
import math
import os
import socket
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from capstone_sim import collection_config
from capstone_sim.profile import assert_generated
from capstone_sim.runner import PX4, prepare_world, run_flight, scenario_parameters


class CollectionConfigurationTest(unittest.TestCase):
    def test_fc_gate_recovers_lost_requests_without_boot_clock_deadline(self):
        gate = collection_config.FCParameterGate()
        sent = []
        # First responses lost. No flight/process retry.
        self.assertFalse(gate.step(1000., sent.append))
        self.assertEqual(sent, list(gate.expected))
        self.assertFalse(gate.step(1000.9, sent.append))
        self.assertEqual(len(sent), 4)
        self.assertFalse(gate.step(1001., sent.append))
        self.assertEqual(len(sent), 8)
        # A delayed, out-of-order partial response: only missing names are requested.
        gate.observe('BAT1_CAPACITY', 12000., 9)
        sent.clear()
        self.assertFalse(gate.step(1015., sent.append))
        self.assertEqual(sent, ['BAT1_N_CELLS', 'BAT1_R_INTERNAL', 'MPC_LAND_SPEED'])
        gate.observe('BAT1_N_CELLS', 5.605193857299268e-45, 6)
        gate.observe('BAT1_R_INTERNAL', gate.expected['BAT1_R_INTERNAL'], 9)
        gate.observe('MPC_LAND_SPEED', 0.3, 9)
        sent.clear()
        self.assertTrue(gate.step(1016., sent.append))
        self.assertEqual(sent, [])
        self.assertTrue(gate.evidence(1016., True, 'verified')['verified'])

    def test_fc_gate_missing_is_bounded_and_wrong_values_never_retried(self):
        gate = collection_config.FCParameterGate()
        sent = []
        self.assertFalse(gate.step(100., sent.append))
        self.assertFalse(gate.step(159.999, sent.append))
        with self.assertRaisesRegex(TimeoutError, 'response timeout'):
            gate.step(160., sent.append)
        self.assertEqual(len(sent), 8)
        wrong = collection_config.FCParameterGate()
        wrong.observe('BAT1_CAPACITY', 6000., 9)
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            wrong.step(100., sent.append)
        self.assertEqual(len(sent), 8)
        with self.assertRaisesRegex(ValueError, 'type mismatch'):
            wrong.observe('BAT1_N_CELLS', 4., 9)

    def test_socket_cleanup_requires_own_fd_exit_and_unchanged_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.bind(str(path))
                identity = collection_config.owned_socket_identity(path, os.getpid())
                self.assertEqual(identity["px4_pid"], os.getpid())
                live = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
                exited = SimpleNamespace(pid=os.getpid(), poll=lambda: 0)
                with self.assertRaisesRegex(ValueError, "exit unconfirmed"):
                    collection_config.release_owned_socket(path, identity, live)
                with self.assertRaisesRegex(ValueError, "still bound"):
                    collection_config.release_owned_socket(path, identity, exited)
            self.assertEqual(collection_config.release_owned_socket(path, identity, exited), "owned_stale_socket_removed")
            self.assertFalse(path.exists())
            path.write_text("replacement must survive")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                collection_config.release_owned_socket(path, identity, exited)
            self.assertEqual(path.read_text(), "replacement must survive")

    def test_touchdown_policy_is_verified_and_old_evidence_keeps_its_scope(self):
        expected = collection_config.fc_parameters()
        self.assertEqual(expected["MPC_LAND_SPEED"], 0.3)
        legacy = {key: value for key, value in expected.items() if key != "MPC_LAND_SPEED"}
        self.assertTrue(collection_config.fc_verified(legacy, expected=legacy))
        self.assertFalse(collection_config.fc_verified(legacy))
        self.assertFalse(collection_config.fc_verified(expected | {"MPC_LAND_SPEED": 0.7}))
        self.assertFalse(collection_config.fc_verified(expected, expected={}))
        wrong = expected | {"MPC_LAND_SPEED": 0.7}
        self.assertFalse(collection_config.fc_verified(wrong, expected=wrong))
        gate = collection_config.FCParameterGate()
        gate.observe("MPC_LAND_SPEED", 0.7, 9)
        with self.assertRaisesRegex(ValueError, "inconsistent.*MPC_LAND_SPEED"):
            gate.step(100., lambda key: None)
        with self.assertRaisesRegex(ValueError, "type mismatch"):
            collection_config.decode_fc_parameter("MPC_LAND_SPEED", 0.3, 6)

    def test_px4_integer_wire_bits_decode_and_real_mismatch_still_fails(self):
        from pymavlink import mavutil

        sender = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        receiver = mavutil.mavlink.MAVLink(None)
        decoded = {}
        for index, (key, value) in enumerate(collection_config.fc_parameters().items()):
            kind = 6 if key == "BAT1_N_CELLS" else 9
            wire = struct.unpack("<f", struct.pack("<i", int(value)))[0] if kind == 6 else value
            frame = mavutil.mavlink.MAVLink_param_value_message(
                key.encode(), wire, kind, len(collection_config.fc_parameters()), index).pack(sender)
            message = receiver.parse_char(frame)
            decoded[key] = collection_config.decode_fc_parameter(key, message.param_value, message.param_type)
        self.assertEqual(decoded["BAT1_N_CELLS"], 4)
        self.assertTrue(collection_config.fc_verified(decoded))
        # Exact value retained in both failed attempts: integer4 transported as float bits.
        self.assertEqual(collection_config.decode_fc_parameter("BAT1_N_CELLS", 5.605193857299268e-45, 6), 4)
        wrong = struct.unpack("<f", struct.pack("<i", 3))[0]
        self.assertFalse(collection_config.fc_verified(decoded | {"BAT1_N_CELLS": collection_config.decode_fc_parameter("BAT1_N_CELLS", wrong, 6)}))
        self.assertFalse(collection_config.fc_verified(decoded | {"BAT1_CAPACITY": 6000}))
        self.assertFalse(collection_config.fc_verified({}))
        for key, value, kind in [("BAT1_N_CELLS", 4., 9), ("BAT1_CAPACITY", 12000., 6),
                                 ("BAT1_R_INTERNAL", math.nan, 9), ("UNKNOWN", 1., 9)]:
            with self.assertRaises(ValueError):
                collection_config.decode_fc_parameter(key, value, kind)

    def test_actual_flight_loop_wall_cap_collection_and_reference_boundaries(self):
        # Evaluate the real loop guard with a synthetic clock; launch no processes.
        tree = ast.parse(inspect.getsource(run_flight))
        guards = [node.test for node in ast.walk(tree) if isinstance(node, ast.While)]
        self.assertEqual(len(guards), 1)
        guard = compile(ast.Expression(guards[0]), "actual_flight_loop_guard", "eval")

        def active(duration, configuration, elapsed):
            return eval(guard, {"time": SimpleNamespace(monotonic=lambda: 100 + elapsed),
                                "start": 100, "duration": duration,
                                "configuration": configuration,
                                "strict": bool(configuration)})

        for duration, cap in [(180, 3240), (480, 5640), (900, 9000)]:
            self.assertTrue(active(duration, "collection", duration + 151))
            self.assertTrue(active(duration, "collection", cap - .001))
            self.assertFalse(active(duration, "collection", cap))
            self.assertFalse(active(duration, "collection", cap + 1))
            self.assertTrue(active(duration, None, duration + 149.999))
            self.assertFalse(active(duration, None, duration + 150))
            self.assertFalse(active(duration, None, duration + 151))

    def test_frozen_guard_rejects_changed_input_before_launch(self):
        from capstone_sim.runtime_manifest import checked_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.txt"
            input_path.write_text("original")
            files = {"input.txt": hashlib.sha256(input_path.read_bytes()).hexdigest()}
            checked_files(root, files)
            input_path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "frozen inputs changed"):
                checked_files(root, files)

    def test_cell_topology_rc_and_capacity_basis(self):
        p = collection_config.battery()
        old = json.loads((collection_config.ROOT / "profiles/cmu_vah07_4s2p.json").read_text())
        self.assertEqual((p["ns"], p["np"], p["capacity_ah"]), (4, 4, 12))
        self.assertEqual(p["capacity_ah"], p["np"] * p["cell_capacity_selected_ah"])
        self.assertAlmostEqual(p["r0_ohm"] - .005, old["r0_ohm"] / 2)
        self.assertAlmostEqual(p["r1_ohm"], old["r1_ohm"] / 2)
        self.assertAlmostEqual(p["c1_f"], old["c1_f"] * 2)
        self.assertAlmostEqual(p["r1_ohm"] * p["c1_f"], old["r1_ohm"] * old["c1_f"])
        self.assertEqual(p["ocv_cell_v"], old["ocv_cell_v"])

    def test_world_mass_inertia_profile_and_sensor_rates(self):
        v = collection_config.vehicle()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "reference").mkdir()
            world = ET.parse(prepare_world(run, .2, configuration="collection"))
            model = world.find("world/model[@name='x500_0']")
            base = model.find("link[@name='base_link']")
            self.assertAlmostEqual(float(base.findtext("inertial/mass")), 2 - .4128 + .8056 + .2)
            for i, axis in enumerate(["ixx", "iyy", "izz"]):
                self.assertAlmostEqual(float(base.findtext("inertial/inertia/" + axis)),
                                       v["geometry"]["body_inertia_diagonal_kg_m2"][i] + .2*.1**2/6)
            total = sum(float(link.findtext("inertial/mass")) for link in model.findall("link"))
            self.assertAlmostEqual(total, 2.6571076923076927)
            self.assertTrue(all(j.get("type") == "fixed" for j in model.findall("joint")))
            plugin = model.find("plugin")
            self.assertEqual(plugin.get("filename"), str(collection_config.PLUGIN))
            digest = hashlib.sha256((collection_config.DIRECTORY / "vehicle.yaml").read_bytes()).hexdigest()
            self.assertEqual(plugin.findtext("profile_sha256"), digest)
            self.assertEqual(float(plugin.findtext("period_s")), .1)
            for name, hz in v["sensors"]["rates_hz"].items():
                self.assertEqual(float(model.findtext(f"link/sensor[@name='{name}']/update_rate")), hz)

    def test_private_fc_startup_and_float32_observed_gate(self):
        source = PX4 / "build/px4_sitl_default/etc"
        airframe = Path("init.d-posix/airframes/4001_gz_x500")
        before = (source / airframe).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            target = collection_config.prepare_fc(Path(tmp), source)
            text = (target / airframe).read_text()
            for name, value in collection_config.fc_parameters().items():
                self.assertIn(f"param set {name} {value:.12g}\n", text)
        self.assertEqual(before, (source / airframe).read_bytes())
        expected = collection_config.fc_parameters()
        self.assertAlmostEqual(expected["BAT1_R_INTERNAL"] * 4, collection_config.battery()["r0_ohm"])
        floats = {key: struct.unpack("f", struct.pack("f", value))[0] for key, value in expected.items()}
        self.assertTrue(collection_config.fc_verified(floats))
        self.assertFalse(collection_config.fc_verified({}))
        self.assertFalse(collection_config.fc_verified(floats | {"BAT1_CAPACITY": 6000}))

    def test_plan_durations_and_fault_onset_do_not_mutate_legacy(self):
        for scenario in collection_config.NORMAL | collection_config.ANOMALY:
            duration = 300 if scenario == "low_soc_normal" else 900 if scenario in collection_config.NORMAL else 480
            onset = 180 if scenario.endswith(("step", "ramp")) else None
            collection_config.validate_request(scenario, duration, onset)
            with self.assertRaises(ValueError):
                collection_config.validate_request(scenario, duration + 1, onset)
            if onset:
                self.assertEqual(scenario_parameters(scenario)["fault_at"], 60)
                self.assertEqual(scenario_parameters(scenario, onset)["fault_at"], 180)
                if scenario.endswith("ramp"):
                    self.assertEqual(scenario_parameters(scenario, onset)["fault_ramp_s"], 20)
        for duration in [math.nan, math.inf, 0, -1]:
            with self.assertRaises(ValueError):
                collection_config.validate_request("hover_normal", duration, None)
        with self.assertRaises(ValueError):
            collection_config.validate_request("propulsion_loss_step", 480, 60)

    def test_profile_digest_and_positive_inferred_dry_inertia(self):
        assert_generated(collection_config.DIRECTORY / "vehicle.yaml", collection_config.HEADER)
        ledger = json.loads((collection_config.DIRECTORY / "mass-ledger.json").read_text())
        inertia = ledger["inferred_body_without_old_pack_I_kgm2"]
        self.assertTrue(all(x > 0 for x in inertia))
        self.assertLessEqual(2*max(inertia), sum(inertia))
        self.assertFalse(ledger["hardware_mass_or_mount_measured"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vehicle.yaml"
            path.write_bytes((collection_config.DIRECTORY / "vehicle.yaml").read_bytes() + b"\n# changed\n")
            with self.assertRaises(ValueError):
                assert_generated(path, collection_config.HEADER)
