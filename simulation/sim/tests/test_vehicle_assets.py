"""Synthetic SDF/startup consistency tests; no PX4 or Gazebo process is started."""

import copy
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from capstone_sim.profile import load_battery, load_profile
from capstone_sim.vehicle_assets import (
    AIRFRAME,
    PX4,
    ROOT,
    SENSOR_NAMES,
    FCParameterGate,
    fc_parameter_types,
    fc_parameters,
    prepare_fc,
    prepare_world,
)

EXAMPLES = ROOT / "profiles/examples"


class VehicleAssetsTest(unittest.TestCase):
    def template(self, root):
        px4 = root / "px4"
        world = px4 / "Tools/simulation/gz/worlds/default.sdf"
        world.parent.mkdir(parents=True)
        world.write_text('<sdf version="1.9"><world name="default"><physics name="ode" '
                         'type="ignored"><max_step_size>.001</max_step_size>'
                         '<real_time_factor>1</real_time_factor></physics></world></sdf>')
        model = px4 / "Tools/simulation/gz/models/x500_base/model.sdf"
        model.parent.mkdir(parents=True)
        sensors = ''.join(f'<sensor name="{name}" type="imu"><update_rate>1</update_rate>'
                          '<always_on>1</always_on></sensor>' for name in SENSOR_NAMES)
        model.write_text('<sdf version="1.9"><model name="x500_base"><link name="base_link">'
                         + sensors + '</link><plugin filename="stock-motor.so" '
                         'name="MulticopterMotorModel"/></model></sdf>')
        return px4

    def etc(self, root):
        etc = root / "upstream_etc"
        airframe = etc / AIRFRAME
        airframe.parent.mkdir(parents=True)
        airframe.write_text("# preserved upstream X500 startup\n")
        (etc / "init.d-posix/px4-rc.mavlink").write_text(
            '[ "$px4_instance" -gt 9 ] && udp_offboard_port_remote=14549 '
            '# use the same ports for more than 10 instances to avoid port overlaps\n'
            'mavlink start -x -u $udp_gcs_port_local -r 4000000 -f\n'
        )
        return etc

    def test_quad_and_hex_geometry_allocation_and_sensors_are_consistent(self):
        for example in ("quad_example", "hex_example"):
            with self.subTest(example=example), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                px4 = self.template(root)
                original = {path: path.read_bytes() for path in px4.rglob("*") if path.is_file()}
                profile_path = EXAMPLES / example / "vehicle.yaml"
                profile = load_profile(profile_path)
                run = root / "run"
                path = prepare_world(run, profile, profile_path=profile_path,
                                     plugin_path=root / "custom.so", payload=.12, px4_root=px4)
                world = ET.parse(path).getroot().find("world")
                model = world.find("model")
                self.assertEqual(model.get("name"), "x500_0")
                self.assertEqual(float(world.findtext("physics/real_time_factor")), 2)
                body = model.find("link[@name='base_link']")
                self.assertAlmostEqual(float(body.findtext("inertial/mass")),
                                       profile["geometry"]["body_mass_kg"] + .12)
                for index, axis in enumerate(("ixx", "iyy", "izz")):
                    self.assertAlmostEqual(float(body.findtext("inertial/inertia/" + axis)),
                                           profile["geometry"]["body_inertia_diagonal_kg_m2"][index]
                                           + .12 * .1**2 / 6)
                self.assertEqual(len(model.findall("joint")), profile["rotor_count"])
                self.assertEqual(len(model.findall("link")), profile["rotor_count"] + 1)
                for index, rotor in enumerate(profile["rotors"]):
                    link = model.find(f"link[@name='rotor_{index}']")
                    position = list(map(float, link.findtext("pose").split()))[:3]
                    x, y, z = rotor["position_m"]
                    self.assertEqual(position, [x, -y, -z])
                    joint = model.find(f"joint[@name='rotor_{index}_joint']")
                    self.assertEqual(joint.get("type"), "fixed")
                    self.assertEqual(joint.findtext("child"), f"rotor_{index}")
                    self.assertIsNone(joint.find("axis"))
                for name in SENSOR_NAMES:
                    self.assertEqual(float(body.findtext(f"sensor[@name='{name}']/update_rate")),
                                     profile["sensors"]["rates_hz"][name])
                self.assertEqual(len(model.findall("plugin")), 1)
                self.assertEqual(model.find("plugin").get("name"), "drone::PowertrainSystem")
                self.assertNotIn("MulticopterMotorModel", path.read_text())
                self.assertEqual(original, {path: path.read_bytes() for path in original})
                self.assertEqual(json.loads((run / "reference/sensor_profile.json").read_text())[
                    "vehicle_profile"], profile["id"])

    def test_existing_world_and_upstream_startup_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            px4 = self.template(root)
            path = EXAMPLES / "hex_example/vehicle.yaml"
            profile, battery = load_profile(path), load_battery(path)
            run = root / "run"
            world = prepare_world(run, profile, profile_path=path, plugin_path=root / "physics.so",
                                  px4_root=px4)
            before = world.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_world(run, profile, profile_path=path, plugin_path=root / "physics.so",
                              px4_root=px4)
            self.assertEqual(world.read_bytes(), before)
            etc = self.etc(root)
            original = {p: p.read_bytes() for p in etc.rglob("*") if p.is_file()}
            private = prepare_fc(run, etc, profile, battery)
            text = (private / AIRFRAME).read_text()
            self.assertIn("param set CA_ROTOR_COUNT 6\n", text)
            self.assertIn("param set SIM_GZ_EC_FUNC6 106\n", text)
            self.assertIn("param set SIM_GZ_EC_FUNC7 0\n", text)
            self.assertNotIn("SIM_GZ_EC_FUNC9", text)
            self.assertIn("udp_gcs_port_remote=$((14600+px4_instance))",
                          (private / "init.d-posix/px4-rc.mavlink").read_text())
            self.assertNotIn("udp_offboard_port_remote=14549",
                             (private / "init.d-posix/px4-rc.mavlink").read_text())
            self.assertEqual(original, {p: p.read_bytes() for p in original})
            with self.assertRaises(FileExistsError):
                prepare_fc(run, etc, profile, battery)

    def test_fc_parameters_match_physics_geometry_and_pack(self):
        path = EXAMPLES / "hex_example/vehicle.yaml"
        profile, battery = load_profile(path), load_battery(path)
        parameters = fc_parameters(profile, battery, .62)
        self.assertEqual(parameters["BAT1_N_CELLS"], 6)
        self.assertEqual(parameters["BAT1_CAPACITY"], 6000.)
        self.assertAlmostEqual(parameters["BAT1_R_INTERNAL"], battery["r0_ohm"] / 6)
        self.assertEqual(parameters["MPC_THR_HOVER"], .62)
        self.assertEqual(parameters["MPC_THR_MIN"], .1)
        self.assertEqual(parameters["THR_MDL_FAC"], 0.)
        self.assertEqual(parameters["MAV_TYPE"], 13)
        ratio = profile["propulsion"]["torque_coefficient"] / profile["propulsion"]["thrust_coefficient"]
        for index, rotor in enumerate(profile["rotors"]):
            for axis, position in zip(("X", "Y", "Z"), rotor["position_m"]):
                self.assertEqual(parameters[f"CA_ROTOR{index}_P{axis}"], position)
            self.assertEqual(parameters[f"CA_ROTOR{index}_AZ"], -1.)
            self.assertEqual(parameters[f"CA_ROTOR{index}_KM"], rotor["spin"] * ratio)
        types = fc_parameter_types({"CA_ROTOR0_PY": 0, "BAT1_CAPACITY": 6000,
                                    "BAT1_N_CELLS": 6})
        self.assertEqual(types, {"CA_ROTOR0_PY": 9, "BAT1_CAPACITY": 9, "BAT1_N_CELLS": 6})

    def test_generic_output_mapping_preserves_hover_duty_and_disarmed_zero(self):
        path = EXAMPLES / "quad_example/vehicle.yaml"
        profile, battery = load_profile(path), load_battery(path)
        for count, mav_type in ((4, 2), (5, 2), (6, 13), (7, 2), (8, 14)):
            selected = copy.deepcopy(profile)
            selected["rotor_count"] = count
            selected["rotors"] = [copy.deepcopy(profile["rotors"][i % 4]) for i in range(count)]
            parameters = fc_parameters(selected, battery, .37)
            self.assertEqual(parameters["MAV_TYPE"], mav_type)
            self.assertEqual(parameters["THR_MDL_FAC"], 0.)
            for index in range(1, 9):
                minimum, maximum = (parameters[f"SIM_GZ_EC_{key}{index}"]
                                    for key in ("MIN", "MAX"))
                self.assertEqual(parameters[f"SIM_GZ_EC_DIS{index}"], 0)
                for duty in (0., .1, .37, .8, 1.):
                    # Pinned FunctionMotors: u ->2u-1, MixerOutput: [-1,1] ->MIN..MAX.
                    output = minimum + ((2 * duty - 1 + 1) / 2) * (maximum - minimum)
                    self.assertAlmostEqual(output / 1000, duty)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler is required for the native capture fixture")
    def test_actual_native_output_guard_accepts_driver_capacity_and_checks_active_channels(self):
        # Compile the production guard itself without linking a PX4 process or its
        # subscriptions. This preserves the real uint32/float wire representation.
        header = (ROOT / "sim/adapters/RawCapture.hpp").read_text()
        start = header.index('if (getenv("CAPSTONE_ROTOR_COUNT"))',
                             header.index("actuator_outputs_s out{};"))
        opening = header.index("{", start)
        depth, end = 1, opening + 1
        while depth:
            depth += (header[end] == "{") - (header[end] == "}")
            end += 1
        guard = header[start:end]
        code = '''
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#define PX4_ERR(...) ((void)0)
struct Outputs { uint32_t noutputs; float output[16]; };
void validate(const Outputs &out, int rotor_count) {
''' + guard + '''
}
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  int rotor_count = std::stoi(argv[1]);
  Outputs out{};
  out.noutputs = std::stoul(argv[2]);
  for (auto &value : out.output) value = 370;
  std::string damage = argv[3];
  if (damage == "nan") out.output[0] = NAN;
  if (damage == "negative") out.output[0] = -1;
  if (damage == "high") out.output[0] = 1001;
  if (damage == "unused_nan") out.output[rotor_count] = NAN;
  validate(out, rotor_count);
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            source, executable = Path(temporary) / "guard.cpp", Path(temporary) / "guard"
            source.write_text(code)
            compiled = subprocess.run(
                [shutil.which("c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 str(source), "-o", str(executable)], capture_output=True, text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            for count, capacity, damage, expected in (
                (4, 16, "valid", 0), (6, 16, "valid", 0), (8, 16, "valid", 0),
                (6, 4, "valid", 74), (4, 17, "valid", 74), (6, 16, "nan", 74),
                (6, 16, "negative", 74), (6, 16, "high", 74), (6, 16, "unused_nan", 0),
            ):
                with self.subTest(count=count, capacity=capacity, damage=damage):
                    result = subprocess.run(
                        [str(executable), str(count), str(capacity), damage],
                        env={**os.environ, "CAPSTONE_ROTOR_COUNT": str(count)},
                    )
                    self.assertEqual(result.returncode, expected)

    def test_parameter_gate_verifies_all_wire_values_before_success(self):
        path = EXAMPLES / "hex_example/vehicle.yaml"
        parameters = fc_parameters(load_profile(path), load_battery(path))
        gate = FCParameterGate(parameters)
        requests = []
        self.assertFalse(gate.step(10., requests.append))
        self.assertEqual(set(requests), set(parameters))
        for name, value in parameters.items():
            wire = (struct.unpack("<f", struct.pack("<i", value))[0]
                    if gate.expected_types[name] == 6 else float(value))
            gate.observe(name, wire, gate.expected_types[name])
        self.assertTrue(gate.step(10.5, requests.append))
        self.assertEqual(gate.evidence(10.5, True, "verified")["timeout_wall_s"], 300.)
        with self.assertRaises(ValueError):
            gate.observe("BAT1_N_CELLS", 6., 9)
        with self.assertRaises(ValueError):
            gate.observe("CA_ROTOR0_PX", math.nan, 9)
        gate.observed["CA_ROTOR0_PX"] += 1.
        with self.assertRaises(ValueError):
            gate.step(11., requests.append)
        missing = FCParameterGate(parameters, 1.)
        self.assertFalse(missing.step(20., requests.append))
        with self.assertRaises(TimeoutError):
            missing.step(21., requests.append)

    def test_invalid_geometry_and_startup_fail_before_output(self):
        path = EXAMPLES / "quad_example/vehicle.yaml"
        profile = copy.deepcopy(load_profile(path))
        profile["rotor_count"] = 9
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            with self.assertRaises(ValueError):
                prepare_world(run, profile, profile_path=path, plugin_path="missing")
            self.assertFalse(run.exists())
            etc = self.etc(Path(temporary))
            (etc / "init.d-posix/px4-rc.mavlink").write_text("unexpected upstream startup\n")
            profile = load_profile(path)
            with self.assertRaises(ValueError):
                prepare_fc(run, etc, profile, load_battery(path))
            self.assertFalse((run / "reference/px4_etc").exists())

    @unittest.skipUnless((PX4 / "build/px4_sitl_default/parameters.json").is_file(),
                         "Optional current pinned PX4 metadata; no build is started")
    def test_all_generated_parameters_exist_with_correct_pinned_px4_types(self):
        metadata = json.loads((PX4 / "build/px4_sitl_default/parameters.json").read_text())
        actual = {item["name"]: item["type"] for item in metadata["parameters"]}
        for example in ("quad_example", "hex_example"):
            path = EXAMPLES / example / "vehicle.yaml"
            parameters = fc_parameters(load_profile(path), load_battery(path))
            for name, wire_type in fc_parameter_types(parameters).items():
                self.assertIn(name, actual)
                self.assertEqual(wire_type, {"Int32": 6, "Float": 9}[actual[name]], name)


if __name__ == "__main__":
    unittest.main()
