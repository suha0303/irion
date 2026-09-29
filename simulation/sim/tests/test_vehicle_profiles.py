"""Profile parsing and compiled multi-rotor physics; no simulator or flight."""

import copy
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml
from capstone_sim import profile


def example_profile(directory, count=6, external_propulsion=False):
    vehicle = copy.deepcopy(profile.load_profile(profile.ROOT / "profiles/collection/vehicle.yaml"))
    vehicle.update(id="synthetic_multirotor", rotor_count=count)
    vehicle["motor_order"] = [f"rotor_{i}" for i in range(count)]
    vehicle["rotors"] = [
        dict(
            position_m=[
                0.3 * math.cos(2 * math.pi * i / count),
                0.3 * math.sin(2 * math.pi * i / count),
                -0.06,
            ],
            spin=1 if i % 2 == 0 else -1,
        )
        for i in range(count)
    ]
    vehicle["geometry"]["body_size_m"] = [0.25, 0.15, 0.1]
    vehicle["propulsion"]["propeller_radius_m"] = 0.12
    vehicle["sensors"]["battery"]["voltage_range_v"] = [0, 30]
    vehicle["battery"].update(
        profile="synthetic_6s2p",
        parameters_file="battery.json",
        nominal_series_cells=6,
        nominal_parallel_cells=2,
        nominal_capacity_ah=6,
    )
    battery = dict(
        profile_id="synthetic_6s2p",
        origin="synthetic_unit_fixture",
        ns=6,
        np=2,
        capacity_ah=6,
        r0_ohm=0.04,
        r1_ohm=0.01,
        c1_f=2000,
        ocv_soc=[i / 30 for i in range(31)],
        ocv_cell_v=[3.0 + 1.2 * i / 30 for i in range(31)],
        thermal=dict(heat_capacity_j_k=650, cooling_w_k=1.5),
    )
    (directory / "battery.json").write_text(json.dumps(battery))
    if external_propulsion:
        (directory / "propulsion.json").write_text(json.dumps(vehicle["propulsion"]))
        vehicle["propulsion"] = {"parameters_file": "propulsion.json"}
    source = directory / "vehicle.yaml"
    source.write_text(yaml.safe_dump(vehicle))
    return source, vehicle, battery


class VehicleProfileTest(unittest.TestCase):
    def test_x500_reference_header_is_still_byte_identical(self):
        path = profile.PROFILE
        expected = (profile.ROOT / "sim/physics/vehicle_generated.hpp").read_text()
        actual = profile.header_text(
            profile.load_profile(path), hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertEqual(actual, expected)

    def test_three_separate_inputs_and_normalized_propulsion(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, original, _ = example_profile(directory, external_propulsion=True)
            loaded = profile.load_profile(path)
            self.assertEqual(loaded["propulsion"]["kv_rpm_per_v"], 920)
            self.assertEqual(profile.load_propulsion(path), loaded["propulsion"])
            self.assertEqual(profile.load_battery(path)["ns"], 6)
            self.assertEqual(
                profile.profile_inputs(path),
                [path, directory / "propulsion.json", directory / "battery.json"],
            )
            original["propulsion"]["kv_rpm_per_v"] = 1000
            path.write_text(yaml.safe_dump(original))
            with self.assertRaisesRegex(ValueError, "propulsion disagree"):
                profile.load_profile(path)

    def test_bad_geometry_and_unsupported_motor_counts_are_rejected(self):
        for count in (3, 9, 12, True):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                path, vehicle, _ = example_profile(Path(tmp), count=6)
                vehicle["rotor_count"] = count
                path.write_text(yaml.safe_dump(vehicle))
                with self.assertRaisesRegex(ValueError, "4..8"):
                    profile.load_profile(path)
        with tempfile.TemporaryDirectory() as tmp:
            path, vehicle, _ = example_profile(Path(tmp))
            for rotor in vehicle["rotors"]:
                rotor["spin"] = 1
            path.write_text(yaml.safe_dump(vehicle))
            with self.assertRaisesRegex(ValueError, "four independent"):
                profile.load_profile(path)

    def test_pack_mismatch_nonmonotone_ocv_and_unsafe_ids_are_rejected(self):
        for field, value in (
            ("ns", 4),
            ("ns", 17),
            ("np", 0),
            ("r0_ohm", -1),
            ("profile_id", "../../escape"),
            ("ocv_cell_v", [4.2] * 30 + [3.0]),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                path, _, battery = example_profile(directory)
                battery[field] = value
                (directory / "battery.json").write_text(json.dumps(battery))
                with self.assertRaises(ValueError):
                    profile.load_battery(path)

    def test_profile_inputs_reject_parent_symlink_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            actual = directory / "actual"
            actual.mkdir()
            path, vehicle, _ = example_profile(actual)
            alias = directory / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                profile.profile_inputs(alias / path.name)
            vehicle["battery"]["parameters_file"] = "../actual/battery.json"
            path.write_text(yaml.safe_dump(vehicle))
            with self.assertRaisesRegex(ValueError, "inside"):
                profile.load_battery(path)

    def test_unsupported_axes_com_inertia_and_boolean_mass_are_rejected(self):
        for field, value in (
            ("center_of_mass_m", [0.1, 0, 0]),
            ("body_inertia_off_diagonal_kg_m2", [0.001, 0, 0]),
            ("body_mass_kg", True),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path, vehicle, _ = example_profile(Path(tmp))
                vehicle["geometry"][field] = value
                path.write_text(yaml.safe_dump(vehicle))
                with self.assertRaises(ValueError):
                    profile.load_profile(path)
        with tempfile.TemporaryDirectory() as tmp:
            path, vehicle, _ = example_profile(Path(tmp))
            vehicle["rotors"][0]["thrust_axis"] = [0, 0.1, -1]
            path.write_text(yaml.safe_dump(vehicle))
            with self.assertRaisesRegex(ValueError, "fixed vertical"):
                profile.load_profile(path)

    def test_pack_sensor_range_must_be_finite_and_cover_full_charge(self):
        for name, limits in (
            ("voltage_range_v", [0, 20]),
            ("current_range_a", [100, 0]),
            ("current_range_a", [0, float("inf")]),
        ):
            with self.subTest(name=name, limits=limits), tempfile.TemporaryDirectory() as tmp:
                path, vehicle, _ = example_profile(Path(tmp))
                vehicle["sensors"]["battery"][name] = limits
                path.write_text(yaml.safe_dump(vehicle))
                with self.assertRaisesRegex(ValueError, "range"):
                    profile.load_battery(path)

    def test_generic_sensor_timing_bias_drop_and_rates_fail_before_runtime(self):
        cases = (
            ("period_s", float("nan")),
            ("period_s", True),
            ("drift_tau_s", 0),
            ("drift_tau_s", float("nan")),
            ("delay_s", float("nan")),
            ("delay_s", 0.21),
            ("voltage_bias_v", float("inf")),
            ("current_bias_a", float("nan")),
            ("drop_begin_s", float("nan")),
            ("drop_end_s", "disabled"),
            ("drift_std_v", False),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value), tempfile.TemporaryDirectory() as tmp:
                path, vehicle, _ = example_profile(Path(tmp))
                vehicle["sensors"]["battery"][name] = value
                path.write_text(yaml.safe_dump(vehicle))
                with self.assertRaisesRegex(ValueError, "sensor"):
                    profile.load_profile(path)
        for name in ("imu_sensor", "magnetometer_sensor", "air_pressure_sensor", "navsat_sensor"):
            for value in (None, 0, float("nan")):
                with self.subTest(rate=name, value=value), tempfile.TemporaryDirectory() as tmp:
                    path, vehicle, _ = example_profile(Path(tmp))
                    if value is None:
                        del vehicle["sensors"]["rates_hz"][name]
                    else:
                        vehicle["sensors"]["rates_hz"][name] = value
                    path.write_text(yaml.safe_dump(vehicle))
                    with self.assertRaisesRegex(ValueError, "rates_hz"):
                        profile.load_profile(path)

    def test_optional_pack_mass_is_validated_without_guessing_or_double_counting(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, vehicle, battery = example_profile(directory)
            self.assertNotIn("assembled_mass_kg", profile.load_battery(path))
            original_mass = vehicle["geometry"]["body_mass_kg"]
            battery["assembled_mass_kg"] = 0.6
            (directory / "battery.json").write_text(json.dumps(battery))
            self.assertEqual(profile.load_battery(path)["assembled_mass_kg"], 0.6)
            self.assertEqual(profile.load_profile(path)["geometry"]["body_mass_kg"], original_mass)
            for invalid in (0, -1, True, float("nan"), original_mass + 0.01):
                with self.subTest(pack_mass=invalid):
                    battery["assembled_mass_kg"] = invalid
                    (directory / "battery.json").write_text(json.dumps(battery))
                    with self.assertRaisesRegex(ValueError, "mass"):
                        profile.load_battery(path)

    def test_custom_header_requires_explicit_destination_and_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, _, _ = example_profile(directory)
            with self.assertRaisesRegex(ValueError, "explicit output"):
                profile.generate(path)
            with self.assertRaisesRegex(ValueError, "validated battery"):
                profile.header_text(profile.load_profile(path), "a" * 64)
            before = (directory / "battery.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite a profile input"):
                profile.generate(path, directory / "battery.json")
            self.assertEqual((directory / "battery.json").read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "preserved X500"):
                profile.generate(path, profile.ROOT / "sim/physics/vehicle_generated.hpp")
            target = directory / "generated.hpp"
            profile.generate(path, target)
            self.assertIn("CAPSTONE_PROFILE_GENERIC", target.read_text())
            self.assertIn('esc_schema = "drone.esc.request/1"', target.read_text())

    @unittest.skipUnless(shutil.which("g++"), "native C++ compiler required")
    def test_compiled_four_six_and_eight_motor_profiles(self):
        for count in (4, 6, 8):
            with self.subTest(rotors=count), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                path, _, _ = example_profile(directory, count=count)
                header = directory / "generated.hpp"
                profile.generate(path, header)
                binary = directory / "numeric-test"
                command = [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I" + str(profile.ROOT / "sim/physics"),
                    '-DDRONE_PROFILE_HEADER="' + str(header) + '"',
                    str(profile.ROOT / "sim/physics/test_multirotor.cpp"),
                    str(profile.ROOT / "sim/physics/powertrain.cpp"),
                    "-o",
                    str(binary),
                ]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"rotors={count} series_cells=6", result.stdout)


if __name__ == "__main__":
    unittest.main()
