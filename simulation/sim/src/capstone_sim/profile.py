"""Physical profiles for the preserved X500 and fixed-axis multirotors."""

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "profiles/x500_reference.yaml"
ORDER = ["front_right", "rear_left", "front_left", "rear_right"]
MOTOR_DEFAULTS = {
    "friction_nms": 2e-6,
    "coulomb_nm": 0.002,
    "motor_current_limit_a": 40.0,
    "esc_current_limit_a": 20.0,
    "esc_efficiency": 0.95,
    "motor_heat_capacity_j_k": 35.0,
    "motor_cooling_w_k": 0.8,
    "rotor_drag_coefficient_n_per_rad_mps": 8.06428e-5,
}


def finite(value, positive=False):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def allocation_rank(rotors):
    """Thrust, roll, pitch and yaw must be independently controllable."""
    rows = [
        [1.0 for _ in rotors],
        [r["position_m"][1] for r in rotors],
        [-r["position_m"][0] for r in rotors],
        [r["spin"] for r in rotors],
    ]
    rows = [[v / max(map(abs, row)) for v in row] if any(row) else row for row in rows]
    rank = 0
    for column in range(len(rotors)):
        pivot = next((i for i in range(rank, 4) if abs(rows[i][column]) > 1e-10), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][column]
        rows[rank] = [v / divisor for v in rows[rank]]
        for i in range(4):
            if i != rank:
                factor = rows[i][column]
                rows[i] = [v - factor * base for v, base in zip(rows[i], rows[rank])]
        rank += 1
        if rank == 4:
            break
    return rank


def profile_file(profile_path, relative):
    """Resolve one explicitly referenced input without following a symlink."""
    if not isinstance(relative, str):
        raise ValueError("profile input must name a profile-local file")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative or not parts.parts:
        raise ValueError("profile input must stay inside the profile directory")
    base = Path(profile_path).absolute().parent
    source = base / relative
    for component in (source, *source.parents):
        if component.is_symlink():
            raise ValueError("profile inputs and parent directories may not use a symlink")
    if not source.resolve().is_relative_to(base.resolve()) or not source.is_file():
        raise ValueError("profile input is missing or escaped the profile directory")
    return source.resolve()


def propulsion_parameters(profile_path, section):
    if not isinstance(section, dict):
        raise ValueError("propulsion must be a mapping")
    if "parameters_file" not in section:
        return section
    source = profile_file(profile_path, section["parameters_file"])
    loaded = json.loads(source.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("propulsion parameters must be a JSON mapping")
    for name in section.keys() & loaded.keys():
        if section[name] != loaded[name]:
            raise ValueError("inline and referenced propulsion disagree: " + name)
    return {**loaded, **section}


def load_profile(path=PROFILE):
    p = yaml.safe_load(Path(path).read_text())
    if not isinstance(p, dict) or p.get("schema") != "drone.vehicle/2.0":
        raise ValueError("requires drone.vehicle/2.0 profile")
    p["propulsion"] = propulsion_parameters(path, p.get("propulsion"))
    identifier, count = p.get("id"), p.get("rotor_count")
    if (
        not isinstance(identifier, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", identifier)
        or type(count) is not int
        or not 4 <= count <= 8
        or p.get("frame") != "FRD"
    ):
        raise ValueError(
            "requires fixed-axis multirotor FRD with 4..8 rotors (PX4 Gazebo ESC output limit)"
        )
    legacy = identifier == "x500_reference"
    if legacy and (count != 4 or p.get("motor_order") != ORDER):
        raise ValueError("X500 motor order mismatch")
    order = p.get("motor_order", [])
    if (
        not isinstance(order, list)
        or len(order) != count
        or any(
            not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
            for name in order
        )
        or len(set(order)) != count
        or len(p.get("rotors", [])) != count
    ):
        raise ValueError("unique motor order and rotor entries must match rotor_count")
    positions = []
    for r in p["rotors"]:
        v = r["position_m"]
        if (
            len(v) != 3
            or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in v
            )
            or isinstance(r["spin"], bool)
            or r["spin"] not in (-1, 1)
        ):
            raise ValueError("rotor geometry/spin")
        positions.append(tuple(v))
        if not legacy and "thrust_axis" in r:
            axis = r["thrust_axis"]
            if (
                not isinstance(axis, list)
                or len(axis) != 3
                or any(
                    not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
                    for x in axis
                )
                or any(
                    not math.isclose(x, target, abs_tol=1e-12)
                    for x, target in zip(axis, (0, 0, -1))
                )
            ):
                raise ValueError(
                    "only fixed vertical rotor thrust_axis [0,0,-1] in FRD is supported"
                )
    if len(set(positions)) != count:
        raise ValueError("duplicate rotor geometry")
    if legacy and [r["spin"] for r in p["rotors"]] != [1, 1, -1, -1]:
        raise ValueError("X500 rotor geometry mismatch")
    if not legacy and allocation_rank(p["rotors"]) != 4:
        raise ValueError("rotor geometry/spin does not provide four independent allocation axes")
    g = p["geometry"]
    inertia = g["body_inertia_diagonal_kg_m2"]
    values = [
        g["body_mass_kg"],
        g["rotor_mass_kg_each"],
        *inertia,
        p["propulsion"]["rotor_inertia_kg_m2"],
    ]
    if (
        len(inertia) != 3
        or any(not finite(x, True) for x in values)
        or 2 * max(inertia) > sum(inertia)
    ):
        raise ValueError("positive physical mass/inertia required")
    for key in ["max_air_speed_mps", "max_body_rate_radps", "max_rotor_speed_radps"]:
        if not math.isfinite(p["validity"][key]) or p["validity"][key] <= 0:
            raise ValueError("invalid model envelope")
    if not legacy:
        sensors = p.get("sensors")
        if not isinstance(sensors, dict) or not isinstance(sensors.get("battery"), dict):
            raise ValueError("generic profile requires a sensors.battery mapping")
        rates = sensors.get("rates_hz")
        if not isinstance(rates, dict) or any(
            not finite(rates.get(name), True)
            for name in (
                "imu_sensor",
                "magnetometer_sensor",
                "air_pressure_sensor",
                "navsat_sensor",
            )
        ):
            raise ValueError("generic profile requires four positive finite sensor rates_hz")
        battery_sensor = sensors["battery"]
        for name in ("period_s", "drift_tau_s"):
            if not finite(battery_sensor.get(name), True):
                raise ValueError("positive finite sensor battery." + name + " required")
        if not finite(battery_sensor.get("delay_s")) or battery_sensor["delay_s"] > 0.2:
            raise ValueError("sensor battery.delay_s must be finite within [0,0.2]")
        for name in ("voltage_bias_v", "current_bias_a", "drop_begin_s", "drop_end_s"):
            value = battery_sensor.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError("finite numeric sensor battery." + name + " required")
        for name in ("voltage_noise_std_v", "current_noise_std_a", "drift_std_v", "quantization_v"):
            if not finite(battery_sensor.get(name)):
                raise ValueError("nonnegative finite sensor battery." + name + " required")
    sensor = p["sensors"]["battery"]
    if sensor["period_s"] <= 0 or not 0 <= sensor["delay_s"] <= 0.2:
        raise ValueError("sensor period/delay")
    for key in ["voltage_noise_std_v", "current_noise_std_a", "drift_std_v", "quantization_v"]:
        if sensor[key] < 0 or not math.isfinite(sensor[key]):
            raise ValueError("sensor noise/quantization")
    if not legacy:
        for name in ("center_of_mass_m", "body_inertia_off_diagonal_kg_m2"):
            components = g.get(name, [0, 0, 0])
            if (
                not isinstance(components, list)
                or len(components) != 3
                or any(
                    not isinstance(x, (int, float))
                    or isinstance(x, bool)
                    or not math.isfinite(x)
                    or abs(x) > 1e-12
                    for x in components
                )
            ):
                raise ValueError(
                    "unsupported nonzero geometry."
                    + name
                    + "; use a COM-centered frame with diagonal inertia"
                )
        if "body_inertia_matrix_kg_m2" in g:
            matrix = g["body_inertia_matrix_kg_m2"]
            if (
                not isinstance(matrix, list)
                or len(matrix) != 3
                or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
                or any(
                    not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
                    for row in matrix
                    for x in row
                )
                or any(abs(matrix[i][j]) > 1e-12 for i in range(3) for j in range(3) if i != j)
                or any(not math.isclose(matrix[i][i], inertia[i], rel_tol=1e-9) for i in range(3))
            ):
                raise ValueError("only a matching diagonal body inertia matrix is supported")
        dimensions = g.get("body_size_m", [])
        if len(dimensions) != 3 or not all(finite(x, True) for x in dimensions):
            raise ValueError(
                "generic profile requires positive geometry.body_size_m [length,width,height]"
            )
        propulsion = p["propulsion"]
        for name in (
            "kv_rpm_per_v",
            "resistance_ohm",
            "rotor_inertia_kg_m2",
            "thrust_coefficient",
            "torque_coefficient",
            "propeller_radius_m",
        ):
            if not finite(propulsion.get(name), True):
                raise ValueError("positive finite propulsion." + name + " required")
        for name, value in MOTOR_DEFAULTS.items():
            propulsion.setdefault(name, value)
            positive = name not in {
                "friction_nms",
                "coulomb_nm",
                "rotor_drag_coefficient_n_per_rad_mps",
            }
            if not finite(propulsion[name], positive):
                raise ValueError("invalid propulsion." + name)
        if propulsion["esc_efficiency"] > 1:
            raise ValueError("propulsion.esc_efficiency must be within (0,1]")
        g.setdefault("drag_area_m2", 0.08)
        if not finite(g["drag_area_m2"], True):
            raise ValueError("positive geometry.drag_area_m2 required")
        validity = p["validity"]
        if not finite(validity.get("min_soc")) or validity["min_soc"] >= 1:
            raise ValueError("validity.min_soc must be within [0,1)")
        if not finite(validity.get("min_pack_voltage_v"), True):
            raise ValueError("positive validity.min_pack_voltage_v required")
    return p


def load_battery(profile_path):
    """Read a profile-local pack definition; no capacity or OCV curve scaling."""
    profile_path = Path(profile_path)
    profile = load_profile(profile_path)
    section = profile["battery"]
    relative = section.get("parameters_file")
    if relative is None and profile["id"] == "x500_reference":
        relative = (
            "battery.json"
            if (profile_path.parent / "battery.json").is_file()
            else section["profile"] + ".json"
        )
    source = profile_file(profile_path, relative)
    battery = json.loads(source.read_text())
    if not isinstance(battery, dict):
        raise ValueError("battery parameters must be a JSON mapping")
    if profile["id"] != "x500_reference" and (
        not isinstance(battery.get("profile_id"), str)
        or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", battery["profile_id"])
    ):
        raise ValueError("generic battery.profile_id must be a safe lowercase identifier")
    if (
        profile["id"] != "x500_reference"
        and section.get("profile", battery["profile_id"]) != battery["profile_id"]
    ):
        raise ValueError("vehicle battery profile and battery.profile_id disagree")
    for name, lower, upper in (("ns", 1, 16), ("np", 1, None)):
        value = battery.get(name)
        if type(value) is not int or value < lower or (upper is not None and value > upper):
            raise ValueError("invalid integer battery " + name)
    for name in ("capacity_ah", "c1_f"):
        if not finite(battery.get(name), True):
            raise ValueError("positive finite battery " + name + " required")
    for name in ("r0_ohm", "r1_ohm"):
        if not finite(battery.get(name)):
            raise ValueError("nonnegative finite battery " + name + " required")
    for name, value in (("interconnect_resistance_ohm", 0.0), ("auxiliary_a", 0.3)):
        battery.setdefault(name, value)
        if not finite(battery[name]):
            raise ValueError("invalid battery " + name)
    if battery["interconnect_resistance_ohm"] > battery["r0_ohm"]:
        raise ValueError("battery interconnect resistance exceeds total pack resistance")
    if profile["id"] != "x500_reference" and "assembled_mass_kg" in battery:
        if not finite(battery["assembled_mass_kg"], True):
            raise ValueError("positive finite battery assembled_mass_kg required when supplied")
        if profile["geometry"]["body_mass_kg"] < battery["assembled_mass_kg"]:
            raise ValueError(
                "body_mass_kg includes the pack and cannot be below battery assembled_mass_kg"
            )
    ocv = battery.get("ocv_cell_v", [])
    soc = battery.get("ocv_soc", [])
    if (
        len(ocv) != 31
        or not all(finite(v, True) for v in ocv)
        or any(ocv[i] > ocv[i + 1] for i in range(30))
        or len(soc) != 31
        or any(
            not finite(v) or not math.isclose(v, i / 30, abs_tol=1e-10) for i, v in enumerate(soc)
        )
    ):
        raise ValueError("battery requires monotone per-cell OCV and uniform SOC grid of 31 points")
    thermal = battery.get("thermal", {})
    for name in ("heat_capacity_j_k", "cooling_w_k"):
        if not finite(thermal.get(name), True):
            raise ValueError("positive finite battery.thermal." + name + " required")
    for name, expected in (
        ("nominal_series_cells", battery["ns"]),
        ("nominal_parallel_cells", battery["np"]),
        ("nominal_capacity_ah", battery["capacity_ah"]),
    ):
        if name in section and (
            not finite(section[name], True)
            or not math.isclose(section[name], expected, rel_tol=1e-9)
        ):
            raise ValueError("vehicle and battery disagree: " + name)
    if (
        profile["id"] != "x500_reference"
        and profile["validity"]["min_pack_voltage_v"] >= battery["ns"] * ocv[-1]
    ):
        raise ValueError("minimum valid pack voltage is not below full-charge OCV")
    sensor = profile["sensors"]["battery"]
    for name in ("voltage_range_v", "current_range_a"):
        limits = sensor.get(name, [])
        if (
            not isinstance(limits, list)
            or len(limits) != 2
            or any(
                not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
                for x in limits
            )
            or limits[0] >= limits[1]
        ):
            raise ValueError("sensor battery." + name + " requires two ordered finite bounds")
    if sensor["voltage_range_v"][1] < battery["ns"] * ocv[-1]:
        raise ValueError("sensor voltage range clips the full-charge pack OCV")
    return battery


def load_propulsion(profile_path):
    return load_profile(profile_path)["propulsion"]


def profile_inputs(profile_path):
    """Return every vehicle/propulsion/pack input that a run must freeze."""
    path = Path(profile_path).absolute()
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError("vehicle profile and parent directories may not use a symlink")
    profile = load_profile(path)
    inputs = [path.resolve()]
    prop = profile["propulsion"]
    if "parameters_file" in prop:
        inputs.append(profile_file(path, prop["parameters_file"]))
    battery = profile["battery"]
    relative = battery.get("parameters_file")
    if relative is None and profile["id"] == "x500_reference":
        relative = (
            "battery.json"
            if (path.parent / "battery.json").is_file()
            else battery["profile"] + ".json"
        )
    inputs.append(profile_file(path, relative))
    load_battery(path)
    return inputs


def header_text(p, digest, battery=None):
    positions = ",".join(
        "{{"
        + ",".join(map(str, [r["position_m"][0], -r["position_m"][1], -r["position_m"][2]]))
        + "}}"
        for r in p["rotors"]
    )
    if p["id"] != "x500_reference":
        if battery is None:
            raise ValueError("generic header generation requires a validated battery definition")
        return generic_header(p, battery, digest, positions)
    return "\n".join(
        [
            "// clang-format off",
            "// Generated by sim/tools/generate_profile.py; edit profiles/x500_reference.yaml.",
            "#pragma once",
            "#include <array>",
            "namespace x500::profile {",
            f'inline constexpr const char *sha256 = "{digest}";',
            "inline constexpr std::array<std::array<double,3>,4> positions{{" + positions + "}};",
            "inline constexpr std::array<double,4> spins{"
            + ",".join(str(r["spin"]) for r in p["rotors"])
            + "};",
            f"inline constexpr double rotor_inertia = {p['propulsion']['rotor_inertia_kg_m2']};",
            *[
                f"inline constexpr double {k} = {p['validity'][k]};"
                for k in ["max_air_speed_mps", "max_body_rate_radps", "max_rotor_speed_radps"]
            ],
            "}",
            "",
        ]
    )


def generic_header(p, battery, digest, positions):
    count = p["rotor_count"]
    prop = p["propulsion"]
    values = {
        "rotor_inertia": prop["rotor_inertia_kg_m2"],
        "motor_ke": 60 / (2 * math.pi * prop["kv_rpm_per_v"]),
        "motor_resistance": prop["resistance_ohm"],
        "kt_prop": prop["thrust_coefficient"],
        "kq_prop": prop["torque_coefficient"],
        "friction": prop["friction_nms"],
        "coulomb": prop["coulomb_nm"],
        "motor_limit": prop["motor_current_limit_a"],
        "dc_limit": prop["esc_current_limit_a"],
        "esc_efficiency": prop["esc_efficiency"],
        "motor_heat_capacity": prop["motor_heat_capacity_j_k"],
        "motor_cooling": prop["motor_cooling_w_k"],
        "battery_heat_capacity": battery["thermal"]["heat_capacity_j_k"],
        "battery_cooling": battery["thermal"]["cooling_w_k"],
        "interconnect_resistance": battery["interconnect_resistance_ohm"],
        "capacity_ah": battery["capacity_ah"],
        "r0": battery["r0_ohm"],
        "r1": battery["r1_ohm"],
        "c1": battery["c1_f"],
        "auxiliary_a": battery["auxiliary_a"],
        "rotor_drag_coefficient": prop["rotor_drag_coefficient_n_per_rad_mps"],
        "body_drag_area": p["geometry"]["drag_area_m2"],
        **{
            name: p["validity"][name]
            for name in (
                "max_air_speed_mps",
                "max_body_rate_radps",
                "max_rotor_speed_radps",
                "min_soc",
                "min_pack_voltage_v",
            )
        },
    }
    return "\n".join(
        [
            "// Generated from the selected vehicle and battery profiles.",
            "#pragma once",
            "#include <array>",
            "#define CAPSTONE_PROFILE_GENERIC 1",
            "namespace x500::profile {",
            f'inline constexpr const char *sha256 = "{digest}";',
            f"inline constexpr std::array<std::array<double,3>,{count}> positions{{{{"
            + positions
            + "}};",
            f"inline constexpr std::array<double,{count}> spins{{"
            + ",".join(str(r["spin"]) for r in p["rotors"])
            + "};",
            'inline constexpr const char *esc_schema = "drone.esc.request/1";',
            f'inline constexpr const char *esc_order = "{",".join(p["motor_order"])}";',
            f"inline constexpr unsigned series_cells = {battery['ns']};",
            "inline constexpr std::array<double,31> ocv_cell{"
            + ",".join(repr(float(v)) for v in battery["ocv_cell_v"])
            + "};",
            *[f"inline constexpr double {name} = {value!r};" for name, value in values.items()],
            "}",
            "",
        ]
    )


def generate(path=PROFILE, target=None):
    path = Path(path)
    p = load_profile(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if target is None:
        if path.resolve() != PROFILE.resolve():
            raise ValueError("an explicit output header is required for a custom profile")
        target = ROOT / "sim/physics/vehicle_generated.hpp"
    target = Path(target)
    inputs = profile_inputs(path) if p["id"] != "x500_reference" else [PROFILE, path]
    if target.resolve() in {source.resolve() for source in inputs}:
        raise ValueError("generated header must not overwrite a profile input")
    if p["id"] != "x500_reference" and target.resolve() in {
        (ROOT / "sim/physics/vehicle_generated.hpp").resolve(),
        (ROOT / "profiles/collection/vehicle_generated.hpp").resolve(),
    }:
        raise ValueError("custom header must not overwrite a preserved X500 header")
    battery = load_battery(path) if p["id"] != "x500_reference" else None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(header_text(p, digest, battery))
    return p


def assert_generated(path=PROFILE, header=None):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if header is None:
        header = ROOT / "sim/physics/vehicle_generated.hpp"
    if digest not in Path(header).read_text():
        raise ValueError("profile changed: run capstone-sim build before flight")
