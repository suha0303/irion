"""Run-local multirotor geometry and PX4 allocation; existing X500 assets stay unchanged."""

import copy
import json
import math
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

from .collection_config import FCParameterGate as ConfiguredFCParameterGate

ROOT = Path(__file__).resolve().parents[3]
PX4 = ROOT / "vendor/PX4-Autopilot"
AIRFRAME = "init.d-posix/airframes/4001_gz_x500"
SENSOR_NAMES = (
    "imu_sensor", "magnetometer_sensor", "air_pressure_sensor", "navsat_sensor",
)
WORLD_SYSTEMS = (
    ("physics", "Physics"), ("user-commands", "UserCommands"),
    ("scene-broadcaster", "SceneBroadcaster"), ("imu", "Imu"),
    ("magnetometer", "Magnetometer"), ("air-pressure", "AirPressure"),
    ("navsat", "NavSat"),
)


def required(element, path):
    result = element.find(path)
    if result is None:
        raise ValueError("Pinned PX4/Gazebo asset is missing " + path)
    return result


def _numbers(values, count, name, positive=False):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} requires {count} numbers")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or (positive and value <= 0)
        for value in values
    ):
        raise ValueError("Invalid finite physical values: " + name)
    return values


def _geometry(profile):
    count = profile["rotor_count"]
    if type(count) is not int or not 4 <= count <= 8 or len(profile["rotors"]) != count:
        raise ValueError("Pinned PX4 Gazebo ESC outputs support 4..8 rotors")
    if profile["frame"] != "FRD":
        raise ValueError("Vehicle geometry and PX4 allocation require FRD coordinates")
    geometry = profile["geometry"]
    _numbers(geometry["body_size_m"], 3, "geometry.body_size_m", positive=True)
    _numbers(geometry["body_inertia_diagonal_kg_m2"], 3, "body inertia", positive=True)
    _numbers(
        [geometry["body_mass_kg"], geometry["rotor_mass_kg_each"],
         profile["propulsion"]["rotor_inertia_kg_m2"],
         profile["propulsion"]["propeller_radius_m"]],
        4, "masses and rotor dimensions", positive=True,
    )
    for rotor in profile["rotors"]:
        _numbers(rotor["position_m"], 3, "rotor.position_m")
        if type(rotor["spin"]) is not int or rotor["spin"] not in {-1, 1}:
            raise ValueError("Rotor spin must be -1 or +1")
    return geometry


def _text(parent, name, value):
    element = ET.SubElement(parent, name)
    element.text = str(value)
    return element


def _inertial(link, mass, inertia):
    element = ET.SubElement(link, "inertial")
    _text(element, "mass", mass)
    values = ET.SubElement(element, "inertia")
    for name, value in zip(("ixx", "iyy", "izz"), inertia):
        _text(values, name, value)
    for name in ("ixy", "ixz", "iyz"):
        _text(values, name, 0)


def _shape(parent, tag, name, dimensions, *, pose=None, cylinder=False):
    element = ET.SubElement(parent, tag, name=name)
    if pose is not None:
        _text(element, "pose", pose)
    geometry = ET.SubElement(element, "geometry")
    if cylinder:
        shape = ET.SubElement(geometry, "cylinder")
        _text(shape, "radius", dimensions[0])
        _text(shape, "length", dimensions[1])
    else:
        _text(ET.SubElement(geometry, "box"), "size", " ".join(map(str, dimensions)))
    return element


def _sensor_settings(plugin, battery):
    for key, value in battery.items():
        if key.endswith(("_range_v", "_range_a")):
            prefix = key.split("_")[0]
            _text(plugin, prefix + "_min", value[0])
            _text(plugin, prefix + "_max", value[1])
        else:
            _text(plugin, key, value)


def prepare_world(
    run, profile, *, profile_path, plugin_path, payload=0., sensor_override=None,
    real_time_factor=2, px4_root=None, model_name="x500_0", world_name="x500_world",
):
    """Build primitive geometry with the pinned upstream sensors, never stock motor plugins.

    Body mass/inertia already include the selected battery and frame. Rotor masses are
    separate fixed links; their axial motion is integrated once by the custom physics.
    """
    geometry = _geometry(profile)
    if not math.isfinite(payload) or payload < 0:
        raise ValueError("Payload must be a finite nonnegative mass")
    if real_time_factor not in (2, 3, 4, 8):
        raise ValueError("Supported requested real-time factors: 2/3/4/8")
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in (model_name, world_name)):
        raise ValueError("Safe Gazebo model and world names required")
    run, px4_root = Path(run), Path(px4_root or PX4)
    path = run / "reference/world.sdf"
    if path.exists():
        raise FileExistsError("Run world already exists; existing evidence preserved: " + str(path))
    world = ET.parse(px4_root / "Tools/simulation/gz/worlds/default.sdf")
    node = required(world.getroot(), "world")
    node.set("name", world_name)
    physics = required(node, "physics")
    required(physics, "max_step_size").text = "0.001"
    required(physics, "real_time_factor").text = str(real_time_factor)
    for filename, name in WORLD_SYSTEMS:
        ET.SubElement(node, "plugin", filename=f"gz-sim-{filename}-system",
                      name=f"gz::sim::systems::{name}")
    template = required(
        ET.parse(px4_root / "Tools/simulation/gz/models/x500_base/model.sdf").getroot(), "model",
    )
    model = ET.SubElement(node, "model", name=model_name)
    spawn_height = max(.24, geometry["body_size_m"][2] / 2 + .05)
    _text(model, "pose", f"0 0 {spawn_height} 0 0 {math.pi / 2}")
    _text(model, "self_collide", "false")
    _text(model, "static", "false")
    body = ET.SubElement(model, "link", name="base_link")
    inertia = [value + payload * .1**2 / 6 for value in geometry["body_inertia_diagonal_kg_m2"]]
    _inertial(body, geometry["body_mass_kg"] + payload, inertia)
    for tag in ("collision", "visual"):
        _shape(body, tag, "body_" + tag, geometry["body_size_m"])
    for name in SENSOR_NAMES:
        sensor = copy.deepcopy(required(template, f"link/sensor[@name='{name}']"))
        required(sensor, "update_rate").text = str(profile["sensors"]["rates_hz"][name])
        body.append(sensor)
    radius = profile["propulsion"]["propeller_radius_m"]
    rotor_inertia = profile["propulsion"]["rotor_inertia_kg_m2"]
    for index, rotor in enumerate(profile["rotors"]):
        # PX4 allocation is FRD; Gazebo rigid bodies and generated physics positions are FLU.
        x, right, down = rotor["position_m"]
        y, z = -right, -down
        link = ET.SubElement(model, "link", name=f"rotor_{index}")
        _text(link, "pose", f"{x} {y} {z} 0 0 0")
        _inertial(link, geometry["rotor_mass_kg_each"],
                  [rotor_inertia / 2, rotor_inertia / 2, rotor_inertia])
        _shape(link, "visual", "rotor_visual", [radius, .002], cylinder=True)
        joint = ET.SubElement(model, "joint", name=f"rotor_{index}_joint", type="fixed")
        _text(joint, "parent", "base_link")
        _text(joint, "child", f"rotor_{index}")
        length = math.hypot(x, y)
        if length:
            _shape(body, "visual", f"arm_{index}", [length, .015, .015],
                   pose=f"{x / 2} {y / 2} {z} 0 0 {math.atan2(y, x)}")
    plugin = ET.SubElement(model, "plugin", filename=str(Path(plugin_path)),
                           name="drone::PowertrainSystem")
    import hashlib

    _text(plugin, "profile_sha256", hashlib.sha256(Path(profile_path).read_bytes()).hexdigest())
    battery = dict(profile["sensors"]["battery"])
    battery.update(sensor_override or {})
    _sensor_settings(plugin, battery)
    path.parent.mkdir(parents=True, exist_ok=True)
    world.write(path, encoding="unicode")
    from .recording import dump

    dump(run / "reference/sensor_profile.json", dict(
        battery=battery, rates_hz=profile["sensors"]["rates_hz"], vehicle_profile=profile["id"],
    ))
    metadata_path = run / "observations/metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["semantic_bindings"]["pack_voltage_current"]["sensor_profile"] = battery
        dump(metadata_path, metadata)
    return path


def fc_parameters(profile, battery, hover_throttle=.5):
    """Use the same FRD rotor geometry/spin as the compiled physical model."""
    _geometry(profile)
    if not math.isfinite(hover_throttle) or not 0 < hover_throttle < 1:
        raise ValueError("Hover throttle must be a finite fraction in (0,1)")
    prop = profile["propulsion"]
    yaw_ratio = prop["torque_coefficient"] / prop["thrust_coefficient"]
    if not math.isfinite(yaw_ratio) or yaw_ratio <= 0:
        raise ValueError("Positive finite thrust/torque coefficients required")
    count = profile["rotor_count"]
    result = dict(
        SIM_GZ_EN=1, CA_AIRFRAME=0, CA_ROTOR_COUNT=count, SIM_BAT_ENABLE=0,
        # PX4's pinned commander recognizes these rotary-wing classes. Five/seven
        # rotors use its quad compatibility class; actual allocation/count stay N.
        MAV_TYPE={4: 2, 6: 13, 8: 14}.get(count, 2),
        BAT1_N_CELLS=battery["ns"], BAT1_CAPACITY=battery["capacity_ah"] * 1000,
        BAT1_R_INTERNAL=battery["r0_ohm"] / battery["ns"],
        BAT1_V_EMPTY=battery.get("fc_cell_empty_v", battery["ocv_cell_v"][0]),
        BAT1_V_CHARGED=battery.get("fc_cell_charged_v", battery["ocv_cell_v"][-1]),
        MPC_THR_HOVER=hover_throttle, MPC_THR_MIN=.1, THR_MDL_FAC=0.,
        MPC_USE_HTE=0, NAV_DLL_ACT=2,
    )
    for index, rotor in enumerate(profile["rotors"]):
        for axis, position in zip(("X", "Y", "Z"), rotor["position_m"]):
            result[f"CA_ROTOR{index}_P{axis}"] = position
        for axis, value in (("X", 0.), ("Y", 0.), ("Z", -1.)):
            result[f"CA_ROTOR{index}_A{axis}"] = value
        result[f"CA_ROTOR{index}_CT"] = 1.
        result[f"CA_ROTOR{index}_KM"] = rotor["spin"] * yaw_ratio
    for index in range(1, 9):
        result[f"SIM_GZ_EC_FUNC{index}"] = 100 + index if index <= count else 0
        result[f"SIM_GZ_EC_DIS{index}"] = 0
        result[f"SIM_GZ_EC_MIN{index}"] = 0
        result[f"SIM_GZ_EC_MAX{index}"] = 1000
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite FC parameter")
    return result


def fc_parameter_types(parameters):
    """PARAM_VALUE uses byte-encoded INT32 and numeric REAL32, not numeric casts."""
    integers = {
        "SIM_GZ_EN", "CA_AIRFRAME", "CA_ROTOR_COUNT", "SIM_BAT_ENABLE", "BAT1_N_CELLS",
        "MPC_USE_HTE", "NAV_DLL_ACT", "MAV_TYPE",
    }
    floats = {"BAT1_CAPACITY", "BAT1_R_INTERNAL", "BAT1_V_EMPTY", "BAT1_V_CHARGED",
              "MPC_THR_HOVER", "MPC_THR_MIN", "THR_MDL_FAC"}
    types = {}
    for name in parameters:
        if name in integers or re.fullmatch(r"SIM_GZ_EC_(FUNC|DIS|MIN|MAX)[1-8]", name):
            types[name] = 6
        elif name in floats or re.fullmatch(r"CA_ROTOR[0-7]_([PA][XYZ]|CT|KM)", name):
            types[name] = 9
        else:
            raise ValueError("Unsupported private FC parameter: " + name)
    return types


class FCParameterGate(ConfiguredFCParameterGate):
    """Verify every selected allocation/output/battery value before arming."""

    def __init__(self, parameters, timeout_wall_s=300.):
        if not math.isfinite(timeout_wall_s) or timeout_wall_s <= 0:
            raise ValueError("Positive finite FC response timeout required")
        if not parameters or any(not math.isfinite(value) for value in parameters.values()):
            raise ValueError("Nonempty finite FC parameters required")
        self.timeout_wall_s = timeout_wall_s
        self.expected = dict(parameters)
        self.expected_types = fc_parameter_types(parameters)
        self.observed, self.wire = {}, {}
        self.started, self.next_request = None, 0.
        self.requests = []

    def observe(self, name, value, param_type):
        if name not in self.expected_types or param_type != self.expected_types[name]:
            raise ValueError(f"Vehicle FC parameter type mismatch: {name} type={param_type}")
        if not math.isfinite(value):
            raise ValueError("Vehicle FC nonfinite parameter: " + name)
        self.wire[name] = dict(param_type=int(param_type), raw_value=float(value))
        self.observed[name] = (
            struct.unpack("<i", struct.pack("<f", value))[0] if param_type == 6 else float(value)
        )


def prepare_fc(run, upstream_etc, profile, battery, hover_throttle=.5):
    """Create a private startup tree, leaving the vendor checkout and X500 airframe intact."""
    parameters = fc_parameters(profile, battery, hover_throttle)
    target = Path(run) / "reference/px4_etc"
    if target.exists():
        raise FileExistsError("Private FC startup already exists; evidence preserved: " + str(target))
    upstream_etc = Path(upstream_etc)
    if not (upstream_etc / AIRFRAME).is_file():
        raise ValueError("Pinned PX4 multirotor startup is missing")
    mavlink = (upstream_etc / "init.d-posix/px4-rc.mavlink").read_text()
    alias = ('[ "$px4_instance" -gt 9 ] && udp_offboard_port_remote=14549 '
             '# use the same ports for more than 10 instances to avoid port overlaps')
    gcs = "mavlink start -x -u $udp_gcs_port_local -r 4000000 -f"
    if mavlink.count(alias) != 1 or mavlink.splitlines().count(gcs) != 1:
        raise ValueError("Pinned PX4 UDP startup policy differs; no private assets created")
    mavlink = mavlink.replace(alias, "# Unique offboard14540+instance for project lanes0..15")
    mavlink = mavlink.replace(gcs, "udp_gcs_port_remote=$((14600+px4_instance))\n"
                              + gcs + " -o $udp_gcs_port_remote", 1)
    shutil.copytree(upstream_etc, target, symlinks=False)
    lines = [
        "#!/bin/sh", "# Run-local multirotor allocation generated from the selected profile.",
        ". ${R}etc/init.d/rc.mc_defaults", "PX4_SIMULATOR=${PX4_SIMULATOR:=gz}",
        "PX4_GZ_WORLD=${PX4_GZ_WORLD:=x500_world}", "PX4_SIM_MODEL=${PX4_SIM_MODEL:=x500}",
    ]
    lines.extend(f"param set {name} {value:.12g}" for name, value in parameters.items())
    (target / AIRFRAME).write_text("\n".join(lines) + "\n")
    (target / "init.d-posix/px4-rc.mavlink").write_text(mavlink)
    return target
