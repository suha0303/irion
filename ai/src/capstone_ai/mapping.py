import math

MOTORS = ["front_right", "front_left", "rear_right", "rear_left"]
RATES = ["body_rate_p", "body_rate_q", "body_rate_r"]
FEATURES = (
    ["motor_request_" + x for x in MOTORS]
    + RATES
    + ["body_rate_error_" + x for x in "pqr"]
    + ["roll", "pitch"]
    + ["specific_force_" + x for x in "xyz"]
    + ["climb_rate", "climb_rate_error", "pack_voltage", "pack_current"]
)
CONTEXT = [
    "mean_cell_voltage_60s",
    "cell_voltage_slope_60s",
    "mean_c_rate_60s",
    "std_c_rate_60s",
    "observed_charge_fraction",
    "observed_energy_fraction",
]
TARGETS = RATES + ["pack_voltage"]
BINDINGS = {
    "motor_request": ("actuator_outputs", {"output": "output_count"}),
    "body_rate": ("vehicle_angular_velocity", {"xyz": "rad/s"}),
    "body_rate_target": (
        "vehicle_rates_setpoint",
        dict.fromkeys(["roll", "pitch", "yaw"], "rad/s"),
    ),
    "attitude": ("vehicle_attitude", {"q": "unitless"}),
    "specific_force": ("vehicle_acceleration", {"xyz": "m/s^2"}),
    "climb_rate": ("vehicle_local_position", {"vz": "m/s"}),
    "climb_rate_target": ("vehicle_local_position_setpoint", {"vz": "m/s"}),
    "pack_voltage_current": ("battery_status", {"voltage_v": "V", "current_a": "A"}),
    "operating_state": ("vehicle_status", {"arming_state": "enum", "nav_state": "enum"}),
    "landed_state": ("vehicle_land_detected", {"landed": "boolean_as_number"}),
}


def capacity_ah(config=None):
    """Explicit X500 nameplate selection; legacy configuration remains6Ah."""
    config = config or {}
    profile = config.get("battery_profile_id")
    capacity = config.get("nominal_capacity_ah", 6)
    if (profile, capacity) not in {(None, 6), ("vtc6_4s4p_r3_long15", 12)}:
        raise ValueError("unsupported explicit X500 battery nameplate")
    return capacity


def validate_metadata(m, config):
    errors = []
    if m.get("schema") != "drone.observation.metadata/2.0":
        errors.append("metadata schema")
    if m["frames"] != {"body": "FRD", "navigation": "NED", "quaternion": "wxyz_body_to_navigation"}:
        errors.append("unsupported frame; explicit source conversion required")
    if m.get("conversion_history"):
        errors.append("already converted source is not the native PX4 profile")
    v = m["vehicle"]
    if (
        v["profile_id"] != "x500_reference"
        or v["rotor_count"] != 4
        or sorted(v["motor_order"]) != sorted(MOTORS)
    ):
        errors.append("X500 motor mapping")
    expected_capacity = capacity_ah(config)
    if v["nominal_series_cells"] != 4 or v["nominal_capacity_ah"] != expected_capacity:
        errors.append("unsupported nameplate")
    if expected_capacity == 12 and (
        v.get("battery_profile_id") != "vtc6_4s4p_r3_long15"
        or v.get("nominal_parallel_cells") != 4
    ):
        errors.append("R3 VTC6 4S4P nameplate mismatch")
    for k, val in {
        "semantic": "esc_request",
        "encoding": "output_count_0_1000",
        "stage": "post_output_mapping",
    }.items():
        if m["actuation"].get(k) != val:
            errors.append("actuation." + k)
    if config["feature_profile"] not in {"common19_ctx6", "common15_ctx6"}:
        errors.append("unsupported feature profile")
    for role, (topic, units) in BINDINGS.items():
        if config["feature_profile"] == "common15_ctx6" and role in {
            "body_rate_target",
            "climb_rate_target",
        }:
            continue
        b = m["semantic_bindings"].get(role, {})
        if b.get("topic") != topic or b.get("source") != "px4.uorb":
            errors.append(role + ": missing binding")
        for field, unit in units.items():
            if b.get("units", {}).get(field) != unit or field not in b.get("fields", []):
                errors.append(role + ": " + field + " unit/field mismatch")
    if m["clock"]["unit"] != "us" or m["clock"]["mode"] not in {"capture_replay", "source_replay"}:
        errors.append("clock")
    if errors:
        raise ValueError("profile_incomplete: " + "; ".join(errors))


def mapped(event, metadata):
    f, q, topic = event["fields"], event["field_quality"], event["topic"]
    out = {}

    def add(name, field, index=None, factor=1.0):
        value = f.get(field)
        if index is not None and value is not None:
            value = value[index]
        quality = q.get(field, "unavailable")
        if value is None:
            quality = "unavailable"
        elif not math.isfinite(value):
            quality = "invalid"
        out[name] = (value * factor if quality == "valid" else None, quality)

    if topic == "actuator_outputs":
        for motor in MOTORS:
            add(
                "motor_request_" + motor,
                "output",
                metadata["vehicle"]["motor_order"].index(motor),
                0.001,
            )
        for name, (v, quality) in list(out.items()):
            if v is not None and not 0 <= v <= 1:
                out[name] = (None, "invalid")
    elif topic == "vehicle_angular_velocity":
        for i, name in enumerate(RATES):
            add(name, "xyz", i)
    elif topic == "vehicle_rates_setpoint":
        for name, field in zip("pqr", ["roll", "pitch", "yaw"]):
            add("rate_target_" + name, field)
    elif topic == "vehicle_acceleration":
        for i, name in enumerate("xyz"):
            add("specific_force_" + name, "xyz", i)
    elif topic == "vehicle_attitude":
        quat = f.get("q")
        quality = q.get("q", "unavailable")
        if quat is None or any(v is None or not math.isfinite(v) for v in quat):
            quality = "unavailable"
        if quality == "valid":
            norm = math.sqrt(sum(x * x for x in quat))
            if not 0.9 <= norm <= 1.1:
                quality = "invalid"
        if quality == "valid":
            w, x, y, z = [v / norm for v in quat]
            roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
            out.update(roll=(roll, quality), pitch=(pitch, quality))
        else:
            out.update(roll=(None, quality), pitch=(None, quality))
    elif topic in {"vehicle_local_position", "vehicle_local_position_setpoint"}:
        name = "climb_rate" if topic == "vehicle_local_position" else "climb_target"
        add(name, "vz", factor=-1)
        if topic == "vehicle_local_position" and f.get("z_valid") != 1:
            out[name] = (None, "invalid")
    elif topic == "battery_status":
        add("pack_voltage", "voltage_v")
        add("pack_current", "current_a")
        if out["pack_voltage"][0] is not None and out["pack_voltage"][0] <= 0:
            out["pack_voltage"] = (None, "invalid")
    elif topic == "vehicle_status":
        add("arming_state", "arming_state")
        add("nav_state", "nav_state")
    elif topic == "vehicle_land_detected":
        add("landed", "landed")
    return out
