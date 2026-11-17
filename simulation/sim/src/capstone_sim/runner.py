import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pymavlink import mavutil

from . import bounded_wind, collection_config, model_envelope, propulsion_health, wind_model
from .profile import assert_generated, load_profile
from .progress_guard import SimulationClockGuard, SimulationClockStalled, record_secondary_failure
from .recording import Recorder, dump, integrity, sha
from .scheduling import receive_before_capture, service_capture

ROOT = Path(__file__).resolve().parents[3]
PX4 = ROOT / "vendor/PX4-Autopilot"
BINARY = PX4 / "build/px4_sitl_default/bin/px4"
MODEL = PX4 / "Tools/simulation/gz/models"
COLLECTION_PHASE_WAIT_S = {"streaming": 300, "takeoff": 180, "landing": 300}


@contextmanager
def termination_guard():
    """Unwind owned-process cleanup on SIGTERM; restore the caller's handler."""
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(number, unused_frame):
        # A second TERM must not interrupt the bounded child cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise RuntimeError(f"Runner received SIGTERM ({number})")

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def native_land_requested(run, phase_policy):
    """Allow the expected mode transition after native LAND publication.

    HEARTBEAT can announce AUTO before an older LOCAL_POSITION_NED timestamp
    reaches the mission end. Publication permits waiting for activation; it
    does not replace the required landing-consumption receipt or final checks.
    """
    if phase_policy is None:
        return False
    from . import joint_phase

    request = joint_phase.receipt(run, "land-request")
    return bool(
        request is not None
        and request.get("kind") == "native_land_publication_not_activation"
        and request["actual_time_us"]
        >= request["actual_start_us"] + request["duration_us"]
    )


def gazebo_seed_transport(seed):
    """Gazebo8 Ruby FFI is int32; native nonzero int converts to uint32 intact.

    Use an attached option because Ruby's optional Integer arg rejects a
    separate negative token. Zero retains Gazebo's existing default sentinel.
    Logical path/wind seeds are never changed by this wire representation.
    """
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Simulation seed must be a uint32 integer")
    signed = seed if seed < 0x80000000 else seed - 0x100000000
    return dict(revision="gz-sim8-int32-bitpattern/1", logical_seed_u32=seed,
                gazebo_cli_seed_i32=signed, gazebo_cli_option=f"--seed={signed}",
                gazebo_native_seed_u32=seed,
                gazebo_seed_mode="explicit_uint32" if seed else "native_default_zero_sentinel")


def fault_evidence(path):
    """Keep only the event timestamps when long parallel flights finalize."""
    result: dict[str, int | None] = dict(first_time_us=None, applied_time_us=None, boundary_time_us=None)
    path = Path(path)
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if result["first_time_us"] is None:
                    result["first_time_us"] = row["time_us"]
                if result["applied_time_us"] is None and row["fault_applied"]:
                    result["applied_time_us"] = row.get("fault_applied_time_us", row["time_us"])
                if result["boundary_time_us"] is None and row.get("fault_boundary_time_us", -1) >= 0:
                    result["boundary_time_us"] = row["fault_boundary_time_us"]
    return result


def capture_health(recorder, px4_now_s, active_since_s, max_lag_s=5.0):
    """A live PX4 process is insufficient proof that the raw capture loop is alive."""
    last_us = recorder.through
    last_s = last_us / 1e6 if last_us >= 0 else active_since_s
    lag_s = max(0.0, px4_now_s - last_s)
    return dict(
        passed=lag_s <= max_lag_s,
        px4_boot_s=px4_now_s,
        capture_through_us=last_us,
        lag_sim_seconds=lag_s,
        max_lag_sim_seconds=max_lag_s,
        recorded_events=recorder.count,
    )


def stop_owned_process(process, name, *, run=None, env=None):
    """Prefer an isolated Gazebo service request; keep bounded signal fallback."""
    grace = 600 if name == "gazebo" else 60
    started = time.monotonic()
    record = dict(name=name, pid=process.pid, grace_seconds=grace, signals=[], grace_expired=False)
    try:
        if name == "gazebo" and run is not None and process.poll() is None:
            request = dict(
                service="/server_control",
                acknowledged=False,
                request_timeout_ms=10000,
                command_timeout_seconds=15,
                exit_grace_seconds=grace,
            )
            record["service_stop"] = request
            partition = "x500_" + Path(run).name
            if (
                env is None
                or not re.fullmatch(r"[0-9a-f]{32}", Path(run).name)
                or env.get("GZ_PARTITION") != partition
                or env.get("GZ_IP") != "127.0.0.1"
            ):
                request["error"] = "Run-specific Gazebo partition/IP mismatch; request not sent"
            else:
                request["partition"] = partition
                try:
                    response = subprocess.run(
                        [
                            "gz",
                            "service",
                            "-s",
                            "/server_control",
                            "--reqtype",
                            "gz.msgs.ServerControl",
                            "--reptype",
                            "gz.msgs.Boolean",
                            "--timeout",
                            "10000",
                            "--req",
                            "stop: true",
                        ],
                        env=env,
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=False,
                    )
                    request.update(
                        returncode=response.returncode,
                        stdout=response.stdout[-4096:],
                        stderr=response.stderr[-4096:],
                    )
                    request["acknowledged"] = (
                        response.returncode == 0 and response.stdout.strip() == "data: true"
                    )
                    if request["acknowledged"]:
                        try:
                            process.wait(timeout=grace)
                        except subprocess.TimeoutExpired:
                            request["exit_wait_expired"] = True
                except (OSError, subprocess.TimeoutExpired) as exc:
                    request["error"] = str(exc)
            request["signal_fallback"] = process.poll() is None
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            record["signals"].append("SIGTERM")
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                record["grace_expired"] = True
                os.killpg(process.pid, signal.SIGKILL)
                record["signals"].append("SIGKILL")
                process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["error"] = str(exc)
    record.update(exit_code=process.returncode, elapsed_seconds=time.monotonic() - started)
    return record


SCENARIOS: dict[str, dict[str, float | str]] = {
    "hover_normal": {},
    "battery_aging_existing": {},
    "battery_aging_step": {"fault": "battery_stage", "fault_at": 180, "fault_ramp_s": 1.0},
    "battery_aging_ramp": {"fault": "battery_stage", "fault_at": 180, "fault_ramp_s": 20.0},
    "sensor_bias_quality": {"sensor_voltage_bias": 0.08, "sensor_current_bias": 0.15},
    "sensor_delay_quality": {"sensor_delay": 0.08},
    "sensor_dropout_quality": {"sensor_drop_begin": 40, "sensor_drop_end": 42},
    "altitude_normal": {"mission": "altitude"},
    "horizontal_normal": {"mission": "horizontal"},
    "yaw_normal": {"mission": "yaw"},
    "waypoint_normal": {"mission": "waypoint"},
    "low_soc_normal": {"soc": 0.4},
    "wind_normal": {"wind_e": 2, "gust_sigma": 0.4},
    "cold_normal": {"temperature": 283.15},
    "payload_normal": {"payload": 0.2},
    "propulsion_loss_existing": {},
    "propulsion_loss_ramp": {
        "fault": "propulsion_profile",
        "fault_at": 60,
        "fault_ramp_s": 20,
    },
    "propulsion_loss_step": {"fault": "propulsion_profile", "fault_at": 60, "fault_ramp_s": 1.0},
}


def required_element(parent: ET.Element, path: str) -> ET.Element:
    value = parent.find(path)
    if value is None:
        raise ValueError(f"upstream model missing {path}")
    return value


def numeric_text(element: ET.Element) -> float:
    if element.text is None:
        raise ValueError("missing model numeric value")
    return float(element.text)


def prepare_random_ground(world, path):
    """Keep physical ground under the whole route, including distant landings."""
    from .random_flight import position_controls

    controls = [
        point
        for a, b in zip(path["program"], path["program"][1:])
        for point in position_controls(a, b)
    ]
    # Both NED horizontal axes fit the square after conversion to Gazebo ENU.
    extent = max(500.0, 2 * max(max(abs(p[0]), abs(p[1])) for p in controls) + 200)
    tree = ET.parse(world)
    ground = required_element(tree.getroot(), "world/model[@name='ground_plane']/link")
    collision = required_element(ground, "collision")
    geometry = required_element(collision, "geometry")
    geometry.remove(required_element(geometry, "plane"))
    # DART converts SDF planes to a fixed 2100 m box, ignoring plane/size.
    # An explicit box uses our dimensions. Its top stays at the original z=0.
    ET.SubElement(ET.SubElement(geometry, "box"), "size").text = f"{extent} {extent} 1"
    pose = collision.find("pose")
    if pose is None:
        pose = ET.SubElement(collision, "pose")
    pose.text = "0 0 -0.5 0 0 0"
    required_element(ground, "visual/geometry/plane/size").text = f"{extent} {extent}"
    tree.write(world, encoding="utf-8", xml_declaration=True)
    return dict(shape="box", size_m=[extent, extent, 1.0], top_z_m=0.0, margin_m=100.0)


def prepare_world(run, payload, sensor_override=None, configuration=None, real_time_factor=2):
    profile_path = (
        collection_config.DIRECTORY / "vehicle.yaml" if configuration else ROOT / "profiles/x500_reference.yaml"
    )
    profile = load_profile(profile_path)
    assert_generated(profile_path, collection_config.HEADER if configuration else None)
    world = ET.parse(PX4 / "Tools/simulation/gz/worlds/default.sdf")
    w = required_element(world.getroot(), "world")
    w.set("name", "x500_world")
    physics = required_element(w, "physics")
    required_element(physics, "max_step_size").text = "0.001"
    if real_time_factor not in (1, 2, 3, 4, 8):
        raise ValueError("Supported requested real-time factors:1/2/3/4/8")
    required_element(physics, "real_time_factor").text = str(real_time_factor)
    for filename, name in [
        ("physics", "Physics"),
        ("user-commands", "UserCommands"),
        ("scene-broadcaster", "SceneBroadcaster"),
        ("imu", "Imu"),
        ("magnetometer", "Magnetometer"),
        ("air-pressure", "AirPressure"),
        ("navsat", "NavSat"),
    ]:
        ET.SubElement(
            w,
            "plugin",
            filename=f"gz-sim-{filename}-system",
            name=f"gz::sim::systems::{name}",
        )
    # Derive a single X500 from the release-pinned upstream model, retaining assets and sensors.
    m = required_element(ET.parse(MODEL / "x500_base/model.sdf").getroot(), "model")
    m.set("name", "x500_0")
    # ENU north is +pi/2 yaw: start aligned with the NED yaw=0 mission.
    # Avoid an unnecessary ninety-degree yaw step during lift-off.
    required_element(m, "pose").text = f"0 0 .24 0 0 {math.pi / 2}"
    mass = required_element(m, "link[@name='base_link']/inertial/mass")
    mass.text = str(profile["geometry"]["body_mass_kg"] + payload)
    for index, axis in enumerate(["ixx", "iyy", "izz"]):
        inertia = required_element(m, f"link[@name='base_link']/inertial/inertia/{axis}")
        inertia.text = str(
            profile["geometry"]["body_inertia_diagonal_kg_m2"][index] + payload * (0.1**2) / 6
        )
    for i, rotor in enumerate(profile["rotors"]):
        rotor_link = required_element(m, f"link[@name='rotor_{i}']")
        x, y, z = rotor["position_m"]
        required_element(rotor_link, "pose").text = f"{x} {-y} {-z} 0 0 0"
        required_element(rotor_link, "inertial/mass").text = str(
            profile["geometry"]["rotor_mass_kg_each"]
        )
        inertia = required_element(rotor_link, "inertial/inertia")
        inertia.clear()
        j = profile["propulsion"]["rotor_inertia_kg_m2"]
        for axis, value in [
            ("ixx", j / 2),
            ("iyy", j / 2),
            ("izz", j),
            ("ixy", 0),
            ("ixz", 0),
            ("iyz", 0),
        ]:
            ET.SubElement(inertia, axis).text = str(value)
        joint = required_element(m, f"joint[@name='rotor_{i}_joint']")
        joint.set("type", "fixed")
        for axis_element in joint.findall("axis"):
            joint.remove(axis_element)
    for name, rate in profile["sensors"]["rates_hz"].items():
        required_element(m, f"link/sensor[@name='{name}']/update_rate").text = str(rate)
    plugin = ET.SubElement(
        m,
        "plugin",
        filename=str(collection_config.PLUGIN if configuration else ROOT / "build/libDronePowertrain.so"),
        name="drone::PowertrainSystem",
    )
    ET.SubElement(plugin, "profile_sha256").text = sha(profile_path)
    sensor = dict(profile["sensors"]["battery"])
    sensor.update(sensor_override or {})
    for key, value in sensor.items():
        if key.endswith("_range_v") or key.endswith("_range_a"):
            prefix = key.split("_")[0]
            ET.SubElement(plugin, prefix + "_min").text = str(value[0])
            ET.SubElement(plugin, prefix + "_max").text = str(value[1])
        else:
            ET.SubElement(plugin, key).text = str(value)
    dump(
        run / "reference/sensor_profile.json",
        dict(battery=sensor, rates_hz=profile["sensors"]["rates_hz"]),
    )
    metadata_path = run / "observations/metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["semantic_bindings"]["pack_voltage_current"]["sensor_profile"] = sensor
        dump(metadata_path, metadata)
    w.append(m)
    path = run / "reference/world.sdf"
    world.write(path, encoding="unicode")
    return path


def scenario_parameters(scenario, fault_at_s=None):
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    parameters = dict(SCENARIOS[scenario])
    if fault_at_s is not None:
        if "fault" not in parameters:
            raise ValueError("fault onset override requires an in-flight physical fault scenario")
        if not math.isfinite(fault_at_s) or not 0 < fault_at_s <= 1800:
            raise ValueError("fault onset must be finite and within (0,1800] PX4 boot seconds")
        parameters["fault_at"] = fault_at_s
    return parameters


def custom_scenario_parameters(scenario, fault_at_s, vehicle, pack):
    """Custom propulsion endpoints are resolved from the prepared healthy profile."""
    return scenario_parameters(scenario, fault_at_s)


@termination_guard()
def run_flight(
    scenario,
    seed=17,
    duration=90.0,
    instance=0,
    parent=None,
    timing=False,
    capture_interval_ms=5.0,
    fault_at_s=None,
    configuration=None,
    normal_variation=None,
    real_time_factor=2,
    vehicle_profile=None,
    gui_preview=False,
    operation_variation=None,
    fault_after_mission_s=None,
    battery_stage=None,
    battery_transition_s=None,
    propulsion_transition_s=None,
    wind_environment=None,
    wind_direction_deg=0.,
    wind_variation=1.,
    phase_policy=None,
    paired_case_id=None,
    mavlink_only=False,
    mavlink_battery_stage=None,
    mavlink_propulsion_stage=None,
):
    from . import battery_aging

    seed_transport = gazebo_seed_transport(seed)
    battery_aging.validate_selection(scenario, battery_stage, battery_transition_s, configuration)
    propulsion_health.resolve(scenario, transition_s=propulsion_transition_s)
    use_cmu = vehicle_profile is None
    aging_selection = None
    custom = vehicle_profile is not None
    mavlink_fixed_health = (
        mavlink_battery_stage is not None or mavlink_propulsion_stage is not None
    )
    if mavlink_fixed_health:
        scenario_for_stages = (
            "propulsion_loss_existing" if mavlink_propulsion_stage is not None
            else "battery_aging_existing"
        )
        if not mavlink_only or scenario != scenario_for_stages or battery_stage is not None:
            raise ValueError("Fixed training stages require the matching pre-existing MAVLink scenario")
    if mavlink_only and (configuration != "collection" or custom or gui_preview
                         or phase_policy is not None or timing or fault_after_mission_s is not None):
        raise ValueError("MAVLink-only capture requires collection X500 without native capture options")
    prepared: Any = None
    selected: Any = None
    if normal_variation is not None and operation_variation is not None:
        raise ValueError("Normal and operation variation are mutually exclusive")
    if configuration != "collection" and (
        operation_variation is not None or fault_after_mission_s is not None
    ):
        raise ValueError("Operation variation and mission onset require collection X500")
    cli_wind_profile = None
    if wind_environment is not None:
        cli_wind_profile = bounded_wind.profile(wind_environment, wind_direction_deg, wind_variation)
        if vehicle_profile is not None or configuration != "collection":
            raise ValueError("Bounded environments require collection X500")
        if operation_variation is not None:
            if "wind_profile" in operation_variation:
                raise ValueError("Select wind profile in operation JSON or CLI, not both")
            operation_variation = dict(operation_variation, wind_profile=cli_wind_profile,
                                       wind_e=0., wind_n=0., gust_sigma=0.)
        if normal_variation is not None:
            if "wind_profile" in normal_variation:
                raise ValueError("Select wind profile in normal JSON or CLI, not both")
            normal_variation = dict(normal_variation, wind_profile=cli_wind_profile,
                                    wind_e=0., wind_n=0., gust_sigma=0.)
    elif wind_direction_deg != 0 or wind_variation != 1:
        raise ValueError("Wind direction/variation require an explicit bounded environment")
    flight_variation = operation_variation if operation_variation is not None else normal_variation
    selected_wind_profile = flight_variation.get("wind_profile") if flight_variation else cli_wind_profile
    if selected_wind_profile is not None:
        bounded_wind.validate(selected_wind_profile)
        if configuration != "collection" or vehicle_profile is not None:
            raise ValueError("Bounded environments require collection X500")
    if custom:
        if battery_stage is not None or scenario in battery_aging.SCENARIOS:
            raise ValueError("CMU stage selection applies to the built-in X500; custom profiles supply their own pack")
        if configuration is not None or flight_variation is not None:
            raise ValueError(
                "Custom vehicle profile cannot be mixed with X500 configuration/variation"
            )
        from . import vehicle_assets, vehicle_runtime

        prepared, selected = vehicle_runtime.read_prepared(vehicle_profile)
    strict = bool(configuration or custom)
    if real_time_factor not in (1, 2, 3, 4, 8):
        raise ValueError("Supported requested real-time factors:1/2/3/4/8")
    if gui_preview and (instance != 0 or configuration != "collection" or custom):
        raise ValueError("GUI preview requires one collection X500 on instance 0")
    if configuration not in (None, "collection"):
        raise ValueError("unknown X500 configuration")
    if configuration:
        collection_config.validate_request(
            scenario, duration, fault_at_s,
            operation_variation=operation_variation,
            fault_after_mission_s=fault_after_mission_s,
            battery_transition_s=battery_transition_s,
            propulsion_transition_s=propulsion_transition_s,
            mavlink_fixed_health=mavlink_fixed_health,
        )
        collection_config.assert_frozen()
        if not collection_config.PLUGIN.is_file():
            raise FileNotFoundError("Collection plugin not built")
    elif not custom:
        collection_config.assert_frozen()
    parameters = (
        custom_scenario_parameters(scenario, fault_at_s, selected["vehicle"], selected["battery"])
        if custom
        else scenario_parameters(scenario, fault_at_s)
    )
    if scenario in battery_aging.SCENARIOS and scenario != "battery_aging_existing":
        if battery_transition_s is not None:
            parameters["fault_ramp_s"] = battery_transition_s
    if scenario in propulsion_health.DYNAMIC and propulsion_transition_s is not None:
        parameters["fault_ramp_s"] = propulsion_transition_s
    if normal_variation is not None:
        from .normal_variation import validate

        validate(normal_variation)
        if configuration != "collection" or scenario not in collection_config.NORMAL:
            raise ValueError("Variation applies only to healthy Collection normal runs")
        if normal_variation["soc"] < 0.6 and duration > 180:
            raise ValueError("Low-SOC variation requires <=180s duration")
    if flight_variation is not None:
        parameters.update({k: v for k, v in flight_variation.items() if k not in {"path", "wind_profile"}})
    if operation_variation is not None:
        parameters["mission"] = operation_variation["path"]["kind"]
    if fault_after_mission_s is not None:
        # The native fault stays disabled until airborne/path zero is acknowledged.
        parameters.update(fault_at=-1, fault_after_mission_s=fault_after_mission_s)
    from . import joint_phase

    joint_phase.validate(phase_policy, configuration, duration, flight_variation, custom, paired_case_id)
    tracking = (joint_phase.policy()["tracking_acceptance"] if phase_policy is not None else dict(
        max_position_error_m=2.0, position_tolerance_m=0.75, position_coverage_min=0.95,
        yaw_tolerance_rad=0.25, yaw_coverage_min=0.95,
    ))
    if not BINARY.exists():
        raise FileNotFoundError("PX4 not built")
    if not math.isfinite(duration) or duration <= 0 or duration > 1800:
        raise ValueError("duration must be finite and within (0,1800] seconds")
    instance_limit = collection_config.MAX_INSTANCES - 1 if configuration else 15 if custom else 9
    if mavlink_only:
        instance_limit = collection_config.MAVLINK_MAX_INSTANCES - 1
    if type(instance) is not int or not 0 <= instance <= instance_limit:
        raise ValueError(f"instance must be 0..{instance_limit} for the selected configuration")
    if not math.isfinite(capture_interval_ms) or not 1 <= capture_interval_ms <= 50:
        raise ValueError("capture interval must be finite and within [1,50] host milliseconds")
    capture_interval_ns = int(capture_interval_ms * 1e6)
    cfg: dict[str, float | str] = {
        "soc": 0.9,
        "temperature": 298.15,
        "wind_e": 0,
        "wind_n": 0,
        "gust_sigma": 0,
        "payload": 0,
        "mission": "hover",
    }
    pack = (
        selected["battery"]
        if custom
        else collection_config.battery()
    )
    vehicle_profile = (
        selected["vehicle"] if custom else collection_config.vehicle() if configuration else load_profile()
    )
    plugin_path = (
        prepared / "libDronePowertrain.so"
        if custom
        else collection_config.PLUGIN
        if configuration
        else ROOT / "build/libDronePowertrain.so"
    )
    propulsion_normal = propulsion_health.read_runtime(
        prepared / "propulsion_runtime.txt" if custom else ROOT / "profiles/propulsion_runtime.txt"
    )
    if not custom and propulsion_normal != propulsion_health.load_catalog()["states"]["normal"]:
        raise ValueError("X500 healthy propulsion differs from the sealed normal profile")
    propulsion_selection = propulsion_health.resolve(
        scenario, propulsion_normal, propulsion_transition_s,
        mavlink_fixed_stage=mavlink_propulsion_stage,
    )
    if custom:
        cfg.update(r0=pack["r0_ohm"], capacity=pack["capacity_ah"])
    cfg.update(parameters)
    if selected_wind_profile is not None:
        cfg.update(wind_e=0., wind_n=0., gust_sigma=0.)
    elif cfg["gust_sigma"] != 0:
        raise ValueError("Gaussian OU is preserved as historical reference; choose an explicit bounded wind environment")
    if use_cmu:
        aging_selection = battery_aging.resolve(
            scenario, battery_stage, battery_transition_s, pack,
            mavlink_fixed_stage=mavlink_battery_stage,
        )
        cfg.update(r0=aging_selection["initial"]["r0_ohm"],
                   capacity=aging_selection["initial"]["capacity_ah"],
                   battery_profile_locked=1)
    run_hover = (
        vehicle_runtime.hover_estimate(vehicle_profile, pack, payload=cfg["payload"])
        if custom
        else None
    )
    lease_path = ROOT / ".cache/instances" / f"{instance}.lock"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease = lease_path.open("a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        raise RuntimeError(
            f"PX4 instance {instance} is already owned by another project run"
        ) from None
    run = ROOT / "runs" / uuid.uuid4().hex
    recorder = None
    state = {}
    try:
        folders = ["evaluation", "reference", "native_logs", "px4"]
        if not mavlink_only:
            folders.append("observations")
        for d in folders:
            (run / d).mkdir(parents=True, exist_ok=True)
        state = {
            "run_id": run.name,
            "parent_attempt_id": parent,
            "origin": "px4_sitl",
            "status": "created",
            "termination_reason": "interrupted_before_completion",
            "start_utc": datetime.now(UTC).isoformat(),
            "end_utc": None,
            "clock": {"id": "px4-boot", "epoch": "simulation_boot", "unit": "us"},
            "observation_metadata": "observations/metadata.json",
            "source_lock": "reference/source_lock.json",
            "native_logs": [],
            "instance": instance,
            "gz_partition": "x500_" + run.name,
        }
        if gui_preview:
            state.update(demo_only=True, dataset_admission="excluded_gui_preview")
        if mavlink_only:
            state.pop("observation_metadata")
            state.update(telemetry_recording="external_mavlink", dataset_admission="mavlink_raw")
        dump(run / "run.json", state)
        receipt = os.getenv("CAPSTONE_MAVLINK_RUN_RECEIPT") if mavlink_only else None
        if receipt:
            progress_path = Path(receipt).absolute()
            if progress_path.name == "simulator.json" and not any(
                path.is_symlink() for path in (progress_path, *progress_path.parents)
            ):
                # Display metadata must not interrupt the flight.
                with suppress(OSError):
                    dump(progress_path, {"run": str(run)})
        if strict:
            state["configuration"] = (
                "vehicle_profile:" + vehicle_profile["id"] if custom else configuration
            )
            state["mavlink_source_checked"] = False
        lock = json.loads((ROOT / "artifacts/source_lock.json").read_text())
        lock["local_sources"] = {
            str(p.relative_to(ROOT)): sha(p)
            for base in ["sim", "profiles", "contracts"]
            for p in (ROOT / base).rglob("*")
            if p.is_file() and "__pycache__" not in str(p)
        }
        lock["runtime_binary_sha256"] = {
            str(p.relative_to(ROOT)): sha(p) for p in [BINARY, plugin_path]
        }
        if strict and collection_config.transport_library() is not None:
            library = collection_config.transport_library()
            lock["runtime_binary_sha256"][str(library.relative_to(ROOT))] = sha(library)
        dump(run / "reference/source_lock.json", lock)
        (run / "reference/resolved_vehicle.yaml").write_text(
            (
                prepared / "vehicle.yaml"
                if custom
                else collection_config.DIRECTORY / "vehicle.yaml"
                if configuration
                else ROOT / "profiles/x500_reference.yaml"
            ).read_text()
        )
        if custom:
            dump(run / "reference/vehicle-profile.json", selected)
            dump(run / "reference/hover-estimate.json", run_hover)
            for name in ("battery.json", "propulsion.json", "profile.hpp"):
                (run / "reference" / name).write_bytes((prepared / name).read_bytes())
        unhealthy = scenario.endswith(("existing", "step", "ramp"))
        dump(
            run / "evaluation/scenario.json",
            {
                "name": scenario,
                "seed": seed,
                "gazebo_seed_transport": seed_transport,
                "wind_profile": selected_wind_profile,
                "wind_model": bounded_wind.definition(seed, selected_wind_profile)
                if selected_wind_profile is not None else wind_model.metadata(seed, cfg),
                "wind_support_status": "bounded_current" if selected_wind_profile is not None
                else "fixed_wind_regression" if cfg["gust_sigma"] == 0
                else "legacy_gaussian_reference_outside_current_support",
                "parameters": cfg,
                "paired_case_id": f"{vehicle_profile['id'] if custom else 'x500'}-{cfg['mission']}-seed{seed}",
                "vehicle_instance_id": vehicle_profile["id"] + "-001"
                if custom
                else "x500-reference-001",
                "battery_instance_id": "virtual-pack-001",
                "propulsion_assembly_ids": [
                    f"propulsion-{i}" for i in range(vehicle_profile["rotor_count"])
                ],
                "health_eligibility": "uncertain"
                if scenario.endswith("quality")
                else "abnormal"
                if unhealthy
                else "confirmed_healthy",
                "normal_range_revision": vehicle_profile["revision"],
                "continuity": "independent_reset",
                "health_basis": "all baseline physical health parameters; engineering prior, not hardware certification",
                "requested_duration_s": duration,
                "requested_real_time_factor": real_time_factor,
                "position_command_schedule": {
                    "clock": "px4-boot",
                    "period_us": 100_000,
                    "policy": "phase_locked_skip_missed_slots",
                },
                "normal_variation": normal_variation,
                "operation_variation": operation_variation,
                "fault_after_mission_s": fault_after_mission_s,
                "fault_schedule": None,
                "producer_lock": str(collection_config.producer_lock_path().relative_to(ROOT)) if strict else None,
                "producer_lock_sha256": sha(collection_config.producer_lock_path()) if strict else None,
            },
        )
        (run / "evaluation/events.jsonl").touch()
        if aging_selection is not None:
            dump(run / "evaluation/battery-selection.json", aging_selection)
            (run / "evaluation/battery-catalog.json").write_bytes(battery_aging.CATALOG.read_bytes())
            battery_aging.write_runtime(run / "evaluation/battery-initial.txt", aging_selection["initial"])
            battery_aging.write_runtime(run / "evaluation/battery-target.txt", aging_selection["target"])
            scenario_path = run / "evaluation/scenario.json"
            scenario_record = json.loads(scenario_path.read_text())
            scenario_record.update(
                battery_selection="evaluation/battery-selection.json",
                battery_model_revision=aging_selection["catalog_revision"],
                normal_range_revision=vehicle_profile["revision"] + "+" + aging_selection["catalog_revision"],
                health_basis="CMU VAH01 initial snapshot" if scenario in collection_config.NORMAL
                else "CMU VAH01 snapshot/transition; motor and propeller health specified separately",
            )
            dump(scenario_path, scenario_record)
        dump(run / "evaluation/propulsion-selection.json", propulsion_selection)
        (run / "evaluation/propulsion-catalog.json").write_bytes(propulsion_health.CATALOG.read_bytes())
        propulsion_health.write_runtime(run / "evaluation/propulsion-initial.txt", propulsion_selection["initial"])
        propulsion_health.write_runtime(run / "evaluation/propulsion-target.txt", propulsion_selection["target"])
        scenario_path = run / "evaluation/scenario.json"
        scenario_record = json.loads(scenario_path.read_text())
        scenario_record.update(
            propulsion_selection="evaluation/propulsion-selection.json",
            propulsion_model_revision=propulsion_selection["catalog_revision"],
            normal_range_revision=scenario_record["normal_range_revision"] + "+" + propulsion_selection["catalog_revision"],
        )
        if mavlink_fixed_health:
            scenario_record["mavlink_training_health"] = dict(
                onset_kind="pre_existing",
                battery_stage=mavlink_battery_stage,
                propulsion_stage=mavlink_propulsion_stage,
            )
        dump(scenario_path, scenario_record)
        if not mavlink_only:
            recorder = Recorder(
                run, timing=timing, vehicle_profile=selected["observation_vehicle"] if custom else None,
                sync_interval_s=0.1 if phase_policy is not None else 0.0,
            )
        if not custom:
            for name in ["battery.json", "mass-ledger.json", "health_policy.yaml"]:
                (run / "reference" / name).write_bytes((collection_config.DIRECTORY / name).read_bytes())
        if not custom and recorder is not None:
            metadata_path = run / "observations/metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["vehicle"].update(
                profile_revision=vehicle_profile["revision"],
                nominal_capacity_ah=pack["capacity_ah"],
                nominal_series_cells=pack["ns"],
                nominal_parallel_cells=pack["np"],
                battery_profile_id=pack["profile_id"],
            )
            dump(metadata_path, metadata)
        timeline = []

        def transition(name, **fields):
            state["status"] = name
            dump(run / "run.json", state)
            timeline.append(dict(state=name, wall_time=time.monotonic(), **fields))
            dump(run / "evaluation/execution.json", timeline)

        env = os.environ.copy()
        if strict or use_cmu:
            env = collection_config.transport_environment(env)
            for key in list(env):
                if key.startswith(("X500_", "PX4_PARAM_")) or key in (
                    "PX4_UXRCE_DDS_NS",
                    "PX4_UXRCE_DDS_PORT",
                    "ROS_DOMAIN_ID",
                ):
                    env.pop(key)
        for name in [
            "X500_CAPTURE_TIMING_PATH",
            "X500_RUN_ID",
            "CAPSTONE_ESC_ORDER",
            "CAPSTONE_ROTOR_COUNT",
            "CAPSTONE_MAVLINK_RUN_RECEIPT",
        ]:
            env.pop(name, None)
        if timing:
            env["X500_CAPTURE_TIMING_PATH"] = str(run / "native_logs/capture-timing.jsonl")
            env["X500_RUN_ID"] = run.name
            dump(
                run / "native_logs/timing-config.json",
                {
                    "run_id": run.name,
                    "clock_id": "px4-boot",
                    "host_clock": "CLOCK_MONOTONIC",
                    "python_clock_implementation": time.get_clock_info("monotonic").implementation,
                    "linux_time_namespace": os.readlink("/proc/self/ns/time"),
                    "native_host_clock": "syscall(SYS_clock_gettime, CLOCK_MONOTONIC)",
                    "process_relationship": "PX4 child inherits producer host/time namespace",
                    "capture_interval_ms": capture_interval_ms,
                    "capture_scheduling": "phase_locked_skip_missed_slots",
                    "mavlink_receive_timeout_s": "min(0.02, time_until_capture_due)",
                    "gazebo_real_time_factor": real_time_factor,
                    "gazebo_max_step_size_s": 0.001,
                    "px4_sim_speed_factor_env": 1,
                    "native_capture_schedule_sim_us": 10000,
                    "stream_and_watermark_fsync": True,
                    "watermark_commit_interval_wall_s": recorder.sync_interval_s,
                    "raw_events_and_native_timestamps_unchanged": True,
                    "native_capture_and_timing_fsync": False,
                    "timing_enabled": True,
                },
            )
        env.update(
            GZ_PARTITION="x500_" + run.name,
            GZ_IP="127.0.0.1",
            GZ_HOMEDIR=str(run / "gz"),
            GZ_SIM_RESOURCE_PATH=str(MODEL),
            GZ_SIM_SYSTEM_PLUGIN_PATH=str(plugin_path.parent),
            PX4_SYS_AUTOSTART="4001",
            PX4_SIM_MODEL="gz_x500",
            PX4_GZ_MODEL_NAME="x500_0",
            PX4_GZ_STANDALONE="1",
            PX4_GZ_WORLD="x500_world",
            HEADLESS="1",
            X500_TRUTH_PATH=str(run / "evaluation/truth.jsonl"),
            X500_SEED=str(seed),
            PX4_SIM_SPEED_FACTOR="1",
            X500_BATTERY_FILE=str(
                prepared / "battery_runtime.txt"
                if custom
                else collection_config.DIRECTORY / "battery_runtime.txt"
                if configuration
                else collection_config.DIRECTORY / "battery_runtime.txt"
            ),
            X500_PROPULSION_FILE=str(
                run / "evaluation/propulsion-initial.txt"
            ),
            X500_PROPULSION_PROFILE_LOCKED="1",
            X500_PROPULSION_TARGET_FILE=str(run / "evaluation/propulsion-target.txt"),
        )
        if mavlink_only:
            env["X500_TRUTH_PATH"] = os.devnull
            env["X500_COUPLED_POWERTRAIN"] = "1"
        else:
            env["X500_CAPTURE_PATH"] = str(run / "native_logs/capture.jsonl")
        joint_info = None
        if phase_policy is not None:
            joint_info = joint_phase.prepare(run, flight_variation, duration)
            env.update(joint_phase.environment(run, joint_info))
            scenario_path = run / "evaluation/scenario.json"
            scenario_record = json.loads(scenario_path.read_text())
            scenario_record.update(joint_phase_policy=joint_info, paired_case_id=paired_case_id,
                position_command_schedule=dict(clock="native_consumption_relative_phase", period_us=100_000,
                    policy="preloaded_zero_order_hold_actual_controller_consumption; Python stream is keepalive"))
            scenario_record["wind_model"] = wind_model.expected_metadata(scenario_record)
            dump(scenario_path, scenario_record)
        if selected_wind_profile is not None:
            env.update(X500_WIND_ENVIRONMENT=selected_wind_profile["environment"],
                       X500_WIND_DIRECTION_DEG=str(selected_wind_profile["direction_deg"]),
                       X500_WIND_VARIATION=str(selected_wind_profile["variation"]))
        if custom:
            env.update(
                CAPSTONE_ESC_ORDER=",".join(vehicle_profile["motor_order"]),
                CAPSTONE_ROTOR_COUNT=str(vehicle_profile["rotor_count"]),
            )
        for k, v in cfg.items():
            if k not in {"r0", "capacity"}:
                env["X500_" + k.upper()] = str(v)
        if aging_selection is not None:
            env["X500_BATTERY_FILE"] = str(run / "evaluation/battery-initial.txt")
            if not aging_selection["fixed_for_flight"]:
                env["X500_BATTERY_TARGET_FILE"] = str(run / "evaluation/battery-target.txt")
        processes = []
        fault_request = None
        logs = []
        conn = None
        fc_socket = Path(f"/tmp/px4-sock-{instance}")
        fc_socket_identity = None
        start = time.monotonic()
        success = False
    except BaseException as exc:
        # Preparation can fail before child-process cleanup owns the lease.
        # Preserve the attempt, release only our handles, and let the CLI report it.
        try:
            if recorder is not None:
                recorder.release()
        finally:
            lease.close()
        if state:
            state.update(
                status="failed",
                termination_reason=str(exc),
                end_utc=datetime.now(UTC).isoformat(),
                primary_failure=dict(stage="startup_preparation", reason=str(exc)),
            )
            try:
                dump(run / "run.json", state)
            except OSError:
                pass
        raise
    try:
        if strict and os.path.lexists(fc_socket):
            raise RuntimeError("Existing PX4 socket before run; preserved")
        transition("preparing")
        sensor_override = {
            dest: cfg[src]
            for src, dest in {
                "sensor_voltage_bias": "voltage_bias_v",
                "sensor_current_bias": "current_bias_a",
                "sensor_delay": "delay_s",
                "sensor_drop_begin": "drop_begin_s",
                "sensor_drop_end": "drop_end_s",
            }.items()
            if src in cfg
        }
        if custom:
            world = vehicle_assets.prepare_world(
                run,
                vehicle_profile,
                profile_path=prepared / "vehicle.yaml",
                plugin_path=plugin_path,
                payload=cfg["payload"],
                sensor_override=sensor_override,
                real_time_factor=real_time_factor,
            )
        else:
            world = prepare_world(
                run, cfg["payload"], sensor_override, configuration, real_time_factor
            )
        if flight_variation is not None and flight_variation["path"]["kind"] == "random":
            state["ground_support"] = prepare_random_ground(world, flight_variation["path"])
        gazebo_args = ["gz", "sim", "-r", "-s", seed_transport["gazebo_cli_option"], str(world)]
        if gui_preview:
            from .preview import prepare_gui

            gui_config = prepare_gui(run, world, flight_variation)
            gazebo_args = [arg for arg in gazebo_args if arg != "-s"]
            gazebo_args.extend(["--gui-config", str(gui_config)])
        px4_etc = PX4 / "build/px4_sitl_default/etc"
        fc_expected: Any = None
        if custom:
            assert run_hover is not None
            hover = run_hover["hover_throttle"]
            px4_etc = vehicle_assets.prepare_fc(
                run, px4_etc, vehicle_profile, pack, hover_throttle=hover
            )
            fc_expected = vehicle_assets.fc_parameters(vehicle_profile, pack, hover_throttle=hover)
        elif configuration:
            px4_etc = (
                collection_config.prepare_fc(run, px4_etc, gui_preview=True)
                if gui_preview
                else collection_config.prepare_fc(run, px4_etc, mavlink_only=True, instance=instance)
                if mavlink_only
                else collection_config.prepare_fc(run, px4_etc)
            )
            fc_expected = collection_config.fc_parameters()
        if strict:
            dump(
                run / "reference/fc-configuration.json",
                dict(
                    expected=fc_expected,
                    observed={},
                    verified=False,
                    reason="awaiting actual PX4 PARAM_VALUE before arming",
                ),
            )
        for name, args, cwd in [
            ("gazebo", gazebo_args, run),
            (
                "px4",
                [
                    str(BINARY),
                    str(px4_etc),
                    "-d",
                    "-i",
                    str(instance),
                    "-w",
                    str(run / "px4"),
                ],
                run / "px4",
            ),
        ]:
            log = (run / f"native_logs/{name}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    args,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
            )
            if gui_preview and name == "gazebo":
                from .preview import follow_camera

                state["preview_camera"] = follow_camera(env)
        state["owned_process_groups"] = {"gazebo": processes[0].pid, "px4": processes[1].pid}
        dump(run / "run.json", state)
        _, offboard_receive_port = collection_config.offboard_ports(instance)
        conn = mavutil.mavlink_connection(f"udpin:127.0.0.1:{offboard_receive_port}", source_system=240)
        conn.target_system = 1 + instance
        conn.target_component = 1
        now = 0.0
        position = None
        armed = False
        mavlink_landed = False
        phase = "waiting"
        phase_t = 0.0
        next_send_us = 0
        next_heartbeat_us = 0
        last_heartbeat = 0.0
        last_command = 0.0
        reached_t = None
        next_capture_ns = 0
        ready_sim = ready_px4 = False
        mode = None
        acknowledgements = {}
        mission_errors = []
        yaw_errors = []
        mission_started = False
        joint_finished = False
        fc_gate: Any = (
            vehicle_assets.FCParameterGate(fc_expected, timeout_wall_s=300)
            if custom
            else collection_config.FCParameterGate(timeout_wall_s=300)
            if configuration
            else None
        )
        fc_checked = False
        next_progress = 0.0
        clock_guard = SimulationClockGuard(time.monotonic(), 300.0) if strict else None
        target_xyz = (0.0, 0.0, -3.0)
        target_yaw = 0.0
        phase_wait = COLLECTION_PHASE_WAIT_S if strict else {"streaming": 45, "takeoff": 30, "landing": 45}
        state["wait_policy"] = dict(
            recording_sync_interval_wall_s=recorder.sync_interval_s if recorder is not None else None,
            clock_stall_wall_seconds=300 if strict else None,
            phase_sim_seconds=phase_wait,
            gazebo_shutdown_method="partition_server_control_then_signals",
            gazebo_service_command_wall_seconds=15,
            gazebo_service_exit_wall_seconds=600,
            gazebo_shutdown_wall_seconds=600,
            px4_shutdown_wall_seconds=60,
        )
        while time.monotonic() - start < (duration * 8 + 1800 if strict else duration + 150):
            if strict and fc_socket_identity is None:
                fc_socket_identity = collection_config.owned_socket_identity(fc_socket, processes[1].pid)
            if any(p.poll() is not None for p in processes):
                raise RuntimeError(
                    "owned process exited: "
                    + str([p.poll() for p in processes])
                    + "; exporter exit74 means recording_failure; inspect native_logs"
                )
            if (
                not ready_sim
                and not mavlink_only
                and (run / "evaluation/truth.jsonl").exists()
                and (run / "evaluation/truth.jsonl").stat().st_size
            ):
                transition("simulator_ready")
                ready_sim = True
            if recorder is not None:
                next_capture_ns = service_capture(recorder, next_capture_ns, capture_interval_ns)
            if fault_request is not None:
                schedule = fault_request.poll()
                if schedule is not None:
                    cfg["fault_at"] = schedule["scheduled_time_us"] / 1e6
                    parameters["fault_at"] = cfg["fault_at"]
                    metadata_path = run / "evaluation/scenario.json"
                    scenario_metadata = json.loads(metadata_path.read_text())
                    scenario_metadata.update(parameters=cfg, fault_schedule=schedule)
                    dump(metadata_path, scenario_metadata)
            if phase_policy is not None and reached_t is None:
                native_origin = joint_phase.receipt(run, "origin")
                if native_origin is not None:
                    reached_t = native_origin["actual_start_us"] / 1e6
                    transition("airborne", sim_time=reached_t, observed_px4_boot_s=now,
                               boundary_source="native_actual_reference_consumption")
                    transition("mission", sim_time=reached_t, observed_px4_boot_s=now,
                               boundary_source="native_actual_reference_consumption")
                    mission_started = True
                    scenario_path = run / "evaluation/scenario.json"
                    scenario_record = json.loads(scenario_path.read_text())
                    scenario_record["joint_phase_policy"]["actual_start"] = native_origin
                    dump(scenario_path, scenario_record)
            if phase_policy is not None and reached_t is not None and fault_after_mission_s is not None:
                physics_path = run / "evaluation/joint-physics-origin.json"
                if physics_path.exists():
                    physics = json.loads(physics_path.read_text())
                    expected_onset = round((reached_t + fault_after_mission_s)*1e6)
                    if physics["origin_us"] != round(reached_t*1e6) or physics["fault_onset_us"] != expected_onset:
                        raise ValueError("Native joint physical fault origin mismatch")
                    cfg["fault_at"] = expected_onset / 1e6
                    scenario_path = run / "evaluation/scenario.json"
                    scenario_record = json.loads(scenario_path.read_text())
                    if scenario_record.get("fault_schedule") is None:
                        scenario_record.update(parameters=cfg, fault_schedule=dict(
                            clock="px4-boot", origin="actual_native_joint_consumption",
                            mission_origin_time_us=round(reached_t*1e6),
                            requested_offset_us=round(fault_after_mission_s*1e6),
                            scheduled_time_us=expected_onset, acknowledged=True,
                            acknowledgement_source="native_physics_receipt",
                            applied_time_us=None))
                        dump(scenario_path, scenario_record)
            batch = receive_before_capture(
                conn, next_capture_ns if recorder is not None else time.monotonic_ns() + 20_000_000
            )
            if recorder is not None:
                next_capture_ns = service_capture(recorder, next_capture_ns, capture_interval_ns)
            for msg in batch:
                typ = msg.get_type()
                if strict and typ in ("HEARTBEAT", "LOCAL_POSITION_NED", "PARAM_VALUE"):
                    if msg.get_srcSystem() != 1 + instance:
                        raise RuntimeError("Collection MAVLink source instance mismatch")
                    state["mavlink_source_checked"] = True
                if typ == "HEARTBEAT":
                    if msg.autopilot != mavutil.mavlink.MAV_AUTOPILOT_PX4:
                        raise RuntimeError("endpoint is not PX4")
                    armed = bool(msg.base_mode & 128)
                    mode = (msg.custom_mode >> 16) & 0xFF
                    last_heartbeat = time.monotonic()
                elif typ == "COMMAND_ACK":
                    acknowledgements[msg.command] = msg.result
                elif typ == "LOCAL_POSITION_NED":
                    now = msg.time_boot_ms / 1000.0
                    position = msg
                    if mavlink_only and not ready_sim and now > 0:
                        transition("simulator_ready", sim_time=now, source="LOCAL_POSITION_NED")
                        ready_sim = True
                elif typ == "EXTENDED_SYS_STATE" and msg.get_srcSystem() == 1 + instance:
                    mavlink_landed = msg.landed_state == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND
                elif typ == "PARAM_VALUE" and strict:
                    key = msg.param_id
                    if isinstance(key, bytes):
                        key = key.decode("ascii")
                    key = key.rstrip("\x00")
                    if key in fc_expected:
                        try:
                            fc_gate.observe(key, msg.param_value, msg.param_type)
                        except ValueError as exc:
                            dump(
                                run / "reference/fc-configuration.json",
                                fc_gate.evidence(time.monotonic(), False, str(exc)),
                            )
                            raise
                elif typ == "STATUSTEXT":
                    with (run / "native_logs/statustext.log").open("a") as f:
                        f.write(str(msg) + "\n")
            if clock_guard is not None:
                clock_guard.observe(time.monotonic(), now, phase)
            if phase != "waiting" and time.monotonic() - last_heartbeat > 5:
                raise TimeoutError("PX4 heartbeat lost")
            if ready_sim and last_heartbeat and not ready_px4:
                transition("px4_ready", custom_main_mode=mode)
                ready_px4 = True
            if position is None:
                continue
            if strict and phase != "waiting" and recorder is not None:
                health = capture_health(recorder, now, phase_t)
                if not health["passed"]:
                    state["capture_health_failure"] = health
                    dump(run / "native_logs/capture-health-failure.json", health)
                    raise RuntimeError(
                        "recording_stalled: PX4 clock advanced but raw capture did not"
                    )
            if strict and time.monotonic() >= next_progress:
                state["progress"] = dict(
                    sim_elapsed_s=max(0.0, now - reached_t) if reached_t is not None else 0.0,
                    target_s=duration,
                    last_update_utc=datetime.now(UTC).isoformat(),
                    px4_boot_s=now,
                )
                if mavlink_only:
                    state["progress"]["host_monotonic_s"] = time.monotonic()
                    if math.isfinite(position.z):
                        state["progress"]["height_m"] = -position.z
                dump(run / "run.json", state)
                next_progress = time.monotonic() + 1.0
            if strict and not fc_checked:
                if not ready_px4:
                    continue
                wall_now = time.monotonic()
                try:
                    fc_checked = fc_gate.step(
                        wall_now,
                        lambda key: conn.mav.param_request_read_send(
                            conn.target_system, 1, key.encode(), -1
                        ),
                    )
                except (ValueError, TimeoutError) as exc:
                    dump(
                        run / "reference/fc-configuration.json",
                        fc_gate.evidence(wall_now, False, str(exc)),
                    )
                    raise
                if not fc_checked:
                    continue
                dump(
                    run / "reference/fc-configuration.json",
                    fc_gate.evidence(wall_now, True, "actual PX4 PARAM_VALUE before arming"),
                )
            if phase == "waiting" and now > 12 and ready_px4:
                transition("preflight_ready", sim_time=now)
                phase = "streaming"
                phase_t = now
            if phase in ["streaming", "flying"] and round(now * 1e6) >= next_send_us:
                x = y = yaw = 0.0
                z = -3.0
                if phase == "flying":
                    t = now - (reached_t or now)
                    if phase_policy is not None:
                        t = max(0., min(duration, math.floor(t*10)/10))
                    if cfg["mission"] == "altitude":
                        z = -3.0 - 0.8 * math.sin(t * 0.15)
                    elif cfg["mission"] == "horizontal":
                        x = 2 * math.sin(t * 0.1)
                    elif cfg["mission"] == "yaw":
                        yaw = 0.5 * math.sin(t * 0.15)
                    elif cfg["mission"] == "waypoint":
                        x = 2 * math.sin(t * 0.08)
                        y = 2 * (1 - math.cos(t * 0.08))
                target_xyz = (x, y, z)
                velocity = acceleration = (0.0, 0.0, 0.0)
                position_mask = 0b100111111000
                if flight_variation is not None and phase == "flying":
                    from .normal_variation import target

                    x, y, z, yaw = target(t, flight_variation)
                    target_xyz = (x, y, z)
                    if flight_variation["path"]["kind"] == "random":
                        from .random_flight import kinematics

                        _, v, a = kinematics(t, flight_variation["path"])
                        velocity, acceleration = v[:3], a[:3]
                        # Position with analytic velocity/acceleration feed-forward.
                        # Preserve existing cadence and tracking acceptance limits.
                        position_mask = 0b100000000000
                        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                target_yaw = yaw
                conn.mav.set_position_target_local_ned_send(
                    int(now * 1000),
                    conn.target_system,
                    1,
                    1,
                    position_mask,
                    0. if phase_policy is not None else x,
                    0. if phase_policy is not None else y,
                    -3. if phase_policy is not None else z,
                    *velocity,
                    *acceleration,
                    0. if phase_policy is not None else yaw,
                    0,
                )
                # Preserve the nominal2x cadence in PX4 time, skipping missed slots.
                # A paused simulator must not receive repeated identical-time commands.
                next_send_us = (round(now * 1e6) // 100_000 + 1) * 100_000
            if phase in ["streaming", "flying", "landing"] and round(now * 1e6) >= next_heartbeat_us:
                # LAND owns the setpoints, but the controller connection stays alive.
                conn.mav.heartbeat_send(6, 8, 0, 0, 4)
                next_heartbeat_us = (round(now * 1e6) // 100_000 + 1) * 100_000
            if phase == "streaming" and now - phase_t > 2 and now - last_command > 2:
                conn.mav.command_long_send(conn.target_system, 1, 176, 0, 1, 6, 0, 0, 0, 0, 0)
                conn.mav.command_long_send(conn.target_system, 1, 400, 0, 1, 0, 0, 0, 0, 0, 0)
                last_command = now
            if (
                phase == "streaming"
                and armed
                and mode == 6
                and acknowledgements.get(400) == 0
                and acknowledgements.get(176) == 0
            ):
                transition("armed", sim_time=now)
                phase = "flying"
                phase_t = now
            if phase_policy is None and phase == "flying" and position.z < -2.5 and reached_t is None:
                reached_t = now
                transition("airborne", sim_time=now)
                if fault_after_mission_s is not None:
                    from .fault_schedule import MissionFaultRequest

                    fault_request = MissionFaultRequest(reached_t, fault_after_mission_s, env)
            if phase == "flying" and reached_t and now - reached_t > 2:
                if mode != 6 and not native_land_requested(run, phase_policy):
                    raise RuntimeError(f"mission mode left OFFBOARD: {mode}")
                if not mission_started:
                    transition(
                        "mission", sim_time=now, custom_main_mode=mode, command_ack=acknowledgements
                    )
                    mission_started = True
                if recorder is not None:
                    error = math.sqrt(
                        sum(
                            (a - b) ** 2
                            for a, b in zip((position.x, position.y, position.z), target_xyz)
                        )
                    )
                    mission_errors.append(error)
                    att = recorder.latest.get("vehicle_attitude", {}).get("q")
                    if att and all(v is not None for v in att):
                        qw, qx, qy, qz = att
                        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                        yaw_errors.append(
                            abs(math.atan2(math.sin(yaw - target_yaw), math.cos(yaw - target_yaw)))
                        )
                    if error > tracking["max_position_error_m"]:
                        raise RuntimeError(
                            f"mission tracking error exceeded {tracking['max_position_error_m']:g}m: {error}"
                        )
            if ((phase_policy is None and phase == "flying" and reached_t and now - reached_t >= duration)
                    or (phase_policy is not None and not joint_finished
                        and joint_phase.receipt(run, "landing") is not None)):
                if recorder is not None and (
                    not mission_errors
                    or sum(e <= tracking["position_tolerance_m"] for e in mission_errors)
                    / len(mission_errors) < tracking["position_coverage_min"]
                ):
                    raise RuntimeError("mission target tolerance coverage below 95 percent")
                if recorder is not None and (
                        not yaw_errors or sum(e <= tracking["yaw_tolerance_rad"] for e in yaw_errors)
                        / len(yaw_errors) < tracking["yaw_coverage_min"]):
                    raise RuntimeError("yaw target tolerance coverage below 95 percent")
                if recorder is not None:
                    dump(
                        run / "evaluation/mission-check.json",
                        dict(
                            status="passed",
                            tracking_acceptance=tracking,
                            position_tolerance_m=tracking["position_tolerance_m"],
                            yaw_tolerance_rad=tracking["yaw_tolerance_rad"],
                            position_coverage=sum(e <= tracking["position_tolerance_m"] for e in mission_errors)
                            / len(mission_errors),
                            yaw_coverage=sum(e <= tracking["yaw_tolerance_rad"] for e in yaw_errors) / len(yaw_errors),
                            max_position_error_m=max(mission_errors),
                            max_yaw_error_rad=max(yaw_errors),
                            samples=len(mission_errors),
                        ),
                    )
                if phase_policy is None:
                    conn.mav.command_long_send(
                        conn.target_system,
                        1,
                        21,
                        0,
                        0,
                        0,
                        0,
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )
                    transition("landing", sim_time=now)
                else:
                    native_landing = joint_phase.receipt(run, "landing")
                    transition("landing", sim_time=native_landing["actual_time_us"]/1e6,
                               observed_px4_boot_s=now, boundary_source="native_actual_auto_land_consumption")
                    joint_finished = True
                    scenario_path = run / "evaluation/scenario.json"
                    scenario_record = json.loads(scenario_path.read_text())
                    scenario_record["joint_phase_policy"]["actual_landing"] = native_landing
                    dump(scenario_path, scenario_record)
                phase = "landing"
                phase_t = now
            if phase == "flying" and reached_t is not None and not armed:
                raise RuntimeError("premature_disarm_or_PX4_safety_landing")
            landed = (
                recorder.latest.get("vehicle_land_detected", {}).get("landed") == 1
                if recorder is not None else mavlink_landed
            )
            if (
                phase == "landing"
                and not armed
                and landed
            ):
                transition("disarmed", sim_time=now)
                success = True
                break
            if phase == "streaming" and now - phase_t > phase_wait["streaming"]:
                raise TimeoutError("arming/offboard rejected")
            if phase == "flying" and reached_t is None and now - phase_t > phase_wait["takeoff"]:
                raise TimeoutError("takeoff target not reached")
            if phase == "landing" and now - phase_t > phase_wait["landing"]:
                state["landing_failure_evidence"] = dict(
                    elapsed_sim_seconds=now - phase_t,
                    land_command_ack=acknowledgements.get(21),
                    custom_main_mode=mode,
                    armed=armed,
                    landed=landed,
                    position_ned_m=[position.x, position.y, position.z],
                    vertical_speed_ned_mps=position.vz,
                )
                raise TimeoutError("landing timeout")
        if not success:
            raise TimeoutError("run wall-clock timeout")
        if phase_policy is None and fault_after_mission_s is not None and (
            fault_request is None or not fault_request.finished
        ):
            success = False
            raise RuntimeError("Mission completed without acknowledged native fault schedule")
        if phase_policy is not None:
            joint_phase.check(run)
        state["termination_reason"] = "mission_landed_disarmed"
    except (RuntimeError, TimeoutError, OSError, ValueError, KeyboardInterrupt) as exc:
        success = False
        state["termination_reason"] = str(exc)
        state["primary_failure"] = dict(
            stage=phase if "phase" in locals() else "startup", reason=str(exc)
        )
        if isinstance(exc, SimulationClockStalled):
            state["clock_stall_evidence"] = exc.evidence
    finally:
        # Cleanup must precede every finalization write: the output filesystem may fail.
        cleanup_errors = []
        if fault_request is not None:
            try:
                fault_request.close()
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup_errors.append(str(exc))
        if conn:
            try:
                conn.close()
            except OSError as exc:
                cleanup_errors.append(str(exc))
        shutdown = []
        for index in reversed(range(len(processes))):
            record = stop_owned_process(
                processes[index], ("gazebo", "px4")[index], run=run, env=env
            )
            shutdown.append(record)
            if record.get("error"):
                cleanup_errors.append(record["error"])
        state["process_shutdown"] = shutdown
        for log in logs:
            try:
                log.close()
            except OSError as exc:
                cleanup_errors.append(str(exc))
        if strict and len(processes) == 2:
            try:
                state["px4_socket_cleanup"] = collection_config.release_owned_socket(
                    fc_socket, fc_socket_identity, processes[1]
                )
            except (OSError, ValueError) as exc:
                cleanup_errors.append(str(exc))
        try:
            transition("finalizing")
            if recorder is not None:
                recorder.close()
        except (OSError, ValueError) as exc:
            success = False
            record_secondary_failure(state, "recording_failure", exc)
        finally:
            if recorder is not None:
                recorder.release()
        if cleanup_errors:
            success = False
            record_secondary_failure(state, "cleanup_failure", repr(cleanup_errors))
        state.update(
            status="completed" if success else "failed",
            end_utc=datetime.now(UTC).isoformat(),
            process_exit_codes=[p.returncode for p in processes],
            wall_duration_s=time.monotonic() - start,
            native_logs=[str(p.relative_to(run)) for p in (run / "px4").rglob("*.ulg")],
        )
        try:
            dump(run / "run.json", state)
            if not mavlink_only:
                events = []
                if scenario.endswith("quality"):
                    events.append(
                        dict(
                            schema="drone.evaluation.event/2.1",
                            run_id=run.name,
                            event_id="observation-quality-1",
                            kind="sensor_quality_change",
                            target="battery_sensor",
                            onset_kind="in_flight_step"
                            if "sensor_drop_begin" in cfg
                            else "pre_existing",
                            physical_onset_time_us=int(float(cfg["sensor_drop_begin"]) * 1e6)
                            if "sensor_drop_begin" in cfg
                            else None,
                            requested_time_us=int(float(cfg.get("sensor_drop_begin", 0)) * 1e6),
                            applied_time_us=int(float(cfg.get("sensor_drop_begin", 0)) * 1e6),
                            end_time_us=int(float(cfg["sensor_drop_end"]) * 1e6)
                            if "sensor_drop_end" in cfg
                            else None,
                            profile_revision=3,
                            parameters={k: v for k, v in cfg.items() if k.startswith("sensor_")},
                        )
                    )
                if unhealthy:
                    evidence = fault_evidence(run / "evaluation/truth.jsonl")
                    applied = evidence["applied_time_us"] if "fault" in cfg else evidence["first_time_us"]
                    if fault_after_mission_s is not None:
                        from .fault_schedule import complete_evidence

                        scenario_path = run / "evaluation/scenario.json"
                        scenario_metadata = json.loads(scenario_path.read_text())
                        schedule = scenario_metadata.get("fault_schedule")
                        if schedule is not None:
                            schedule["applied_time_us"] = applied
                        try:
                            scenario_metadata["fault_schedule"] = complete_evidence(schedule, applied, duration)
                        except ValueError as exc:
                            record_secondary_failure(state, "fault_schedule_failure", exc)
                            dump(run / "run.json", state)
                        dump(scenario_path, scenario_metadata)
                    if applied is not None:
                        events.append(
                            {
                                "schema": "drone.evaluation.event/2.1",
                                "run_id": run.name,
                                "event_id": "physical-change-1",
                                "kind": scenario,
                                "target": "battery_pack" if "battery" in scenario else "propulsion",
                                "onset_kind": (
                                    "in_flight_step" if scenario.endswith("step") else "in_flight_ramp"
                                )
                                if "fault" in cfg
                                else "pre_existing",
                                "physical_onset_time_us": applied if "fault" in cfg else None,
                                "requested_time_us": int(float(cfg.get("fault_at", 0)) * 1e6),
                                "applied_time_us": applied if "fault" in cfg else 0,
                                "end_time_us": None,
                                "profile_revision": 1,
                                "parameters": dict(parameters, battery_stage=aging_selection["target_stage"],
                                                   battery_selection="evaluation/battery-selection.json")
                                if aging_selection is not None else parameters,
                                **(
                                    {
                                        "ramp_duration_us": int(float(cfg["fault_ramp_s"]) * 1e6),
                                        "degradation_boundary_time_us": evidence["boundary_time_us"],
                                        "degradation_boundary_definition": "50 percent of configured parameter change",
                                    }
                                    if "fault_ramp_s" in cfg
                                    else {}
                                ),
                            }
                        )
                control_events = run / "evaluation/truth.jsonl.control-events.jsonl"
                if control_events.exists():
                    for line in control_events.read_text().splitlines():
                        e = json.loads(line)
                        events.append(
                            dict(
                                schema="drone.evaluation.event/2.1",
                                run_id=run.name,
                                event_id="request-" + str(e["request_id"]),
                                kind="physical_parameter_change",
                                target={0: "battery_pack", 1: "propulsion", 2: "battery_pack"}[
                                    e["field"]
                                ],
                                onset_kind="in_flight_step",
                                physical_onset_time_us=e["applied_time_us"],
                                requested_time_us=e["requested_time_us"],
                                applied_time_us=e["applied_time_us"],
                                end_time_us=None,
                                profile_revision=e["revision"],
                                parameters=dict(field=e["field"], value=e["value"]),
                            )
                        )
                (run / "evaluation/events.jsonl").write_text(
                    "".join(json.dumps(e) + "\n" for e in events)
                )
                if events and not scenario.endswith("quality"):
                    scenario_path = run / "evaluation/scenario.json"
                    scenario_metadata = json.loads(scenario_path.read_text())
                    scenario_metadata["health_eligibility"] = "abnormal"
                    scenario_metadata["health_basis"] = (
                        "physical parameter anomaly initialized or applied during run"
                    )
                    dump(scenario_path, scenario_metadata)
                if aging_selection is not None:
                    try:
                        check = battery_aging.verify_truth(run / "evaluation/truth.jsonl", aging_selection)
                    except (ValueError, KeyError, OSError) as exc:
                        check = {"status": "failed", "error": str(exc)}
                        record_secondary_failure(state, "battery_profile_failure", exc)
                        dump(run / "run.json", state)
                    dump(run / "evaluation/battery-profile-check.json", check)
                envelope = model_envelope.check(run / "evaluation/truth.jsonl", timeline)
                dump(run / "evaluation/envelope-check.json", envelope)
                scenario_path = run / "evaluation/scenario.json"
                scenario_metadata = json.loads(scenario_path.read_text())
                scenario_metadata.update(
                    health_eligibility_scope=model_envelope.POLICY,
                    health_eligibility_interval_us=[envelope["mission_start_us"], envelope["mission_end_us"]],
                )
                if (not envelope["normal_eligibility_passed"]
                        and scenario_metadata["health_eligibility"] == "confirmed_healthy"):
                    scenario_metadata.update(
                        health_eligibility="uncertain",
                        health_basis="missing mission interval or outside declared model validity envelope during mission",
                    )
                dump(scenario_path, scenario_metadata)
                integrity(run)
        except (OSError, ValueError) as exc:
            record_secondary_failure(state, "finalization_failure", exc)
            try:
                dump(run / "run.json", state)
            except OSError:
                # A separate project path can survive a run-specific quota/error.
                try:
                    dump(ROOT / "artifacts" / (run.name + "-failure.json"), state)
                except OSError:
                    print(json.dumps(state), file=sys.stderr, flush=True)
        finally:
            lease.close()
    return {
        "run": str(run),
        "status": state["status"],
        "reason": state["termination_reason"],
    }
