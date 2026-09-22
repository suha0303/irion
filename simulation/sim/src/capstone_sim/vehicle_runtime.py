"""Explicit profile preparation: compile and validate assets without starting a flight."""

import copy
import fcntl
import json
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import yaml

from . import collection_config
from .profile import (
    generate,
    header_text,
    load_battery,
    load_profile,
    load_propulsion,
    profile_inputs,
)
from .recording import dump, sha

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build/vehicles"
PREPARED_FILES = frozenset(
    (
        "vehicle.yaml",
        "battery.json",
        "propulsion.json",
        "profile.hpp",
        "battery_runtime.txt",
        "propulsion_runtime.txt",
        "libDronePowertrain.so",
    )
)


def json_digest(value):
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def ocv(pack, soc=0.9):
    index = soc * 30
    lower = min(29, int(index))
    return pack["ns"] * (
        pack["ocv_cell_v"][lower]
        + (index - lower) * (pack["ocv_cell_v"][lower + 1] - pack["ocv_cell_v"][lower])
    )


def hover_estimate(profile, pack, *, payload=0):
    """Steady level-flight electrical feasibility; not a closed-loop flight validation."""
    prop, geometry = profile["propulsion"], profile["geometry"]
    count = profile["rotor_count"]
    allocation = np.asarray(
        [
            [1.0] * count,
            [-rotor["position_m"][1] for rotor in profile["rotors"]],
            [rotor["position_m"][0] for rotor in profile["rotors"]],
            [
                -rotor["spin"] * prop["torque_coefficient"] / prop["thrust_coefficient"]
                for rotor in profile["rotors"]
            ],
        ]
    )
    if (
        not isinstance(payload, (int, float))
        or isinstance(payload, bool)
        or not math.isfinite(payload)
        or payload < 0
    ):
        raise ValueError("Payload must be a finite nonnegative mass")
    mass = geometry["body_mass_kg"] + count * geometry["rotor_mass_kg_each"] + payload
    requested = np.asarray([mass * 9.80665, 0.0, 0.0, 0.0])
    thrust = np.linalg.lstsq(allocation, requested, rcond=None)[0]
    if (
        not np.allclose(allocation @ thrust, requested, rtol=1e-8, atol=1e-8)
        or not np.isfinite(thrust).all()
        or np.any(thrust <= 0)
    ):
        raise ValueError("Profile has no positive level-hover allocation in the supported model")
    omega = np.sqrt(thrust / prop["thrust_coefficient"])
    ke = 60 / (2 * math.pi * prop["kv_rpm_per_v"])
    motor_current = (
        prop["torque_coefficient"] * omega**2 + prop["friction_nms"] * omega + prop["coulomb_nm"]
    ) / ke
    motor_voltage = ke * omega + prop["resistance_ohm"] * motor_current
    power = float(np.sum(motor_current * motor_voltage) / prop["esc_efficiency"])
    resistance = pack["r0_ohm"] + pack["r1_ohm"]
    unloaded = ocv(pack) - resistance * pack["auxiliary_a"]
    discriminant = unloaded**2 - 4 * resistance * power
    if discriminant <= 0:
        raise ValueError("Battery cannot supply the estimated hover power")
    voltage = (unloaded + math.sqrt(discriminant)) / 2
    duty = motor_voltage / voltage
    esc_current = motor_current * duty / prop["esc_efficiency"]
    if (
        np.max(omega) > profile["validity"]["max_rotor_speed_radps"]
        or np.max(motor_current) > prop["motor_current_limit_a"]
        or np.max(esc_current) > prop["esc_current_limit_a"]
        or voltage <= profile["validity"]["min_pack_voltage_v"]
        or float(np.min(duty)) < 0.1
        or float(np.max(duty)) > 0.8
    ):
        raise ValueError("Profile lacks hover voltage/current/speed or 20% duty headroom")
    return dict(
        basis="steady uniform propulsion, SOC0.9, pack R0+R1, no wind; engineering estimate",
        mass_kg=mass,
        rotor_thrust_n=thrust.tolist(),
        rotor_speed_rad_s=omega.tolist(),
        rotor_duty=duty.tolist(),
        loaded_pack_voltage_v=voltage,
        pack_current_a=power / voltage + pack["auxiliary_a"],
        hover_throttle=float(np.mean(duty)),
        flight_verified=False,
    )


def recipe(path):
    path = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Simulation profile inputs cannot contain symlinks")
    profile = load_profile(path)
    if profile["id"] == "x500_reference":
        raise ValueError(
            "Use the existing X500 configuration; custom profiles require a distinct ID"
        )
    pack = load_battery(path)
    inputs = {str(p): sha(p) for p in profile_inputs(path)}
    source = {}
    for folder in (ROOT / "sim/physics", ROOT / "sim/adapters", ROOT / "sim/src/capstone_sim"):
        source.update(
            {
                str(p.relative_to(ROOT)): sha(p)
                for p in folder.iterdir()
                if p.is_file() and p.suffix in {".py", ".hpp", ".cpp"}
            }
        )
    source["sim/CMakeLists.txt"] = sha(ROOT / "sim/CMakeLists.txt")
    return dict(
        source_profile=str(path),
        profile=profile,
        battery=pack,
        inputs=inputs,
        sources=source,
        hover=hover_estimate(profile, pack),
    )


def inspection(path):
    item = recipe(path)
    return dict(
        status="PROFILE_VALIDATED_NO_FLIGHT",
        profile_id=item["profile"]["id"],
        rotor_count=item["profile"]["rotor_count"],
        recipe_sha256=json_digest(item),
        inputs=item["inputs"],
        hover_estimate=item["hover"],
        ai_compatible=False,
        actual_flight_started=False,
        training_started=False,
    )


def location(item):
    return BUILD / item["profile"]["id"] / json_digest(item)[:24]


def physical_content(profile):
    """Ignore input filenames, retaining every resolved physical/configuration value."""
    result = copy.deepcopy(profile)
    for section in ("propulsion", "battery"):
        for key in ("parameters_file", "resolved_parameters_sha256"):
            result[section].pop(key, None)
    return result


def observation_profile(profile, pack):
    return dict(
        id=profile["id"],
        revision=profile["revision"],
        rotor_count=profile["rotor_count"],
        motor_order=profile["motor_order"],
        battery={key: pack[key] for key in ("ns", "np", "capacity_ah", "profile_id")},
        ai_compatible=False,
        simulation_profile_manifest="reference/vehicle-profile.json",
    )


def read_prepared(path):
    item = recipe(path)
    folder = location(item)
    target = folder / "manifest.json"
    if not target.is_file():
        raise ValueError(
            "Profile not prepared for these inputs/sources; run capstone-sim profile prepare <path>"
        )
    record = json.loads(target.read_text())
    if (
        record.get("schema") != "drone.simulation.profile/1"
        or record.get("recipe_sha256") != json_digest(item)
        or record.get("native_runtime_sha256") != sha(collection_config.producer_lock_path())
        or record.get("profile_id") != item["profile"]["id"]
        or record.get("source_profile") != item["source_profile"]
        or record.get("source_inputs") != item["inputs"]
        or not isinstance(record.get("files"), dict)
        or record["files"].keys() != PREPARED_FILES
    ):
        raise ValueError("Prepared profile or native runtime changed; prepare again explicitly")
    collection_config.assert_frozen()
    for name, digest in record["files"].items():
        relative = Path(name)
        file = folder / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(p.is_symlink() for p in (file, *file.parents))
            or not file.is_file()
            or sha(file) != digest
        ):
            raise ValueError("Prepared vehicle file changed: " + name)
    # The runner consumes these resolved dictionaries, not just the input files.
    # Bind them to both the cached snapshot and the original validated recipe.
    profile = load_profile(folder / "vehicle.yaml")
    pack = load_battery(folder / "vehicle.yaml")
    prop = load_propulsion(folder / "vehicle.yaml")
    for key in ("parameters_file", "resolved_parameters_sha256"):
        prop.pop(key, None)
    if (
        physical_content(profile) != physical_content(item["profile"])
        or pack != item["battery"]
        or record.get("vehicle") != profile
        or record.get("battery") != pack
        or record.get("propulsion") != prop
        or record.get("observation_vehicle") != observation_profile(profile, pack)
        or record.get("hover_estimate") != item["hover"]
        or record.get("ai_compatible") is not False
        or record.get("flight_verified") is not False
    ):
        raise ValueError("Prepared manifest disagrees with its validated physical inputs")
    battery_values = [
        pack["r0_ohm"],
        pack["r1_ohm"],
        pack["c1_f"],
        pack["capacity_ah"],
        *pack["ocv_cell_v"],
    ]
    prop_values = [
        prop[key]
        for key in (
            "thrust_coefficient",
            "torque_coefficient",
            "friction_nms",
            "coulomb_nm",
            "esc_efficiency",
            "resistance_ohm",
        )
    ]
    texts = {
        "battery_runtime.txt": " ".join(map(str, battery_values)) + "\n",
        "propulsion_runtime.txt": " ".join(map(str, prop_values)) + "\n",
        "profile.hpp": header_text(profile, sha(folder / "vehicle.yaml"), pack),
    }
    if any((folder / name).read_text() != expected for name, expected in texts.items()):
        raise ValueError("Prepared native inputs disagree with the physical profile")
    return folder, record


def prepare(path, *, jobs=4, retry=False):
    """A finite compile/asset/numeric check, never a Gazebo/PX4 flight process."""
    from . import vehicle_assets

    if type(jobs) is not int or jobs < 1:
        raise ValueError("Positive build job count required")
    collection_config.assert_frozen()
    item = recipe(path)
    folder = location(item)
    lock = ROOT / ".cache/vehicle-prepare.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Existing data jobs keep exclusive control of the shared source/runtime.
        collection_lock = ROOT.parent / "Data/operations/collection.lock"
        collection_lock.parent.mkdir(parents=True, exist_ok=True)
        with collection_lock.open("a") as collection_guard:
            fcntl.flock(collection_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _prepare(item, folder, jobs, retry, vehicle_assets)


def _prepare(item, folder, jobs, retry, assets):
    if folder.exists():
        if (folder / "manifest.json").is_file():
            try:
                read_prepared(item["source_profile"])
            except (ValueError, KeyError):
                if not retry:
                    raise ValueError(
                        "Previous preparation no longer matches; preserved. Use --retry explicitly"
                    ) from None
            else:
                return dict(
                    status="ALREADY_PREPARED_NO_FLIGHT", manifest=str(folder / "manifest.json")
                )
        if not retry:
            raise ValueError(
                "Previous preparation preserved; --retry is required for a new attempt"
            )
        preserved = (
            BUILD / "failed" / (folder.parent.name + "-" + folder.name + "-" + uuid.uuid4().hex)
        )
        preserved.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(preserved)
    folder.mkdir(parents=True, exist_ok=False)
    status = folder / "prepare-state.json"
    dump(status, dict(status="PREPARING", pid=os.getpid(), flight_started=False))
    try:
        pack = item["battery"]
        prop = copy.deepcopy(item["profile"]["propulsion"])
        for key in ("parameters_file", "resolved_parameters_sha256"):
            prop.pop(key, None)
        dump(folder / "battery.json", pack)
        dump(folder / "propulsion.json", prop)
        profile = copy.deepcopy(item["profile"])
        profile["battery"]["parameters_file"] = "battery.json"
        profile["battery"]["resolved_parameters_sha256"] = sha(folder / "battery.json")
        profile["propulsion"] = dict(
            parameters_file="propulsion.json",
            resolved_parameters_sha256=sha(folder / "propulsion.json"),
        )
        (folder / "vehicle.yaml").write_text(yaml.safe_dump(profile, sort_keys=True))
        profile = load_profile(folder / "vehicle.yaml")
        generate(folder / "vehicle.yaml", folder / "profile.hpp")
        (folder / "battery_runtime.txt").write_text(
            " ".join(
                map(
                    str,
                    [
                        pack["r0_ohm"],
                        pack["r1_ohm"],
                        pack["c1_f"],
                        pack["capacity_ah"],
                        *pack["ocv_cell_v"],
                    ],
                )
            )
            + "\n"
        )
        (folder / "propulsion_runtime.txt").write_text(
            " ".join(
                map(
                    str,
                    [
                        prop["thrust_coefficient"],
                        prop["torque_coefficient"],
                        prop["friction_nms"],
                        prop["coulomb_nm"],
                        prop["esc_efficiency"],
                        prop["resistance_ohm"],
                    ],
                )
            )
            + "\n"
        )
        commands = [
            [
                "cmake",
                "-S",
                str(ROOT / "sim"),
                "-B",
                str(folder / "build"),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DDRONE_PROFILE_HEADER=" + str(folder / "profile.hpp"),
            ],
            [
                "cmake",
                "--build",
                str(folder / "build"),
                "--target",
                "DronePowertrain",
                "profile_physics_test",
                "--parallel",
                str(jobs),
            ],
            [str(folder / "build/profile_physics_test")],
        ]
        env = {
            **os.environ,
            "PATH": str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", ""),
        }
        for argv in commands:
            with (folder / "prepare.log").open("a") as log:
                log.write(json.dumps(argv) + "\n")
                log.flush()
                subprocess.run(
                    argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        shutil.copy2(folder / "build/libDronePowertrain.so", folder / "libDronePowertrain.so")
        preview = folder / "preview"
        (preview / "reference").mkdir(parents=True)
        assets.prepare_world(
            preview,
            profile,
            profile_path=folder / "vehicle.yaml",
            plugin_path=folder / "libDronePowertrain.so",
        )
        assets.prepare_fc(
            preview,
            ROOT / "vendor/PX4-Autopilot/build/px4_sitl_default/etc",
            profile,
            pack,
            hover_throttle=item["hover"]["hover_throttle"],
        )
        observation = observation_profile(profile, pack)
        record = dict(
            schema="drone.simulation.profile/1",
            profile_id=profile["id"],
            recipe_sha256=json_digest(item),
            source_profile=item["source_profile"],
            source_inputs=item["inputs"],
            native_runtime_sha256=sha(collection_config.producer_lock_path()),
            vehicle=profile,
            battery=pack,
            propulsion=prop,
            observation_vehicle=observation,
            hover_estimate=item["hover"],
            ai_compatible=False,
            flight_verified=False,
            files={name: sha(folder / name) for name in sorted(PREPARED_FILES)},
        )
        if json_digest(recipe(item["source_profile"])) != json_digest(item):
            raise ValueError("Profile or simulator sources changed during preparation")
        collection_config.assert_frozen()
        dump(folder / "manifest.json", record)
        dump(status, dict(status="PREPARED_NO_FLIGHT", pid=os.getpid(), flight_started=False))
        return dict(
            status="PREPARED_NO_FLIGHT",
            manifest=str(folder / "manifest.json"),
            hover_estimate=item["hover"],
            ai_compatible=False,
            flight_verified=False,
        )
    except BaseException as error:
        dump(
            status,
            dict(status="FAILED_PRESERVED", reason=str(error), retry=False, flight_started=False),
        )
        raise
