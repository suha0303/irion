"""Built-in X500 collection configuration; physical values are explicit profiles."""

import json
import math
import os
import shutil
import stat
import struct
from pathlib import Path

import yaml

from .runtime_manifest import PLUGIN as CURRENT_PLUGIN
from .runtime_manifest import RUNTIME_MANIFEST, TRANSPORT, verify_runtime

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "profiles/collection"
PLUGIN = ROOT / CURRENT_PLUGIN
HEADER = DIRECTORY / "vehicle_generated.hpp"
NAME = "collection"
# Read-only identities for previously recorded evidence. New runs use NAME.
LEGACY_NAME = "r3_long15"
LEGACY_BATTERY_ID = "vtc6_4s4p_r3_long15"
LEGACY_REVISION = "vtc6-4s4p-r3-long15"
LEGACY_PROFILE_PATH = "profiles/r3_long15/vehicle.yaml"
LEGACY_PROFILE_SHA256 = "ed634305dae2b3d147b82303b096e1d7c558be79940bbd4114c00c7793819958"
NORMAL = {
    "hover_normal", "altitude_normal", "horizontal_normal", "yaw_normal",
    "waypoint_normal", "low_soc_normal", "wind_normal", "cold_normal", "payload_normal",
}
ANOMALY = {
    "battery_resistance_existing", "propulsion_loss_existing",
    "battery_resistance_step", "propulsion_loss_step",
    "battery_resistance_ramp", "propulsion_loss_ramp",
}


def validate_request(scenario, duration, fault_at_s):
    if scenario not in NORMAL | ANOMALY:
        raise ValueError("scenario outside collection scope")
    maximum = 300 if scenario == "low_soc_normal" else 900 if scenario in NORMAL else 480
    if not math.isfinite(duration) or not 0 < duration <= maximum:
        raise ValueError(f"Collection {scenario} duration must be within (0,{maximum}] seconds")
    if scenario.endswith(("step", "ramp")):
        if fault_at_s != 180:
            raise ValueError("Collection dynamic anomaly requires --fault-at-s 180 PX4 boot seconds")
    elif fault_at_s is not None:
        raise ValueError("Collection fault onset only applies to dynamic anomalies")


def battery():
    return json.loads((DIRECTORY / "battery.json").read_text())


def producer_lock_path():
    return ROOT / RUNTIME_MANIFEST


def transport_library():
    library = ROOT / TRANSPORT
    alias = library.parent / "libgz-transport13.so.13"
    if not library.is_file() or alias.resolve() != library:
        raise ValueError("Pinned private Gazebo transport library/SONAME missing or replaced")
    return library


def transport_environment(environment):
    library = transport_library()
    result = dict(environment)
    if library is not None:
        previous = result.get("LD_LIBRARY_PATH", "")
        result["LD_LIBRARY_PATH"] = str(library.parent) + (":" + previous if previous else "")
    return result


def assert_frozen():
    verify_runtime(ROOT)


def vehicle():
    return yaml.safe_load((DIRECTORY / "vehicle.yaml").read_text())


def fc_parameters():
    p = battery()
    return {"BAT1_N_CELLS": p["ns"], "BAT1_CAPACITY": p["capacity_ah"] * 1000,
            # PX4 applies this per-series-cell equivalent to total pack current.
            "BAT1_R_INTERNAL": p["r0_ohm"] / p["ns"],
            # No range sensor is fitted: PX4's range-sensor-only crawl speed
            # does not protect touchdown from the default 0.7 m/s impact.
            "MPC_LAND_SPEED": 0.3}


def prepare_fc(run, upstream_etc, *, gui_preview=False):
    """Patch a private run-local startup copy, never the PX4 binary or vendor tree."""
    target = Path(run) / "reference/px4_etc"
    shutil.copytree(upstream_etc, target, symlinks=False)
    airframe = target / "init.d-posix/airframes/4001_gz_x500"
    lines = airframe.read_text().splitlines()
    expected = fc_parameters()
    for key, value in expected.items():
        matches = [i for i, line in enumerate(lines) if line.startswith(f"param set {key} ")]
        if key == "MPC_LAND_SPEED" and not matches:
            lines.append(f"param set {key} {value:.12g}")
            continue
        if len(matches) != 1:
            raise ValueError(f"ambiguous upstream FC parameter {key}")
        lines[matches[0]] = f"param set {key} {value:.12g}"
    airframe.write_text("\n".join(lines) + "\n")
    # PX4's default aliases instances above9 onto14549. Only the private collection
    # startup copy changes; every one of our16 lanes needs its own receiver.
    mavlink = target / "init.d-posix/px4-rc.mavlink"
    text = mavlink.read_text()
    alias = '[ "$px4_instance" -gt 9 ] && udp_offboard_port_remote=14549 # use the same ports for more than 10 instances to avoid port overlaps'
    if text.count(alias) != 1:
        raise ValueError("unexpected PX4 offboard port alias policy")
    text = text.replace(alias, "# collection: unique14540+instance for lanes0..15")
    # The default GCS destination14550 is lane10's offboard receiver. Keep GCS
    # telemetry in a separate range, including when no ground station is open.
    gcs = "mavlink start -x -u $udp_gcs_port_local -r 4000000 -f"
    lines = text.splitlines()
    if lines.count(gcs) != 1:
        raise ValueError("unexpected PX4 GCS startup policy")
    index = lines.index(gcs)
    lines[index:index + 1] = [
        "udp_gcs_port_remote=$((14600+px4_instance))",
        gcs + " -o $udp_gcs_port_remote",
    ]
    if gui_preview:
        # QGC's default UDP receiver. Disable inter-link forwarding so QGC's
        # heartbeat cannot reach the strict PX4-only offboard receiver.
        lines = [line.replace(" -f", "") for line in lines]
        lines[index] = "udp_gcs_port_remote=14550"
    mavlink.write_text("\n".join(lines) + "\n")
    return target


def fc_verified(observed, *, expected=None):
    current = fc_parameters()
    if expected is None:
        expected = current
    # Old sealed runs recorded only the battery parameters. Verify the exact
    # recorded policy without retroactively imposing a new touchdown setting.
    if set(expected) not in (set(current), set(current) - {"MPC_LAND_SPEED"}):
        return False
    if any(not math.isclose(value, current[key], rel_tol=1e-5, abs_tol=1e-6)
           for key, value in expected.items()):
        return False
    return all(key in observed and math.isclose(observed[key], value, rel_tol=1e-5, abs_tol=1e-6)
               for key, value in expected.items())


class FCParameterGate:
    """Bounded read-only handshake on wall time, independent of SITL speed."""

    def __init__(self, timeout_wall_s=60.0):
        if not math.isfinite(timeout_wall_s) or timeout_wall_s <= 0:
            raise ValueError("positive finite FC response timeout required")
        self.timeout_wall_s = timeout_wall_s
        self.expected = fc_parameters()
        self.observed = {}
        self.wire = {}
        self.started = None
        self.next_request = 0.0
        self.requests = []

    def observe(self, name, value, param_type):
        self.wire[name] = dict(param_type=int(param_type), raw_value=float(value))
        self.observed[name] = decode_fc_parameter(name, value, param_type)

    def step(self, wall_now, send):
        if self.started is None:
            self.started = wall_now
        for name, value in self.observed.items():
            if not math.isclose(value, self.expected[name], rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(f"Collection FC parameter inconsistent: {name}={value}")
        missing = [name for name in self.expected if name not in self.observed]
        if not missing:
            return True
        if wall_now - self.started >= self.timeout_wall_s:
            raise TimeoutError("Collection FC parameter response timeout: " + ", ".join(missing))
        if wall_now >= self.next_request:
            for name in missing:
                send(name)
            self.requests.append(dict(elapsed_wall_s=wall_now - self.started, names=missing))
            self.next_request = wall_now + 1.0
        return False

    def evidence(self, wall_now, verified, reason):
        return dict(expected=self.expected, observed=self.observed, wire=self.wire,
                    verified=verified, reason=reason, requests=self.requests,
                    timeout_wall_s=self.timeout_wall_s, request_interval_wall_s=1.0,
                    elapsed_wall_s=wall_now - self.started if self.started is not None else 0.0)


def decode_fc_parameter(name, value, param_type):
    """Decode PX4 PARAM_VALUE's byte-wise INT32 / numeric REAL32 encoding.

    PX4 mavlink_parameters.cpp copies INT32 bits into the float transport field.
    Do not numerically cast that float or accept a different declared wire type.
    """
    expected_types = {"BAT1_N_CELLS": 6, "BAT1_CAPACITY": 9, "BAT1_R_INTERNAL": 9,
                      "MPC_LAND_SPEED": 9}
    if name not in expected_types or param_type != expected_types[name]:
        raise ValueError(f"Collection FC parameter type mismatch: {name} type={param_type}")
    if not math.isfinite(value):
        raise ValueError(f"Collection FC nonfinite parameter: {name}")
    if param_type == 6:  # MAV_PARAM_TYPE_INT32: bit-preserving, not float-to-int.
        return struct.unpack("<i", struct.pack("<f", value))[0]
    return float(value)  # MAV_PARAM_TYPE_REAL32


def owned_socket_identity(path, pid):
    """Claim only a pathname backed by an FD in this run's own PX4 process."""
    if not path.exists():
        return None
    info = path.lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("Collection PX4 socket is not an owned socket")
    inodes = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    for line in Path("/proc/net/unix").read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 8 and fields[7] == str(path) and fields[6] in inodes:
            return dict(device=info.st_dev, inode=info.st_ino, uid=info.st_uid, px4_pid=pid)
    return None


def release_owned_socket(path, identity, process):
    """After own PX4 exit, remove only its unchanged, no-longer-bound socket."""
    if not os.path.lexists(path):
        return "already_removed"
    if identity is None or process.poll() is None or process.pid != identity["px4_pid"]:
        raise ValueError("Collection PX4 socket ownership/exit unconfirmed; preserved")
    info = path.lstat()
    if (not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid()
            or (info.st_dev, info.st_ino, info.st_uid) !=
            (identity["device"], identity["inode"], identity["uid"])):
        raise ValueError("Collection PX4 socket identity changed; preserved")
    for line in Path("/proc/net/unix").read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 8 and fields[7] == str(path):
            raise ValueError("Collection PX4 socket still bound; preserved")
    path.unlink()
    return "owned_stale_socket_removed"
