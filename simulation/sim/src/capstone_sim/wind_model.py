"""Evaluator-only OU definitions and checks; no observation or health feature changes."""

import json
import math
from pathlib import Path

from . import bounded_wind

REVISION = "stationary-exact-ou-enu-v1"
TRACE_SUFFIX = ".wind-trace.jsonl"


def metadata(seed, conditions):
    sigma = conditions["gust_sigma"]
    return dict(revision=REVISION, process="independent_exact_discrete_Ornstein_Uhlenbeck",
                seed_u32=seed % 2**32, sigma_per_axis_mps=sigma, tau_s=2.0,
                axes=["east", "north", "up"], mean_enu_mps=[conditions["wind_e"], conditions["wind_n"], 0.0],
                velocity_convention="air_mass_velocity_towards_ENU_not_meteorological_from",
                clock="native_simulation_boot_seconds_shared_with_PX4",
                update="native_elapsed_at_least0.01s_exact_recorded_dt_stationary_initial_state",
                gaussian_support="unbounded_no_clipping_mean5_is_not_instantaneous_peak5",
                radial_planning_allowance_mps=3*sigma,
                radial3sigma_point_coverage=1.0 if sigma == 0 else math.erf(3/math.sqrt(2))-math.sqrt(2/math.pi)*3*math.exp(-4.5),
                qualification="actual_mission_airspeed8_mu0.1_and_existing_gates_fail_closed_for_every_seed")


def expected_metadata(scenario):
    value = scenario.get("wind_profile")
    if value is not None:
        return bounded_wind.definition(scenario["seed"], value, scenario.get("joint_phase_policy", {}).get("revision"))
    return metadata(scenario["seed"], scenario["parameters"])


def verify_record(entry, definition):
    if definition["revision"] == bounded_wind.REVISION:
        return bounded_wind.verify_record(entry, definition)
    if type(entry["wind_update_index"]) is not int or entry["wind_update_index"] < 1:
        raise ValueError("wind update index")
    stamp = entry["wind_update_time_us"]
    if type(stamp) is not int or not 0 <= stamp <= entry["time_us"]:
        raise ValueError("wind update clock")
    for key in ("gust_enu_mps", "wind_enu_mps", "ground_velocity_enu_mps"):
        values = entry[key]
        if (not isinstance(values, list) or len(values) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise ValueError("finite ENU vector: " + key)
    if any(abs(total-mean-gust) > 2e-9 for total, mean, gust in zip(
            entry["wind_enu_mps"], definition["mean_enu_mps"], entry["gust_enu_mps"])):
        raise ValueError("mean plus gust vector")
    if definition["sigma_per_axis_mps"] == 0 and any(entry["gust_enu_mps"]):
        raise ValueError("nonzero gust at sigma0")
    if (type(entry["relative_airspeed_mps"]) not in (int, float)
            or not math.isfinite(entry["relative_airspeed_mps"]) or entry["relative_airspeed_mps"] < 0):
        raise ValueError("finite relative airspeed")
    relative = math.dist(entry["ground_velocity_enu_mps"], entry["wind_enu_mps"])
    if abs(relative-entry["relative_airspeed_mps"]) > 1e-8:
        raise ValueError("wind-relative frame norm")


def trace_records(path, definition):
    with Path(path).open() as stream:
        header = json.loads(stream.readline())
        expected = dict(schema="drone.ou-wind-trace/1", kind="configuration",
                        seed_u32=definition["seed_u32"], sigma_per_axis_mps=definition["sigma_per_axis_mps"],
                        tau_s=definition["tau_s"], mean_enu_mps=definition["mean_enu_mps"])
        if header != expected:
            raise ValueError("OU native configuration")
        previous = None
        for index, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ValueError("Incomplete OU update")
            point = json.loads(line)
            if (point.get("kind") != "update" or point.get("index") != index
                    or type(point.get("index")) is not int
                    or type(point.get("time_us")) is not int or point["time_us"] < 0
                    or type(point.get("dt_s")) not in (int, float)
                    or not math.isfinite(point["dt_s"]) or point["dt_s"] < .01-1e-9):
                raise ValueError("OU update order/clock/dt")
            gust = point["gust_enu_mps"]
            if (not isinstance(gust, list) or len(gust) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in gust)):
                raise ValueError("OU update vector")
            if previous is not None and (point["time_us"] <= previous["time_us"]
                    or abs((point["time_us"]-previous["time_us"])/1e6-point["dt_s"]) > 2e-6):
                raise ValueError("OU elapsed time mismatch")
            previous = point
            yield point
        if previous is None:
            raise ValueError("No OU updates")


def verify_trace(truth_path, definition):
    """Bind every applied wind to its native update, including pre/post mission."""
    if definition["revision"] == bounded_wind.REVISION:
        return bounded_wind.verify_trace(truth_path, definition)
    truth_path = Path(truth_path)
    updates = iter(trace_records(str(truth_path)+TRACE_SUFFIX, definition))
    current = None
    samples = 0
    with truth_path.open() as stream:
        for line in stream:
            entry = json.loads(line)
            verify_record(entry, definition)
            while current is None or current["index"] < entry["wind_update_index"]:
                try:
                    current = next(updates)
                except StopIteration:
                    raise ValueError("Truth references missing OU update") from None
            if (current["index"] != entry["wind_update_index"]
                    or current["time_us"] != entry["wind_update_time_us"]
                    or any(abs(a-b) > 2e-9 for a, b in zip(current["gust_enu_mps"], entry["gust_enu_mps"]))):
                raise ValueError("Applied wind does not match native OU trace")
            samples += 1
    for current in updates:
        pass
    if not samples or current is None:
        raise ValueError("Empty OU wind evidence")
    return dict(status="passed", truth_samples=samples, native_updates=current["index"])
