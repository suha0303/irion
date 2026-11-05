"""Explicit bounded environments; latent OU scale is dimensionless, not wind sigma."""

import json
import math
from pathlib import Path

REVISION = "bounded-latent-ou-c2-enu-v1"
FIELDS = {"revision", "environment", "direction_deg", "variation"}
ENVIRONMENTS = {"calm": (0., 0., 0.), "low": (1.25, 1., 2.5), "strong": (3.75, 1., 5.)}


def profile(environment, direction_deg=0., variation=1.):
    if type(direction_deg) not in (int, float) or not math.isfinite(direction_deg):
        raise ValueError("Finite wind direction required")
    return validate(dict(revision=REVISION, environment=environment,
                         direction_deg=direction_deg % 360, variation=variation))


def validate(value):
    if not isinstance(value, dict) or set(value) != FIELDS or value["revision"] != REVISION:
        raise ValueError("Exact bounded wind profile required")
    if value["environment"] not in ENVIRONMENTS:
        raise ValueError("Bounded wind environment must be calm/low/strong")
    if (type(value["direction_deg"]) not in (int, float)
            or not math.isfinite(value["direction_deg"]) or not 0 <= value["direction_deg"] < 360
            or type(value["variation"]) not in (int, float)
            or not math.isfinite(value["variation"]) or not .5 <= value["variation"] <= 1.5):
        raise ValueError("Bounded wind direction0..360 and dimensionless variation0.5..1.5 required")
    return value


def speed_bound(value):
    return ENVIRONMENTS[validate(value)["environment"]][2]


def definition(seed, value, phase_revision=None):
    validate(value)
    center, amplitude, bound = ENVIRONMENTS[value["environment"]]
    result = dict(revision=REVISION, profile=value, seed_u32=seed % 2**32,
                axes=["east", "north", "up"], direction_convention="towards_north0_east90_degrees",
                speed_midpoint_mps=center, speed_amplitude_mps=amplitude,
                generator_speed_bounds_mps=[center-amplitude, center+amplitude],
                environment_speed_bound_mps=bound,
                latent_axes=["speed", "azimuth", "elevation"], latent_std_dimensionless=1.,
                latent_tau_s=2., latent_knot_period_s=1.,
                latent_std_scope="OU knots; C2 interpolation is not variance normalized", interpolation="quintic_smoothstep_C2",
                direction_amplitude_rad=math.pi/3, elevation_amplitude_rad=math.pi/12,
                transformation="speed=center+amplitude*tanh(alpha*z0); azimuth/elevation=tanh(alpha*z1/z2)",
                variation_meaning="dimensionless_latent_multiplier_not_transformed_wind_standard_deviation",
                clock="absolute_native_simulation_boot_seconds",
                bound_policy="by_construction_for_all_latent_values_and_times_no_posthoc_clipping",
                scope="engineering_environment_not_fitted_to_measured_turbulence_spectrum")

    if phase_revision is not None:
        if phase_revision != "joint-native-consumption-c2-v2":
            raise ValueError("Unknown joint wind phase revision")
        result.update(phase_revision=phase_revision, clock="native_controller_consumption_relative_seconds_with_C2_initial_plateau")
    return result


def native_header(value):
    keys = ("profile", "seed_u32", "latent_std_dimensionless", "latent_tau_s", "latent_knot_period_s",
            "speed_midpoint_mps", "speed_amplitude_mps", "direction_amplitude_rad", "elevation_amplitude_rad")
    result = dict(schema="drone.bounded-wind-trace/1", kind="configuration", **{key: value[key] for key in keys})
    if "phase_revision" in value:
        result.update(schema="drone.bounded-wind-trace/2", phase_revision=value["phase_revision"])
    return result


def verify_record(entry, value):
    if type(entry["wind_sample_index"]) is not int or entry["wind_sample_index"] < 1:
        raise ValueError("bounded sample index")
    for key in ("wind_enu_mps", "ground_velocity_enu_mps"):
        vector = entry[key]
        if (not isinstance(vector, list) or len(vector) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
            raise ValueError("finite bounded ENU vector")
    magnitude = math.sqrt(sum(v*v for v in entry["wind_enu_mps"]))
    minimum, maximum = entry["wind_observed_min_norm_mps"], entry["wind_observed_max_norm_mps"]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (minimum, maximum)):
        raise ValueError("finite physical-step wind extrema")
    environment = value["profile"]["environment"]
    inside = (minimum == maximum == 0 if environment == "calm" else
              0 < minimum <= maximum <= 2.5 if environment == "low" else 2.5 < minimum <= maximum <= 5)
    if not inside or not minimum-2e-9 <= magnitude <= maximum+2e-9:
        raise ValueError("bounded wind environment or physical-step extrema")
    relative = entry["relative_airspeed_mps"]
    if (type(relative) not in (int, float) or not math.isfinite(relative)
            or abs(relative-math.dist(entry["ground_velocity_enu_mps"], entry["wind_enu_mps"])) > 1e-8):
        raise ValueError("bounded wind-relative frame norm")
    if (type(entry["wind_force_evaluations"]) is not int
            or entry["wind_force_evaluations"] < entry["wind_sample_index"]):
        raise ValueError("physical wind evaluations")


def trace_records(path, value):
    with Path(path).open() as stream:
        if json.loads(stream.readline()) != native_header(value):
            raise ValueError("bounded native configuration")
        previous = None
        for index, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ValueError("Incomplete bounded wind sample")
            point = json.loads(line)
            if (point.get("kind") != "sample" or type(point.get("index")) is not int or point["index"] != index
                    or type(point.get("time_us")) is not int or point["time_us"] < 0
                    or type(point.get("sample_time_s")) not in (int, float)
                    or not math.isfinite(point["sample_time_s"])
                    or abs(point["sample_time_s"]*1e6-point["time_us"]) > 2
                    or point.get("latent_knot_index") != math.floor(point["weather_time_s"] if "phase_revision" in value else point["sample_time_s"])):
                raise ValueError("bounded sample clock/index")
            for key in ("wind_enu_mps", "latent_state"):
                vector = point[key]
                if (not isinstance(vector, list) or len(vector) != 3
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
                    raise ValueError("finite bounded native vector")
            if "phase_revision" in value and (type(point.get("weather_time_s")) not in (int, float)
                    or not math.isfinite(point["weather_time_s"]) or point["weather_time_s"] < 0):
                raise ValueError("Joint weather argument")
            if previous is not None and point["sample_time_s"] <= previous:
                raise ValueError("bounded time reversal")
            previous = point["sample_time_s"]
            yield point
        if previous is None:
            raise ValueError("Empty bounded trace")


def verify_trace(truth_path, value):
    truth_path = Path(truth_path)
    records = iter(trace_records(str(truth_path)+".wind-trace.jsonl", value))
    samples = 0
    origin = None
    if "phase_revision" in value:
        receipt = json.loads((truth_path.parent / "joint-origin.json").read_text())
        origin = receipt["actual_start_us"]
    with truth_path.open() as stream:
        for line in stream:
            entry = json.loads(line)
            verify_record(entry, value)
            try:
                point = next(records)
            except StopIteration:
                raise ValueError("Missing bounded wind sample") from None
            if (point["index"] != entry["wind_sample_index"] or point["time_us"] != entry["time_us"]
                    or any(abs(a-b) > 2e-9 for a, b in zip(point["wind_enu_mps"], entry["wind_enu_mps"]))):
                raise ValueError("Applied wind differs from bounded trace")
            if origin is not None and abs(point["weather_time_s"] - max(0., (point["time_us"]-origin-1_000_000)/1e6)) > 2e-6:
                raise ValueError("Applied joint weather phase differs from actual native origin")
            samples += 1
    if next(records, None) is not None or not samples:
        raise ValueError("Bounded trace length mismatch")
    return dict(status="passed", truth_samples=samples, native_samples=samples)
