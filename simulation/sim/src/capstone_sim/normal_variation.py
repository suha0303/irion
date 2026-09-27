"""Bounded healthy-condition and smooth-path variation, independent of faults."""

import math
import random
from typing import Any

FIELDS = {"soc", "temperature", "payload", "wind_e", "wind_n", "gust_sigma", "path"}
PATH_FIELDS = {
    "kind",
    "height",
    "radius",
    "omega",
    "vertical_amplitude",
    "yaw_amplitude",
    "phase",
    "heading",
}
KINDS = {"hover", "line", "ellipse", "figure8", "lissajous", "altitude", "yaw", "mixed", "random"}


def validate(spec):
    if set(spec) != FIELDS:
        raise ValueError("Exact normal variation fields required; health/fault overrides forbidden")
    for key in FIELDS - {"path"}:
        if type(spec[key]) not in (int, float) or not math.isfinite(spec[key]):
            raise ValueError("Finite normal condition required: " + key)
    if not (
        0.4 <= spec["soc"] <= 0.95
        and 283.15 <= spec["temperature"] <= 303.15
        and 0 <= spec["payload"] <= 0.2
        and 0 <= spec["gust_sigma"] <= 0.4
        and math.hypot(spec["wind_e"], spec["wind_n"]) <= 2.000001
    ):
        raise ValueError("Normal variation exceeds declared engineering envelope")
    p = spec["path"]
    validate_path(p)
    if p["kind"] == "random":
        from .random_flight import speed_limit

        if p["limits"]["horizontal_speed_mps"] > speed_limit(spec) + 1e-8:
            raise ValueError("Random speed exceeds wind-adjusted airspeed margin")
    return spec


def validate_path(p):
    if p.get("kind") == "random":
        from .random_flight import validate

        return validate(p)
    extended = p.get("kind") in {"mixed", "random"}
    expected = PATH_FIELDS | {"program", "duration_s"} if extended else PATH_FIELDS
    if set(p) != expected:
        raise ValueError("Exact path fields required")
    if p["kind"] not in KINDS or any(
        type(p[k]) not in (int, float) or not math.isfinite(p[k]) for k in PATH_FIELDS - {"kind"}
    ):
        raise ValueError("Invalid smooth path")
    if not (
        3 <= p["height"] <= 4
        and 0.5 <= p["radius"] <= 2
        and 0.04 <= p["omega"] <= 0.12
        and 0 <= p["vertical_amplitude"] <= 0.6
        and 0 <= p["yaw_amplitude"] <= 0.5
        and 0 <= p["phase"] <= 2 * math.pi
        and -math.pi <= p["heading"] <= math.pi
    ):
        raise ValueError("Path exceeds declared engineering envelope")
    if extended:
        from .flight_program import validate_program

        validate_program(p, validate_path)


def sample(scenario, seed):
    """Freeze all sampled values in the case manifest before any run starts."""
    rng = random.Random(seed)
    u = rng.uniform
    speed = u(0, 0.7)
    direction = u(-math.pi, math.pi)
    spec: dict[str, Any] = dict(
        soc=u(0.87, 0.95),
        temperature=u(293.15, 303.15),
        payload=u(0, 0.08),
        wind_e=0.0,
        wind_n=0.0,
        gust_sigma=u(0, 0.12),
    )
    kind = {
        "altitude_normal": "altitude",
        "yaw_normal": "yaw",
        "horizontal_normal": rng.choice(["line", "ellipse"]),
        "waypoint_normal": rng.choice(["ellipse", "figure8", "lissajous"]),
    }.get(scenario, "hover")
    if scenario == "wind_normal":
        speed = u(0.5, 2)
        spec["gust_sigma"] = u(0.1, 0.4)
        kind = rng.choice(["hover", "line", "ellipse"])
    elif scenario == "payload_normal":
        spec["payload"] = u(0.1, 0.2)
        kind = rng.choice(["hover", "line", "yaw"])
    elif scenario == "cold_normal":
        spec.update(temperature=u(283.15, 291.15), soc=u(0.9, 0.95), payload=u(0, 0.05))
        kind = rng.choice(["hover", "line"])
    elif scenario == "low_soc_normal":
        spec.update(soc=u(0.4, 0.5), payload=u(0, 0.05))
        speed = u(0, 0.4)
        kind = rng.choice(["hover", "line", "yaw"])
    spec.update(wind_e=speed * math.cos(direction), wind_n=speed * math.sin(direction))
    spec["path"] = dict(
        kind=kind,
        height=u(3, 4),
        radius=u(0.5, 2),
        omega=u(0.04, 0.12),
        vertical_amplitude=u(0.2, 0.6),
        yaw_amplitude=u(0.15, 0.5),
        phase=u(0, 2 * math.pi),
        heading=u(-math.pi, math.pi),
    )
    return validate(spec)


def target(t, spec):
    """Continuous, smoothly introduced NED positions; no instantaneous waypoint jumps."""
    p = spec["path"]
    if p["kind"] in {"mixed", "random"}:
        from .flight_program import target as program_target

        return program_target(t, p)
    a, w, q = p["radius"], p["omega"], p["phase"]
    theta = w * max(t, 0)
    blend = 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0) / 20, 1))
    x = y = yaw = 0.0
    z = -3 - blend * (p["height"] - 3)
    if p["kind"] == "line":
        x = a * math.sin(theta)
    elif p["kind"] == "ellipse":
        x, y = a * math.sin(theta), 0.7 * a * (1 - math.cos(theta))
    elif p["kind"] == "figure8":
        x, y = a * math.sin(theta), 0.5 * a * math.sin(2 * theta)
    elif p["kind"] == "lissajous":
        x, y = a * math.sin(theta), 0.5 * a * (math.sin(1.5 * theta + q) - math.sin(q))
    elif p["kind"] == "altitude":
        z -= blend * p["vertical_amplitude"] * math.sin(theta)
    elif p["kind"] == "yaw":
        yaw = blend * p["yaw_amplitude"] * math.sin(theta)
    h = p["heading"]
    return (
        blend * (x * math.cos(h) - y * math.sin(h)),
        blend * (x * math.sin(h) + y * math.cos(h)),
        z,
        yaw,
    )
