"""Bounded C3 waypoint and composite paths; finite seeded programs, no planner service."""

import copy
import math
import random

BASIC = ("hover", "line", "ellipse", "figure8", "lissajous", "altitude", "yaw")


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u**4 * (35 + u * (-84 + u * (70 - 20 * u)))


def build(spec, mode, seed, duration, basic_index):
    from .normal_variation import validate

    if mode == "random":
        from .random_flight import build as build_random

        result = copy.deepcopy(spec)
        result["path"] = build_random(spec, seed, duration)
        return validate(result)
    rng = random.Random(seed ^ 0xC3B0300)
    result = copy.deepcopy(spec)
    p = result["path"]
    allowed = ("hover", "line", "yaw") if spec["soc"] < 0.6 else BASIC
    p["kind"] = allowed[basic_index % len(allowed)]
    if mode == "basic":
        return validate(result)
    p.update(
        kind=mode, duration_s=duration, radius=min(p["radius"], 1.2), omega=min(p["omega"], 0.08)
    )
    if mode == "mixed":
        count = 2 if duration == 180 else 6
        kinds = list(allowed)
        rng.shuffle(kinds)
        p["program"] = []
        for i in range(count):
            component = {k: v for k, v in p.items() if k not in {"program", "duration_s"}}
            component["kind"] = kinds[i % len(kinds)]
            p["program"].append(
                dict(start_s=i * duration / count, duration_s=duration / count, path=component)
            )
    else:
        raise ValueError("Unknown flight program mode")
    return validate(result)


def validate_program(p, validate_path):
    if p.get("kind") == "random":
        from .random_flight import validate

        return validate(p)
    if p["duration_s"] not in (180, 480, 900) or p["kind"] != "mixed":
        raise ValueError("Finite180/480/900s flight program required")
    if p["radius"] > 1.2 or p["omega"] > 0.08 or not isinstance(p["program"], list):
        raise ValueError("Program motion bound exceeded")
    if p["kind"] == "mixed":
        if len(p["program"]) not in (2, 6):
            raise ValueError("Finite composite component count required")
        end = 0.0
        for item in p["program"]:
            if set(item) != {"start_s", "duration_s", "path"} or item["path"]["kind"] not in BASIC:
                raise ValueError("Explicit basic composite component required")
            validate_path(item["path"])
            if (
                not math.isfinite(item["start_s"])
                or not 80 <= item["duration_s"] <= 180
                or abs(item["start_s"] - end) > 1e-6
            ):
                raise ValueError("Contiguous bounded composite timing required")
            if item["path"]["radius"] > 1.2 or item["path"]["omega"] > 0.08:
                raise ValueError("Composite component motion exceeds bound")
            end += item["duration_s"]
        if abs(end - p["duration_s"]) > 1e-6:
            raise ValueError("Composite duration mismatch")


def component(t, p):
    a, w, q = p["radius"], p["omega"], p["phase"]
    theta = w * t
    x = y = yaw = 0.0
    z = -(p["height"] - 3)
    kind = p["kind"]
    if kind == "line":
        x = a * math.sin(theta)
    elif kind == "ellipse":
        x, y = a * math.sin(theta), 0.7 * a * (1 - math.cos(theta))
    elif kind == "figure8":
        x, y = a * math.sin(theta), 0.5 * a * math.sin(2 * theta)
    elif kind == "lissajous":
        x, y = a * math.sin(theta), 0.5 * a * (math.sin(1.5 * theta + q) - math.sin(q))
    elif kind == "altitude":
        z -= p["vertical_amplitude"] * math.sin(theta)
    elif kind == "yaw":
        yaw = p["yaw_amplitude"] * math.sin(theta)
    h = p["heading"]
    return x * math.cos(h) - y * math.sin(h), x * math.sin(h) + y * math.cos(h), z, yaw


def target(t, p):
    t = min(max(t, 0.0), p["duration_s"])
    if p["kind"] == "random":
        from .random_flight import kinematics

        return kinematics(t, p)[0]
    item = next((v for v in p["program"] if t < v["start_s"] + v["duration_s"]), p["program"][-1])
    local = t - item["start_s"]
    envelope = smooth(local / 25) * smooth((item["duration_s"] - local) / 25)
    x, y, z, yaw = component(local, item["path"])
    return envelope * x, envelope * y, -3 + envelope * z, envelope * yaw
