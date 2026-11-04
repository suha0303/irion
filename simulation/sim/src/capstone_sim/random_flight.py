"""Continuous steering integrated into consistent position/velocity targets.

Rows: [seconds, north, east, down, unwrapped_yaw, vn, ve, vd]. These mark
input changes, not stopping points. Quintic velocity blends are C3 in position.
"""

import bisect
import math
import random

REVISION = "random-continuous-v3"
FIELDS = {"kind", "revision", "duration_s", "limits", "program"}
LIMITS = {
    "min_altitude_m": 3.0,
    "max_altitude_m": 30.0,
    "horizontal_speed_mps": 3.0,
    "vertical_speed_mps": 1.2,
    "acceleration_mps2": 1.4,
    "jerk_mps3": 1.5,
    "yaw_rate_radps": 0.6,
    "min_cruise_speed_fraction": 0.30,
    "min_input_seconds": 8.0,
    "max_input_seconds": 18.0,
}
# Exact maxima of the first two derivatives of quintic smoothstep.
D1, D2 = 1.875, 10 / math.sqrt(3)


def speed_limit(spec):
    # Planning margin, not a hard bound on stochastic wind.
    return min(3.0, 4.5 - math.hypot(spec["wind_e"], spec["wind_n"]) - 3 * spec["gust_sigma"])


def clamp(value, low, high):
    return min(max(value, low), high)


def position_controls(a, b):
    """Degree-six Bezier controls: convex hull bounds the entire segment."""
    dt = b[0] - a[0]
    result = [tuple(a[1:4])]
    for velocity in [a[5:8]] * 3 + [b[5:8]] * 3:
        result.append(tuple(x + dt * v / 6 for x, v in zip(result[-1], velocity)))
    return result


def minimum_speed(a, b):
    """Exact minimum norm on the line between horizontal velocity vectors."""
    dx, dy = b[5] - a[5], b[6] - a[6]
    squared = dx * dx + dy * dy
    u = clamp(-(a[5] * dx + a[6] * dy) / squared, 0, 1) if squared else 0
    return math.hypot(a[5] + u * dx, a[6] + u * dy)


def build(spec, seed, duration):
    if type(duration) not in (int, float) or duration not in (180, 480, 900):
        raise ValueError("Random flight requires 180, 480 or 900 seconds")
    limits = dict(LIMITS, horizontal_speed_mps=speed_limit(spec))
    if not 1 <= limits["horizontal_speed_mps"] <= 3:
        raise ValueError("Wind budget outside supported random flight limits")
    rng = random.Random(seed ^ 0x434F4E5433)
    for _ in range(64):
        # Only initial acceleration and final braking reach zero horizontal speed.
        count = round((duration - 22) / 13)
        durations = [rng.uniform(10, 16) for _ in range(count)]
        total = sum(durations)
        durations = [dt * (duration - 22) / total for dt in durations]
        points = [[0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0]]
        course = rng.uniform(-math.pi, math.pi)
        altitude_goal = rng.uniform(10, 26)
        speeds = [0.4, 0.65, 1.0, 0.8]
        for index, dt in enumerate([12.0, *durations, 10.0]):
            left = points[-1]
            final = index == len(durations) + 1
            if final:
                velocity, yaw = [0.0, 0.0, 0.0], left[4]
            else:
                if index % 4 == 0:
                    rng.shuffle(speeds)
                    altitude_goal = rng.uniform(10, 26)
                turn = rng.choice((8, 30, 45, 80))
                course += math.radians(rng.uniform(-turn, turn))
                speed = speeds[index % 4] * limits["horizontal_speed_mps"]
                # Climb/descent inputs change independently of horizontal steering.
                vd = -0.65 if index == 0 else clamp((-altitude_goal - left[3]) * 0.08, -0.65, 0.65)
                velocity = [speed * math.cos(course), speed * math.sin(course), vd]
                heading = course + rng.choice((0, 0, 0, math.pi / 6, -math.pi / 6))
                turn_yaw = math.atan2(math.sin(heading - left[4]), math.cos(heading - left[4]))
                yaw = left[4] + clamp(turn_yaw, -0.6 * dt / D1, 0.6 * dt / D1)
            position = [x + dt * (v + w) / 2 for x, v, w in zip(left[1:4], left[5:8], velocity)]
            points.append([left[0] + dt, *position, yaw, *velocity])
        points[-1][0] = float(duration)
        path = dict(
            kind="random", revision=REVISION, duration_s=duration, limits=limits, program=points
        )
        try:
            validate(path)
        except ValueError:
            continue
        return path
    raise ValueError("Could not generate a diverse continuous random flight for this seed")


def metrics(path):
    points = path["program"]
    controls = [p for a, b in zip(points, points[1:]) for p in position_controls(a, b)]
    acceleration, jerk, yaw_rates, distances = [], [], [], []
    for a, b in zip(points, points[1:]):
        dt = b[0] - a[0]
        dv = math.dist(a[5:8], b[5:8])
        acceleration.append(D1 * dv / dt)
        jerk.append(D2 * dv / dt**2)
        yaw_rates.append(D1 * abs(b[4] - a[4]) / dt)
        # Simpson quadrature for reporting arc length, not derivative acceptance.
        speed = [
            math.sqrt(sum(v * v for v in kinematics(a[0] + dt * i / 8, path)[1][:3]))
            for i in range(9)
        ]
        distances.append(
            dt
            / 24
            * sum(v * (1 if i in (0, 8) else 4 if i % 2 else 2) for i, v in enumerate(speed))
        )
    cruise_min = min(minimum_speed(a, b) for a, b in zip(points[1:-2], points[2:-1]))
    cruise_speeds = [math.hypot(p[5], p[6]) for p in points[1:-1]]
    return {
        "segments": len(points) - 1,
        "distance_m": sum(distances),
        "north_span_m": max(p[1] for p in points) - min(p[1] for p in points),
        "east_span_m": max(p[2] for p in points) - min(p[2] for p in points),
        "max_distance_from_start_m": max(math.hypot(p[1], p[2]) for p in points),
        "end_distance_from_start_m": math.hypot(points[-1][1], points[-1][2]),
        "altitude_lower_bound_m": min(-p[2] for p in controls),
        "altitude_upper_bound_m": max(-p[2] for p in controls),
        "cruise_speed_min_mps": cruise_min,
        "cruise_knot_speed_min_mps": min(cruise_speeds),
        "horizontal_speed_mps": max(cruise_speeds),
        "vertical_speed_mps": max(abs(p[7]) for p in points),
        "climb_speed_mps": max(-p[7] for p in points),
        "descent_speed_mps": max(p[7] for p in points),
        "acceleration_mps2": max(acceleration),
        "jerk_mps3": max(jerk),
        "yaw_rate_radps": max(yaw_rates),
    }


def validate(path):
    if set(path) != FIELDS or path.get("revision") != REVISION or path.get("kind") != "random":
        raise ValueError("Retired random path; regenerate with random-continuous-v3")
    if type(path["duration_s"]) not in (int, float) or path["duration_s"] not in (180, 480, 900):
        raise ValueError("Random flight duration must be 180/480/900 seconds")
    limits = path["limits"]
    if not isinstance(limits, dict) or limits.keys() != LIMITS.keys():
        raise ValueError("Exact random flight limits required")
    for key, value in limits.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Finite random flight limits required")
        if key == "horizontal_speed_mps":
            if not 1 <= value <= 3:
                raise ValueError("Random flight speed outside supported limits")
        elif value != LIMITS[key]:
            raise ValueError("Random flight limits cannot be overridden")
    points = path["program"]
    if not isinstance(points, list) or not 6 <= len(points) <= 120:
        raise ValueError("Finite continuous steering program required")
    for i, point in enumerate(points):
        if (
            not isinstance(point, list)
            or len(point) != 8
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
        ):
            raise ValueError("Eight finite steering state values required")
        if i:
            prev = points[i - 1]
            dt = point[0] - prev[0]
            if not limits["min_input_seconds"] <= dt <= limits["max_input_seconds"]:
                raise ValueError("Steering interval outside supported limits")
            expected = position_controls(prev, point)[-1]
            if math.dist(expected, point[1:4]) > 1e-7:
                raise ValueError("Position must integrate the continuous velocity commands")
            if abs(point[4] - prev[4]) > math.pi + 1e-8:
                raise ValueError("Yaw must take the shortest continuous turn")
    if (
        points[0] != [0, 0, 0, -3, 0, 0, 0, 0]
        or points[-1][0] != float(path["duration_s"])
        or points[-1][5:8] != [0, 0, 0]
    ):
        raise ValueError("Continuous takeoff start and final braking required")
    result = metrics(path)
    for key in (
        "horizontal_speed_mps",
        "vertical_speed_mps",
        "acceleration_mps2",
        "jerk_mps3",
        "yaw_rate_radps",
    ):
        if result[key] > limits[key] + 1e-8:
            raise ValueError("Random flight derivative limit exceeded: " + key)
    if result["altitude_lower_bound_m"] < 3 - 1e-8 or result["altitude_upper_bound_m"] > 30 + 1e-8:
        raise ValueError("Continuous altitude bounds exceeded")
    if (
        result["cruise_speed_min_mps"]
        < limits["min_cruise_speed_fraction"] * limits["horizontal_speed_mps"] - 1e-8
    ):
        raise ValueError("Cruise must not stop or linger at steering transitions")
    if min(result["climb_speed_mps"], result["descent_speed_mps"]) < 0.15:
        raise ValueError("Random flight requires both climb and descent inputs")
    if (
        min(result["north_span_m"], result["east_span_m"]) < 15
        or result["horizontal_speed_mps"] / result["cruise_knot_speed_min_mps"] < 1.6
    ):
        raise ValueError("Random flight lacks spatial or speed diversity")
    return path


def kinematics(t, path):
    """Analytically integrated input with consistent feed-forward derivatives."""
    points = path["program"]
    t = clamp(t, 0.0, path["duration_s"])
    index = min(max(bisect.bisect_right([p[0] for p in points], t) - 1, 0), len(points) - 2)
    a, b = points[index : index + 2]
    dt = b[0] - a[0]
    u = (t - a[0]) / dt
    s = u**3 * (10 + u * (-15 + 6 * u))
    integral = u**4 * (2.5 + u * (-3 + u))
    ds = 30 * u * u * (1 - u) ** 2 / dt
    dds = 60 * u * (1 - u) * (1 - 2 * u) / dt**2
    dv = [w - v for v, w in zip(a[5:8], b[5:8])]
    dyaw = b[4] - a[4]
    return (
        tuple(x + dt * (v * u + d * integral) for x, v, d in zip(a[1:4], a[5:8], dv))
        + (a[4] + dyaw * s,),
        tuple(v + d * s for v, d in zip(a[5:8], dv)) + (dyaw * ds,),
        tuple(d * ds for d in dv) + (dyaw * dds,),
    )
