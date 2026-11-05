"""Wind-relative planning limits for the X500 engineering simulation revision."""

import math

from . import bounded_wind

REVISION = "x500-bounded-wind5-engineering-v3"
MEAN_WIND_MAX_MPS = 5.0
MODEL_AIRSPEED_MAX_MPS = 8.0
PLANNED_AIRSPEED_MAX_MPS = 7.0  # 1m/s reserve for actual tracking, not a changed quality gate.
MAX_VERTICAL_SPEED_MPS = 1.2
MAX_INPLANE_ADVANCE_RATIO = .1  # Engineering qualification constraint, not a manufacturer rating.


def random_speed_limit(spec, *, legacy=False, directional=False):
    if "wind_profile" in spec:
        budget = PLANNED_AIRSPEED_MAX_MPS - bounded_wind.speed_bound(spec["wind_profile"])
        return min(3., math.sqrt(max(0., budget*budget-MAX_VERTICAL_SPEED_MPS**2)))
    wind = math.hypot(spec["wind_e"], spec["wind_n"])
    gust = spec["gust_sigma"]
    if legacy:
        return min(3., 4.5-wind-3*gust)
    budget = PLANNED_AIRSPEED_MAX_MPS-3*gust
    horizontal = math.sqrt(max(0., budget*budget-MAX_VERTICAL_SPEED_MPS**2))
    cap = min(3., horizontal-wind)
    return max(1., cap) if directional else cap


def basic_speed_bounds(path, envelope_rate=math.pi/40):
    kind, radius, omega = path["kind"], path["radius"], path["omega"]
    positions = {"line": 1., "ellipse": math.hypot(1, 1.4),
                 "figure8": math.hypot(1, .5), "lissajous": math.sqrt(2)}
    derivatives = {"line": 1., "ellipse": math.hypot(1, .7),
                   "figure8": math.sqrt(2), "lissajous": 1.25}
    horizontal = (radius*(omega*derivatives[kind]+envelope_rate*positions[kind])
                  if kind in positions else 0.)
    vertical = envelope_rate*(path["height"]-3)
    if kind == "altitude":
        vertical += path["vertical_amplitude"]*(envelope_rate+omega)
    return horizontal, vertical


def commanded_airspeed_bound(spec):
    path = spec["path"]
    if "wind_profile" in spec:
        if path["kind"] == "random":
            ground = max(math.sqrt(sum(v*v for v in row[5:8])) for row in path["program"])
        elif path["kind"] == "mixed":
            pairs = [basic_speed_bounds(item["path"], 2.1875/25) for item in path["program"]]
            ground = math.hypot(max(p[0] for p in pairs), max(p[1] for p in pairs))
        else:
            ground = math.hypot(*basic_speed_bounds(path))
        return bounded_wind.speed_bound(spec["wind_profile"]) + ground
    if path["kind"] == "random":
        # 3sigma is a radial planning allowance, not a Gaussian hard bound.
        # Velocity is a convex blend of adjacent knot velocities, so relative
        # speed in constant wind cannot exceed the largest knot norm.
        vectors = path["program"]
        wind_n, wind_e = spec["wind_n"], spec["wind_e"]
        bound = max(math.sqrt((p[5]-wind_n)**2+(p[6]-wind_e)**2+p[7]**2) for p in vectors)
        return bound+3*spec["gust_sigma"]
    if path["kind"] == "mixed":
        bounds = [basic_speed_bounds(item["path"], 2.1875/25) for item in path["program"]]
        horizontal = max(x[0] for x in bounds)
        vertical = max(x[1] for x in bounds)
    else:
        horizontal, vertical = basic_speed_bounds(path)
    return math.hypot(math.hypot(spec["wind_n"], spec["wind_e"])+horizontal, vertical)+3*spec["gust_sigma"]


def directional_speed_limit(spec, course, vertical_speed):
    """Positive ray/ball intersection; bound the full mean-relative velocity."""
    budget = PLANNED_AIRSPEED_MAX_MPS - 3 * spec["gust_sigma"]
    projection = spec["wind_n"] * math.cos(course) + spec["wind_e"] * math.sin(course)
    squared = (projection * projection + budget * budget - vertical_speed * vertical_speed
               - spec["wind_n"]**2 - spec["wind_e"]**2)
    if squared <= 0:
        raise ValueError("No directional wind-relative planning margin")
    return projection + math.sqrt(squared)


def validate_wind(spec, *, revision=None):
    if "wind_profile" in spec:
        bounded_wind.validate(spec["wind_profile"])
        if any(spec[key] != 0 for key in ("wind_e", "wind_n", "gust_sigma")):
            raise ValueError("Bounded profile cannot mix legacy mean wind or Gaussian sigma")
        if revision not in (None, "random-continuous-v6-bounded-wind"):
            raise ValueError("Preserved v3/v4/v5 paths cannot acquire bounded wind semantics")
        return
    wind = math.hypot(spec["wind_n"], spec["wind_e"])
    sigma = spec["gust_sigma"]
    if not math.isfinite(wind) or wind > MEAN_WIND_MAX_MPS+1e-9:
        raise ValueError("Mean wind exceeds supported5m/s")
    if type(sigma) not in (int, float) or not math.isfinite(sigma) or not 0 <= sigma <= .4:
        raise ValueError("OU per-axis sigma outside supported0..0.4m/s")
    if revision == "random-continuous-v3" and wind > 2+1e-9:
        raise ValueError("Preserved v3 paths cannot acquire a new high-wind operating scope")
    if revision == "random-continuous-v4-wind5" and wind > 2+1e-9 and sigma != 0:
        raise ValueError("Preserved v4 high-wind paths require fixed wind")


def validate_wind_path(spec):
    path = spec["path"]
    validate_wind(spec, revision=path.get("revision"))
    wind = math.hypot(spec["wind_n"], spec["wind_e"])
    if path["kind"] == "random":
        legacy = path["revision"] == "random-continuous-v3"
        if legacy and wind > 2+1e-9:
            raise ValueError("Preserved v3 paths cannot acquire a new high-wind operating scope")
        cap = random_speed_limit(spec, legacy=legacy,
                                 directional=path["revision"] == "random-continuous-v5-ou")
        if path["limits"]["horizontal_speed_mps"] > cap+1e-8:
            raise ValueError("Random speed exceeds wind-adjusted airspeed margin")
    if commanded_airspeed_bound(spec) > PLANNED_AIRSPEED_MAX_MPS+1e-8:
        raise ValueError("Commanded path exceeds the7m/s wind-relative planning budget")
