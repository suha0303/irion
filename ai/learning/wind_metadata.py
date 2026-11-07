"""Describe audited operating metadata. Wind metadata never becomes a neural input."""

import math

BOUNDED = "bounded-latent-ou-c2-enu-v1"
RANDOM_BOUNDED = "random-continuous-v6-bounded-wind"
KINDS = {
    "hover",
    "altitude",
    "line",
    "ellipse",
    "figure8",
    "lissajous",
    "yaw",
    "mixed",
    "random",
}
ENVIRONMENT_BOUND = dict(calm=0.0, low=2.5, strong=5.0)
FIXED_RANDOM = {"random-continuous-v3", "random-continuous-v4-wind5"}


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def describe(scenario, *, wind_clock=None):
    """Classify explicit profile/path metadata, without reconstructing physics/traces.

    Source/hash/actual trace integrity and pair eligibility are separate gates.
    No fallback from bounded-profile legacy zero fields or common scenario name.
    """
    operation = scenario.get("operation_variation") or {}
    path = operation.get("path") or {}
    params = scenario.get("parameters", {})
    kind = path.get("kind")
    if kind not in KINDS or params.get("mission") != kind:
        raise ValueError("Explicit path kind and executed mission must agree")
    profile, model = (
        scenario.get("wind_profile"),
        scenario.get("wind_model", {}),
    )
    if profile is not None:
        environment = profile.get("environment")
        if (
            profile.get("revision") != BOUNDED
            or model.get("revision") != BOUNDED
            or model.get("profile") != profile
            or operation.get("wind_profile") != profile
            or scenario.get("wind_support_status") != "bounded_current"
            or environment not in ENVIRONMENT_BOUND
        ):
            raise ValueError(
                "Explicit matching bounded wind profile/model/environment required"
            )
        if (
            any(params.get(k) != 0 for k in ("wind_e", "wind_n", "gust_sigma"))
            or model.get("axes") != ["east", "north", "up"]
            or model.get("clock")
            != (wind_clock or "absolute_native_simulation_boot_seconds")
            or not finite(profile.get("direction_deg"))
            or not finite(profile.get("variation"))
            or not 0.5 <= profile["variation"] <= 1.5
        ):
            raise ValueError(
                "Bounded ENU clock/profile must not mix legacy wind parameters"
            )
        bounds = model.get("generator_speed_bounds_mps", [])
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(finite(x) for x in bounds)
        ):
            raise ValueError("Finite declared generator speed bounds required")
        lower, upper = bounds
        valid = (
            lower == upper == 0
            if environment == "calm"
            else 0 < lower <= upper <= 2.5
            if environment == "low"
            else 2.5 < lower <= upper <= 5
        )
        if (
            not valid
            or model.get("environment_speed_bound_mps")
            != ENVIRONMENT_BOUND[environment]
        ):
            raise ValueError(
                "Declared speed bounds contradict instantaneous wind environment"
            )
        if kind == "random" and path.get("revision") != RANDOM_BOUNDED:
            raise ValueError(
                "Bounded random requires v6; do not reinterpret old v3/v4/v5 paths"
            )
        return dict(
            wind_mode="bounded_profile",
            wind_revision=BOUNDED,
            environment=environment,
            generator_speed_bounds_mps=bounds,
            hard_environment_bound_mps=ENVIRONMENT_BOUND[environment],
            time_varying=environment != "calm",
            wind_clock=model["clock"],
            path_kind=kind,
            path_revision=path.get("revision"),
            metadata_only=True,
        )
    if (
        scenario.get("wind_support_status") != "fixed_wind_regression"
        or operation.get("wind_profile") is not None
        or model.get("revision") != "stationary-exact-ou-enu-v1"
        or params.get("gust_sigma") != 0
        or model.get("sigma_per_axis_mps") != 0
    ):
        raise ValueError(
            "No bounded profile: only explicitly marked fixed-wind regression accepted"
        )
    if kind == "random" and path.get("revision") not in FIXED_RANDOM:
        raise ValueError(
            "Fixed regression retains explicit v3/v4 path semantics"
        )
    return dict(
        wind_mode="legacy_fixed_regression",
        wind_revision=model["revision"],
        environment=None,
        time_varying=False,
        path_kind=kind,
        path_revision=path.get("revision"),
        metadata_only=True,
    )
