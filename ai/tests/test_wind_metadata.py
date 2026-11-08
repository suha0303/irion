"""Metadata-only fixtures; no flight, neural forward, scaler fit or
evaluation.
"""

import copy

import pytest

from learning.wind_metadata import BOUNDED, KINDS, describe


def fixture(environment, kind):
    bounds = dict(calm=[0.0, 0.0], low=[0.25, 2.25], strong=[2.75, 4.75])[
        environment
    ]
    profile = dict(
        revision=BOUNDED,
        environment=environment,
        direction_deg=0.0,
        variation=1.0,
    )
    return dict(
        name="hover_normal",
        wind_profile=profile,
        wind_support_status="bounded_current",
        parameters=dict(wind_e=0.0, wind_n=0.0, gust_sigma=0.0, mission=kind),
        operation_variation=dict(
            wind_profile=copy.deepcopy(profile),
            path=dict(
                kind=kind,
                revision="random-continuous-v6-bounded-wind"
                if kind == "random"
                else None,
            ),
        ),
        wind_model=dict(
            revision=BOUNDED,
            profile=copy.deepcopy(profile),
            axes=["east", "north", "up"],
            clock="absolute_native_simulation_boot_seconds",
            generator_speed_bounds_mps=bounds,
            environment_speed_bound_mps=dict(calm=0.0, low=2.5, strong=5.0)[
                environment
            ],
        ),
    )


@pytest.mark.parametrize("environment", ["calm", "low", "strong"])
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_all27_profile_and_actual_kind_despite_common_scenario_and_legacy_zeros(
    environment, kind
):
    result = describe(fixture(environment, kind))
    assert result["environment"] == environment and result["path_kind"] == kind
    assert result["time_varying"] is (environment != "calm")
    assert result["metadata_only"] is True


@pytest.mark.parametrize(
    "revision",
    [
        "random-continuous-v3",
        "random-continuous-v4-wind5",
        "random-continuous-v5-ou",
    ],
)
def test_bounded_profile_cannot_relabel_older_random_paths(revision):
    scenario = fixture("strong", "random")
    scenario["operation_variation"]["path"]["revision"] = revision
    with pytest.raises(ValueError, match="v6"):
        describe(scenario)


@pytest.mark.parametrize(
    "damage", ["profile", "bounds", "legacy_mix", "mission", "missing_profile"]
)
def test_ambiguous_metadata_is_rejected(damage):
    scenario = fixture("strong", "ellipse")
    if damage == "profile":
        scenario["wind_model"]["profile"]["environment"] = "calm"
    elif damage == "bounds":
        scenario["wind_model"]["generator_speed_bounds_mps"][1] = 5.1
    elif damage == "legacy_mix":
        scenario["parameters"]["gust_sigma"] = 0.4
    elif damage == "mission":
        scenario["parameters"]["mission"] = "hover"
    elif damage == "missing_profile":
        scenario.pop("wind_profile")
    with pytest.raises(ValueError):
        describe(scenario)


@pytest.mark.parametrize(
    "revision", ["random-continuous-v3", "random-continuous-v4-wind5"]
)
def test_fixed_regression_is_kept_separate(revision):
    scenario = fixture("strong", "random")
    scenario.update(
        wind_profile=None,
        wind_support_status="fixed_wind_regression",
        wind_model=dict(
            revision="stationary-exact-ou-enu-v1", sigma_per_axis_mps=0.0
        ),
    )
    scenario["operation_variation"]["path"]["revision"] = revision
    scenario["operation_variation"].pop("wind_profile")
    result = describe(scenario)
    assert (
        result["environment"] is None
        and result["wind_mode"] == "legacy_fixed_regression"
    )
