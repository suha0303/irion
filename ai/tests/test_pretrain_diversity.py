"""Offline bounds, reproducibility and immutable executed-configuration checks."""

import copy
import sys

import numpy as np
import pytest

from pretrain import collection, common, composition

sys.path.insert(0, str(common.ROOT / "simulation/sim/src"))
from capstone_sim.normal_variation import sample, target, validate


def test_diverse_membership_and_reproducible_healthy_parameters():
    first = collection.cases(500000, "diverse104")["cases"]
    assert first == collection.cases(500000, "diverse104")["cases"]
    assert len(first) == 104
    train = [c for c in first if c["role"] == "normal_train"]
    for scenario in {c["scenario"] for c in train}:
        specs = [c["normal_variation"] for c in train if c["scenario"] == scenario]
        assert len({s["soc"] for s in specs}) == 7
        assert len({s["path"]["radius"] for s in specs}) == 7
        assert all(s["payload"] > 0 and s["wind_e"] != 0 for s in specs)
    assert not any("normal_variation" in c for c in first if c["mission_seconds"] == 480)
    damaged = copy.deepcopy(first)
    next(c for c in damaged if "normal_variation" in c)["normal_variation"]["payload"] += 0.0001
    with pytest.raises(ValueError):
        composition.validate_cases(damaged, "diverse104")


@pytest.mark.parametrize(
    "kind", ["hover", "line", "ellipse", "figure8", "lissajous", "altitude", "yaw"]
)
def test_paths_are_smooth_bounded_and_have_no_start_jump(kind):
    for seed in range(12):
        spec = sample("waypoint_normal", seed)
        spec["path"]["kind"] = kind
        validate(spec)
        times = np.arange(0, 900, 0.1)
        xyz = np.asarray([target(float(t), spec) for t in times])
        assert np.array_equal(xyz[0], [0, 0, -3, 0])
        assert np.max(np.linalg.norm(np.diff(xyz[:, :3], axis=0) / 0.1, axis=1)) < 0.65
        assert np.max(np.linalg.norm(np.diff(xyz[:, :3], n=2, axis=0) / 0.01, axis=1)) < 0.2
        assert np.min(xyz[:, 2]) >= -4.6 and np.max(xyz[:, 2]) <= -2.4


@pytest.mark.parametrize(
    "key,value", [("r0", 0.03), ("fault", "r0"), ("capacity", 8), ("soc", 0.2), ("payload", 0.3)]
)
def test_fault_health_and_out_of_range_overrides_rejected(key, value):
    spec = sample("hover_normal", 17)
    spec[key] = value
    with pytest.raises(ValueError):
        validate(spec)
