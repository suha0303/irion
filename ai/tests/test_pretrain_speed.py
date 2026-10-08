"""Offline matched-seed speed scope and information-gate rejection tests."""

import copy

import numpy as np
import pytest

from pretrain import common, speed_test


def samples():
    return dict(
        features=["pack_voltage", "motor_request_front_right"],
        config=dict(history=100),
        aligned=np.zeros((901, 2)).tolist(),
        valid_airborne_fraction=0.98,
        windows=80,
        topic_quality=dict(
            battery_status=dict(count=1000, first=0, last=10_000_000, max_gap_us=100000)
        ),
    )


def test_speed_gate_rejects_information_loss_and_changed_contract():
    base = samples()
    repeat = copy.deepcopy(base)
    candidate = copy.deepcopy(base)
    assert speed_test.compare(base, repeat, candidate)["passed"]
    assert speed_test.compare(base, repeat, candidate)["bitwise_equal"]
    candidate["topic_quality"]["battery_status"]["count"] = 800
    assert not speed_test.compare(base, repeat, candidate)["passed"]
    candidate = copy.deepcopy(base)
    candidate["aligned"][30][0] = 10
    assert not speed_test.compare(base, repeat, candidate)["passed"]
    candidate = copy.deepcopy(base)
    candidate["config"]["history"] = 50
    with pytest.raises(ValueError):
        speed_test.compare(base, repeat, candidate)


@pytest.mark.parametrize("protocol", ["information-v1", "information-v2"])
def test_matched_seed_exception_is_test_only_and_requires_prior_owned_completion(
    tmp_path, protocol
):
    collector = common.collector()
    config = tmp_path / "config.json"
    common.write(
        config,
        dict(mode="test", seed_base=100, speed_test=dict(factors=[2, 2, 4, 8], protocol=protocol)),
    )
    plan = dict(
        pretrain_memory_override=dict(config=dict(path=str(config), sha256=common.sha(config))),
        pretrain_cli_approval=str(config),
        phase="benchmark",
        concurrency=1,
        unit_attempts_max=1,
        speed_comparison=dict(protocol=protocol, index=0, factor=2),
    )
    case = dict(role="benchmark", scenario="hover_normal", seed=2100)
    assert collector.speed_comparison_scope(plan, [case]) == set()
    bad = copy.deepcopy(plan)
    bad["phase"] = "collection"
    with pytest.raises(collector.Stop):
        collector.speed_comparison_scope(bad, [case])
    bad = copy.deepcopy(plan)
    bad["unit_attempts_max"] = 3
    with pytest.raises(collector.Stop):
        collector.speed_comparison_scope(bad, [case])
    plan["speed_comparison"]["index"] = 1
    with pytest.raises(FileNotFoundError):
        collector.speed_comparison_scope(plan, [case])
    common.write(tmp_path / "speed-01/plan.json", dict(pretrain_cli_approval=str(config)))
    common.write(
        tmp_path / "speed-01/execution/ledger.json",
        dict(
            status="COLLECTED_PENDING_QUALITY_STOPPED", cases=dict(one=dict(run_id="owned-first"))
        ),
    )
    assert collector.speed_comparison_scope(plan, [case]) == {"owned-first"}
