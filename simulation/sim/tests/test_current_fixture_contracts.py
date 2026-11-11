"""Keep current production policy and historical fixture policy distinct."""

import json

import pytest
import test_joint_duration
import test_vehicle_runner
from capstone_sim import joint_phase

from test_support import contracts


@pytest.mark.parametrize(
    "limit,removed,accepted", [(40_000, 5, False), (3_000_000, 5, True), (3_000_000, 310, False)]
)
def test_declared_gap_and_complete_reference_coverage(
    tmp_path, monkeypatch, limit, removed, accepted
):
    original = joint_phase.POLICY.read_bytes()
    assert json.loads(original)["acceptance"]["native_consumption_max_gap_us"] == 3_000_000
    contracts.bind_gap_policy(joint_phase, monkeypatch, tmp_path, limit)
    case = test_joint_duration.Consumption180Test()
    case.setUp()
    try:
        del case.rows[10_001 : 10_001 + removed]
        case.duplicate()  # Repetition must not repair a gap or missing reference.
        case.write()
        if accepted:
            result = joint_phase.verify(case.run)
            assert result["max_consumption_gap_us"] == 60_000
            assert result["native_consumption_max_gap_us"] == limit
            assert result["reference_rows_seen"] == 1801
        else:
            with pytest.raises(ValueError, match="Incomplete actual"):
                joint_phase.verify(case.run)
    finally:
        case.doCleanups()
    assert (contracts.ROOT / "profiles/collection/joint-phase-policy.json").read_bytes() == original


def test_recorder_fake_keeps_current_sync_setting(tmp_path, monkeypatch):
    contracts.bind_recorder(test_vehicle_runner, monkeypatch)
    result = test_vehicle_runner.VehicleRunnerTest().run_mocked(tmp_path)
    assert result["recorder"].sync_interval_s == 0.0
    assert result["recorder"].closed and result["recorder"].released
    folder = tmp_path / "direct"
    (folder / "observations").mkdir(parents=True)
    (folder / "evaluation").mkdir()
    fake = test_vehicle_runner.RecorderFixture(folder, sync_interval_s=0.1)
    assert fake.sync_interval_s == 0.1


def test_historical_generator_is_pinned_and_current_generator_unchanged():
    before = joint_phase.POLICY
    historical = contracts.historical_generator()
    assert historical.REVISION == "joint-native-consumption-c2-v1"
    assert joint_phase.REVISION == "joint-native-consumption-c2-v2"
    assert joint_phase.POLICY == before


def test_historical_dependency_tampering_rejected(tmp_path, monkeypatch):
    for path in contracts.FROZEN.iterdir():
        if path.is_file():
            (tmp_path / path.name).write_bytes(path.read_bytes())
    with (tmp_path / "joint_phase.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# altered\n")
    monkeypatch.setattr(contracts, "FROZEN", tmp_path)
    with pytest.raises(ValueError, match="Historical audit dependency changed"):
        contracts.historical_generator()
