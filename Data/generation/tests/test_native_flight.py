"""Exercise the real Sim interpreter boundary without a flight or dependency install."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pretrain import common, native_flight, pilot45


def test_actual_sim_launcher_imports_runtime_while_data_lacks_pymavlink():
    assert importlib.util.find_spec("pymavlink") is None
    result = native_flight.environment_check()
    assert result["status"] == "PASSED_NO_FLIGHT"
    assert result["probe"]["run_flight_imported"] is True
    assert result["probe"]["flight_started"] is False
    assert Path(result["probe"]["executable"]) == common.ROOT / "simulation/.venv/bin/python"


def test_native_argv_keeps_parent_operation_and_health_selection():
    case = pilot45.build_cases(1)[-1]
    args = native_flight.command(case, "failed-raw")
    assert args[:2] == [str(common.ROOT / "simulation/capstone-sim"), "run"]
    assert "_case" not in args and str(common.PYTHON) not in args
    assert args[args.index("--parent-attempt-id") + 1] == "failed-raw"
    assert json.loads(args[args.index("--operation-variation") + 1]) == case["operation_variation"]
    assert args[args.index("--battery-stage") + 1] == "rpt_16"
    assert args[args.index("--fault-after-mission-s") + 1] == str(case["fault_after_mission_s"])
    with pytest.raises(ValueError):
        native_flight.command(case, paired_options={"invented_kwarg": True})


def test_paired_cli_resolves_current_case_id_and_keeps_only_pair_members_equal():
    cases = pilot45.build_cases(99)
    options = [
        "--phase-policy",
        "joint-native-consumption-c2-v1",
        "--paired-case-id",
        "{paired_case_id}",
    ]

    def selected(case):
        argv = native_flight.command(case, paired_options=options)
        assert not any("{" in token or "}" in token for token in argv[-4:])
        return argv[argv.index("--paired-case-id") + 1]

    assert selected(cases[0]) == cases[0]["paired_case_id"]
    assert selected(cases[0]) != selected(cases[1])
    assert selected(cases[-2]) == selected(cases[-1])


def test_child_collects_json_after_real_other_interpreter_exits(monkeypatch, tmp_path):
    # A real process using Sim Python emits a result and separate stderr. Native
    # flight itself is a fixture so this test never launches PX4/Gazebo.
    raw = tmp_path / "simulation/runs/fixture"
    raw.mkdir(parents=True)
    task = tmp_path / "task"
    request = task / "case/flight-02.request.json"
    receipt = task / "case/flight-02.receipt.json"
    case = pilot45.build_cases(1)[0]
    common.write(task / "config.json", dict(source_sha256={}))
    common.write(task / "operator/state.json", {})
    common.write(request, dict(action="flight", task=str(task), case=case, receipt=str(receipt)))
    program = (
        "import json,sys;print('fixture stderr',file=sys.stderr);print(json.dumps("
        + repr(dict(run=str(raw), status="completed"))
        + "))"
    )
    monkeypatch.setattr(
        native_flight,
        "command",
        lambda *args: [str(common.ROOT / "simulation/.venv/bin/python"), "-B", "-c", program],
    )
    monkeypatch.setattr(native_flight, "ROOT", tmp_path)
    monkeypatch.setattr(pilot45, "ROOT", tmp_path)
    monkeypatch.setattr(
        pilot45,
        "resources",
        lambda *args: dict(
            available_bytes=2_000_000_000, disk_free_bytes=100_000_000_000, owned_tree_rss_bytes=1
        ),
    )
    progress = SimpleNamespace(matches={}, sample=lambda: {}, terminal_evidence=lambda: {})
    monkeypatch.setattr(
        pilot45,
        "collector",
        lambda: SimpleNamespace(MissionProgress=lambda *args, **kwargs: progress),
    )
    log = task / "case/flight-02.log"
    result = pilot45.child(task, request, log)
    assert result["exit_code"] == 0
    assert common.read(receipt) == dict(run=str(raw), status="completed")
    assert log.read_text().strip() == "fixture stderr"
    assert json.loads(Path(result["stdout_path"]).read_text())["run"] == str(raw)


def test_invalid_stdout_is_preserved_instead_of_fabricating_result(tmp_path):
    p = tmp_path / "stdout.json"
    p.write_text('diagnostic\n{"status":"completed"}')
    with pytest.raises(ValueError):
        native_flight.read_result(p)
    assert p.read_text().startswith("diagnostic")
