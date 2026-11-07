"""Short-cycle parameters preserve480s defaults and never launch a native flight."""

import copy
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from pretrain import common, native_flight, pilot45, pilot_parallel


@pytest.mark.local_evidence
def test_default480_case_values_are_identical_to_reviewed_previous_source():
    previous = (
        common.ROOT
        / "docs/data/reports/short-cycle-20261107/attempt-01/before/Data/generation/pretrain/pilot45.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pretrain._before_short", previous
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    assert pilot45.build_cases(12345) == old.build_cases(12345)


def test180_has_full45_27cells9pairs_and_safe_original_endpoints():
    rows = pilot45.build_cases(12345, duration=180, case_prefix="s180")
    pilot45.validate_cases(rows, duration=180)
    assert sum(r["duration"] for r in rows) == 8100
    assert all(r["case_id"].startswith("s180-") for r in rows)
    assert Counter(r["category"] for r in rows) == dict(
        standalone_normal=27, paired_normal=9, anomaly=9
    )
    assert Counter(r["role"] for r in rows) == dict(
        normal_test=36, anomaly_test=9
    )
    dynamic = [r for r in rows if r["anomaly_timing"] in {"step", "ramp"}]
    assert {r["fault_after_mission_s"] for r in dynamic} <= {45, 60, 75}
    assert all(r["fault_after_mission_s"] + 20 <= 95 for r in dynamic)
    assert all(
        r["attempts_max"] == 3
        and r["cohort"] == "short_cycle_development_reference"
        for r in rows
    )
    assert all(
        r["battery_stage"] == "rpt_16"
        for r in rows
        if r["anomaly_family"] == "combined"
    )


@pytest.mark.parametrize("duration", [1, 100, 179, 181, 900, 180.0, True])
def test_only_explicit180_and480_are_allowed(duration):
    with pytest.raises(ValueError):
        pilot45.build_cases(1, duration=duration)


def test_short_manifest_requires_exact_scope_and_new_duration(monkeypatch):
    monkeypatch.setattr(pilot45, "short_authorization", lambda: {})
    rows = pilot45.build_cases(12, duration=180, case_prefix="s180")
    value = dict(
        short_cycle=True, mission_seconds=180, seed_base=12, cases=rows
    )
    assert pilot45.validate_manifest(value) == rows
    changed = copy.deepcopy(value)
    changed["cases"][0]["duration"] = 480
    with pytest.raises(ValueError):
        pilot45.validate_manifest(changed)
    with pytest.raises(ValueError):
        pilot45.validate_cases(rows)


def test_short_argv_carries180_actual_instance_and_timing_without_quality_override():
    case = pilot45.build_cases(1, duration=180, case_prefix="s180")[-1]
    argv = native_flight.command(
        case,
        "previous",
        [
            "--phase-policy",
            "joint-native-consumption-c2-v2",
            "--paired-case-id",
            "{paired_case_id}",
        ],
        15,
        True,
    )
    assert argv[argv.index("--duration") + 1] == "180"
    assert argv[argv.index("--real-time-factor") + 1] == "2"
    assert argv[argv.index("--instance") + 1] == "15"
    assert argv[argv.index("--fault-after-mission-s") + 1] in {"45", "60", "75"}
    assert "--timing" in argv
    assert "--capture-interval-ms" not in argv
    assert "--timing" not in native_flight.command(case)


def test_progress_sampler_receives_requested_duration(monkeypatch, tmp_path):
    # Construction boundary, before Popen: no child/native execution.
    case = pilot45.build_cases(1, duration=180, case_prefix="s180")[0]
    task = tmp_path / "task"
    common.write(task / "config.json", {})
    request = task / "flight.request.json"
    common.write(request, dict(action="flight", case=case, instance=15))
    (tmp_path / "simulation/runs").mkdir(parents=True)
    observed = []

    class Done(Exception):
        pass

    def progress(root, baseline, selected, lane, duration, **kwargs):
        observed.append((lane["instance"], duration))
        raise Done

    from types import SimpleNamespace

    monkeypatch.setattr(pilot45, "ROOT", tmp_path)
    monkeypatch.setattr(
        pilot45, "collector", lambda: SimpleNamespace(MissionProgress=progress)
    )
    with pytest.raises(Done):
        pilot45.child(task, request, tmp_path / "never.log")
    assert observed == [(15, 180)]


def test_quality_requested_time_uses45_actual_case_durations(tmp_path):
    common.write(
        tmp_path / "manifest.json",
        dict(cases=pilot45.build_cases(1, duration=180, case_prefix="s180")),
    )
    common.write(tmp_path / "config.json", {})
    q = pilot45.quality_snapshot(
        tmp_path, dict(cases={}, attempts=[]), dict(status="NOT_STARTED")
    )
    assert q["requested_mission_seconds"] == 8100 and q["planned_cases"] == 45
    assert q["completed_raw"] == q["prepared_and_audited"] == 0


def test_short_parallel_scope_never_uses480_continuation(monkeypatch):
    monkeypatch.setattr(pilot45, "short_authorization", lambda: {})
    cfg = dict(
        short_cycle=True,
        mission_seconds=180,
        flight_workers=16,
        requested_real_time_factor=2,
        cpu_workers_total_max=4,
        planned_cases=45,
        training_authorized=False,
        fit_authorized=False,
    )
    pilot_parallel.validate_config(cfg)
    with pytest.raises(pilot45.Blocked):
        pilot_parallel.validate_config(
            dict(cfg, continuation=dict(previous_task="old480"))
        )


@pytest.mark.parametrize(
    "change",
    [
        None,
        "retry",
        "inherited",
        "missing_ai",
        "failure",
        "missing_case",
        "missing_final",
    ],
)
def test_clean_cycle_input_gate_requires_all45_first_attempts_and_two_readers(
    change,
):
    q = dict(
        status="PILOT45_QUALITY_PASSED_STOPPED",
        planned_cases=45,
        completed_raw=45,
        prepared_and_audited=45,
        online_available=45,
        run_available=45,
        ai_case_checks_passed=45,
        short_ai_final_passed=True,
    )
    ledger = dict(
        attempts=[dict(state="COMPLETED", number=1) for _ in range(45)],
        cases={str(i): {} for i in range(45)},
    )
    if change == "retry":
        ledger["attempts"].append(dict(state="FAILED_PRESERVED", number=2))
    elif change == "inherited":
        ledger["cases"]["0"]["inherited_from"] = "old-cycle"
    elif change == "missing_ai":
        q["ai_case_checks_passed"] = 44
    elif change == "failure":
        q["status"] = "BLOCKED_STOPPED"
    elif change == "missing_case":
        q["completed_raw"] = 44
    elif change == "missing_final":
        q["short_ai_final_passed"] = False
    assert pilot_parallel.clean_short_inputs(q, ledger) is (change is None)


@pytest.mark.parametrize("phase", ["prerequisites", "case", "final"])
def test_short_bridge_uses_public_ai_connector_for_each_phase(
    monkeypatch, tmp_path, phase
):
    from learning.history19 import short_cycle

    from pretrain import pilot_bridge

    task = tmp_path / "task"
    common.write(
        task / "config.json", dict(short_cycle=True, task_id=short_cycle.TASK)
    )
    for member in (
        "expected-producer.json",
        "consumer-capsule.json",
        "ai-manifest.json",
    ):
        common.write(task / member, {})
    source = tmp_path / "simulation/config/producer-sources.json"
    runtime = tmp_path / "simulation/artifacts/runtime-manifest.json"
    common.write(source, {})
    common.write(runtime, {})
    spec = dict(paths={}, origin_relative_path="evaluation/mission-origin.json")
    for key in ("native_policy", "paired_policy"):
        path = tmp_path / (key + ".json")
        common.write(path, {})
        spec["paths"][key] = str(path)
    entries = {}
    for i in range(45):
        actual = task / "prior-checks" / f"{i}.json"
        wrapper = task / "prior-checks" / f"{i}.wrapper.json"
        common.write(actual, dict(fixture_only=True))
        common.write(
            wrapper, dict(receipt=str(actual), sha256=common.sha(actual))
        )
        entries[str(i)] = dict(ai_receipt=str(wrapper))
    raw, output = task / "raw", task / "output"
    case = dict(case_id="0")
    entries["0"].update(run=str(raw), output=str(output), case=case)
    common.write(task / "ledger.json", dict(cases=entries))
    for member in (
        "run.json",
        "integrity.json",
        "evaluation/execution.json",
        "evaluation/mission-origin.json",
        "evaluation/scenario.json",
        "evaluation/battery-selection.json",
        "evaluation/propulsion-selection.json",
    ):
        common.write(raw / member, {})
    for member in ("frames.jsonl", "online.json", "run.json"):
        common.write(output / member, {})
    monkeypatch.setattr(pilot_bridge, "ROOT", tmp_path)

    def snapshot(_task, directory):
        common.write(directory / "registry.json", {})
        common.write(directory / "ledger.json", {})

    def static_gate(request, receipt):
        # Only public request construction is under test, never actual flight admission.
        common.write(receipt, dict(actual_data_verified=True, phase=phase))
        return 0

    monkeypatch.setattr(pilot_bridge, "snapshot", snapshot)
    monkeypatch.setattr(short_cycle, "main", static_gate)
    result = pilot_bridge.check(
        task, spec, phase=phase, case_id="0" if phase == "case" else None
    )
    request = common.read(Path(result["receipt"]).with_name("request.json"))
    assert request["schema"] == "drone-ai.short-cycle.check/1"
    assert request["phase"] == phase
    assert request["purpose"] == "short_cycle_development_reference"
    assert "native_completion" not in request["refs"]
    assert "development_receipt" not in request["refs"]
    if phase == "case":
        assert set(request["case_origin"]) == {"run", "execution", "origin"}
        assert request["case_id"] == "task:0"
    if phase == "final":
        assert len(request["case_receipts"]) == 45
        assert all("wrapper" not in r["path"] for r in request["case_receipts"])


def test_short_global_registry_excludes_old_raw_without_blocking_own_new_cases(
    monkeypatch, tmp_path
):
    for name in (
        "wind5-support-20261107/development-reference-runs.json",
        "wind5-gust-support-20261107/development-reference-runs.json",
        "wind5-gust-support-20261107/superseded-mean-ou-runs.json",
    ):
        common.write(tmp_path / "docs/data/reports" / name, dict(records=[]))
    monkeypatch.setattr(pilot45, "ROOT", tmp_path)
    old = dict(records=[dict(run_id="old480", cohort="old_development")])
    value = pilot45.global_protection(
        old,
        [dict(case_id="s180-new")],
        "short-new",
        cohort=pilot45.SHORT_COHORT,
    )
    prior = next(
        e for e in value["entries"] if e["identities"] == ["run:old480"]
    )
    fresh = next(
        e
        for e in value["entries"]
        if e["identities"] == ["case:short-new:s180-new"]
    )
    assert "short_cycle_input_check" in prior["forbidden_uses"]
    assert "short_cycle_input_check" not in fresh["forbidden_uses"]
    assert {"fit", "calibration", "final_test"} <= set(fresh["forbidden_uses"])


@pytest.mark.local_evidence
def test_short_draft_has_its_own_parallel_namespace_without_old_scope_authority(
    monkeypatch, tmp_path
):
    from test_pilot45 import init_fixture

    init_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(pilot45, "short_authorization", lambda: {})
    monkeypatch.setattr(
        pilot45,
        "global_protection",
        lambda *a, **kw: dict(entries=[], purpose=kw["cohort"]),
    )
    task = pilot45.init("short-new-45", 90210, short_cycle=True)
    cfg = common.read(task / "config.json")
    assert cfg["lineage_namespace"] == task.name
    assert cfg["mission_seconds"] == 180
    assert (cfg["flight_workers"], cfg["cpu_workers_total_max"]) == (16, 4)
    assert cfg["authorization"]["path"] == str(pilot45.SHORT_AUTH)
    assert cfg["sealed"] is False
    assert common.read(task / "ledger.json")["attempts"] == []
    assert not (task / "operator/service.json").exists()


@pytest.mark.parametrize(
    "change", [None, "old6", "v1", "fixture", "480", "hash", "flight_argv"]
)
def test_short_handoff_separates_static180_binding_from_old_ready(
    monkeypatch, tmp_path, change
):
    from types import SimpleNamespace

    from learning import admission19

    from pretrain import ai19, pilot_bridge

    producer = {k: "a" * 64 for k in admission19.PRODUCER_FIELDS}
    producer.update(
        battery_policy=admission19.BATTERY_POLICY,
        battery_revision="cmu-vah01-baseline/1",
        propulsion_revision=admission19.PROPULSION_REVISION,
    )
    native = dict(
        revision="joint-native-consumption-c2-v2",
        supported_duration_us=[180000000, 480000000],
    )
    native_path, paired_path = (
        tmp_path / "native.json",
        tmp_path / "paired.json",
    )
    common.write(native_path, native)
    policy = dict(
        approved_for_short_cycle=True,
        approved_for_pilot=False,
        scope_task="SHORT_CYCLE_20261107",
        purpose=pilot45.SHORT_COHORT,
        duration_profile="short180",
        fixture_only=False,
        producer=producer,
        native_adapter=native["revision"],
        native_policy_sha256=common.sha(native_path),
        mission_source="evaluation/joint-origin.json",
    )
    if change == "v1":
        policy["native_adapter"] = "joint-native-consumption-c2-v1"
    if change == "fixture":
        policy["fixture_only"] = True
    if change == "480":
        policy["duration_profile"] = "standard480"
    common.write(paired_path, policy)
    code_path = tmp_path / "ai/fixture.py"
    common.write(code_path, {})
    freeze = tmp_path / "freeze.json"
    common.write(
        freeze,
        dict(
            code_frozen=True,
            task_id="SHORT_CYCLE_20261107",
            files={"ai/fixture.py": common.sha(code_path)},
        ),
    )
    sim_contract = tmp_path / "sim-contract.json"
    common.write(sim_contract, {})
    value = dict(
        code_frozen=True,
        duration_profile="short180",
        mission_seconds=180,
        expected_cases=45,
        expected_pairs=9,
        limits=dict(
            flight_workers=16,
            RTF=2,
            aggregate_nonflight_workers=4,
            max_attempts_per_logical_case=3,
        ),
        no_extra_development6=True,
        no_LR108_or_main_training_link=True,
        source_freeze=dict(path=str(freeze), sha256=common.sha(freeze)),
        simulation_evidence=dict(
            contract=dict(
                path=str(sim_contract), sha256=common.sha(sim_contract)
            )
        ),
        schema="drone-ai.short-cycle.data-handoff/1",
        task_id="SHORT_CYCLE_20261107",
        status="READY",
        producer=producer,
        paths=dict(
            native_policy=str(native_path), paired_policy=str(paired_path)
        ),
        sha256=dict(
            native_policy=common.sha(native_path),
            paired_policy=common.sha(paired_path),
        ),
        origin_relative_path=policy["mission_source"],
        paired_cli_options=[
            "--phase-policy",
            native["revision"],
            "--paired-case-id",
            "{paired_case_id}",
        ],
        pair_verifier_argv=[
            str(tmp_path / "simulation/capstone-sim"),
            "pair-check",
            "--normal-run",
            "{normal_run}",
            "--anomaly-run",
            "{anomaly_run}",
            "--output",
            "{output}",
        ],
    )
    if change == "old6":
        value["paths"]["native_completion"] = "historical-READY.json"
    if change == "hash":
        value["sha256"]["paired_policy"] = "b" * 64
    if change == "flight_argv":
        value["pair_verifier_argv"][1] = "run"
    path = tmp_path / "handoff.json"
    common.write(path, value)
    monkeypatch.setattr(ai19, "producer_binding", lambda: producer)
    monkeypatch.setattr(pilot_bridge, "ROOT", tmp_path)
    monkeypatch.setattr(
        pilot_bridge.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout="--phase-policy --paired-case-id"
        ),
    )
    if change is None:
        assert pilot_bridge.dependencies(path) == value
    else:
        with pytest.raises(ValueError):
            pilot_bridge.dependencies(path)


@pytest.mark.local_evidence
def test_bounded_short_repair_keeps_attempt1_false_and_admits_exact_parent_as_attempt2(
    monkeypatch, tmp_path
):
    from test_pilot_parallel import fixture

    task, _ = fixture(monkeypatch, tmp_path)
    cases = pilot45.build_cases(98765, duration=180, case_prefix="s180")
    cfg = common.read(task / "config.json")
    cfg.pop("continuation")
    case = cases[0]
    proof = tmp_path / "repair-acceptance.json"
    common.write(proof, dict(scope="synthetic-boundary-only"))
    previous = dict(
        number=1,
        state="FAILED_PRESERVED",
        run_id="failed-native-parent",
        retry_allowed=False,
        source_task="previous",
        parent_attempt_id=None,
    )
    cfg.update(
        short_cycle=True,
        mission_seconds=180,
        stop_on_first_physical_failure=True,
        short_repair=dict(
            previous_task="previous",
            files={str(proof): common.sha(proof)},
            failures={
                case["case_id"]: dict(
                    run_id=previous["run_id"], attempt=1, seed=case["seed"]
                )
            },
        ),
    )
    entries = {c["case_id"]: dict(case=c, attempts=[]) for c in cases}
    entries[case["case_id"]]["attempts"] = [copy.deepcopy(previous)]
    common.write(task / "config.json", cfg)
    common.write(task / "manifest.json", dict(cases=cases))
    common.write(
        task / "ledger.json",
        dict(cases=entries, attempts=[dict(previous, case_id=case["case_id"])]),
    )
    coordinator = pilot_parallel.Coordinator(task)
    job = coordinator.begin_flight(case, 0)
    request = common.read(job["request"])
    imported, current = coordinator.ledger["cases"][case["case_id"]]["attempts"]
    assert imported == previous and imported["retry_allowed"] is False
    assert (
        current["number"] == 2
        and current["parent_attempt_id"] == previous["run_id"]
    )
    assert current["execution_task"] == str(task)
    assert request["parent_attempt_id"] == previous["run_id"]
    assert not Path(
        request["receipt"]
    ).exists()  # admission boundary only, no native child
    current.update(
        number=2,
        state="FAILED_PRESERVED",
        retry_allowed=False,
        run_id="new-failure",
    )
    with pytest.raises(pilot45.Blocked):
        coordinator.begin_flight(
            case, 0
        )  # repair never permits an unrelated new failure
    entries = coordinator.ledger["cases"][case["case_id"]]["attempts"]
    entries.append(dict(current, number=3, retry_allowed=True))
    with pytest.raises(pilot45.Blocked):
        coordinator.begin_flight(
            case, 0
        )  # historical attempts count toward max3
    assert len(entries) == 3


@pytest.mark.parametrize(
    "change", [None, "retry", "failed", "unknown", "inherited", "missing_case"]
)
def test_current_cycle_and_cumulative_failure_denominators_stay_separate(
    change,
):
    task = Path("current")
    current = [
        dict(
            case_id=str(i),
            execution_task=str(task),
            state="COMPLETED",
            number=2 if i < 16 else 1,
        )
        for i in range(45)
    ]
    history = [
        dict(
            case_id=str(i),
            source_task="previous",
            state="FAILED_PRESERVED",
            number=1,
            retry_allowed=False,
        )
        for i in range(16)
    ]
    ledger = dict(
        cases={str(i): {} for i in range(45)}, attempts=history + current
    )
    if change == "retry":
        ledger["attempts"].append(
            dict(current[0], state="FAILED_PRESERVED", number=3)
        )
    elif change == "failed":
        current[0]["state"] = "FAILED_PRESERVED"
    elif change == "unknown":
        current[0]["state"] = "RUNNING"
    elif change == "missing_case":
        current[0]["case_id"] = "other"
    q = dict(
        status="PILOT45_QUALITY_PASSED_STOPPED",
        short_ai_final_passed=True,
        planned_cases=45,
        completed_raw=45,
        prepared_and_audited=45,
        online_available=45,
        run_available=45,
        ai_case_checks_passed=45,
        inherited_successes=int(change == "inherited"),
    )
    q.update(pilot_parallel.current_execution_counts(task, ledger))
    assert pilot_parallel.clean_short_inputs(q, ledger) is False
    assert q["historical_failed_attempts"] == 16
    assert len(ledger["attempts"]) == (62 if change == "retry" else 61)
    assert pilot_parallel.current_execution_inputs_passed(q) is (change is None)


@pytest.mark.parametrize("stop_first", [False, True])
def test_explicit_stop_first_preserves_failure_instead_of_spending_next_attempt(
    monkeypatch, tmp_path, stop_first
):
    from test_pilot_parallel import fixture

    task, cases = fixture(monkeypatch, tmp_path)
    cfg = common.read(task / "config.json")
    cfg["stop_on_first_physical_failure"] = stop_first
    common.write(task / "config.json", cfg)
    c = pilot_parallel.Coordinator(task)
    case = cases[3]
    job = c.begin_flight(case, 0)
    raw = tmp_path / "simulation/runs/failed-clock-parent"
    common.write(
        raw / "run.json",
        dict(status="failed", instance=0, parent_attempt_id=None),
    )
    common.write(
        raw / "evaluation/scenario.json",
        dict(
            name=case["scenario"],
            seed=case["seed"],
            operation_variation=case["operation_variation"],
        ),
    )
    reason = "recording_stalled: PX4 clock advanced but raw capture did not"
    common.write(
        common.read(job["request"])["receipt"],
        dict(status="failed", run=str(raw), reason=reason),
    )
    if stop_first:
        with pytest.raises(
            pilot45.Blocked, match="automatic retry not allowed"
        ):
            c.finish_flight(job, RuntimeError(reason))
    else:
        assert c.finish_flight(job, RuntimeError(reason)) is False
    a = c.ledger["cases"][case["case_id"]]["attempts"]
    assert len(a) == 1 and a[0]["state"] == "FAILED_PRESERVED"
    assert a[0]["retry_allowed"] is (not stop_first)
    assert a[0]["run_id"] == raw.name
    assert (
        len(c.ledger["attempts"]) == 4
    )  # three fixture successes + this one failure
