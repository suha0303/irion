"""Recovery/projection/wait gates use fixtures; native flight remains unexecuted."""

from pathlib import Path

import pytest
from test_pilot45 import init_fixture

from pretrain import common, pilot45, pilot_bridge, pilot_flow


@pytest.mark.local_evidence
def test_recovery_keeps_failed_attempt_and_exact_old_case(
    monkeypatch, tmp_path
):
    init_fixture(monkeypatch, tmp_path)
    old = pilot45.init("old", 99, single=True)
    original = common.read(old / "manifest.json")
    case = original["cases"][0]
    attempt = dict(
        number=1,
        state="FAILED_PRESERVED",
        parent_attempt_id=None,
        retry_allowed=False,
        error="launcher import failed",
    )
    common.write(
        old / "ledger.json",
        dict(
            cases={case["case_id"]: dict(case=case, attempts=[attempt])},
            attempts=[dict(case_id=case["case_id"], **attempt)],
        ),
    )
    log = old / "pilot/execution" / case["case_id"] / "flight-01.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "ModuleNotFoundError: No module named 'pymavlink'\n", encoding="utf-8"
    )
    audit = (
        tmp_path
        / "ai/reports/ai19-single-run-integration-20261107/attempt-02/result-review-01/termination-audit.json"
    )
    common.write(
        audit, dict(actual_raw_runs=0, checks=dict(shared_locks_free=True))
    )
    env = tmp_path / "environment.json"
    common.write(
        env,
        dict(status="PASSED_NO_FLIGHT", probe=dict(run_flight_imported=True)),
    )
    before = {str(p): common.sha(p) for p in old.rglob("*") if p.is_file()}
    monkeypatch.setattr(pilot45, "ROOT", tmp_path)
    monkeypatch.setattr(pilot45, "recovery_authorization", lambda: {})
    monkeypatch.setattr(
        pilot45,
        "service_identity",
        lambda *args: ("owned", 0, dict(ActiveState="failed")),
    )
    new = pilot45.init(
        "new", single=True, recovery_from=old, launch_receipt=env
    )
    newplan = common.read(new / "manifest.json")
    assert len(pilot45.validate_manifest(newplan)) == 1
    expected = dict(case, attempts_max=3)
    assert newplan["cases"] == [expected] and newplan["seed_base"] == 99
    ledger = common.read(new / "ledger.json")
    assert (
        ledger["attempts"][0]["number"] == 1
        and not ledger["attempts"][0]["retry_allowed"]
    )
    assert ledger["attempts"][0]["source_task"] == str(old)
    assert common.read(new / "config.json")["lineage_namespace"] == "old"
    assert before == {
        str(p): common.sha(p) for p in old.rglob("*") if p.is_file()
    }
    with pytest.raises(pilot45.Blocked, match="replacement seed"):
        pilot45.init(
            "bad", 123, single=True, recovery_from=old, launch_receipt=env
        )


@pytest.mark.local_evidence
def test_ai_projection_keeps_all45_pending_and_no_fit_guard(
    monkeypatch, tmp_path
):
    init_fixture(monkeypatch, tmp_path)
    task = pilot45.init("draft", 123)
    manifest = pilot_bridge.plan(task)
    assert len(manifest["cases"]) == 45 and manifest["no_fit"] is True
    assert len({c["case_id"] for c in manifest["cases"]}) == 45
    common.write(task / "ai-manifest.json", manifest)
    output = tmp_path / "projection"
    pilot_bridge.snapshot(task, output)
    ledger = common.read(output / "ledger.json")
    assert len(ledger["cases"]) == 45
    assert all(
        c["status"] == "pending" and c["attempts"] == []
        for c in ledger["cases"]
    )
    assert all(c["case_id"].startswith("draft:") for c in manifest["cases"])


def test_missing_or_waiting_handoff_cannot_enable_flights(tmp_path):
    p = tmp_path / "handoff.json"
    with pytest.raises(pilot_bridge.Pending):
        pilot_bridge.dependencies(p)
    common.write(
        p,
        dict(
            schema="drone-ai.pilot45.data-handoff/1",
            task_id=pilot45.TASK_ID,
            status="WAITING",
            reason="native6 missing",
        ),
    )
    with pytest.raises(pilot_bridge.Pending, match="native6 missing"):
        pilot_bridge.dependencies(p)
    common.write(
        p,
        dict(
            schema="drone-ai.pilot45.data-handoff/1",
            task_id=pilot45.TASK_ID,
            status="FAILED",
        ),
    )
    with pytest.raises(ValueError, match="no blind"):
        pilot_bridge.dependencies(p)


def test_flow_wait_is_finite_and_does_not_call_llm_or_launch_without_receipt(
    monkeypatch, tmp_path
):
    task = tmp_path / "flow"
    cfg = dict(
        flow=True,
        source_sha256={},
        dependency_handoff=str(tmp_path / "missing.json"),
        dependency_wait_seconds=0,
    )
    common.write(task / "config.json", cfg)
    common.write(task / "ledger.json", dict(cases={}, attempts=[], phases={}))
    (task / "operator").mkdir()
    monkeypatch.setattr(pilot_flow, "HOME", tmp_path)
    monkeypatch.setattr(pilot_flow, "preflight", lambda *args: {})
    monkeypatch.setattr(pilot45, "resources", lambda *args: {})
    monkeypatch.setattr(
        pilot_flow.time,
        "sleep",
        lambda *args: pytest.fail("Expired wait must stop immediately"),
    )
    with pytest.raises(RuntimeError, match="wait6h exhausted"):
        pilot_flow.worker(task)
    state = common.read(task / "operator/state.json")
    assert state["status"] == "BLOCKED_STOPPED"
    assert common.read(task / "ledger.json")["phases"] == {}
    assert not (task / "operator/service.json").exists()


@pytest.mark.parametrize("reuse", [False, True])
def test_flow_executes_only_one_integration_then_one_finite45_stage(
    monkeypatch, tmp_path, reuse
):
    task = tmp_path / "flow"
    common.write(
        task / "config.json",
        dict(
            flow=True,
            source_sha256={},
            dependency_handoff="final.json",
            dependency_wait_seconds=1,
            previous_integration="old",
            draft="draft",
            reuse_integration=str(tmp_path / "one") if reuse else None,
        ),
    )
    common.write(task / "ledger.json", dict(cases={}, attempts=[], phases={}))
    (task / "operator").mkdir()
    (task / "final-admission-regression.log").write_text(
        "fixture control-stage proof\n", encoding="utf-8"
    )
    common.write(task / "final-admission-regression.json", dict(exit_code=0))
    monkeypatch.setattr(pilot_flow, "HOME", tmp_path)
    monkeypatch.setattr(pilot_flow, "preflight", lambda *args: {})
    monkeypatch.setattr(pilot45, "resources", lambda *args: {})
    monkeypatch.setattr(
        pilot_bridge,
        "dependencies",
        lambda *args: dict(
            paths=dict(native_completion="actual-evidence.json")
        ),
    )
    monkeypatch.setattr(
        pilot_flow.native_flight,
        "environment_check",
        lambda: dict(status="PASSED_NO_FLIGHT"),
    )

    def init(*args, **kwargs):
        assert not reuse, "Successful integration must never be regenerated"
        return tmp_path / "one"

    monkeypatch.setattr(pilot45, "init", init)
    monkeypatch.setattr(pilot45, "task_path", lambda path: Path(path))
    monkeypatch.setattr(
        pilot45, "prepare_pilot", lambda *args, **kwargs: tmp_path / "pilot45"
    )
    phases = []
    q = dict(
        completed_raw=45,
        prepared_and_audited=45,
        physical_attempts=45,
        failed_physical_attempts=0,
        online_available=45,
        run_available=45,
        valid_target_seconds=21000,
        run_post_initial100_seconds=21150,
    )
    if reuse:
        from pretrain import integration_reuse

        common.write(task / "accepted-integration.json", dict(fixture=True))
        monkeypatch.setattr(
            integration_reuse,
            "validate",
            lambda *args: dict(
                quality=q, termination=dict(basis="fixture historical success")
            ),
        )

    def execute(flow, child, state):
        phases.append(child.name)
        return dict(quality=q)

    monkeypatch.setattr(pilot_flow, "execute", execute)
    pilot_flow.worker(task)
    assert phases == (["pilot45"] if reuse else ["one", "pilot45"])
    result = common.read(task / "completion.json")
    assert result["no_fit"] and result["no_model_forward"] and result["no_gpu"]
    assert not result["next_stage_authorized"]


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "missing_result",
        "failed_result",
        "missing_status",
        "nonzero_status",
        "not_exited",
        "unloaded",
    ],
    ids=[
        "success",
        "missing-result",
        "failed-result",
        "missing-status",
        "nonzero-status",
        "not-exited",
        "unloaded",
    ],
)
def test_successful_exited_unit_is_stopped_before_stop_hook_audit(
    monkeypatch, tmp_path, failure
):
    import os
    from types import SimpleNamespace

    root = tmp_path
    flow, child = root / "flow", root / "child"
    unit = "drone-pretrain-child.service"
    common.write(
        child / "config.json",
        dict(
            mode="pilot45",
            owner_uid=os.getuid(),
            service_unit=unit,
            collection_profile="integration19_common19",
        ),
    )
    common.write(child / "operator/service.json", dict(unit=unit))
    common.write(
        child / "quality.json",
        dict(status="INTEGRATION19_QUALITY_PASSED_STOPPED"),
    )
    monkeypatch.setattr(pilot_flow, "ROOT", root)
    monkeypatch.setattr(pilot_flow, "HOME", root)
    commands = []
    stopped = False
    original = dict(
        MainPID="0",
        ActiveState="active",
        SubState="exited",
        LoadState="loaded",
        ExecStart="pretrain.cli _worker " + str(child),
        ControlGroup="/" + unit,
        Result="success",
        ExecMainCode="1",
        ExecMainStatus="0",
        InvocationID="a" * 32,
    )
    if failure == "missing_result":
        original.pop("Result")
    if failure == "failed_result":
        original["Result"] = "exit-code"
    if failure == "missing_status":
        original.pop("ExecMainStatus")
    if failure == "nonzero_status":
        original["ExecMainStatus"] = "1"
    if failure == "not_exited":
        original["ExecMainCode"] = "0"
    if failure == "unloaded":
        original.update(
            LoadState="not-found",
            ActiveState="inactive",
            SubState="dead",
            ExecMainCode="0",
        )

    def command(argv, **kwargs):
        nonlocal stopped
        commands.append(argv)
        if "stop" in argv:
            stopped = True
            common.write(
                child / "operator/socket-cleanup.json",
                dict(status="CLEANUP_COMPLETE", service_result="success"),
            )
            return SimpleNamespace(returncode=0, stdout="")
        assert "show" in argv
        values = dict(original)
        if stopped:
            values.update(
                LoadState="not-found",
                ActiveState="inactive",
                SubState="dead",
                ExecMainCode="0",
                ExecStart="",
            )
        requested = [
            x.split("=", 1)[1] for x in argv if x.startswith("--property=")
        ]
        return SimpleNamespace(
            returncode=0,
            stdout="".join(
                k + "=" + values[k] + "\n" for k in requested if k in values
            ),
        )

    monkeypatch.setattr(pilot45.subprocess, "run", command)
    for p in (
        root / "pretrain.lock",
        root / "Data/operations/collection.lock",
        root / "simulation/.cache/setup.lock",
        root / "simulation/.cache/instances/0.lock",
    ):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    if failure:
        with pytest.raises(RuntimeError):
            pilot_flow.execute(flow, child, {})
        assert not stopped
    else:
        result = pilot_flow.execute(flow, child, {})
        assert stopped and result["locks4_released"]
        receipt = common.read(child / "operator/termination-before-stop.json")
        assert receipt["properties"]["Result"] == "success"
        assert receipt["properties"]["ExecMainStatus"] == "0"
    assert "--property=Result" in commands[0]
    assert "--property=ExecMainStatus" in commands[0]


@pytest.mark.parametrize("reject", [False, True])
def test_data_pair_uses_ai_projector_for_status_passed_native_report(
    monkeypatch, tmp_path, reject
):
    # The native report intentionally has no boolean passed/AI metric keys.
    # Actual native validation is owned/tested by AI; this checks Data delegation.
    import learning.history19.pilot_connector as connector

    task = tmp_path / "task"
    normal, anomaly = tmp_path / "normal", tmp_path / "anomaly"
    case = dict(group_id="one-pair", paired_case_id="actual-pair-id")
    common.write(
        task / "ledger.json",
        dict(
            cases={
                "n": dict(case=case, run=str(normal)),
                "a": dict(case=case, run=str(anomaly)),
            }
        ),
    )
    common.write(task / "expected-producer.json", dict(fixture=True))
    policy = tmp_path / "policy.json"
    common.write(policy, dict(native_adapter="fixture-only"))
    report = task / "pilot/pairs/one-pair/native.json"
    common.write(report, dict(status="passed", fixture_only=True))
    called = []

    def project(path, **kwargs):
        called.append((path, kwargs))
        if reject:
            raise ValueError("actual native coverage rejected")
        return dict(
            passed=True,
            normal_run_id="normal",
            anomaly_run_id="anomaly",
            normal_origin_receipt_sha256="1" * 64,
            anomaly_origin_receipt_sha256="2" * 64,
        )

    monkeypatch.setattr(connector, "project_pair_receipt", project)
    spec = dict(
        paths=dict(paired_policy=str(policy)),
        pair_verifier_argv=[
            str(common.ROOT / "simulation/capstone-sim"),
            "joint-phase-verify-pair",
            "--normal-run",
            "{normal_run}",
            "--anomaly-run",
            "{anomaly_run}",
            "--out",
            "{output}",
        ],
    )
    if reject:
        with pytest.raises(ValueError, match="coverage rejected"):
            pilot_bridge.pair(task, spec, "n", "a")
        assert not (report.parent / "projection.json").exists()
    else:
        result = common.read(pilot_bridge.pair(task, spec, "n", "a"))
        assert result["passed"] and result["pair_id"] == "actual-pair-id"
        assert result["normal_origin_receipt_sha256"] == "1" * 64
        assert result["native_evidence"]["sha256"] == common.sha(report)
    assert called == [
        (
            report,
            dict(
                policy_path=str(policy),
                normal_run=str(normal),
                anomaly_run=str(anomaly),
            ),
        )
    ]


def test_reuse_source_gate_only_allows_controller_fix(monkeypatch, tmp_path):
    from pretrain import integration_reuse

    task = tmp_path / "old"
    control, preprocessor = (
        tmp_path / "controller.py",
        tmp_path / "preprocess.py",
    )
    control.write_text("old control", encoding="utf-8")
    preprocessor.write_text("same preprocessing", encoding="utf-8")
    common.write(
        task / "config.json",
        dict(
            source_sha256={
                str(p): common.sha(p) for p in (control, preprocessor)
            }
        ),
    )
    monkeypatch.setattr(integration_reuse, "CONTROL_FILES", {str(control)})
    control.write_text("reviewed termination correction", encoding="utf-8")
    result = integration_reuse.compatible_sources(task)
    assert set(result["reviewed_control_changes"]) == {str(control)}
    preprocessor.write_text("changed signal processing", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency changed"):
        integration_reuse.compatible_sources(task)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing_result",
        "failed_result",
        "child_nonzero",
        "child_missing",
        "unit_mismatch",
        "live",
    ],
)
def test_historical_reuse_requires_actual_cleanup_and_child_exit_evidence(
    monkeypatch, tmp_path, fault
):
    from pretrain import integration_reuse

    task = tmp_path / "old"
    unit = "owned-completed.service"
    group = "0::/fixture/" + unit + "\n"
    cleanup = dict(
        status="CLEANUP_COMPLETE",
        service_result="success",
        errors=[],
        remaining_unremoved_sockets=[],
    )
    if fault == "missing_result":
        cleanup.pop("service_result")
    if fault == "failed_result":
        cleanup["service_result"] = "exit-code"
    common.write(task / "operator/socket-cleanup.json", cleanup)
    common.write(task / "operator/service.json", dict(unit=unit))
    common.write(
        task / "operator/state.json",
        dict(
            status="INTEGRATION19_QUALITY_PASSED_STOPPED",
            service_unit=unit if fault != "unit_mismatch" else "foreign",
            pid=999999,
            pid_start_ticks="10",
            owner_uid=1000,
            cgroup=group,
        ),
    )
    for stage in ("flight", "prepare", "audit"):
        point = dict(
            exit_code=0,
            owner_uid=1000,
            cgroup=group,
            pid=999998,
            pid_start_ticks="11",
        )
        if stage == "prepare" and fault == "child_nonzero":
            point["exit_code"] = 1
        if stage == "prepare" and fault == "child_missing":
            point.pop("exit_code")
        common.write(
            task
            / "pilot/execution/case"
            / (stage + "-02.request.process.json"),
            point,
        )
    monkeypatch.setattr(
        integration_reuse,
        "identity",
        lambda p: "10" if fault == "live" else None,
    )
    if fault:
        with pytest.raises(ValueError):
            integration_reuse.historical_termination(task)
    else:
        result = integration_reuse.historical_termination(task)
        assert result["unloaded_Result_not_used"]
        assert result["service_result"] == "success"
