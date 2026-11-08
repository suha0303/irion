"""Finite dependency handoff inside the existing collector. No LLM callbacks."""

import fcntl
import os
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from x500_storage import capture

from . import native_flight, pilot45, pilot_bridge
from .common import (
    HOME,
    ROOT,
    STORE,
    identity,
    now,
    read,
    runtime_identity,
    sha,
    verify,
    write,
)

DEFAULT_HANDOFF = (
    ROOT
    / "ai/reports/p27-pilot45-admission-20261107/recovery-01/data-handoff.json"
)


def init(
    name, previous, draft, handoff=DEFAULT_HANDOFF, *, reuse_integration=None
):
    pilot45.recovery_authorization()
    if not pilot45.re.fullmatch(r"[A-Za-z0-9_-]{1,56}", name):
        raise ValueError("Safe flow name required")
    task = pilot45.task_path(HOME / "tasks" / name)
    if task.exists():
        raise ValueError(
            (
                "Existing flow preserved; inspect/resume it instead of "
                "duplicate submission"
            )
        )
    previous, draft = pilot45.task_path(previous), pilot45.task_path(draft)
    task.mkdir(parents=True)
    sources = {
        str(p): sha(p) for p in (ROOT / "Data/generation/pretrain").glob("*.py")
    }
    sources.update(
        {str(p): sha(p) for p in (ROOT / "Data/generation/tests").glob("*.py")}
    )
    sources[str(pilot45.RECOVERY_AUTH)] = sha(pilot45.RECOVERY_AUTH)
    config = dict(
        mode="pilot45",
        flow=True,
        sealed=True,
        single_integration=False,
        planned_cases=46,
        owner_uid=os.getuid(),
        task_id=pilot45.TASK_ID,
        service_unit="drone-pretrain-" + name + ".service",
        python_runtime=runtime_identity(),
        external_storage=str(STORE),
        storage_policy=capture(STORE),
        source_sha256=sources,
        use_available_ram=True,
        use_swap=True,
        min_available_ram_gb=1,
        flight_workers=1,
        cpu_workers_total_max=1,
        wall_seconds=36 * 3600,
        dependency_wait_seconds=6 * 3600,
        dependency_handoff=str(Path(handoff).absolute()),
        previous_integration=str(previous),
        draft=str(draft),
        reuse_integration=str(pilot45.task_path(reuse_integration))
        if reuse_integration
        else None,
        training_authorized=False,
        fit_authorized=False,
    )
    write(task / "config.json", config)
    (task / "config.sha256").write_text(
        sha(task / "config.json") + "\n", encoding="utf-8"
    )
    write(task / "budget.json", dict(per_case_conservative_increment_bytes=0))
    write(task / "ledger.json", dict(cases={}, attempts=[], phases={}))
    write(
        task / "operator/state.json",
        dict(
            status="READY_NOT_STARTED",
            stage="dependency_wait",
            task=name,
            updated_utc=now(),
        ),
    )
    return task


def preflight(task):
    cfg = read(task / "config.json")
    if not cfg.get("flow") or cfg.get("owner_uid") != os.getuid():
        raise ValueError("Owned finite flow required")
    verify(cfg["source_sha256"])
    if cfg.get("final16"):
        from .final16 import validate_config

        validate_config(cfg)
    else:
        pilot45.short_authorization() if cfg.get(
            "short_cycle"
        ) else pilot45.recovery_authorization()
    return pilot45.resources()


def owned_status(task):
    unit, pid, properties = pilot45.service_identity(task)
    return dict(unit=unit, pid=pid, properties=properties)


def completed_properties(properties):
    required = {
        "MainPID",
        "LoadState",
        "ActiveState",
        "SubState",
        "Result",
        "ExecMainCode",
        "ExecMainStatus",
        "InvocationID",
    }
    if required - set(properties):
        raise RuntimeError("Actual completion properties missing")
    if properties["LoadState"] != "loaded":
        raise RuntimeError(
            "Unloaded unit defaults are not historical exit evidence"
        )
    if (
        properties["MainPID"] != "0"
        or properties["Result"] != "success"
        or properties["ExecMainCode"] != "1"
        or properties["ExecMainStatus"] != "0"
        or properties["ActiveState"] not in {"active", "inactive"}
        or properties["SubState"] not in {"exited", "dead"}
        or not pilot45.re.fullmatch(r"[0-9a-f]{32}", properties["InvocationID"])
    ):
        raise RuntimeError("Actual owned service did not exit successfully")


def execute(flow, task, state):
    """Only this finite program waits for its explicitly owned task service."""
    from .cli import start

    state.update(
        status="RUNNING",
        active_task=str(task),
        stage=read(task / "config.json")["collection_profile"],
        updated_utc=now(),
    )
    write(flow / "operator/state.json", state)
    if not (task / "operator/service.json").exists():
        start(task)
    elif (flow / "operator/resuming.json").exists():
        existing = owned_status(task)
        quality_path = task / "quality.json"
        done = quality_path.exists() and read(quality_path)["status"] in {
            "INTEGRATION19_QUALITY_PASSED_STOPPED",
            "PILOT45_QUALITY_PASSED_STOPPED",
        }
        if not existing["pid"] and not done:
            pilot45.start(task, resume=True)
    deadline = time.monotonic() + 24 * 3600
    while True:
        current = owned_status(task)
        if not current["pid"] and current["properties"].get(
            "ActiveState"
        ) not in {
            "activating",
            "deactivating",
        }:
            break
        if time.monotonic() > deadline:
            pilot45.stop(task, immediate=True)
            raise RuntimeError(
                "Finite task wait24h exceeded; only this own task stopped"
            )
        if (flow / "operator/stop-request.json").exists():
            if not (task / "operator/stop-request.json").exists():
                pilot45.stop(task)
        time.sleep(10)
    completed_properties(current["properties"])
    write(
        task / "operator/termination-before-stop.json",
        dict(
            schema="drone-data.owned-termination/1",
            task=str(task),
            unit=current["unit"],
            owner_uid=os.getuid(),
            config_sha256=sha(task / "config.json"),
            observed_utc=now(),
            properties=current["properties"],
        ),
    )
    if current["properties"].get("ActiveState") == "active":
        # Existing task units retain their exited state. Explicitly stop this
        # identity-checked PID0 unit so its ExecStopPost owns socket cleanup.
        subprocess.run(
            ["systemctl", "--user", "stop", current["unit"]],
            check=True,
            timeout=120,
        )
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                current["unit"],
                "--property=MainPID",
                "--property=ActiveState",
                "--property=LoadState",
                "--property=SubState",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        properties = dict(
            line.split("=", 1)
            for line in result.stdout.splitlines()
            if "=" in line
        )
        if (
            {"MainPID", "ActiveState", "LoadState", "SubState"}
            - set(properties)
            or properties["MainPID"] != "0"
            or properties["ActiveState"] != "inactive"
            or properties["SubState"] != "dead"
            or properties["LoadState"] not in {"loaded", "not-found"}
        ):
            raise RuntimeError("Own completed task unit did not terminate")
        current["after_explicit_stop"] = dict(
            exit_code=result.returncode, properties=properties
        )
    write(
        task / "operator/termination-confirmed.json",
        dict(
            schema="drone-data.owned-termination/1",
            task=str(task),
            unit=current["unit"],
            owner_uid=os.getuid(),
            config_sha256=sha(task / "config.json"),
            observed_utc=now(),
            termination=current,
        ),
    )
    quality = read(task / "quality.json")
    cleanup = read(task / "operator/socket-cleanup.json")
    passed = (
        (quality["status"] == "FINAL16_QUALITY_PASSED_STOPPED")
        if read(task / "config.json").get("final16")
        else quality["status"]
        in {
            "INTEGRATION19_QUALITY_PASSED_STOPPED",
            "PILOT45_QUALITY_PASSED_STOPPED",
        }
    )
    if (
        not passed
        or cleanup["status"] != "CLEANUP_COMPLETE"
        or cleanup.get("service_result") != "success"
    ):
        raise RuntimeError(
            "Owned task/quality/termination requires correction: " + str(task)
        )
    with ExitStack() as stack:
        for p in (
            HOME / "pretrain.lock",
            ROOT / "Data/operations/collection.lock",
            ROOT / "simulation/.cache/setup.lock",
            ROOT / "simulation/.cache/instances/0.lock",
        ):
            handle = stack.enter_context(p.open("rb"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = dict(
        task=str(task),
        quality=quality,
        termination=current,
        socket_cleanup=cleanup,
        locks4_released=True,
    )
    write(flow / (task.name + "-result.json"), result)
    state.update(active_task=None, updated_utc=now())
    write(flow / "operator/state.json", state)
    return result


def worker(task):
    cfg = read(task / "config.json")
    preflight(task)
    state = dict(
        status="RUNNING",
        stage="dependency_wait",
        task=task.name,
        pid=os.getpid(),
        pid_start_ticks=identity(os.getpid()),
        owner_uid=os.getuid(),
        updated_utc=now(),
        active_task=None,
    )
    ledger = read(task / "ledger.json")
    with (HOME / "pilot45-flow.lock").open("a", encoding="utf-8") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (task / "operator/claimed").exists():
            request = task / "operator/resume-request.json"
            if (
                not request.exists()
                or read(request).get("config_sha256")
                != sha(task / "config.json")
                or read(request).get("owner_uid") != os.getuid()
            ):
                raise RuntimeError("Explicit owned flow resume required")
            request.rename(task / "operator/resuming.json")
            stop = task / "operator/stop-request.json"
            if stop.exists():
                stop.rename(
                    task
                    / "operator"
                    / ("consumed-stop-" + str(time.time_ns()) + ".json")
                )
        else:
            with (task / "operator/claimed").open("x", encoding="utf-8") as out:
                out.write(now())
        write(task / "operator/state.json", state)
        try:
            deadline = time.monotonic() + cfg["dependency_wait_seconds"]
            while True:
                verify(cfg["source_sha256"])
                if (task / "operator/stop-request.json").exists():
                    raise RuntimeError(
                        "USER_STOP while waiting; no flight started"
                    )
                pilot45.resources()
                try:
                    spec = pilot_bridge.dependencies(cfg["dependency_handoff"])
                    break
                except pilot_bridge.Pending as exc:
                    state.update(
                        status="WAITING_DEPENDENCIES",
                        reason=str(exc),
                        dependency_file=cfg["dependency_handoff"],
                        updated_utc=now(),
                    )
                    write(task / "operator/state.json", state)
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        (
                            "Finite dependency wait6h exhausted; no automatic "
                            "LLM resumption"
                        )
                    )
                time.sleep(30)
            environment = task / "native-environment.json"
            regression = task / "final-admission-regression.log"
            if not regression.exists():
                argv = [
                    str(pilot45.PYTHON),
                    "-B",
                    "-m",
                    "pytest",
                    str(ROOT / "Data/generation/tests/test_ai19.py"),
                    str(ROOT / "Data/generation/tests/test_pilot45.py"),
                    str(ROOT / "Data/generation/tests/test_native_flight.py"),
                    str(ROOT / "Data/generation/tests/test_pilot_recovery.py"),
                    str(
                        ROOT
                        / "Data/generation/tests/test_collection_identity.py"
                    ),
                    "-q",
                ]
                with regression.open("x", encoding="utf-8") as log:
                    code = subprocess.run(
                        argv,
                        cwd=ROOT,
                        env=dict(
                            os.environ,
                            PYTHONPATH=":".join(
                                str(ROOT / p)
                                for p in (
                                    "Data/generation",
                                    "ai",
                                    "ai/src",
                                    "simulation/sim/src",
                                )
                            ),
                        ),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                        check=False,
                    ).returncode
                write(
                    task / "final-admission-regression.json",
                    dict(argv=argv, exit_code=code, log=str(regression)),
                )
                if code:
                    raise RuntimeError(
                        (
                            "Final stable-producer admission regression "
                            "failed; no flight"
                        )
                    )
            elif read(task / "final-admission-regression.json")["exit_code"]:
                raise RuntimeError(
                    "Prior regression failure preserved; correction required"
                )
            write(environment, native_flight.environment_check())
            if cfg.get("reuse_integration"):
                integration = pilot45.task_path(cfg["reuse_integration"])
            elif "integration" in ledger["phases"]:
                integration = pilot45.task_path(
                    ledger["phases"]["integration"]["task"]
                )
            else:
                integration = pilot45.init(
                    task.name + "-one",
                    single=True,
                    recovery_from=cfg["previous_integration"],
                    launch_receipt=environment,
                    dependency_handoff=cfg["dependency_handoff"],
                )
            ledger["phases"]["integration"] = dict(task=str(integration))
            write(task / "ledger.json", ledger)
            acceptance = None
            if cfg.get("reuse_integration"):
                from .integration_reuse import validate

                acceptance = task / "accepted-integration.json"
                if not acceptance.exists():
                    argv = [
                        str(pilot45.PYTHON),
                        "-B",
                        "-m",
                        "pretrain.integration_reuse",
                        "--task",
                        str(integration),
                        "--output",
                        str(acceptance),
                    ]
                    with (task / "accepted-integration.log").open(
                        "x", encoding="utf-8"
                    ) as log:
                        subprocess.run(
                            argv,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=1800,
                            check=True,
                        )
                proof = validate(integration, acceptance)
                first = dict(
                    task=str(integration),
                    quality=proof["quality"],
                    historical_termination=proof["termination"],
                    acceptance_receipt=str(acceptance),
                    reused=True,
                    new_flights=0,
                )
            else:
                first = execute(task, integration, state)
            if (task / "operator/stop-request.json").exists():
                raise RuntimeError(
                    "USER_STOP after current integration; no pilot started"
                )
            pilot = (
                pilot45.task_path(ledger["phases"]["pilot45"]["task"])
                if "pilot45" in ledger["phases"]
                else pilot45.prepare_pilot(
                    task.name + "-45",
                    cfg["draft"],
                    integration,
                    cfg["dependency_handoff"],
                    integration_acceptance=acceptance,
                )
            )
            ledger["phases"]["pilot45"] = dict(task=str(pilot))
            write(task / "ledger.json", ledger)
            final = execute(task, pilot, state)
            state.update(
                status="PILOT45_COMPLETE_TERMINATION_VERIFIED_STOPPED",
                stage="done",
                finished_utc=now(),
                next_stage_authorized=False,
            )
            result = dict(
                status=state["status"],
                integration=first,
                pilot45=final,
                native_dependency_completion=spec["paths"]["native_completion"],
                no_fit=True,
                no_model_forward=True,
                no_gpu=True,
                next_stage_authorized=False,
            )
            write(task / "completion.json", result)
            q = final["quality"]
            (task / "FINAL_REPORT.md").write_text(
                "# 예비45 종료 보고\n\n상태: "
                + state["status"]
                + "\n\n"
                + (
                    f"계획45, 완료원본{q['completed_raw']}, "
                    f"처리/검산{q['prepared_and_audited']}, "
                    f"물리시도{q['physical_attempts']}, "
                    f"실패시도{q['failed_physical_attempts']}. "
                    f"온라인가용{q['online_available']}, run가용{q['run_available']}. "
                    f"유효target {q['valid_target_seconds']}초, 초기100이후 "
                    f"{q['run_post_initial100_seconds']}초.\n\n"
                )
                + (
                    "실패/불가용·run별유효길이·RSS·저장량·실제unit/PID종료·소켓·잠금반납 근거는 "
                    "completion.json과각task quality.json/ledger.json에있다. "
                    "기존launcher실패도 integration 장부에보존했다. "
                    "학습·보정·모델forward·성능평가·GPU·다음회차는실행하지않고정지했다.\n"
                ),
                encoding="utf-8",
            )
        except BaseException as exc:
            state.update(
                status="BLOCKED_STOPPED",
                reason=f"{type(exc).__name__}: {exc}",
                finished_utc=now(),
                next_stage_authorized=False,
            )
            write(task / "failure.json", state)
            raise
        finally:
            write(task / "operator/state.json", state)
            write(task / "ledger.json", ledger)
