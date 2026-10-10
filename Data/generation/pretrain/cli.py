"""Drone data console (default X500): optional gates -> collection -> preparation -> quality -> stop."""

import argparse
import faulthandler
import json
import os
import re
import secrets
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

from x500_storage import capture, environment, guard, selected_root

# Support manually invoked workers as well as systemd launches. No execution occurs here.
if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "_worker":
    faulthandler.enable()
    worker_config = json.loads((Path(sys.argv[2]) / "config.json").read_text())
    os.environ.update(environment(worker_config))

from .common import (
    ADMISSION,
    CHOICES,
    CPU_WORKER_BYTES,
    HOME,
    MAX_FLIGHT_WORKERS,
    PACKAGE,
    PRODUCER_MANIFEST,
    PYTHON,
    RESOURCES,
    ROOT,
    SIMULATION,
    STORE,
    TASK_WALL_SECONDS,
    collector,
    identity,
    load,
    now,
    producer_sources,
    pythonpath,
    read,
    runtime_identity,
    service_unit,
    sha,
    verify,
    verify_runtime,
    write,
)
from .composition import SIZES, seed_stride
from .display import flight_cards, grid, screen


def duration(seconds):
    if seconds is None:
        return "미확인"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def snapshot(task):
    path = task / "operator/state.json"
    state = read(path) if path.exists() else dict(status="NOT_STARTED", completed=[], active=[])
    if state["status"] in {"RUNNING", "DRAINING"} and (
        not state.get("pid_start_ticks")
        or identity(state.get("pid")) != state.get("pid_start_ticks")
    ):
        state["status"] = "UNKNOWN_NOT_RUNNING"
        state["reason"] = (
            "기록 PID가 실행 중이 아님. 성공으로 처리하지 않습니다. run.log / systemctl 상태 확인 필요"
        )
    # Older successful flight tasks cleared the ledger before copying its count.
    # Repair the view only; preserve the original evidence files.
    if state["status"] in {
        "TEST_COMPLETE_STOPPED", "PRETRAIN_READY_STOPPED_PENDING_USER_REVIEW",
        "PRETRAIN_COMPACT_READY_STOPPED_RESTORE_REQUIRED",
    }:
        for stage in state.get("stages", []):
            if stage["name"] == state.get("stage"):
                if "completed" in stage:
                    state["done"] = stage["completed"]
                elif stage.get("passed") is True and stage["name"] in state.get("completed", []):
                    state["done"] = state.get("total", 0)
    processing = task / "operator/processing-state.json"
    if state.get("pipeline_overlap") and processing.exists():
        cpu = read(processing)
        state["processing"] = cpu
        for key in ("stream_completed", "stream_retired_paths"):
            state[key] = cpu.get(key, state.get(key, 0))
        state["completed"] = list(
            dict.fromkeys(state.get("completed", []) + cpu.get("completed", []))
        )
    return state


def render(task, tick=0):
    state = snapshot(task)
    config = read(task / "config.json")
    finished = state.get("finished")
    elapsed = (finished or time.time()) - state.get("started", time.time())
    total, done = state.get("total", 0), state.get("done", 0)
    collection = None
    ledger = state.get("collection_ledger")
    if ledger:
        data = load("pretrain_data_view", ROOT / "ai/x500_data.py")
        collection = data.snapshot(Path(ledger).parent.parent / "plan.json")
        total, done = collection["total"], collection["archived"] + collection.get("failed", 0)
    percent = 100 * done / total if total else None
    complete = state["status"] in {
        "TEST_COMPLETE_STOPPED",
        "PRETRAIN_READY_STOPPED_PENDING_USER_REVIEW",
        "PRETRAIN_COMPACT_READY_STOPPED_RESTORE_REQUIRED",
    }
    if config["mode"] == "test":
        planned = len(config["levels"]) * (
            (0 if config.get("flights_only") else 5) + bool(config["test_flights"])
        )
    elif config["mode"] == "resume_preprocess":
        planned = config["collection_sets"] + 6
    elif config["mode"] == "finish_preprocess":
        planned = 5
    elif config["mode"] == "finish_audit":
        planned = 3
    elif config["mode"] == "merge_prepared":
        planned = 7
    elif config["mode"] == "recover_capsules":
        planned = 8
    else:
        planned = (
            (
                (
                    2 * config.get("collection_sets", 1) + 6
                    if config.get("dataset_scope") == "all_selected_runs"
                    else 7 * config.get("collection_sets", 1)
                )
                + 2 * bool(config["gates"])
            )
            if config["mode"] == "collect"
            else 5
        )
    stopped = state["status"] in {"FAILED_STOPPED", "BLOCKED_STOPPED", "UNKNOWN_NOT_RUNNING"}
    if config.get("storage_profile") == "compact" and config["mode"] not in {
        "test",
        "merge_prepared",
    }:
        planned += 1
    if state.get("planned_stage_count"):
        planned = state["planned_stage_count"]
    overall = (
        "완료 100%"
        if complete
        else "미완료 · 중지"
        if stopped
        else "진행 중"
        if state["status"] in {"RUNNING", "DRAINING"}
        else "시작 전"
    )
    if config["mode"] == "test":
        settings = "시험 병렬 " + " → ".join(map(str, config["levels"]))
        settings += f"  ·  현재 단계 {state.get('workers', '-')}개 (일반 작업 설정과 별개)"
        if config.get("speed_test"):
            factors = config["speed_test"]["factors"]
            settings = f"배속{'→'.join(map(str, factors))} ·120초·동일 seed·단일 기체 | 현재 {state.get('speed_trial', '-')}/{len(factors)} · {state.get('speed_factor', '-')}x"
    elif config["mode"] == "resume_preprocess":
        settings = f"기존 {config['collection_run_count']}run 재개 / 추출 {config['io_workers']} / 전처리 {config['workers']} / 검산 {config['audit_workers']} · 재수집 없음"
    elif config["mode"] == "finish_preprocess":
        settings = f"실패1개 전처리1병렬 / 성공 {config['reused_run_count']}개 재사용 / 검산 {config['audit_workers']} · 재수집·재추출 없음"
    elif config["mode"] == "finish_audit":
        settings = f"실패1개 검산1병렬 / 성공 검산 {config['reused_audit_count']}개 재사용 · 전처리·집계 반복 없음"
    elif config["mode"] == "merge_prepared":
        settings = f"기존 {config['collection_run_count']}run 통합 / 검산 {config['audit_workers']}병렬 · 공통 train scaler · 수집·학습 없음"
    elif config["mode"] == "recover_capsules":
        settings = f"보존 캡슐 {config['collection_run_count']}run 복원·통합 · 공통 train scaler · 재수집·학습 없음"
    else:
        settings = f"비행 {config['flight_workers']} / 검증·압축 {config['export_workers']} / 추출 {config['io_workers']} / 전처리 {config['workers']} / 검산 {config['audit_workers']}"
    lines = [
        "드론  /  학습 전 통합 콘솔 (기체 X500)  /  " + task.name,
        f"  {state['status']}  |  전체 {overall}",
        f"  {time.strftime('%m/%d %H:%M:%S %Z')}  ·  화면 #{tick}  ·  경과 {duration(elapsed)}  ·  PID {state.get('pid', '-')}",
        "─" * 78,
        "흐름   보존 캡슐 인수·복원 → 통합 집계 → 독립 검산 → 압축 보관 → 정지"
        if config["mode"] == "recover_capsules"
        else "흐름   기존 압축본 인수·복원 → 통합 집계 → 독립 검산 → 정리 → 정지"
        if config["mode"] == "merge_prepared"
        else "흐름   기존 결과 인수 → 실패1개 검산 → 전체 검산 → 정지"
        if config["mode"] == "finish_audit"
        else "흐름   기존 결과 인수 → 실패1개 전처리 → 전체 집계 → 독립 검산 → 정지"
        if config["mode"] == "finish_preprocess"
        else "흐름   압축본 검증·인수 → 추출 → 전처리 → 집계 → 독립 검산 → 정지"
        if config["mode"] == "resume_preprocess"
        else "흐름   [선택 게이트] → 수집 → 추출 → 전처리 → 집계 → 독립 검산 → 정지",
        "설정   " + settings,
        f"현재   {state.get('stage', '-')}  |  완료 단계 {len(state.get('completed', []))}/{planned}  (시간 진행률 아님)",
    ]
    storage = state.get("storage_relocation", {}).get(
        "storage_root", config.get("external_storage")
    )
    if storage:
        lines.append("저장   " + storage)
    if config["mode"] == "collect":
        sets = config.get("collection_sets", 1)
        size = SIZES[config.get("collection_profile", "legacy50")]
        lines.append(
            f"수집 계획   {size}개 × {sets}묶음 = {size * sets}개 · seed {config.get('seed_base', '-')} · 학습 없음"
        )
    if config.get("storage_profile") in {"compact", "stream_compact"}:
        if config["mode"] != "merge_prepared":
            lines[4] = lines[4].replace(" → 정지", " → 압축 패키지 → 정지")
        else:
            lines[4] = (
                "흐름   기존 압축본 인수·복원 → 통합 집계 → 독립 검산 → 압축 → 임시 정리 → 정지"
            )
        if config["storage_profile"] == "stream_compact":
            lines[4] = (
                "흐름   보존 캡슐 인수·복원 → 공통 scaler·통합 검산 → 압축 보관 → 정지"
                if config["mode"] == "recover_capsules"
                else "흐름   동일 길이 묶음 비행 ∥ 이전 묶음 후처리 / 전체 집계·검산→보관"
                if config.get("pipeline_overlap")
                else "흐름   유한 묶음: 비행→검증→전처리→run 검산→압축·정리 / 전체 집계·검산→보관"
            )
            raw_limit = config["flight_workers"] * (2 if config.get("pipeline_overlap") else 1)
            lines.append(
                "보관   원본 캡슐·실패 기록 보존 · 새 작업의 임시 파일만 정리"
                if config["mode"] == "recover_capsules"
                else f"보관   검증 후 새 원시 정리 · 임시 원시 최대{raw_limit}run + 실패 증거 별도 보존"
            )
            if config["mode"] != "recover_capsules":
                lines.append(
                    f"압축 완료   {state.get('stream_completed', 0)}/{config['collection_run_count']}run · 정리 경로 {state.get('stream_retired_paths', 0)}개"
                )
            if state.get("portable_dataset"):
                lines.append("보관본   " + state["portable_dataset"])
            if state.get("dataset_requires_restore"):
                lines.append(
                    "최종 데이터셋은 무손실 압축 보관 중 · 학습 인수 전 새 준비 경로에 복원 필요"
                )
        else:
            lines.append("보관   검산 후 무손실 데이터셋·평가 정답 분리 압축 / 원시 자동 삭제 없음")
        if config.get("dataset_scope") == "all_selected_runs":
            lines.append("전처리   전체 통합 데이터셋 1개 · 전체 train 기준 공통 scaler 1개")
        if config["mode"] == "collect":
            lines.append(
                f"수집 완료   {len(state.get('collected_sets', []))}/{sets}묶음 (현재 묶음 진행률은 아래 표시)"
            )
    if config.get("use_available_ram"):
        lines.append("메모리 설정   예상·고정 RSS 상한 해제 · 최소 가용 RAM1GB")
        memory_file = task / "operator/memory-policy.json"
        if memory_file.exists():
            memory = read(memory_file)
            lines.append(
                f"OS 메모리 상한   RAM {memory['max_bytes'] / 1e9:.2f} GB + 스왑 {memory.get('swap_max_bytes', 0) / 1e9:.2f} GB (가용 RAM1GB 보호)"
            )
    elif config.get("ignore_memory_estimate"):
        lines.append(
            f"시험 예외   예상 차단 해제 · 최소 가용 RAM {config.get('min_available_ram_gb', 4)}GB · RSS8GB / OS9GB 유지"
        )
    processing = state.get("processing")
    if processing:
        lines.append(
            f"병행 후처리   {processing.get('stage', '-')} · {processing.get('done', 0)}/{processing.get('total', 0)} · {processing.get('status', '-')} · 비행과 별도"
        )
        if processing.get("status") == "FAILED_STOPPED":
            lines.append("후처리 오류   " + str(processing.get("reason")))
    if total:
        width = 24
        filled = min(width, int(width * done / total))
        lines.append(
            f"개별 처리 [{'█' * filled}{'·' * (width - filled)}] {percent:5.1f}%  {done}/{total}건"
        )
        if stopped:
            lines.append("단계 판정: 통과 아님. 개별 처리 완료율은 자원·품질 승인율이 아닙니다.")
        eta = (
            ((time.time() - state.get("stage_started", time.time())) / done * (total - done))
            if done >= state.get("workers", 1)
            else None
        )
        if not stopped and not complete:
            lines.append(
                f"단계 잔여 예상 {duration(eta)}  ·  완료 건수 평균 기준, 다음 단계 시간 제외"
            )
    resources = state.get("resources", {})
    estimate = state.get("admission")
    if estimate:
        decision = (
            "추정 차단 해제 · 실측 보호 유지"
            if estimate.get("estimate_only_override")
            else "통과"
            if estimate["passed"]
            else "부족: 실행 안 함"
        )
        lines.append(
            f"자원 계획   예상 {estimate['estimated_bytes'] / 1e9:.2f} GB  ·  시작 전 가용 {estimate['available_bytes'] / 1e9:.2f} GB  ·  {decision}"
        )
    if resources:
        lines.append(
            f"마지막 자원 기록   가용 RAM {resources.get('available_bytes', 0) / 1e9:.1f} GB  ·  사용 RSS {resources.get('owned_tree_rss_bytes', 0) / 1e9:.2f} GB  ·  저장 여유 {resources.get('external_free_bytes', 0) / 1e9:.1f} GB"
        )
    violation = state.get("first_resource_violation")
    if violation:
        lines.append(f"한도 위반 순간   {violation['observed_utc']}  ·  {violation['reason']}")
    elif stopped and "RAM" in str(state.get("reason")):
        lines.append(
            "한도 위반 순간 수치: 이전 버전에서 저장하지 않아 미확인. 마지막 자원 기록과 다릅니다."
        )
    if collection:
        lines.append(
            f"수집   보관 {collection['archived']} / 실패 {collection.get('failed', 0)} / 전체 {collection['total']} · 잔여 예상 {duration(collection.get('eta', {}).get('seconds'))}"
        )
        failures = [w for w in collection["workers"] if w["status"] == "FAILED_PRESERVED"]
        active_workers = [w for w in collection["workers"] if w["status"] != "FAILED_PRESERVED"]
        lines.extend(grid(flight_cards(active_workers + failures, duration)))
        if failures:
            lines.append(f"실패 증거 {len(failures)}건 · 최근: " + str(failures[-1].get("error")))
        for reason in collection.get("reason", []):
            lines.append("수집 중지: " + str(reason))
    cpu_cards = []
    # During flights keep the 8+8 flight grid; once drained, show CPU worker detail.
    active_cpu = state.get("active", [])
    if not collection and processing and processing.get("status") == "RUNNING":
        active_cpu = processing.get("active", [])
    for index, active in enumerate(active_cpu):
        path = Path(active["status_file"])
        item = read(path) if path.exists() else {}
        amount, limit = item.get("bytes_done", 0), item.get("bytes_total", 0)
        progress = (
            f"{100 * amount / limit:5.1f}% ({amount / 1e6:.0f}/{limit / 1e6:.0f} MB)"
            if limit
            else "건별 처리 중"
        )
        cpu_cards.append(
            f"{index:>2} {active['run_id'][:6]} {item.get('phase', item.get('action', '시작'))[:10]} {progress.split('(')[0].strip()} {active.get('attempt', 1)}/{active.get('attempts_max', 1)}"
        )
        if "windows" in item:
            cpu_cards[-1] += f" 창{item['windows']:,}"
    lines.extend(grid(cpu_cards))
    if state.get("reason"):
        lines.append("사유   " + state["reason"])
    cleanup_file = task / "operator/socket-cleanup.json"
    if cleanup_file.exists():
        cleanup = read(cleanup_file)
        removed = sum(a["result"] == "removed_owned_stale_socket" for a in cleanup["actions"])
        lines.append(
            f"소켓 정리   {cleanup['status']} · 제거 {removed}개 · 남은 소켓 {len(cleanup['remaining_unremoved_sockets'])}개 (미확인 소유는 보존)"
        )
    if state.get("result"):
        lines.append("결과   " + state["result"])
    if state.get("stages"):
        lines.append("최근 완료")
        for stage in state["stages"][-5:]:
            lines.append(
                f"  ✓ {stage['name']}  ·  병렬 {stage.get('workers', 1)}  ·  {duration(stage.get('elapsed_s'))}"
            )
    lines += [
        "─" * 78,
        "로그   " + str(task / "operator/run.log"),
        "화면만 자동 갱신 · q / Ctrl+C: 화면 닫기 · 작업은 독립 실행",
        f"run별 최대 {config.get('unit_attempts_max', 1)}회 시도 · 전체 작업 재시작·학습·LLM 호출 없음.",
    ]
    return "\n".join(lines)


def watch(task, once=False):
    previous = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() and not once else None
    try:
        if previous:
            tty.setcbreak(sys.stdin.fileno())
        tick = 0
        while True:
            tick += 1
            text = render(task, tick)
            if sys.stdout.isatty() and not once:
                size = shutil.get_terminal_size()
                text = screen(text, size.columns, size.lines - 1)
                print("\033[2J\033[H", end="")
            print(text, flush=True)
            if once:
                return
            if previous:
                ready, _, _ = select.select([sys.stdin], [], [], 2)
                if ready and sys.stdin.read(1).lower() == "q":
                    return
            else:
                time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        if previous:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, previous)


def create(args):
    if args.mode not in {"test", "collect"}:
        raise ValueError("attempt-07 재현은 제거됐습니다. test/collect만 실행할 수 있습니다")
    if not 1 <= args.flight_workers <= MAX_FLIGHT_WORKERS:
        raise ValueError("PX4 instance 계약상 비행 병렬은1..16입니다")
    if any(
        type(n) is not int or n <= 0
        for n in (args.workers, args.io_workers, args.audit_workers, args.export_workers)
    ):
        raise ValueError("병렬 수는 양의 정수여야 합니다")
    args.test_flights = args.mode == "test"
    args.flights_only = False
    if args.use_available_ram:
        args.ignore_memory_estimate = True
        args.min_available_ram_gb = 1
    if args.ignore_memory_estimate and args.mode != "test" and not args.use_available_ram:
        raise ValueError("예상 메모리 차단 해제는 병렬 시험 전용입니다")
    if args.min_available_ram_gb != 4 and not args.ignore_memory_estimate:
        raise ValueError("가용 RAM 완화는 예상 차단을 해제한 CPU 시험 전용입니다")
    if args.mode != "test" and (args.test_flights or args.flights_only):
        raise ValueError("비행 시험 옵션은 test 모드 전용입니다")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.name):
        raise ValueError("Task name must contain1..64 ASCII letters/digits/-/_")
    if args.export_workers > args.flight_workers:
        raise ValueError("검증·압축 병렬 수는 비행 병렬 수 이하여야 합니다")
    if args.gates and args.mode != "collect":
        raise ValueError("게이트는 신규 수집 모드에서만 선택합니다")
    if args.collection_sets < 1 or (args.collection_sets != 1 and args.mode != "collect"):
        raise ValueError("수집 묶음 수는 collect 모드의 양의 정수입니다")
    # Reserve room for16-way benchmark programs and optional gate seeds as well.
    seed_max = 2**31 - 20_000 - seed_stride(args.collection_profile) * args.collection_sets
    if seed_max < 0:
        raise ValueError("수집 배율이 seed 범위를 초과합니다")
    automatic_seed = args.seed_base is None
    if automatic_seed:
        args.seed_base = secrets.randbelow(seed_max + 1)
    if args.levels != sorted(set(args.levels)) or not 0 <= args.seed_base <= seed_max:
        raise ValueError("시험 병렬 수는 오름차순·중복 없이, seed는 작업별 허용 범위여야 합니다")
    storage_root = guard(getattr(args, "storage_root", selected_root()))
    storage_policy = capture(storage_root)
    sources = list((ROOT / "Data/generation/pretrain").glob("*.py")) + [
        ROOT / "Data/generation/drone-pretrain",
        ROOT / "ai/x500_storage.py",
        ROOT / "Data/generation/tools/collect_long_runs.py",
        ROOT / "Data/generation/tools/collection_support.py",
        ADMISSION / "derived.py",
        ADMISSION / "freeze.py",
        RESOURCES / "protected-inventory.json",
        RESOURCES / "case_manifest.json",
        PACKAGE / "PACKAGE.json",
        PRODUCER_MANIFEST,
    ]
    sources += [
        p
        for p in RESOURCES.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and not p.is_relative_to(PACKAGE)
    ]
    native_inventory = producer_sources()
    package = read(PACKAGE / "PACKAGE.json")
    inventory = {str(p.resolve()): sha(p) for p in sources}
    runtime = runtime_identity()
    inventory[runtime["executable"]] = runtime["sha256"]
    inventory.update(
        {str((PACKAGE / name).resolve()): digest for name, digest in package["files"].items()}
    )
    inventory.update(
        {str((SIMULATION / name).absolute()): digest for name, digest in native_inventory.items()}
    )
    task = HOME / "tasks" / args.name
    task.mkdir(parents=True, exist_ok=False)
    (task / "operator").mkdir()
    config = dict(
        schema="x500.pretrain-cli/1",
        name=args.name,
        service_unit="drone-pretrain-" + args.name + ".service",
        vehicle="x500_reference",
        mode=args.mode,
        gates=args.gates,
        workers=args.workers,
        flight_workers=args.flight_workers,
        io_workers=args.io_workers,
        audit_workers=args.audit_workers,
        export_workers=args.export_workers,
        levels=args.levels,
        test_flights=args.test_flights,
        flights_only=args.flights_only,
        ignore_memory_estimate=args.ignore_memory_estimate,
        min_available_ram_gb=args.min_available_ram_gb,
        use_available_ram=args.use_available_ram,
        use_swap=args.use_swap,
        seed_base=args.seed_base,
        seed_source="os_secure_random" if automatic_seed else "user_explicit",
        collection_sets=args.collection_sets,
        collection_profile=args.collection_profile,
        storage_profile=args.storage_profile if args.mode == "collect" else "standard",
        pipeline_overlap=args.mode == "collect" and args.storage_profile == "stream_compact",
        collection_order="mission_seconds_desc_then_frozen_manifest",
        dataset_scope="all_selected_runs",
        collection_run_count=SIZES[args.collection_profile] * args.collection_sets
        if args.mode == "collect"
        else 0,
        source_sha256=inventory,
        python_runtime=runtime,
        created_utc=now(),
        training_authorized=False,
        continue_after_case_failure=True,
        unit_attempts_max=3,
        scope="Finite collection; per-unit maximum3 attempts; no final test or training",
        external_storage=str(storage_root),
        storage_policy=storage_policy,
        wall_seconds=TASK_WALL_SECONDS,
        external_additional_bytes_max=80_000_000_000
        * (
            SIZES[args.collection_profile] * args.collection_sets / 50
            if args.mode == "collect"
            else 1
        ),
        cpu_worker_bytes=CPU_WORKER_BYTES,
        test_scope="Fresh finite flights at requested concurrency, archive/extract/preprocess/independent audit; no historical raw data or training",
    )
    if args.mode == "collect":
        from .collection import cases

        planned = [
            cases(
                args.seed_base + seed_stride(args.collection_profile) * i, args.collection_profile
            )["cases"]
            for i in range(args.collection_sets)
        ]
        write(
            task / "planned-cases.json",
            dict(profile=args.collection_profile, sets=planned, execution_started=False),
        )
        config["planned_cases"] = str(task / "planned-cases.json")
        config["source_sha256"][config["planned_cases"]] = sha(task / "planned-cases.json")
    write(task / "config.json", config)
    (task / "config.sha256").write_text(sha(task / "config.json") + "\n")
    print("준비됨:", task)
    return task


def systemd_command(arguments):
    """Quote one ExecStart-style argument vector, including percent specifiers."""
    return " ".join(
        '"'
        + argument.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$")
        + '"'
        for argument in arguments
    )


def start(task):
    task = task.resolve()
    if task.parent != HOME / "tasks":
        raise ValueError("등록된 로컬 task 디렉터리가 필요합니다")
    if (task / "operator/claimed").exists():
        raise ValueError(
            "STOP: Attempt already claimed; no automatic retry. 새 task 이름을 사용하세요."
        )
    verify({str(task / "config.json"): (task / "config.sha256").read_text().strip()})
    cfg = read(task / "config.json")
    if cfg.get("python_allocator") not in (None, "debug"):
        raise ValueError("Unsupported task Python allocator")
    verify_runtime(cfg)
    verify(cfg["source_sha256"])
    storage_env = environment(cfg)
    memory = service_memory(cfg)
    write(task / "operator/memory-policy.json", memory)
    unit = service_unit(task, cfg).removesuffix(".service")
    command = [
        "systemd-run",
        "--user",
        "--expand-environment=no",
        "--unit=" + unit,
        "--property=Type=exec",
        "--property=Restart=no",
        "--property=KillMode=control-group",
        "--property=OOMPolicy=stop",
        "--property=ExecStopPost="
        + systemd_command([str(PYTHON), "-B", "-m", "pretrain.socket_cleanup", str(task)]),
        "--property=RemainAfterExit=yes",
        "--property=MemoryAccounting=yes",
        "--property=MemoryHigh=" + str(memory["high_bytes"]),
        "--property=MemoryMax=" + str(memory["max_bytes"]),
        "--property=RuntimeMaxSec=" + str(cfg.get("wall_seconds", TASK_WALL_SECONDS)),
        "--property=TimeoutStopSec=900",
        "--property=WorkingDirectory=" + str(ROOT),
        "--property=StandardOutput=append:" + str(task / "operator/run.log"),
        "--property=StandardError=append:" + str(task / "operator/run.log"),
        "--setenv=PYTHONPATH=" + pythonpath(),
        "--setenv=PYTHONDONTWRITEBYTECODE=1",
        "--setenv=PYTHONFAULTHANDLER=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        str(PYTHON),
        "-B",
        "-m",
        "pretrain.cli",
        "_worker",
        str(task),
    ]
    for key, value in storage_env.items():
        command.insert(2, "--setenv=" + key + "=" + value)
    if cfg.get("python_allocator"):
        command.insert(2, "--setenv=PYTHONMALLOC=" + cfg["python_allocator"])
    if cfg.get("use_available_ram"):
        command.insert(2, "--property=MemorySwapMax=" + str(memory["swap_max_bytes"]))
    subprocess.run(command, check=True)
    write(
        task / "operator/service.json",
        dict(
            unit=unit + ".service",
            submitted_utc=now(),
            restart="no",
            memory=memory,
            python_allocator=cfg.get("python_allocator", "default"),
            session_lifetime="user systemd; survives terminal closure, host shutdown/logout may stop user manager",
        ),
    )
    write(HOME / "latest.json", dict(task=str(task)))
    print(
        "독립 실행 제출 완료. 터미널 화면을 닫아도 작업은 유지됩니다. PC 종료/로그아웃은 피하세요."
    )


def service_memory(config):
    if not config.get("use_available_ram"):
        return dict(high_bytes=8_000_000_000, max_bytes=9_000_000_000)
    if (
        config["mode"]
        not in {
            "test",
            "collect",
            "resume_preprocess",
            "finish_preprocess",
            "finish_audit",
            "merge_prepared",
            "recover_capsules",
        }
        or config.get("min_available_ram_gb") != 1
    ):
        raise ValueError("가용 RAM 확장은 학습 전 작업에서1GB를 남겨야 합니다")
    available = collector().memory_snapshot()["available_bytes"]
    cap = available - 1_000_000_000
    if cap <= 0:
        raise ValueError("가용 RAM이1GB 이하라 시험을 시작하지 않습니다")
    return dict(
        high_bytes=cap,
        max_bytes=cap,
        available_bytes=available,
        reserve_bytes=1_000_000_000,
        observed_utc=now(),
        swap_max_bytes=free_swap_bytes() if config.get("use_swap") else 0,
    )


def free_swap_bytes():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["SwapFree"].split()[0]) * 1024


def choose(prompt, options, default):
    while True:
        try:
            value = input(
                f"{prompt} [{'/'.join(map(str, options))}] (기본 {default}): "
            ).strip().lower() or str(default)
        except UnicodeDecodeError:
            print("터미널 입력 인코딩 오류입니다. 영문 입력으로 전환한 뒤 다시 입력하세요.")
            continue
        if value in {str(x) for x in options}:
            return value
        print("목록에 있는 값을 선택하세요.")


def input_integer(prompt, default, maximum=None):
    while True:
        try:
            value = positive_int(input(f"{prompt} (기본 {default}): ").strip() or str(default))
            if maximum is not None and value > maximum:
                raise ValueError(f"현재 PX4 인스턴스/포트 계약은 1..{maximum}입니다")
            return value
        except UnicodeDecodeError:
            print("터미널 입력 인코딩 오류: 숫자로 다시 입력하세요.")
        except (ValueError, argparse.ArgumentTypeError) as exc:
            print(str(exc))


def existing_task():
    tasks = sorted(
        (HOME / "tasks").glob("*/config.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not tasks:
        print("기존 작업이 없습니다.")
        return
    for i, path in enumerate(tasks, 1):
        state_path = path.parent / "operator/state.json"
        state = read(state_path).get("status", "UNKNOWN") if state_path.exists() else "NOT_STARTED"
        print(f"{i}. {path.parent.name} · {state}")
    while True:
        index = input_integer("조회할 작업 번호", 1)
        if index <= len(tasks):
            watch(tasks[index - 1].parent)
            return
        print("표시된 작업 번호를 입력하세요.")


def menu():
    print("드론 · 학습 전 통합 CLI (기본 기체 X500)\n1. 테스트  2. 새 수집→전처리  3. 기존 작업 화면  0. 종료")
    choice = choose("작업", (1, 2, 3, 0), 2)
    if choice == "0":
        return
    if choice == "3":
        existing_task()
        return
    args = parser().parse_args(["init", "--name", time.strftime("local-%Y%m%d-%H%M%S")])
    args.mode = "test" if choice == "1" else "collect"
    if args.mode == "test":
        args.levels = [input_integer("시험 병렬 수 (1..16 정수)", 2, MAX_FLIGHT_WORKERS)]
        args.test_flights = True
        print("새120초 비행 → 검증·압축 → 추출·전처리 → 집계·독립 검산. 학습에 투입하지 않습니다.")
    else:
        args.workers = input_integer("전처리 병렬 수", 4)
        args.io_workers = input_integer("압축 해제·검증 병렬 수", 4)
        args.audit_workers = input_integer("품질 검산 병렬 수", 4)
        print("300run: 학습180·검증26·보정26·정상 개발평가26+대조6·이상36")
        print("정상258run 경로: 기본103 / 복합77 / 제한된 부드러운 난수78 · 기본2배속")
        print("조건은 실행 전 manifest에 고정 · 새 조건의 실제 비행 통과는 수집 시 검증")
        print("같은 비행 길이끼리 묶음 / 비행과 이전 묶음 후처리 병행 / 원시 최대2묶음 + 실패 증거")
        args.collection_sets = input_integer("수집 배율 (1=300run)", 1)
        args.flight_workers = input_integer("비행 병렬 수 (1..16 정수)", 16, MAX_FLIGHT_WORKERS)
        while True:
            args.export_workers = input_integer("검증·압축 병렬 수", min(4, args.flight_workers))
            if args.export_workers <= args.flight_workers:
                break
            print("검증·압축 병렬 수는 비행 병렬 수 이하여야 합니다.")
        args.gates = choose("사전 비행·내구 게이트 실행", ("y", "n"), "n") == "y"
        args.seed_base = input_seed()
    print("예상·고정 RSS 제한 해제 · 가용 RAM1GB 보호 · 제출 시 비어 있는 스왑 사용 허용")
    print("개별 run 최대 3회 시도 · 실패 증거 보존 후 다음 run · 불완전 데이터셋 인수 금지")
    while True:
        try:
            raw = input(f"최종 데이터 저장 폴더 (기본 {selected_root()}): ").strip()
            args.storage_root = guard(raw or selected_root())
            break
        except (ValueError, UnicodeDecodeError) as exc:
            print(f"저장 경로 확인: {exc}")
    print(f"저장: {args.storage_root} · 학습 없음")
    if choose("선택한 유한 작업 시작", ("y", "n"), "n") != "y":
        return
    task = create(args)
    start(task)
    watch(task)


def positive_int(value):
    try:
        number = int(value)
        if number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요")


def input_seed():
    while True:
        try:
            value = input("seed 직접 입력 (빈칸=매번 새 난수): ").strip()
            if not value:
                return None
            seed = int(value)
            if 0 <= seed <= 2**31 - 20_000:
                return seed
            print("seed는0..2^31-20000 정수입니다. 배율에 따라 추가 여유가 필요합니다.")
        except (ValueError, UnicodeDecodeError):
            print("정수 seed를 입력하거나 빈칸으로 두세요.")


class Levels(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1 and values[0].lower() == "all":
            levels = list(CHOICES)
        else:
            try:
                levels = [int(value) for value in values]
            except ValueError:
                parser.error("--levels: 1..16 정수를 입력하세요")
            if levels != sorted(set(levels)) or any(
                not 1 <= n <= MAX_FLIGHT_WORKERS for n in levels
            ):
                parser.error("--levels: 중복 없는 오름차순1..16 정수가 필요합니다")
        setattr(namespace, self.dest, levels)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action")
    init = sub.add_parser("init", help="실행 없이 설정 작성")
    init.add_argument("--name", required=True)
    init.add_argument("--mode", choices=("test", "collect"), default="test")
    for key, default in (
        ("workers", 4),
        ("io-workers", 4),
        ("audit-workers", 4),
        ("flight-workers", 16),
        ("export-workers", 4),
    ):
        init.add_argument("--" + key, type=positive_int, default=default)
    init.add_argument(
        "--storage-root",
        type=Path,
        default=selected_root(),
        help="최종·중간 데이터 저장 루트; 기본 프로젝트/Data",
    )
    init.add_argument("--gates", action="store_true")
    init.add_argument(
        "--collection-profile",
        choices=("balanced300", "diverse104", "train104"),
        default="balanced300",
    )
    init.add_argument(
        "--storage-profile",
        choices=("stream_compact", "compact", "standard"),
        default="stream_compact",
        help="stream_compact: 유한 묶음 처리 후 검증된 새 성공 원시 정리; compact: 원시 유지; standard: 압축 추가 안 함",
    )
    init.add_argument(
        "--collection-sets",
        type=positive_int,
        default=1,
        help="기본300run 배율; 전체 통합 데이터셋·공통 train scaler로 전처리·검산",
    )
    init.set_defaults(test_flights=False, flights_only=False)
    init.add_argument("--no-swap", dest="use_swap", action="store_false", default=True)
    init.add_argument(
        "--ignore-memory-estimate",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    init.add_argument(
        "--min-available-ram-gb", type=int, choices=(1, 4), default=1, help=argparse.SUPPRESS
    )
    init.add_argument(
        "--use-available-ram",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    init.add_argument("--levels", nargs="+", action=Levels, default=[2], metavar="N|all")
    init.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="생략하면 매번 새 보안 난수 seed; 지정하면 구성 재현",
    )
    resume = sub.add_parser(
        "resume-preprocess", help="완료된 수집만 새 작업에서 전처리; 재수집 없음"
    )
    resume.add_argument("source", type=Path)
    resume.add_argument("--name", required=True)
    capsules = sub.add_parser(
        "recover-capsules", help="완료된 run 캡슐을 새 작업에서 복원·통합·검산; 재수집·학습 없음"
    )
    capsules.add_argument("source", type=Path)
    capsules.add_argument("--name", required=True)
    partial = sub.add_parser(
        "finish-preprocess", help="실패1개만 처리 후 기존 성공 결과와 집계·검산"
    )
    partial.add_argument("source", type=Path)
    partial.add_argument("--name", required=True)
    audit = sub.add_parser("finish-audit", help="실패1개 검산 후 기존 결과와 전체 검산")
    audit.add_argument("source", type=Path)
    audit.add_argument("--name", required=True)
    speed = sub.add_parser(
        "speed-test", help="2x twice,4x,8x finite matched-seed comparison; no training"
    )
    speed.add_argument("--name", required=True)
    speed.add_argument("--seed-base", type=int, default=int(time.time()) % 1_000_000_000)
    speed.add_argument(
        "--baseline-review",
        type=Path,
        help="Run one speed candidate using its preserved2x reference",
    )
    speed.add_argument("--candidate-factor", type=int, choices=(3, 8), default=8)
    speed.add_argument(
        "--previous-candidate",
        type=Path,
        action="append",
        default=[],
        help="Stopped candidate task sharing the same seed; identity-checked evidence only",
    )
    for action in ("start", "status", "watch", "_worker"):
        command = sub.add_parser(action)
        command.add_argument("task", nargs="?", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action is None:
        menu()
    elif args.action == "init":
        create(args)
    elif args.action == "speed-test":
        if (args.candidate_factor != 8 or args.previous_candidate) and not args.baseline_review:
            raise ValueError("Single candidate requires --baseline-review")
        baseline = None
        if args.baseline_review:
            baseline = read(args.baseline_review)
            args.seed_base = read(Path(baseline["source_task"]) / "config.json")["seed_base"]
        fresh = parser().parse_args(
            [
                "init",
                "--name",
                args.name,
                "--mode",
                "test",
                "--levels",
                "1",
                "--flight-workers",
                "1",
                "--export-workers",
                "1",
                "--workers",
                "1",
                "--io-workers",
                "1",
                "--audit-workers",
                "1",
                "--seed-base",
                str(args.seed_base),
            ]
        )
        task = create(fresh)
        config = read(task / "config.json")
        config.update(
            speed_test=dict(factors=[2, 2, 4, 8], protocol="information-v2", mission_seconds=120),
            storage_profile="standard",
            unit_attempts_max=1,
            continue_after_case_failure=False,
        )
        if baseline:
            source_config = Path(baseline["source_task"]) / "config.json"
            config["speed_test"].update(
                factors=[args.candidate_factor],
                candidate_only=True,
                continue_after_lower_speed_failure=True,
                baseline_review=dict(
                    path=str(args.baseline_review.resolve()), sha256=sha(args.baseline_review)
                ),
                baseline_config=dict(path=str(source_config), sha256=sha(source_config)),
                previous_candidates=[
                    dict(
                        config=dict(
                            path=str(p.resolve() / "config.json"), sha256=sha(p / "config.json")
                        ),
                        results=dict(
                            path=str(STORE / "speed-tests" / p.name / "results.json"),
                            sha256=sha(STORE / "speed-tests" / p.name / "results.json"),
                        ),
                    )
                    for p in args.previous_candidate
                ],
            )
            collector().candidate_baseline(config)
            for key in ("baseline_review", "baseline_config"):
                ref = config["speed_test"][key]
                config["source_sha256"][ref["path"]] = ref["sha256"]
            for previous in config["speed_test"]["previous_candidates"]:
                for ref in previous.values():
                    config["source_sha256"][ref["path"]] = ref["sha256"]
        write(task / "config.json", config)
        (task / "config.sha256").write_text(sha(task / "config.json") + "\n")
        start(task)
    elif args.action == "resume-preprocess":
        from .recovery import create_recovery

        task = create_recovery(args.source, args.name)
        start(task)
    elif args.action == "recover-capsules":
        from .capsule_recovery import create_recovery

        task = create_recovery(args.source, args.name)
        start(task)
    elif args.action == "finish-preprocess":
        from .partial import create_partial

        task = create_partial(args.source, args.name)
        start(task)
    elif args.action == "finish-audit":
        from .audit_recovery import create_audit_recovery

        task = create_audit_recovery(args.source, args.name)
        start(task)
    else:
        task = args.task or Path(read(HOME / "latest.json")["task"])
        if args.action == "start":
            start(task)
        elif args.action in {"status", "watch"}:
            watch(task, once=args.action == "status")
        else:
            from .runner import worker

            worker(task)


if __name__ == "__main__":
    main()
