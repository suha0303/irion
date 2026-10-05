#!/usr/bin/env python3
"""Read-only progress display for the finite long-run collector. No LLM calls."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

DEFAULT_PLAN = Path(__file__).resolve().parent / "experiments/ROUND_001/long-runs-01/plan.json"
ARCHIVED = {"ARCHIVED_PENDING_QUALITY"}
TERMINAL = {
    "FAILED_DRAINED_STOPPED",
    "COLLECTED_PENDING_QUALITY_STOPPED",
    "COLLECTED_WITH_ERRORS_STOPPED",
}


def load(path):
    return json.loads(Path(path).read_text())


def estimate_eta(plan, ledger, resources, cases):
    completed = [row for row in cases.values() if row.get("status") in ARCHIVED]
    elapsed = max(ledger.get("elapsed_s", 0), resources.get("elapsed_s", 0))
    completed_work = sum(
        row.get("case", {}).get("mission_seconds", plan.get("mission_seconds", 0))
        for row in completed
    )
    total_work = ledger.get("planned_mission_seconds_total")
    result = {"seconds": None, "basis": "완료 실측 부족", "confidence": "unavailable"}
    if ledger.get("status") in {"FAILED_DRAINED_STOPPED", "COLLECTED_WITH_ERRORS_STOPPED"}:
        result["basis"] = "실패 중지: 완료 ETA를 계산하지 않음"
    elif ledger.get("status") == "COLLECTED_PENDING_QUALITY_STOPPED":
        result.update(seconds=0, basis="수집 종료 기록", confidence="observed")
    elif (
        len(completed) >= plan.get("concurrency", 2)
        and completed_work > 0
        and total_work
        and elapsed > 0
    ):
        result.update(
            seconds=max(0, total_work - completed_work) * elapsed / completed_work,
            basis=f"완료 {len(completed)}run / 명목임무 {completed_work:.0f}s 처리에 wall {elapsed:.0f}s; 병렬·검증·압축 포함 처리율",
            confidence="low; 조건·길이·동시성·저장속도 변화에 민감, 보장/신뢰구간 아님",
        )
    return result


def snapshot(plan_path):
    plan = load(plan_path)
    execution = Path(plan["execution_dir"])
    ledger_path = execution / "ledger.json"
    if not ledger_path.exists():
        return {
            "campaign": plan["campaign_id"],
            "status": "NOT_STARTED_OR_NO_LEDGER",
            "total": plan.get("attempts_max", 0),
            "archived": 0,
            "percent": None,
            "workers": [],
            "ledger": str(ledger_path),
            "reason": [],
            "updated_unix": None,
        }
    ledger = load(ledger_path)
    resource_path = execution / "resource-progress.json"
    resources = load(resource_path) if resource_path.exists() else ledger.get("resources", {})
    cases = ledger.get("cases", {})
    done = sum(row.get("status") in ARCHIVED for row in cases.values())
    total = plan.get("attempts_max", 0)
    workers = []
    for case_id, row in cases.items():
        if row.get("status") in ARCHIVED:
            continue
        # Case IDs are untrusted even in a status file. Never read outside execution.
        if not case_id or Path(case_id).name != case_id or case_id in {".", ".."}:
            continue
        worker = {
            "case": case_id,
            "status": row.get("status", "UNKNOWN"),
            "instance": row.get("lane", {}).get("instance"),
            "error": row.get("error"),
            "scenario": row.get("case", {}).get("scenario", "미확인"),
            "target_mission_seconds": row.get("case", {}).get(
                "mission_seconds", plan.get("mission_seconds")
            ),
            "flight_wall_s": row.get("timings_s", {}).get("flight_command"),
            "assessment": row.get("assessment"),
        }
        attempt = row.get("attempt", 1)
        if type(attempt) is not int or not 1 <= attempt <= 3:
            continue
        attempt_dir = execution / case_id
        if attempt > 1:
            attempt_dir /= f"attempt-{attempt:02d}"
        worker.update(attempt=attempt, attempts_max=row.get("attempts_max", 1))
        flight_path = attempt_dir / "flight-progress.json"
        worker["flight"] = load(flight_path) if flight_path.exists() else {}
        flight = worker["flight"]
        if flight.get("observed_simulation_elapsed_s") is not None:
            try:
                producer_time = datetime.fromisoformat(flight["producer_progress_updated_utc"])
                fresh = (
                    producer_time.tzinfo is not None
                    and 0 <= time.time() - producer_time.timestamp() <= 10
                )
            except (KeyError, ValueError, TypeError):
                fresh = False
            if not fresh:
                flight.update(
                    observed_simulation_elapsed_s=None,
                    live_rtf=None,
                    simulation_clock_basis="Producer progress stale or timestamp invalid",
                )
        progress = []
        for name in ("archive-progress.json", "archive-verify-progress.json"):
            path = attempt_dir / name
            if path.exists():
                item = load(path)
                progress.append(
                    {k: item.get(k) for k in ("stage", "status", "bytes_processed", "elapsed_s")}
                )
        worker["progress"] = progress
        workers.append(worker)
    return {
        "campaign": plan["campaign_id"],
        "status": ledger.get("status", "UNKNOWN"),
        "total": total,
        "archived": done,
        "failed": sum(row.get("status") == "FAILED_PRESERVED" for row in cases.values()),
        "not_started": len(ledger.get("not_started", [])),
        "percent": min(100.0, 100 * done / total) if total else None,
        "workers": workers,
        "ledger": str(ledger_path),
        "phase": plan.get("phase", "collection"),
        "resources": resources,
        "eta": estimate_eta(plan, ledger, resources, cases),
        "measurements": {
            key: row["assessment"] for key, row in cases.items() if "assessment" in row
        },
        "reason": ledger.get("stop_reason", []),
        "updated_unix": ledger_path.stat().st_mtime,
    }


def render(state):
    percent = "미정" if state["percent"] is None else f"{state['percent']:.1f}%"
    lines = [
        f"X500 {state['campaign']}",
        f"압축·검증 완료 {percent}  {state['archived']}/{state['total']}  {state['status']}",
    ]
    if state["updated_unix"] is not None:
        age = max(0, time.time() - state["updated_unix"])
        lines.append(f"ledger 갱신 후 {age:.0f}초 — 파일 기록 기준, 프로세스 생존은 미확인")
    resources = state.get("resources", {})
    if resources:
        for key, label in (
            ("available_bytes", "MemAvailable"),
            ("owned_tree_rss_bytes", "process tree RSS"),
            ("cgroup_total_rss_bytes", "cgroup total RSS"),
            ("cgroup_memory_current_bytes", "cgroup memory.current"),
            ("local_free_bytes", "로컬 여유"),
            ("external_free_bytes", "외부 여유"),
        ):
            value = resources.get(key)
            lines.append(
                f"{label}: {value / 1e9:.2f} GB" if value is not None else f"{label}: 미측정"
            )
    for case_id, result in state.get("measurements", {}).items():
        lines.append(
            f"{case_id}: 유효관측 {result['observed_valid_seconds']:.1f}s / "
            f"실제 wall {result['flight_command_wall_s']:.1f}s / 유효 RTF {result['effective_rtf']:.3f}"
        )
    if state.get("phase") in {"benchmark", "endurance"}:
        lines.append("사전 gate 자료: 학습·최종 시험 데이터에 포함하지 않음")
    eta = state.get(
        "eta", {"seconds": None, "basis": "완료 실측 부족", "confidence": "unavailable"}
    )
    lines.append(
        "남은 예상시간: "
        + (f"{eta['seconds'] / 60:.1f}분" if eta["seconds"] is not None else "미확인")
    )
    lines.append(f"  근거: {eta['basis']}; 불확실성: {eta['confidence']}")
    for worker in state["workers"]:
        lines.append(
            f"instance {worker['instance']}: {worker['case']} / {worker['scenario']} / {worker['status']}"
        )
        flight = worker["flight"]
        simulation_elapsed = flight.get("observed_simulation_elapsed_s")
        live_rtf = flight.get("live_rtf")
        wall = (
            worker.get("flight_wall_s")
            if worker.get("flight_wall_s") is not None
            else flight.get("wall_elapsed_s")
        )
        lines.append(
            f"  비행 simulation 경과 {simulation_elapsed if simulation_elapsed is not None else '미확인'} / 목표 {worker['target_mission_seconds']}s; "
            f"wall {round(wall, 1) if wall is not None else '미확인'}s (기록값); 실시간 RTF {live_rtf if live_rtf is not None else '미확인'}"
        )
        if flight.get("producer_run_id"):
            lines.append(
                f"  관측 run {flight['producer_run_id']} / producer 갱신 {flight.get('producer_progress_updated_utc')}"
            )
        for progress in worker["progress"]:
            count = progress.get("bytes_processed") or 0
            lines.append(f"  {progress['stage']}: {count / 1e6:.1f} MB 처리 / {progress['status']}")
        if worker["status"] == "VERIFYING":
            lines.append("  producer verify byte 진행: 미확인 (별도 byte sidecar 없음)")
        if worker.get("error"):
            lines.append("  " + worker["error"])
    for reason in state["reason"]:
        lines.append("중지 사유: " + reason)
    lines += [
        "상태: " + state["ledger"],
        "100%는 archive 검증 완료입니다. 품질 승인·전처리·학습 완료가 아닙니다.",
        "화면만 갱신합니다. Ctrl+C는 화면만 닫습니다. LLM 호출 없음.",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "watch"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("interval must be at least one second")
    try:
        while True:
            try:
                state = snapshot(args.plan)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"UNKNOWN: {type(exc).__name__}: {exc}")
                return 2
            if args.command == "watch" and not args.json:
                print("\033[2J\033[H", end="")
            print(json.dumps(state, ensure_ascii=False) if args.json else render(state), flush=True)
            if (
                args.command == "status"
                or state["status"] in TERMINAL
                or state["status"] == "NOT_STARTED_OR_NO_LEDGER"
            ):
                return 0 if state["status"] == "COLLECTED_PENDING_QUALITY_STOPPED" else 2
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
