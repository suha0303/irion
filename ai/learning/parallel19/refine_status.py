"""Read-only progress for both adopted and newly dispatched GPU workers."""

import argparse
import json
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def visible_workers(state, plan):
    active = [{**row, "source": "현재"} for row in state.get("active", [])]
    seen = {row["pid"] for row in active}
    completed = set(state.get("adopted_complete", []))
    if state.get("status") == "RUNNING" and plan.get(
        "adopted_allocation"
    ) == state.get("allocation", {}).get("leader"):
        for row in plan.get("adopted_steps", []):
            if row["pid"] not in completed and row["pid"] not in seen:
                active.append(
                    {**row, "source": "인수", "kind": row.get("kind", "search")}
                )
                seen.add(row["pid"])
    return sorted(
        active, key=lambda r: (r["node"], r["source"], r["model"], r["pid"])
    )


def show(base, details=False):
    root = Path(base)
    path = root / "output/state.json"
    print("X500 · 학습 → 보정·개발 평가 → 다음 LR 비교 → 최종 검증")
    if not path.exists():
        print("실행 준비 중:", root)
        return
    state = read(path)
    plan = (
        read(root / "output/plan.json")
        if (root / "output/plan.json").exists()
        else {}
    )
    workers = visible_workers(state, plan)
    adopted = sum(r["source"] == "인수" for r in workers)
    print(
        state["status"],
        "|",
        state["stage"],
        "| job",
        state.get("allocation", {}).get("leader", "-"),
    )
    print(
        (
            f"실행 작업 {len(workers)}/9 = 인수한 학습 {adopted} + 현재 관리자 작업 "
            f"{len(workers) - adopted} · 기록 기준"
        )
    )
    print(
        "DIS05",
        sum(r["node"] == "DIS05" for r in workers),
        "/5 · DIS06",
        sum(r["node"] == "DIS06" for r in workers),
        "/4 · 대기",
        state.get("queued", 0),
    )
    print(
        "개발 평가",
        state.get("development_completed", 0),
        "/",
        state.get("development_total", 0),
        "후보 완료",
    )
    print(
        "최종 평가",
        state.get("final_completed", 0),
        "/16종 완료 · 모델별 LR 탐색 종료 후 순차 배정",
    )
    print("─" * 100)
    for row in workers:
        p = Path(row["output"]) / "progress.json"
        v = read(p) if p.exists() else {}
        phase = v.get("stage", "시작 중")
        if "epoch" in v:
            progress = (
                f"epoch {v['epoch']}/60 · batch "
                f"{v.get('batch', '-')}/{v.get('batches', '-')}"
            )
        else:
            progress = (
                f"run {v.get('completed_runs', '-')}/{v.get('total_runs', '-')}"
            )
        print(
            (
                f"{row['node']} {row['source']} {row.get('kind', 'search')} "
                f"{row['model']:32} | {phase} | {progress}"
            )
        )
    print("─" * 100)
    finished = sum(
        m["status"].startswith("LOCAL_")
        for m in state.get("models", {}).values()
    )
    print(
        f"LR 탐색 종료 {finished}/16종 · 인수 = 전환 전부터 계속 학습하던 작업"
    )
    if details:
        for name, m in state.get("models", {}).items():
            if "best" not in m:
                print(name, m["status"])
                continue
            print(
                (
                    f"{name:34} LR {m['best']['lr']:.6g} val "
                    f"{m['best']['best_validation_loss']:.6g} 비교 "
                    f"{m['rounds']}차 {m['status']}"
                )
            )
    if state.get("reason"):
        print(state["reason"])
    print("Ctrl+C: 화면만 종료 · 작업 유지 · LLM 호출 없음")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--details", action="store_true")
    args = p.parse_args()
    try:
        while True:
            if args.watch:
                print("\033[2J\033[H", end="")
            show(args.root, args.details)
            if not args.watch:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
