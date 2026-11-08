"""Read-only terminal progress; no LLM, submissions or automatic retries."""

import argparse
import json
import time
from pathlib import Path


def show(root):
    root = Path(root)
    p = root / "state.json"
    if not p.exists():
        print("할당·인수 준비 중:", root)
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    print("X500 · 16모델 LR / 본학습")
    print(
        d["status"], "|", d["stage"], "|", d.get("completed_models", 0), "/16"
    )
    for e in d.get("active", []):
        q = root / e["model"] / d["stage"] / "progress.json"
        s = json.loads(q.read_text(encoding="utf-8")) if q.exists() else {}
        print(
            e["node"],
            e["model"],
            s.get("stage", "시작 중"),
            (
                f"epoch {s.get('epoch', '-')}/60 batch "
                f"{s.get('batch', '-')}/{s.get('batches', '-')}"
            ),
            (
                f"LR {s.get('lr', s.get('candidate', '-'))} GPU "
                f"{s.get('gpu_allocated_bytes', 0) / 1e9:.2f}GB"
            ),
        )
    if d.get("reason"):
        print(d["reason"])
    print("q 대신 Ctrl+C: 화면만 종료 · 작업 유지 · LLM 호출 없음")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("root")
    cli_parser.add_argument("--watch", action="store_true")
    a = cli_parser.parse_args()
    try:
        while True:
            if a.watch:
                print("\033[2J\033[H", end="")
            show(a.root)
            if not a.watch:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
