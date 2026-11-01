import argparse
import json
from pathlib import Path
import unicodedata

from .storage import Store


def main():
    parser = argparse.ArgumentParser(description="ARION 로그 가져오기 / 첨부 자료 초기화")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--seed-downloads", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / ".data")
    args = parser.parse_args()
    if args.seed_downloads:
        expected = "비행수집_2026-10-05_185146_mavlink_raw.jsonl"
        args.log = next((p for p in (Path.home() / "Downloads").glob("*.jsonl") if unicodedata.normalize("NFC", p.name) == expected), None)
        candidate = Path.home() / "Downloads" / "real-training.npz"
        args.npz = candidate if candidate.exists() else None
    if not args.log:
        parser.error("--log 경로가 필요합니다. 화면에서도 JSONL 파일을 가져올 수 있습니다.")
    store = Store(args.data_dir)
    run = store.import_bytes(args.log.read_bytes(), unicodedata.normalize("NFC", args.log.name))
    report = {"manifest": run.manifest(), "npz": store.inspect_npz(run.run_id, args.npz) if args.npz else None}
    (args.data_dir / "baseline-inspection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"run_id": run.run_id, "records": len(run.samples), "excluded": len(run.issues),
                      "sha256": run.sha256, "report": str(args.data_dir / "baseline-inspection.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
