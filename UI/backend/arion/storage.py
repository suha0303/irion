from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np

from .telemetry import TelemetryRun


class Store:
    def __init__(self, root: Path):
        self.root = root
        (root / "logs").mkdir(parents=True, exist_ok=True)
        self.database = root / "arion.sqlite3"
        with closing(self.connect()) as db, db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, sha TEXT NOT NULL, name TEXT NOT NULL,
                path TEXT NOT NULL, created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS diagnostics(run_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS detections(run_id TEXT NOT NULL, event_id TEXT NOT NULL,
                model_version TEXT NOT NULL, origin TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(run_id,event_id,model_version,origin));
            """)

    def connect(self):
        db = sqlite3.connect(self.database)
        db.row_factory = sqlite3.Row
        return db

    def import_bytes(self, content: bytes, name: str):
        sha = hashlib.sha256(content).hexdigest()
        path = self.root / "logs" / (sha + ".jsonl")
        newly_created = not path.exists()
        if newly_created:
            path.write_bytes(content)
        try:
            run = TelemetryRun(path, Path(name).name)
        except Exception:
            if newly_created:
                path.unlink(missing_ok=True)
            raise
        with closing(self.connect()) as db, db:
            db.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?)", (run.run_id, sha, run.name, str(path), datetime.now(timezone.utc).isoformat()))
        return self.get(run.run_id)

    @lru_cache(maxsize=6)
    def get(self, run_id):
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(run_id)
        return TelemetryRun(Path(row["path"]), row["name"])

    def list(self):
        with closing(self.connect()) as db:
            rows = db.execute("SELECT id,created_at FROM runs ORDER BY created_at DESC").fetchall()
        return [dict(self.get(row["id"]).manifest(), created_at=row["created_at"]) for row in rows]

    def npz_report(self, run_id):
        with closing(self.connect()) as db:
            row = db.execute("SELECT payload FROM diagnostics WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def inspect_npz(self, run_id, path):
        run = self.get(run_id)
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist()) > 128 * 1024 * 1024:
                raise ValueError("NPZ 압축 해제 크기는 128 MiB 이하여야 합니다.")
        with np.load(path, allow_pickle=False) as data:
            required = {"metadata", "X", "mask", "tick_ns", "epoch", "window_end", "valid_end", "invalid_end", "model_ready_end", "age_ns"}
            if not required.issubset(data.files):
                raise ValueError("지원되는 전처리 NPZ 필드가 부족합니다.")
            metadata = json.loads(str(data["metadata"].item()))
            if not isinstance(metadata, dict):
                raise ValueError("NPZ metadata는 JSON 객체여야 합니다.")
            x, mask = data["X"], data["mask"]
            if x.ndim != 2 or x.dtype.kind not in "fiu" or mask.shape != x.shape or mask.dtype.kind != "b":
                raise ValueError("X/mask 크기 또는 dtype이 올바르지 않습니다.")
            features = metadata.get("features", [])
            if not isinstance(features, list) or any(not isinstance(f, dict) or not isinstance(f.get('name'), str) or not isinstance(f.get('unit'), str) for f in features):
                raise ValueError("특징 목록에 name과 unit이 필요합니다.")
            if len(features) != x.shape[1] or data["tick_ns"].shape != (x.shape[0],):
                raise ValueError("특징 목록·시각 길이와 X 크기가 일치하지 않습니다.")
            for key in ('window_end', 'valid_end', 'invalid_end', 'model_ready_end'):
                indices = data[key]
                if indices.ndim != 1 or (indices.size and (indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= x.shape[0]))):
                    raise ValueError(f"{key}의 행 인덱스가 유효하지 않습니다.")
            if not set(data['valid_end']).issubset(set(data['window_end'])) or not set(data['model_ready_end']).issubset(set(data['valid_end'])):
                raise ValueError("NPZ 준비 상태 인덱스의 관계가 올바르지 않습니다.")
            input_hashes = [item.get("sha256") for item in metadata.get("input_files", [])]
            report = {"filename": "real-training.npz" if not hasattr(path, "name") else Path(str(path.name)).name,
                      "shape": list(x.shape), "windows": int(data["window_end"].size),
                      "data_ready_windows": int(data["valid_end"].size), "model_ready_windows": int(data["model_ready_end"].size),
                      "invalid_windows": int(data["invalid_end"].size), "missing_cells": int((~mask).sum()),
                      "provenance_matches": len(input_hashes) == 1 and input_hashes[0] == run.sha256,
                      "input_hashes": input_hashes, "current_input_sha256": run.sha256,
                      "schema_version": metadata.get("schema_version"), "normalization": metadata.get("normalization"),
                      "features": features, "config": metadata.get("config", {}),
                      "note": "독립 전처리 산출물 검사입니다. 해시 불일치 시 선택 비행의 입력으로 결합하지 않습니다."}
        with closing(self.connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO diagnostics VALUES (?,?)", (run_id, json.dumps(report, ensure_ascii=False)))
        return report

    def save_events(self, run_id, events):
        with closing(self.connect()) as db, db:
            for event in events:
                db.execute("INSERT OR REPLACE INTO detections VALUES (?,?,?,?,?)", (run_id, event["event_id"], event["model_version"], event["origin"], json.dumps(event, ensure_ascii=False, allow_nan=False)))

    def events(self, run_id, origin="model"):
        with closing(self.connect()) as db:
            rows = db.execute("SELECT payload FROM detections WHERE run_id=? AND origin=?", (run_id, origin)).fetchall()
        return sorted([json.loads(row[0]) for row in rows], key=lambda e: (e["decided_at_ms"], e["event_id"]))
