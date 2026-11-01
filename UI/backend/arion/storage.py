"""File-backed flight and event catalog. Original source files are never rewritten."""
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from threading import RLock
import zipfile
import numpy as np
from .telemetry import TelemetryRun
from .file_io import atomic_json, read_json, append_json, hash_file

class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.catalog_path = self.root / 'catalog.json'
        self.catalog = read_json(self.catalog_path, {'schema_version':'arion-files-v1','runs':{},'sessions':{}})
        self._event_cache = {}
        self._migrate_sqlite()

    def _catalog_save(self): atomic_json(self.catalog_path, self.catalog)

    def _migrate_sqlite(self):
        marker = self.root/'backups'/'file-migration.json'
        database = self.root/'arion.sqlite3'
        if marker.exists() or not database.exists(): return
        # Only this migration imports sqlite3. The source is opened read-only.
        import sqlite3
        from contextlib import closing
        backup=self.root/'backups'/'before-file-storage.sqlite3'; backup.parent.mkdir(exist_ok=True)
        before=hash_file(database)
        with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)) as db:
            db.row_factory=sqlite3.Row
            if not backup.exists():
                with closing(sqlite3.connect(backup)) as destination: db.backup(destination)
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            counts={}
            for table in ('runs','sessions'):
                if table not in tables: continue
                rows=[dict(r) for r in db.execute('SELECT * FROM '+table)]
                counts[table]=len(rows)
                for row in rows: self.catalog[table].setdefault(row['id'], row)
            for table in ('profiles','providers'):
                existing=read_json(self.root/(table+'.json'), {})
                if table in tables:
                    for row in db.execute('SELECT id,payload FROM '+table): existing.setdefault(row['id'], json.loads(row['payload']))
                    counts[table]=len(existing)
                atomic_json(self.root/(table+'.json'),existing)
            for table in ('diagnostics','detections','analysis_results'):
                if table not in tables: continue
                rows=db.execute('SELECT * FROM '+table).fetchall();counts[table]=len(rows)
                for row in rows:
                    payload=json.loads(row['payload'])
                    if table=='diagnostics': atomic_json(self.root/'records'/row['run_id']/'diagnostics.json',payload)
                    else: self._save_rows(payload['run_id'],[payload],'events' if table=='detections' else 'analysis')
        # Legacy rows remain legacy; never classify them as LSTM or Transformer.
        for run_id in self.catalog['runs']:
            for event in self.events(run_id,'fixture')+self.events(run_id):
                e=dict(event,provider_id='legacy',kind='detector',input_ref={'kind':'file','sha256':event.get('input_sha256')})
                self._save_rows(run_id,[e],'analysis')
        self._catalog_save()
        atomic_json(marker,{'schema_version':'arion-file-migration-v1','source_sha256':before,'backup':str(backup),'counts':counts})

    def import_bytes(self, content: bytes, name: str):
        if Path(name).suffix.lower()=='.npz': return self.import_npz(content,name)
        sha=hashlib.sha256(content).hexdigest();path=self.root/'logs'/(sha+'.jsonl')
        with self.lock:
            newly_created=not path.exists()
            if newly_created: path.write_bytes(content)
            try: run=TelemetryRun(path,Path(name).name)
            except Exception:
                if newly_created:path.unlink(missing_ok=True)
                raise
            self.catalog['runs'].setdefault(run.run_id, {'id':run.run_id,'sha':sha,'name':run.name,'path':str(path),'kind':'jsonl','created_at':datetime.now(timezone.utc).isoformat()})
            self._catalog_save()
        return self.get(run.run_id)

    @lru_cache(maxsize=6)
    def get(self, run_id):
        row=self.catalog['runs'].get(run_id)
        if not row:raise KeyError(run_id)
        return TelemetryRun(Path(row['path']),row['name'])

    def list(self):
        return [dict(self.get(r['id']).manifest(),created_at=r['created_at']) for r in sorted(self.catalog['runs'].values(),key=lambda x:x['created_at'],reverse=True)]

    def npz_report(self, run_id):
        return read_json(self.root/'records'/run_id/'diagnostics.json')

    @staticmethod
    def _key(e):return (e.get('provider_id','legacy'),e['event_id'],e['model_version'],e['origin'])

    def _rows(self,run_id,kind):
        path=self.root/'records'/run_id/(kind+'.jsonl')
        stamp=(path.stat().st_mtime_ns,path.stat().st_size) if path.exists() else (0,0)
        cache=self._event_cache.get((run_id,kind))
        if cache and cache[0]==stamp:return cache[1]
        rows={}
        if path.exists():
            with path.open() as f:
                for line in f:
                    try:e=json.loads(line);rows[self._key(e)]=e
                    except (ValueError,KeyError):continue
        if len(self._event_cache)>12:self._event_cache.pop(next(iter(self._event_cache)))
        self._event_cache[run_id,kind]=(stamp,rows)
        return rows

    def _save_rows(self,run_id,events,kind):
        with self.lock:
            rows=self._rows(run_id,kind)
            for event in events:
                key=self._key(event)
                if rows.get(key)!=event:
                    append_json(self.root/'records'/run_id/(kind+'.jsonl'),event);rows[key]=event
            self._event_cache.pop((run_id,kind),None)

    def save_events(self, run_id, events): self._save_rows(run_id,events,'events')

    def events(self, run_id, origin='model'):
        with self.lock:
            return sorted([e for e in self._rows(run_id,'events').values() if e['origin']==origin],key=lambda e:(e['decided_at_ms'],e['event_id']))

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
        atomic_json(self.root / "records" / run_id / "diagnostics.json", report)
        return report
