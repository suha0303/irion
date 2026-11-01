from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import uuid

from .storage import Store
from .contracts import InputProfile, Provider, builtin_profiles
from .streaming import SessionRun

class StoreV2(Store):
    def __init__(self,root):
        self.active={}
        super().__init__(root)
        with closing(self.connect()) as db, db:
            if db.execute('PRAGMA user_version').fetchone()[0]<2:
                backup=self.root/'backups'/'before-v2.sqlite3';backup.parent.mkdir(exist_ok=True)
                if not backup.exists():
                    with closing(sqlite3.connect(backup)) as destination: db.backup(destination)
            db.executescript('''
                CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,path TEXT NOT NULL,sha TEXT,created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS providers(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS analysis_results(run_id TEXT NOT NULL,provider_id TEXT NOT NULL,event_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,origin TEXT NOT NULL,payload TEXT NOT NULL,
                    PRIMARY KEY(run_id,provider_id,event_id,model_version,origin));
            ''')
            for p in builtin_profiles():
                db.execute('INSERT OR IGNORE INTO profiles VALUES(?,?)',(p.profile_id,p.model_dump_json()))
                existing=json.loads(db.execute('SELECT payload FROM profiles WHERE id=?',(p.profile_id,)).fetchone()[0])
                if 'voltage_encoding' not in existing:
                    existing.update(voltage_encoding=p.voltage_encoding,missing_current_raw=p.missing_current_raw)
                    db.execute('UPDATE profiles SET payload=? WHERE id=?',(json.dumps(existing,ensure_ascii=False),p.profile_id))
            for p in [Provider(provider_id='lstm',name='LSTM'),Provider(provider_id='transformer',name='Transformer'),Provider(provider_id='legacy',name='이전 분석 결과')]:
                db.execute('INSERT OR IGNORE INTO providers VALUES(?,?)',(p.provider_id,p.model_dump_json()))
            if db.execute('PRAGMA user_version').fetchone()[0]<2:
                for row in db.execute('SELECT payload FROM detections').fetchall():
                    event=json.loads(row[0]);event['provider_id']='legacy';event['kind']='detector'
                    event['input_ref']={'kind':'file','sha256':event.get('input_sha256')}
                    db.execute('INSERT OR IGNORE INTO analysis_results VALUES(?,?,?,?,?,?)',
                        (event['run_id'],'legacy',event['event_id'],event['model_version'],event['origin'],json.dumps(event,ensure_ascii=False)))
                db.execute('PRAGMA user_version=2')

    def profiles(self):
        with closing(self.connect()) as db: return [json.loads(r[0]) for r in db.execute('SELECT payload FROM profiles ORDER BY id')]

    def profile(self, id):
        return InputProfile.model_validate(next((p for p in self.profiles() if p['profile_id']==id),{}))

    def register_profile(self,p):
        # Immutable profile IDs: a changed unit or period requires a new version.
        with closing(self.connect()) as db, db:
            row=db.execute('SELECT payload FROM profiles WHERE id=?',(p.profile_id,)).fetchone()
            if row and json.loads(row[0])!=p.model_dump(): raise ValueError('프로필 변경은 새 profile_id로 등록하세요.')
            db.execute('INSERT OR IGNORE INTO profiles VALUES(?,?)',(p.profile_id,p.model_dump_json()))
        return p.model_dump()

    def providers(self):
        with closing(self.connect()) as db: return [json.loads(r[0]) for r in db.execute('SELECT payload FROM providers ORDER BY id')]

    def provider(self,id):
        for p in self.providers():
            if p['provider_id']==id:return Provider.model_validate(p)
        raise KeyError(id)

    def register_provider(self,p):
        if p.provider_id=='legacy': raise ValueError('legacy는 이전 결과 보존용입니다.')
        with closing(self.connect()) as db, db:
            db.execute('INSERT OR REPLACE INTO providers VALUES(?,?)',(p.provider_id,p.model_dump_json()))
        return p.model_dump()

    def create_session(self,config):
        profile=self.profile(config.profile_id)
        if (config.kind=='normalized') != (profile.input_format=='normalized'): raise ValueError('입력 경로와 프로필 형식이 다릅니다.')
        id='live-'+uuid.uuid4().hex[:16]
        run=SessionRun(self.root/'sessions'/(id+'.jsonl'),config,profile,id)
        with closing(self.connect()) as db,db:
            db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(id,str(run.path),None,datetime.now(timezone.utc).isoformat()))
        self.active[id]=run
        return run

    def finish_session(self,id):
        run=self.active.pop(id)
        sha=run.finish()
        with closing(self.connect()) as db,db: db.execute('UPDATE sessions SET sha=? WHERE id=?',(sha,id))
        self.journal.cache_clear()
        return self.get(id)

    @lru_cache(maxsize=2)
    def journal(self,id,path): return SessionRun(Path(path))

    def get(self,id):
        if id in self.active:return self.active[id]
        with closing(self.connect()) as db: row=db.execute('SELECT path FROM sessions WHERE id=?',(id,)).fetchone()
        if row: return self.journal(id,row[0])
        return super().get(id)

    def list(self):
        rows=super().list()
        with closing(self.connect()) as db: sessions=db.execute('SELECT id,created_at FROM sessions ORDER BY created_at DESC').fetchall()
        for r in sessions:
            try: rows.append(dict(self.get(r['id']).manifest(),created_at=r['created_at']))
            except (OSError,ValueError): continue
        return sorted(rows,key=lambda r:r['created_at'],reverse=True)

    def save_analysis(self,events):
        with closing(self.connect()) as db,db:
            for e in events:
                db.execute('INSERT OR REPLACE INTO analysis_results VALUES(?,?,?,?,?,?)',
                    (e['run_id'],e['provider_id'],e['event_id'],e['model_version'],e['origin'],json.dumps(e,ensure_ascii=False,allow_nan=False)))

    def analysis(self,run_id,cursor=None):
        with closing(self.connect()) as db:
            where='run_id=?';params=[run_id]
            if cursor is not None:
                where+=" AND json_extract(payload,'$.decided_at_ms')<=?";params.append(cursor)
            return [json.loads(r[0]) for r in db.execute(
                f"SELECT payload FROM analysis_results WHERE {where} ORDER BY json_extract(payload,'$.decided_at_ms') DESC,rowid DESC LIMIT 2000",params)]
