"""Profiles, sessions, provider results and exports stored as local files."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import uuid
import math
import os
import shutil
import tempfile
import numpy as np
from .storage import Store
from .contracts import InputProfile, SignalDefinition, Provider, SourceConfig, builtin_profiles
from .streaming import SessionRun, encode
from .file_io import atomic_json, read_json, hash_file
from .npz_archive import NpzArchive, load_npz, export_npz, row_arrays, atomic_npz, npz_metadata
from .npz_replay import NpzReplay
from .telemetry import LABELS
from copy import deepcopy

class StoreV2(Store):
    def __init__(self,root):
        self.active={};self.export_executor=None
        super().__init__(root)
        self._profiles=read_json(self.root/'profiles.json',{})
        for p in builtin_profiles():self._profiles.setdefault(p.profile_id,p.model_dump())
        self._profiles={k:InputProfile.model_validate(v).model_dump() for k,v in self._profiles.items()}
        atomic_json(self.root/'profiles.json',self._profiles)
        self._providers=read_json(self.root/'providers.json',{})
        for p in [Provider(provider_id='lstm',name='LSTM'),Provider(provider_id='transformer',name='Transformer'),Provider(provider_id='legacy',name='이전 분석 결과')]:self._providers.setdefault(p.provider_id,p.model_dump())
        atomic_json(self.root/'providers.json',self._providers)
        for run_id in self.catalog['runs']:
            for e in self.events(run_id,'fixture')+self.events(run_id):
                self.save_analysis([dict(e,provider_id='legacy',kind='detector',input_ref={'kind':'file','sha256':e.get('input_sha256')})])
        # Interrupted export jobs are explicit failures and may be requested again.
        for path in (self.root/'exports').glob('*.json'):
            job=read_json(path)
            if job['state'] in ('queued','running'):
                job.update(state='error',error='서버가 종료되어 내보내기가 중단됐습니다. 다시 요청하세요.');atomic_json(path,job)
        # Recover only incomplete processed archives. Original journals stay immutable.
        for row in self.catalog['sessions'].values():
            if not row.get('sha'):
                path=Path(row['path'])
                if path.exists():
                    archive=self.archive_for(row['id'])
                    if archive:archive.recover();archive.close()
                    row['sha']=hash_file(path)
        self._catalog_save()

    def profiles(self):return sorted(self._profiles.values(),key=lambda p:p['profile_id'])
    def profile(self,id):
        if id not in self._profiles:raise ValueError('등록되지 않은 데이터 프로필입니다.')
        return InputProfile.model_validate(self._profiles[id])
    def register_profile(self,p):
        with self.lock:
            if p.profile_id in self._profiles and self._profiles[p.profile_id]!=p.model_dump():raise ValueError('프로필 변경은 새 profile_id로 등록하세요.')
            self._profiles[p.profile_id]=p.model_dump();atomic_json(self.root/'profiles.json',self._profiles)
        return p.model_dump()
    def providers(self):return sorted(self._providers.values(),key=lambda p:p['provider_id'])
    def provider(self,id):
        if id not in self._providers:raise KeyError(id)
        return Provider.model_validate(self._providers[id])
    def register_provider(self,p):
        if p.provider_id=='legacy':raise ValueError('legacy는 이전 결과 보존용입니다.')
        with self.lock:self._providers[p.provider_id]=p.model_dump();atomic_json(self.root/'providers.json',self._providers)
        return p.model_dump()

    def create_session(self,config):
        profile=self.profile(config.profile_id)
        if (config.kind=='normalized')!=(profile.input_format=='normalized'):raise ValueError('입력 경로와 프로필 형식이 다릅니다.')
        id='live-'+uuid.uuid4().hex[:16];run=SessionRun(self.root/'sessions'/(id+'.jsonl'),config,profile,id)
        with self.lock:
            self.catalog['sessions'][id]={'id':id,'path':str(run.path),'sha':None,'created_at':datetime.now(timezone.utc).isoformat()}
            self._catalog_save();self.active[id]=run
        return run

    def finish_session(self,id):
        run=self.active[id];sha=run.finish()
        atomic_json(self.root/'records'/id/'manifest.json',run.manifest())
        with self.lock:
            self.catalog['sessions'][id]['sha']=sha;self._catalog_save();self.active.pop(id)
        self.journal.cache_clear()
        return self.get(id)

    @lru_cache(maxsize=4)
    def archive_for(self,id):
        if id in self.active:return self.active[id].archive
        row=self.catalog['sessions'].get(id)
        if not row:
            if id in self.catalog['runs']:return None
            raise KeyError(id)
        path=Path(row['path'])
        with path.open('rb') as f:header=json.loads(f.readline())
        if header['config']['kind']!='normalized':return None
        profile=InputProfile.model_validate(header['profile'])
        archive=NpzArchive(path.parent/id/'npz',path,profile,id)
        if not archive.data['finished'] or archive.data['jobs']:
            archive.recover()
        if archive.data['received_rows']==0 and path.stat().st_size:
            archive.recover()
        return archive

    @lru_cache(maxsize=2)
    def journal(self,id,path):
        archive=self.archive_for(id)
        if archive and archive.data['chunks']:
            summary=read_json(self.root/'records'/id/'manifest.json')
            return NpzReplay(Path(path),archive,summary)
        return SessionRun(Path(path))

    def get(self,id):
        if id in self.active:return self.active[id]
        row=self.catalog['sessions'].get(id)
        if row:
            run=self.journal(id,row['path'])
            if row.get('input_npz_sha256'):run.sha256=row['input_npz_sha256']
            return run
        return super().get(id)

    def list(self):
        rows=super().list()
        for r in list(self.catalog['sessions'].values()):
            try:rows.append(dict(self.get(r['id']).manifest(),created_at=r['created_at']))
            except (OSError,ValueError):continue
        return sorted(rows,key=lambda r:r['created_at'],reverse=True)

    def save_analysis(self,events):
        grouped={}
        for e in events:grouped.setdefault(e['run_id'],[]).append(e)
        for run_id,rows in grouped.items():self._save_rows(run_id,rows,'analysis')

    def analysis(self,run_id,cursor=None):
        with self.lock:
            rows=[e for e in self._rows(run_id,'analysis').values() if cursor is None or e['decided_at_ms']<=cursor]
            return sorted(rows,key=lambda e:e['decided_at_ms'],reverse=True)[:2000]

    def chunks(self,id):
        self.get(id);archive=self.archive_for(id)
        if not archive:return {'npz':None,'chunks':[]}
        with archive.lock:return {'npz':archive.status(),'chunks':deepcopy(archive.data['chunks'])}

    def request_export(self,id):
        run=self.get(id)
        if run.live:raise ValueError('수집을 종료한 후 전체 NPZ를 내보내세요.')
        archive=self.archive_for(id)
        if archive is None:raise ValueError('외부 전처리 입력으로 기록한 비행만 NPZ 내보내기를 지원합니다.')
        job={'job_id':'export-'+uuid.uuid4().hex[:16],'run_id':id,'state':'queued','error':'','download_url':None}
        atomic_json(self.root/'exports'/(job['job_id']+'.json'),job)
        if self.export_executor is None:self.export_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='arion-export')
        self.export_executor.submit(self._export,archive,job)
        return job

    def _export(self,archive,job):
        path=self.root/'exports'/(job['job_id']+'.json')
        try:
            job.update(state='running');atomic_json(path,job)
            target=self.root/'exports'/(job['job_id']+'.npz');export_npz(archive,target)
            job.update(state='ready',sha256=hash_file(target),download_url='/api/exports/'+job['job_id']+'/download',rows=archive.data['saved_rows'])
        except Exception as e:job.update(state='error',error=str(e)[:500])
        atomic_json(path,job)

    def export_status(self,id):
        if not id.startswith('export-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in id):raise KeyError(id)
        job=read_json(self.root/'exports'/(id+'.json'))
        if job is None:raise KeyError(id)
        return job

    def import_npz(self,content,name):
        sha=hashlib.sha256(content).hexdigest();id='npz-'+sha[:16]
        if id in self.catalog['sessions']:return self.get(id)
        meta,arrays=load_npz(io.BytesIO(content))
        features=meta['features'];config=meta.get('config',{})
        period=(meta.get('profile',{}).get('sample_period_ms') or config.get('tick_ns',100_000_000)/1e6)
        if meta.get('profile'):
            profile=InputProfile.model_validate(meta['profile'])
        else:
            ttl={'BATTERY_STATUS':config.get('battery_max_age_ns',600_000_000)/1e6,'ESC_STATUS':config.get('esc_max_age_ns',250_000_000)/1e6,
                 'LOCAL_POSITION_NED':config.get('motion_max_age_ns',300_000_000)/1e6,'ATTITUDE':config.get('motion_max_age_ns',300_000_000)/1e6}
            profile=InputProfile(profile_id='npz-'+sha[:16],input_format='normalized',preprocess_version=meta.get('preprocess_version','imported-physical-v1'),sample_period_ms=period,
                signals={f['name']:SignalDefinition(label=LABELS.get(f['name'],(f['name'],))[0],unit=f['unit'],ttl_ms=ttl.get(f.get('message'),60000),message=f.get('message','NORMALIZED')) for f in features},feature_order=[f['name'] for f in features])
        if profile.feature_order!=[f['name'] for f in features]:raise ValueError('NPZ 특징 순서와 계약 불일치')
        for feature in features:
            if profile.signals[feature['name']].unit!=feature['unit']:raise ValueError('NPZ 특징 단위와 계약 불일치')
        original=self.root/'logs'/(sha+'.npz');original.write_bytes(content)
        (self.root/'imports').mkdir(exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='npz-',dir=self.root/'imports'))
        provenance=meta.get('source',config)
        run=SessionRun(staging/(id+'.jsonl'),SourceConfig(name=Path(name).name,kind='normalized',profile_id=profile.profile_id,
                       source_system=provenance.get('source_system'),source_component=provenance.get('source_component')),profile,id)
        run.archive.data['metadata']['imported_source']={'sha256':sha,'metadata':meta}
        try:
            ticks=arrays['tick_ns'];epochs=arrays['epoch'];base=int(ticks[0]);last_epoch=-1
            for i in range(len(ticks)):
                values={}
                for j,key in enumerate(profile.feature_order):
                    value=float(arrays['X'][i,j]);value=value if math.isfinite(value) else None
                    mask=bool(arrays['mask'][i,j]);age=int(arrays['age_ns'][i,j])
                    if not mask and 'missing_fill' in meta:value=None
                    status='valid' if mask else 'missing' if age<0 or value is None else 'invalid'
                    if 'status' in arrays:
                        if arrays['status'].shape!=arrays['X'].shape:raise ValueError('status 크기 오류')
                        status={0:'missing',1:'valid',2:'stale',3:'invalid'}.get(int(arrays['status'][i,j]))
                        if status is None:raise ValueError('status 코드 오류')
                    values[key]={'value':value,'unit':profile.signals[key].unit,'mask':mask,'status':status,'age_ns':age if age>=0 else None}
                # Preserve exact source nanoseconds and physical values; this remains an independent flight.
                source=int(arrays['source_time_ns'][i]) if 'source_time_ns' in arrays else int(ticks[i])
                state={}
                if 'rows' in arrays:
                    saved=json.loads(str(arrays['rows'][i])).get('frame',{})
                    state={k:saved[k] for k in ('armed','mode','flight_state') if k in saved}
                run.append_normalized({'schema_version':'arion-processed-v1','session_id':id,'profile_id':profile.profile_id,
                    'sequence':i+1,'epoch':int(epochs[i]),'source_time_ns':source,'relative_time_ns':int(ticks[i])-base,
                    'preprocess_version':profile.preprocess_version,'sample_period_ms':profile.sample_period_ms,'values':values,**state})
            run.finish()
            summary=run.manifest();summary.update(source='NPZ 독립 기록',input_sha256=sha)
            atomic_json(self.root/'records'/id/'manifest.json',summary)
            with self.lock:
                if id in self.catalog['sessions']:return self.get(id)
                destination=self.root/'sessions'/(id+'.jsonl');destination.parent.mkdir(exist_ok=True)
                os.replace(run.path,destination)
                archive_destination=destination.parent/id
                if archive_destination.exists():
                    archive_destination.rename(destination.parent/(id+'-orphan-'+uuid.uuid4().hex[:8]))
                (staging/id).rename(archive_destination)
                self.catalog['sessions'][id]={'id':id,'path':str(destination),'sha':hash_file(destination),'input_npz_sha256':sha,'original_path':str(original),'created_at':datetime.now(timezone.utc).isoformat()}
                self._catalog_save()
            return self.get(id)
        except Exception:
            run.finish()
            # Unaccepted files remain inspectable, never listed as successful imports.
            raise
        finally:shutil.rmtree(staging,ignore_errors=True)

    def close(self):
        for id in list(self.active):self.finish_session(id)
        for archive in list(self.archives()):archive.close()
        if self.export_executor:self.export_executor.shutdown(wait=True)

    def archives(self):
        for id in self.catalog['sessions']:
            try:
                archive=self.archive_for(id)
                if archive:yield archive
            except (OSError,ValueError):continue
