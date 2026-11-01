"""NPZ chunks from durable journal byte ranges. Compression never blocks ingest."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
import tempfile
import zipfile
import numpy as np
from .file_io import atomic_json, read_json, canonical, hash_file

CHUNK_NS=60_000_000_000
MAX_UNPACKED=128*1024*1024
MAX_CHUNK_UNPACKED=48*1024*1024
STATUSES={'missing':0,'valid':1,'stale':2,'invalid':3}


def journal_items(path,start=0,end=None):
    with Path(path).open('rb') as f:
        f.seek(start)
        while end is None or f.tell()<end:
            offset=f.tell();line=f.readline()
            if not line:break
            try:item=json.loads(line)
            except ValueError:continue
            if item.get('type')=='sample':yield offset,f.tell(),item


def npz_metadata(profile,run_id):
    return {'schema_version':'arion-npz-v1','normalization':'physical_units','session_id':run_id,
            'profile':profile.model_dump(),'preprocess_version':profile.preprocess_version,
            'features':[{'name':k,'unit':profile.signals[k].unit} for k in profile.feature_order],
            'status_codes':STATUSES,'note':'관측 기록입니다. 학습 및 모델 정규화는 외부에서 수행합니다.'}


def row_arrays(items,profile):
    names=profile.feature_order;n=len(items);f=len(names)
    x=np.full((n,f),np.nan,dtype=np.float64);mask=np.zeros((n,f),dtype=np.bool_)
    status=np.zeros((n,f),dtype=np.uint8);age=np.full((n,f),-1,dtype=np.int64)
    for i,item in enumerate(items):
        signals=item.get('processed',{}).get('values')
        if signals is None:
            signals={k:{'value':v['value'],'mask':v['status']=='valid','status':v['status'],
                        'age_ns':round(v['age_ms']*1e6) if v['age_ms'] is not None else None}
                     for k,v in item['frame']['values'].items()}
        for j,key in enumerate(names):
            s=signals.get(key)
            if s is None:continue
            if s['value'] is not None:x[i,j]=s['value']
            mask[i,j]=s['mask'];status[i,j]=STATUSES[s['status']]
            if s['age_ns'] is not None:age[i,j]=s['age_ns']
    return {'X':x,'mask':mask,'status':status,'age_ns':age,
            'tick_ns':np.array([i.get('tick_ns',round(i['t_ms']*1e6)) for i in items],dtype=np.int64),
            'source_time_ns':np.array([i.get('processed',i.get('raw',{})).get('source_time_ns',0) for i in items],dtype=np.int64),
            'sequence':np.array([i['sequence'] for i in items],dtype=np.int64),
            'producer_sequence':np.array([i.get('processed',i.get('raw',{})).get('sequence',i['sequence']) for i in items],dtype=np.int64),
            'epoch':np.array([i['epoch'] for i in items],dtype=np.int64),
            'row_id':np.array([i.get('row_id','') for i in items]),
            'row_sha256':np.array([i.get('row_sha256','') for i in items]),
            'rows':np.array([canonical(i).decode() for i in items])}


def atomic_npz(path,arrays,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.npz-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,metadata=json.dumps(metadata,ensure_ascii=False),**arrays)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


class NpzArchive:
    def __init__(self,path,journal,profile,run_id):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.journal=Path(journal);self.profile=profile;self.run_id=run_id
        self.manifest_path=self.path/'manifest.json';self.lock=RLock()
        self.data=read_json(self.manifest_path,{'schema_version':'arion-npz-archive-v1','state':'recording','error':'',
                'received_rows':0,'saved_rows':0,'chunks':[],'jobs':[],'finished':False,'metadata':npz_metadata(profile,run_id)})
        self.executor=None;self.futures=[];self.pending=None

    def status(self):
        with self.lock:
            return {k:deepcopy(self.data[k]) for k in ('state','error','received_rows','saved_rows','finished')} | {
                'chunk_count':len(self.data['chunks']),'pending_rows':self.data['received_rows']-self.data['saved_rows'],
                'schema_version':'arion-processed-v1','chunk_seconds':60}

    def _save(self):atomic_json(self.manifest_path,self.data)

    def observe(self,item,start,end):
        tick=item.get('tick_ns',round(item['t_ms']*1e6))
        estimated_bytes=len(canonical(item).decode())*4 + len(self.profile.signals)*18 + 700
        with self.lock:
            if self.pending and (self.pending['epoch']!=item['epoch'] or tick-self.pending['tick_ns']>=CHUNK_NS or self.pending['rows']>=60000 or self.pending['estimated_bytes']+estimated_bytes>MAX_CHUNK_UNPACKED):self._seal()
            if self.pending is None:self.pending={'start':start,'end':end,'start_sequence':item['sequence'],'end_sequence':item['sequence'],
                        'epoch':item['epoch'],'tick_ns':tick,'rows':0,'estimated_bytes':0}
            self.pending.update(end=end,end_sequence=item['sequence']);self.pending['rows']+=1
            self.pending['estimated_bytes']+=estimated_bytes
            self.data['received_rows']+=1

    def _seal(self):
        if not self.pending:return
        job=self.pending;self.pending=None
        self.data['jobs'].append(dict(job,state='queued'));self.data['state']='saving';self._save()
        self._submit(job)

    def _submit(self,job):
        if self.executor is None:self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='arion-npz')
        self.futures=[f for f in self.futures if not f.done()]
        self.futures.append(self.executor.submit(self._write_chunk,dict(job)))

    def _write_chunk(self,job):
        try:
            items=[i for _,_,i in journal_items(self.journal,job['start'],job['end'])]
            if len(items)!=job['rows']:raise ValueError('저널 구간과 NPZ 행 수 불일치')
            filename=f"chunk-{job['start_sequence']:012d}-{job['end_sequence']:012d}.npz"
            path=self.path/filename
            with self.lock:meta=deepcopy(self.data['metadata'])
            meta.update(start_sequence=job['start_sequence'],end_sequence=job['end_sequence'])
            atomic_npz(path,row_arrays(items,self.profile),meta)
            chunk={'filename':filename,'rows':len(items),'start_sequence':items[0]['sequence'],'end_sequence':items[-1]['sequence'],
                   'start_ms':items[0]['t_ms'],'end_ms':items[-1]['t_ms'],'epoch':items[0]['epoch'],'sha256':hash_file(path)}
            with self.lock:
                self.data['chunks']=[c for c in self.data['chunks'] if c['filename']!=filename]+[chunk]
                self.data['chunks'].sort(key=lambda c:c['start_sequence'])
                self.data['saved_rows']=sum(c['rows'] for c in self.data['chunks'])
                self.data['jobs']=[j for j in self.data['jobs'] if j['start_sequence']!=job['start_sequence']]
                self.data['error']=next((j.get('error','저장 오류') for j in self.data['jobs'] if j['state']=='error'),'')
                self.data['state']='error' if self.data['error'] else 'saving' if self.data['jobs'] else 'saved' if self.data['finished'] else 'recording'
                self._save()
        except Exception as e:
            with self.lock:
                for j in self.data['jobs']:
                    if j['start_sequence']==job['start_sequence']:j.update(state='error',error=str(e)[:500])
                self.data.update(state='error',error=str(e)[:500]);self._save()

    def finish(self,wait=True):
        with self.lock:
            self.data['finished']=True;self._seal()
            if not self.data['jobs']:self.data['state']='saved';self._save()
        if wait:self.wait()

    def wait(self):
        for future in list(self.futures):future.result()

    def retry(self):
        with self.lock:
            for job in self.data['jobs']:
                if job['state']=='error':job['state']='queued';self._submit(job)
            if self.data['jobs']:self.data['state']='saving'
            self._save()

    def recover(self):
        # Reconstruct pending boundaries from the immutable journal, one chunk at a time.
        self.wait()
        with self.lock:
            self.data['received_rows']=0;self.data['jobs']=[];self.data['finished']=False;self.pending=None
        saved_ranges=[(c['start_sequence'],c['end_sequence']) for c in self.data['chunks']]
        for start,end,item in journal_items(self.journal):
            if any(a<=item['sequence']<=b for a,b in saved_ranges):
                with self.lock:self._seal()
                self.data['received_rows']+=1
            else:self.observe(item,start,end)
        self.finish()

    def close(self):
        self.finish()
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor=None


def load_npz(path):
    try:
        with zipfile.ZipFile(path) as z:
            if sum(i.file_size for i in z.infolist())>MAX_UNPACKED:raise ValueError('NPZ 압축 해제 크기는 128 MiB 이하여야 합니다.')
        with np.load(path,allow_pickle=False) as d:
            arrays={k:d[k] for k in d.files}
        meta=json.loads(str(arrays.pop('metadata').item()))
        x=arrays['X'];mask=arrays['mask'];ticks=arrays['tick_ns'];epoch=arrays['epoch']
        n,f=x.shape if x.ndim==2 else (0,0)
        if not n or f>256 or x.dtype.kind not in 'fiu' or mask.shape!=x.shape or mask.dtype.kind!='b':raise ValueError('X/mask 크기 또는 자료형 오류')
        features=meta['features']
        if len(features)!=f or any(not isinstance(s.get('name'),str) or not isinstance(s.get('unit'),str) for s in features):raise ValueError('특징 이름·단위·순서 오류')
        if len({s['name'] for s in features})!=f:raise ValueError('특징 이름 중복')
        for key in ('tick_ns','epoch'):
            if arrays[key].shape!=(n,) or arrays[key].dtype.kind not in 'iu' or np.any(arrays[key]<0):raise ValueError(f'{key} 크기·자료형 오류')
        if np.any(np.diff(ticks.astype(np.int64))<0) or np.any(np.diff(epoch.astype(np.int64))<0):raise ValueError('시간 또는 epoch 순서 오류')
        if np.any(~np.isfinite(x[mask])):raise ValueError('유효한 셀에 유한한 값이 필요합니다.')
        age=arrays['age_ns']
        if age.shape!=x.shape and meta.get('age_messages'):
            messages=meta['age_messages']
            if age.shape!=(n,len(messages)) or any(s.get('message') not in messages for s in features):
                raise ValueError('메시지 그룹별 관측 나이 매핑 오류')
            age=age[:,[messages.index(s['message']) for s in features]]
            arrays['age_ns']=age
        if age.shape!=x.shape or age.dtype.kind not in 'iu' or np.any(age[mask]<0):raise ValueError('관측 나이 크기·자료형 오류')
        for key in ('source_time_ns','sequence','producer_sequence','row_id','row_sha256','rows'):
            if key in arrays and arrays[key].shape!=(n,):raise ValueError(f'{key} 길이 오류')
        for key in ('source_time_ns','sequence','producer_sequence'):
            if key in arrays and (arrays[key].dtype.kind not in 'iu' or np.any(arrays[key]<0)):raise ValueError(f'{key} 자료형 오류')
        if 'status' in arrays:
            s=arrays['status']
            if s.shape!=x.shape or s.dtype.kind not in 'iu' or np.any(s>3) or np.any(s<0) or np.any((s==1)!=mask):raise ValueError('status/mask 불일치')
        if meta.get('normalization') not in ('physical_units','identity_not_model_ready','identity'):raise ValueError('물리 단위가 확인되지 않은 NPZ는 재생할 수 없습니다.')
        return meta,arrays
    except (KeyError,TypeError,IndexError,zipfile.BadZipFile,EOFError) as e:raise ValueError(f'NPZ 구조 오류: {e}') from e


def export_npz(archive,target):
    """Disk-backed arrays and streamed .npy ZIP members; no full-flight RAM copy."""
    chunks=deepcopy(archive.data['chunks']);count=sum(c['rows'] for c in chunks)
    if not count:raise ValueError('내보낼 NPZ 관측이 없습니다.')
    if archive.data['jobs']:raise ValueError('NPZ 구간 저장이 완료되지 않았습니다. 저장 재시도 후 내보내세요.')
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='arion-export-',dir=target.parent) as directory:
        scratch=Path(directory);maps={};offset=0;string_sizes={}
        for chunk in chunks:
            _,arrays=load_npz(archive.path/chunk['filename'])
            for key,a in arrays.items():
                if a.dtype.kind=='U':string_sizes[key]=max(string_sizes.get(key,0),a.dtype.itemsize//4)
        for chunk in chunks:
            _,arrays=load_npz(archive.path/chunk['filename'])
            for key,a in arrays.items():
                if key not in maps:
                    dtype=f'U{string_sizes[key]}' if key in string_sizes else a.dtype
                    maps[key]=np.lib.format.open_memmap(scratch/(key+'.npy'),mode='w+',dtype=dtype,shape=(count,*a.shape[1:]))
                maps[key][offset:offset+len(a)]=a
            offset+=chunk['rows']
        for a in maps.values():a.flush()
        meta=deepcopy(archive.data['metadata']);meta.update(chunks=chunks,final_log_sha256=hash_file(archive.journal))
        np.save(scratch/'metadata.npy',json.dumps(meta,ensure_ascii=False),allow_pickle=False)
        temporary=target.with_suffix('.tmp')
        try:
            with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
                for path in sorted(scratch.glob('*.npy')):z.write(path,path.name)
            with temporary.open('rb') as f:os.fsync(f.fileno())
            os.replace(temporary,target)
        finally:temporary.unlink(missing_ok=True)
    return target
