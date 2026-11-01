"""Append-only session journals and passive MAVLink transports."""
import asyncio
from bisect import bisect_right
from collections import deque, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import time
from statistics import median
import uuid

from .contracts import InputProfile, SourceConfig, NormalizedInput
from .telemetry import TelemetryRun, MAX_AGE

BUFFER_MS = 300_000
MAX_BUFFER_RECORDS = 60_000

def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def safe(value):
    import math
    if isinstance(value, (bytes, bytearray)): return {'$bytes':bytes(value).hex()}
    if isinstance(value, float) and not math.isfinite(value): return {'$float': str(value)}
    if isinstance(value, dict): return {k:safe(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [safe(v) for v in value]
    return value

class SessionRun:
    def __init__(self, path, config=None, profile=None, session_id=None):
        self.path = Path(path)
        self.live = config is not None
        self.entries = []
        self.times = []
        self.issues = deque(maxlen=200)
        self.boundaries = []
        self.epoch_starts = [0.0]
        self.total_lines = 0
        self.accepted = 0
        self.excluded = 0
        self.dropped_sequences = 0
        self.stream_stats = defaultdict(lambda: {'count':0, 'last':None, 'intervals':deque(maxlen=2048)})
        self.last_receipt = time.perf_counter()
        self.duration_ms = 0.0
        self.last_external_sequence = None
        self.external_epoch = None
        self.force_boundary = False
        self.state = 'waiting'
        self.error = ''
        self.core = TelemetryRun.__new__(TelemetryRun)
        self.core.initialize()
        self.sha256 = None
        if self.live:
            self.run_id = session_id or 'live-' + uuid.uuid4().hex[:16]
            self.config, self.profile = config, profile
            self.name = config.name
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = self.path.open('xb')
            self.writer.write(encode({'schema_version':'arion-journal-v2', 'session_id':self.run_id,
                                     'config':config.model_dump(), 'profile':profile.model_dump()})+b'\n')
            self.writer.flush()
        else:
            with self.path.open('rb') as f:
                header=json.loads(f.readline())
                self.run_id=header['session_id'];self.config=SourceConfig.model_validate(header['config'])
                self.profile=InputProfile.model_validate(header['profile']);self.name=self.config.name
                for line in f:
                    self.total_lines += 1
                    try:item=json.loads(line)
                    except ValueError:
                        self.excluded+=1;self.issues.append({'line':self.total_lines,'reason':'중단된 수집의 손상 행 · 원문 보존','severity':'excluded'})
                        continue
                    if item['type']=='sample': self._remember(item)
                    elif item['type']=='issue':
                        self.excluded+=1;self.issues.append(item['issue'])
            self.state='finished'
            with self.path.open('rb') as f: self.sha256=hashlib.file_digest(f, 'sha256').hexdigest()
        for spec in self.profile.signals.values():
            if spec.message in self.core.max_age: self.core.max_age[spec.message]=spec.ttl_ms
        self.core.rates = dict(self.profile.expected_rates_hz)
        self.core.gap_ms = self.profile.gap_ms
        self.core.voltage_encoding = self.profile.voltage_encoding
        self.core.missing_current_raw = self.profile.missing_current_raw
        self.core.run_id = self.run_id
        self.source = getattr(self, "source", (self.config.source_system, self.config.source_component))

    @property
    def now_ms(self):
        return self.duration_ms + (time.perf_counter()-self.last_receipt)*1000 if self.live and self.entries else self.duration_ms

    def _remember(self, item):
        self.accepted += 1
        self.entries.append(item);self.times.append(item['t_ms']);self.duration_ms=item['t_ms']
        epoch=item['epoch']
        if epoch >= len(self.epoch_starts):
            self.epoch_starts.append(item['t_ms'])
            self.boundaries.append({'t_ms':item['t_ms'],'epoch':epoch,'reason':'연결·시각·수신 경계','line':item['sequence']})
        if item.get('raw',{}).get('source_system') is not None:
            self.source=(item['raw']['source_system'],item['raw']['source_component'])
        name=item.get('message','NORMALIZED'); stat=self.stream_stats[name]
        stat['count'] += 1
        if stat['last'] and stat['last'][1]==epoch and item['t_ms']>stat['last'][0]:
            stat['intervals'].append(item['t_ms']-stat['last'][0])
        stat['last']=(item['t_ms'],epoch)
        if self.live:
            cut=max(bisect_right(self.times,self.duration_ms-BUFFER_MS)-1,len(self.entries)-MAX_BUFFER_RECORDS,0)
            if cut: del self.entries[:cut];del self.times[:cut]
            # The decoding cache is not an archive. Retain only the display window.
            for key in list(self.core.by_stream):
                times=self.core.stream_times[key]
                n=max(0,bisect_right(times,self.duration_ms-BUFFER_MS)-1,len(times)-MAX_BUFFER_RECORDS)
                if n: del times[:n];del self.core.by_stream[key][:n]
            n=max(0,bisect_right(self.core.sample_times,self.duration_ms-BUFFER_MS)-1,len(self.core.samples)-MAX_BUFFER_RECORDS)
            if n: del self.core.samples[:n];del self.core.sample_times[:n]

    def issue(self, reason, raw=None):
        self.excluded+=1;self.total_lines+=1
        issue={'line':self.total_lines,'reason':str(reason)[:500],'severity':'excluded'}
        self.issues.append(issue)
        self._write({'type':'issue','issue':issue,'raw':safe(raw)})

    def _write(self, item):
        self.writer.write(encode(item)+b'\n');self.writer.flush()

    def append_raw(self, record):
        if not self.live: raise ValueError('종료된 세션입니다.')
        if self.config.kind=='normalized': raise ValueError('입력 형식 불일치')
        if self.config.source_system is not None and record['source_system'] != self.config.source_system:
            self.issue('다른 system 제외',record);return False
        if self.config.source_component is not None and record['source_component'] != self.config.source_component:
            self.issue('다른 component 제외',record);return False
        record=dict(record,quality={'profile':self.profile.profile_id})
        ok=self.core.ingest(record, force_boundary=self.force_boundary)
        if not ok:
            self.issue(self.core.issues[-1]['reason'],record);self.core.issues=self.core.issues[-200:];return False
        self.force_boundary=False
        frame=self.core.frame(self.core.duration_ms)
        frame['values']={k:v for k,v in frame['values'].items() if k in self.profile.signals}
        self.source=self.core.source
        return self._append(frame,record['message_name'],safe(record))

    def append_normalized(self, payload):
        if not self.live or self.config.kind!='normalized' or self.state in ('disconnected','error'): raise ValueError('활성 전처리 세션이 아닙니다.')
        try:
            p=NormalizedInput.model_validate(payload)
            if p.session_id!=self.run_id: raise ValueError('session_id 불일치')
            if p.preprocess_version!=self.profile.preprocess_version: raise ValueError('전처리 버전 불일치')
            if p.sample_period_ms!=self.profile.sample_period_ms: raise ValueError('전처리 주기 변경: 새 프로필·세션이 필요합니다.')
            if self.external_epoch is not None and p.epoch < self.external_epoch: raise ValueError('오래된 producer epoch')
            if self.external_epoch==p.epoch and self.last_external_sequence is not None and p.sequence<=self.last_external_sequence:
                raise ValueError('중복 또는 순서가 역전된 sequence')
            for key, value in p.values.items():
                if key not in self.profile.signals: raise ValueError(f'미등록 신호: {key}')
                if value.unit!=self.profile.signals[key].unit: raise ValueError(f'단위 불일치: {key}')
            boundary=self.force_boundary or (self.external_epoch is not None and self.external_epoch!=p.epoch)
            if self.external_epoch==p.epoch and self.last_external_sequence is not None:
                self.dropped_sequences+=max(0,p.sequence-self.last_external_sequence-1)
            self.core.ingest({'receive_monotonic_ns':p.source_time_ns,'source_system':1,'source_component':1,
                              'message_name':'HEARTBEAT','fields':{}},force_boundary=boundary)
            t=self.core.duration_ms; values={}
            for key, spec in self.profile.signals.items():
                v=p.values.get(key)
                status='missing' if v is None else 'invalid' if not v.valid or v.value is None else 'stale' if v.age_ms>spec.ttl_ms else 'valid'
                values[key]={'value':v.value if status=='valid' else None,'status':status,'unit':spec.unit,'label':spec.label,
                             'message':spec.message,'age_ms':v.age_ms if v else None,'reason':'mask=false' if v and not v.valid else ''}
            frame={'epoch':self.core._epoch,'t_ms':t,'values':values,'streams':[], 'armed':p.armed,'mode':p.mode,'flight_state':p.flight_state}
            self.external_epoch=p.epoch;self.last_external_sequence=p.sequence;self.force_boundary=False
            return self._append(frame,'NORMALIZED',p.model_dump())
        except (ValueError,TypeError) as error:
            self.issue(str(error),payload);raise ValueError(str(error)) from error

    def _append(self, frame, message, raw):
        self.total_lines+=1
        frame.update(schema_version='arion-telemetry-v2',run_id=self.run_id,sequence=self.accepted+1,
                     profile_id=self.profile.profile_id,preprocess_version=self.profile.preprocess_version,
                     sample_period_ms=self.profile.sample_period_ms)
        item={'type':'sample','sequence':self.accepted+1,'t_ms':frame['t_ms'],'epoch':frame['epoch'],
              'message':message,'raw':raw,'frame':frame}
        self._write(item);self._remember(item)
        self.last_receipt=time.perf_counter();self.state='connected';self.error=''
        return True

    def epoch_at(self, cursor): return max(0,bisect_right(self.epoch_starts,cursor)-1)

    def frame(self, cursor):
        cursor=max(0,float(cursor))
        i=bisect_right(self.times,cursor)-1
        if i<0:
            return {'schema_version':'arion-telemetry-v2','run_id':self.run_id,'epoch':self.epoch_at(cursor),'t_ms':cursor,
                    'sequence':0,'values':{},'streams':[],'armed':None,'mode':'미확인','flight_state':'미확인'}
        f=deepcopy(self.entries[i]['frame']);delta=cursor-f['t_ms'];f['t_ms']=cursor
        for key,v in f['values'].items():
            if v['age_ms'] is not None: v['age_ms']=round(v['age_ms']+delta,3)
            if v['status']=='valid' and v['age_ms']>self.profile.signals[key].ttl_ms:
                v.update(value=None,status='stale')
        if delta>self.profile.gap_ms: f.update(armed=None,mode='미확인',flight_state='미확인')
        if self.config.kind=='normalized':
            f['streams']=[{'name':'NORMALIZED','last_t_ms':self.entries[i]['t_ms'],'age_ms':delta,
                           'status':'valid' if delta<=self.profile.gap_ms else 'stale'}]
        else:
            for stream in f['streams']:
                if stream['age_ms'] is not None:
                    stream['age_ms']+=delta
                    if stream['age_ms']>self.core.max_age[stream['name']]: stream['status']='stale'
        return f

    def manifest(self):
        streams=[]
        for name,s in self.stream_stats.items():
            xs=list(s['intervals'])
            streams.append({'name':name,'count':s['count'],'median_hz':1000/median(xs) if xs else None,
                            'target_hz':self.profile.expected_rates_hz.get(name),'max_interval_ms':max(xs) if xs else None})
        return {'schema_version':'arion-run-v2','run_id':self.run_id,'name':self.name,'input_sha256':self.sha256,
                'source':f'{self.config.kind.upper()} '+('실시간' if self.live else '수집 기록'),
                'source_kind':'live' if self.live else 'replay','source_system':self.source[0], 'source_component':self.source[1],
                'duration_ms':self.now_ms,'total_records':self.total_lines,'accepted_records':self.accepted,
                'excluded_records':self.excluded,'issues':list(self.issues),'boundaries':self.boundaries[-200:],
                'epochs':len(self.epoch_starts),'streams':streams,'supported_signals':list(self.profile.signals),
                'profile':self.profile.model_dump(),'session':self.session_status()}

    def session_status(self):
        state=self.state
        if self.live and state=='connected' and self.now_ms-self.duration_ms>self.profile.gap_ms: state='stale'
        return {'session_id':self.run_id,'state':state,'message':self.error,'kind':self.config.kind,
                'buffer_start_ms':self.times[0] if self.times else 0,'recording':self.live,
                'last_sequence':self.accepted,'dropped_sequences':self.dropped_sequences,
                'config':self.config.model_dump()}

    def series(self):
        points=[];last=-1;epoch=None
        for item in self.entries:
            f=item['frame']
            if f['t_ms']-last<100 and f['epoch']==epoch: continue
            if epoch is not None and (f['epoch']!=epoch or f['t_ms']-last>self.profile.gap_ms):
                points.append({'t_ms':f['t_ms']-.001,'epoch':f['epoch'],'values':{k:None for k in self.profile.signals}})
            points.append({'t_ms':f['t_ms'],'epoch':f['epoch'],'values':{k:v['value'] for k,v in f['values'].items()}})
            last=f['t_ms'];epoch=f['epoch']
        return {'step_ms':100,'points':points}

    def segment(self, start, end):
        items=[x for x in self.entries if start<=x['sequence']<=end]
        if not items or items[0]['sequence']!=start or items[-1]['sequence']!=end or len(items)!=end-start+1:
            raise ValueError('입력 구간이 표시 버퍼에 없거나 sequence가 누락됐습니다.')
        if len({x['epoch'] for x in items})!=1: raise ValueError('입력 구간이 epoch 경계를 넘습니다.')
        digest=hashlib.sha256(b'\n'.join(encode(x['frame']) for x in items)).hexdigest()
        return {'kind':'stream','session_id':self.run_id,'start_sequence':start,'end_sequence':end,
                'segment_sha256':digest,'preprocess_version':self.profile.preprocess_version},items

    def finish(self):
        if self.live:
            self.writer.close();self.live=False;self.state='finished'
            with self.path.open('rb') as f: self.sha256=hashlib.file_digest(f,'sha256').hexdigest()
        return self.sha256

class LiveSources:
    def __init__(self,store): self.store=store;self.tasks={};self.handles={}

    async def connect(self, run):
        await self.disconnect(run)
        if not run.live: raise ValueError('종료된 세션은 다시 연결할 수 없습니다.')
        c=run.config
        if c.kind=='normalized': run.state='waiting';return
        try:
            if c.kind=='udp':
                handle=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                try: handle.bind((c.host,c.port));handle.setblocking(False)
                except Exception: handle.close();raise
            else:
                import serial
                if not c.device.startswith('/dev/'): raise ValueError('명시적인 /dev/ 시리얼 장치가 필요합니다.')
                handle=serial.Serial(c.device,c.baud,timeout=0,exclusive=True)
            self.handles[run.run_id]=handle
            self.tasks[run.run_id]=asyncio.create_task(self._read(run,handle))
            run.state='waiting';run.error=''
        except (OSError,ValueError) as e:
            run.state='error';run.error=str(e);raise ValueError(str(e)) from e

    async def disconnect(self,run):
        task=self.tasks.pop(run.run_id,None)
        if task:
            task.cancel()
            try: await task
            except asyncio.CancelledError: pass
        handle=self.handles.pop(run.run_id,None)
        if handle: handle.close()
        run.force_boundary=bool(run.entries)
        if run.live: run.state='disconnected'

    async def _read(self,run,handle):
        from pymavlink.dialects.v20 import common
        decoder=common.MAVLink(None);decoder.robust_parsing=True
        loop=asyncio.get_running_loop()
        try:
            while True:
                if run.config.kind=='udp':
                    try: data,_=handle.recvfrom(65535)
                    except BlockingIOError:
                        await asyncio.sleep(.005);continue
                else:
                    data=handle.read(65535)
                    if not data: await asyncio.sleep(.01);continue
                for msg in decoder.parse_buffer(data) or []:
                    if msg.get_type()=='BAD_DATA': run.issue('MAVLink 프레임 해석 실패');continue
                    fields=msg.to_dict();fields.pop('mavpackettype',None)
                    run.append_raw({'receive_monotonic_ns':time.monotonic_ns(),'receive_utc_ns':time.time_ns(),
                                    'source_system':msg.get_srcSystem(),'source_component':msg.get_srcComponent(),
                                    'message_name':msg.get_type(),'fields':fields})
        except asyncio.CancelledError: raise
        except Exception as e:
            run.state='error';run.error=f'{type(e).__name__}: {e}'[:300]
        finally:
            handle.close()

    async def close(self):
        for run in list(self.store.active.values()):
            await self.disconnect(run);self.store.finish_session(run.run_id)
