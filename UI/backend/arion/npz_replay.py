"""Causal replay with a two-chunk cache, also used for imported physical-unit NPZ."""
from bisect import bisect_right
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from threading import RLock
import json
from .contracts import InputProfile, SourceConfig
from .streaming import SessionRun
from .telemetry import TelemetryRun
from .npz_archive import NpzArchive, load_npz
from .file_io import read_json, hash_file

class NpzReplay(SessionRun):
    def __init__(self,journal,archive,summary=None):
        self.path=Path(journal);self.archive=archive;self.run_id=archive.run_id
        self.lock=RLock()
        self.profile=archive.profile
        with self.path.open('rb') as f:header=json.loads(f.readline())
        self.config=SourceConfig.model_validate(header['config']);self.name=self.config.name
        self.live=False;self.state='finished';self.error='';self.sha256=hash_file(self.path)
        self.source=(self.config.source_system,self.config.source_component)
        self.entries=[];self.times=[];self.accepted=archive.data['received_rows'];self.dropped_sequences=0
        self.total_lines=self.accepted;self.excluded=0;self.issues=deque(maxlen=200);self.boundaries=[]
        self.duration_ms=max((c['end_ms'] for c in archive.data['chunks']),default=0)
        self.epoch_starts=[0.0];self.stream_stats=defaultdict(lambda:{'count':0,'last':None,'intervals':deque(maxlen=2048)})
        self.core=TelemetryRun.__new__(TelemetryRun);self.core.initialize()
        self.starts=[c['start_ms'] for c in archive.data['chunks']]
        if summary:
            self.total_lines=summary['total_records'];self.excluded=summary['excluded_records'];self.issues.extend(summary['issues'])
            self.boundaries=summary.get('boundaries',[]);self.dropped_sequences=summary.get('session',{}).get('dropped_sequences',0)
            self.source=(summary.get('source_system'),summary.get('source_component'))
        else:
            with self.path.open('rb') as f:
                f.readline()
                for line in f:
                    try:item=json.loads(line)
                    except ValueError:
                        self.excluded+=1;self.issues.append({'line':self.total_lines+1,'reason':'중단된 수집의 손상 행 · 원문 보존','severity':'excluded'});continue
                    if item.get('type')=='issue':self.excluded+=1;self.issues.append(item['issue'])
                    elif item.get('type')=='sample':
                        raw=item.get('raw',{});sequence=raw.get('sequence')
                        if sequence is not None and raw.get('epoch')==getattr(self,'_producer_epoch',None):self.dropped_sequences+=max(0,sequence-self._producer_sequence-1)
                        self._producer_epoch=raw.get('epoch');self._producer_sequence=sequence
            self.total_lines=self.accepted+self.excluded
        for chunk in archive.data['chunks']:
            if chunk['epoch']>=len(self.epoch_starts):
                self.epoch_starts.append(chunk['start_ms'])
                if not summary:self.boundaries.append({'t_ms':chunk['start_ms'],'epoch':chunk['epoch'],'reason':'수집 경계','line':chunk['start_sequence']})
        self._summary=summary

    @lru_cache(maxsize=2)
    def _chunk(self,index):
        chunk=self.archive.data['chunks'][index];meta,arrays=load_npz(self.archive.path/chunk['filename'])
        if hash_file(self.archive.path/chunk['filename'])!=chunk['sha256']:raise ValueError('NPZ 구간 파일 해시 불일치')
        return [json.loads(str(r)) for r in arrays['rows']]

    def _at(self,cursor):
        index=bisect_right(self.starts,cursor)-1
        if index<0:return None
        rows=self._chunk(index);j=bisect_right([r['t_ms'] for r in rows],cursor)-1
        return rows[j] if j>=0 else None

    def frame(self,cursor):
        with self.lock:
            cursor=max(0,min(float(cursor),self.duration_ms));item=self._at(cursor)
            if item is None:self.entries=[];self.times=[]
            else:self.entries=[item];self.times=[item['t_ms']]
            return super().frame(cursor)

    def session_status(self):
        return {'session_id':self.run_id,'state':'finished','message':'','kind':'normalized','recording':False,
                'buffer_start_ms':0,'last_sequence':self.accepted,'dropped_sequences':self.dropped_sequences,'config':self.config.model_dump()}

    def manifest(self):
        if self._summary:
            output=dict(self._summary,source_kind='replay',source='NPZ 독립 기록' if self._summary.get('source')=='NPZ 독립 기록' else 'NORMALIZED 수집 기록',input_sha256=self.sha256,
                        duration_ms=self.duration_ms,session=self.session_status(),npz=self.archive.status())
            return output
        streams=[{'name':'NORMALIZED','count':self.accepted,'median_hz':1000/self.profile.sample_period_ms if self.profile.sample_period_ms else None,
                  'target_hz':1000/self.profile.sample_period_ms if self.profile.sample_period_ms else None,'max_interval_ms':None}]
        return {'schema_version':'arion-run-v3','run_id':self.run_id,'name':self.name,'input_sha256':self.sha256,
                'source':'NPZ 독립 기록','source_kind':'replay','source_system':self.source[0],'source_component':self.source[1],
                'duration_ms':self.duration_ms,'total_records':self.total_lines,'accepted_records':self.accepted,'excluded_records':self.excluded,
                'issues':list(self.issues),'boundaries':self.boundaries,'epochs':len(self.epoch_starts),'streams':streams,
                'profile':self.profile.model_dump(),'session':self.session_status(),'npz':self.archive.status()}

    def series(self,start_ms=0,end_ms=None,max_points=6000):
        end_ms=self.duration_ms if end_ms is None else end_ms
        # A bounded chart representation; a seek loads its surrounding five minutes.
        step=max(100,(end_ms-start_ms)/max_points);points=[];last=-float('inf');epoch=None
        for n,c in enumerate(self.archive.data['chunks']):
            if c['end_ms']<start_ms or c['start_ms']>end_ms:continue
            for item in self._chunk(n):
                f=item['frame']
                if f['t_ms']<start_ms or f['t_ms']>end_ms:continue
                if f['t_ms']-last<step and f['epoch']==epoch:continue
                if epoch is not None and (f['epoch']!=epoch or f['t_ms']-last>max(self.profile.gap_ms,step*1.5)):
                    points.append({'t_ms':f['t_ms']-.001,'epoch':f['epoch'],'values':{k:None for k in f['values']}})
                points.append({'t_ms':f['t_ms'],'epoch':f['epoch'],'values':{k:v['value'] for k,v in f['values'].items()}})
                last=f['t_ms'];epoch=f['epoch']
        return {'step_ms':step,'points':points}

    def segment(self,start,end):
        with self.lock:return self._segment(start,end)

    def _segment(self,start,end):
        if end-start+1>60000:raise ValueError('한 번에 최대 60000행을 참조할 수 있습니다.')
        items=[]
        for n,c in enumerate(self.archive.data['chunks']):
            if c['end_sequence']<start or c['start_sequence']>end:continue
            items.extend(i for i in self._chunk(n) if start<=i['sequence']<=end)
        old=self.entries
        try:self.entries=items;return super().segment(start,end)
        finally:self.entries=old
