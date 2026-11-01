import asyncio
import hashlib
import json
import time
from urllib.parse import urlparse
import httpx
from .contracts import AnalysisResponse, AnalysisResult
from .streaming import SessionRun
from .telemetry import LABELS


def contract_for(run):
    if isinstance(run,SessionRun):
        return {'signals':{k:v.unit for k,v in run.profile.signals.items()},'sample_period_ms':run.profile.sample_period_ms,
                'preprocess_version':run.profile.preprocess_version,'profile_id':run.profile.profile_id,
                'schema_version':'arion-processed-v1' if run.config.kind=='normalized' else 'arion-telemetry-v2',
                'feature_order':run.profile.feature_order}
    return {'signals':{k:v[1] for k,v in LABELS.items()},'sample_period_ms':None,'preprocess_version':'raw-v1'}


def compatibility(run,provider):
    c=contract_for(run);req=provider.requirements
    reasons=[]
    for signal,unit in req.signals.items():
        if c['signals'].get(signal)!=unit:reasons.append(f'{signal} 단위/특징 불일치')
    if req.sample_period_ms is not None and req.sample_period_ms!=c['sample_period_ms']:reasons.append('입력 주기 불일치')
    if req.preprocess_version and req.preprocess_version!=c['preprocess_version']:reasons.append('전처리 버전 불일치')
    return reasons


def validate_result(run,provider,event):
    e=AnalysisResult.model_validate(event)
    if e.run_id!=run.run_id or e.provider_id!=provider.provider_id:raise ValueError('비행/제공자 불일치')
    if e.kind!=provider.kind:raise ValueError('결과 종류 불일치')
    if provider.model_version!='unconfigured' and e.model_version!=provider.model_version:raise ValueError('모델 버전 불일치')
    if compatibility(run,provider):raise ValueError('입력 호환 안 됨')
    end=run.now_ms if getattr(run,'live',False) else run.duration_ms
    if e.decided_at_ms>end or e.epoch>=len(run.epoch_starts) or run.epoch_at(e.start_ms)!=e.epoch or run.epoch_at(e.decided_at_ms)!=e.epoch:
        raise ValueError('미래 시각 또는 epoch/비행 경계 오류')
    if not getattr(run,'live',False) and e.valid_until_ms>end:raise ValueError('유효기간이 비행 범위를 벗어납니다.')
    if e.epoch+1<len(run.epoch_starts) and e.valid_until_ms>=run.epoch_starts[e.epoch+1]:raise ValueError('유효기간이 epoch 경계를 넘습니다.')
    ref=e.input_ref
    if ref.kind=='file':
        if not run.sha256 or ref.sha256!=run.sha256:raise ValueError('원본 파일 해시 불일치')
    else:
        if not isinstance(run,SessionRun) or ref.session_id!=run.run_id:raise ValueError('실시간 세션 불일치')
        if ref.start_sequence is None or ref.end_sequence is None:raise ValueError('입력 sequence 필요')
        expected,items=run.segment(ref.start_sequence,ref.end_sequence)
        if ref.segment_sha256!=expected['segment_sha256'] or ref.preprocess_version!=expected['preprocess_version']:raise ValueError('입력 구간 해시/전처리 불일치')
        if items[0]['epoch']!=e.epoch or items[-1]['t_ms']>e.decided_at_ms or e.start_ms<items[0]['t_ms'] or e.end_ms>items[-1]['t_ms']:
            raise ValueError('근거 구간과 입력 참조 불일치')
    if e.kind=='detector' and e.status in ('normal','suspect') and provider.requirements.signals:
        frames=[run.frame(e.start_ms),run.frame(e.end_ms)]
        if ref.kind=='stream':frames.extend(x['frame'] for x in items)
        for frame in frames:
            for signal in provider.requirements.signals:
                if frame['values'].get(signal,{}).get('status')!='valid':
                    raise ValueError(f'필수 입력이 결측·만료된 판정: {signal}')
    units=contract_for(run)['signals']
    for evidence in e.evidence:
        if units.get(evidence.signal)!=evidence.unit:raise ValueError('근거 신호 또는 단위 불일치')
    return e.model_dump()

class ProviderRegistry:
    def __init__(self,store):
        self.store=store;self.tasks={};self.states={};self.received={};self.demo=set();self.api_keys={};self.demo_keys={}

    @staticmethod
    def key(e):return ':'.join([e['run_id'],e['provider_id'],e['origin'],e['model_version'],e['event_id']])

    def states_for(self,run):
        output=[]
        for raw in self.store.providers():
            if raw['provider_id']=='legacy':continue
            p=self.store.provider(raw['provider_id']);key=(run.run_id,p.provider_id)
            mismatch=compatibility(run,p)
            state=self.states.get(key,{'state':'disconnected','message':'','url':''})
            if mismatch:state=dict(state,state='incompatible',message=' · '.join(mismatch))
            output.append(dict(raw,**state,input_compatible=not mismatch))
        return output

    async def connect(self,run_id,provider_id,url):
        run=self.store.get(run_id);p=self.store.provider(provider_id)
        reasons=compatibility(run,p)
        if reasons:raise ValueError('입력 호환 안 됨: '+' · '.join(reasons))
        parsed=urlparse(url)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.password or parsed.fragment:
            raise ValueError('로컬 HTTP 결과 API만 지원합니다.')
        await self.disconnect(run_id,provider_id)
        self.demo.discard(run_id)
        key=(run_id,provider_id);self.api_keys[key]=set();self.states[key]={'state':'connecting','message':'','url':url}
        self.tasks[key]=asyncio.create_task(self._poll(run_id,p,url))

    async def disconnect(self,run_id,provider_id):
        key=(run_id,provider_id);task=self.tasks.pop(key,None)
        if task:
            task.cancel()
            try:await task
            except asyncio.CancelledError:pass
        self.states[key]={'state':'disconnected','message':'','url':''}

    async def _poll(self,run_id,p,url):
        key=(run_id,p.provider_id)
        async with httpx.AsyncClient(timeout=3,trust_env=False,follow_redirects=False) as client:
            while True:
                try:
                    run=self.store.get(run_id)
                    reasons=compatibility(run,p)
                    if reasons:raise ValueError('입력 호환 안 됨: '+' · '.join(reasons))
                    params={'run_id':run_id,'provider_id':p.provider_id,'input_sha256':run.sha256 or '',
                            'latest_sequence':getattr(run,'accepted',0)}
                    response=await client.get(url,params=params);response.raise_for_status()
                    payload=AnalysisResponse.model_validate(response.json())
                    if payload.run_id!=run_id or payload.provider_id!=p.provider_id:raise ValueError('응답 비행/제공자 불일치')
                    events=[validate_result(run,p,e) for e in payload.results]
                    known={self.key(e):e for e in self.store.analysis(run_id)}
                    for e in events:
                        if any(ref not in {x['event_id'] for x in known.values()} for ref in e['related_event_ids']):raise ValueError('존재하지 않는 관련 이벤트')
                    self.store.save_analysis(events)
                    self.api_keys.setdefault(key,set()).update(self.key(e) for e in events)
                    for e in events:
                        if known.get(self.key(e))!=e:self.received[self.key(e)]=(time.perf_counter(),e['decided_at_ms'])
                    # Bound diagnostics bookkeeping; file journals retain complete event history.
                    if len(self.received)>5000:self.received=dict(list(self.received.items())[-4000:])
                    self.states[key]={'state':payload.state,'message':'','url':url,'last_sync_utc_ms':time.time()*1000,
                                      'fixture_only':bool(events) and all(e['origin']=='fixture' for e in events)}
                except (httpx.HTTPError,ValueError,TypeError) as e:
                    self.states[key]={'state':'error','message':str(e)[:500],'url':url}
                await asyncio.sleep(p.poll_interval_ms/1000)

    def snapshot(self,run,cursor):
        states=self.states_for(run);by_id={s['provider_id']:s for s in states}
        events=[]
        for e in self.store.analysis(run.run_id,cursor):
            if e['provider_id']=='legacy':continue # legacy snapshot is served by the original compatibility route
            state=by_id.get(e['provider_id'],{})
            if e['origin']=='fixture':
                allowed=self.demo_keys.get(run.run_id,set()) if run.run_id in self.demo else self.api_keys.get((run.run_id,e['provider_id']),set()) if state.get('fixture_only') else set()
                if self.key(e) not in allowed:continue
            if e['decided_at_ms']<=cursor:events.append(e)
        events.sort(key=lambda e:(e['decided_at_ms'],e['provider_id'],e['event_id']))
        if getattr(run,'live',False) and cursor-run.duration_ms>run.profile.gap_ms:
            for s in states:s.update(state='unavailable',message='입력 수신이 만료되었습니다.')
        active={}
        for s in states:
            xs=[e for e in events if e['provider_id']==s['provider_id'] and e['epoch']==run.epoch_at(cursor) and e['valid_until_ms']>=cursor]
            required=self.store.provider(s['provider_id']).requirements.signals
            if required and any(run.frame(cursor)['values'].get(k,{}).get('status')!='valid' for k in required):
                s.update(state='unavailable',message='필수 입력이 결측 또는 만료되었습니다.')
            usable=s['state'] not in ('error','incompatible','connecting','warming_up','unavailable')
            active[s['provider_id']]=xs[-1] if xs and usable else None
            if s['state']=='disconnected' and any(e['provider_id']==s['provider_id'] and e['origin']=='model' for e in events):s['state']='saved'
        return states,events,active

    def fixtures(self,run):
        if isinstance(run,SessionRun) and run.live:raise ValueError('실시간 시험은 별도 시험 API를 사용하세요.')
        from .ai import fixture_events
        rows=[]
        for pid in ('lstm','transformer'):
            for e in fixture_events(run):
                e.pop('input_sha256');e.update(provider_id=pid,kind='detector',schema_version='arion-analysis-v2',
                    input_ref={'kind':'file','sha256':run.sha256},model_version=f'{pid}-fixture-v2')
                if pid=='transformer' and e['status']=='suspect':e.update(score=.56,threshold=.65,status='normal',title='정상 범위 · 비교용 시험 응답')
                rows.append(validate_result(run,self.store.provider(pid),e))
        self.store.save_analysis(rows)
        self.demo_keys[run.run_id]={self.key(e) for e in rows}

    async def close(self):
        for run_id,pid in list(self.tasks):await self.disconnect(run_id,pid)
