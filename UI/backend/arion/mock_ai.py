"""Explicitly artificial local API for integration checks: port 8766."""
from pathlib import Path
from fastapi import FastAPI, HTTPException

from .storage import Store
from .ai import fixture_events

app = FastAPI(title="ARION 시험용 AI API — 실제 모델 아님")
store = Store(Path(__file__).resolve().parents[2] / ".data")


@app.get("/results")
def results(run_id: str, input_sha256: str):
    try:
        run = store.get(run_id)
    except KeyError as error:
        raise HTTPException(404, "비행 없음") from error
    if input_sha256 != run.sha256:
        raise HTTPException(409, "입력 해시 불일치")
    events = fixture_events(run)
    for event in events:
        event['event_id'] = 'api-' + event['event_id']
    return {"schema_version": "arion-ai-results-v1", "run_id": run_id,
            "input_sha256": run.sha256, "state": "ready", "results": events}

import asyncio
import os
from urllib.parse import urlparse
import httpx

DASHBOARD_URL=os.environ.get('ARION_DASHBOARD_URL','http://127.0.0.1:8765')
if urlparse(DASHBOARD_URL).hostname not in ('127.0.0.1','localhost','::1'):
    raise ValueError('시험 API는 로컬 대시보드만 참조합니다.')

@app.get('/{provider}/results')
async def results_v2(provider: str,run_id: str,input_sha256: str='',latest_sequence: int=0,mode: str='ready',delay_ms: int=0):
    if provider not in ('lstm','transformer'):raise HTTPException(404,'시험 제공자 없음')
    if delay_ms:await asyncio.sleep(min(max(0,delay_ms),5000)/1000)
    if mode=='error':raise HTTPException(503,'의도적인 시험 API 오류')
    base={'schema_version':'arion-provider-results-v2','provider_id':provider,'run_id':run_id,
          'state':'warming_up' if mode=='warming_up' else 'ready','results':[]}
    if mode=='warming_up':return base
    async with httpx.AsyncClient(base_url=DASHBOARD_URL,timeout=3,trust_env=False) as client:
        response=await client.get('/api/runs/'+run_id);response.raise_for_status();manifest=response.json()
        if manifest.get('source_kind')=='live':
            if latest_sequence<3:return dict(base,state='warming_up')
            # New fixture every 20 source records. This is a UI test, never an anomaly model.
            end=(latest_sequence//20)*20
            if end<3:return dict(base,state='warming_up')
            start=max(1,end-19)
            segment=await client.get(f'/api/runs/{run_id}/input-segment',params={'start_sequence':start,'end_sequence':end})
            segment.raise_for_status();data=segment.json();frames=data['frames']
            candidates=[(frames[0]['t_ms'],frames[-1]['t_ms'],frames[-1]['t_ms'],frames[-1]['t_ms']+3000,
                         frames[-1]['epoch'],data['input_ref'],f'stream-{end}')]
        else:
            duration=manifest['duration_ms'];candidates=[]
            if input_sha256!=manifest['input_sha256']:raise HTTPException(409,'입력 해시 불일치')
            for n,fraction in enumerate((.30,.60,.82)):
                end=round(duration*fraction,3)
                frame=(await client.get(f'/api/runs/{run_id}/frame',params={'t_ms':end})).json()
                epoch=frame['epoch'];begins=[b['t_ms'] for b in manifest['boundaries'] if b['epoch']==epoch]
                boundary=next((b['t_ms']-.001 for b in manifest['boundaries'] if b['epoch']==epoch+1),duration)
                candidates.append((max(begins[0] if begins else 0,end-1800),end,end,min(boundary,end+4000),epoch,
                                   {'kind':'file','sha256':input_sha256},f'file-{n}'))
        for start,end,decided,valid,epoch,ref,eid in candidates:
            suspect=provider=='lstm' and not eid.endswith('2')
            base['results'].append({'schema_version':'arion-analysis-v2','provider_id':provider,'run_id':run_id,
                'event_id':f'api-{provider}-{eid}','input_ref':ref,'epoch':epoch,'start_ms':start,'end_ms':end,
                'decided_at_ms':decided,'valid_until_ms':valid,'model_version':provider+'-fixture-v2',
                'kind':'detector','status':'suspect' if suspect else 'normal','target':'powertrain',
                'score':.82 if suspect else .56,'threshold':.65,'title':provider+' 비교용 시험 판정',
                'description':'화면과 연결 검증용 인공 결과입니다. 실제 고장 또는 모델 추론이 아닙니다.',
                'evidence':[],'origin':'fixture'})
    return base
