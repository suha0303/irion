import { useEffect, useState } from 'react';
import { Plug, Radio, X, Activity, Settings2 } from 'lucide-react';
import { api, post } from './api';
import type { Frame, InputProfile, Manifest, ProviderState } from './types';
import { numeric, statusLabel, targetLabel, timeLabel } from './types';

export function ProviderCards({frame,connected,configure}:{frame:Frame|null;connected:boolean;configure:(id:string)=>void}) {
  const providers=frame?.providers??[{provider_id:'lstm',name:'LSTM'},{provider_id:'transformer',name:'Transformer'}] as ProviderState[];
  return <div className="provider-grid">{providers.map(p=>{
    const active=connected?frame?.active_by_provider?.[p.provider_id]:null;
    const state=connected?(active?statusLabel[active.status]:statusLabel[p.state]??'미연결'):'서버 연결 중';
    return <section className="panel provider-card" key={p.provider_id}><div className="card-label"><Activity size={17}/><b>{p.name}</b><button className="icon-button" aria-label={`${p.name} API 설정`} onClick={()=>configure(p.provider_id)}><Settings2 size={16}/></button></div>
      <div className={`provider-status ${active?.status==='suspect'?'suspect':''}`}>{state}</div>
      {active?<><small>{targetLabel[active.target]??active.target} · {active.model_version}</small>{active.kind==='explanation'?<p>{active.description}</p>:<p>점수 <b>{numeric(active.score,3)}</b> / 임계값 {numeric(active.threshold,3)}</p>}<small>근거 {timeLabel(active.start_ms,true)}–{timeLabel(active.end_ms,true)}</small>{active.origin==='fixture'&&<span className="pill amber">시험용 응답</span>}</>:<p>{p.message||'입력·모델이 준비되면 결과를 표시합니다.'}</p>}
      <div className="card-foot">{p.input_compatible===false?'입력 계약 확인 필요':`결과 확인 주기 ${p.poll_interval_ms??500}ms · 추론 주기와 구분`}</div></section>;
  })}</div>;
}

export function ProviderDialog({runId,initial,providers,close}:{runId:string;initial:string;providers:ProviderState[];close:()=>void}) {
  const [id,setId]=useState(initial),[url,setUrl]=useState(`http://127.0.0.1:8766/${initial}/results`),[error,setError]=useState(''),[busy,setBusy]=useState(false);
  const submit=async(connect:boolean)=>{setBusy(true);setError('');try{await post(`/runs/${runId}/analysis/${id}/${connect?'connect':'disconnect'}`,connect?{url}:{});close();}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
  return <div className="modal-backdrop" onClick={close}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="provider-heading" onClick={e=>e.stopPropagation()}><div className="panel-heading"><h2 id="provider-heading">분석 제공자 연결</h2><button className="icon-button" aria-label="설정 닫기" onClick={close}><X size={18}/></button></div><div className="inner">
    <label className="form-label">분석 제공자<select className="text-input" value={id} onChange={e=>{setId(e.target.value);setUrl(`http://127.0.0.1:8766/${e.target.value}/results`);}}>{providers.filter(p=>p.provider_id!=='legacy').map(p=><option key={p.provider_id} value={p.provider_id}>{p.name}</option>)}</select></label>
    <label className="form-label">결과 API 주소<input className="text-input" value={url} onChange={e=>setUrl(e.target.value)}/></label><p>각 모델을 별도로 연결합니다. 입력 계약·비행·근거 구간이 일치하는 결과만 표시합니다.</p><div className="notice warning">기본 주소는 별도로 실행하는 시험 API입니다. 실제 LSTM·Transformer 모델은 포함되어 있지 않습니다.</div>
    {error&&<div className="notice error" role="alert">{error}</div>}<div className="modal-actions"><button className="button" disabled={busy} onClick={()=>submit(false)}>연결 해제</button><button className="button primary" disabled={busy} onClick={()=>submit(true)}><Plug size={15}/>API 연결</button></div></div></section></div>;
}

export function SourceDialog({close,created}:{close:()=>void;created:(m:Manifest)=>void}) {
  const [kind,setKind]=useState('udp'),[profiles,setProfiles]=useState<InputProfile[]>([]),[profile,setProfile]=useState('mavlink_common_v1');
  const [name,setName]=useState('실시간 비행'),[port,setPort]=useState('14560'),[device,setDevice]=useState(''),[baud,setBaud]=useState('115200'),[source,setSource]=useState('');
  const [error,setError]=useState(''),[busy,setBusy]=useState(false);
  useEffect(()=>{api<InputProfile[]>('/profiles').then(setProfiles).catch(e=>setError(e.message));},[]);
  const submit=async()=>{setBusy(true);setError('');try{const m=await post<Manifest>('/sessions',{kind,name,profile_id:profile,host:'127.0.0.1',port:Number(port),device,baud:Number(baud),source_system:source?Number(source):null});created(m);close();}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
  return <div className="modal-backdrop" onClick={close}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="source-heading" onClick={e=>e.stopPropagation()}><div className="panel-heading"><h2 id="source-heading">실시간 입력 연결</h2><button className="icon-button" aria-label="설정 닫기" onClick={close}><X size={18}/></button></div><div className="inner">
    <label className="form-label">세션 이름<input className="text-input" value={name} onChange={e=>setName(e.target.value)}/></label>
    <label className="form-label">입력 방식<select className="text-input" value={kind} onChange={e=>{setKind(e.target.value);setProfile(e.target.value==='normalized'?'normalized_flight_v1':'mavlink_common_v1');}}><option value="udp">MAVLink UDP</option><option value="serial">MAVLink 시리얼</option><option value="normalized">공용 수집기 / 전처리 스트림</option></select></label>
    <label className="form-label">데이터 프로필<select className="text-input" value={profile} onChange={e=>setProfile(e.target.value)}>{profiles.filter(p=>(p.input_format==='normalized')===(kind==='normalized')).map(p=><option key={p.profile_id}>{p.profile_id}</option>)}</select></label>
    {kind==='udp'&&<label className="form-label">전용 UDP 포트 · 127.0.0.1<input className="text-input" type="number" value={port} onChange={e=>setPort(e.target.value)}/></label>}
    {kind==='serial'&&<><label className="form-label">시리얼 장치<input className="text-input" placeholder="/dev/cu.usbmodem…" value={device} onChange={e=>setDevice(e.target.value)}/></label><label className="form-label">Baud rate<input className="text-input" type="number" value={baud} onChange={e=>setBaud(e.target.value)}/></label></>}
    {kind!=='normalized'&&<label className="form-label">Source system · 비워두면 첫 유효 기체<input className="text-input" type="number" value={source} onChange={e=>setSource(e.target.value)}/></label>}
    <p>{kind==='normalized'?'세션을 만든 뒤 공통 프레임 API로 데이터를 전달하세요. 주기·단위·mask·age를 검사하며 추가 전처리는 하지 않습니다.':'전용 링크로 전달된 텔레메트리만 수신합니다. 기존 QGC·공용 수집기와 같은 포트를 사용하지 마세요.'}</p>
    {error&&<div className="notice error" role="alert">{error}</div>}<div className="modal-actions"><button className="button" onClick={close}>닫기</button><button className="button primary" disabled={busy} onClick={submit}><Radio size={15}/>수집 시작</button></div></div></section></div>;
}
