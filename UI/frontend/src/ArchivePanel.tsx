import { useEffect, useState } from 'react';
import { Archive, ChevronDown, Download, LoaderCircle, RotateCcw } from 'lucide-react';
import { api, post } from './api';
import type { Frame, Manifest, NpzChunk, NpzState, ExportJob } from './types';
import { numeric, timeLabel } from './types';
import './archive.css';

const labels:Record<string,string>={recording:'구간 기록 중',saving:'NPZ 저장 중',saved:'저장 완료',error:'저장 오류'};
export function ArchivePanel({run,frame,expanded=false}:{run:Manifest;frame:Frame|null;expanded?:boolean}) {
  const [data,setData]=useState<{npz:NpzState|null;chunks:NpzChunk[]}|null>(null);
  const [job,setJob]=useState<ExportJob|null>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false);
  const [open,setOpen]=useState(expanded);
  const applicable=run.profile?.input_format==='normalized';
  useEffect(()=>{
    setData(null);setJob(null);setError('');setOpen(expanded);
    if(!applicable)return;
    let stale=false;
    const update=()=>api<{npz:NpzState|null;chunks:NpzChunk[]}>(`/runs/${run.run_id}/npz/chunks`).then(d=>{if(!stale)setData(d);}).catch(e=>{if(!stale)setError(e.message);});
    update();const timer=setInterval(update,2000);return()=>{stale=true;clearInterval(timer);};
  },[run.run_id,applicable,expanded]);
  useEffect(()=>{
    if(!job||!['queued','running'].includes(job.state))return;
    let stale=false;
    const timer=setTimeout(()=>api<ExportJob>(`/exports/${job.job_id}`).then(j=>{if(!stale)setJob(j);}).catch(e=>{if(!stale){setError(e.message);setJob({...job,state:'error',error:e.message});}}),500);
    return()=>{stale=true;clearTimeout(timer);};
  },[job]);
  const state=data?.npz??frame?.manifest?.npz??run.npz;
  const exportFile=async()=>{setBusy(true);setError('');try{setJob(await post<ExportJob>(`/runs/${run.run_id}/npz/export`));}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
  const retry=async()=>{setBusy(true);try{await post(`/runs/${run.run_id}/npz/retry`);setError('');}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
  if(!applicable)return expanded?<section className="panel archive-raw"><Archive size={19}/><div><strong>원본 텔레메트리 기록</strong><p>이 비행은 원본 JSONL 경로입니다. NPZ 저장은 외부 전처리 스트림 입력에 적용됩니다.</p></div></section>:null;
  const live=run.source_kind==='live';
  return <section className={`panel archive-panel ${expanded?'expanded':''}`} aria-label="전처리와 NPZ 저장">
    <div className="archive-bar"><button className="archive-title" onClick={()=>setOpen(!open)} aria-expanded={open}><Archive size={18}/><span><strong>전처리 → 화면 + NPZ</strong><small>{run.profile?.sample_period_ms?`${numeric(1000/run.profile.sample_period_ms,1)}Hz 전처리`:'출력 주기 미확인'} · 공통 계약 v1</small></span><ChevronDown size={14}/></button>
      <span className={`archive-state ${state?.state==='error'?'error':''}`} role="status"><i/>{labels[state?.state??'recording']}<small>{(state?.saved_rows??0).toLocaleString()} / {(state?.received_rows??run.accepted_records).toLocaleString()}행 · {state?.chunk_count??0}구간</small></span>
      <div className="archive-actions">{state?.state==='error'&&<button className="button small" disabled={busy} onClick={retry}><RotateCcw size={14}/>저장 재시도</button>}{job?.state==='ready'?<a className="button small" href={job.download_url!} download><Download size={14}/>전체 NPZ 다운로드</a>:<button className="button small" disabled={live||busy||job?.state==='queued'||job?.state==='running'||!state?.saved_rows} onClick={exportFile}>{job&&['queued','running'].includes(job.state)?<LoaderCircle size={14} className="spin"/>:<Download size={14}/>}전체 NPZ 내보내기</button>}</div>
    </div>
    {(error||state?.error||job?.state==='error')&&<div className="archive-error" role="alert">{error||state?.error||job?.error}</div>}
    {open&&<div className="archive-details"><p>수락한 프레임을 복구용 JSONL에 먼저 기록하고 같은 값·mask·관측 나이를 60초 구간 NPZ에 저장합니다. {live?'전체 내보내기는 수집 종료 후 사용할 수 있습니다.':'전체 내보내기는 저장 구간을 합치는 백그라운드 작업입니다.'}</p><dl className="compact-meta"><div><dt>프로필 / 전처리 버전</dt><dd>{run.profile?.profile_id} / {run.profile?.preprocess_version}</dd></div><div><dt>현재 구간 대기</dt><dd>{state?.pending_rows??0}행</dd></div><div><dt>입력 주소</dt><dd><code>POST /api/sessions/{run.run_id}/frames</code></dd></div></dl><div className="table-scroll"><table><thead><tr><th>저장 구간</th><th>Epoch</th><th>행 수</th><th>입력 순서</th></tr></thead><tbody>{(data?.chunks??[]).map(c=><tr key={c.filename}><td>{timeLabel(c.start_ms,true)}–{timeLabel(c.end_ms,true)}</td><td>{c.epoch}</td><td>{c.rows}</td><td>{c.start_sequence}–{c.end_sequence}</td></tr>)}</tbody></table></div>{!data?.chunks.length&&<p className="muted">60초 도달, epoch 변경 또는 수집 종료 시 첫 구간을 저장합니다.</p>}</div>}
  </section>;
}
