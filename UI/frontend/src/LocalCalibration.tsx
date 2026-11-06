import { useEffect, useState } from 'react';
import { api } from './api';
import type { Job } from './local-flight-types';

type Calibration = { id:string; model_id:string; profile_id:string; threshold:number; quantile:number; normal_flights:number; active:boolean; binding:{weight_sha256:string;clock:string} };
export default function LocalCalibration({jobs,model,disabled,submit}:{jobs:Job[];model:string;disabled:boolean;submit:(path:string,body:unknown)=>Promise<void>}) {
  const [selected,setSelected]=useState<string[]>([]),[quantile,setQuantile]=useState(.99),[adapted,setAdapted]=useState(true);
  const [records,setRecords]=useState<Calibration[]>([]),[error,setError]=useState('');
  const revision=jobs.map(j=>j.id+j.status).join(',');
  useEffect(()=>{let stopped=false; api<{calibrations:Calibration[]}>('/local-ai/calibrations').then(r=>{if(!stopped)setRecords(r.calibrations);}).catch(e=>{if(!stopped)setError(String(e));});return()=>{stopped=true;};},[revision]);
  return <section className="panel"><h2>기체별 임계값 보정</h2>
    <p>학습·모델 선택·평가와 다른 정상 비행 2~20개를 사용합니다. 선택 모델 A: {model || '미선택'}</p>
    <p>예측 모델은 비행별 20개 이상의 1초 앞 예측/실측 쌍이 필요합니다. 경험적 기준이며 지정 백분위가 실제 오경보율을 보장하지 않습니다.</p>
    {jobs.filter(j=>j.status==='complete'&&j.action==='prepare'&&j.result?.role==='calibration'&&j.result?.health==='confirmed_normal').map(j=><label className="local-segment" key={j.id}><input type="checkbox" checked={selected.includes(j.id)} onChange={()=>setSelected(s=>s.includes(j.id)?s.filter(id=>id!==j.id):[...s,j.id])}/>{j.id.slice(0,8)} · {j.result?.frames}시점</label>)}
    <label className="form-label">경험적 기준 백분위<select className="text-input" value={quantile} onChange={e=>setQuantile(Number(e.target.value))}><option value={.95}>95%</option><option value={.99}>99%</option><option value={.999}>99.9%</option></select></label>
    <label><input type="checkbox" checked={adapted} onChange={e=>setAdapted(e.target.checked)}/> 이 기체에서 채택한 파인튜닝 모델 보정 · 없으면 원본</label>
    <button className="button primary" disabled={disabled||!model||selected.length<2||selected.length>20} onClick={()=>submit('/local-ai/calibrations',{model_id:model,prepared_jobs:selected,quantile,use_active:adapted})}>새 보정값 만들기</button>
    <p>생성 후 아래에서 ‘판정에 사용’을 선택하세요. 기체·가중치·입력 방식이 일치하는 파일 분석과 실시간 분석에 적용됩니다. 과거 버전을 다시 선택하거나 적용을 해제할 수 있습니다.</p>
    {error&&<p role="alert">{error}</p>}
    {records.map(c=><article key={c.id} className="local-job"><strong>{c.profile_id} · {c.model_id}</strong><p>가중치 {c.binding.weight_sha256.slice(0,8)} · {c.binding.clock==='receipt_monotonic'?'실시간 수신 기록':'온보드 로그'}<br/>정상 {c.normal_flights}회 · 기준 {c.threshold.toFixed(5)} · {c.active?'판정에 사용 중':'미선택'}</p><button className="button small" disabled={disabled} onClick={()=>submit('/local-ai/calibrations/select',{id:c.id,enabled:!c.active})}>{c.active?'적용 해제':'판정에 사용'}</button></article>)}
  </section>;
}
