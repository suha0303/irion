import { useEffect, useRef, useState } from 'react';
import { api, upload } from './api';
import type { Job } from './local-flight-types';

type SavedModel = {id: string; base_model_id: string; profile: {name: string}; origin: string};
type Selection = {id: string; key: string; selected: string | null; previous: string | null; profile_id: string; base_model_id: string};
type Library = {models: SavedModel[]; selection: {active: Record<string,string|null>; history: Selection[]}};
export default function LocalModels({ jobs, disabled, submit, imported }: {jobs:Job[];disabled:boolean;submit:(path:string,body:unknown)=>Promise<void>;imported:(job:Job)=>void}) {
  const [library,setLibrary] = useState<Library>({models:[],selection:{active:{},history:[]}});
  const [error,setError] = useState('');
  const [adapted,setAdapted] = useState('');
  const [model,setModel] = useState('');
  const [evaluation,setEvaluation] = useState('');
  const [comparison,setComparison] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const revision = jobs.filter(j=>j.action.startsWith('model_') && j.status==='complete').map(j=>j.id).join(',');
  const ready = (action:string) => jobs.filter(j=>j.action===action && j.status==='complete');
  useEffect(()=>{
    let active=true;
    api<Library>('/local-ai/models').then(value=>{if(active){setLibrary(value);setError('');}}).catch(e=>{if(active)setError(String(e));});
    return ()=>{active=false;};
  },[revision]);
  const latest = library.selection.history.filter((e,i,all)=>!all.slice(i+1).some(next=>next.key===e.key));
  return <section className="panel"><h2>6. 적응 모델 보관·선택·원복</h2>
    <p>기체 프로파일 버전과 원본 모델별로 선택합니다. 비교 결과가 있어야 적응 모델을 선택할 수 있습니다. 선택해도 건강 판정 임계값은 자동 적용되지 않습니다.</p>
    {error && <p role="alert">{error}</p>}
    <label className="form-label">보관할 적응 작업<select className="text-input" value={adapted} onChange={e=>setAdapted(e.target.value)}><option value="">선택</option>{ready('adapt').map(j=><option key={j.id} value={j.id}>{j.id.slice(0,8)}</option>)}</select></label>
    <button className="button" disabled={disabled || !adapted} onClick={()=>submit('/local-ai/models/register',{adapted_job:adapted})}>모델 보관함에 등록</button>
    <input hidden ref={input} type="file" accept=".zip" onChange={e=>{const file=e.target.files?.[0];if(file)upload<Job>('/local-ai/models/import',file).then(imported).catch(e=>setError(String(e)));e.target.value='';}}/>
    <button className="button" disabled={disabled} onClick={()=>input.current?.click()}>모델 백업 가져오기</button>
    <label className="form-label">보관된 모델<select className="text-input" value={model} onChange={e=>setModel(e.target.value)}><option value="">선택</option>{library.models.map(m=><option key={m.id} value={m.id}>{m.profile.name} · {m.base_model_id} · {m.id.slice(0,8)} · {m.origin==='synthetic_unit_fixture'?'합성 시험':'실기체 적응'}</option>)}</select></label>
    <button className="button" disabled={disabled || !model} onClick={()=>submit(`/local-ai/models/${model}/export`,{})}>백업 파일 만들기</button>
    <label className="form-label">독립 평가 비행<select className="text-input" value={evaluation} onChange={e=>setEvaluation(e.target.value)}><option value="">선택</option>{ready('prepare').filter(j=>j.result?.role==='evaluation').map(j=><option key={j.id} value={j.id}>{j.id.slice(0,8)}</option>)}</select></label>
    <button className="button" disabled={disabled || !model || !evaluation} onClick={()=>submit(`/local-ai/models/${model}/compare`,{prepared_job:evaluation})}>보관 모델 독립 비교</button>
    <label className="form-label">확인한 비교 작업<select className="text-input" value={comparison} onChange={e=>setComparison(e.target.value)}><option value="">선택</option>{ready('compare').map(j=><option key={j.id} value={j.id}>{j.id.slice(0,8)}</option>)}</select></label>
    <button className="button primary" disabled={disabled || !model || !comparison} onClick={()=>submit(`/local-ai/models/${model}/select`,{comparison_job:comparison})}>이 기체의 분석 모델로 선택</button>
    <p>5번 분석 화면에서 ‘선택한 적응 모델 사용’을 켜면 적용됩니다. 원본 A/B 분석도 계속 사용할 수 있습니다.</p>
    {latest.map(e=><div className="local-job" key={e.id}><p>{e.profile_id} · {e.base_model_id}<br/>현재: {e.selected?.slice(0,12) ?? '원본 모델'}</p><button className="button small" disabled={disabled} onClick={()=>submit('/local-ai/models/rollback',{event_id:e.id})}>이전 선택으로 원복</button></div>)}
    {ready('model_export').map(j=><p key={j.id}><a href={`/api/local-ai/model-exports/${j.id}`}>모델 백업 다운로드 · {j.id.slice(0,8)}</a></p>)}
    <small>백업에는 가중치·고정 scaler·기체 프로파일·학습 계보가 포함됩니다. 원시 비행 로그와 실행 코드는 포함하지 않습니다.</small>
  </section>;
}
