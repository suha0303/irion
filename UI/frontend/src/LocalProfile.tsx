import { useEffect, useRef, useState } from 'react';
import { post } from './api';
import type { Aircraft, Job, Profile, ProfileSuggestion } from './local-flight-types';

export const emptyProfile = ():Aircraft => ({schema:'drone.real-aircraft/1',id:'',name:'',autopilot:'unknown',firmware:'',rotor_count:null,series_cells:null,capacity_ah:null,battery_id:'',motor_channels:{},output_min:null,output_max:null,mapping_evidence:'',notes:'',imu_instance:null,battery_instance:null,actuator_instance:null});
const fields = [
  ['rotor_count','로터 수'],['series_cells','배터리 직렬 셀 수'],['capacity_ah','정격 용량 (Ah)'],
  ['output_min','모터 출력 최솟값'],['output_max','모터 출력 최댓값'],['imu_instance','분석할 IMU 인스턴스'],
  ['battery_instance','배터리 인스턴스'],['actuator_instance','출력 인스턴스'],
] as const;
const positions = [['front_right','앞 오른쪽'],['front_left','앞 왼쪽'],['rear_right','뒤 오른쪽'],['rear_left','뒤 왼쪽']] as const;
export function missingProfileFields(p:Aircraft):string[] {
  const missing:string[] = fields.filter(([key])=>p[key]==null).map(([key])=>key);
  if(p.autopilot==='unknown')missing.push('autopilot');
  if(positions.some(([pos])=>p.motor_channels[pos]==null))missing.push('motor_channels');
  return missing;
}
export function editableProfileFields(p:Aircraft):string[] {return [...new Set([...missingProfileFields(p),'series_cells'])];}
export function fromLog(s:ProfileSuggestion):Aircraft {return {...emptyProfile(),...s.profile};}
export function editProfile(p:Aircraft,key:string,value:Aircraft[keyof Aircraft]):Aircraft {
  const changed={...p,[key]:value};
  if(p.provenance?.fields){const evidence={...p.provenance.fields};delete evidence[key];changed.provenance={...p.provenance,fields:evidence};}
  return changed;
}

export default function LocalProfile({profiles,jobs,pendingLog,disabled,onSaved,reinspect}:{profiles:Profile[];jobs:Job[];pendingLog:string;disabled:boolean;onSaved:(p:Profile)=>void;reinspect:(id:string)=>Promise<void>}) {
  const [profile,setProfile]=useState<Aircraft>(emptyProfile),[source,setSource]=useState('');
  const [notice,setNotice]=useState(''),[error,setError]=useState(''),[saving,setSaving]=useState(false);
  const consumed=useRef('');
  const [requested,setRequested]=useState<string[]>(()=>editableProfileFields(emptyProfile()));
  function openProfile(p:Aircraft){setProfile(p);setRequested(editableProfileFields(p));}
  const inspections=jobs.filter(j=>j.action==='inspect'&&j.status==='complete');
  const selected=inspections.find(j=>j.id===source), suggestion=selected?.result?.profile_suggestion;
  useEffect(()=>{
    const job=jobs.find(j=>j.id===pendingLog&&j.status==='complete');
    if(!job?.result?.profile_suggestion||consumed.current===pendingLog)return;
    consumed.current=pendingLog;setSource(pendingLog);
    const proposed=fromLog(job.result.profile_suggestion);
    if(JSON.stringify(profile)===JSON.stringify(emptyProfile()))openProfile(proposed);
    setNotice('로그 검사가 완료됐습니다. 빈 프로파일에는 확인된 값을 채웠습니다. 기존 입력이 있으면 보존하며, 아래에서 새 초안으로 열 수 있습니다.');
  },[jobs,pendingLog,profile]);
  const missing=missingProfileFields(profile), evidence=profile.provenance?.fields??{};
  const update=(key:string,value:Aircraft[keyof Aircraft])=>setProfile(p=>editProfile(p,key,value));
  async function save(){setSaving(true);setError('');try{const saved=await post<Profile>('/local-ai/profiles',profile);setProfile({...emptyProfile(),...saved.profile});onSaved(saved);setNotice(`저장됨 · 분석에 필요한 미확인 항목 ${saved.eligibility.log_missing_fields?.length??saved.eligibility.missing_fields.length}개`);}catch(e){setError(String(e));}finally{setSaving(false);}}
  function editor(key:string){
    if(key==='autopilot')return <label className="form-label" key={key}>비행제어기<select className="text-input" value={profile.autopilot} onChange={e=>update(key,e.target.value)}><option value="unknown">미확정</option><option value="px4">PX4</option><option value="ardupilot">ArduPilot</option></select></label>;
    if(key==='motor_channels')return <fieldset key={key}><legend>모터 위치별 출력 채널</legend>{positions.map(([pos,label])=><label className="form-label" key={pos}>{label}<input className="text-input" type="number" min={1} max={16} value={profile.motor_channels[pos]??''} onChange={e=>{const channels={...profile.motor_channels};if(e.target.value==='')delete channels[pos];else channels[pos]=Number(e.target.value);update(key,channels);}}/></label>)}</fieldset>;
    const entry=fields.find(([name])=>name===key);if(!entry)return null;
    const [field,label]=entry,question=suggestion?.questions.find(q=>q.field===key);
    return <label className="form-label" key={key}>{label}{question?.candidates.length?<select className="text-input" value={profile[field]??''} onChange={e=>update(field,e.target.value===''?null:Number(e.target.value))}><option value="">선택 필요</option>{question.candidates.map(v=><option key={v} value={v}>{v}</option>)}</select>:<input className="text-input" type="number" step={field==='capacity_ah'||field.startsWith('output_')?'any':1} value={profile[field]??''} placeholder="미확정" onChange={e=>update(field,e.target.value===''?null:Number(e.target.value))}/>}<small>{evidence[field]?.status==='estimated'?'전압 기반 추정값입니다. 실제 구성이 다르면 수정하세요.':question?.reason}</small></label>;
  }
  return <section className="panel"><h2>2. 기체 프로파일</h2>
    <p>로그 설정과 근거가 충분한 셀 수 추정값을 채웁니다. 셀 수는 아래에서 바로 수정할 수 있습니다. 기록용 정보는 비워도 됩니다.</p>
    <label className="form-label">저장된 프로파일<select className="text-input" value="" onChange={e=>{const p=profiles.find(p=>p.version===e.target.value);if(p){openProfile({...emptyProfile(),...p.profile});setSource('');setNotice('저장된 버전을 열었습니다. 변경 후에는 새 버전으로 저장됩니다.');}}}><option value="">선택</option>{profiles.map(p=><option key={p.version} value={p.version}>{p.profile.name} · {p.version.slice(0,8)}</option>)}</select></label>
    <label className="form-label">자동 입력에 사용할 로그<select className="text-input" value={source} onChange={e=>setSource(e.target.value)}><option value="">선택</option>{inspections.map(j=><option key={j.id} value={j.id}>{j.result?.filename} · {j.id.slice(0,8)}</option>)}</select></label>
    {suggestion?<><button className="button" disabled={disabled} onClick={()=>{openProfile(fromLog(suggestion));setNotice('로그 기반 새 초안을 열었습니다. 기존 저장 버전은 보존됩니다.');}}>이 로그로 새 프로파일 초안 열기</button><p>{suggestion.note}</p></>:null}{selected&&<button className="button" disabled={disabled} onClick={()=>reinspect(selected.id)}>로그 다시 검사 · 자동 입력 갱신</button>}
    <p role="status">{notice}</p>{error&&<p role="alert">{error}</p>}
    <h3>분석에 필요한 정보 · 미확인 {missing.length}개</h3>
    {profile.rotor_count!=null&&profile.rotor_count!==4&&<p role="alert">현재 모델은 4로터 입력만 지원합니다.</p>}
    {!missing.length&&<p>필수 입력 설정이 채워졌습니다. 유효한 연속 구간과 건강 상태는 별도로 확인합니다.</p>}
    <div className="local-form-grid">{requested.map(editor)}</div>
    <details><summary>채워진 분석 정보 확인·수정</summary><div className="local-form-grid">{['autopilot',...fields.map(([key])=>key),'motor_channels'].filter(key=>!requested.includes(key)).map(editor)}</div></details>
    {Object.keys(evidence).length>0&&<details open><summary>자동 입력 값과 출처</summary><table><thead><tr><th>항목</th><th>값</th><th>근거</th></tr></thead><tbody>{Object.entries(evidence).map(([key,e])=><tr key={key}><td>{fields.find(([k])=>k===key)?.[1]??({autopilot:'비행제어기',motor_channels:'모터 채널',firmware:'펌웨어'}[key]??key)}</td><td>{JSON.stringify(e.value)}{e.status==='estimated'?' · 추정':''}</td><td>{e.source}</td></tr>)}</tbody></table><small>원본 {profile.provenance?.source_sha256?.slice(0,12)} · 직접 수정한 값은 로그 자동 입력 표시에서 제외됩니다.</small></details>}
    <details><summary>기록용 정보 · 선택 사항</summary><p>기체 이름·모델명·펌웨어·배터리 식별명·확인 메모는 분석을 차단하지 않습니다. ID와 이름은 비우면 자동 생성합니다.</p>{([['id','관리 ID'],['name','기체 이름'],['firmware','펌웨어 기록'],['battery_id','배터리 식별명'],['mapping_evidence','설정 확인 메모'],['notes','기타 메모']] as const).map(([key,label])=><label className="form-label" key={key}>{label}<input className="text-input" value={profile[key]} onChange={e=>update(key,e.target.value)}/></label>)}</details>
    <button className="button primary" disabled={disabled||saving} onClick={save}>새 버전으로 저장</button><button className="button" disabled={saving} onClick={()=>{openProfile(emptyProfile());setSource('');setNotice('빈 프로파일 초안');}}>빈 초안</button>
  </section>;
}
