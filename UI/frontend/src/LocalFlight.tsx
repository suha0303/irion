import LocalRunResult from './LocalRunResult';
import { useEffect, useRef, useState } from 'react';
import { api, post, upload } from './api';
import './local-flight.css';
import LocalPreparation, { AnalysisResult, TimingResult, FlightEventResult } from './LocalPreparation';
import type { Profile, Job } from './local-flight-types';
import LocalModels from './LocalModels';
import LocalCalibration from './LocalCalibration';
import LocalProfile from './LocalProfile';

type Model = { id: string; task: 'forecast' | 'run_health'; label: string; version: string };
const labels: Record<string, string> = { queued: '대기', running: '실행 중', complete: '완료', failed: '실패', cancelled: '중지', interrupted: '이전 실행 중단' };

export default function LocalFlight({ close }: { close: () => void }) {
  const [models, setModels] = useState<Model[]>([]);
  const [mode, setMode] = useState<'forecast' | 'run_health'>('forecast');
  const [selected, setSelected] = useState<string[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profileLog, setProfileLog] = useState('');
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [available, setAvailable] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const running = jobs.some(j => j.status === 'running' || j.status === 'queued');

  useEffect(() => {
    let active = true;
    Promise.all([api<{available: boolean; models: Model[]; reason?: string}>('/local-ai/catalog'),
      api<{profiles: Profile[]}>('/local-ai/profiles')]).then(([c, p]) => {
      if (!active) return;
      setModels(c.models); setAvailable(c.available); setProfiles(p.profiles);
      if (c.reason) setError(c.reason);
    }).catch(e => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const ids = models.filter(m => m.task === mode).map(m => m.id);
    setSelected(ids.length ? [ids[0], ids.find(id => id.includes('transformer')) ?? ids[0]] : []);
  }, [mode, models]);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const result = await api<{jobs: Job[]}>('/local-ai/jobs');
        if (active) setJobs(result.jobs);
      } catch (e) { if (active) setError(String(e)); }
      if (active) timer = setTimeout(refresh, 1500);
    };
    void refresh();
    return () => { active = false; clearTimeout(timer); };
  }, []);

  async function perform(work: () => Promise<void>) {
    setBusy(true); setError('');
    try { await work(); } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }

  return <div className="local-flight-backdrop"><section className="local-flight" role="dialog" aria-modal="true" aria-labelledby="local-flight-title">
    <header><div><span className="eyebrow">ON DEVICE · CPU</span><h1 id="local-flight-title">실기체 로그와 로컬 AI</h1><p>기체별 입력을 확인하고, 예측 8종과 비행 종료 후 분석 8종을 선택합니다.</p></div><button className="button" onClick={close}>닫기</button></header>
    {error && <div className="notice error" role="alert">{error}</div>}
    <div className="local-flight-grid">
      <section className="panel"><h2>1. 분석 모델 선택</h2><label className="form-label">분석 모드<select className="text-input" value={mode} onChange={e => setMode(e.target.value as 'forecast' | 'run_health')}><option value="forecast">비행 중 예측 · 8종</option><option value="run_health">비행 종료 후 분석 · 8종</option></select></label>
        {['A', 'B'].map((slot, index) => <label className="form-label" key={slot}>비교 모델 {slot}<select className="text-input" value={selected[index] ?? ''} onChange={e => setSelected(s => s.map((v, i) => i === index ? e.target.value : v))}>{models.filter(m => m.task === mode).map(m => <option key={m.id} value={m.id}>{m.label}</option>)}</select></label>)}
        <p>최신 2026-11-08 묶음 · {models.length}종 · CPU 실행<br/>검증 범위는 X500 시뮬레이션입니다. 실기체 입력 연결·적응 성능은 별도 검증합니다.</p>
        <button className="button" disabled={!available || busy || running} onClick={() => perform(async () => { const job = await post<Job>('/local-ai/benchmark'); setJobs(j => [job, ...j]); })}>16종 CPU 실행 점검</button>
        <small>합성 입력 기능·속도 점검입니다. 비행의 정상 여부를 판정하지 않습니다.</small>
      </section>
      <LocalProfile profiles={profiles} jobs={jobs} pendingLog={profileLog} disabled={!available||busy} onSaved={saved=>setProfiles(previous=>[saved,...previous.filter(p=>p.version!==saved.version)])} reinspect={id=>perform(async()=>{const job=await post<Job>(`/local-ai/logs/${id}/reinspect`);setJobs(j=>[job,...j]);setProfileLog(job.id);})}/>
      <section className="panel"><h2>3. 실기체 로그 검사</h2><p>PX4 ULog 또는 ArduPilot BIN의 메시지·필드·시간축·파서 경고를 확인합니다. 검사가 끝나면 로그에 명시된 프로파일 값을 자동으로 채웁니다. 정상 여부는 추정하지 않습니다.</p>
        <input hidden ref={input} type="file" accept=".bin,.BIN,.ulg" onChange={e => { const file = e.target.files?.[0]; if (file) void perform(async () => { const job = await upload<Job>('/local-ai/logs',file); setJobs(j => [job, ...j]); setProfileLog(job.id); }); e.target.value = ''; }}/>
        <button className="button primary" disabled={!available || busy || running} onClick={() => input.current?.click()}>ULog / BIN 가져오기</button><small>최대 256 MiB · 원본 보관 · PC 내부 처리</small>
        <div className="notice warning">적응 학습에는 정상임이 확인된 비행과 검증된 신호 매핑이 필요합니다. 현재 정보가 없는 로그는 검사·열람 단계에 머뭅니다.</div>
      </section>
      <LocalPreparation jobs={jobs} profiles={profiles} selectedModels={selected} disabled={!available || busy || running} submit={(path,body) => perform(async () => {const job=await post<Job>(path,body);setJobs(j => [job,...j]);})}/>
      <LocalCalibration jobs={jobs} model={selected[0]??''} disabled={!available||busy||running} submit={(path,body)=>perform(async()=>{const job=await post<Job>(path,body);setJobs(j=>[job,...j]);})}/>
      <LocalModels jobs={jobs} disabled={!available || busy || running} submit={(path,body) => perform(async () => {const job=await post<Job>(path,body);setJobs(j => [job,...j]);})} imported={job=>setJobs(j=>[job,...j])}/>
      <section className="panel"><h2>작업 기록</h2>{jobs.length === 0 && <p>아직 실행한 작업이 없습니다.</p>}{jobs.map(job => <article className="local-job" key={job.id}><strong>{{inspect:'로그 검사',benchmark:'CPU 실행 점검',timeline:'보존 구간 검사',prepare:'구간 저장',analyze:'예측·분석',adapt:'적응 학습',compare:'독립 비교',calibration_fit:'임계값 보정',calibration_select:'보정 적용',admit_live:'수신 비행 구간 등록'}[job.action] ?? job.action} · {job.id.slice(0,8)} · {labels[job.status] ?? job.status}</strong>
        {(job.status === 'running' || job.status === 'queued') && <button className="button small" onClick={() => perform(async () => { await post(`/local-ai/jobs/${job.id}/cancel`); })}>중지</button>}
        {job.error && <p role="alert">{job.error}</p>}
        {job.result && job.action === 'inspect' && <><p>{job.result.filename} · {job.result.autopilot}<br/>건강 상태 미확정 · 정상 학습에 사용하지 않음<br/>후보 필드 누락: {Object.keys(job.result.missing_candidate_fields ?? {}).length}개 메시지</p>{job.result.parser_warnings && <p className="local-warning">파서 경고: {job.result.parser_warnings}</p>}<details><summary>입력 진단 상세</summary><pre>{JSON.stringify({missing: job.result.missing_candidate_fields, parameters: job.result.parameters, messages: job.result.messages}, null, 2)}</pre></details></>}
        {job.result && job.action === 'benchmark' && <details open><summary>16종 합성 입력 측정</summary>{job.result.models?.map(m => <p key={m.id}>{m.id}: p95 {m.p95_ms.toFixed(2)} ms</p>)}</details>}
      {job.result?.flight_events&&<FlightEventResult events={job.result.flight_events}/>}
      {job.result?.timing && <TimingResult timing={job.result.timing}/>}
        {job.result?.analyses&&<LocalRunResult summary={job.result.run_summary} analyses={job.result.analyses}/>}
        {job.result?.original && <><h3>원본</h3><AnalysisResult result={job.result.original}/></>}
        {job.result?.adapted && <><h3>적응 모델</h3><AnalysisResult result={job.result.adapted}/></>}
        {job.progress && <p>{job.progress.epoch} epoch · loss {job.progress.train_loss.toFixed(5)}</p>}
        {job.action === 'adapt' && ['cancelled','interrupted','failed'].includes(job.status) && <button className="button" disabled={busy || running} onClick={() => perform(async () => {const row=await post<Job>(`/local-ai/jobs/${job.id}/resume`);setJobs(j => [row,...j]);})}>마지막 완료 epoch에서 재개</button>}
      </article>)}</section>
    </div>
  </section></div>;
}
