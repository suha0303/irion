import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, ArrowDownToLine, ArrowRight, Battery, Bell, ChevronDown, ChevronRight, CircleAlert, Compass, Database, FileUp, FlaskConical, FolderOpen, Gauge, Info, LayoutDashboard, Link2, Pause, Play, Plug, Radio, RotateCcw, Settings2, ShieldCheck, SkipBack, SkipForward, Wind, X } from 'lucide-react';
import { ApiError, api, post, upload } from './api';
import { PathChart, TimeChart, motorColors } from './Charts';
import { Diagnostics, EventDetail, EventList } from './Panels';
import type { Detection, Manifest, Metrics, Point, ProviderState } from './types';
import { detectionKey, numeric, statusLabel, targetLabel, timeLabel } from './types';
import { useReplay } from './useReplay';
import { Flight3D } from './Flight3D';
import { ProviderCards, ProviderDialog, SourceDialog } from './Integrations';
import { useServerHealth } from './useServerHealth';
import { compareModels } from './comparison';
import './extension.css';

type Tab = 'monitor' | 'events' | 'diagnostics';
const titles = { monitor: ['비행 모니터링', '관측과 두 모델의 판단을 같은 시점에서 확인하세요.'], events: ['이상 구간 분석', '판정 시점과 관측 데이터를 나란히 살펴보세요.'], diagnostics: ['데이터 진단', '좋은 판단은 신뢰할 수 있는 입력에서 시작됩니다.'] };

export default function App() {
  const [runs, setRuns] = useState<Manifest[]>([]);
  const [runId,setRunId]=useState(()=>{try{return localStorage.getItem('arion.selectedRunId')??'';}catch{return '';}});
  const [tab, setTab] = useState<Tab>('monitor');
  const [points, setPoints] = useState<Point[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState('');
  const [retryFile,setRetryFile]=useState<File|null>(null);
  const [imported,setImported]=useState<{name:string;accepted:number;excluded:number}|null>(null);
  const server=useServerHealth();
  const [busy, setBusy] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [batteryMetric, setBatteryMetric] = useState('battery_current_a');
  const [aiModal, setAiModal] = useState(false);
  const [providerId,setProviderId]=useState('lstm');
  const [providerFilter,setProviderFilter]=useState('all');
  const [sourceModal,setSourceModal]=useState(false);
  const [providerDefs,setProviderDefs]=useState<ProviderState[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const run = runs.find(r => r.run_id === runId);
  const isLive=run?.source_kind==='live';
  const { frame, playback, connection, command } = useReplay(run?.run_id??'',isLive);
  useEffect(()=>{if(runId){try{localStorage.setItem('arion.selectedRunId',runId);}catch{/* Browser storage can be unavailable. */}}},[runId]);
  const configure=(id='lstm')=>{setProviderId(id);setAiModal(true);};
  const cursor = frame?.t_ms ?? 0;
  const connected = connection === 'connected';
  const events = frame?.events ?? [];
  const selectedEvent = events.find(e => detectionKey(e) === selected) ?? null;
  const filteredEvents=events.filter(e=>providerFilter==='all'||(e.provider_id??'legacy')===providerFilter);

  const reload = useCallback(async (preferred?: string) => {
    const data = await api<Manifest[]>('/runs'); setRuns(data);
    setRunId(previous => preferred ?? (data.some(r => r.run_id === previous) ? previous : data[0]?.run_id ?? ''));
  }, []);
  useEffect(()=>{
    if(server.status==='offline'){setInitialLoading(false);return;}
    if(server.status!=='online')return;
    let cancelled=false;
    reload().catch(e=>{if(!cancelled)setError(e.message);}).finally(()=>{if(!cancelled)setInitialLoading(false);});
    api<ProviderState[]>('/analysis/providers').then(data=>{if(!cancelled)setProviderDefs(data);}).catch(e=>{if(!cancelled)setError(e.message);});
    return()=>{cancelled=true;};
  },[server.status,reload]);
  useEffect(() => {
    if (!runId) return;
    let stale = false; setPoints([]); setSelected(null);
    api<{points:Point[]}>(`/runs/${runId}/series`).then(series=>{if(!stale)setPoints(series.points);}).catch(e=>{if(!stale)setError(e.message);});
    return () => { stale = true; };
  }, [runId,isLive]);

  useEffect(()=>{
    if(frame?.live_series)setPoints(frame.live_series);
    if(frame?.manifest)setRuns(previous=>previous.map(r=>r.run_id===frame.run_id?{...r,...frame.manifest!}:r));
  },[frame?.live_series,frame?.manifest]);
  const sessionAction=async(action:string)=>{setBusy(true);try{await post(`/sessions/${runId}/${action}`);await reload(runId);}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
  useEffect(() => {
    if (tab !== 'diagnostics') return;
    let stale = false;
    const update = () => api<Metrics>('/metrics').then(data => { if (!stale) setMetrics(data); }).catch(() => {});
    update(); const interval = setInterval(update, 2000);
    return () => { stale = true; clearInterval(interval); };
  }, [tab]);
  useEffect(() => {
    const keypress = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !['INPUT', 'BUTTON', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName) && !aiModal && !sourceModal && runId) {
        e.preventDefault(); command((isLive?playback.following:playback.playing) ? 'pause' : 'play');
      }
      if (e.code === 'Escape') {setAiModal(false);setSourceModal(false);}
    };
    document.addEventListener('keydown', keypress); return () => document.removeEventListener('keydown', keypress);
  });

  async function importLog(file?: File) {
    if (!file) return;
    setBusy(true); setError('');setImported(null);setRetryFile(null);
    try {
      const manifest = await upload<Manifest>('/runs/import', file);
      setRuns(previous=>[manifest,...previous.filter(r=>r.run_id!==manifest.run_id)]);
      setRunId(manifest.run_id);setTab('monitor');setSelected(null);
      if(manifest.run_id===runId){command('pause');command('seek',{t_ms:0});}
      setImported({name:file.name,accepted:manifest.accepted_records,excluded:manifest.excluded_records});
    } catch (e) {
      setError((e as Error).message);
      if(e instanceof ApiError&&e.kind==='network'){setRetryFile(file);void server.check();}
    }
    finally { setBusy(false); if (fileInput.current) fileInput.current.value = ''; }
  }
  async function toggleDemo() {
    try {
      await post(`/runs/${runId}/analysis-demo`, { enabled: !frame?.fixture_mode }); setSelected(null);
    }
    catch (e) { setError((e as Error).message); }
  }
  const value = (key: string) => connected ? frame?.values[key]?.value : null;
  const chooseEvent = (event: Detection) => { command('pause'); command('seek', { t_ms: event.decided_at_ms }); setSelected(detectionKey(event)); setTab('events'); };
  const comparison=compareModels(connected?frame?.active_by_provider?.lstm:null,connected?frame?.active_by_provider?.transformer:null);
  const extraProviders=frame?.providers?.filter(p=>!['lstm','transformer'].includes(p.provider_id)).map(p=>p.provider_id)??[];
  const motorValues=[1,2,3,4].map(i=>value(`motor_${i}_rpm`));
  const motorRange=motorValues.every(v=>v!=null)?`${numeric(Math.min(...motorValues as number[]),0)}–${numeric(Math.max(...motorValues as number[]),0)}`:'—';
  const titleDate = run?.name.match(/\d{4}-\d{2}-\d{2}/)?.[0]?.replaceAll('-', '.') ?? '선택한 비행';

  return <div className="app-shell balanced-workspace">
    <input ref={fileInput} type="file" accept=".jsonl" className="visually-hidden" aria-label="JSONL 로그 파일" onChange={e => importLog(e.target.files?.[0])}/>
    <div className="app-main">
      <header className="workspace-topbar"><a className="workspace-brand" href="#" onClick={e=>{e.preventDefault();setTab('monitor');}} aria-label="ARION 비행 모니터링">ARION<small>FLIGHT INSIGHTS</small></a><nav aria-label="주요 화면">{(['monitor','events','diagnostics'] as Tab[]).map(key=><button key={key} className={tab===key?'active':''} onClick={()=>setTab(key)} aria-current={tab===key?'page':undefined}>{titles[key][0]}{key==='events'&&events.length>0&&<span className="nav-event-count">{events.length}</span>}</button>)}</nav><span className={`server-indicator ${server.status}`} role="status"><i/>{server.status==='online'?'로컬 서버 연결됨':server.status==='offline'?'서버 오프라인':'연결 확인 중'}</span></header>
      <main>
        <div className="page-heading"><div><div className="eyebrow">FLIGHT WORKSPACE</div><h1>{titles[tab][0]}</h1><p>{titles[tab][1]}</p></div><div className="heading-actions"><button className="button" onClick={()=>setSourceModal(true)} disabled={server.status!=='online'||busy}><Radio size={17}/>실시간 연결</button><button className="button primary" onClick={() => fileInput.current?.click()} disabled={busy}><FileUp size={17}/>{busy ? '자료 검사 중…' : '비행 로그 가져오기'}</button></div></div>
        {server.status==='offline'&&<section className="server-offline" role="alert"><Link2 size={19}/><div><strong>대시보드 서버에 연결할 수 없습니다</strong><p>터미널에서 대시보드 폴더의 <code>./server.sh start</code>를 실행하세요. 화면과 업로드는 서버가 실행 중일 때 동작합니다.</p><small>실행 위치: /Users/ijunseo/Capstone/dashboard</small></div><button className="button small" onClick={()=>server.check()}>연결 다시 확인</button></section>}
        {error && <div className="notice error upload-notice" role="alert"><CircleAlert size={18}/><div><strong>{retryFile?`${retryFile.name} · 가져오기 대기`:'요청을 완료하지 못했습니다'}</strong><p>{error}</p></div>{retryFile&&<button className="button small" onClick={()=>importLog(retryFile)} disabled={busy}>가져오기 다시 시도</button>}<button className="icon-button" aria-label="오류 메시지 닫기" onClick={()=>{setError('');setRetryFile(null);}}><X size={16}/></button></div>}
        {imported&&<div className="import-success" role="status"><ShieldCheck size={17}/><span><strong>가져오기 완료</strong> · {imported.accepted.toLocaleString()}건 관측{imported.excluded>0?` · 오류 ${imported.excluded}건 제외`:''}</span><button className="text-button" onClick={()=>setTab('diagnostics')}>검사 결과 보기<ArrowRight size={13}/></button><button className="icon-button" aria-label="가져오기 완료 안내 닫기" onClick={()=>setImported(null)}><X size={14}/></button></div>}
        {!run ? <div className="panel welcome"><div className="welcome-icon"><Wind size={40}/></div><span className="eyebrow">YOUR NEXT FLIGHT INSIGHT</span><h2>{server.status==='offline'?'서버 실행 후 비행 기록을 불러옵니다':initialLoading?'비행 기록을 불러오고 있습니다':'첫 비행 기록을 열어보세요'}</h2><p>JSONL 로그 하나로 배터리, 모터, 비행 상태를<br/>같은 시간축에서 확인할 수 있습니다.</p><button className="button primary" onClick={() => fileInput.current?.click()} disabled={busy}><FileUp size={17}/>JSONL 파일 선택</button><small>최대 64 MiB · 원본 보존 · 로컬 처리</small></div> : <>
          <section className="flight-bar"><div className="flight-icon"><Wind size={23}/></div><div className="flight-selector"><label htmlFor="flight-select">선택한 비행</label><div><select id="flight-select" value={runId} disabled={busy} onChange={e=>{setRunId(e.target.value);setImported(null);}} aria-label="비행 선택">{runs.map(r => <option key={r.run_id} value={r.run_id}>{r.name.includes('2026-10-05') ? '이동 비행 · 2026.10.05' : r.name}</option>)}</select><ChevronDown size={15}/></div></div><div className="flight-meta"><span>{run.source}</span><span>{(run.duration_ms / 1000).toFixed(2)}초</span><span>{run.accepted_records.toLocaleString()}개 관측</span></div><span className="pill blue"><span className="status-dot"/>{isLive?'LIVE STREAM':'LOG REPLAY'}</span></section>
          {isLive && <div className="session-toolbar"><span><Radio size={16}/>{statusLabel[frame?.session?.state??run.session?.state??'waiting']??'수집 중'} · 수집 {frame?.session?.last_sequence??0}건</span><code>{runId}</code><button className="button small" disabled={busy} onClick={()=>sessionAction('connect')}>입력 재연결</button><button className="button small" disabled={busy} onClick={()=>sessionAction('finish')}>수집 종료·저장</button>{frame?.session?.message&&<p className="inline-error">{frame.session.message}</p>}</div>}
          {connection === 'reconnecting' && server.status==='online' && <div className="notice error" role="alert"><Link2 size={18}/><div><strong>서버 연결이 끊겼습니다</strong><p>자동 재연결 중입니다. 마지막 시각을 보존하며 재생은 일시정지됩니다.</p></div></div>}
          {frame?.fixture_mode && <div className="notice warning fixture-banner"><FlaskConical size={18}/><div><strong>시험용 AI 응답 모드</strong><span>표시되는 판정은 화면 검증용이며, 실제 고장이나 모델 탐지 결과가 아닙니다.</span></div><button className="text-button" onClick={toggleDemo}>시험 종료<X size={13}/></button></div>}
          {tab === 'monitor' && <>
            <div className="monitor-hero"><Flight3D frame={frame} points={points} connected={connected}/><aside className="model-comparison" aria-label="LSTM과 Transformer 비교"><div className="comparison-heading"><h2>모델 비교</h2><span>관측과 별도 판정</span></div><ProviderCards frame={frame} connected={connected} configure={configure} providerIds={['lstm','transformer']} choose={chooseEvent} offline={server.status==='offline'}/><div className={`comparison-summary ${comparison.state}`}><Info size={15}/><span>{comparison.label}</span></div></aside></div>
            <section className="observation-strip" aria-label="현재 비행 관측"><div className="observation-item"><span><Battery size={14}/>배터리 SOC</span><strong>{numeric(value('battery_soc_pct'),0)}<small>%</small></strong><em>충전 잔량 · 건강도와 구분</em></div><div className="observation-item"><span>팩 전압 / 전류</span><strong>{numeric(value('battery_voltage_v'),2)}<small>V</small><i>/</i>{numeric(value('battery_current_a'),2)}<small>A</small></strong><em>수신된 관측값</em></div><div className="observation-item"><span><Compass size={14}/>상대 높이 / 속력</span><strong>{numeric(value('height_m'),2)}<small>m</small><i>/</i>{numeric(value('speed_m_s'),2)}<small>m/s</small></strong><em>{connected?frame?.flight_state??'미확인':'연결 끊김'} · {connected&&frame?.armed!=null?(frame.armed?'ARMED':'DISARMED'):'—'} · {connected?frame?.mode??'—':'—'}</em></div><div className="observation-item"><span><Wind size={14}/>모터 1–4 · RPM</span><strong>{motorRange}</strong><em>슬롯 번호 · 물리 위치 미지정</em></div></section>
            {extraProviders.length>0&&<details className="extra-providers"><summary>추가 분석 제공자 {extraProviders.length}개</summary><ProviderCards frame={frame} connected={connected} configure={configure} providerIds={extraProviders} choose={chooseEvent} offline={server.status==='offline'}/></details>}
            <div className="charts-grid"><section className="panel"><div className="panel-heading"><div><span className="eyebrow">POWER TREND</span><h2>배터리 추이</h2></div><div className="segmented" aria-label="배터리 지표">{[{ key: 'battery_current_a', label: '전류' }, { key: 'battery_voltage_v', label: '전압' }, { key: 'battery_soc_pct', label: 'SOC' }].map(m => <button key={m.key} className={batteryMetric === m.key ? 'active' : ''} onClick={() => setBatteryMetric(m.key)}>{m.label}</button>)}</div></div><TimeChart points={points} cursor={cursor} signals={[{ key: batteryMetric, name: batteryMetric === 'battery_current_a' ? '팩 전류' : batteryMetric === 'battery_voltage_v' ? '팩 전압' : 'SOC' }]} colors={['#0aab97']} unit={batteryMetric === 'battery_current_a' ? 'A' : batteryMetric === 'battery_voltage_v' ? 'V' : '%'}/></section>
              <section className="panel"><div className="panel-heading"><div><span className="eyebrow">MOTOR TREND</span><h2>모터 RPM 추이</h2></div><div className="chart-legend">{[1, 2, 3, 4].map(i => <span key={i}><i style={{ background: motorColors[i - 1] }}/>{i}</span>)}</div></div><TimeChart points={points} cursor={cursor} signals={[1, 2, 3, 4].map(i => ({ key: `motor_${i}_rpm`, name: `모터 ${i}` }))} colors={motorColors} unit="rpm"/></section></div>
            <section className="panel motor-section"><div className="panel-heading"><div><span className="eyebrow">PROPULSION</span><h2>모터 관측 <span className="heading-note">4 CHANNELS</span></h2></div><span className="muted">슬롯 번호 기준 · 물리 위치 미지정</span></div><div className="motor-grid">{[1, 2, 3, 4].map((i) => <div className="motor-card" key={i}><div className="motor-card-top"><span className="motor-number" style={{ color: motorColors[i - 1], background: `${motorColors[i - 1]}12` }}><Wind size={17}/></span><strong>모터 {i}</strong><span className={`status-dot ${connected && frame?.values[`motor_${i}_rpm`]?.status === 'valid' ? 'teal' : 'gray'}`}/></div><div className="motor-rpm">{numeric(value(`motor_${i}_rpm`), 0)}<small>rpm</small></div><div className="motor-current"><span>전류</span><b>{numeric(value(`motor_${i}_current_a`), 2)} <small>A</small></b></div><div className="motor-progress"><i style={{ width: `${Math.min(100, Math.abs(value(`motor_${i}_rpm`) ?? 0) / 100)}%`, background: motorColors[i - 1] }}/></div></div>)}</div></section>
            <div className="bottom-grid"><section className="panel"><div className="panel-heading"><div><span className="eyebrow">LOCAL TRAJECTORY</span><h2>비행 궤적</h2></div><span className="pill neutral">NED</span></div><PathChart points={points} cursor={cursor}/></section><section className="panel event-summary"><div className="panel-heading"><div><span className="eyebrow">DETECTION TIMELINE</span><h2>판정 이력 <span className="heading-note">{events.length}건</span></h2></div><button className="text-button" onClick={() => setTab('events')}>전체 보기<ArrowUpRightIcon/></button></div><EventList events={events} selected={selected} choose={chooseEvent}/><div className="event-summary-foot"><Info size={14}/>{frame?.fixture_mode ? '시험용 응답의 판정 시각을 따라 표시합니다.' : '이상 점수는 고장 확률이나 건강도 %가 아닙니다.'}</div></section></div>
          </>}
          {tab === 'events' && <><div className="event-tools"><div><Info size={16}/><span>재생 시각까지 도착한 판정만 표시됩니다. 이벤트를 누르면 판정 시점으로 이동합니다.</span></div><button className={`button ${frame?.fixture_mode ? 'amber-button' : ''}`} disabled={isLive} onClick={toggleDemo}><FlaskConical size={16}/>{frame?.fixture_mode ? '시험 이벤트 끄기' : '시험 이벤트 켜기'}</button></div><ProviderCards frame={frame} connected={connected} configure={configure} choose={chooseEvent} offline={server.status==='offline'}/><label className="provider-filter">제공자 필터 <select className="text-input" value={providerFilter} onChange={e=>setProviderFilter(e.target.value)}><option value="all">전체 제공자</option>{providerDefs.map(p=><option value={p.provider_id} key={p.provider_id}>{p.name}</option>)}</select></label><div className="events-layout"><section className="panel"><div className="panel-heading"><h2>판정 이력</h2><span className="pill neutral">{events.length}건</span></div><EventList events={filteredEvents} selected={selected} choose={chooseEvent}/></section><EventDetail event={selectedEvent} points={points} cursor={cursor}/></div></>}
          {tab === 'diagnostics' && <Diagnostics run={run} frame={connected ? frame : null} metrics={metrics} connectAI={()=>configure()}/>}
          <div className="page-footer"><span><ShieldCheck size={13}/>원본 보존 · 읽기 전용 분석</span><span>{titleDate} <i>•</i> Source {run.source_system}/{run.source_component} <i>•</i> Epoch {frame?.epoch ?? 0}</span></div>
        </>}
      </main>
      {run && <footer className="playback-dock"><div className="playback-top"><div className="playback-title"><span className={`status-dot ${playback.playing ? 'teal' : 'gray'}`}/>{isLive?(playback.following?'실시간 관측':'화면 일시정지 · 수집 중'):(playback.playing?'재생 중':'일시정지')}<span className="keyboard-hint">SPACE</span></div><div className="playback-time"><strong>{timeLabel(cursor, true)}</strong><span>/ {timeLabel(Math.max(run.duration_ms,cursor), true)}</span></div><div className="playback-quality"><button onClick={() => setTab('diagnostics')} className="text-button"><CircleAlert size={14}/>제외 {run.excluded_records}건</button><span>최대 10Hz 화면 갱신</span></div></div><div className="playback-controls"><button className="icon-button" aria-label="처음으로" onClick={() => { command('pause'); command('seek', { t_ms: 0 }); }} disabled={!connected}><RotateCcw size={17}/></button><button className="icon-button" aria-label="5초 뒤로" onClick={() => command('seek', { t_ms: cursor - 5000 })} disabled={!connected}><SkipBack size={17}/></button><button className="play-button" aria-label={(isLive?playback.following:playback.playing)?'일시정지':'재생'} onClick={() => command((isLive?playback.following:playback.playing)?'pause':'play')} disabled={!connected}>{(isLive?playback.following:playback.playing) ? <Pause size={18} fill="currentColor"/> : <Play size={18} fill="currentColor"/>}</button><button className="icon-button" aria-label="5초 앞으로" onClick={() => command('seek', { t_ms: cursor + 5000 })} disabled={!connected}><SkipForward size={17}/></button><div className="timeline-track"><input type="range" aria-label="재생 시점" min={isLive?frame?.session?.buffer_start_ms??0:0} max={Math.max(run.duration_ms,cursor)} step="100" value={cursor} onChange={e => command('seek', { t_ms: Number(e.target.value) })} disabled={!connected} style={{ '--progress': `${cursor / Math.max(1, run.duration_ms) * 100}%` } as React.CSSProperties}/><div className="timeline-markers">{events.map(e => <button key={detectionKey(e)} title={`${timeLabel(e.decided_at_ms, true)} ${e.title}`} aria-label={e.title} style={{ left: `${e.decided_at_ms / run.duration_ms * 100}%` }} onClick={() => chooseEvent(e)}/>)}</div></div>{isLive?<button className="button small live-return" onClick={()=>command('go_live')}>실시간으로 복귀</button>:<label className="speed-select"><select aria-label="재생 속도" value={playback.speed} onChange={e => command('speed', { speed: Number(e.target.value) })}>{[.25, .5, 1, 2, 4].map(s => <option value={s} key={s}>{s}×</option>)}</select><ChevronDown size={12}/></label>}</div></footer>}
    </div>
    {aiModal && run && <ProviderDialog runId={runId} initial={providerId} providers={providerDefs} close={()=>setAiModal(false)}/>}
    {sourceModal && <SourceDialog close={()=>setSourceModal(false)} created={m=>{setRuns(previous=>[m,...previous.filter(r=>r.run_id!==m.run_id)]);setRunId(m.run_id);setTab('monitor');setImported(null);}}/>}

  </div>;
}

function ArrowUpRightIcon() { return <ArrowRight size={14}/>; }
