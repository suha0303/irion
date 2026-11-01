import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, ArrowDownToLine, ArrowRight, Battery, Bell, ChevronDown, ChevronRight, CircleAlert, Compass, Database, FileUp, FlaskConical, FolderOpen, Gauge, Info, LayoutDashboard, Link2, Pause, Play, Plug, Radio, RotateCcw, Settings2, ShieldCheck, SkipBack, SkipForward, Wind, X } from 'lucide-react';
import { api, post, upload } from './api';
import { PathChart, TimeChart, motorColors } from './Charts';
import { Diagnostics, EventDetail, EventList } from './Panels';
import type { Detection, Manifest, Metrics, Point, ProviderState } from './types';
import { detectionKey, numeric, statusLabel, targetLabel, timeLabel } from './types';
import { useReplay } from './useReplay';
import { Flight3D } from './Flight3D';
import { ProviderCards, ProviderDialog, SourceDialog } from './Integrations';
import './extension.css';

type Tab = 'monitor' | 'events' | 'diagnostics';
const titles = { monitor: ['비행 모니터링', '비행의 흐름을 읽고, 이상 징후의 근거를 확인하세요.'], events: ['이상 구간 분석', '판정 시점과 관측 데이터를 나란히 살펴보세요.'], diagnostics: ['데이터 진단', '좋은 판단은 신뢰할 수 있는 입력에서 시작됩니다.'] };

export default function App() {
  const [runs, setRuns] = useState<Manifest[]>([]);
  const [runId, setRunId] = useState('');
  const [tab, setTab] = useState<Tab>('monitor');
  const [points, setPoints] = useState<Point[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState('');
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
  const { frame, playback, connection, command } = useReplay(runId,isLive);
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
  useEffect(() => { reload().catch(e => setError(e.message)).finally(() => setInitialLoading(false)); }, [reload]);
  useEffect(() => {
    if (!runId) return;
    let stale = false; setPoints([]); setSelected(null);
    api<{points:Point[]}>(`/runs/${runId}/series`).then(series=>{if(!stale)setPoints(series.points);}).catch(e=>{if(!stale)setError(e.message);});
    return () => { stale = true; };
  }, [runId]);
  useEffect(()=>{api<ProviderState[]>('/analysis/providers').then(setProviderDefs).catch(e=>setError(e.message));},[]);
  useEffect(()=>{
    if(frame?.live_series)setPoints(frame.live_series);
    if(frame?.manifest)setRuns(previous=>previous.map(r=>r.run_id===frame.run_id?{...r,...frame.manifest!}:r));
  },[frame?.live_series,frame?.manifest]);
  const sessionAction=async(action:string)=>{try{await post(`/sessions/${runId}/${action}`);await reload(runId);}catch(e){setError((e as Error).message);}};
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
      if (e.code === 'Escape') setAiModal(false);
    };
    document.addEventListener('keydown', keypress); return () => document.removeEventListener('keydown', keypress);
  });

  async function importLog(file?: File) {
    if (!file) return;
    setBusy(true); setError('');
    try { const manifest = await upload<Manifest>('/runs/import', file); await reload(manifest.run_id); setTab('monitor'); }
    catch (e) { setError((e as Error).message); }
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
  const titleDate = run?.name.match(/\d{4}-\d{2}-\d{2}/)?.[0]?.replaceAll('-', '.') ?? '선택한 비행';

  return <div className="app-shell">
    <input ref={fileInput} type="file" accept=".jsonl" className="visually-hidden" aria-label="JSONL 로그 파일" onChange={e => importLog(e.target.files?.[0])}/>
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); setTab('monitor'); }}><span className="brand-mark">A<span/></span><span>ARION<small>FLIGHT INTELLIGENCE</small></span></a>
      <div className="workspace-tag"><span className="status-dot teal"/>캡스톤 연구 워크스페이스</div>
      <div className="nav-label">ANALYSIS</div>
      <nav aria-label="주요 화면">{([{ key: 'monitor', name: '비행 모니터링', icon: LayoutDashboard }, { key: 'events', name: '이상 구간 분석', icon: Activity }, { key: 'diagnostics', name: '데이터 진단', icon: Database }] as const).map(item => <button key={item.key} className={`nav-item ${tab === item.key ? 'active' : ''}`} onClick={() => setTab(item.key)} aria-current={tab === item.key ? 'page' : undefined}><item.icon size={19}/>{item.name}{item.key === 'events' && events.length > 0 && <span className="nav-count">{events.length}</span>}</button>)}</nav>
      <div className="nav-label source-label">DATA SOURCE</div>
      <div className="source-card"><div><FolderOpen size={18}/><span>{isLive?'실시간 수집':'로그 재생'}</span><span className="tiny-pill">LOCAL</span></div><p>관측·분석·입력 품질을 확인하세요.</p><div className="source-divider"/><small><Radio size={13}/>UDP · 시리얼 · 전처리 입력</small></div>
      <div className="sidebar-bottom"><div className="read-only"><ShieldCheck size={16}/><span>조회 전용 워크스페이스</span></div><p>상태 관찰과 판단 근거 확인<br/>ARION × CAPSTONE 2026</p><div className="version">RESEARCH PREVIEW <span>v0.2</span></div></div>
    </aside>
    <div className="app-main">
      <header className="topbar"><div className="breadcrumb">워크스페이스<ChevronRight size={13}/><strong>{titles[tab][0]}</strong></div><div className="topbar-right"><span className="local-indicator"><span className={`status-dot ${connected ? 'teal' : 'gray'}`}/>{connected ? '로컬 서버 연결됨' : '로컬 모드'}</span><span className="topbar-divider"/><span className="avatar">AR</span></div></header>
      <nav className="compact-nav" aria-label="좁은 화면 메뉴">{(['monitor', 'events', 'diagnostics'] as Tab[]).map(key => <button key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>{titles[key][0]}</button>)}</nav>
      <main>
        <div className="page-heading"><div><div className="eyebrow">ARION / FLIGHT INSIGHTS</div><h1>{titles[tab][0]}</h1><p>{titles[tab][1]}</p></div><div className="heading-actions"><button className="button" onClick={()=>setSourceModal(true)}><Radio size={17}/>실시간 연결</button><button className="button primary" onClick={() => fileInput.current?.click()} disabled={busy}><FileUp size={17}/>{busy ? '자료 검사 중…' : '비행 로그 가져오기'}</button></div></div>
        {error && <div className="notice error" role="alert"><CircleAlert size={18}/><span>{error}</span><button className="icon-button" aria-label="오류 메시지 닫기" onClick={() => setError('')}><X size={16}/></button></div>}
        {!run ? <div className="panel welcome"><div className="welcome-icon"><Wind size={40}/></div><span className="eyebrow">YOUR NEXT FLIGHT INSIGHT</span><h2>{initialLoading ? '비행 기록을 불러오고 있습니다' : '첫 비행 기록을 열어보세요'}</h2><p>JSONL 로그 하나로 배터리, 모터, 비행 상태를<br/>같은 시간축에서 확인할 수 있습니다.</p><button className="button primary" onClick={() => fileInput.current?.click()} disabled={busy}><FileUp size={17}/>JSONL 파일 선택</button><small>최대 64 MiB · 원본 보존 · 로컬 처리</small></div> : <>
          <section className="flight-bar"><div className="flight-icon"><Wind size={23}/></div><div className="flight-selector"><label htmlFor="flight-select">선택한 비행</label><div><select id="flight-select" value={runId} onChange={e => setRunId(e.target.value)} aria-label="비행 선택">{runs.map(r => <option key={r.run_id} value={r.run_id}>{r.name.includes('2026-10-05') ? '이동 비행 · 2026.10.05' : r.name}</option>)}</select><ChevronDown size={15}/></div></div><div className="flight-meta"><span>{run.source}</span><span>{(run.duration_ms / 1000).toFixed(2)}초</span><span>{run.accepted_records.toLocaleString()}개 관측</span></div><span className="pill blue"><span className="status-dot"/>{isLive?'LIVE STREAM':'LOG REPLAY'}</span></section>
          {isLive && <div className="session-toolbar"><span><Radio size={16}/>{statusLabel[frame?.session?.state??run.session?.state??'waiting']??'수집 중'} · 수집 {frame?.session?.last_sequence??0}건</span><code>{runId}</code><button className="button small" onClick={()=>sessionAction('connect')}>입력 재연결</button><button className="button small" onClick={()=>sessionAction('finish')}>수집 종료·저장</button>{frame?.session?.message&&<p className="inline-error">{frame.session.message}</p>}</div>}
          {connection === 'reconnecting' && <div className="notice error" role="alert"><Link2 size={18}/><div><strong>서버 연결이 끊겼습니다</strong><p>자동 재연결 중입니다. 마지막 시각을 보존하며 재생은 일시정지됩니다.</p></div></div>}
          {frame?.fixture_mode && <div className="notice warning fixture-banner"><FlaskConical size={18}/><div><strong>시험용 AI 응답 모드</strong><span>표시되는 판정은 화면 검증용이며, 실제 고장이나 모델 탐지 결과가 아닙니다.</span></div><button className="text-button" onClick={toggleDemo}>시험 종료<X size={13}/></button></div>}
          {tab === 'monitor' && <>
            <div className="overview-grid two-overviews"><section className="panel battery-overview"><div className="card-label"><Battery size={18}/><span>배터리 상태</span><span className="pill neutral">SOC</span></div><div className="battery-main"><strong>{numeric(value('battery_soc_pct'), 0)}<span>%</span></strong><div className="battery-glyph"><div style={{ height: `${value('battery_soc_pct') ?? 0}%` }}/><span/></div></div><div className="battery-readings"><div><small>팩 전압</small><b>{numeric(value('battery_voltage_v'), 2)} <em>V</em></b></div><div><small>팩 전류</small><b>{numeric(value('battery_current_a'), 2)} <em>A</em></b></div></div><div className="card-foot">충전 잔량 · 배터리 건강도와 구분</div></section>
              <section className="panel flight-overview"><div className="card-label"><Compass size={18}/><span>비행 상태</span><span className="pill neutral">{connected ? frame?.flight_state ?? '미확인' : '연결 끊김'}</span></div><div className="flight-numbers"><div><small>상대 높이</small><strong>{numeric(value('height_m'), 2)}<span>m</span></strong></div><div><small>속력</small><strong>{numeric(value('speed_m_s'), 2)}<span>m/s</span></strong></div></div><div className="flight-flags"><span><i className={`status-dot ${frame?.armed && connected ? 'teal' : 'gray'}`}/>{connected && frame?.armed != null ? frame.armed ? 'ARMED' : 'DISARMED' : '미확인'}</span><b>{connected ? frame?.mode ?? '—' : '—'}</b></div><div className="card-foot">epoch 첫 위치 기준 · 지면고도와 다를 수 있음</div></section>
            </div><Flight3D frame={frame} points={points} connected={connected}/><ProviderCards frame={frame} connected={connected} configure={configure}/>
            <section className="panel motor-section"><div className="panel-heading"><div><span className="eyebrow">PROPULSION</span><h2>모터 관측 <span className="heading-note">4 CHANNELS</span></h2></div><span className="muted">슬롯 번호 기준 · 물리 위치 미지정</span></div><div className="motor-grid">{[1, 2, 3, 4].map((i) => <div className="motor-card" key={i}><div className="motor-card-top"><span className="motor-number" style={{ color: motorColors[i - 1], background: `${motorColors[i - 1]}12` }}><Wind size={17}/></span><strong>모터 {i}</strong><span className={`status-dot ${connected && frame?.values[`motor_${i}_rpm`]?.status === 'valid' ? 'teal' : 'gray'}`}/></div><div className="motor-rpm">{numeric(value(`motor_${i}_rpm`), 0)}<small>rpm</small></div><div className="motor-current"><span>전류</span><b>{numeric(value(`motor_${i}_current_a`), 2)} <small>A</small></b></div><div className="motor-progress"><i style={{ width: `${Math.min(100, Math.abs(value(`motor_${i}_rpm`) ?? 0) / 100)}%`, background: motorColors[i - 1] }}/></div></div>)}</div></section>
            <div className="charts-grid"><section className="panel"><div className="panel-heading"><div><span className="eyebrow">POWER TREND</span><h2>배터리 추이</h2></div><div className="segmented" aria-label="배터리 지표">{[{ key: 'battery_current_a', label: '전류' }, { key: 'battery_voltage_v', label: '전압' }, { key: 'battery_soc_pct', label: 'SOC' }].map(m => <button key={m.key} className={batteryMetric === m.key ? 'active' : ''} onClick={() => setBatteryMetric(m.key)}>{m.label}</button>)}</div></div><TimeChart points={points} cursor={cursor} signals={[{ key: batteryMetric, name: batteryMetric === 'battery_current_a' ? '팩 전류' : batteryMetric === 'battery_voltage_v' ? '팩 전압' : 'SOC' }]} colors={['#0aab97']} unit={batteryMetric === 'battery_current_a' ? 'A' : batteryMetric === 'battery_voltage_v' ? 'V' : '%'}/></section>
              <section className="panel"><div className="panel-heading"><div><span className="eyebrow">MOTOR TREND</span><h2>모터 RPM 추이</h2></div><div className="chart-legend">{[1, 2, 3, 4].map(i => <span key={i}><i style={{ background: motorColors[i - 1] }}/>{i}</span>)}</div></div><TimeChart points={points} cursor={cursor} signals={[1, 2, 3, 4].map(i => ({ key: `motor_${i}_rpm`, name: `모터 ${i}` }))} colors={motorColors} unit="rpm"/></section></div>
            <div className="bottom-grid"><section className="panel"><div className="panel-heading"><div><span className="eyebrow">LOCAL TRAJECTORY</span><h2>비행 궤적</h2></div><span className="pill neutral">NED</span></div><PathChart points={points} cursor={cursor}/></section><section className="panel event-summary"><div className="panel-heading"><div><span className="eyebrow">DETECTION TIMELINE</span><h2>판정 이력 <span className="heading-note">{events.length}건</span></h2></div><button className="text-button" onClick={() => setTab('events')}>전체 보기<ArrowUpRightIcon/></button></div><EventList events={events} selected={selected} choose={chooseEvent}/><div className="event-summary-foot"><Info size={14}/>{frame?.fixture_mode ? '시험용 응답의 판정 시각을 따라 표시합니다.' : '이상 점수는 고장 확률이나 건강도 %가 아닙니다.'}</div></section></div>
          </>}
          {tab === 'events' && <><div className="event-tools"><div><Info size={16}/><span>재생 시각까지 도착한 판정만 표시됩니다. 이벤트를 누르면 판정 시점으로 이동합니다.</span></div><button className={`button ${frame?.fixture_mode ? 'amber-button' : ''}`} disabled={isLive} onClick={toggleDemo}><FlaskConical size={16}/>{frame?.fixture_mode ? '시험 이벤트 끄기' : '시험 이벤트 켜기'}</button></div><ProviderCards frame={frame} connected={connected} configure={configure}/><label className="provider-filter">제공자 필터 <select className="text-input" value={providerFilter} onChange={e=>setProviderFilter(e.target.value)}><option value="all">전체 제공자</option>{providerDefs.map(p=><option value={p.provider_id} key={p.provider_id}>{p.name}</option>)}</select></label><div className="events-layout"><section className="panel"><div className="panel-heading"><h2>판정 이력</h2><span className="pill neutral">{events.length}건</span></div><EventList events={filteredEvents} selected={selected} choose={chooseEvent}/></section><EventDetail event={selectedEvent} points={points} cursor={cursor}/></div></>}
          {tab === 'diagnostics' && <Diagnostics run={run} frame={connected ? frame : null} metrics={metrics} connectAI={()=>configure()}/>}
          <div className="page-footer"><span><ShieldCheck size={13}/>원본 보존 · 읽기 전용 분석</span><span>{titleDate} <i>•</i> Source {run.source_system}/{run.source_component} <i>•</i> Epoch {frame?.epoch ?? 0}</span></div>
        </>}
      </main>
      {run && <footer className="playback-dock"><div className="playback-top"><div className="playback-title"><span className={`status-dot ${playback.playing ? 'teal' : 'gray'}`}/>{isLive?(playback.following?'실시간 관측':'화면 일시정지 · 수집 중'):(playback.playing?'재생 중':'일시정지')}<span className="keyboard-hint">SPACE</span></div><div className="playback-time"><strong>{timeLabel(cursor, true)}</strong><span>/ {timeLabel(Math.max(run.duration_ms,cursor), true)}</span></div><div className="playback-quality"><button onClick={() => setTab('diagnostics')} className="text-button"><CircleAlert size={14}/>제외 {run.excluded_records}건</button><span>최대 10Hz 화면 갱신</span></div></div><div className="playback-controls"><button className="icon-button" aria-label="처음으로" onClick={() => { command('pause'); command('seek', { t_ms: 0 }); }} disabled={!connected}><RotateCcw size={17}/></button><button className="icon-button" aria-label="5초 뒤로" onClick={() => command('seek', { t_ms: cursor - 5000 })} disabled={!connected}><SkipBack size={17}/></button><button className="play-button" aria-label={(isLive?playback.following:playback.playing)?'일시정지':'재생'} onClick={() => command((isLive?playback.following:playback.playing)?'pause':'play')} disabled={!connected}>{(isLive?playback.following:playback.playing) ? <Pause size={18} fill="currentColor"/> : <Play size={18} fill="currentColor"/>}</button><button className="icon-button" aria-label="5초 앞으로" onClick={() => command('seek', { t_ms: cursor + 5000 })} disabled={!connected}><SkipForward size={17}/></button><div className="timeline-track"><input type="range" aria-label="재생 시점" min={isLive?frame?.session?.buffer_start_ms??0:0} max={Math.max(run.duration_ms,cursor)} step="100" value={cursor} onChange={e => command('seek', { t_ms: Number(e.target.value) })} disabled={!connected} style={{ '--progress': `${cursor / Math.max(1, run.duration_ms) * 100}%` } as React.CSSProperties}/><div className="timeline-markers">{events.map(e => <button key={detectionKey(e)} title={`${timeLabel(e.decided_at_ms, true)} ${e.title}`} aria-label={e.title} style={{ left: `${e.decided_at_ms / run.duration_ms * 100}%` }} onClick={() => chooseEvent(e)}/>)}</div></div>{isLive?<button className="button small live-return" onClick={()=>command('go_live')}>실시간으로 복귀</button>:<label className="speed-select"><select aria-label="재생 속도" value={playback.speed} onChange={e => command('speed', { speed: Number(e.target.value) })}>{[.25, .5, 1, 2, 4].map(s => <option value={s} key={s}>{s}×</option>)}</select><ChevronDown size={12}/></label>}</div></footer>}
    </div>
    {aiModal && run && <ProviderDialog runId={runId} initial={providerId} providers={providerDefs} close={()=>setAiModal(false)}/>}
    {sourceModal && <SourceDialog close={()=>setSourceModal(false)} created={m=>{reload(m.run_id);setTab('monitor');}}/>}

  </div>;
}

function ArrowUpRightIcon() { return <ArrowRight size={14}/>; }
