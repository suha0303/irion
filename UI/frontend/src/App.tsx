import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import LocalLive from './LocalLive';
import LocalInsights, { LocalModelCards } from './LocalInsights';
import { useLocalAI } from './useLocalAI';
import { localAlarms } from './local-insights';
import {
  ArrowRight,
  Battery,
  ChevronDown,
  CircleAlert,
  Compass,
  FileUp,
  FlaskConical,
  Info,
  Link2,
  Pause,
  Play,
  Radio,
  RotateCcw,
  SkipBack,
  SkipForward,
  Wind,
  X,
} from 'lucide-react';
import { ApiError, api, post, upload } from './api';
import { PathChart, TimeChart } from './Charts';
import { Diagnostics, EventDetail, EventList } from './Panels';
import type { Detection, Manifest, Metrics, Point, ProviderState } from './types';
import type { Job } from './local-flight-types';
import { detectionKey, numeric, statusLabel, timeLabel } from './types';
import { useReplay } from './useReplay';
import { ProviderCards, ProviderDialog, SourceDialog } from './Integrations';
import { useServerHealth } from './useServerHealth';
import { compareModels } from './comparison';
import { ArchivePanel } from './ArchivePanel';
import './extension.css';

const LocalFlight = lazy(() => import('./LocalFlight'));
const V02Workspace = lazy(() => import('./V02Workspace'));
const Flight3D = lazy(() => import('./Flight3D').then((module) => ({ default: module.Flight3D })));

type Tab = 'monitor' | 'events' | 'diagnostics' | 'local';
const titles = {
  monitor: ['비행 모니터링', '관측과 선택한 모델의 판단을 확인하세요.'],
  local: [
    '로컬 AI · 기체 관리',
    '기체 정보부터 로그 준비, 분석과 파인튜닝까지 단계별로 진행하세요.',
  ],
  events: ['이상 구간 분석', '판정 시점과 관측 데이터를 나란히 살펴보세요.'],
  diagnostics: ['데이터 진단', '선택한 비행의 수신 주기, 누락과 입력 상태를 확인합니다.'],
};

export default function App() {
  const [runs, setRuns] = useState<Manifest[]>([]);
  const [runId, setRunId] = useState(() => {
    try {
      return localStorage.getItem('arion.selectedRunId') ?? '';
    } catch {
      return '';
    }
  });
  const [selectedTab, setTab] = useState<Tab>('monitor');
  const [points, setPoints] = useState<Point[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState('');
  const [retryFile, setRetryFile] = useState<File | null>(null);
  const [uploadedLog, setUploadedLog] = useState<Job | null>(null);
  const server = useServerHealth();
  const [busy, setBusy] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [batteryMetric, setBatteryMetric] = useState('battery_current_a');
  const [aiModal, setAiModal] = useState(false);
  const [providerId, setProviderId] = useState('lstm');
  const [providerFilter, setProviderFilter] = useState('all');
  const [sourceModal, setSourceModal] = useState(false);
  const [localVisited, setLocalVisited] = useState(false);
  const openLocal = () => {
    setLocalVisited(true);
    setTab('local');
  };
  const [providerDefs, setProviderDefs] = useState<ProviderState[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const run = runs.find((r) => r.run_id === runId);
  const availableRunId = run?.run_id ?? '';
  const local = useLocalAI(availableRunId);
  const hasLocal = availableRunId.startsWith('live-');
  const wholeRun = local.state.mode === 'px4_whole_run';
  const bufferedRun = local.state.mode === 'px4_buffered' || local.state.buffered_parallel === true;
  const isV02 = server.edition === 'v0.2';
  const Workspace = isV02 ? V02Workspace : LocalFlight;
  const legacyAnalysis = server.edition === 'legacy' && !wholeRun && !bufferedRun;
  const externalAnalysis = legacyAnalysis && !hasLocal;
  const tab = selectedTab === 'events' && !legacyAnalysis ? 'monitor' : selectedTab;
  const localRows = legacyAnalysis ? localAlarms(local.state) : [];
  const isLive = run?.source_kind === 'live';
  const { frame, playback, connection, command } = useReplay(availableRunId, isLive);
  useEffect(() => {
    if (runId) {
      try {
        localStorage.setItem('arion.selectedRunId', runId);
      } catch {
        /* Browser storage can be unavailable. */
      }
    }
  }, [runId]);
  const configure = (id = 'lstm') => {
    setProviderId(id);
    setAiModal(true);
  };
  const cursor = frame?.t_ms ?? 0;
  const connected = connection === 'connected';
  const events = legacyAnalysis ? (frame?.events ?? []) : [];
  const selectedEvent = events.find((e) => detectionKey(e) === selected) ?? null;
  const filteredEvents = events.filter(
    (e) => providerFilter === 'all' || (e.provider_id ?? 'legacy') === providerFilter,
  );

  const reload = useCallback(async (preferred?: string) => {
    const data = await api<Manifest[]>('/runs');
    setRuns(data);
    setRunId(
      (previous) =>
        preferred ?? (data.some((r) => r.run_id === previous) ? previous : (data[0]?.run_id ?? '')),
    );
  }, []);
  useEffect(() => {
    if (server.status === 'offline') {
      setInitialLoading(false);
      return;
    }
    if (server.status !== 'online') return;
    let cancelled = false;
    reload()
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setInitialLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [server.status, reload]);
  useEffect(() => {
    if (server.status !== 'online' || !externalAnalysis) return;
    let cancelled = false;
    api<ProviderState[]>('/analysis/providers')
      .then((data) => {
        if (!cancelled) setProviderDefs(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [server.status, externalAnalysis]);
  useEffect(() => {
    // A remembered selection may belong to a different local data directory.
    if (!availableRunId) return;
    let stale = false;
    setPoints([]);
    setSelected(null);
    api<{ points: Point[] }>(`/runs/${availableRunId}/series`)
      .then((series) => {
        if (!stale) setPoints(series.points);
      })
      .catch((e) => {
        if (!stale) setError(e.message);
      });
    return () => {
      stale = true;
    };
  }, [availableRunId, isLive]);

  useEffect(() => {
    if (frame?.live_series) setPoints(frame.live_series);
    if (frame?.manifest)
      setRuns((previous) =>
        previous.map((r) => (r.run_id === frame.run_id ? { ...r, ...frame.manifest! } : r)),
      );
  }, [frame?.live_series, frame?.manifest]);
  const sessionAction = async (action: string) => {
    setBusy(true);
    try {
      await post(`/sessions/${runId}/${action}`);
      await reload(runId);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  useEffect(() => {
    if (tab !== 'diagnostics' || !externalAnalysis) return;
    let stale = false;
    const update = () =>
      api<Metrics>('/metrics')
        .then((data) => {
          if (!stale) setMetrics(data);
        })
        .catch(() => {});
    update();
    const interval = setInterval(update, 2000);
    return () => {
      stale = true;
      clearInterval(interval);
    };
  }, [tab, externalAnalysis]);
  useEffect(() => {
    const keypress = (e: KeyboardEvent) => {
      if (
        e.code === 'Space' &&
        !['INPUT', 'BUTTON', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName) &&
        !aiModal &&
        !sourceModal &&
        tab !== 'local' &&
        runId
      ) {
        e.preventDefault();
        command((isLive ? playback.following : playback.playing) ? 'pause' : 'play');
      }
      if (e.code === 'Escape' && (aiModal || sourceModal)) {
        setAiModal(false);
        setSourceModal(false);
      }
    };
    document.addEventListener('keydown', keypress);
    return () => document.removeEventListener('keydown', keypress);
  });

  async function importLog(file?: File) {
    if (!file) return;
    setBusy(true);
    setError('');
    setRetryFile(null);
    try {
      setUploadedLog(await upload<Job>('/local-ai/logs', file));
      openLocal();
    } catch (e) {
      setError((e as Error).message);
      if (e instanceof ApiError && e.kind === 'network') {
        setRetryFile(file);
        void server.check();
      }
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = '';
    }
  }
  async function toggleDemo() {
    try {
      await post(`/runs/${runId}/analysis-demo`, { enabled: !frame?.fixture_mode });
      setSelected(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const value = (key: string) => (connected ? frame?.values[key]?.value : null);
  const chooseEvent = (event: Detection) => {
    command('pause');
    command('seek', { t_ms: event.decided_at_ms });
    setSelected(detectionKey(event));
    setTab('events');
  };
  const comparison = compareModels(
    connected ? frame?.active_by_provider?.lstm : null,
    connected ? frame?.active_by_provider?.transformer : null,
  );
  const extraProviders = externalAnalysis
    ? (frame?.providers
        ?.filter((p) => !['lstm', 'transformer'].includes(p.provider_id))
        .map((p) => p.provider_id) ?? [])
    : [];
  return (
    <div className="app-shell balanced-workspace">
      <input
        ref={fileInput}
        type="file"
        accept=".ulg,.ULG,.bin,.BIN"
        className="visually-hidden"
        aria-label="PX4 ULog 또는 ArduPilot BIN 비행 로그"
        onChange={(e) => importLog(e.target.files?.[0])}
      />
      <div className="app-main">
        <header className="workspace-topbar">
          <a
            className="workspace-brand"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              setTab('monitor');
            }}
            aria-label="ARION 비행 모니터링"
          >
            ARION<small>{isV02 ? 'FLIGHT INSIGHTS · V0.2' : 'FLIGHT INSIGHTS'}</small>
          </a>
          <nav aria-label="주요 화면">
            {(['monitor', 'events', 'local', 'diagnostics'] as Tab[])
              .filter((key) => key !== 'events' || legacyAnalysis)
              .map((key) => (
                <button
                  key={key}
                  className={tab === key ? 'active' : ''}
                  onClick={() => (key === 'local' ? openLocal() : setTab(key))}
                  aria-current={tab === key ? 'page' : undefined}
                >
                  {titles[key][0]}
                  {key === 'events' && (hasLocal ? localRows.length : events.length) > 0 && (
                    <span className="nav-event-count">
                      {hasLocal ? localRows.length : events.length}
                    </span>
                  )}
                </button>
              ))}
          </nav>
          <span className={`server-indicator ${server.status}`} role="status">
            <i />
            {server.status === 'online'
              ? '로컬 서버 연결됨'
              : server.status === 'offline'
                ? '서버 오프라인'
                : '연결 확인 중'}
          </span>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <h1>{titles[tab][0]}</h1>
              <p>
                {tab === 'local' && server.wholeRunOnly
                  ? '등록 기체와 연결된 전체 런 모델을 확인하세요.'
                  : titles[tab][1]}
              </p>
            </div>
            {(tab === 'monitor' || (tab === 'local' && server.wholeRunOnly)) && (
              <div className="heading-actions">
                {tab === 'monitor' && (
                  <button
                    className="button"
                    onClick={() => setSourceModal(true)}
                    disabled={server.status !== 'online' || busy}
                  >
                    <Radio size={17} />
                    실시간 연결
                  </button>
                )}
                <button
                  className="button primary"
                  onClick={() => fileInput.current?.click()}
                  disabled={server.status !== 'online' || busy}
                >
                  <FileUp size={17} />
                  비행 로그 가져오기
                </button>
              </div>
            )}
          </div>
          {server.status === 'offline' && (
            <section className="server-offline" role="alert">
              <Link2 size={19} />
              <div>
                <strong>대시보드 서버에 연결할 수 없습니다</strong>
                <p>
                  터미널에서 대시보드 폴더의 <code>./server.sh start</code>를 실행하세요. 화면과
                  업로드는 서버가 실행 중일 때 동작합니다.
                </p>
                <small>소스 실행: 프로젝트의 UI 폴더 · 설치형 실행: ./drone-local start</small>
              </div>
              <button className="button small" onClick={() => server.check()}>
                연결 다시 확인
              </button>
            </section>
          )}
          {error && (
            <div className="notice error upload-notice" role="alert">
              <CircleAlert size={18} />
              <div>
                <strong>
                  {retryFile ? `${retryFile.name} · 가져오기 대기` : '요청을 완료하지 못했습니다'}
                </strong>
                <p>{error}</p>
              </div>
              {retryFile && (
                <button
                  className="button small"
                  onClick={() => importLog(retryFile)}
                  disabled={busy}
                >
                  가져오기 다시 시도
                </button>
              )}
              <button
                className="icon-button"
                aria-label="오류 메시지 닫기"
                onClick={() => {
                  setError('');
                  setRetryFile(null);
                }}
              >
                <X size={16} />
              </button>
            </div>
          )}
          {localVisited && server.edition && (
            <div hidden={tab !== 'local'} className="local-management-page">
              <Suspense fallback={<div role="status">기체 관리 화면을 불러오는 중…</div>}>
                <Workspace
                  active={tab === 'local'}
                  uploadedLog={uploadedLog}
                  chooseLog={() => fileInput.current?.click()}
                  uploading={busy}
                />
              </Suspense>
            </div>
          )}
          {tab === 'local' ? null : !run ? (
            <div className="panel welcome">
              <div className="welcome-icon">
                <Wind size={40} />
              </div>
              <h2>
                {server.status === 'offline'
                  ? '서버 실행 후 비행 기록을 불러옵니다'
                  : initialLoading
                    ? '비행 기록을 불러오고 있습니다'
                    : '저장된 비행이 없습니다'}
              </h2>
              {server.status === 'online' && !initialLoading && (
                <p>실시간으로 연결하거나 PX4 .ulg 또는 ArduPilot .bin 로그를 가져오세요.</p>
              )}
            </div>
          ) : (
            <>
              <section className="flight-bar">
                <div className="flight-icon">
                  <Wind size={23} />
                </div>
                <div className="flight-selector">
                  <label htmlFor="flight-select">선택한 비행</label>
                  <div>
                    <select
                      id="flight-select"
                      value={runId}
                      disabled={busy}
                      onChange={(e) => setRunId(e.target.value)}
                      aria-label="비행 선택"
                    >
                      {runs.map((r) => (
                        <option key={r.run_id} value={r.run_id}>
                          {r.name}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={15} />
                  </div>
                </div>
                <div className="flight-meta">
                  <span>{run.source}</span>
                  <span>{(run.duration_ms / 1000).toFixed(2)}초</span>
                  <span>{run.accepted_records.toLocaleString()}개 관측</span>
                </div>
                <span className="pill blue">
                  <span className="status-dot" />
                  {isLive ? '실시간 수신' : '기록 재생'}
                </span>
              </section>
              {isLive && (
                <div className="session-toolbar">
                  <span>
                    <Radio size={16} />
                    {statusLabel[frame?.session?.state ?? run.session?.state ?? 'waiting'] ??
                      '수집 중'}{' '}
                    · 수집 {frame?.session?.last_sequence ?? 0}건
                  </span>
                  <code>{runId}</code>
                  <button
                    className="button small"
                    disabled={busy}
                    onClick={() => sessionAction('connect')}
                  >
                    입력 재연결
                  </button>
                  <button
                    className="button small"
                    disabled={busy}
                    onClick={() => sessionAction('finish')}
                  >
                    비행 종료 · 결과 분석
                  </button>
                  {frame?.session?.message && (
                    <p className="inline-error">{frame.session.message}</p>
                  )}
                </div>
              )}
              {connection === 'reconnecting' && server.status === 'online' && (
                <div className="notice error" role="alert">
                  <Link2 size={18} />
                  <div>
                    <strong>서버 연결이 끊겼습니다</strong>
                    <p>자동 재연결 중입니다. 마지막 시각을 보존하며 재생은 일시정지됩니다.</p>
                  </div>
                </div>
              )}
              {externalAnalysis && frame?.fixture_mode && (
                <div className="notice warning fixture-banner">
                  <FlaskConical size={18} />
                  <div>
                    <strong>시험용 AI 응답 모드</strong>
                    <span>
                      표시되는 판정은 화면 검증용이며, 실제 고장이나 모델 탐지 결과가 아닙니다.
                    </span>
                  </div>
                  <button className="text-button" onClick={toggleDemo}>
                    시험 종료
                    <X size={13} />
                  </button>
                </div>
              )}
              {tab === 'monitor' && (
                <>
                  {hasLocal &&
                    (legacyAnalysis ||
                      bufferedRun ||
                      (wholeRun && run.session?.kind === 'udp')) && (
                      <LocalLive
                        key={runId}
                        runId={runId}
                        live={!!isLive}
                        manage={openLocal}
                        state={local.state}
                        onState={local.setState}
                      />
                    )}
                  <ArchivePanel run={run} frame={frame} />
                  <div className={legacyAnalysis ? 'monitor-hero' : 'monitor-hero single-panel'}>
                    <Suspense
                      fallback={
                        <section className="panel" role="status">
                          3D 비행 화면을 불러오는 중…
                        </section>
                      }
                    >
                      <Flight3D frame={frame} points={points} connected={connected} />
                    </Suspense>
                    {legacyAnalysis && (
                      <aside
                        className="model-comparison"
                        aria-label={hasLocal ? '선택한 로컬 모델 상태' : '외부 모델 비교'}
                      >
                        <div className="comparison-heading">
                          <h2>모델 비교</h2>
                          <span>관측과 별도 판정</span>
                        </div>
                        {hasLocal ? (
                          <LocalModelCards
                            state={local.state}
                            error={local.error}
                            showEvents={() => setTab('events')}
                          />
                        ) : (
                          <>
                            <ProviderCards
                              frame={frame}
                              connected={connected}
                              configure={configure}
                              providerIds={['lstm', 'transformer']}
                              choose={chooseEvent}
                              offline={server.status === 'offline'}
                            />
                            <div className={`comparison-summary ${comparison.state}`}>
                              <Info size={15} />
                              <span>{comparison.label}</span>
                            </div>
                          </>
                        )}
                      </aside>
                    )}
                  </div>
                  <section className="observation-strip" aria-label="현재 비행 관측">
                    <div className="observation-item">
                      <span>
                        <Battery size={14} />
                        배터리 SOC
                      </span>
                      <strong>
                        {numeric(value('battery_soc_pct'), 0)}
                        <small>%</small>
                      </strong>
                      <em>충전 잔량 · 건강도와 구분</em>
                    </div>
                    <div className="observation-item">
                      <span>팩 전압 / 전류</span>
                      <strong>
                        {numeric(value('battery_voltage_v'), 2)}
                        <small>V</small>
                        <i>/</i>
                        {numeric(value('battery_current_a'), 2)}
                        <small>A</small>
                      </strong>
                      <em>수신된 관측값</em>
                    </div>
                    <div className="observation-item">
                      <span>
                        <Compass size={14} />
                        상대 높이 / 속력
                      </span>
                      <strong>
                        {numeric(value('height_m'), 2)}
                        <small>m</small>
                        <i>/</i>
                        {numeric(value('speed_m_s'), 2)}
                        <small>m/s</small>
                      </strong>
                      <em>
                        {connected ? (frame?.flight_state ?? '미확인') : '연결 끊김'} ·{' '}
                        {connected && frame?.armed != null
                          ? frame.armed
                            ? 'ARMED'
                            : 'DISARMED'
                          : '—'}{' '}
                        · {connected ? (frame?.mode ?? '—') : '—'}
                      </em>
                    </div>
                  </section>
                  {extraProviders.length > 0 && (
                    <details className="extra-providers">
                      <summary>추가 분석 제공자 {extraProviders.length}개</summary>
                      <ProviderCards
                        frame={frame}
                        connected={connected}
                        configure={configure}
                        providerIds={extraProviders}
                        choose={chooseEvent}
                        offline={server.status === 'offline'}
                      />
                    </details>
                  )}
                  <section className="panel battery-trend">
                    <div className="panel-heading">
                      <div>
                        <span className="eyebrow">POWER TREND</span>
                        <h2>배터리 추이</h2>
                      </div>
                      <div className="segmented" aria-label="배터리 지표">
                        {[
                          { key: 'battery_current_a', label: '전류' },
                          { key: 'battery_voltage_v', label: '전압' },
                          { key: 'battery_soc_pct', label: 'SOC' },
                        ].map((m) => (
                          <button
                            key={m.key}
                            className={batteryMetric === m.key ? 'active' : ''}
                            onClick={() => setBatteryMetric(m.key)}
                          >
                            {m.label}
                          </button>
                        ))}
                      </div>
                    </div>
                    <TimeChart
                      points={points}
                      cursor={cursor}
                      signals={[
                        {
                          key: batteryMetric,
                          name:
                            batteryMetric === 'battery_current_a'
                              ? '팩 전류'
                              : batteryMetric === 'battery_voltage_v'
                                ? '팩 전압'
                                : 'SOC',
                        },
                      ]}
                      colors={['#0aab97']}
                      unit={
                        batteryMetric === 'battery_current_a'
                          ? 'A'
                          : batteryMetric === 'battery_voltage_v'
                            ? 'V'
                            : '%'
                      }
                    />
                  </section>
                  <div className={legacyAnalysis ? 'bottom-grid' : undefined}>
                    <section className="panel">
                      <div className="panel-heading">
                        <div>
                          <span className="eyebrow">LOCAL TRAJECTORY</span>
                          <h2>비행 궤적</h2>
                        </div>
                        <span className="pill neutral">NED</span>
                      </div>
                      <PathChart points={points} cursor={cursor} />
                    </section>
                    {legacyAnalysis &&
                      (hasLocal ? (
                        <section className="panel event-summary">
                          <div className="panel-heading">
                            <h2>로컬 AI 판정 이력</h2>
                            <button className="text-button" onClick={() => setTab('events')}>
                              전체 보기
                            </button>
                          </div>
                          <div className="inner">
                            <p>
                              {local.state.insights?.history_available
                                ? `실시간 기록 경보 ${local.state.insights.alarm_count}건`
                                : '실시간 경보 이력 없음'}
                            </p>
                            <p>
                              저장 로그 기준 초과{' '}
                              {localRows.filter((row) => row.source === 'log').length}시점
                            </p>
                            <p>예측과 종료 후 모델의 결과를 이상 구간 분석에서 확인하세요.</p>
                          </div>
                        </section>
                      ) : (
                        <section className="panel event-summary">
                          <div className="panel-heading">
                            <div>
                              <span className="eyebrow">DETECTION TIMELINE</span>
                              <h2>
                                판정 이력 <span className="heading-note">{events.length}건</span>
                              </h2>
                            </div>
                            <button className="text-button" onClick={() => setTab('events')}>
                              전체 보기
                              <ArrowUpRightIcon />
                            </button>
                          </div>
                          <EventList events={events} selected={selected} choose={chooseEvent} />
                          <div className="event-summary-foot">
                            <Info size={14} />
                            {frame?.fixture_mode
                              ? '시험용 응답의 판정 시각을 따라 표시합니다.'
                              : '이상 점수는 고장 확률이나 건강도 %가 아닙니다.'}
                          </div>
                        </section>
                      ))}
                  </div>
                </>
              )}
              {tab === 'events' && hasLocal && (
                <LocalInsights key={runId} state={local.state} error={local.error} />
              )}
              {tab === 'events' && !hasLocal && (
                <>
                  <div className="event-tools">
                    <div>
                      <Info size={16} />
                      <span>
                        재생 시각까지 도착한 판정만 표시됩니다. 이벤트를 누르면 판정 시점으로
                        이동합니다.
                      </span>
                    </div>
                    <button
                      className={`button ${frame?.fixture_mode ? 'amber-button' : ''}`}
                      disabled={isLive}
                      onClick={toggleDemo}
                    >
                      <FlaskConical size={16} />
                      {frame?.fixture_mode ? '시험 이벤트 끄기' : '시험 이벤트 켜기'}
                    </button>
                  </div>
                  <ProviderCards
                    frame={frame}
                    connected={connected}
                    configure={configure}
                    choose={chooseEvent}
                    offline={server.status === 'offline'}
                  />
                  <label className="provider-filter">
                    제공자 필터{' '}
                    <select
                      className="text-input"
                      value={providerFilter}
                      onChange={(e) => setProviderFilter(e.target.value)}
                    >
                      <option value="all">전체 제공자</option>
                      {providerDefs.map((p) => (
                        <option value={p.provider_id} key={p.provider_id}>
                          {p.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="events-layout">
                    <section className="panel">
                      <div className="panel-heading">
                        <h2>판정 이력</h2>
                        <span className="pill neutral">{events.length}건</span>
                      </div>
                      <EventList events={filteredEvents} selected={selected} choose={chooseEvent} />
                    </section>
                    <EventDetail event={selectedEvent} points={points} cursor={cursor} />
                  </div>
                </>
              )}
              {tab === 'diagnostics' && (
                <Diagnostics
                  run={run}
                  frame={connected ? frame : null}
                  metrics={metrics}
                  connectAI={externalAnalysis ? () => configure() : undefined}
                />
              )}
            </>
          )}
        </main>
        {run && tab !== 'local' && (
          <footer className="playback-dock">
            <div className="playback-top">
              <div className="playback-title">
                <span className={`status-dot ${playback.playing ? 'teal' : 'gray'}`} />
                {isLive
                  ? playback.following
                    ? '실시간 관측'
                    : '화면 일시정지 · 수집 중'
                  : playback.playing
                    ? '재생 중'
                    : '일시정지'}
                <span className="keyboard-hint">SPACE</span>
              </div>
              <div className="playback-time">
                <strong>{timeLabel(cursor, true)}</strong>
                <span>/ {timeLabel(Math.max(run.duration_ms, cursor), true)}</span>
              </div>
            </div>
            <div className="playback-controls">
              <button
                className="icon-button"
                aria-label="처음으로"
                onClick={() => {
                  command('pause');
                  command('seek', { t_ms: 0 });
                }}
                disabled={!connected}
              >
                <RotateCcw size={17} />
              </button>
              <button
                className="icon-button"
                aria-label="5초 뒤로"
                onClick={() => command('seek', { t_ms: cursor - 5000 })}
                disabled={!connected}
              >
                <SkipBack size={17} />
              </button>
              <button
                className="play-button"
                aria-label={(isLive ? playback.following : playback.playing) ? '일시정지' : '재생'}
                onClick={() =>
                  command((isLive ? playback.following : playback.playing) ? 'pause' : 'play')
                }
                disabled={!connected}
              >
                {(isLive ? playback.following : playback.playing) ? (
                  <Pause size={18} fill="currentColor" />
                ) : (
                  <Play size={18} fill="currentColor" />
                )}
              </button>
              <button
                className="icon-button"
                aria-label="5초 앞으로"
                onClick={() => command('seek', { t_ms: cursor + 5000 })}
                disabled={!connected}
              >
                <SkipForward size={17} />
              </button>
              <div className="timeline-track">
                <input
                  type="range"
                  aria-label="재생 시점"
                  min={isLive ? (frame?.session?.buffer_start_ms ?? 0) : 0}
                  max={Math.max(run.duration_ms, cursor)}
                  step="100"
                  value={cursor}
                  onChange={(e) => command('seek', { t_ms: Number(e.target.value) })}
                  disabled={!connected}
                  style={
                    {
                      '--progress': `${(cursor / Math.max(1, run.duration_ms)) * 100}%`,
                    } as React.CSSProperties
                  }
                />
                <div className="timeline-markers">
                  {events.map((e) => (
                    <button
                      key={detectionKey(e)}
                      title={`${timeLabel(e.decided_at_ms, true)} ${e.title}`}
                      aria-label={e.title}
                      style={{ left: `${(e.decided_at_ms / run.duration_ms) * 100}%` }}
                      onClick={() => chooseEvent(e)}
                    />
                  ))}
                </div>
              </div>
              {isLive ? (
                <button className="button small live-return" onClick={() => command('go_live')}>
                  실시간으로 복귀
                </button>
              ) : (
                <label className="speed-select">
                  <select
                    aria-label="재생 속도"
                    value={playback.speed}
                    onChange={(e) => command('speed', { speed: Number(e.target.value) })}
                  >
                    {[0.25, 0.5, 1, 2, 4].map((s) => (
                      <option value={s} key={s}>
                        {s}×
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={12} />
                </label>
              )}
            </div>
          </footer>
        )}
      </div>
      {externalAnalysis && aiModal && run && (
        <ProviderDialog
          runId={runId}
          initial={providerId}
          providers={providerDefs}
          close={() => setAiModal(false)}
        />
      )}
      {sourceModal && (
        <SourceDialog
          isV02={isV02}
          close={() => setSourceModal(false)}
          created={(m) => {
            setRuns((previous) => [m, ...previous.filter((r) => r.run_id !== m.run_id)]);
            setRunId(m.run_id);
            setTab('monitor');
          }}
        />
      )}
    </div>
  );
}

function ArrowUpRightIcon() {
  return <ArrowRight size={14} />;
}
