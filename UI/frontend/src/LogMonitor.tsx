import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { FileClock } from 'lucide-react';
import { api } from './api';
import { FlightBar, FlightPose, MonitorCharts, ObservationStrip } from './FlightMonitorViews';
import PlaybackDock, { type PlaybackCommand } from './PlaybackDock';
import LogAnalysis from './LogAnalysis';
import type { MonitorTimeline } from './FlightAnalysisViews';
import { jobFilename, type Job } from './local-flight-types';
import type { Point, Signal } from './types';
import './local-flight.css';

type LogSeries = {
  points: Point[];
  duration_ms: number;
  step_ms: number;
  start_us?: number;
  warning?: string | null;
};

function pointAt(points: Point[], cursor: number): Point | undefined {
  let left = 0;
  let right = points.length;
  while (left < right) {
    const middle = Math.floor((left + right) / 2);
    if (points[middle].t_ms <= cursor) left = middle + 1;
    else right = middle;
  }
  return points[left - 1];
}

function LogPlayback({
  series,
  logId,
  children,
}: {
  series?: LogSeries;
  logId: string;
  children: (timeline?: MonitorTimeline) => ReactNode;
}) {
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [metric, setMetric] = useState('battery_current_a');
  const {
    points,
    duration_ms: duration,
    step_ms: step,
    warning,
  } = series ?? {
    points: [],
    duration_ms: 0,
    step_ms: 0,
  };
  const available = points.length > 0;
  const running = playing && cursor < duration;

  useEffect(() => {
    if (!running) return;
    let previous = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      const elapsed = (now - previous) * speed;
      previous = now;
      setCursor((time) => Math.min(duration, time + elapsed));
    }, 100);
    return () => window.clearInterval(timer);
  }, [duration, running, speed]);

  const seekMs = useCallback(
    (time: number) => {
      setCursor(Math.max(0, Math.min(duration, time)));
    },
    [duration],
  );
  const command: PlaybackCommand = (action, args = {}) => {
    if (!available) return;
    switch (action) {
      case 'seek':
        if (cursor >= duration) setPlaying(false);
        seekMs(args.t_ms ?? 0);
        break;
      case 'speed':
        setSpeed(args.speed ?? 1);
        break;
      case 'play':
        if (cursor >= duration) setCursor(0);
        setPlaying(true);
        break;
      case 'pause':
        setPlaying(false);
        break;
    }
  };

  useEffect(() => {
    function handleSpace(event: KeyboardEvent) {
      if (
        event.code === 'Space' &&
        !event.repeat &&
        !['INPUT', 'BUTTON', 'SELECT', 'TEXTAREA', 'A', 'SUMMARY'].includes(
          (event.target as HTMLElement)?.tagName,
        ) &&
        available
      ) {
        event.preventDefault();
        setPlaying(!running);
        if (cursor >= duration) setCursor(0);
      }
    }
    document.addEventListener('keydown', handleSpace);
    return () => document.removeEventListener('keydown', handleSpace);
  }, [available, cursor, duration, running]);

  const sample = pointAt(points, cursor);
  const fresh = sample && cursor - sample.t_ms <= Math.max(500, step * 2);
  const values: Record<string, Signal> = {};
  if (fresh) {
    for (const [key, value] of Object.entries(sample.values)) {
      values[key] = {
        value,
        unit: '',
        status: value == null ? 'missing' : 'valid',
        age_ms: null,
        label: key,
        message: '',
        reason: '',
      };
    }
  }
  const frame = {
    run_id: logId,
    epoch: sample?.epoch ?? 0,
    t_ms: sample?.t_ms ?? 0,
    values,
    armed: null,
    flight_state: '로그 관측',
    mode: '미확인',
  };

  return (
    <>
      {series && <ObservationStrip frame={frame} available={available} recorded />}
      {children(
        available && series?.start_us != null
          ? { originUs: series.start_us, cursorMs: cursor, seekMs }
          : undefined,
      )}
      {series && (
        <>
          {warning && <p className="notice warning">{warning}</p>}
          {!available && <p className="notice warning">표시할 수 있는 비행 관측값이 없습니다.</p>}
          <FlightPose frame={frame} points={points} available={available} />
          <MonitorCharts points={points} cursor={cursor} metric={metric} onMetric={setMetric} />
          <PlaybackDock
            isLive={false}
            cursor={cursor}
            duration={duration}
            playback={{ playing: running, following: false, speed }}
            command={command}
            available={available && duration > 0}
          />
        </>
      )}
    </>
  );
}

export default function LogMonitor({
  isV02,
  uploadedLog,
  selectedId,
  selectLog,
  openInspection,
}: {
  isV02: boolean;
  uploadedLog: Job | null;
  selectedId: string;
  selectLog: (id: string) => void;
  openInspection: (log: Job) => void;
}) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [listError, setListError] = useState('');
  const [liveBusy, setLiveBusy] = useState(false);
  const [view, setView] = useState<{ id: string; data?: LogSeries; error?: string }>();

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const result = await api<{ jobs: Job[]; live_busy?: boolean }>('/local-ai/jobs', {
          signal: controller.signal,
        });
        if (!controller.signal.aborted) {
          setJobs(result.jobs);
          setLiveBusy(Boolean(result.live_busy));
          setListError('');
        }
      } catch (failure) {
        if (!controller.signal.aborted) setListError(String(failure));
      }
      if (!controller.signal.aborted) {
        setLoaded(true);
        timer = setTimeout(refresh, 3000);
      }
    }
    void refresh();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  const inspections = jobs.filter((job) => job.action === 'inspect');
  if (uploadedLog && !inspections.some((job) => job.id === uploadedLog.id)) {
    inspections.unshift(uploadedLog);
  }
  const sourceKey = (job: Job) => job.request?.path ?? job.id;
  const sources = new Map<string, Job>();
  for (const inspection of inspections) {
    const key = sourceKey(inspection);
    if (!sources.has(key)) sources.set(key, inspection);
  }
  const logs = [...sources.values()];
  const selection = inspections.find((job) => job.id === selectedId);
  const log =
    (selection && sources.get(sourceKey(selection))) ??
    logs.find((job) => job.status === 'complete') ??
    logs[0];
  const completeId = log?.status === 'complete' ? log.id : '';

  useEffect(() => {
    if (!completeId) return;
    const controller = new AbortController();
    api<LogSeries>(`/local-ai/logs/${completeId}/monitor`, { signal: controller.signal }).then(
      (data) => {
        if (!controller.signal.aborted) setView({ id: completeId, data });
      },
      (failure: unknown) => {
        if (!controller.signal.aborted) setView({ id: completeId, error: String(failure) });
      },
    );
    return () => controller.abort();
  }, [completeId]);

  if (!log) {
    return (
      <section className="panel welcome">
        <div className="welcome-icon">
          <FileClock size={36} />
        </div>
        <h2>
          {listError
            ? '로그 목록을 불러오지 못했습니다'
            : loaded
              ? '가져온 비행 로그가 없습니다'
              : '로그 목록을 불러오는 중…'}
        </h2>
        {listError && <p role="alert">{listError}</p>}
        {loaded && !listError && (
          <p>위의 비행 로그 가져오기에서 .ulg 또는 .bin 파일을 선택하세요.</p>
        )}
      </section>
    );
  }

  const current = view?.id === log.id ? view : undefined;
  const source =
    log.result?.autopilot === 'px4'
      ? 'PX4 ULog'
      : log.result?.autopilot === 'ardupilot'
        ? 'ArduPilot BIN'
        : '비행제어기 로그';
  return (
    <>
      <FlightBar
        icon={<FileClock size={23} />}
        id="log-monitor-select"
        label="비행 로그 선택"
        value={log.id}
        onChange={selectLog}
        options={logs.map((item) => ({
          id: item.id,
          label: jobFilename(item) ?? item.id.slice(0, 8),
        }))}
        source={source}
        duration={current?.data?.duration_ms}
        status="로그 재생"
      />
      {listError && (
        <p className="notice error" role="alert">
          {listError}
        </p>
      )}
      <LogPlayback key={log.id} series={current?.data} logId={log.id}>
        {(timeline) => (
          <LogAnalysis
            log={log}
            jobs={jobs}
            isV02={isV02}
            openInspection={openInspection}
            onJob={(job) => setJobs((rows) => [job, ...rows.filter((row) => row.id !== job.id)])}
            liveBusy={liveBusy}
            timeline={timeline}
          />
        )}
      </LogPlayback>
      {log.status === 'queued' || log.status === 'running' ? (
        <p className="panel" role="status">
          가져온 로그를 검사하는 중…
        </p>
      ) : !completeId ? (
        <p className="notice error" role="alert">
          {log.error || '로그 검사 결과를 확인할 수 없습니다.'}
        </p>
      ) : current?.error ? (
        <p className="notice error" role="alert">
          {current.error}
        </p>
      ) : !current?.data ? (
        <p className="panel" role="status">
          로그 관측값을 불러오는 중…
        </p>
      ) : null}
    </>
  );
}
