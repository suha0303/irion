import { lazy, Suspense, useState, type ReactNode } from 'react';
import { Battery, ChevronDown, Compass } from 'lucide-react';
import { PathChart, TimeChart } from './Charts';
import { numeric, type Frame, type Point } from './types';

const Flight3D = lazy(() => import('./Flight3D').then((module) => ({ default: module.Flight3D })));

export function FlightBar({
  icon,
  id,
  label,
  value,
  onChange,
  options,
  source,
  duration,
  observations,
  status,
  disabled,
}: {
  icon: ReactNode;
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { id: string; label: string; keywords?: string }[];
  source: string;
  duration?: number;
  observations?: number;
  status: string;
  disabled?: boolean;
}) {
  const [query, setQuery] = useState('');
  const needle = query.trim().toLocaleLowerCase();
  const matching = options.filter((item) =>
    `${item.label} ${item.keywords ?? ''}`.toLocaleLowerCase().includes(needle),
  );
  const selected = options.find((item) => item.id === value);
  const visible = selected && !matching.includes(selected) ? [selected, ...matching] : matching;
  return (
    <section className="flight-bar">
      <div className="flight-icon">{icon}</div>
      <div className="flight-selector">
        <label htmlFor={id}>{label}</label>
        <div>
          <select
            id={id}
            value={value}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value)}
          >
            {visible.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
          <ChevronDown size={15} />
        </div>
        {options.length > 12 && (
          <>
            <input
              className="text-input flight-search"
              type="search"
              aria-label="비행 기록 검색"
              placeholder="비행 이름 · 입력 방식 검색"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            {needle && <small role="status">검색 결과 {matching.length}건</small>}
          </>
        )}
      </div>
      <div className="flight-meta">
        <span>{source}</span>
        {duration != null && <span>{(duration / 1000).toFixed(1)}초</span>}
        {observations != null && <span>{observations.toLocaleString()}개 관측</span>}
      </div>
      <span className="pill blue">
        <span className="status-dot" />
        {status}
      </span>
    </section>
  );
}

export function FlightPose({
  frame,
  points,
  available,
  aside,
}: {
  frame: Pick<Frame, 'run_id' | 'epoch' | 't_ms' | 'values'> | null;
  points: Point[];
  available: boolean;
  aside?: ReactNode;
}) {
  return (
    <div className={aside ? 'monitor-hero' : 'monitor-hero single-panel'}>
      <Suspense
        fallback={
          <section className="panel" role="status">
            3D 비행 화면을 불러오는 중…
          </section>
        }
      >
        <Flight3D frame={frame} points={points} connected={available} />
      </Suspense>
      {aside}
    </div>
  );
}

export function ObservationStrip({
  frame,
  available,
  recorded = false,
}: {
  frame: Pick<Frame, 'values' | 'armed' | 'flight_state' | 'mode'> | null;
  available: boolean;
  recorded?: boolean;
}) {
  const value = (key: string) => (available ? frame?.values[key]?.value : null);
  return (
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
        <em>{recorded ? '기록된 관측값' : '수신된 관측값'}</em>
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
          {recorded ? '로그 관측값' : available ? (frame?.flight_state ?? '미확인') : '연결 끊김'}
          {!recorded && (
            <>
              {' · '}
              {available && frame?.armed != null ? (frame.armed ? 'ARMED' : 'DISARMED') : '—'}
              {' · '}
              {available ? (frame?.mode ?? '—') : '—'}
            </>
          )}
        </em>
      </div>
    </section>
  );
}

export function MonitorCharts({
  points,
  cursor,
  metric,
  onMetric,
}: {
  points: Point[];
  cursor: number;
  metric: string;
  onMetric: (value: string) => void;
}) {
  return (
    <div className="monitor-chart-grid">
      <section className="panel">
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
                className={metric === m.key ? 'active' : ''}
                onClick={() => onMetric(m.key)}
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
              key: metric,
              name:
                metric === 'battery_current_a'
                  ? '팩 전류'
                  : metric === 'battery_voltage_v'
                    ? '팩 전압'
                    : 'SOC',
            },
          ]}
          colors={['#0aab97']}
          unit={metric === 'battery_current_a' ? 'A' : metric === 'battery_voltage_v' ? 'V' : '%'}
        />
      </section>
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
    </div>
  );
}
