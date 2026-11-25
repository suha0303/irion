import type { ReactNode } from 'react';

export function FlightAnalysisPanel({
  status,
  action,
  children,
}: {
  status?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="panel local-live" aria-label="비행 분석">
      <div className="panel-heading">
        <h2>비행 분석</h2>
        <div className="flight-analysis-actions">
          {status && <span className="pill neutral">{status}</span>}
          {action}
        </div>
      </div>
      {children}
    </section>
  );
}

export function FlightModelControls({ children }: { children: ReactNode }) {
  return (
    <section className="local-postflight" aria-label="이 비행의 모델 선택">
      <h3>이 비행의 모델 선택</h3>
      {children}
    </section>
  );
}

export type MonitorTimeline = {
  originUs: number;
  cursorMs: number;
  seekMs: (time: number) => void;
};

export function FlightResultSelect({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}) {
  if (options.length < 2) return null;
  return (
    <label className="form-label">
      표시할 분석 결과
      <select
        className="text-input"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((item) => (
          <option key={item.value} value={item.value}>
            {item.label}
          </option>
        ))}
      </select>
    </label>
  );
}

type Verdict = {
  key: string;
  model: string;
  candidate?: boolean;
  alarm: boolean | null;
  score: number | null;
  threshold?: number;
  detail?: string;
};

export function FlightVerdictCards({ items }: { items: Verdict[] }) {
  return (
    <div className="whole-run-model-grid">
      {items.map((result) => (
        <div className="panel provider-card" key={result.key}>
          <h4>{result.model}</h4>
          {result.candidate && <small>파인튜닝 모델 · 전용 보정 기준</small>}
          <div className={`provider-status ${result.alarm ? 'suspect' : ''}`}>
            {result.alarm === null ? '판정 불가' : result.alarm ? '이상 의심' : '정상 범위'}
          </div>
          {result.score != null && result.threshold != null && (
            <p>
              점수 {result.score.toFixed(3)} · 기준 {result.threshold.toFixed(3)}
            </p>
          )}
          {result.detail && <p>{result.detail}</p>}
        </div>
      ))}
    </div>
  );
}

export function FlightPostflight({ children }: { children: ReactNode }) {
  return (
    <section className="local-postflight" aria-label="비행 후 판정">
      <h3>비행 후 판정</h3>
      {children}
    </section>
  );
}
