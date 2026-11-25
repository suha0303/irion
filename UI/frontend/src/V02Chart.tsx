import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { api } from './api';
import { PredictionPlot, PredictionSignalSelect, type PredictionData } from './BufferedFlightChart';
import { V02_RULES, type V02Prediction, type V02Rule } from './local-flight-types';
import type { MonitorTimeline } from './FlightAnalysisViews';

type Value = number | null;
type Values = Value | Value[];
type Rule = {
  threshold: number[];
  ema: Value[][];
  raw_score?: Value[][];
  domains: (boolean | null)[][];
};
type Segment = {
  time_us: number[];
  observed: Values[];
  prediction: Partial<Record<V02Prediction, Values[]>>;
  residual: Partial<Record<V02Prediction, Values[]>>;
  rules?: Partial<Record<V02Rule, Rule>>;
};
type Data = { task: 'inflight' | 'whole-run'; segments: Segment[] };

function voltageOrVector(value: Values | undefined): Value[] {
  return Array.isArray(value) ? value : [null, null, null, value ?? null];
}

export default function V02Chart({
  jobId,
  modelId,
  index,
  primaryRule,
  compare,
  modelSelect,
  timeline,
}: {
  jobId: string;
  modelId: string;
  index: number;
  primaryRule: V02Rule;
  compare: boolean;
  modelSelect?: ReactNode;
  timeline?: MonitorTimeline;
}) {
  const [data, setData] = useState<Data>();
  const [error, setError] = useState('');
  const [part, setPart] = useState(0);
  const [signal, setSignal] = useState(3);
  const [comparison, setComparison] = useState<V02Rule>();
  useEffect(() => {
    const controller = new AbortController();
    api<Data>(`/local-ai/v02/jobs/${jobId}/graphs/${modelId}/${index}`, {
      signal: controller.signal,
    })
      .then((series) => {
        if (!controller.signal.aborted) setData(series);
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setError(String(failure));
      });
    return () => controller.abort();
  }, [jobId, modelId, index]);

  const segment = data?.segments[part];
  const rule = compare && comparison && segment?.rules?.[comparison] ? comparison : primaryRule;
  const origin = timeline?.originUs;
  const projection = useMemo<PredictionData | undefined>(() => {
    if (!segment) return undefined;
    const prediction = rule === 'original' ? 'original' : 'adapted';
    const scores = segment.rules?.[rule];
    return {
      model: `${modelId}-${part}-${rule}`,
      thresholds: scores?.threshold ?? [],
      timeLabel: origin == null ? '비행제어기 부팅 후 경과(초)' : '로그 재생 시간(초)',
      rawScores: Boolean(scores?.raw_score),
      frames: segment.time_us.map((time_us, i) => ({
        time_us: time_us - (origin ?? 0),
        segment: part,
        observed: voltageOrVector(segment.observed[i]),
        predicted: voltageOrVector(segment.prediction[prediction]?.[i]),
        residual: voltageOrVector(segment.residual[prediction]?.[i]),
        raw_score: scores?.raw_score?.[i] ?? [],
        score: scores?.ema[i] ?? [],
        alarm: scores?.domains[i]?.map(Boolean) ?? [],
      })),
    };
  }, [modelId, segment, part, rule, origin]);

  return (
    <section className="buffered-chart" aria-label="실측·예측·이상 점수 그래프">
      <h3>실측·예측·이상 점수 그래프</h3>
      {data && segment && (
        <div className="buffered-chart-controls">
          {modelSelect}
          {data.segments.length > 1 && (
            <label className="form-label">
              분석 구간
              <select
                className="text-input"
                value={part}
                onChange={(event) => setPart(Number(event.target.value))}
              >
                {data.segments.map((_, i) => (
                  <option key={i} value={i}>
                    구간 {i + 1}
                  </option>
                ))}
              </select>
            </label>
          )}
          {data.task === 'inflight' && (
            <PredictionSignalSelect value={signal} onChange={setSignal} />
          )}
          {compare && segment.rules && (
            <label className="form-label">
              비교할 모델·판정 조합
              <select
                className="text-input"
                value={rule}
                onChange={(event) => setComparison(event.target.value as V02Rule)}
              >
                {V02_RULES.filter(([key]) => segment.rules?.[key]).map(([key, label]) => (
                  <option key={key} value={key}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      )}
      {error ? (
        <p role="alert">{error}</p>
      ) : projection ? (
        <PredictionPlot
          data={projection}
          target={data?.task === 'inflight' ? signal : 3}
          timeline={timeline}
        />
      ) : (
        <p role="status">그래프 불러오는 중…</p>
      )}
    </section>
  );
}
