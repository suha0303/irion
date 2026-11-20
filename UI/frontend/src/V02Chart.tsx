import { useEffect, useMemo, useState } from 'react';
import { use } from 'echarts/core';
import { AxisPointerComponent, DataZoomComponent, LegendComponent } from 'echarts/components';
import { api } from './api';
import { Chart } from './Charts';
import type { V02Prediction, V02Rule } from './local-flight-types';

use([AxisPointerComponent, DataZoomComponent, LegendComponent]);

type Value = number | null;
type Series<T> = {
  time_us: number[];
  observed: T[];
  prediction: Partial<Record<V02Prediction, T[]>>;
  residual: Partial<Record<V02Prediction, T[]>>;
};
type FlightSeries = Series<Value[]> & {
  rules: Partial<
    Record<V02Rule, { threshold: number[]; ema: Value[][]; domains: (boolean | null)[][] }>
  >;
};
type Data =
  | { task: 'inflight'; segments: FlightSeries[] }
  | { task: 'whole-run'; segments: Series<Value>[] };

const signals = [
  ['롤 각속도', 'rad/s'],
  ['피치 각속도', 'rad/s'],
  ['요 각속도', 'rad/s'],
  ['팩 전압', 'V'],
] as const;
const modelLines = [
  ['original', '원본', '#c24b12'],
  ['adapted', '후보', '#107b72'],
  ['persistence', '유지 기준', '#7958a1'],
] as const;
export const ruleNames = [
  ['original', '원본 · 기존 기준'],
  ['x500', '후보 · 기존 기준'],
  ['calibrated', '후보 · 보정 기준'],
] as const;

export default function V02Chart({
  jobId,
  modelId,
  index,
}: {
  jobId: string;
  modelId: string;
  index: number;
}) {
  const [data, setData] = useState<Data>();
  const [error, setError] = useState('');
  const [part, setPart] = useState(0);
  const [signal, setSignal] = useState(3);
  const [rule, setRule] = useState<V02Rule>('original');
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

  const option = useMemo(() => {
    const segment = data?.segments[part];
    if (!data || !segment) return undefined;
    const asValue = (value: Value | Value[]) => (Array.isArray(value) ? value[signal] : value);
    const flightRule = data.task === 'inflight' ? data.segments[part].rules[rule] : undefined;
    const times = segment.time_us.map((t) => t / 1e6);
    const domain = signal === 3 ? 1 : 0;
    const count = flightRule ? 3 : 2;
    const axes = Array.from({ length: count }, (_, i) => i);
    const line = (
      name: string,
      color: string,
      i: number,
      values: readonly (Value | Value[])[],
      dashed = false,
    ) => ({
      name,
      type: 'line',
      showSymbol: false,
      connectNulls: false,
      xAxisIndex: i,
      yAxisIndex: i,
      lineStyle: { color, width: 1.7, type: dashed ? 'dashed' : 'solid' },
      itemStyle: { color },
      data: times.map((t, j) => [t, asValue(values[j])]),
    });
    const lines = [
      line('실측', '#203b63', 0, segment.observed),
      ...modelLines.flatMap(([key, label, color]) => {
        const forecast = segment.prediction[key];
        const residual = segment.residual[key];
        return forecast && residual
          ? [
              line(label, color, 0, forecast, key !== 'adapted'),
              line(label, color, 1, residual, key !== 'adapted'),
            ]
          : [];
      }),
    ];
    const threshold = flightRule?.threshold[domain];
    const score =
      flightRule && threshold !== undefined
        ? [
            {
              ...line(
                '이상 점수 (EMA)',
                '#365778',
                2,
                flightRule.ema.map((row) => row[domain]),
              ),
              markLine: {
                symbol: 'none',
                label: { formatter: `임계값 ${threshold.toPrecision(4)}` },
                data: [{ yAxis: threshold }],
              },
              markArea: {
                itemStyle: { color: 'rgba(210,60,46,0.18)' },
                data: alarmIntervals(
                  times,
                  flightRule.domains.map((row) => row[domain]),
                ),
              },
            },
          ]
        : [];
    const [name, unit] = signals[data.task === 'inflight' ? signal : 3];
    return {
      animation: false,
      legend: { top: 0 },
      tooltip: {
        trigger: 'axis',
        valueFormatter: (v: unknown) => (typeof v === 'number' ? v.toFixed(3) : '—'),
      },
      axisPointer: { link: [{ xAxisIndex: axes }] },
      dataZoom: [
        { type: 'inside', xAxisIndex: axes },
        { type: 'slider', xAxisIndex: axes, bottom: 3, height: 20 },
      ],
      grid: axes.map((i) => ({ left: 78, right: 32, top: 62 + i * 136, height: 106 })),
      xAxis: axes.map((i) => ({
        type: 'value',
        gridIndex: i,
        min: times[0],
        max: times.at(-1),
        axisLabel: { show: i === count - 1, formatter: '{value}초' },
      })),
      yAxis: [
        { name: `${name} (${unit})`, scale: true },
        { name: `실측 − 예측 (${unit})`, scale: true },
        ...(threshold !== undefined
          ? [
              {
                name: domain === 1 ? '전압 이상 점수' : '각속도 이상 점수',
                max: ({ max }: { max: number }) => Math.max(max || 0, threshold * 1.08, 1e-6),
              },
            ]
          : []),
      ].map((axis, i) => ({ type: 'value', gridIndex: i, ...axis })),
      series: [...lines, ...score],
    };
  }, [data, part, signal, rule]);

  if (error) return <p role="alert">{error}</p>;
  if (!data || !option) return <p role="status">그래프를 불러오는 중…</p>;
  const flight = data.task === 'inflight' ? data.segments[part] : undefined;
  const label = data.task === 'inflight' ? signals[signal][0] : '팩 전압';
  return (
    <section aria-label="평가 비교 그래프">
      <div className="buffered-chart-controls">
        <label className="form-label">
          분석 구간
          <select
            className="text-input"
            value={part}
            onChange={(e) => setPart(Number(e.target.value))}
          >
            {data.segments.map((_, i) => (
              <option key={i} value={i}>
                구간 {i + 1}
              </option>
            ))}
          </select>
        </label>
        {flight && (
          <>
            <label className="form-label">
              비교 신호
              <select
                className="text-input"
                value={signal}
                onChange={(e) => setSignal(Number(e.target.value))}
              >
                {signals.map(([name, unit], i) => (
                  <option value={i} key={name}>
                    {name} ({unit})
                  </option>
                ))}
              </select>
            </label>
            <label className="form-label">
              이상 점수 기준
              <select
                className="text-input"
                value={rule}
                onChange={(e) => setRule(e.target.value as V02Rule)}
              >
                {ruleNames
                  .filter(([key]) => flight.rules[key])
                  .map(([key, name]) => (
                    <option key={key} value={key}>
                      {name}
                    </option>
                  ))}
              </select>
            </label>
          </>
        )}
      </div>
      <Chart
        option={option}
        height={flight ? 480 : 344}
        label={`${label} 실측·예측·잔차 비교 그래프`}
      />
    </section>
  );
}

function alarmIntervals(times: number[], alarm: (boolean | null)[]) {
  const ranges: [number, number][] = [];
  for (let i = 0; i < times.length; i += 1) {
    if (!alarm[i]) continue;
    const last = ranges.at(-1);
    if (i > 0 && alarm[i - 1] && last) last[1] = times[i] + 0.1;
    else ranges.push([times[i], times[i] + 0.1]);
  }
  return ranges.map(([start, end]) => [{ xAxis: start }, { xAxis: end }]);
}
