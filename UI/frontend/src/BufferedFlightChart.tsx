import { useCallback, useEffect, useMemo, useState } from 'react';
import { use } from 'echarts/core';
import { AxisPointerComponent, DataZoomComponent } from 'echarts/components';
import { api } from './api';
import { Chart } from './Charts';
import { numeric } from './types';
import type { MonitorTimeline } from './FlightAnalysisViews';

use([AxisPointerComponent, DataZoomComponent]);

type Frame = {
  time_us: number;
  segment: number;
  observed: (number | null)[];
  predicted: (number | null)[];
  residual: (number | null)[];
  raw_score: (number | null)[];
  score: (number | null)[];
  alarm: boolean[];
};
export type PredictionData = {
  model: string;
  thresholds: number[];
  frames: Frame[];
  timeLabel?: string;
  rawScores?: boolean;
};
type ChartData = PredictionData & {
  models: string[];
  candidate_models?: string[];
  live: boolean;
  hold_ms: number;
};

const signals = [
  { label: '롤 각속도', unit: 'rad/s', valueDigits: 3, residualDigits: 4, scoreDomain: 0 },
  { label: '피치 각속도', unit: 'rad/s', valueDigits: 3, residualDigits: 4, scoreDomain: 0 },
  { label: '요 각속도', unit: 'rad/s', valueDigits: 3, residualDigits: 4, scoreDomain: 0 },
  { label: '팩 전압', unit: 'V', valueDigits: 2, residualDigits: 3, scoreDomain: 1 },
];
const domainNames = ['각속도 종합', '전압'];
const observedColor = '#203b63';
const predictedColor = '#c24b12';
const residualColor = '#72519c';
const residualGrid = { top: 179, height: 83 };
const observedGrid = { top: 34, height: 104 };
const scoreDividerTop = residualGrid.top + residualGrid.height + 12;
const scoreGrid = { top: scoreDividerTop + 46, height: 95 };
const predictionGrids = [observedGrid, residualGrid];
const scoredGrids = [...predictionGrids, scoreGrid];
const chartHeight = scoreGrid.top + scoreGrid.height + 83;
const scoreDigits = 3;
const timeTicks = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 1 });
const secondsLabel = (value: number) => `${numeric(value, 1)}초`;

function modelName(model: string) {
  return model
    .replace(/^inflight-/, '')
    .replace('lstm-', 'LSTM ')
    .replace('transformer-', 'Transformer ')
    .replace('x', '×');
}

function points(frames: Frame[], value: (frame: Frame) => number | null) {
  const data: [number, number | null][] = [];
  let previous: Frame | undefined;
  for (const frame of frames) {
    if (
      previous &&
      (previous.segment !== frame.segment || frame.time_us - previous.time_us !== 1e5)
    )
      data.push([(previous.time_us + frame.time_us) / 2e6, null]);
    data.push([frame.time_us / 1e6, value(frame)]);
    previous = frame;
  }
  return data;
}

function alarmAreas(frames: Frame[], domain: number) {
  const intervals: [number, number][] = [];
  for (const frame of frames) {
    if (!frame.alarm[domain]) continue;
    const last = intervals.at(-1);
    if (last && last[1] + 1e5 === frame.time_us) last[1] = frame.time_us;
    else intervals.push([frame.time_us, frame.time_us]);
  }
  return intervals.map(([start, end]) => [{ xAxis: start / 1e6 }, { xAxis: (end + 1e5) / 1e6 }]);
}

export function chartOption(data: PredictionData, target: number) {
  const { frames } = data;
  const signal = signals[target];
  const domain = signal.scoreDomain;
  const scoreName = domainNames[domain];
  const threshold = data.thresholds[domain];
  const hasScores = threshold !== undefined;
  const axes = hasScores ? [0, 1, 2] : [0, 1];
  const lastAxis = axes.length - 1;
  const digits = [signal.valueDigits, signal.residualDigits, scoreDigits];
  const line = (
    name: string,
    axis: number,
    color: string,
    value: (frame: Frame) => number | null,
    stroke: { width?: number; type?: 'solid' | 'dashed' } = {},
  ) => ({
    name,
    type: 'line',
    data: points(frames, value),
    encode: { x: 0, y: 1, tooltip: [1] },
    xAxisIndex: axis,
    yAxisIndex: axis,
    showSymbol: false,
    connectNulls: false,
    lineStyle: { color, width: 1.5, type: 'solid', ...stroke },
    itemStyle: { color },
    tooltip: {
      valueFormatter: (value: unknown) => {
        if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
        const precision = digits[axis];
        const amount = Math.abs(value) < 0.5 * 10 ** -precision ? 0 : value;
        return `${numeric(amount, precision)}${axis < 2 ? ` ${signal.unit}` : ''}`;
      },
    },
  });
  const latest = frames.at(-1)!.time_us / 1e6;
  return {
    animation: false,
    textStyle: { fontSize: 11 },
    grid: [
      { left: 72, right: 35, ...observedGrid },
      { left: 72, right: 35, ...residualGrid },
      ...(hasScores
        ? [
            {
              left: 72,
              right: 35,
              ...scoreGrid,
              show: true,
              backgroundColor: '#f1f6fa',
              borderWidth: 0,
            },
          ]
        : []),
    ],
    tooltip: {
      trigger: 'axis',
    },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    dataZoom: [
      { type: 'inside', xAxisIndex: axes, filterMode: 'none' },
      {
        type: 'slider',
        xAxisIndex: axes,
        filterMode: 'none',
        height: 24,
        bottom: 12,
        labelFormatter: secondsLabel,
      },
    ],
    xAxis: axes.map((index) => ({
      type: 'value',
      gridIndex: index,
      min: frames[0].time_us / 1e6,
      max: Math.max(latest, frames[0].time_us / 1e6 + 0.1),
      name: index === lastAxis ? (data.timeLabel ?? '첫 UDP 수신 후 경과(초)') : undefined,
      nameLocation: 'middle',
      nameGap: 27,
      minInterval: 0.1,
      axisLabel: { show: index === lastAxis, hideOverlap: true, formatter: timeTicks.format },
      axisPointer: { label: { formatter: ({ value }: { value: number }) => secondsLabel(value) } },
      splitLine: { show: false },
    })),
    yAxis: [
      `${signal.label} (${signal.unit})`,
      `잔차 (${signal.unit})`,
      ...(hasScores ? [`${scoreName} 이상 점수`] : []),
    ].map((name, index) => ({
      type: 'value',
      gridIndex: index,
      name,
      scale: index < 2,
      minInterval: 10 ** -digits[index],
      axisLabel: { formatter: (value: number) => numeric(value, digits[index]) },
      max:
        index === 2
          ? ({ max }: { max: number }) => Math.max(max || 0, threshold * 1.08)
          : undefined,
      splitNumber: 3,
      splitLine: { lineStyle: { color: '#e5edf2' } },
    })),
    series: [
      line('실측', 0, observedColor, (f) => f.observed[target], { width: 2 }),
      {
        ...line('판정에 사용한 예측', 0, predictedColor, (f) => f.predicted[target], {
          width: 2,
          type: 'dashed',
        }),
        z: 4,
      },
      line('실측 − 예측', 1, residualColor, (f) => f.residual[target]),
      ...(hasScores && data.rawScores !== false
        ? [line(`${scoreName} 원점수`, 2, '#a9b4bf', (f) => f.raw_score[domain], { width: 1 })]
        : []),
      ...(hasScores
        ? [
            {
              ...line(`${scoreName} EMA`, 2, '#365778', (f) => f.score[domain]),
              markLine: {
                silent: true,
                symbol: 'none',
                lineStyle: { type: 'dashed', width: 1.4, color: '#9f5517' },
                label: {
                  formatter: `임계값 ${numeric(threshold, scoreDigits)}`,
                  position: 'insideEndTop',
                },
                data: [{ yAxis: threshold }],
              },
              markArea: {
                silent: true,
                itemStyle: { color: 'rgba(204, 55, 73, 0.18)' },
                data: alarmAreas(frames, domain),
              },
            },
          ]
        : []),
    ],
  };
}

export function PredictionSignalSelect({
  value,
  onChange,
}: {
  value: number;
  onChange: (signal: number) => void;
}) {
  return (
    <label className="form-label">
      비교 신호
      <select
        className="text-input"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {signals.map(({ label, unit }, index) => (
          <option key={label} value={index}>
            {label} ({unit})
          </option>
        ))}
      </select>
    </label>
  );
}

export function PredictionPlot({
  data,
  target,
  merge,
  timeline,
}: {
  data: PredictionData;
  target: number;
  merge?: boolean;
  timeline?: MonitorTimeline;
}) {
  const option = useMemo(
    () => (data.frames.length ? chartOption(data, target) : undefined),
    [data, target],
  );
  const seekMs = timeline?.seekMs;
  const seekSeconds = useCallback((value: number) => seekMs?.(value * 1000), [seekMs]);
  if (!option) return <p role="status">표시할 분석 프레임이 아직 없습니다.</p>;
  const signal = signals[target];
  const hasScores = data.thresholds[signal.scoreDomain] !== undefined;
  const firstAlarm = timeline && data.frames.find((frame) => frame.alarm[signal.scoreDomain]);
  const cursorSeconds = (timeline?.cursorMs ?? 0) / 1000;
  const insideSegment =
    timeline &&
    timeline.cursorMs * 1000 >= data.frames[0].time_us &&
    timeline.cursorMs * 1000 <= data.frames.at(-1)!.time_us;
  const label = `${signal.label} 실측, 예측, 잔차`;
  return (
    <>
      {timeline && (
        <div className="flight-chart-navigation">
          <span>
            재생 위치 {secondsLabel(cursorSeconds)}
            {insideSegment ? '' : ' · 표시 구간 밖'}
          </span>
          {firstAlarm && (
            <button className="button small" onClick={() => seekMs?.(firstAlarm.time_us / 1000)}>
              첫 경보로 이동
            </button>
          )}
        </div>
      )}
      <div className="buffered-chart-legend">
        <span className="line-key" style={{ color: observedColor }}>
          실측
        </span>
        <span className="line-key dashed" style={{ color: predictedColor }}>
          예측
        </span>
        <span className="line-key" style={{ color: residualColor }}>
          잔차
        </span>
        {hasScores && (
          <span>
            {data.rawScores !== false ? '회색 원점수 · ' : ''}남색 EMA · 점선 임계값 · 붉은 구간
            경보
          </span>
        )}
      </div>
      <div className="buffered-chart-plot">
        <Chart
          key={`${data.model}-${target}-${Boolean(timeline)}`}
          option={option}
          height={hasScores ? chartHeight : residualGrid.top + residualGrid.height + 83}
          label={hasScores ? `${label}와 ${domainNames[signal.scoreDomain]} 점수 및 임계값` : label}
          merge={merge}
          timeNavigation={
            timeline && {
              cursor: cursorSeconds,
              grids: hasScores ? scoredGrids : predictionGrids,
              seek: seekSeconds,
            }
          }
        />
        {hasScores && (
          <div
            aria-hidden="true"
            className="buffered-chart-divider"
            style={{ top: scoreDividerTop }}
          />
        )}
      </div>
      <p>
        {timeline
          ? '그래프를 누르면 해당 시각으로 이동합니다. 아래 슬라이더로 시간 범위를 확대합니다.'
          : '예측선은 각 실측 시점의 판정에 사용한 값입니다. 아래 슬라이더로 시간 범위를 확대합니다.'}
      </p>
    </>
  );
}

export default function BufferedFlightChart({
  runId,
  live,
  comparison,
}: {
  runId: string;
  live: boolean;
  comparison?: string;
}) {
  const [model, setModel] = useState('');
  const [target, setTarget] = useState(3);
  const [data, setData] = useState<ChartData>();
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams();
    if (model) params.set('model', model);
    if (comparison) params.set('comparison', comparison);
    let timer: ReturnType<typeof setTimeout>;
    const update = async () => {
      try {
        const result = await api<ChartData>(
          `/sessions/${runId}/local-ai/buffered-chart?${params}`,
          {
            signal: controller.signal,
          },
        );
        if (controller.signal.aborted) return;
        setData(result);
        setError('');
      } catch (failure) {
        if (controller.signal.aborted) return;
        setError(String(failure));
      }
      if (live) timer = setTimeout(update, 3000);
    };
    void update();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [model, runId, live, comparison]);
  const current = data && (!model || model === data.model) ? data : undefined;
  return (
    <section className="buffered-chart" aria-labelledby="buffered-chart-heading">
      <h3 id="buffered-chart-heading">실측·예측·이상 점수 그래프</h3>
      <div className="buffered-chart-controls">
        {data && data.models.length > 1 && (
          <label className="form-label">
            모델
            <select
              className="text-input"
              value={model || data.models[0]}
              onChange={(event) => setModel(event.target.value)}
            >
              {data.models.map((id) => (
                <option key={id} value={id}>
                  {modelName(id)}
                  {data.candidate_models?.includes(id) ? ' · 후보' : ' · 원본'}
                </option>
              ))}
            </select>
          </label>
        )}
        <PredictionSignalSelect value={target} onChange={setTarget} />
      </div>
      {error && (
        <p className="notice error" role="alert">
          {error}
        </p>
      )}
      {!error && !current && <p role="status">그래프 불러오는 중…</p>}
      {current && (
        <>
          <p>
            {modelName(current.model)} · {(current.hold_ms / 1000).toLocaleString()}초 버퍼
            {current.live ? ' · 최근 최대 3분 · 자동 갱신' : ' · 전체 비행'}
          </p>
          <PredictionPlot data={current} target={target} merge />
        </>
      )}
    </section>
  );
}
