import { useEffect, useMemo, useState } from 'react';
import { use } from 'echarts/core';
import { AxisPointerComponent, DataZoomComponent } from 'echarts/components';
import { api } from './api';
import { Chart } from './Charts';
import { numeric } from './types';

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
type ChartData = {
  model: string;
  models: string[];
  live: boolean;
  hold_ms: number;
  thresholds: number[];
  frames: Frame[];
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
const scoreGridTop = 303;
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

export function chartOption(data: ChartData, target: number) {
  const { frames } = data;
  const axes = [0, 1, 2];
  const signal = signals[target];
  const domain = signal.scoreDomain;
  const scoreName = domainNames[domain];
  const threshold = data.thresholds[domain];
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
      { left: 72, right: 35, top: 34, height: 104 },
      { left: 72, right: 35, top: 179, height: 83 },
      {
        left: 72,
        right: 35,
        top: scoreGridTop,
        height: 95,
        show: true,
        backgroundColor: '#f1f6fa',
        borderWidth: 0,
      },
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
      name: index === 2 ? '첫 UDP 수신 후 경과(초)' : undefined,
      nameLocation: 'middle',
      nameGap: 27,
      minInterval: 0.1,
      axisLabel: { show: index === 2, hideOverlap: true, formatter: timeTicks.format },
      axisPointer: { label: { formatter: ({ value }: { value: number }) => secondsLabel(value) } },
      splitLine: { show: false },
    })),
    yAxis: [
      `${signal.label} (${signal.unit})`,
      `잔차 (${signal.unit})`,
      `${scoreName} 이상 점수`,
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
      line(`${scoreName} 원점수`, 2, '#a9b4bf', (f) => f.raw_score[domain], { width: 1 }),
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
    ],
  };
}

export default function BufferedFlightChart({ runId, live }: { runId: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  const [model, setModel] = useState('');
  const [target, setTarget] = useState(3);
  const [data, setData] = useState<ChartData>();
  const [error, setError] = useState('');
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    const query = model ? `?model=${encodeURIComponent(model)}` : '';
    let timer: ReturnType<typeof setTimeout>;
    const update = async () => {
      try {
        const result = await api<ChartData>(`/sessions/${runId}/local-ai/buffered-chart${query}`, {
          signal: controller.signal,
        });
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
  }, [open, model, runId, live]);
  const current = data && (!model || model === data.model) ? data : undefined;
  const option = useMemo(
    () => (open && current?.frames.length ? chartOption(current, target) : undefined),
    [open, current, target],
  );
  return (
    <details className="buffered-chart" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>실측·예측·이상 점수 그래프</summary>
      {open && (
        <div className="buffered-chart-body">
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
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="form-label">
              비교 신호
              <select
                className="text-input"
                value={target}
                onChange={(event) => setTarget(Number(event.target.value))}
              >
                {signals.map(({ label, unit }, index) => (
                  <option key={label} value={index}>
                    {label} ({unit})
                  </option>
                ))}
              </select>
            </label>
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
              {option ? (
                <>
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
                    <span>회색 원점수 · 남색 EMA · 점선 임계값 · 붉은 구간 경보</span>
                  </div>
                  <div className="buffered-chart-plot">
                    <Chart
                      key={`${current.model}-${target}`}
                      option={option}
                      height={469}
                      label={`${signals[target].label} 실측, 예측, 잔차와 ${domainNames[signals[target].scoreDomain]} 점수 및 임계값`}
                      merge
                    />
                    <div
                      aria-hidden="true"
                      className="buffered-chart-divider"
                      style={{ top: scoreGridTop - 23 }}
                    />
                  </div>
                  <p>
                    예측선은 각 실측 시점의 판정에 사용한 예측입니다. 아래 슬라이더로 시간 범위를
                    확대할 수 있습니다.
                  </p>
                </>
              ) : (
                <p role="status">표시할 분석 프레임이 아직 없습니다.</p>
              )}
            </>
          )}
        </div>
      )}
    </details>
  );
}
