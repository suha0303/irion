import { useMemo, useState } from 'react';
import { Chart } from './Charts';
import type { ForecastSeries } from './local-flight-types';

export function chartSeries(segments: ForecastSeries[], target: number, residual: boolean) {
  return segments.flatMap((s) =>
    (residual ? (['residual'] as const) : (['observed', 'predicted'] as const)).map((key) => ({
      name: key === 'observed' ? '실측' : key === 'predicted' ? '1초 앞 예측' : '실측 − 예측',
      type: 'line',
      showSymbol: false,
      connectNulls: false,
      smooth: false,
      lineStyle: {
        color: key === 'observed' ? '#208778' : key === 'predicted' ? '#526bd7' : '#b86c23',
        width: 2,
      },
      data: s.time_us.map((t, i) => [t / 1e6, s[key][i][target]]),
    })),
  );
}

export default function LocalForecastChart({
  segments,
  targets,
  units,
}: {
  segments: ForecastSeries[];
  targets: string[];
  units: string[];
}) {
  const [target, setTarget] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const options = useMemo(
    () =>
      !expanded
        ? []
        : [false, true].map((residual) => ({
            animation: false,
            grid: { left: 65, right: 15, top: 28, bottom: 35 },
            tooltip: { trigger: 'axis' },
            xAxis: {
              type: 'value',
              name: '부팅 후 초',
              nameLocation: 'middle',
              nameGap: 22,
              scale: true,
            },
            yAxis: { type: 'value', name: units[target], scale: true },
            series: chartSeries(segments, target, residual),
          })),
    [expanded, segments, target, units],
  );
  return (
    <details onToggle={(event) => setExpanded(event.currentTarget.open)}>
      <summary>예측·실측·잔차 시계열</summary>
      <label className="form-label">
        표시 신호
        <select
          className="text-input"
          value={target}
          onChange={(e) => setTarget(Number(e.target.value))}
        >
          {targets.map((name, i) => (
            <option key={name} value={i}>
              {name} ({units[i]})
            </option>
          ))}
        </select>
      </label>
      <p>
        청록: 실측 · 파랑: 1초 앞 예측 · 아래 주황: 실측 − 예측. 공백 사이 선은 연결하지 않습니다.
        위 RMSE는 0.1~1초 전체 예측 시점 기준입니다.
      </p>
      {expanded && (
        <>
          <Chart option={options[0]} height={210} label={`${targets[target]} 실측과 1초 앞 예측`} />
          <Chart option={options[1]} height={160} label={`${targets[target]} 예측 잔차`} />
        </>
      )}
    </details>
  );
}
