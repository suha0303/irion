import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import LocalScoreChart, { scoreChartData } from './LocalScoreChart';
import type { Analysis, ForecastSeries } from './local-flight-types';

const segment = (time_us: number[], index: number): ForecastSeries => ({
  segment: index,
  time_us,
  horizon_us: 1e6,
  origin_us: [],
  observed: [],
  predicted: [],
  residual: [],
});
const analysis: Analysis = {
  model_id: 'forecast_lstm_h64_l2',
  retained_segments: 2,
  scoring: {
    scores: [5.28, 1, 5.33, 7.48, 4.97],
    time_us: [20.4e6, 21.4e6, 131.4e6, 132.4e6, 133.4e6],
    unit: 'residual',
  },
  decision: { status: 'suspect', threshold: 4.97, alarms: [false, true, true, true, true] },
  metrics: { series: [segment([20.4e6, 21.4e6], 0), segment([131.4e6, 132.4e6, 133.4e6], 1)] },
};

it('marks every scalar exceedance, including the early point, independently of persisted alarm state', () => {
  const data = scoreChartData(analysis)!;
  expect(data.points).toHaveLength(5);
  expect(data.exceeded).toEqual([
    [20.4, 5.28],
    [131.4, 5.33],
    [132.4, 7.48],
  ]);
  expect(data.threshold).toBe(4.97);
});

it('does not connect retained sections or fabricate a line for aggregate postflight scores', () => {
  expect(scoreChartData(analysis)?.lines.map((line) => line.map(([t]) => t))).toEqual([
    [20.4, 21.4],
    [131.4, 132.4, 133.4],
  ]);
  expect(scoreChartData({ ...analysis, metrics: {} })?.lines).toEqual([]);
});

it('keeps zero thresholds and rejects unavailable or nonfinite thresholds', () => {
  expect(
    scoreChartData({ ...analysis, decision: { status: 'suspect', threshold: 0 } })?.exceeded,
  ).toHaveLength(5);
  for (const decision of [
    { status: 'unavailable' as const, threshold: 0 },
    { status: 'normal' as const, threshold: NaN },
  ]) {
    const data = scoreChartData({ ...analysis, decision })!;
    expect(data.threshold).toBeUndefined();
    expect(data.exceeded).toEqual([]);
  }
});

it('rejects mismatched score/time arrays and breaks lines at invalid points', () => {
  expect(
    scoreChartData({ ...analysis, scoring: { scores: [1, 2], time_us: [1e6], unit: 'residual' } }),
  ).toBeUndefined();
  const broken = structuredClone(analysis);
  broken.scoring!.scores[3] = NaN;
  expect(scoreChartData(broken)?.lines.map((line) => line.length)).toEqual([2, 1, 1]);
});

it('identifies the model, saved-result origin, clock and threshold in the visible chart', () => {
  const html = renderToStaticMarkup(
    <LocalScoreChart
      analysis={analysis}
      sourceLabel="저장 로그 재분석"
      clockLabel="수신 시작 후 시간 (초)"
    />,
  );
  for (const label of [
    'forecast_lstm_h64_l2',
    '저장 로그 재분석',
    '수신 시작 후 시간 (초)',
    '4.97000',
    '3 / 5시점',
    '실제 고장 구간과 구분',
  ])
    expect(html).toContain(label);
});
