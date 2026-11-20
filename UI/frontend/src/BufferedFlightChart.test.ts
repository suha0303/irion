import { expect, test } from 'vitest';
import { chartOption } from './BufferedFlightChart';

const chart = {
  model: 'inflight-lstm-64x2',
  models: ['inflight-lstm-64x2'],
  live: false,
  hold_ms: 10_000,
  thresholds: [3.25, 4.5],
  frames: [
    {
      time_us: 2_000_000,
      segment: 0,
      observed: [0, 0, 0, 15],
      predicted: [0, 0, 0, 16],
      residual: [0, 0, 0, -1],
      raw_score: [1.25, 5.4],
      score: [1, 5],
      alarm: [false, true],
    },
  ],
};

test.each([0, 1, 2])(
  'angular signal %i uses the combined angular score and threshold',
  (signal) => {
    const option = chartOption(chart, signal);
    expect(option.yAxis).toHaveLength(3);
    expect(option.series.map((line) => line.name)).not.toContain('전압 EMA');
    const score = option.series.find((line) => line.name === '각속도 종합 EMA');
    expect(score).toMatchObject({
      data: [[2, 1]],
      markLine: { data: [{ yAxis: 3.25 }] },
      markArea: { data: [] },
    });
  },
);

test('pack voltage uses the voltage score, threshold and alarm region', () => {
  const option = chartOption(chart, 3);
  expect(option.yAxis).toHaveLength(3);
  expect(option.grid[2]).toMatchObject({ show: true, backgroundColor: '#f1f6fa' });
  expect(option.series[0].lineStyle.type).toBe('solid');
  expect(option.series[1].lineStyle.type).toBe('dashed');
  expect(option.series[0].lineStyle.color).not.toBe(option.series[1].lineStyle.color);
  expect(option.series.map((line) => line.name)).not.toContain('각속도 종합 EMA');
  const score = option.series.find((line) => line.name === '전압 EMA');
  expect(score).toMatchObject({
    data: [[2, 5]],
    markLine: { data: [{ yAxis: 4.5 }] },
    markArea: { data: [[{ xAxis: 2 }, { xAxis: 2.1 }]] },
  });
});
