import { describe, expect, it } from 'vitest';
import { chartSeries } from './LocalForecastChart';

describe('local forecast segment display', () => {
  it('keeps gap boundaries and physical time when plotting residuals', () => {
    const segments = [0, 10].map((time, segment) => ({
      segment,
      horizon_us: 1000000,
      origin_us: [time * 1e6],
      time_us: [(time + 1) * 1e6],
      observed: [[16]],
      predicted: [[15]],
      residual: [[1]],
    }));
    const lines = chartSeries(segments, 0, false);
    expect(lines).toHaveLength(4);
    expect(lines[0].data).toEqual([[1, 16]]);
    expect(lines[2].data).toEqual([[11, 16]]);
    expect(chartSeries(segments, 0, true)[1].data).toEqual([[11, 1]]);
  });
});
