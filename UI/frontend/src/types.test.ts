import { describe, expect, it } from 'vitest';
import { numeric, timeLabel } from './types';

describe('telemetry display boundaries', () => {
  it('distinguishes a stopped motor from a missing observation', () => {
    expect(numeric(0, 0)).toBe('0');
    expect(numeric(null, 0)).toBe('—');
    expect(numeric(NaN)).toBe('—');
    expect(numeric(Infinity)).toBe('—');
  });
  it('keeps negative measurements and fractions', () => {
    expect(numeric(-2.5)).toBe('-2.50');
    expect(numeric(15.26)).toBe('15.26');
  });
  it('formats run time rather than epoch-sized timestamps', () => {
    expect(timeLabel(47040, true)).toBe('00:47.0');
    expect(timeLabel(65000)).toBe('01:05');
    expect(timeLabel(0, true)).toBe('00:00.0');
    expect(timeLabel(59990, true)).toBe('01:00.0');
  });
});
