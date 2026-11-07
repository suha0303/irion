import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import LocalRunResult, { RunSummaryView } from './LocalRunResult';
import type { RunSummary } from './local-flight-types';

const summary: RunSummary = {
  status: 'unavailable',
  partial: true,
  context: { start_us: 0, end_us: 30e6, extent_known: true, clock: 'receipt_monotonic', notes: [] },
  duration_us: 30e6,
  analyzed_us: 20e6,
  excluded_us: 10e6,
  coverage: 2 / 3,
  note: '미분석 구간의 건강 상태는 알 수 없습니다.',
  models: [],
  timeline: [
    { start_us: 0, end_us: 20e6, reason: 'analyzed' },
    {
      start_us: 20e6,
      end_us: 30e6,
      reason: 'missing_or_stale',
      missing_features: ['pack_voltage'],
    },
  ],
};
it('shows overall duration, coverage, exclusions and unresolved verdict', () => {
  const html = renderToStaticMarkup(<RunSummaryView summary={summary} />);
  for (const text of [
    '전체 런 요약',
    '30.0초',
    '20.0초',
    '10.0초',
    '66.7%',
    '판정 불가 구간 있음',
    '필수 입력 누락',
    'pack_voltage',
  ])
    expect(html).toContain(text);
  expect(html).not.toContain('정상 범위');
});
it('does not turn unknown extent or empty old results into full coverage', () => {
  expect(
    renderToStaticMarkup(<RunSummaryView summary={{ ...summary, coverage: null }} />),
  ).toContain('범위 미확정');
  expect(renderToStaticMarkup(<LocalRunResult analyses={[]} />)).toContain('이전 결과');
});
it('offers readable segment metrics for postflight analysis', () => {
  const html = renderToStaticMarkup(
    <LocalRunResult
      summary={summary}
      analyses={[
        {
          model_id: 'run_health_lstm_h64_l2',
          retained_segments: 1,
          metrics: {},
          segment_results: [
            {
              start_us: 0,
              end_us: 20e6,
              frames: 200,
              decision: { status: 'unavailable', reason: '기체별 보정 없음' },
              metrics: {
                samples: 100,
                pack_voltage_rmse: 1.25,
                short_cell_v: 0.02,
                long_cell_v: 0.1,
              },
            },
          ],
        },
      ]}
    />,
  );
  expect(html).toContain('구간별 상세 결과');
  expect(html).toContain('1.2500');
  expect(html).toContain('기체별 보정 없음');
});
