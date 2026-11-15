import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { localAlarms, modelDecision } from './local-insights';
import LocalInsights, { LocalModelCards } from './LocalInsights';
import type { LocalState } from './useLocalAI';

const state: LocalState = {
  status: 'finished',
  forecast_model: 'forecast_lstm_h64_l2',
  post_model: 'run_health_lstm_h64_l2',
  insights: {
    history_available: false,
    alarm_count: 0,
    alarms: [],
    report_source: 'saved_log_analysis',
    report: {
      analyses: [
        {
          model_id: 'forecast_lstm_h64_l2',
          retained_segments: 1,
          metrics: {},
          decision: { status: 'suspect', threshold: 5, observations: 3, exceedances: 1 },
          scoring: { scores: [1, 6, 2], time_us: [1e6, 2e6, 3e6], unit: 'residual' },
        },
        {
          model_id: 'run_health_lstm_h64_l2',
          retained_segments: 1,
          metrics: {},
          decision: { status: 'normal', threshold: 2, observations: 1, exceedances: 0 },
          scoring: { scores: [1], time_us: [3e6], unit: 'segment' },
        },
      ],
    },
  },
};

it('shows only the selected models with their actual, different verdicts', () => {
  const html = renderToStaticMarkup(
    <LocalModelCards state={state} error="" showEvents={() => {}} />,
  );
  expect(html).toContain('forecast_lstm_h64_l2');
  expect(html).toContain('run_health_lstm_h64_l2');
  expect(html).toContain('이상 의심');
  expect(html).toContain('정상 범위');
  expect(html).not.toContain('Transformer');
  expect(html).not.toContain('미연결');
});

it('does not turn log reanalysis into historical live alarms', () => {
  const alarms = localAlarms(state);
  expect(alarms).toHaveLength(1);
  expect(alarms[0]).toMatchObject({ time: 2e6, source: 'log', score: 6 });
  const html = renderToStaticMarkup(<LocalInsights state={state} error="" />);
  expect(html).toContain('당시 실시간 경보 이력과 구분');
  expect(html).toContain('2.0초');
  expect(html).toContain('6.000');
});

it('does not make a zero score or unavailable calibration into an alarm', () => {
  const unknown = structuredClone(state);
  unknown.insights!.report!.analyses[0].decision = { status: 'unavailable', threshold: 0 };
  expect(localAlarms(unknown)).toEqual([]);
});

it('aggregates all retained sections and keeps missing decisions visible', () => {
  const base = state.insights!.report!.analyses[0];
  const second = {
    ...base,
    decision: { status: 'normal' as const, threshold: 5, observations: 4, exceedances: 0 },
  };
  expect(modelDecision([base, second])).toMatchObject({
    status: 'suspect',
    observations: 7,
    exceedances: 1,
  });
  expect(modelDecision([second, { ...second, decision: undefined }])?.status).toBe('unavailable');
});

it('uses temporal alarm flags instead of reapplying a scalar threshold', () => {
  const temporal = structuredClone(state);
  temporal.insights!.report!.analyses[0].decision = {
    status: 'suspect',
    threshold: 5,
    alarms: [false, false, true],
  };
  expect(localAlarms(temporal)).toHaveLength(1);
  expect(localAlarms(temporal)[0]).toMatchObject({ time: 3e6, score: 2 });
});
