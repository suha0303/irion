import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import LocalTuning from './LocalTuning';
import type { Job } from './local-flight-types';

it.each([
  ['rejected', true, false],
  ['pending', true, false],
  ['passed', false, false],
  ['passed', true, true],
])('gates model application on review=%s and budget=%s', (status, withinBudget, enabled) => {
  const job = {
    id: 'test',
    action: 'tune',
    status: 'complete',
    result: {
      within_budget: withinBudget,
      tuned_models: [{ model_id: 'forecast_transformer_h64_l4', review_status: status }],
    },
  } as Job;
  const html = renderToStaticMarkup(
    <LocalTuning jobs={[job]} disabled={false} submit={async () => {}} />,
  );
  const button = html.match(/<button[^>]*>검증된 모델과 판정 기준 적용<\/button>/)?.[0];
  expect(button).toBeDefined();
  expect(button?.includes('disabled')).toBe(!enabled);
});

it('summarizes scored evaluations without counting unavailable or irrelevant events', () => {
  const jobs: Job[] = [
    {
      id: 'completed-comparison',
      action: 'tune',
      status: 'complete',
      result: {
        within_budget: true,
        tuned_models: [
          {
            model_id: 'run_health_transformer_h64_l2',
            review_status: 'rejected',
            review: {
              promotion_check: {
                status: 'rejected',
                missing: [],
                reasons: ['known_anomaly_still_missed'],
                flights: [
                  {
                    health: 'confirmed_normal',
                    decisions: { original: { status: 'normal' }, adapted: { status: 'normal' } },
                  },
                  {
                    health: 'confirmed_normal',
                    decisions: {
                      original: { status: 'unavailable' },
                      adapted: { status: 'unavailable' },
                    },
                  },
                  {
                    health: 'abnormal',
                    events: [
                      {
                        original: { covered: true, detected: false },
                        adapted: { covered: true, detected: false },
                      },
                      {
                        original: { covered: false, detected: false },
                        adapted: { covered: false, detected: false },
                      },
                    ],
                  },
                ],
              },
            },
          },
        ],
      },
    },
  ];
  const html = renderToStaticMarkup(
    <LocalTuning jobs={jobs} disabled={false} submit={async () => {}} />,
  );
  expect(html.match(/0 \/ 1회/g)).toHaveLength(2);
  expect(html.match(/0 \/ 1건/g)).toHaveLength(2);
  expect(html).toContain('알려진 이상을 원본과 새 모델 모두 놓쳤습니다.');
  expect(html).toContain('disabled="">검증된 모델과 판정 기준 적용');
});
