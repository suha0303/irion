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
