import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import { Diagnostics } from './Panels';
import type { Manifest } from './types';

test('diagnostics show input errors and omit display filters', () => {
  const inputError = { line: 1, reason: 'MAVLink 프레임 해석 실패', severity: 'excluded' };
  const run: Manifest = {
    run_id: 'live-test',
    name: '시험 비행',
    input_sha256: 'a'.repeat(64),
    source: 'UDP 수집 기록',
    storage: 'udp-wire/1',
    source_system: 1,
    source_component: 1,
    duration_ms: 1000,
    total_records: 4001,
    accepted_records: 3000,
    excluded_records: 1001,
    input_error_records: 1,
    input_issues: [inputError],
    issues: [{ line: 2, reason: '미지원 메시지: HIGHRES_IMU', severity: 'excluded' }],
    epochs: 1,
    boundaries: [],
    streams: [],
  };
  const html = renderToStaticMarkup(<Diagnostics run={run} frame={null} metrics={null} />);
  expect(html).toContain(inputError.reason);
  expect(html).not.toContain('HIGHRES_IMU');
  expect(html).not.toContain((1001).toLocaleString());
  expect(html).not.toContain('AI API 설정');
  expect(html).toContain('MAVLink UDP (.mavlink.bin)');
  expect(html).toContain('MAVLink UDP 원문 SHA-256');
  expect(html).not.toContain('원본 JSONL');
});
