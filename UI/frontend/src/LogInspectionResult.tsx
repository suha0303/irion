import type { Job } from './local-flight-types';

export default function LogInspectionResult({ result }: { result: NonNullable<Job['result']> }) {
  return (
    <>
      <p>
        {result.filename} ·{' '}
        {result.autopilot === 'px4'
          ? 'PX4'
          : result.autopilot === 'ardupilot'
            ? 'ArduPilot'
            : '제어기 미확인'}
      </p>
      <p>후보 필드가 빠진 메시지 {Object.keys(result.missing_candidate_fields ?? {}).length}종</p>
      {result.parser_warnings && (
        <p className="local-warning">파서 경고: {result.parser_warnings}</p>
      )}
      <details>
        <summary>입력 진단 상세</summary>
        <pre>
          {JSON.stringify(
            {
              missing: result.missing_candidate_fields,
              parameters: result.parameters,
              messages: result.messages,
            },
            null,
            2,
          )}
        </pre>
      </details>
    </>
  );
}
