import type { FlightEvents, Timing } from './local-flight-types';

export function FlightEventResult({ events }: { events: FlightEvents }) {
  return (
    <div>
      <h3>비행·제어기 이벤트</h3>
      <p>이륙·착륙과 센서 전환으로 나눈 구간 {events.intervals.length}개 · 건강 상태 미확정</p>
      {events.error_counts.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>기록</th>
              <th>코드</th>
              <th>횟수</th>
            </tr>
          </thead>
          <tbody>
            {events.error_counts.map((r) => (
              <tr key={`${r.subsystem}-${r.code}`}>
                <td>{r.label}</td>
                <td>
                  {r.code}
                  {[5, 6, 8].includes(r.subsystem) && events.supported
                    ? r.code === 1
                      ? ' · 발생'
                      : r.code === 0
                        ? ' · 해제'
                        : ''
                    : ''}
                </td>
                <td>{r.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {events.notes.map((note, i) => (
        <p key={i}>{note}</p>
      ))}
    </div>
  );
}

export function TimingResult({ timing }: { timing: Timing }) {
  return (
    <>
      {timing.coverage !== undefined && (
        <p>
          보존 구간 사용률 {(timing.coverage * 100).toFixed(1)}% · {timing.retained_frames}개 시점
        </p>
      )}
      {timing.missing_or_stale_frames !== undefined && (
        <p>결측·오래된 관측 제외: {timing.missing_or_stale_frames}개 시점</p>
      )}
      {timing.filled_frames !== undefined && (
        <p>짧은 공백 보충: {timing.filled_frames}개 시점 · 보충값은 실측 오차 계산에서 제외</p>
      )}
      {!!timing.operating_excluded_frames && (
        <p>지상·이착륙·비행 상태 미확인 제외: {timing.operating_excluded_frames}개 시점</p>
      )}
      {timing.recorded_intersections && (
        <>
          <p>공통 기록 구간입니다. 비행 중 여부·정상 여부·모델 입력 적격성은 별도로 확인합니다.</p>
          {timing.recorded_intersections.map((s, i) => (
            <p key={i}>
              {(s.start_us / 1e6).toFixed(2)}–{(s.end_us / 1e6).toFixed(2)}초 ·{' '}
              {s.duration_s.toFixed(1)}초 보존
            </p>
          ))}
        </>
      )}
      {timing.streams && (
        <details>
          <summary>센서별 기록 공백·역행</summary>
          {Object.entries(timing.streams).map(([k, s]) => (
            <p key={k}>
              {k}: 최대 공백 {(s.max_gap_us / 1e6).toFixed(3)}초 · 역행 {s.reversals} · 중복{' '}
              {s.duplicates}
            </p>
          ))}
        </details>
      )}
      {timing.clock_corrections &&
        Object.entries(timing.clock_corrections).map(([k, c]) => (
          <p key={k}>
            {k}: 보정 기울기 편차 {c.drift_ppm.toFixed(1)} ppm · 시각 이동{' '}
            {(c.offset_us / 1000).toFixed(1)} ms · 동기점 잔차 {c.residual_max_us.toFixed(1)} μs
          </p>
        ))}
    </>
  );
}
