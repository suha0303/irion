import { useState } from 'react';
import type {
  Analysis,
  Decision,
  FlightEvents,
  Job,
  Profile,
  Timing,
  SegmentMetrics,
} from './local-flight-types';
import LocalForecastChart from './LocalForecastChart';
import LocalTuning from './LocalTuning';

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

export function DecisionResult({ decision }: { decision?: Decision }) {
  if (!decision) return <p>판정 불가 · 기체별 보정값 없음</p>;
  return (
    <div className={`local-decision ${decision.status}`}>
      <strong>
        {decision.status === 'normal'
          ? '정상 범위'
          : decision.status === 'suspect'
            ? '이상 의심'
            : '판정 불가'}
      </strong>
      <p>
        {decision.reason ??
          `보존된 평가 시점 ${decision.observations}개 중 ${decision.exceedances}개 기준 초과 · 기준 ${decision.threshold?.toFixed(5)}`}
      </p>
      {decision.calibration_id && (
        <small>보정 {decision.calibration_id.slice(0, 8)} · 실기체 성능 검증 대기</small>
      )}
    </div>
  );
}

export function MetricTable({ metrics }: { metrics: SegmentMetrics }) {
  return (
    <>
      {metrics.targets && (
        <table>
          <thead>
            <tr>
              <th>신호</th>
              <th>예측 RMSE</th>
              <th>직전값 유지 RMSE</th>
            </tr>
          </thead>
          <tbody>
            {metrics.targets.map((t, i) => (
              <tr key={t}>
                <td>
                  {t} ({metrics.units?.[i]})
                </td>
                <td>{metrics.rmse?.[i].toFixed(4) ?? '—'}</td>
                <td>{metrics.persistence_rmse?.[i].toFixed(4) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {metrics.pack_voltage_rmse !== undefined && (
        <p>
          초기 이력 이후 {metrics.samples}시점 · 팩 전압 기준 RMSE{' '}
          {metrics.pack_voltage_rmse.toFixed(4)} V · 셀 전압 단기 잔차{' '}
          {metrics.short_cell_v?.toFixed(4)} V · 장기 잔차 {metrics.long_cell_v?.toFixed(4)} V
        </p>
      )}
    </>
  );
}

export function AnalysisResult({ result }: { result: Analysis }) {
  return (
    <div>
      <strong>
        {result.model_id} · {result.adapted ? `적응 ${result.adapted_sha256?.slice(0, 8)}` : '원본'}
      </strong>
      <p>{result.retained_segments}개 구간 각각 분석 · 전체 기체 건강 인증과 구분</p>
      <DecisionResult decision={result.decision} />
      <p>
        처리 시간 {result.elapsed_seconds?.toFixed(2) ?? '—'}초 · 기록 사용률{' '}
        {result.coverage === undefined ? '—' : (result.coverage * 100).toFixed(1) + '%'} · 독립 정상
        관측의 기준 초과율:{' '}
        {result.normal_alarm_rate == null
          ? '산출 불가'
          : (result.normal_alarm_rate * 100).toFixed(2) + '%'}{' '}
        (비행 단위 오경보율과 구분)
      </p>
      {result.metrics.targets && (
        <table>
          <thead>
            <tr>
              <th>신호</th>
              <th>예측 RMSE</th>
              <th>직전값 유지 RMSE</th>
            </tr>
          </thead>
          <tbody>
            {result.metrics.targets.map((t, i) => (
              <tr key={t}>
                <td>
                  {t} ({result.metrics.units?.[i]})
                </td>
                <td>{result.metrics.rmse?.[i].toFixed(4)}</td>
                <td>{result.metrics.persistence_rmse?.[i].toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {!result.segment_results && result.metrics.segments && (
        <details>
          <summary>구간별 비행 종료 후 분석</summary>
          <pre>{JSON.stringify(result.metrics.segments, null, 2)}</pre>
        </details>
      )}
      {!result.segment_results &&
        result.metrics.series &&
        result.metrics.targets &&
        result.metrics.units && (
          <LocalForecastChart
            segments={result.metrics.series}
            targets={result.metrics.targets}
            units={result.metrics.units}
          />
        )}
      {result.segment_results?.map((segment, i) => (
        <details key={`${segment.start_us}-${i}`}>
          <summary>
            구간 {i + 1} · {(segment.start_us / 1e6).toFixed(1)}–{(segment.end_us / 1e6).toFixed(1)}
            초 · {segment.frames}시점 ·{' '}
            {segment.decision.status === 'suspect'
              ? '이상 의심'
              : segment.decision.status === 'normal'
                ? '정상 범위'
                : '판정 불가'}
          </summary>
          <DecisionResult decision={segment.decision} />
          <MetricTable metrics={segment.metrics} />
          {segment.series && segment.metrics.targets && segment.metrics.units && (
            <LocalForecastChart
              segments={segment.series}
              targets={segment.metrics.targets}
              units={segment.metrics.units}
            />
          )}
        </details>
      ))}
    </div>
  );
}

type Props = {
  page?: 'prepare' | 'analysis' | 'adapt' | 'hidden';
  jobs: Job[];
  profiles: Profile[];
  selectedModels: string[];
  disabled: boolean;
  submit: (path: string, body: unknown) => Promise<void>;
};
export default function LocalPreparation({
  page,
  jobs,
  profiles,
  selectedModels,
  disabled,
  submit,
}: Props) {
  const [log, setLog] = useState('');
  const [version, setVersion] = useState('');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [flightInterval, setFlightInterval] = useState('');
  const [fastAge, setFastAge] = useState(200);
  const [batteryAge, setBatteryAge] = useState(350);
  const [clock, setClock] = useState({
    stream: '',
    source1: '',
    common1: '',
    source2: '',
    common2: '',
    evidence: '',
  });
  const [corrections, setCorrections] = useState<
    Record<string, { anchors_us: number[][]; evidence: string }>
  >({});
  const [previewId, setPreviewId] = useState('');
  const [segments, setSegments] = useState<number[]>([]);
  const [evidence, setEvidence] = useState('');
  const [role, setRole] = useState('evaluation');
  const [health, setHealth] = useState('unknown');
  const [healthEvidence, setHealthEvidence] = useState('');
  const [preparedId, setPreparedId] = useState('');
  const [additionalPrepared, setAdditionalPrepared] = useState<string[]>([]);
  const [train, setTrain] = useState<string[]>([]);
  const [development, setDevelopment] = useState('');
  const [adaptedId, setAdaptedId] = useState('');
  const [epochs, setEpochs] = useState(5);
  const [useActive, setUseActive] = useState(false);
  const ready = (action: string) =>
    jobs.filter((j) => j.action === action && j.status === 'complete');
  const preview = jobs.find((j) => j.id === previewId)?.result?.timing;
  const profile = profiles.find((p) => p.version === version);
  const inspection = jobs.find((j) => j.id === log)?.result;
  const toggle = (items: string[], value: string) =>
    items.includes(value) ? items.filter((i) => i !== value) : [...items, value];
  const jobLabel = (j: Job) =>
    `${j.result?.filename ?? j.id.slice(0, 8)} · ${j.result?.role ?? j.action}`;
  const corrected = Object.keys(corrections).length;
  const anchorReady =
    Object.values(clock).every((v) => v.trim() !== '') &&
    [clock.source1, clock.common1, clock.source2, clock.common2].every((v) =>
      Number.isFinite(Number(v)),
    );
  return (
    <>
      <section className="panel" hidden={page !== undefined && page !== 'prepare'}>
        <h2>보존 구간과 시간축 조정</h2>
        <p>
          공백 전후를 분리하고 11초 이상 보존된 구간을 찾습니다. 검사 범위는 최대 2시간, 저장할 보존
          구간은 합계 1000초입니다. 시각은 로그의 부팅 후 초 단위입니다. 기록이 유지됐다는
          이유만으로 정상 비행으로 지정하지 않습니다.
        </p>
        <label className="form-label">
          검사한 로그
          <select
            className="text-input"
            value={log}
            onChange={(e) => {
              setLog(e.target.value);
              setFlightInterval('');
              setStart('');
              setEnd('');
            }}
          >
            <option value="">선택</option>
            {ready('inspect').map((j) => (
              <option key={j.id} value={j.id}>
                {jobLabel(j)}
              </option>
            ))}
          </select>
        </label>
        {inspection?.flight_events && (
          <>
            <FlightEventResult events={inspection.flight_events} />
            <label className="form-label">
              자동 비행 구간
              <select
                className="text-input"
                value={flightInterval}
                onChange={(e) => {
                  setFlightInterval(e.target.value);
                  const interval = inspection.flight_events?.intervals.find(
                    (v) => v.index === Number(e.target.value),
                  );
                  if (e.target.value !== '' && interval) {
                    setStart(String(interval.start_us / 1e6));
                    setEnd(String(interval.end_us / 1e6));
                    setEvidence(interval.evidence);
                  }
                }}
              >
                <option value="">수동 구간 입력</option>
                {inspection.flight_events.intervals.map((v) => (
                  <option key={v.index} value={v.index} disabled={!v.usable}>
                    비행 {v.flight} · {(v.start_us / 1e6).toFixed(1)}–{(v.end_us / 1e6).toFixed(1)}
                    초 · IMU {v.imu_instance ?? '프로파일 선택값'}
                  </option>
                ))}
              </select>
            </label>
            <small>
              자동 구간은 기록된 이착륙·센서 전환을 따릅니다. 정상 여부는 별도로 확인합니다.
            </small>
          </>
        )}
        {inspection?.timing && <TimingResult timing={inspection.timing} />}
        <label className="form-label">
          입력 매핑에 사용할 프로파일 버전
          <select
            className="text-input"
            value={version}
            onChange={(e) => setVersion(e.target.value)}
          >
            <option value="">선택</option>
            {profiles.map((p) => (
              <option key={p.version} value={p.version}>
                {p.profile.name} · {p.version.slice(0, 8)}
              </option>
            ))}
          </select>
        </label>
        <div className="local-form-grid">
          <label className="form-label">
            시작 (초)
            <input
              className="text-input"
              type="number"
              min="0"
              step="0.1"
              value={start}
              onChange={(e) => {
                setStart(e.target.value);
                setFlightInterval('');
              }}
            />
          </label>
          <label className="form-label">
            종료 (초)
            <input
              className="text-input"
              type="number"
              min="0"
              step="0.1"
              value={end}
              onChange={(e) => {
                setEnd(e.target.value);
                setFlightInterval('');
              }}
            />
          </label>
        </div>
        <details>
          <summary>관측 유지 시간과 시계 보정</summary>
          <p>
            큰 관측값 자체는 제거하지 않습니다. 센서 갱신이 느리다면 마지막 관측을 유지할 시간을
            조정하세요. 20–2000 ms 범위이며 설정은 결과에 기록됩니다.
          </p>
          <label className="form-label">
            운동·명령 관측 유지 (ms)
            <input
              className="text-input"
              type="number"
              min="20"
              max="2000"
              value={fastAge}
              onChange={(e) => setFastAge(Number(e.target.value))}
            />
          </label>
          <label className="form-label">
            배터리 관측 유지 (ms)
            <input
              className="text-input"
              type="number"
              min="20"
              max="2000"
              value={batteryAge}
              onChange={(e) => setBatteryAge(Number(e.target.value))}
            />
          </label>
          <p>
            시계가 어긋났다는 근거가 있을 때만 공통 시계와 일치하는 두 동기점을 등록하세요. 동기점
            사이만 보정하며, 기체의 실제 반응 지연을 자동으로 지우지 않습니다. 보정 결과는 사후
            분석용입니다.
          </p>
          <label className="form-label">
            센서 기록 이름
            <input
              className="text-input"
              placeholder="BAT:0 / battery_status:0"
              value={clock.stream}
              onChange={(e) => setClock((c) => ({ ...c, stream: e.target.value }))}
            />
          </label>
          <div className="local-form-grid">
            {(['source1', 'common1', 'source2', 'common2'] as const).map((key, i) => (
              <label className="form-label" key={key}>
                {
                  [
                    '동기점 1 원래 시각 (초)',
                    '동기점 1 공통 시각 (초)',
                    '동기점 2 원래 시각 (초)',
                    '동기점 2 공통 시각 (초)',
                  ][i]
                }
                <input
                  className="text-input"
                  type="number"
                  value={clock[key]}
                  onChange={(e) => setClock((c) => ({ ...c, [key]: e.target.value }))}
                />
              </label>
            ))}
          </div>
          <label className="form-label">
            동기화 근거
            <input
              className="text-input"
              value={clock.evidence}
              onChange={(e) => setClock((c) => ({ ...c, evidence: e.target.value }))}
            />
          </label>
          <button
            className="button"
            disabled={!anchorReady}
            onClick={() =>
              setCorrections((c) => ({
                ...c,
                [clock.stream]: {
                  anchors_us: [
                    [Number(clock.source1) * 1e6, Number(clock.common1) * 1e6],
                    [Number(clock.source2) * 1e6, Number(clock.common2) * 1e6],
                  ],
                  evidence: clock.evidence,
                },
              }))
            }
          >
            보정 추가
          </button>
          {Object.keys(corrections).map((k) => (
            <p key={k}>
              {k}{' '}
              <button
                className="button small"
                onClick={() =>
                  setCorrections((c) => {
                    const next = { ...c };
                    delete next[k];
                    return next;
                  })
                }
              >
                제거
              </button>
            </p>
          ))}
        </details>
        <button
          className="button primary"
          disabled={disabled || !log || !profile || start === '' || end === ''}
          onClick={() =>
            submit('/local-ai/timeline', {
              log_job: log,
              profile_id: profile?.profile.id,
              profile_version: version,
              ...(flightInterval !== ''
                ? { flight_interval: Number(flightInterval) }
                : {
                    start_us: Math.round(Number(start) * 1e6),
                    end_us: Math.round(Number(end) * 1e6),
                  }),
              fast_max_age_ms: fastAge,
              battery_max_age_ms: batteryAge,
              clock_corrections: corrections,
            })
          }
        >
          사용 가능 구간 찾기{corrected ? ' · 시계 보정 적용' : ''}
        </button>
        <label className="form-label">
          완료된 구간 검사
          <select
            className="text-input"
            value={previewId}
            onChange={(e) => {
              setPreviewId(e.target.value);
              setSegments([]);
              setEvidence(
                jobs.find((j) => j.id === e.target.value)?.result?.interval_evidence ?? '',
              );
            }}
          >
            <option value="">선택</option>
            {ready('timeline').map((j) => (
              <option key={j.id} value={j.id}>
                {jobLabel(j)}
              </option>
            ))}
          </select>
        </label>
        {preview && (
          <>
            <TimingResult timing={preview} />
            <p>
              이 검사에 사용한 프로파일·시간 설정으로 저장합니다. 설정을 바꿨다면 구간 검사를 새로
              실행하세요.
            </p>
            {preview.candidate_segments?.map((s, i) => (
              <label className="local-segment" key={i}>
                <input
                  type="checkbox"
                  disabled={!s.usable}
                  checked={segments.includes(i)}
                  onChange={() =>
                    setSegments((old) =>
                      old.includes(i)
                        ? old.filter((v) => v !== i)
                        : [...old, i].sort((a, b) => a - b),
                    )
                  }
                />
                {(s.start_us / 1e6).toFixed(1)}–{(s.end_us / 1e6).toFixed(1)}초 · {s.frames}시점 ·{' '}
                {s.usable ? '사용 가능' : '11초 미만'}
              </label>
            ))}
          </>
        )}
        <label className="form-label">
          비행 중 구간 선택 근거
          <input
            className="text-input"
            value={evidence}
            onChange={(e) => setEvidence(e.target.value)}
            placeholder="비행 기록·이착륙 시각 확인 등"
          />
        </label>
        <label className="form-label">
          비행 용도
          <select className="text-input" value={role} onChange={(e) => setRole(e.target.value)}>
            <option value="evaluation">독립 평가·예측</option>
            <option value="adaptation">적응 학습</option>
            <option value="development">학습 중 모델 선택</option>
            <option value="calibration">임계값 보정 전용</option>
          </select>
        </label>
        <label className="form-label">
          건강 상태
          <select className="text-input" value={health} onChange={(e) => setHealth(e.target.value)}>
            <option value="unknown">모름</option>
            <option value="confirmed_normal">정상 확인</option>
            <option value="abnormal">이상 확인</option>
          </select>
        </label>
        <label className="form-label">
          건강 상태 확인 근거
          <input
            className="text-input"
            value={healthEvidence}
            onChange={(e) => setHealthEvidence(e.target.value)}
          />
        </label>
        <button
          className="button primary"
          disabled={disabled || !previewId || !segments.length || !evidence.trim()}
          onClick={() =>
            submit('/local-ai/prepare', {
              timeline_job: previewId,
              selected_segments: segments,
              interval_evidence: evidence,
              role,
              health,
              health_evidence: healthEvidence,
            })
          }
        >
          선택 구간 저장
        </button>
      </section>
      <section
        className="panel"
        hidden={page !== undefined && page !== 'analysis' && page !== 'adapt'}
      >
        <h2>{page === 'adapt' ? '파인튜닝 및 독립 비교' : '준비된 비행 분석'}</h2>
        <p>
          같은 비행의 여러 구간은 하나의 비행으로 취급합니다. 공백을 넘어 입력·정답 창을 만들지
          않습니다.
        </p>
        <label className="form-label">
          준비된 비행
          <select
            className="text-input"
            value={preparedId}
            onChange={(e) => {
              setPreparedId(e.target.value);
              setAdditionalPrepared([]);
            }}
          >
            <option value="">선택</option>
            {ready('prepare').map((j) => (
              <option key={j.id} value={j.id}>
                {jobLabel(j)}
              </option>
            ))}
          </select>
        </label>
        <div hidden={page === 'adapt'}>
          {preparedId && (
            <details>
              <summary>같은 원본의 다른 준비 구간 함께 분석</summary>
              <p>
                전체 런에 포함할 구간을 추가하세요. 겹치는 구간은 중복 합산하지 않고 선택을 다시
                요청합니다. 선택하지 않은 부분은 미분석으로 표시합니다.
              </p>
              {ready('prepare')
                .filter(
                  (j) =>
                    j.id !== preparedId &&
                    j.result?.source_sha256 &&
                    j.result.source_sha256 ===
                      jobs.find((v) => v.id === preparedId)?.result?.source_sha256,
                )
                .map((j) => (
                  <label className="local-segment" key={j.id}>
                    <input
                      type="checkbox"
                      checked={additionalPrepared.includes(j.id)}
                      onChange={() => setAdditionalPrepared((v) => toggle(v, j.id))}
                    />
                    {jobLabel(j)}
                  </label>
                ))}
            </details>
          )}
          <label className="local-segment">
            <input
              type="checkbox"
              checked={useActive}
              onChange={(e) => setUseActive(e.target.checked)}
            />
            선택한 적응 모델 사용 (등록된 기체·원본 모델이 일치할 때 적용)
          </label>
          <button
            className="button primary"
            disabled={disabled || !preparedId || !selectedModels.length}
            onClick={() =>
              submit('/local-ai/analyze', {
                prepared_jobs: [preparedId, ...additionalPrepared],
                models: [...new Set(selectedModels)],
                use_active: useActive,
              })
            }
          >
            선택 모델 A/B로 분석
          </button>
        </div>
        <div hidden={page === 'analysis'}>
          <LocalTuning jobs={jobs} disabled={disabled} submit={submit} />
          <details>
            <summary>출력층만 학습 · 기존 작업 비교</summary>
            <h3>CPU 적응 학습</h3>
            <p>
              선택한 모델 A의 출력층을 적응합니다. 정상 확인 비행 1–3개를 선택하세요. 원본 모델과
              scaler는 보존합니다.
            </p>
            {ready('prepare')
              .filter(
                (j) => j.result?.role === 'adaptation' && j.result?.health === 'confirmed_normal',
              )
              .map((j) => (
                <label className="local-segment" key={j.id}>
                  <input
                    type="checkbox"
                    checked={train.includes(j.id)}
                    onChange={() => setTrain((t) => toggle(t, j.id))}
                  />
                  {jobLabel(j)}
                </label>
              ))}
            <label className="form-label">
              모델 선택용 별도 비행 (선택)
              <select
                className="text-input"
                value={development}
                onChange={(e) => setDevelopment(e.target.value)}
              >
                <option value="">없음 · 마지막 epoch 사용</option>
                {ready('prepare')
                  .filter(
                    (j) =>
                      j.result?.role === 'development' && j.result?.health === 'confirmed_normal',
                  )
                  .map((j) => (
                    <option key={j.id} value={j.id}>
                      {jobLabel(j)}
                    </option>
                  ))}
              </select>
            </label>
            <label className="form-label">
              학습 반복 횟수
              <input
                className="text-input"
                type="number"
                min="1"
                max="20"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
              />
            </label>
            <button
              className="button primary"
              disabled={disabled || train.length < 1 || train.length > 3 || !selectedModels[0]}
              onClick={() =>
                submit('/local-ai/adapt', {
                  model_id: selectedModels[0],
                  adaptation_jobs: train,
                  development_jobs: development ? [development] : [],
                  epochs,
                })
              }
            >
              새 적응 모델 만들기
            </button>
            <label className="form-label">
              비교할 적응 모델
              <select
                className="text-input"
                value={adaptedId}
                onChange={(e) => setAdaptedId(e.target.value)}
              >
                <option value="">선택</option>
                {ready('adapt').map((j) => (
                  <option key={j.id} value={j.id}>
                    {jobLabel(j)}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="button"
              disabled={disabled || !adaptedId || !preparedId}
              onClick={() =>
                submit('/local-ai/compare', { adapted_job: adaptedId, prepared_job: preparedId })
              }
            >
              선택한 독립 비행에서 원본과 비교
            </button>
            <p>
              비행 종료 후 모델도 출력층을 적응할 수 있습니다. 1–3회 학습의 성능은 독립 비행에서
              비교해야 하며, 기존 경보 임계값은 자동 적용하지 않습니다.
            </p>
          </details>
        </div>
      </section>
    </>
  );
}
