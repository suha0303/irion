import { useState } from 'react';
import { api } from './api';
import type { Job } from './local-flight-types';

const variants = [
  'lstm_h64_l2',
  'lstm_h64_l4',
  'lstm_h128_l2',
  'lstm_h128_l4',
  'transformer_h64_l2',
  'transformer_h64_l4',
  'transformer_h128_l2',
  'transformer_h128_l4',
];
const reasons: Record<string, string> = {
  normal_prediction_regression: '정상 비행의 예측 오차가 증가했습니다.',
  normal_false_alarm_regression: '정상 비행의 오경보가 증가했습니다.',
  lost_anomaly_detection: '원본이 잡던 이상을 놓쳤습니다.',
  known_anomaly_still_missed: '알려진 이상을 원본과 새 모델 모두 놓쳤습니다.',
  anomaly_detection_delayed: '기존보다 이상을 늦게 잡았습니다.',
  anomaly_event_timing_or_target_missing: '이상 발생 구간 확인이 필요합니다.',
  anomaly_event_not_scored: '이상 구간에 분석 결과가 없습니다.',
  evaluation_health_unconfirmed: '평가 비행의 건강 상태를 확인해야 합니다.',
  normal_evaluation_missing: '별도 정상 평가 비행이 필요합니다.',
  abnormal_evaluation_missing: '별도 이상 평가 비행이 필요합니다.',
};

export default function LocalTuning({
  jobs,
  disabled,
  submit,
}: {
  jobs: Job[];
  disabled: boolean;
  submit: (path: string, body: unknown) => Promise<void>;
}) {
  const [models, setModels] = useState([
    'forecast_transformer_h64_l4',
    'run_health_transformer_h64_l2',
  ]);
  const [selection, setSelection] = useState<Record<string, string[]>>({
    adaptation: [],
    development: [],
    calibrations: [],
    evaluations: [],
  });
  const [device, setDevice] = useState('cuda');
  const [budget, setBudget] = useState(3600);
  const [gpu, setGpu] = useState('');
  const [normalPool, setNormalPool] = useState<string[]>([]);
  const [planId, setPlanId] = useState('');
  const [events, setEvents] = useState<
    Record<string, { start: string; end: string; target: string }>
  >({});
  const ready = jobs.filter((j) => j.action === 'prepare' && j.status === 'complete');
  const plans = jobs.filter((j) => j.action === 'normal_plan' && j.status === 'complete');
  const plan = plans.find((j) => j.id === planId);
  const groups = [
    ['adaptation', '학습할 정상 비행 1~3개', 'adaptation'],
    ['development', '학습 중 비교·정상 오차 기준용 비행 1~20개', 'development'],
    ['calibrations', '판정 기준을 만들 별도 정상 비행 2개 이상', 'calibration'],
    ['evaluations', '마지막 비교용 정상·이상 비행', 'evaluation'],
  ];
  function toggle(group: string, id: string) {
    setSelection((old) => ({
      ...old,
      [group]: old[group].includes(id) ? old[group].filter((v) => v !== id) : [...old[group], id],
    }));
  }
  function eventValue(id: string, key: 'start' | 'end' | 'target', value: string) {
    setEvents((old) => ({
      ...old,
      [id]: { ...(old[id] ?? { start: '', end: '', target: 'any' }), [key]: value },
    }));
  }
  function start() {
    const annotations: Record<string, { start_us: number; end_us: number; target: string }[]> = {};
    for (const id of selection.evaluations) {
      const hash = ready.find((j) => j.id === id)?.result?.source_sha256;
      const event = events[id];
      if (hash && event?.start !== undefined && event.start !== '' && event.end !== '') {
        annotations[hash] = [
          {
            start_us: Math.round(Number(event.start) * 1e6),
            end_us: Math.round(Number(event.end) * 1e6),
            target: event.target,
          },
        ];
      }
    }
    return submit('/local-ai/tune', {
      models,
      device,
      budget_seconds: budget,
      epochs: 30,
      adaptation_jobs: selection.adaptation,
      development_jobs: selection.development,
      calibrations_jobs: selection.calibrations,
      evaluations_jobs: selection.evaluations,
      events: annotations,
      normal_plan_job: planId || undefined,
    });
  }
  return (
    <section className="local-tuning">
      <h3>기체별 파인튜닝 · 두 모델 함께</h3>
      <p>
        비행 중 예측 1종과 비행 후 판정 1종을 순서대로 학습합니다. 정상 오경보와 기존 이상 검출이
        나빠지면 새 모델 선택을 막습니다.
      </p>
      {['forecast', 'run_health'].map((task, index) => (
        <label className="form-label" key={task}>
          {index === 0 ? '비행 중 예측 모델' : '비행 후 판정 모델'}
          <select
            className="text-input"
            value={models[index]}
            onChange={(e) =>
              setModels((old) => old.map((v, i) => (i === index ? e.target.value : v)))
            }
          >
            {variants.map((variant) => (
              <option key={variant} value={`${task}_${variant}`}>
                {variant.replaceAll('_', ' · ')}
              </option>
            ))}
          </select>
        </label>
      ))}
      <p className="local-note">
        기본 선택은 기존 X500 시뮬레이션 평가의 추천 모델입니다. 실기체 정확도를 뜻하지 않습니다.
        여기서 ‘원본’은 원본 가중치에 현재 기체의 로컬 판정 기준을 적용한 결과이며, 과거 최종 평가의
        입력·경보 처리 전체를 재현한 결과와 구분합니다.
      </p>
      <label className="form-label">
        학습 장치
        <select className="text-input" value={device} onChange={(e) => setDevice(e.target.value)}>
          <option value="cuda">이 PC의 NVIDIA GPU</option>
          <option value="cpu">CPU</option>
        </select>
      </label>
      <button
        className="button"
        disabled={disabled}
        onClick={() =>
          api<{ name?: string; total_memory_bytes?: number; reason?: string }>('/local-ai/gpu')
            .then((r) =>
              setGpu(
                r.reason ??
                  `${r.name} · ${((r.total_memory_bytes ?? 0) / 1024 ** 3).toFixed(1)} GB`,
              ),
            )
            .catch((e) => setGpu(String(e)))
        }
      >
        GPU 확인
      </button>
      {gpu && <p role="status">{gpu}</p>}
      <label className="form-label">
        학습·비교·저장 전체 시간 예산
        <select
          className="text-input"
          value={budget}
          onChange={(e) => setBudget(Number(e.target.value))}
        >
          <option value={1800}>최대 30분</option>
          <option value={3600}>최대 60분</option>
        </select>
      </label>
      <details>
        <summary>정상 비행 자동 배치 · 권장 10~20회</summary>
        <p>
          정상 확인 로그를 학습·개발·보정·평가에 나눕니다. 같은 비행을 여러 파일로 나눠도 한 회로
          셉니다. 20회는 3·3·8·6회로 배치합니다. 1~3회는 초기 적응용이며 독립 검증 자료가 더
          필요합니다.
        </p>
        {ready
          .filter((j) => j.result?.health === 'confirmed_normal')
          .map((j) => (
            <label className="local-segment" key={j.id}>
              <input
                type="checkbox"
                checked={normalPool.includes(j.id)}
                onChange={() =>
                  setNormalPool((old) =>
                    old.includes(j.id) ? old.filter((id) => id !== j.id) : [...old, j.id],
                  )
                }
              />
              {j.id.slice(0, 8)} · 정상 확인
            </label>
          ))}
        <button
          className="button"
          disabled={disabled || normalPool.length === 0}
          onClick={() => submit('/local-ai/normal-plan', { prepared_jobs: normalPool })}
        >
          선택한 {normalPool.length}개로 배치 만들기
        </button>
      </details>
      <label className="form-label">
        정상 비행 배치
        <select className="text-input" value={planId} onChange={(e) => setPlanId(e.target.value)}>
          <option value="">아래에서 직접 선택</option>
          {plans.map((j) => (
            <option key={j.id} value={j.id}>
              {j.id.slice(0, 8)} · 학습 {j.result?.counts?.adaptation} / 개발{' '}
              {j.result?.counts?.development} / 보정 {j.result?.counts?.calibration} / 평가{' '}
              {j.result?.counts?.evaluation}
            </option>
          ))}
        </select>
      </label>
      {plan && <p>{plan.result?.note} 별도 이상 비행은 아래에서 선택하세요.</p>}
      {groups
        .filter(([group]) => !plan || group === 'evaluations')
        .map(([group, title, role]) => (
          <details key={group}>
            <summary>{title}</summary>
            {ready.flatMap((j) =>
              j.result?.role === role &&
              (!plan || j.result?.health !== 'confirmed_normal') &&
              (role === 'evaluation' || j.result?.health === 'confirmed_normal')
                ? [
                    <div key={j.id}>
                      <label className="local-segment">
                        <input
                          type="checkbox"
                          checked={selection[group].includes(j.id)}
                          onChange={() => toggle(group, j.id)}
                        />
                        {j.id.slice(0, 8)} ·{' '}
                        {j.result.health === 'confirmed_normal'
                          ? '정상 확인'
                          : j.result.health === 'abnormal'
                            ? '이상 확인'
                            : '건강 미확정'}
                      </label>
                      {group === 'evaluations' &&
                        selection[group].includes(j.id) &&
                        j.result.health === 'abnormal' && (
                          <div>
                            <p>
                              이상 발생 구간: 기록 시각 기준 초. 모르면 비워 두세요. 검증 대기로
                              남습니다.
                            </p>
                            <label className="form-label">
                              이상 시작(초)
                              <input
                                className="text-input"
                                type="number"
                                min="0"
                                value={events[j.id]?.start ?? ''}
                                onChange={(e) => eventValue(j.id, 'start', e.target.value)}
                              />
                            </label>
                            <label className="form-label">
                              이상 종료(초)
                              <input
                                className="text-input"
                                type="number"
                                min="0"
                                value={events[j.id]?.end ?? ''}
                                onChange={(e) => eventValue(j.id, 'end', e.target.value)}
                              />
                            </label>
                            <label className="form-label">
                              확인할 신호
                              <select
                                className="text-input"
                                value={events[j.id]?.target ?? 'any'}
                                onChange={(e) => eventValue(j.id, 'target', e.target.value)}
                              >
                                <option value="any">전체 경보</option>
                                <option value="pack_voltage">배터리 전압 경보</option>
                                <option value="propulsion">추진계 경보</option>
                              </select>
                            </label>
                          </div>
                        )}
                    </div>,
                  ]
                : [],
            )}
          </details>
        ))}
      <p>
        정상 자료는 학습뿐 아니라 입력 분포 확인, 정상 오차 기준, 임계값 보정, 독립 평가에
        사용합니다. 점수와 경보 후처리는 기존 최종 평가와 같은 계산을 적용하며, 기준값은 해당 기체의
        정상 자료로 다시 만듭니다. 실기체 정확도는 별도 검증이 필요합니다.
      </p>
      <button
        className="button primary"
        disabled={
          disabled ||
          (plan
            ? !plan.result?.counts?.development
            : selection.adaptation.length < 1 ||
              selection.adaptation.length > 3 ||
              selection.development.length < 1 ||
              selection.development.length > 20)
        }
        onClick={start}
      >
        두 모델 파인튜닝 시작
      </button>
      {jobs.flatMap((job) =>
        job.action === 'tune'
          ? [
              <div key={job.id}>
                <h4>
                  파인튜닝 {job.id.slice(0, 8)} ·{' '}
                  {job.status === 'running'
                    ? '진행 중'
                    : job.status === 'complete'
                      ? '작업 종료'
                      : job.status}
                </h4>
                {job.progress && (
                  <p>
                    {job.progress.stage === 'review'
                      ? '정상·이상 비교 중'
                      : job.progress.stage === 'finished'
                        ? '작업 종료'
                        : '학습 중'}{' '}
                    · {((job.progress.elapsed_seconds ?? 0) / 60).toFixed(1)}분
                  </p>
                )}
                {job.error && <p role="alert">{job.error}</p>}
                {job.result?.note && <p>{job.result.note}</p>}
                {job.result?.elapsed_seconds !== undefined && (
                  <p>
                    전체 처리 {job.result.elapsed_seconds.toFixed(1)}초 ·{' '}
                    {job.result.within_budget ? '시간 예산 이내' : '시간 예산 초과 · 적용 보류'}
                  </p>
                )}
                {job.result?.tuned_models?.map((row, index) => (
                  <div key={row.model_id}>
                    <strong>
                      {row.model_id} ·{' '}
                      {row.review_status === 'passed'
                        ? '퇴행 검사 통과'
                        : row.review_status === 'rejected'
                          ? '적용 보류 · 성능 문제'
                          : '검증 대기'}
                    </strong>
                    {row.review?.promotion_check &&
                      [
                        ...row.review.promotion_check.reasons,
                        ...row.review.promotion_check.missing,
                      ].map((reason) => <p key={reason}>{reasons[reason] ?? reason}</p>)}
                    <ReviewTotals flights={row.review?.promotion_check.flights} />
                    <details>
                      <summary>비행별 예측 오차 상세</summary>
                      {row.review?.promotion_check.flights
                        ?.filter((f) => f.error_before && f.error_after)
                        .map((flight, flightIndex) => (
                          <table key={flightIndex}>
                            <caption>별도 정상 비행 {flightIndex + 1} · 예측 오차 비교</caption>
                            <thead>
                              <tr>
                                <th>신호</th>
                                <th>원본</th>
                                <th>파인튜닝</th>
                              </tr>
                            </thead>
                            <tbody>
                              {flight.error_before?.map((before, i) => (
                                <tr key={i}>
                                  <td>
                                    {row.model_id.startsWith('forecast')
                                      ? [
                                          'Roll (rad/s)',
                                          'Pitch (rad/s)',
                                          'Yaw (rad/s)',
                                          '팩 전압 (V)',
                                        ][i]
                                      : `구간 ${i + 1} · 셀 전압 복합 오차 (V)`}
                                  </td>
                                  <td>{before.toFixed(5)}</td>
                                  <td>{flight.error_after?.[i].toFixed(5)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        ))}
                    </details>
                    <button
                      className="button"
                      disabled={
                        disabled || row.review_status !== 'passed' || !job.result?.within_budget
                      }
                      onClick={() => submit(`/local-ai/tune/${job.id}/select/${index}`, {})}
                    >
                      검증된 모델과 판정 기준 적용
                    </button>
                  </div>
                ))}
              </div>,
            ]
          : [],
      )}
    </section>
  );
}

type ReviewFlights = NonNullable<
  NonNullable<NonNullable<Job['result']>['tuned_models']>[number]['review']
>['promotion_check']['flights'];

function ReviewTotals({ flights }: { flights: ReviewFlights }) {
  if (!flights) return null;
  const normal = flights.filter(
    (f) =>
      f.health === 'confirmed_normal' &&
      f.decisions &&
      ['original', 'adapted'].every((v) =>
        ['normal', 'suspect'].includes(f.decisions![v as 'original' | 'adapted'].status),
      ),
  );
  const events = flights.flatMap((f) => f.events ?? []);
  return (
    <table>
      <caption>별도 평가 결과 · 정상 오경보 / 알려진 이상 검출</caption>
      <thead>
        <tr>
          <th>지표</th>
          <th>원본</th>
          <th>파인튜닝</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>정상 오경보</th>
          {(['original', 'adapted'] as const).map((variant) => (
            <td key={variant}>
              {normal.length
                ? `${normal.filter((f) => f.decisions?.[variant].status === 'suspect').length} / ${normal.length}회`
                : '자료 없음'}
            </td>
          ))}
        </tr>
        <tr>
          <th>이상 검출</th>
          {(['original', 'adapted'] as const).map((variant) => {
            const covered = events.filter((e) => e[variant].covered);
            return (
              <td key={variant}>
                {covered.length
                  ? `${covered.filter((e) => e[variant].detected).length} / ${covered.length}건`
                  : '평가 가능한 이상 없음'}
              </td>
            );
          })}
        </tr>
      </tbody>
    </table>
  );
}
