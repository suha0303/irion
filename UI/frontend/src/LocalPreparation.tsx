import { useEffect, useState } from 'react';
import type { Analysis, Decision, Job, Profile, SegmentMetrics } from './local-flight-types';
import { jobFilename } from './local-flight-types';
import LocalForecastChart from './LocalForecastChart';
import LocalTuning from './LocalTuning';
import FlightPreparation from './FlightPreparation';

export { FlightEventResult, TimingResult } from './FlightLogDetails';

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
  sourceLog: string;
  onSourceLog: (id: string) => void;
  preparedId: string;
  onPrepared: (id: string) => void;
  goToLogs: () => void;
  disabled: boolean;
  submit: (path: string, body: unknown) => Promise<void>;
};
export default function LocalPreparation({
  page,
  jobs,
  profiles,
  selectedModels,
  sourceLog: log,
  onSourceLog,
  preparedId,
  onPrepared,
  goToLogs,
  disabled,
  submit,
}: Props) {
  const [additionalPrepared, setAdditionalPrepared] = useState<string[]>([]);
  const [train, setTrain] = useState<string[]>([]);
  const [development, setDevelopment] = useState('');
  const [adaptedId, setAdaptedId] = useState('');
  const [epochs, setEpochs] = useState(5);
  const [useActive, setUseActive] = useState(false);
  const ready = (action: string) =>
    jobs.filter((job) => job.action === action && job.status === 'complete');
  const prepared = ready('prepare');
  const toggle = (items: string[], value: string) =>
    items.includes(value) ? items.filter((key) => key !== value) : [...items, value];
  const jobLabel = (job: Job) => jobFilename(job) ?? job.id.slice(0, 8);
  useEffect(() => {
    setAdditionalPrepared([]);
  }, [preparedId]);
  return (
    <>
      <FlightPreparation
        jobs={jobs}
        profiles={profiles}
        sourceLog={log}
        onSourceLog={onSourceLog}
        disabled={disabled}
        submit={submit}
        hidden={page !== undefined && page !== 'prepare'}
        legacyRoles
      />
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
            onChange={(e) => onPrepared(e.target.value)}
          >
            <option value="">선택</option>
            {prepared.map((j) => (
              <option key={j.id} value={j.id}>
                {jobLabel(j)} · {j.id.slice(0, 8)}
              </option>
            ))}
          </select>
        </label>
        {page === 'analysis' && !prepared.length && (
          <p>
            구간을 저장하면 여기에서 분석할 수 있습니다.{' '}
            <button className="button small" onClick={goToLogs}>
              로그 준비 열기
            </button>
          </p>
        )}
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
