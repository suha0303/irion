import { lazy, Suspense, useState, type ReactNode } from 'react';
import { V02_RULES, type V02Evaluation, type V02Result, type V02Task } from './local-flight-types';
import { modelLabel } from './V02ModelSelection';
import type { MonitorTimeline } from './FlightAnalysisViews';

const V02Chart = lazy(() => import('./V02Chart'));
const metrics = [
  ['original', '원본'],
  ['adapted', '파인튜닝 모델'],
  ['persistence', '직전값 유지'],
] as const;
const seconds = (us: number) => `${(us / 1e6).toFixed(1)}초`;

function Evaluation({
  row,
  index,
  jobId,
  modelId,
  task,
  monitor,
  modelSelect,
  timeline,
}: {
  row: V02Evaluation;
  index: number;
  jobId: string;
  modelId: string;
  task: V02Task;
  monitor: boolean;
  modelSelect?: ReactNode;
  timeline?: MonitorTimeline;
}) {
  const [graph, setGraph] = useState(false);
  const [compare, setCompare] = useState(false);
  const whole = task === 'whole-run';
  const paired = Boolean(row.decisions?.calibrated ?? row.whole_decisions?.calibrated);
  const primaryRule = paired ? 'calibrated' : 'original';
  const primaryDecision = row.decisions?.[primaryRule];
  const shownRules = V02_RULES.filter(
    ([key]) =>
      (compare || key === primaryRule) &&
      (whole ? row.whole_decisions?.[key] : row.decisions?.[key]),
  );
  const shownMetrics = metrics.filter(
    ([key]) => compare || key === (paired ? 'adapted' : 'original'),
  );
  const summary = (
    <>
      <h4>
        {row.filename}
        {row.date ? ` · ${row.date}` : ''}
        {row.health === 'abnormal'
          ? ' · 이상 확인'
          : row.health === 'confirmed_normal'
            ? ' · 정상 확인'
            : ''}
      </h4>
      <p>
        {row.segments.length}구간 · 공통 평가 {row.scored_frames.toLocaleString()}프레임
      </p>
      <div className="v02-result-table">
        <table>
          <caption>예측 RMSE</caption>
          <thead>
            <tr>
              <th scope="col">모델</th>
              {!whole && (
                <>
                  <th scope="col">롤 (rad/s)</th>
                  <th scope="col">피치 (rad/s)</th>
                  <th scope="col">요 (rad/s)</th>
                </>
              )}
              <th scope="col">팩 전압 (V)</th>
            </tr>
          </thead>
          <tbody>
            {shownMetrics.map(([key, name]) => {
              const values = row.rmse[key];
              return (
                values && (
                  <tr key={key}>
                    <th scope="row">{name}</th>
                    {values.map((value, i) => (
                      <td key={i}>{value.toFixed(4)}</td>
                    ))}
                  </tr>
                )
              );
            })}
          </tbody>
        </table>
        {whole && (
          <table>
            <caption>종료 후 구간별 판정 · 점수 / 임계값</caption>
            <thead>
              <tr>
                <th scope="col">구간</th>
                {shownRules.map(([key, name]) => (
                  <th scope="col" key={key}>
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {row.segments.map((segment, i) => (
                <tr key={i}>
                  <th scope="row">
                    {seconds(segment.start_us)}–{seconds(segment.end_us)}
                  </th>
                  {shownRules.map(([key]) => {
                    const decision = segment.whole_decisions?.[key];
                    return (
                      <td key={key}>
                        {decision?.score != null
                          ? `${decision.score.toFixed(3)} / ${decision.threshold.toFixed(3)} · ${decision.alarm ? '이상 의심' : '정상 범위'}`
                          : '판정 불가'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!whole && (
          <table>
            <caption>비행 중 경보</caption>
            <thead>
              <tr>
                <th scope="col">판정 조합</th>
                <th scope="col">횟수</th>
                <th scope="col">시간</th>
                <th scope="col">판정 시간 대비</th>
                <th scope="col">첫 경보</th>
              </tr>
            </thead>
            <tbody>
              {shownRules.map(([key, name]) => {
                const item = row.decisions?.[key];
                return (
                  item && (
                    <tr key={key}>
                      <th scope="row">{name}</th>
                      <td>{item.episodes}</td>
                      <td>{item.alarm_seconds.toFixed(1)}초</td>
                      <td>{item.alarm_percent.toFixed(2)}%</td>
                      <td>
                        {item.episode_starts_us.length
                          ? seconds(item.episode_starts_us[0])
                          : '없음'}
                      </td>
                    </tr>
                  )
                );
              })}
            </tbody>
          </table>
        )}
      </div>
      <button className="button small" aria-pressed={compare} onClick={() => setCompare(!compare)}>
        {compare ? '비교 접기' : '원본·기준 비교'}
      </button>
    </>
  );
  const plot = (
    <Suspense fallback={<p role="status">그래프 기능을 불러오는 중…</p>}>
      <V02Chart
        jobId={jobId}
        modelId={modelId}
        index={index}
        primaryRule={primaryRule}
        compare={compare}
        modelSelect={modelSelect}
        timeline={timeline}
      />
    </Suspense>
  );
  return monitor ? (
    <>
      {primaryDecision && (
        <p>
          경보 {primaryDecision.episodes}회 · 채점 시간의 {primaryDecision.alarm_percent.toFixed(2)}
          %
        </p>
      )}
      {plot}
      <details>
        <summary>분석 상세</summary>
        {summary}
      </details>
    </>
  ) : (
    <section className="v02-flight-result">
      {summary}
      <details onToggle={(e) => setGraph(e.currentTarget.open)}>
        <summary>실측·예측·잔차 그래프</summary>
        {graph && plot}
      </details>
    </section>
  );
}

export default function V02JobResult({
  jobId,
  result,
  monitor = false,
  timeline,
}: {
  jobId: string;
  result: V02Result;
  monitor?: boolean;
  timeline?: MonitorTimeline;
}) {
  const models = [...result.analyses, ...result.candidates];
  const [modelId, setModelId] = useState(models[0]?.model_id ?? '');
  const model = models.find((entry) => entry.model_id === modelId) ?? models[0];
  const candidate = result.candidates.find((entry) => entry.model_id === model?.model_id);
  const coverage = result.coverage;
  const modelSelect =
    model && models.length > 1 ? (
      <label className="form-label">
        모델
        <select
          className="text-input"
          value={model.model_id}
          onChange={(event) => setModelId(event.target.value)}
        >
          {models.map((item) => (
            <option value={item.model_id} key={item.model_id}>
              {modelLabel(item.model_id)}
            </option>
          ))}
        </select>
      </label>
    ) : null;
  return (
    <>
      {!monitor && result.candidate_job && (
        <p role="status">
          후보 {result.candidate_job.slice(0, 8)} · 파인튜닝 모델 + 전용 보정 기준
        </p>
      )}
      {!monitor && coverage && (
        <p className="analysis-coverage">
          분석 ${''}
          {seconds(coverage.analyzed_us)} · {coverage.retained_frames.toLocaleString()}
          프레임
          {coverage.context.extent_known
            ? ` / 원본 기록 ${seconds(coverage.context.end_us - coverage.context.start_us)}`
            : ' · 원본 전체 범위 미확정'}
        </p>
      )}
      {Object.entries(result.failures).map(([id, error]) => (
        <p key={id} role="alert">
          {modelLabel(id)} · {error}
        </p>
      ))}
      {model && (
        <>
          {models.length === 1 && <p>{modelLabel(model.model_id)}</p>}
          {!monitor && modelSelect}
          {candidate && (
            <>
              <p role="status">{candidate.profile_name} · 검증 대기 · 적용 안 됨</p>
              <p>
                학습 {candidate.counts.adaptation}건 · 보정 {candidate.counts.calibration}건 · 평가{' '}
                {candidate.counts.evaluation}건
              </p>
              {!candidate.anomaly_evidence && (
                <p className="notice warning">
                  확인된 이상 평가 로그를 받으면 이상 검출을 검증할 수 있습니다.
                </p>
              )}
            </>
          )}
          {model.evaluation.map((row, index) => (
            <Evaluation
              key={`${model.model_id}-${row.prepared_sha256}`}
              row={row}
              index={index}
              jobId={jobId}
              modelId={model.model_id}
              task={model.task}
              monitor={monitor}
              modelSelect={monitor && index === 0 ? modelSelect : undefined}
              timeline={timeline}
            />
          ))}
        </>
      )}
    </>
  );
}
