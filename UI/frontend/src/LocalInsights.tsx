import { useState } from 'react';
import type { LocalState } from './useLocalAI';
import { localAlarms, localStatus, modelDecision } from './local-insights';
import LocalRunResult from './LocalRunResult';

const verdict = { normal: '정상 범위', suspect: '이상 의심', unavailable: '판정 불가' };
const seconds = (us: number) => `${(us / 1e6).toFixed(1)}초`;

export function LocalModelCards({
  state,
  error,
  showEvents,
}: {
  state: LocalState;
  error: string;
  showEvents: () => void;
}) {
  return (
    <div className="provider-grid" aria-label="선택한 로컬 모델">
      {(['forecast', 'post'] as const).map((task) => {
        const model = task === 'forecast' ? state.forecast_model : state.post_model;
        const hash = task === 'forecast' ? state.adapted_sha256 : state.post_adapted_sha256;
        const results = state.insights?.report?.analyses.filter((a) => a.model_id === model) ?? [];
        const finished = state.status === 'finished';
        const decision = finished
          ? modelDecision(results)
          : task === 'forecast'
            ? state.decision
            : undefined;
        return (
          <section
            className="panel provider-card"
            key={task}
            aria-label={`${task === 'forecast' ? '비행 중 예측' : '비행 종료 후 분석'} 모델`}
          >
            <h3>{task === 'forecast' ? '비행 중 예측' : '비행 종료 후 분석'}</h3>
            <p className="local-model-name">{model ?? '선택된 모델 없음'}</p>
            <div className={`provider-status ${decision?.status === 'suspect' ? 'suspect' : ''}`}>
              {error
                ? '상태 조회 실패'
                : decision
                  ? verdict[decision.status]
                  : task === 'post' && model && !finished
                    ? '비행 종료 후 실행'
                    : (localStatus[state.status] ?? state.status)}
            </div>
            <p>
              {hash
                ? `파인튜닝 ${hash.slice(0, 8)}`
                : model
                  ? '원본 모델'
                  : '기체와 모델을 연결하세요.'}
            </p>
            {decision?.threshold !== undefined && (
              <p>
                기준 {decision.threshold.toFixed(5)} · 초과 {decision.exceedances ?? 0} /{' '}
                {decision.observations ?? 0}
              </p>
            )}
            {finished && (
              <small>
                {state.insights?.report_source === 'saved_log_analysis'
                  ? '저장 로그 재분석'
                  : '종료 후 저장 결과'}{' '}
                · 전체 런 기준
              </small>
            )}
            {!finished && task === 'forecast' && <small>{state.message}</small>}
            {error && <p role="alert">{error}</p>}
            <button className="button small" onClick={showEvents}>
              판정·구간 상세 보기
            </button>
          </section>
        );
      })}
    </div>
  );
}

export default function LocalInsights({ state, error }: { state: LocalState; error: string }) {
  const [source, setSource] = useState('all');
  const [model, setModel] = useState('all');
  const [selected, setSelected] = useState('');
  const [page, setPage] = useState(0);
  const rows = localAlarms(state);
  const filtered = rows.filter(
    (r) => (source === 'all' || r.source === source) && (model === 'all' || r.model === model),
  );
  const pages = Math.max(1, Math.ceil(filtered.length / 20));
  const current = Math.min(page, pages - 1);
  const chosen = rows.find((r) => r.id === selected);
  const series = chosen?.analysis?.metrics.series?.find((s) => s.time_us.includes(chosen.time));
  const point = series && chosen ? series.time_us.indexOf(chosen.time) : -1;
  const report = state.insights?.report;
  return (
    <section className="local-insights" aria-label="로컬 AI 이상 구간 분석">
      {error && (
        <p role="alert" className="notice error">
          {error}
        </p>
      )}
      <div className="panel local-insight-panel">
        <h2>선택한 비행의 로컬 AI 결과</h2>
        <p>
          {state.insights?.history_available
            ? `실시간 기록 경보 ${state.insights.alarm_count}건`
            : '실시간 경보 이력 없음'}{' '}
          · 저장 로그 기준 초과 {rows.filter((r) => r.source === 'log').length}시점
        </p>
        {state.insights && !state.insights.history_available && state.status === 'finished' && (
          <p className="muted">
            이전 버전에서 실시간 경보 이력을 저장하지 않은 비행입니다. 아래 로그 재분석은 당시
            실시간 경보 이력과 구분합니다.
          </p>
        )}
        <div className="local-insight-filters">
          <label>
            결과 출처
            <select
              className="text-input"
              value={source}
              onChange={(e) => {
                setSource(e.target.value);
                setPage(0);
              }}
            >
              <option value="all">전체 출처</option>
              <option value="live">실시간 기록</option>
              <option value="log">저장 로그 재분석</option>
            </select>
          </label>
          <label>
            분석 모델
            <select
              className="text-input"
              value={model}
              onChange={(e) => {
                setModel(e.target.value);
                setPage(0);
              }}
            >
              <option value="all">전체 모델</option>
              {[
                ...new Set(
                  [state.forecast_model, state.post_model].filter((v): v is string => !!v),
                ),
              ].map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </label>
        </div>
        {state.insights && state.insights.alarm_count > state.insights.alarms.length && (
          <p>
            실시간 경보는 최근 {state.insights.alarms.length}건을 표시합니다. 전체 이력은 원본 경보
            파일에 보존됩니다.
          </p>
        )}
        <div className="local-run-table">
          <table>
            <thead>
              <tr>
                <th>수신 시작 후 시점</th>
                <th>모델</th>
                <th>출처</th>
                <th>점수 / 기준</th>
                <th>근거</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(current * 20, current * 20 + 20).map((r) => (
                <tr key={r.id}>
                  <td>{seconds(r.time)}</td>
                  <td>{r.model}</td>
                  <td>{r.source === 'live' ? '실시간 기록' : '로그 재분석'}</td>
                  <td>
                    {r.score.toFixed(3)} / {r.threshold.toFixed(3)}
                  </td>
                  <td>
                    <button className="button small" onClick={() => setSelected(r.id)}>
                      근거 보기 · {seconds(r.time)}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!filtered.length && (
          <p>
            {state.status === 'loading'
              ? '로컬 결과를 불러오는 중…'
              : '현재 선택한 출처·모델에서 표시할 기준 초과가 없습니다. 건강 인증을 뜻하지 않습니다.'}
          </p>
        )}
        {pages > 1 && (
          <div className="local-pagination">
            <button
              className="button small"
              disabled={current === 0}
              onClick={() => setPage(current - 1)}
            >
              이전
            </button>
            <span>
              {current + 1} / {pages}
            </span>
            <button
              className="button small"
              disabled={current === pages - 1}
              onClick={() => setPage(current + 1)}
            >
              다음
            </button>
          </div>
        )}
        {chosen && (
          <section className="local-alarm-detail" aria-label="선택한 로컬 경보 근거">
            <h3>
              {seconds(chosen.time)} · {chosen.model}
            </h3>
            <p>
              이상 의심 · 점수 {chosen.score.toFixed(5)} / 기준 {chosen.threshold.toFixed(5)}. 예측
              잔차이며 고장 원인 확정과 구분합니다. 경보의 지속·해제 조건에 따라 점수가 기준 아래로
              내려가도 경보가 잠시 유지될 수 있습니다.
            </p>
            {series && point >= 0 && (
              <table>
                <thead>
                  <tr>
                    <th>신호</th>
                    <th>예측</th>
                    <th>실측</th>
                    <th>잔차</th>
                  </tr>
                </thead>
                <tbody>
                  {chosen.analysis?.metrics.targets?.map((target, i) => (
                    <tr key={target}>
                      <td>
                        {target} ({chosen.analysis?.metrics.units?.[i]})
                      </td>
                      <td>{series.predicted[point][i].toFixed(4)}</td>
                      <td>{series.observed[point][i].toFixed(4)}</td>
                      <td>{series.residual[point][i].toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {chosen.targetScores && (
              <p>
                표준화 절대 잔차 · Roll / Pitch / Yaw / 전압:{' '}
                {chosen.targetScores.map((v) => v.toFixed(3)).join(' / ')}
              </p>
            )}
          </section>
        )}
      </div>
      {report && (
        <div className="panel local-insight-panel">
          <LocalRunResult summary={report.run_summary} analyses={report.analyses} />
        </div>
      )}
    </section>
  );
}
