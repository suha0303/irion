import { useState } from 'react';
import type { Analysis, RunSummary } from './local-flight-types';
import { AnalysisResult, MetricTable } from './LocalPreparation';

const reasons: Record<string, string> = {
  analyzed: '분석 입력 사용',
  outside_selection: '선택 범위 밖 · 미분석',
  not_selected: '사용 가능하지만 선택하지 않음',
  too_short: '연속 11초 미만',
  missing_or_stale: '필수 입력 누락·오래됨·무효 값',
  unclassified: '이전 자료 · 위치별 제외 사유 미기록',
};
const statuses = {
  normal: '분석 구간 정상 범위',
  suspect: '이상 의심 구간 있음',
  incomplete: '일부 구간만 판정됨',
  unavailable: '판정 불가 구간 있음',
};
const duration = (us: number) => `${(us / 1e6).toFixed(1)}초`;

export function RunSummaryView({ summary }: { summary: RunSummary }) {
  const [onlyExcluded, setOnlyExcluded] = useState(false);
  const [page, setPage] = useState(0);
  const rows = onlyExcluded
    ? summary.timeline.filter((r) => r.reason !== 'analyzed')
    : summary.timeline;
  const maxPage = Math.max(0, Math.ceil(rows.length / 12) - 1);
  const current = Math.min(page, maxPage);
  return (
    <section className="local-run-summary" aria-label="전체 런 요약">
      <h3>전체 런 요약</h3>
      <strong>{statuses[summary.status]}</strong>
      <p>
        {summary.context.filename} ·{' '}
        {summary.context.clock === 'receipt_monotonic' ? '수신 시작 기준' : '로그 부팅 후 시각'}{' '}
        {duration(summary.context.start_us)}–{duration(summary.context.end_us)}
      </p>
      <div className="local-run-totals">
        <div>
          기록 범위<strong>{duration(summary.duration_us)}</strong>
        </div>
        <div>
          분석 입력 사용<strong>{duration(summary.analyzed_us)}</strong>
        </div>
        <div>
          미분석<strong>{duration(summary.excluded_us)}</strong>
        </div>
        <div>
          전체 사용률
          <strong>
            {summary.coverage === null ? '범위 미확정' : `${(summary.coverage * 100).toFixed(1)}%`}
          </strong>
        </div>
      </div>
      {summary.coverage !== null && (
        <div
          className="local-run-coverage"
          role="img"
          aria-label={`분석 입력 사용 ${(summary.coverage * 100).toFixed(1)}%, 나머지 미분석`}
        >
          <span style={{ width: `${summary.coverage * 100}%` }} />
        </div>
      )}
      <p>{summary.note} 사용 시간은 모델의 초기 이력도 포함하며, 실제 점수 산출 횟수와 다릅니다.</p>
      {summary.context.notes.map((note, i) => (
        <p className="local-warning" key={i}>
          {note}
        </p>
      ))}
      {summary.models.map((model) => (
        <div key={model.model_id}>
          <h4>{model.model_id}</h4>
          <p>
            정상 범위 {model.segment_counts.normal}구간 · 이상 의심 {model.segment_counts.suspect}
            구간 · 판정 불가 {model.segment_counts.unavailable}구간
          </p>
          <p>
            판정 가능 관측 {model.observations}개 중 기준 초과 {model.exceedances}개
            {model.adapted_variants > 1 ? ' · 구간별 가중치 버전이 다릅니다.' : ''}
          </p>
          <MetricTable metrics={model} />
        </div>
      ))}
      <details open>
        <summary>분석·미분석 구간과 제외 이유</summary>
        <label className="local-segment">
          <input
            type="checkbox"
            checked={onlyExcluded}
            onChange={(e) => {
              setOnlyExcluded(e.target.checked);
              setPage(0);
            }}
          />
          미분석 구간만 보기
        </label>
        <div className="local-run-table">
          <table>
            <thead>
              <tr>
                <th>시작–종료</th>
                <th>길이</th>
                <th>처리 결과·이유</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(current * 12, (current + 1) * 12).map((r, i) => (
                <tr key={`${r.start_us}-${i}`}>
                  <td>
                    {duration(r.start_us)}–{duration(r.end_us)}
                  </td>
                  <td>{duration(r.end_us - r.start_us)}</td>
                  <td>
                    {reasons[r.reason] ?? r.reason}
                    {!!r.missing_features?.length && (
                      <details>
                        <summary>해당 입력 {r.missing_features.length}개</summary>
                        <p>{r.missing_features.join(', ')}</p>
                      </details>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!rows.length && <p>해당 구간이 없습니다.</p>}
        {maxPage > 0 && (
          <div>
            <button
              className="button small"
              disabled={current === 0}
              onClick={() => setPage(current - 1)}
            >
              이전 구간
            </button>
            <span>
              {' '}
              {current + 1} / {maxPage + 1}{' '}
            </span>
            <button
              className="button small"
              disabled={current === maxPage}
              onClick={() => setPage(current + 1)}
            >
              다음 구간
            </button>
          </div>
        )}
      </details>
    </section>
  );
}

export default function LocalRunResult({
  summary,
  analyses,
}: {
  summary?: RunSummary;
  analyses: Analysis[];
}) {
  return (
    <>
      {summary ? (
        <RunSummaryView summary={summary} />
      ) : (
        <p>
          이전 결과에는 전체 런 범위가 기록되어 있지 않습니다. 로그 구간을 다시 저장·분석하면 전체
          요약을 볼 수 있습니다.
        </p>
      )}
      {!!analyses.length && <h3>구간별 상세 결과</h3>}
      {analyses.map((analysis, i) => (
        <details key={`${analysis.model_id}-${i}`}>
          <summary>
            {analysis.model_id} ·{' '}
            {analysis.segment_results
              ? `${duration(analysis.segment_results[0].start_us)}–${duration(analysis.segment_results.at(-1)!.end_us)}`
              : `결과 ${i + 1}`}{' '}
            · {analysis.retained_segments}구간
          </summary>
          <AnalysisResult result={analysis} />
        </details>
      ))}
    </>
  );
}
