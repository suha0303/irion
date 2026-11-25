import { useEffect, useState } from 'react';
import { api, post } from './api';
import {
  FlightAnalysisPanel,
  FlightPostflight,
  FlightResultSelect,
  FlightVerdictCards,
  type MonitorTimeline,
} from './FlightAnalysisViews';
import {
  jobFilename,
  type Job,
  type V02Model,
  type V02Result,
  type V02Threshold,
} from './local-flight-types';
import { CandidatePicker } from './V02FlightCandidates';
import V02JobResult from './V02JobResult';
import ModelSelection, { modelIds, modelLabel } from './V02ModelSelection';

function belongsToLog(job: Job, log: Job) {
  const hash = log.result?.sha256;
  return (
    (!!hash &&
      (job.result?.source_sha256 === hash ||
        job.result?.v02?.analyses.some((model) =>
          model.evaluation.some((row) => row.source_sha256 === hash),
        ))) ||
    (!!log.request?.path && job.request?.path === log.request.path)
  );
}

function LogPostflight({ result }: { result?: V02Result }) {
  const models = result?.analyses.filter((model) => model.task === 'whole-run') ?? [];
  const failed = Object.keys(result?.failures ?? {}).some((id) => id.startsWith('whole-run-'));
  return (
    <FlightPostflight>
      {models.length === 0 ? (
        <p>
          {failed
            ? '종료 후 판정 모델 실행에 실패했습니다. 아래 모델별 오류를 확인하세요.'
            : result?.analyses.some((model) => model.task === 'inflight')
              ? '선택한 결과는 비행 중 예측용입니다. 종료 후 판정 모델을 실행하면 이곳에 표시됩니다.'
              : '분석이 끝나면 판정이 표시됩니다.'}
        </p>
      ) : (
        <FlightVerdictCards
          items={models.map((model) => {
            const decisions = model.evaluation.flatMap((row) => {
              const rule = row.whole_decisions?.calibrated ? 'calibrated' : 'original';
              return row.segments.map((segment) => segment.whole_decisions?.[rule]);
            });
            const available = decisions.filter(
              (item): item is V02Threshold => item != null && item.alarm != null,
            );
            const worst = available.reduce<(typeof available)[number] | undefined>(
              (best, current) =>
                !best || (current.score ?? 0) > (best.score ?? 0) ? current : best,
              undefined,
            );
            const alarms = available.filter((item) => item.alarm).length;
            return {
              key: model.model_id,
              model: modelLabel(model.model_id).split(' · ')[1],
              candidate: model.evaluation.some((row) => Boolean(row.whole_decisions?.calibrated)),
              alarm: available.length ? alarms > 0 : null,
              score: worst?.score ?? null,
              threshold: worst?.threshold,
              detail: available.length ? `경보 구간 ${alarms} / ${available.length}` : undefined,
            };
          })}
        />
      )}
    </FlightPostflight>
  );
}

const active = (job: Job) => job.status === 'queued' || job.status === 'running';
const dateTime = new Intl.DateTimeFormat('ko-KR', {
  timeZone: 'Asia/Seoul',
  dateStyle: 'short',
  timeStyle: 'medium',
});

function resultDescription(job: Job) {
  const result = job.result?.v02;
  const models = result?.analyses.map((model) => model.model_id) ?? job.request?.models ?? [];
  const candidate = result?.candidate_job ?? job.request?.candidate_job;
  const scope =
    models.length === 1
      ? modelLabel(models[0])
      : [
          models.filter((id) => id.startsWith('inflight-')).length,
          models.filter((id) => id.startsWith('whole-run-')).length,
        ]
          .map((count, i) => (count ? `${i === 0 ? '비행 중' : '종료 후'} ${count}종` : ''))
          .filter(Boolean)
          .join(' / ');
  const length = result?.coverage?.analyzed_us;
  return [
    candidate ? `후보 ${candidate.slice(0, 8)} · 전용 보정 기준` : '원본 모델 · 기존 기준',
    scope,
    length == null ? '' : `${(length / 1e6).toFixed(1)}초 분석`,
    job.created_at == null ? '' : dateTime.format(job.created_at * 1000),
  ]
    .filter(Boolean)
    .join(' · ');
}

export default function LogAnalysis({
  log,
  jobs,
  isV02,
  openInspection,
  onJob,
  liveBusy,
  timeline,
}: {
  log: Job;
  jobs: Job[];
  isV02: boolean;
  openInspection: (job: Job) => void;
  onJob: (job: Job) => void;
  liveBusy: boolean;
  timeline?: MonitorTimeline;
}) {
  const [models, setModels] = useState<V02Model[]>([]);
  const [base, setBase] = useState('all');
  const [packageId, setPackageId] = useState('');
  const [preparedId, setPreparedId] = useState('');
  const [analysisId, setAnalysisId] = useState('');
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');
  const prepared = jobs.filter(
    (job) => job.action === 'prepare' && job.status === 'complete' && belongsToLog(job, log),
  );
  const input = prepared.find((job) => job.id === preparedId) ?? prepared[0];
  const analyses = jobs.filter((job) => job.action === 'v02_analyze' && belongsToLog(job, log));
  const analysis =
    analyses.find((job) => job.id === analysisId) ??
    analyses.find((job) => job.status === 'complete') ??
    analyses[0];
  const result = analysis?.result?.v02;
  const coverage = result?.coverage;
  const interval = coverage?.timeline.filter((item) => item.reason === 'analyzed') ?? [];
  const origin = timeline?.originUs;
  const localBusy = jobs.some(active);
  const blocked = liveBusy
    ? '실시간 비행 분석이 끝나면 새 로그 분석을 시작할 수 있습니다.'
    : localBusy
      ? '진행 중인 로컬 AI 작업이 끝나면 새 분석을 시작할 수 있습니다.'
      : '';
  const enabled = isV02 && Boolean(input);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    void api<{ models: V02Model[] }>('/local-ai/catalog', { signal: controller.signal }).then(
      (value) => {
        if (!controller.signal.aborted) setModels(value.models);
      },
      (failure: unknown) => {
        if (!controller.signal.aborted) setError(String(failure));
      },
    );
    return () => controller.abort();
  }, [enabled]);

  async function start() {
    if (!input || blocked) return;
    setStarting(true);
    setError('');
    try {
      const job = await post<Job>('/local-ai/v02/analyze', {
        prepared_job: input.id,
        ...(packageId ? { candidate_job: packageId } : { models: modelIds(models, base) }),
      });
      setAnalysisId(job.id);
      onJob(job);
    } catch (failure) {
      setError(String(failure));
    } finally {
      setStarting(false);
    }
  }

  return (
    <FlightAnalysisPanel
      status={
        analysis && active(analysis)
          ? '분석 중'
          : result
            ? '분석 완료'
            : analysis?.error
              ? '분석 실패'
              : '분석 대기'
      }
      action={
        log.status === 'complete' && (
          <button className="button small" onClick={() => openInspection(log)}>
            로그 검사 · 구간 준비
          </button>
        )
      }
    >
      {analysis && (
        <section className="flight-analysis-current" aria-label="현재 표시 중인 분석">
          <h3>현재 표시 중인 분석</h3>
          <p>{resultDescription(analysis)}</p>
          {coverage && (
            <p className="analysis-coverage">
              {coverage.retained_frames.toLocaleString()}프레임
              {coverage.context.extent_known &&
                ` / 원본 기록 ${((coverage.context.end_us - coverage.context.start_us) / 1e6).toFixed(1)}초`}
              {origin != null &&
                interval.length === 1 &&
                ` · 재생 ${((interval[0].start_us - origin) / 1e6).toFixed(1)}–${((interval[0].end_us - origin) / 1e6).toFixed(1)}초`}
              {interval.length > 1 && ` · ${interval.length}개 구간`}
            </p>
          )}
          <FlightResultSelect
            value={analysis.id}
            onChange={setAnalysisId}
            options={analyses.map((item) => ({
              value: item.id,
              label: `${resultDescription(item)} · ${item.id.slice(0, 8)}`,
            }))}
          />
        </section>
      )}
      {enabled && blocked && (
        <p id="log-analysis-blocked" role="status">
          {blocked}
        </p>
      )}
      <details className="flight-new-analysis" open={!result || undefined}>
        <summary>새 분석 실행</summary>
        <div className="flight-new-analysis-body">
          {enabled && input ? (
            <>
              {prepared.length > 1 && (
                <label className="form-label">
                  준비한 비행 구간
                  <select
                    className="text-input"
                    value={input.id}
                    onChange={(event) => {
                      setPreparedId(event.target.value);
                      setPackageId('');
                    }}
                  >
                    {prepared.map((item) => (
                      <option key={item.id} value={item.id}>
                        {jobFilename(item)} · {item.result?.frames?.toLocaleString()}시점 ·{' '}
                        {item.id.slice(0, 8)}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <div className="buffered-chart-controls">
                <CandidatePicker
                  profile={input.request?.profile?.id}
                  value={packageId}
                  onChange={setPackageId}
                />
                {!packageId && (
                  <ModelSelection models={models} selected={base} onChange={setBase} />
                )}
              </div>
              <button
                className="button small"
                disabled={starting || Boolean(blocked) || (!packageId && !models.length)}
                aria-describedby={blocked ? 'log-analysis-blocked' : undefined}
                onClick={() => void start()}
              >
                {packageId ? '선택한 후보로 분석' : '선택한 모델로 분석'}
              </button>
            </>
          ) : (
            <p>
              {isV02
                ? '로그 검사에서 기체 프로파일과 분석 구간을 저장하세요.'
                : '기체 관리에서 로그 분석을 실행하세요.'}
            </p>
          )}
          {error && <p role="alert">{error}</p>}
        </div>
      </details>
      <LogPostflight result={result} />
      {result && analysis ? (
        <V02JobResult
          key={analysis.id}
          jobId={analysis.id}
          result={result}
          monitor
          timeline={timeline}
        />
      ) : (
        <section className="buffered-chart">
          <h3>실측·예측·이상 점수 그래프</h3>
          {analysis?.error ? (
            <p role="alert">{analysis.error}</p>
          ) : (
            <p role="status">
              {analysis && active(analysis)
                ? '선택한 구간을 분석하고 있습니다.'
                : '분석이 끝나면 그래프가 표시됩니다.'}
            </p>
          )}
        </section>
      )}
    </FlightAnalysisPanel>
  );
}
