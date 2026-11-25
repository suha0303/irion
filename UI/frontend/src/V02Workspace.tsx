import { useEffect, useState } from 'react';
import { api, post } from './api';
import FlightPreparation from './FlightPreparation';
import { FlightEventResult, TimingResult } from './FlightLogDetails';
import LocalProfile from './LocalProfile';
import LogInspectionResult from './LogInspectionResult';
import V02Adaptation from './V02Adaptation';
import ModelSelection, { modelIds, modelLabel } from './V02ModelSelection';
import V02JobResult from './V02JobResult';
import { CandidatePicker } from './V02FlightCandidates';
import { jobFilename, type Job, type Profile, type V02Model } from './local-flight-types';
import './local-flight.css';

const pages = [
  ['profile', '기체 프로파일'],
  ['logs', '로그 준비'],
  ['analysis', '로그 분석'],
  ['training', '파인튜닝 · 보정'],
  ['results', '작업 결과'],
] as const;
const actionLabels: Record<string, string> = {
  inspect: '로그 검사',
  timeline: '구간 검사',
  prepare: '구간 저장',
  admit_live: '비행 구간 등록',
  v02_analyze: '로그 분석',
  v02_adapt: '파인튜닝 · 보정',
  v02_candidate_import: '기존 파인튜닝 후보 등록',
  v02_candidate_replay: '저장 비행 후보 분석',
};
const statusLabels: Record<string, string> = {
  queued: '대기',
  running: '실행 중',
  complete: '완료',
  cancelled: '중지',
  interrupted: '중단됨',
  failed: '실패',
};
const jobTime = new Intl.DateTimeFormat('ko-KR', {
  timeZone: 'Asia/Seoul',
  dateStyle: 'short',
  timeStyle: 'short',
});

export default function V02Workspace({
  active,
  uploadedLog,
  preparationLog,
  chooseLog,
  uploading,
}: {
  active: boolean;
  uploadedLog: Job | null;
  preparationLog?: Job | null;
  chooseLog: () => void;
  uploading: boolean;
}) {
  const [page, setPage] = useState<(typeof pages)[number][0]>('profile');
  const [catalog, setCatalog] = useState<{
    available: boolean;
    models: V02Model[];
    reason?: string;
  }>();
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [catalogError, setCatalogError] = useState('');
  const [jobsError, setJobsError] = useState('');
  const [liveBusy, setLiveBusy] = useState(false);
  const [profileLog, setProfileLog] = useState('');
  const [returnToPreparation, setReturnToPreparation] = useState(false);
  const [sourceLog, setSourceLog] = useState('');
  const [analysisLog, setAnalysisLog] = useState('');
  const [analysisCandidate, setAnalysisCandidate] = useState('');
  const [model, setModel] = useState('inflight-lstm-128x4');
  const [selected, setSelected] = useState('');
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(12);
  const running = jobs.some((job) => job.status === 'queued' || job.status === 'running');
  const unavailable = !active || !catalog?.available || busy || running || liveBusy;
  const models = catalog?.models ?? [];
  const selectedModels = modelIds(models, model);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    const options = { signal: controller.signal };
    Promise.all([
      api<{ available: boolean; models: V02Model[]; reason?: string }>(
        '/local-ai/catalog',
        options,
      ),
      api<{ profiles: Profile[] }>('/local-ai/profiles', options),
    ])
      .then(([data, aircraft]) => {
        if (controller.signal.aborted) return;
        setCatalog(data);
        setProfiles(aircraft.profiles);
        setCatalogError(data.reason ?? '');
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setCatalogError(String(failure));
      });
    return () => controller.abort();
  }, [active]);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const data = await api<{ jobs: Job[]; live_busy?: boolean }>('/local-ai/jobs', {
          signal: controller.signal,
        });
        if (!controller.signal.aborted) {
          setJobs(data.jobs);
          setLiveBusy(Boolean(data.live_busy));
          setJobsError('');
        }
      } catch (failure) {
        if (!controller.signal.aborted) setJobsError(String(failure));
      }
      if (!controller.signal.aborted) timer = setTimeout(refresh, 1500);
    }
    void refresh();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [active]);
  useEffect(() => {
    if (!uploadedLog) return;
    setJobs((rows) =>
      rows.some((row) => row.id === uploadedLog.id) ? rows : [uploadedLog, ...rows],
    );
    setProfileLog(uploadedLog.id);
    setSelected(uploadedLog.id);
    setQuery('');
    setLimit(12);
    setPage('results');
  }, [uploadedLog]);
  useEffect(() => {
    if (!preparationLog) return;
    setJobs((rows) =>
      rows.some((row) => row.id === preparationLog.id) ? rows : [preparationLog, ...rows],
    );
    setSourceLog(preparationLog.id);
    setReturnToPreparation(false);
    setPage('logs');
  }, [preparationLog]);

  async function submit(path: string, body?: unknown) {
    setBusy(true);
    setError('');
    try {
      const row = await post<Job>(path, body);
      setJobs((previous) => [row, ...previous.filter((job) => job.id !== row.id)]);
      if (row.action !== 'timeline') {
        setSelected(row.id);
        setQuery('');
        setLimit(12);
        setPage('results');
      }
      return row;
    } catch (failure) {
      setError(String(failure));
    } finally {
      setBusy(false);
    }
  }

  const prepared = jobs.filter((job) => job.action === 'prepare' && job.status === 'complete');
  const results = jobs.filter((job) =>
    [
      jobFilename(job),
      actionLabels[job.action],
      job.id,
      job.result?.v02?.filename,
      ...(job.result?.v02?.candidates ?? []).map((candidate) => candidate.profile_name),
    ]
      .join(' ')
      .toLowerCase()
      .includes(query.trim().toLowerCase()),
  );
  return (
    <section className="local-flight" aria-label="V0.2 로컬 AI">
      <header>
        <p role="status">
          V0.2 · 연결 모델 {models.length}종 ·{' '}
          {liveBusy ? '비행 분석에서 AI 사용 중' : running ? '작업 실행 중' : '작업 대기'}
        </p>
      </header>
      <nav className="local-task-nav" aria-label="로컬 AI 관리 메뉴">
        {pages.map(([id, name]) => (
          <button
            key={id}
            className={page === id ? 'active' : ''}
            aria-current={page === id ? 'page' : undefined}
            onClick={() => setPage(id)}
          >
            {name}
          </button>
        ))}
      </nav>
      {(error || catalogError || jobsError) && (
        <p role="alert" className="notice error">
          {error || catalogError || jobsError}
        </p>
      )}
      <div className="local-flight-grid" data-page={page === 'logs' ? 'logs' : 'v02'}>
        <div hidden={page !== 'profile'}>
          <LocalProfile
            profiles={profiles}
            jobs={jobs}
            pendingLog={profileLog}
            disabled={unavailable}
            onSaved={(item) => {
              setProfiles((rows) => [
                item,
                ...rows.filter((profile) => profile.version !== item.version),
              ]);
              if (returnToPreparation) {
                setReturnToPreparation(false);
                setPage('logs');
              }
            }}
            reinspect={async (id) => {
              await submit(`/local-ai/logs/${id}/reinspect`);
            }}
          />
        </div>
        <section className="panel" hidden={page !== 'logs'}>
          <h2>비행 로그 가져오기</h2>
          <p>PX4 ULog · ArduPilot BIN</p>
          <button
            className="button primary"
            disabled={unavailable || uploading}
            onClick={chooseLog}
          >
            {uploading ? '가져오는 중…' : 'ULog / BIN 가져오기'}
          </button>
        </section>
        <FlightPreparation
          jobs={jobs}
          profiles={profiles}
          sourceLog={sourceLog}
          onSourceLog={setSourceLog}
          onCreateProfile={(id) => {
            setProfileLog(id);
            setReturnToPreparation(true);
            setPage('profile');
          }}
          disabled={unavailable}
          submit={submit}
          hidden={page !== 'logs'}
        />
        <section className="panel" hidden={page !== 'analysis'}>
          <h2>개별 로그 분석</h2>
          <p>준비한 로그 구간의 팩 전압·각속도 예측 또는 종료 후 판정을 확인합니다.</p>
          <label className="form-label">
            준비된 로그
            <select
              className="text-input"
              value={analysisLog}
              onChange={(e) => {
                setAnalysisLog(e.target.value);
                setAnalysisCandidate('');
              }}
            >
              <option value="">선택</option>
              {prepared.map((job) => (
                <option key={job.id} value={job.id}>
                  {jobFilename(job) ?? job.id.slice(0, 8)} ·{' '}
                  {job.request?.profile?.name ?? '저장 구간'} · {job.result?.frames ?? '—'}프레임
                </option>
              ))}
            </select>
          </label>
          {analysisLog && (
            <CandidatePicker
              active={active && page === 'analysis'}
              profile={prepared.find((job) => job.id === analysisLog)?.request?.profile?.id}
              value={analysisCandidate}
              onChange={setAnalysisCandidate}
            />
          )}
          {!analysisCandidate && (
            <ModelSelection models={models} selected={model} onChange={setModel} />
          )}
          {prepared.length === 0 && (
            <button className="button" onClick={() => setPage('logs')}>
              로그 준비 열기
            </button>
          )}
          <button
            className="button primary"
            disabled={unavailable || !analysisLog || (!analysisCandidate && !selectedModels.length)}
            onClick={() =>
              void submit('/local-ai/v02/analyze', {
                prepared_job: analysisLog,
                ...(analysisCandidate
                  ? { candidate_job: analysisCandidate }
                  : { models: selectedModels }),
              })
            }
          >
            {analysisCandidate
              ? '파인튜닝 모델로 분석'
              : selectedModels.length > 1
                ? `${selectedModels.length}종 일괄 분석`
                : '로그 분석 실행'}
          </button>
        </section>
        <div hidden={page !== 'training'}>
          <V02Adaptation
            jobs={jobs}
            models={models}
            disabled={unavailable}
            goToLogs={() => setPage('logs')}
            submit={async (logs, ids) => {
              await submit('/local-ai/v02/adapt', { logs, models: ids });
            }}
          />
        </div>
        <section className="panel" hidden={page !== 'results'}>
          <h2>작업 결과 · {results.length}개</h2>
          <label className="form-label">
            결과 검색
            <input
              className="text-input"
              placeholder="파일명 · 작업 · 기체"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setLimit(12);
              }}
            />
          </label>
          {!results.length && <p>검색 조건에 맞는 작업이 없습니다.</p>}
          {results.slice(0, limit).map((job) => (
            <article className="local-job" key={job.id}>
              <button
                className="local-job-toggle"
                aria-expanded={selected === job.id}
                onClick={() => setSelected(selected === job.id ? '' : job.id)}
              >
                <strong>
                  {[
                    jobFilename(job) ??
                      job.result?.v02?.filename ??
                      job.result?.v02?.candidates[0]?.profile_name,
                    actionLabels[job.action] ?? job.action,
                    job.status === 'complete' && Object.keys(job.result?.v02?.failures ?? {}).length
                      ? '일부 모델 실패'
                      : (statusLabels[job.status] ?? job.status),
                  ]
                    .filter(Boolean)
                    .join(' · ')}
                </strong>
                <span>{selected === job.id ? '접기' : '결과 보기'}</span>
              </button>
              {selected === job.id && (
                <div className="local-job-body">
                  {job.created_at && <p>요청 시각 {jobTime.format(job.created_at * 1000)}</p>}
                  {job.error && <p role="alert">{job.error}</p>}
                  {job.progress?.model_id && (
                    <p role="status">
                      {modelLabel(job.progress.model_id)} · {job.progress.finished}/
                      {job.progress.total} 처리
                    </p>
                  )}
                  {['queued', 'running'].includes(job.status) && (
                    <button
                      className="button"
                      disabled={busy}
                      onClick={() => void submit(`/local-ai/jobs/${job.id}/cancel`)}
                    >
                      작업 중지
                    </button>
                  )}
                  {job.action === 'inspect' && job.result && (
                    <>
                      <LogInspectionResult result={job.result} />
                      <div className="local-job-actions">
                        <button
                          className="button"
                          onClick={() => {
                            setProfileLog(job.id);
                            setReturnToPreparation(false);
                            setPage('profile');
                          }}
                        >
                          기체 프로파일 확인
                        </button>
                        <button
                          className="button primary"
                          onClick={() => {
                            setSourceLog(job.id);
                            setPage('logs');
                          }}
                        >
                          이 로그의 구간 준비
                        </button>
                      </div>
                    </>
                  )}
                  {job.result?.flight_events && (
                    <FlightEventResult events={job.result.flight_events} />
                  )}
                  {job.result?.timing && <TimingResult timing={job.result.timing} />}
                  {job.action === 'prepare' && job.status === 'complete' && (
                    <div className="local-job-actions">
                      <button
                        className="button"
                        onClick={() => {
                          setAnalysisLog(job.id);
                          setAnalysisCandidate('');
                          setPage('analysis');
                        }}
                      >
                        이 로그 분석
                      </button>
                      <button className="button" onClick={() => setPage('training')}>
                        파인튜닝 · 보정 열기
                      </button>
                    </div>
                  )}
                  {job.result?.v02 && <V02JobResult jobId={job.id} result={job.result.v02} />}
                </div>
              )}
            </article>
          ))}
          {limit < results.length && (
            <button className="button" onClick={() => setLimit(limit + 12)}>
              작업 더 보기
            </button>
          )}
        </section>
      </div>
    </section>
  );
}
