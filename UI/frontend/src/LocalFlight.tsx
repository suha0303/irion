import LocalRunResult from './LocalRunResult';
import { useEffect, useRef, useState } from 'react';
import { api, post, upload } from './api';
import './local-flight.css';
import LocalPreparation, {
  AnalysisResult,
  TimingResult,
  FlightEventResult,
} from './LocalPreparation';
import type { Profile, Job } from './local-flight-types';
import LocalModels from './LocalModels';
import LocalCalibration from './LocalCalibration';
import LocalProfile from './LocalProfile';

type Model = { id: string; task: 'forecast' | 'run_health'; label: string; version: string };
const labels: Record<string, string> = {
  queued: '대기',
  running: '실행 중',
  complete: '완료',
  failed: '실패',
  cancelled: '중지',
  interrupted: '이전 실행 중단',
};

type Page = 'profile' | 'logs' | 'analysis' | 'training' | 'models' | 'results';
const pages: [Page, string, string][] = [
  ['profile', '기체 프로파일', '분석에 필요한 값과 로그에서 확인한 근거를 관리합니다.'],
  ['logs', '로그 준비', '로그를 가져오고 보존된 비행 구간을 선택합니다.'],
  ['analysis', '로그 분석', '모델을 선택해 준비된 비행을 분석합니다.'],
  ['training', '파인튜닝 · 보정', '정상 학습, 독립 비교, 판정 기준 보정을 순서대로 진행합니다.'],
  ['models', '모델 보관함', '비교한 모델을 선택하거나 이전 버전으로 돌아갑니다.'],
  ['results', '작업 결과', '작업을 선택하면 전체 런과 구간별 결과가 펼쳐집니다.'],
];
export default function LocalFlight({ active }: { active: boolean }) {
  const [page, setPage] = useState<Page>('profile');
  const [query, setQuery] = useState('');
  const [resultPage, setResultPage] = useState(0);
  const [selectedJob, setSelectedJob] = useState('');
  const [jobFilter, setJobFilter] = useState('all');
  const [jobsLoading, setJobsLoading] = useState(true);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [models, setModels] = useState<Model[]>([]);
  const [mode, setMode] = useState<'forecast' | 'run_health'>('forecast');
  const [selected, setSelected] = useState<string[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profileLog, setProfileLog] = useState('');
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [available, setAvailable] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const running = jobs.some((j) => j.status === 'running' || j.status === 'queued');

  useEffect(() => {
    let mounted = true;
    Promise.all([
      api<{ available: boolean; models: Model[]; reason?: string }>('/local-ai/catalog'),
      api<{ profiles: Profile[] }>('/local-ai/profiles'),
    ])
      .then(([c, p]) => {
        if (!mounted) return;
        setModels(c.models);
        setAvailable(c.available);
        setProfiles(p.profiles);
        if (c.reason) setError(c.reason);
      })
      .catch((e) => {
        if (mounted) setError(String(e));
      })
      .finally(() => {
        if (mounted) setCatalogLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const ids = models.filter((m) => m.task === mode).map((m) => m.id);
    setSelected(ids.length ? [ids[0], ids.find((id) => id.includes('transformer')) ?? ids[0]] : []);
  }, [mode, models]);

  useEffect(() => {
    if (!active) return;
    let mounted = true;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const result = await api<{ jobs: Job[] }>('/local-ai/jobs');
        if (mounted) {
          setJobs(result.jobs);
          setJobsLoading(false);
        }
      } catch (e) {
        if (mounted) setError(String(e));
      }
      if (mounted) timer = setTimeout(refresh, 1500);
    };
    void refresh();
    return () => {
      mounted = false;
      clearTimeout(timer);
    };
  }, [active]);

  async function perform(work: () => Promise<void>) {
    setBusy(true);
    setError('');
    try {
      await work();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const filteredJobs = jobs.filter(
    (j) =>
      (jobFilter === 'all' || j.action === jobFilter) &&
      `${j.id} ${j.result?.filename ?? ''} ${j.result?.analyses?.map((a) => a.model_id).join(' ') ?? ''}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const maxResultPage = Math.max(0, Math.ceil(filteredJobs.length / 12) - 1);
  const currentResultPage = Math.min(resultPage, maxResultPage);
  return (
    <div>
      <section className="local-flight" aria-label="로컬 AI 관리 작업">
        <header>
          <div>
            <span className="eyebrow">ON DEVICE · CPU</span>
            <p>
              {catalogLoading
                ? '모델 목록 확인 중…'
                : `${models.length}종 모델 · 등록 기체 ${profiles.length}개 · ${running ? '작업 실행 중' : '작업 대기'}`}
            </p>
          </div>
          <button className="button" onClick={() => setPage('results')}>
            작업 결과 {jobs.length}개
          </button>
        </header>
        <nav className="local-task-nav" aria-label="로컬 AI 관리 메뉴">
          {pages.map(([id, label]) => (
            <button
              className={page === id ? 'active' : ''}
              key={id}
              aria-current={page === id ? 'page' : undefined}
              onClick={() => setPage(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="local-task-intro">
          <h2>{pages.find(([id]) => id === page)?.[1]}</h2>
          <p>{pages.find(([id]) => id === page)?.[2]}</p>
        </div>
        {error && (
          <div className="notice error" role="alert">
            {error}
          </div>
        )}
        <div className="local-flight-grid" data-page={page}>
          <section className="panel" hidden={page !== 'analysis' && page !== 'training'}>
            <h2>분석 모델 선택</h2>
            <label className="form-label">
              분석 모드
              <select
                className="text-input"
                value={mode}
                onChange={(e) => setMode(e.target.value as 'forecast' | 'run_health')}
              >
                <option value="forecast">비행 중 예측 · 8종</option>
                <option value="run_health">비행 종료 후 분석 · 8종</option>
              </select>
            </label>
            {['A', 'B'].map((slot, index) => (
              <label className="form-label" key={slot}>
                비교 모델 {slot}
                <select
                  className="text-input"
                  value={selected[index] ?? ''}
                  onChange={(e) =>
                    setSelected((s) => s.map((v, i) => (i === index ? e.target.value : v)))
                  }
                >
                  {models
                    .filter((m) => m.task === mode)
                    .map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label}
                      </option>
                    ))}
                </select>
              </label>
            ))}
            <p>
              최신 2026-11-08 묶음 · {models.length}종 · CPU 실행
              <br />
              검증 범위는 X500 시뮬레이션입니다. 실기체 입력 연결·적응 성능은 별도 검증합니다.
            </p>
            <button
              className="button"
              disabled={!available || busy || running}
              onClick={() =>
                perform(async () => {
                  const job = await post<Job>('/local-ai/benchmark');
                  setJobs((j) => [job, ...j]);
                })
              }
            >
              16종 CPU 실행 점검
            </button>
            <small>합성 입력 기능·속도 점검입니다. 비행의 정상 여부를 판정하지 않습니다.</small>
          </section>
          <div hidden={page !== 'profile'}>
            <LocalProfile
              profiles={profiles}
              jobs={jobs}
              pendingLog={profileLog}
              disabled={!available || busy}
              onSaved={(saved) =>
                setProfiles((previous) => [
                  saved,
                  ...previous.filter((p) => p.version !== saved.version),
                ])
              }
              reinspect={(id) =>
                perform(async () => {
                  const job = await post<Job>(`/local-ai/logs/${id}/reinspect`);
                  setJobs((j) => [job, ...j]);
                  setProfileLog(job.id);
                })
              }
            />
          </div>
          <section className="panel" hidden={page !== 'logs'}>
            <h2>비행 로그 가져오기</h2>
            <p>
              PX4 ULog 또는 ArduPilot BIN의 메시지·필드·시간축·파서 경고를 확인합니다. 검사가 끝나면
              로그에 명시된 프로파일 값을 자동으로 채웁니다. 정상 여부는 추정하지 않습니다.
            </p>
            <input
              hidden
              ref={input}
              type="file"
              accept=".bin,.BIN,.ulg"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file)
                  void perform(async () => {
                    const job = await upload<Job>('/local-ai/logs', file);
                    setJobs((j) => [job, ...j]);
                    setProfileLog(job.id);
                  });
                e.target.value = '';
              }}
            />
            <button
              className="button primary"
              disabled={!available || busy || running}
              onClick={() => input.current?.click()}
            >
              ULog / BIN 가져오기
            </button>
            <small>최대 256 MiB · 원본 보관 · PC 내부 처리</small>
            <div className="notice warning">
              적응 학습에는 정상임이 확인된 비행과 검증된 신호 매핑이 필요합니다. 현재 정보가 없는
              로그는 검사·열람 단계에 머뭅니다.
            </div>
          </section>
          <LocalPreparation
            page={
              page === 'logs'
                ? 'prepare'
                : page === 'training'
                  ? 'adapt'
                  : page === 'analysis'
                    ? 'analysis'
                    : 'hidden'
            }
            jobs={jobs}
            profiles={profiles}
            selectedModels={selected}
            disabled={!available || busy || running}
            submit={(path, body) =>
              perform(async () => {
                const job = await post<Job>(path, body);
                setJobs((j) => [job, ...j]);
              })
            }
          />
          <div hidden={page !== 'training'}>
            <LocalCalibration
              jobs={jobs}
              model={selected[0] ?? ''}
              disabled={!available || busy || running}
              submit={(path, body) =>
                perform(async () => {
                  const job = await post<Job>(path, body);
                  setJobs((j) => [job, ...j]);
                })
              }
            />
          </div>
          <div hidden={page !== 'models'}>
            <LocalModels
              jobs={jobs}
              disabled={!available || busy || running}
              submit={(path, body) =>
                perform(async () => {
                  const job = await post<Job>(path, body);
                  setJobs((j) => [job, ...j]);
                })
              }
              imported={(job) => setJobs((j) => [job, ...j])}
            />
          </div>
          <section className="panel" hidden={page !== 'results'}>
            <div className="local-insight-filters">
              <label>
                결과 검색
                <input
                  className="text-input"
                  value={query}
                  placeholder="파일명 · 모델명 · 작업 번호"
                  onChange={(e) => {
                    setQuery(e.target.value);
                    setResultPage(0);
                  }}
                />
              </label>
              <label>
                작업 종류
                <select
                  className="text-input"
                  value={jobFilter}
                  onChange={(e) => {
                    setJobFilter(e.target.value);
                    setResultPage(0);
                  }}
                >
                  <option value="all">전체 작업</option>
                  <option value="analyze">예측·분석</option>
                  <option value="inspect">로그 검사</option>
                  <option value="prepare">구간 저장</option>
                  <option value="adapt">파인튜닝</option>
                  <option value="compare">독립 비교</option>
                  <option value="calibration_fit">보정</option>
                </select>
              </label>
            </div>
            <h2>작업 결과 · {filteredJobs.length}개</h2>
            {jobsLoading && <p role="status">작업 목록을 불러오는 중…</p>}
            {!jobsLoading && jobs.length === 0 && <p>아직 실행한 작업이 없습니다.</p>}
            {filteredJobs.slice(currentResultPage * 12, currentResultPage * 12 + 12).map((job) => (
              <article className="local-job" key={job.id}>
                <button
                  className="local-job-toggle"
                  aria-expanded={selectedJob === job.id}
                  onClick={() => setSelectedJob((old) => (old === job.id ? '' : job.id))}
                >
                  <strong>
                    {{
                      inspect: '로그 검사',
                      benchmark: 'CPU 실행 점검',
                      timeline: '보존 구간 검사',
                      prepare: '구간 저장',
                      analyze: '예측·분석',
                      adapt: '적응 학습',
                      compare: '독립 비교',
                      calibration_fit: '임계값 보정',
                      calibration_select: '보정 적용',
                      admit_live: '수신 비행 구간 등록',
                    }[job.action] ?? job.action}{' '}
                    · {job.id.slice(0, 8)} · {labels[job.status] ?? job.status}
                  </strong>
                  <span>{selectedJob === job.id ? '접기' : '결과 보기'}</span>
                </button>
                {selectedJob === job.id && (
                  <div className="local-job-body">
                    {(job.status === 'running' || job.status === 'queued') && (
                      <button
                        className="button small"
                        onClick={() =>
                          perform(async () => {
                            await post(`/local-ai/jobs/${job.id}/cancel`);
                          })
                        }
                      >
                        중지
                      </button>
                    )}
                    {job.error && <p role="alert">{job.error}</p>}
                    {job.result && job.action === 'inspect' && (
                      <>
                        <p>
                          {job.result.filename} · {job.result.autopilot}
                          <br />
                          건강 상태 미확정 · 정상 학습에 사용하지 않음
                          <br />
                          후보 필드 누락:{' '}
                          {Object.keys(job.result.missing_candidate_fields ?? {}).length}개 메시지
                        </p>
                        {job.result.parser_warnings && (
                          <p className="local-warning">파서 경고: {job.result.parser_warnings}</p>
                        )}
                        <details>
                          <summary>입력 진단 상세</summary>
                          <pre>
                            {JSON.stringify(
                              {
                                missing: job.result.missing_candidate_fields,
                                parameters: job.result.parameters,
                                messages: job.result.messages,
                              },
                              null,
                              2,
                            )}
                          </pre>
                        </details>
                      </>
                    )}
                    {job.result && job.action === 'benchmark' && (
                      <details open>
                        <summary>16종 합성 입력 측정</summary>
                        {job.result.models?.map((m) => (
                          <p key={m.id}>
                            {m.id}: p95 {m.p95_ms.toFixed(2)} ms
                          </p>
                        ))}
                      </details>
                    )}
                    {job.result?.flight_events && (
                      <FlightEventResult events={job.result.flight_events} />
                    )}
                    {job.result?.timing && <TimingResult timing={job.result.timing} />}
                    {job.result?.analyses && (
                      <LocalRunResult
                        summary={job.result.run_summary}
                        analyses={job.result.analyses}
                      />
                    )}
                    {job.result?.original && (
                      <>
                        <h3>원본</h3>
                        <AnalysisResult result={job.result.original} />
                      </>
                    )}
                    {job.result?.adapted && (
                      <>
                        <h3>적응 모델</h3>
                        <AnalysisResult result={job.result.adapted} />
                      </>
                    )}
                    {job.progress && (
                      <p>
                        {job.progress.epoch} epoch · loss {job.progress.train_loss.toFixed(5)}
                      </p>
                    )}
                    {job.action === 'adapt' &&
                      ['cancelled', 'interrupted', 'failed'].includes(job.status) && (
                        <button
                          className="button"
                          disabled={busy || running}
                          onClick={() =>
                            perform(async () => {
                              const row = await post<Job>(`/local-ai/jobs/${job.id}/resume`);
                              setJobs((j) => [row, ...j]);
                            })
                          }
                        >
                          마지막 완료 epoch에서 재개
                        </button>
                      )}
                  </div>
                )}
              </article>
            ))}
            {!jobsLoading && filteredJobs.length === 0 && jobs.length > 0 && (
              <p>검색 조건에 맞는 작업이 없습니다.</p>
            )}
            {maxResultPage > 0 && (
              <div className="local-pagination">
                <button
                  className="button"
                  disabled={currentResultPage === 0}
                  onClick={() => setResultPage(currentResultPage - 1)}
                >
                  이전
                </button>
                <span>
                  {currentResultPage + 1} / {maxResultPage + 1}
                </span>
                <button
                  className="button"
                  disabled={currentResultPage === maxResultPage}
                  onClick={() => setResultPage(currentResultPage + 1)}
                >
                  다음
                </button>
              </div>
            )}
          </section>
        </div>
      </section>
    </div>
  );
}
