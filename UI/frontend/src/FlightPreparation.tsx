import { useEffect, useState } from 'react';
import { FlightEventResult, TimingResult } from './FlightLogDetails';
import { jobFilename, type Job, type Profile } from './local-flight-types';

export default function FlightPreparation({
  jobs,
  profiles,
  sourceLog: log,
  onSourceLog,
  onCreateProfile,
  disabled,
  submit,
  hidden,
  legacyRoles = false,
}: {
  jobs: Job[];
  profiles: Profile[];
  sourceLog: string;
  onSourceLog: (id: string) => void;
  onCreateProfile?: (id: string) => void;
  disabled: boolean;
  submit: (path: string, body: unknown) => Promise<Job | void>;
  hidden?: boolean;
  legacyRoles?: boolean;
}) {
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
  const [previewChoice, setPreviewChoice] = useState<{ id: string; settings: string }>();
  const [segments, setSegments] = useState<number[]>([]);
  const [evidence, setEvidence] = useState('');
  const [role, setRole] = useState('evaluation');
  const [health, setHealth] = useState('unknown');
  const [healthEvidence, setHealthEvidence] = useState('');
  const ready = (action: string) =>
    jobs.filter((j) => j.action === action && j.status === 'complete');
  const inspected = ready('inspect');
  const source = inspected.find((j) => j.id === log);
  const inspection = source?.result;
  const compatible = profiles.filter(
    ({ profile, eligibility }) =>
      (!inspection?.autopilot || profile.autopilot === inspection.autopilot) &&
      eligibility.model_structure_compatible !== false &&
      (eligibility.timeline_ready ?? eligibility.profile_complete),
  );
  const profile = compatible.find((p) => p.version === version);
  const previews = legacyRoles
    ? ready('timeline').filter(
        (job) =>
          source?.request?.path &&
          job.request?.path === source.request.path &&
          job.request?.profile?.id === profile?.profile.id,
      )
    : [];
  const settings = JSON.stringify([
    log,
    version,
    start,
    end,
    flightInterval,
    fastAge,
    batteryAge,
    corrections,
  ]);
  const previewId = previewChoice?.settings === settings ? previewChoice.id : '';
  const previewJob = (legacyRoles ? previews : jobs).find(
    (job) => job.id === previewId && job.action === 'timeline',
  );
  const preview = previewJob?.status === 'complete' ? previewJob.result?.timing : undefined;
  const validRange =
    start !== '' && end !== '' && Number(start) >= 0 && Number(end) > Number(start);
  const validHealth =
    (role === 'evaluation' || health === 'confirmed_normal') &&
    (health === 'unknown' || Boolean(healthEvidence.trim()));
  useEffect(() => {
    setVersion('');
    setFlightInterval('');
    setStart('');
    setEnd('');
    setCorrections({});
    setClock({ stream: '', source1: '', common1: '', source2: '', common2: '', evidence: '' });
    setPreviewChoice(undefined);
    setSegments([]);
    setEvidence('');
    setHealth('unknown');
    setHealthEvidence('');
  }, [log]);
  const jobLabel = (j: Job) =>
    [jobFilename(j) ?? j.id.slice(0, 8), j.request?.profile?.name].filter(Boolean).join(' · ');
  const corrected = Object.keys(corrections).length;
  const anchorReady =
    Object.values(clock).every((v) => v.trim() !== '') &&
    [clock.source1, clock.common1, clock.source2, clock.common2].every((v) =>
      Number.isFinite(Number(v)),
    );
  async function inspectRange() {
    const automatic = flightInterval === '' ? null : Number(flightInterval);
    const row = await submit('/local-ai/timeline', {
      log_job: log,
      profile_id: profile?.profile.id,
      profile_version: version,
      ...(automatic !== null
        ? { flight_interval: automatic }
        : {
            start_us: Math.round(Number(start) * 1e6),
            end_us: Math.round(Number(end) * 1e6),
          }),
      fast_max_age_ms: fastAge,
      battery_max_age_ms: batteryAge,
      clock_corrections: corrections,
    });
    if (legacyRoles || row?.action !== 'timeline') return;
    setPreviewChoice({ id: row.id, settings });
    setSegments([]);
    setEvidence(
      inspection?.flight_events?.intervals.find((v) => v.index === automatic)?.evidence ?? '',
    );
  }
  return (
    <section className="panel" hidden={hidden}>
      <h2>보존 구간과 시간축 조정</h2>
      <p>부팅 후 시각을 기준으로 공백 전후를 분리하고 11초 이상 기록된 구간을 찾습니다.</p>
      <label className="form-label">
        검사한 로그
        <select className="text-input" value={log} onChange={(e) => onSourceLog(e.target.value)}>
          <option value="">선택</option>
          {inspected.map((j) => (
            <option key={j.id} value={j.id}>
              {jobLabel(j)}
            </option>
          ))}
        </select>
      </label>
      {!inspected.length && <p>가져오기와 검사가 끝난 비행 로그가 여기에 표시됩니다.</p>}
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
                  비행 {v.flight} · {(v.start_us / 1e6).toFixed(1)}–{(v.end_us / 1e6).toFixed(1)}초
                  · IMU {v.imu_instance ?? '프로파일 선택값'}
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
          value={profile?.version ?? ''}
          onChange={(e) => setVersion(e.target.value)}
          disabled={!source || !compatible.length}
        >
          <option value="">선택</option>
          {compatible.map((p) => (
            <option key={p.version} value={p.version}>
              {p.profile.name} · {p.version.slice(0, 8)}
            </option>
          ))}
        </select>
      </label>
      {source && (
        <>
          {!compatible.length && <p>이 로그의 구간 검사에 사용할 수 있는 프로파일이 없습니다.</p>}
          {onCreateProfile && (
            <button className="button small" onClick={() => onCreateProfile(log)}>
              이 로그로 프로파일 만들기
            </button>
          )}
          {profile && !profile.eligibility.profile_complete && (
            <p>구간은 검사할 수 있습니다. 저장 전 프로파일의 필수 입력을 확인하세요.</p>
          )}
        </>
      )}
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
          조정하세요. 20–1000 ms 범위이며 두 신호 모두 공백 보충 상한은 1000 ms입니다.
        </p>
        <label className="form-label">
          운동·명령 관측 유지 (ms)
          <input
            className="text-input"
            type="number"
            min="20"
            max="1000"
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
            max="1000"
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
        disabled={disabled || !log || !profile || !validRange}
        onClick={() => void inspectRange()}
      >
        사용 가능 구간 찾기{corrected ? ' · 시계 보정 적용' : ''}
      </button>
      {legacyRoles && (
        <label className="form-label">
          완료된 구간 검사
          <select
            className="text-input"
            value={previewId}
            disabled={!previews.length}
            onChange={(e) => {
              setPreviewChoice({ id: e.target.value, settings });
              setSegments([]);
              setEvidence(
                jobs.find((j) => j.id === e.target.value)?.result?.interval_evidence ?? '',
              );
            }}
          >
            <option value="">선택</option>
            {previews.map((j) => (
              <option key={j.id} value={j.id}>
                {jobLabel(j)} · {j.id.slice(0, 8)}
              </option>
            ))}
          </select>
        </label>
      )}
      {!legacyRoles && (previewJob?.status === 'queued' || previewJob?.status === 'running') && (
        <p role="status">선택한 설정으로 구간을 검사하고 있습니다.</p>
      )}
      {!legacyRoles && previewJob?.error && <p role="alert">{previewJob.error}</p>}
      {preview && (
        <>
          <TimingResult timing={preview} />
          <p>저장할 구간을 선택하세요. 시간이나 입력 설정을 바꾸면 구간을 다시 검사합니다.</p>
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
      {preview && (
        <>
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
              {legacyRoles && <option value="development">학습 중 모델 선택</option>}
              <option value="calibration">임계값 보정 전용</option>
            </select>
          </label>
          <label className="form-label">
            건강 상태
            <select
              className="text-input"
              value={health}
              onChange={(e) => setHealth(e.target.value)}
            >
              <option value="unknown">모름</option>
              <option value="confirmed_normal">정상 확인</option>
              <option value="abnormal">이상 확인</option>
            </select>
            {role !== 'evaluation' && <small>이 용도에는 확인된 정상 비행이 필요합니다.</small>}
          </label>
          <label className="form-label">
            건강 상태 확인 근거 {health !== 'unknown' && '(필수)'}
            <input
              className="text-input"
              value={healthEvidence}
              onChange={(e) => setHealthEvidence(e.target.value)}
            />
          </label>
          <button
            className="button primary"
            disabled={
              disabled ||
              !profile?.eligibility.profile_complete ||
              !segments.length ||
              !evidence.trim() ||
              !validHealth
            }
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
        </>
      )}
    </section>
  );
}
