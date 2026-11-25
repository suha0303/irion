import { useEffect, useState } from 'react';
import { api, post } from './api';
import type { Profile } from './local-flight-types';
import type { LocalState } from './useLocalAI';
import { CandidatePicker } from './V02FlightCandidates';
import { FlightAnalysisPanel, FlightPostflight, FlightVerdictCards } from './FlightAnalysisViews';

function label(id: string) {
  return id
    .replace('whole-run-', '')
    .replace('lstm-', 'LSTM · ')
    .replace('transformer-', 'Transformer · ')
    .replace('x', ' × ');
}

export function WholeRunCards({
  state,
  recordedDurationMs,
}: {
  state: LocalState;
  recordedDurationMs?: number;
}) {
  const summary = state.whole_summary;
  if (!summary) return <p>비행 종료 후 판정이 표시됩니다.</p>;
  return (
    <>
      <p>
        {recordedDurationMs != null && recordedDurationMs > 0
          ? `전체 수신 기록 ${(recordedDurationMs / 1000).toFixed(1)}초 · `
          : '전체 수신 범위 미확인 · '}
        판정 입력 {summary.duration_s.toFixed(1)}초 · 유효 프레임{' '}
        {summary.valid_frames.toLocaleString()} / {summary.frames.toLocaleString()}
      </p>
      <FlightVerdictCards
        items={summary.models.map((model) => ({
          key: model.model_id,
          model: label(model.model_id),
          candidate: Boolean(model.candidate_job),
          alarm: model.alarm,
          score: model.score,
          threshold: model.threshold,
          detail:
            model.score == null
              ? undefined
              : model.coverage == null
                ? undefined
                : `판정 입력 내 채점률 ${(model.coverage * 100).toFixed(1)}%`,
        }))}
      />
    </>
  );
}

export default function WholeRunLive({
  runId,
  live,
  state,
  onState,
  recordedDurationMs,
}: {
  runId: string;
  live: boolean;
  state: LocalState;
  onState: (state: LocalState) => void;
  recordedDurationMs?: number;
}) {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [version, setVersion] = useState('');
  const [candidate, setCandidate] = useState('');
  const [scope, setScope] = useState('real_aircraft');
  const [system, setSystem] = useState(1);
  const [component, setComponent] = useState(1);
  const [imuId, setImuId] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!live || state.status !== 'unconfigured') return;
    let active = true;
    void api<{ profiles: Profile[] }>('/local-ai/profiles')
      .then(({ profiles }) => {
        if (active)
          setProfiles(
            profiles.filter(
              ({ profile, eligibility }) =>
                eligibility.profile_complete &&
                profile.autopilot === 'px4' &&
                profile.series_cells === 4 &&
                profile.capacity_ah === 12,
            ),
          );
      })
      .catch((failure) => {
        if (active) setError(String(failure));
      });
    return () => {
      active = false;
    };
  }, [live, state.status]);
  async function start() {
    setBusy(true);
    setError('');
    try {
      onState(
        await post<LocalState>(`/sessions/${runId}/local-ai`, {
          mode: 'px4_whole_run',
          profile_version: version,
          ...(candidate ? { candidate_job: candidate } : {}),
          scope,
          source_system: system,
          source_component: component,
          imu_message: 'HIGHRES_IMU',
          imu_id: imuId,
        }),
      );
    } catch (failure) {
      setError(String(failure));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      {live && state.status === 'unconfigured' && (
        <FlightAnalysisPanel status="모델 연결 대기">
          {error && (
            <p className="notice error" role="alert">
              {error}
            </p>
          )}
          <div className="local-live-settings">
            <label className="form-label">
              기체 프로파일
              <select
                className="text-input"
                value={version}
                onChange={(e) => {
                  setVersion(e.target.value);
                  setCandidate('');
                }}
              >
                <option value="">PX4 X500 프로파일 선택</option>
                {profiles.map((p) => (
                  <option value={p.version} key={p.version}>
                    {p.profile.name}
                  </option>
                ))}
              </select>
            </label>
            {version && (
              <CandidatePicker
                profile={profiles.find((profile) => profile.version === version)?.profile.id}
                scope={scope}
                models={state.whole_run_models}
                value={candidate}
                onChange={setCandidate}
              />
            )}
            <label className="form-label">
              연결 대상
              <select
                className="text-input"
                value={scope}
                onChange={(e) => {
                  setScope(e.target.value);
                  setCandidate('');
                }}
              >
                <option value="real_aircraft">실기체</option>
                <option value="simulation_test">시뮬레이터 통신 시험</option>
              </select>
            </label>
            <label className="form-label">
              MAVLink system ID
              <input
                className="text-input"
                type="number"
                min={1}
                max={255}
                value={system}
                onChange={(e) => setSystem(Number(e.target.value))}
              />
            </label>
            <label className="form-label">
              Component ID
              <input
                className="text-input"
                type="number"
                min={1}
                max={255}
                value={component}
                onChange={(e) => setComponent(Number(e.target.value))}
              />
            </label>
            <label className="form-label">
              HIGHRES_IMU ID
              <input
                className="text-input"
                type="number"
                min={0}
                max={255}
                value={imuId}
                onChange={(e) => setImuId(Number(e.target.value))}
              />
            </label>
          </div>
          <button className="button primary" onClick={start} disabled={busy || !version}>
            {busy ? '연결 중…' : '전체 런 모델 4종 연결'}
          </button>
        </FlightAnalysisPanel>
      )}
      {state.status !== 'unconfigured' && (
        <FlightAnalysisPanel>
          <p role={state.status === 'error' ? 'alert' : 'status'}>{state.message}</p>
          <FlightPostflight>
            <WholeRunCards state={state} recordedDurationMs={recordedDurationMs} />
          </FlightPostflight>
        </FlightAnalysisPanel>
      )}
    </>
  );
}
