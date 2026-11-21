import { useEffect, useState } from 'react';
import { api, post } from './api';
import type { Profile } from './local-flight-types';
import type { LocalState } from './useLocalAI';

function label(id: string) {
  return id
    .replace('whole-run-', '')
    .replace('lstm-', 'LSTM · ')
    .replace('transformer-', 'Transformer · ')
    .replace('x', ' × ');
}

function WholeRunResults({ state }: { state: LocalState }) {
  const summary = state.whole_summary;
  return (
    <section className="panel local-live" aria-label="전체 런 모델 결과">
      <div className="panel-heading">
        <h2>전체 런 모델 4종</h2>
      </div>
      <p role={state.status === 'error' ? 'alert' : 'status'}>{state.message}</p>
      {summary && (
        <p>
          {summary.duration_s.toFixed(1)}초 · 유효 프레임 {summary.valid_frames.toLocaleString()} /{' '}
          {summary.frames.toLocaleString()}
        </p>
      )}
      <div className="whole-run-model-grid">
        {state.whole_run_models?.map((id) => {
          const result = summary?.models.find((model) => model.model_id === id);
          const verdict = !result
            ? '비행 종료 후 분석'
            : result.alarm === null
              ? '판정 불가'
              : result.alarm
                ? '이상 의심'
                : '정상 범위';
          return (
            <section className="panel provider-card" key={id}>
              <h3>{label(id)}</h3>
              <div className={`provider-status ${result?.alarm ? 'suspect' : ''}`}>{verdict}</div>
              {result && result.score !== null && (
                <p>
                  점수 {result.score.toFixed(3)} · 기준 {result.threshold.toFixed(3)}
                  <br />
                  채점 범위 {((result.coverage ?? 0) * 100).toFixed(1)}%
                </p>
              )}
            </section>
          );
        })}
      </div>
    </section>
  );
}

export default function WholeRunLive({
  runId,
  live,
  state,
  onState,
}: {
  runId: string;
  live: boolean;
  state: LocalState;
  onState: (state: LocalState) => void;
}) {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [version, setVersion] = useState('');
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
        <section className="panel local-live" aria-label="전체 런 모델 연결">
          <div className="panel-heading">
            <h2>PX4 · 전체 런 모델 연결</h2>
          </div>
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
                onChange={(e) => setVersion(e.target.value)}
              >
                <option value="">PX4 X500 프로파일 선택</option>
                {profiles.map((p) => (
                  <option value={p.version} key={p.version}>
                    {p.profile.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="form-label">
              연결 대상
              <select
                className="text-input"
                value={scope}
                onChange={(e) => setScope(e.target.value)}
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
        </section>
      )}
      {state.status !== 'unconfigured' && <WholeRunResults state={state} />}
    </>
  );
}
