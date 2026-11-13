import LocalRunResult from './LocalRunResult';
import { useEffect, useState } from 'react';
import { api, post } from './api';
import type { Analysis, Decision, Profile, RunSummary } from './local-flight-types';
import { DecisionResult } from './LocalPreparation';
import './local-flight.css';

type Model = { id: string; task: string; label: string };
type Rates = {
  state: string;
  error?: string;
  rows: {
    name: string;
    hz: number;
    observed_hz: number;
    ack: number | null;
    readback_us: number | null;
    verified: boolean;
    state: string;
  }[];
};
type State = {
  finish_reason?: string;
  decision?: Decision;
  rates?: Rates;
  status: string;
  message?: string;
  scope?: string;
  forecast_model?: string;
  adapted_sha256?: string;
  missing_features?: string[];
  prediction?: { origin_us: number; values: number[]; units: string[] } | null;
  result?: {
    run_summary?: RunSummary;
    message: string;
    analyses: { manifest: string; analysis: Analysis }[];
  };
};
const statusNames: Record<string, string> = {
  unconfigured: '설정 대기',
  starting: '모델 준비',
  warming_up: '이력 수집',
  waiting: '연결 확인',
  predicting: '실시간 예측 중',
  stale: '수신·처리 지연',
  analyzing: '비행 후 분석 중',
  finished: '비행 후 결과',
  interrupted: '중단됨',
  error: '오류',
};

export default function LocalLive({
  runId,
  live,
  manage,
}: {
  runId: string;
  live: boolean;
  manage: () => void;
}) {
  const [state, setState] = useState<State>({ status: 'unconfigured' });
  const [models, setModels] = useState<Model[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [version, setVersion] = useState('');
  const [forecast, setForecast] = useState('forecast_lstm_h64_l2');
  const [postModel, setPostModel] = useState('run_health_lstm_h64_l2');
  const [system, setSystem] = useState(1),
    [component, setComponent] = useState(1);
  const [scope, setScope] = useState('real_aircraft');
  const [imu, setImu] = useState('HIGHRES_IMU'),
    [imuId, setImuId] = useState(0);
  const [autoFinish, setAutoFinish] = useState(false);
  const [active, setActive] = useState(true),
    [autoRates, setAutoRates] = useState(true);
  const [evidence, setEvidence] = useState(''),
    [admitRole, setAdmitRole] = useState('calibration'),
    [admitHealth, setAdmitHealth] = useState('unknown'),
    [admitted, setAdmitted] = useState('');
  const [error, setError] = useState(''),
    [busy, setBusy] = useState(false);
  const refresh = async () => {
    const [c, p] = await Promise.all([
      api<{ models: Model[] }>('/local-ai/catalog'),
      api<{ profiles: Profile[] }>('/local-ai/profiles'),
    ]);
    setModels(c.models);
    setProfiles(p.profiles);
  };
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    setState({ status: 'unconfigured' });
    setError('');
    const poll = async () => {
      try {
        const next = await api<State>(`/sessions/${runId}/local-ai`);
        if (!cancelled) setState(next);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
      if (!cancelled) timer = setTimeout(poll, 1000);
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [runId]);
  async function start() {
    setBusy(true);
    setError('');
    try {
      setState(
        await post<State>(`/sessions/${runId}/local-ai`, {
          profile_version: version,
          forecast_model: forecast,
          post_model: postModel,
          source_system: system,
          source_component: component,
          scope,
          imu_message: imu,
          imu_id: imuId,
          use_active: active,
          auto_rates: autoRates,
          auto_finish_disarm: autoFinish,
        }),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="panel local-live" aria-label="실기체 로컬 AI">
      <div className="panel-heading">
        <h2>기체별 로컬 AI · {statusNames[state.status] ?? state.status}</h2>
        <button className="button small" onClick={manage}>
          프로파일·파인튜닝 관리
        </button>
      </div>
      {error && (
        <p className="notice error" role="alert">
          {error}
        </p>
      )}
      {live && state.status === 'unconfigured' && (
        <details
          onToggle={(e) => {
            if (e.currentTarget.open) void refresh().catch((e) => setError(String(e)));
          }}
        >
          <summary>기체와 모델 연결</summary>
          <div className="local-live-settings">
            <label className="form-label">
              기체 프로파일
              <select
                className="text-input"
                value={version}
                onChange={(e) => setVersion(e.target.value)}
              >
                <option value="">등록된 버전 선택</option>
                {profiles.map((p) => (
                  <option
                    key={p.version}
                    value={p.version}
                    disabled={!p.eligibility.profile_complete}
                  >
                    {p.profile.name} · {p.version.slice(0, 8)}
                    {!p.eligibility.profile_complete ? ' · 정보 부족' : ''}
                  </option>
                ))}
              </select>
            </label>
            <label className="form-label">
              비행 중 예측 · 8종
              <select
                className="text-input"
                value={forecast}
                onChange={(e) => setForecast(e.target.value)}
              >
                {models
                  .filter((m) => m.task === 'forecast')
                  .map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
              </select>
            </label>
            <label className="form-label">
              비행 종료 후 · 8종
              <select
                className="text-input"
                value={postModel}
                onChange={(e) => setPostModel(e.target.value)}
              >
                {models
                  .filter((m) => m.task === 'run_health')
                  .map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
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
              component ID
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
              가속도 메시지
              <select className="text-input" value={imu} onChange={(e) => setImu(e.target.value)}>
                {['HIGHRES_IMU', 'SCALED_IMU', 'SCALED_IMU2', 'SCALED_IMU3'].map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            </label>
            {imu === 'HIGHRES_IMU' && (
              <label className="form-label">
                MAVLink IMU ID · 로그 인스턴스와 별도
                <input
                  className="text-input"
                  type="number"
                  min={0}
                  max={255}
                  value={imuId}
                  onChange={(e) => setImuId(Number(e.target.value))}
                />
              </label>
            )}
          </div>
          <label>
            <input type="checkbox" checked={active} onChange={(e) => setActive(e.target.checked)} />{' '}
            이 기체에서 채택한 파인튜닝 모델 사용 · 없으면 원본
          </label>
          <label>
            <input
              type="checkbox"
              checked={autoRates}
              onChange={(e) => setAutoRates(e.target.checked)}
            />{' '}
            필수 메시지 주기 자동 설정 · 기체로 주기 설정 요청 전송
          </label>
          <label>
            <input
              type="checkbox"
              checked={autoFinish}
              onChange={(e) => setAutoFinish(e.target.checked)}
            />{' '}
            시동 해제 확인 시 수집 종료·비행 후 분석 자동 실행
          </label>
          <small>
            연결 후 시동 상태를 관측한 뒤, 2초 이상 연속된 시동 해제 HEARTBEAT 3개를 확인합니다.
            통신 단절만으로 종료하지 않습니다.
          </small>
          <p>
            출력 port·배터리 ID·모터 채널과 출력 범위는 선택한 프로파일을 사용합니다. 필요한
            메시지가 없으면 예측을 보류합니다.
          </p>
          <button className="button primary" disabled={busy || !version} onClick={start}>
            {busy ? '연결 중…' : '로컬 AI 시작'}
          </button>
        </details>
      )}
      {state.finish_reason === 'confirmed_disarm' && (
        <p>기체 시동 해제 확인으로 수집을 종료했습니다.</p>
      )}
      <p role="status">{state.message}</p>
      {state.scope && (
        <p>
          {state.scope === 'simulation_test' ? '시뮬레이터 통신 시험' : '실기체 입력'} ·{' '}
          {state.forecast_model} ·{' '}
          {state.adapted_sha256 ? `파인튜닝 ${state.adapted_sha256.slice(0, 8)}` : '원본 모델'}
        </p>
      )}
      <DecisionResult decision={state.decision} />
      {state.rates && (
        <details>
          <summary>
            메시지 주기 설정 ·{' '}
            {state.rates.state === 'waiting_heartbeat'
              ? '기체 연결 대기 · 연결되면 자동 요청'
              : state.rates.state}
          </summary>
          {state.rates.error && <p role="alert">{state.rates.error}</p>}
          <p>ACK·설정 조회·실수신률을 함께 확인합니다. 신호 값의 유효성은 별도입니다.</p>
          <p>ESC_STATUS는 선택 항목입니다. FC가 전송하지 않으면 RPM은 미수신으로 표시합니다.</p>
          <table>
            <thead>
              <tr>
                <th>메시지</th>
                <th>요청 / 수신 Hz</th>
                <th>확인</th>
              </tr>
            </thead>
            <tbody>
              {state.rates.rows.map((r) => (
                <tr key={r.name}>
                  <td>{r.name}</td>
                  <td>
                    {r.hz} / {r.observed_hz.toFixed(1)}
                  </td>
                  <td>{r.verified ? '확인됨' : `${r.state} · ACK ${r.ack ?? '—'}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {live && (
            <button
              className="button small"
              onClick={async () => {
                try {
                  await post(`/sessions/${runId}/message-rates`);
                } catch (e) {
                  setError(String(e));
                }
              }}
            >
              주기 설정 요청
            </button>
          )}
        </details>
      )}
      {state.prediction && (
        <>
          <p>관측 시작 후 {(state.prediction.origin_us / 1e6).toFixed(1)}초 기준 · 1초 뒤 예측</p>
          <div className="local-live-values">
            {['Roll 각속도', 'Pitch 각속도', 'Yaw 각속도', '팩 전압'].map((name, i) => (
              <div key={name}>
                <span>{name}</span>
                <strong>
                  {state.prediction!.values[i].toFixed(3)}{' '}
                  <small>{state.prediction!.units[i]}</small>
                </strong>
              </div>
            ))}
          </div>
        </>
      )}
      {!!state.missing_features?.length && (
        <p>미수신·오래된 입력: {state.missing_features.join(', ')}</p>
      )}
      {state.result && (
        <div>
          <h3>비행 후 분석 결과</h3>
          <p>{state.result.message}</p>
          <LocalRunResult
            summary={state.result.run_summary}
            analyses={state.result.analyses.map((r) => r.analysis)}
          />
          {state.result.analyses.map((r, i) => (
            <div key={r.manifest}>
              <button
                className="button small"
                disabled={
                  !evidence.trim() ||
                  (admitRole === 'calibration' && admitHealth !== 'confirmed_normal')
                }
                onClick={async () => {
                  try {
                    await post('/local-ai/live-prepared', {
                      run_id: runId,
                      index: i,
                      role: admitRole,
                      health: admitHealth,
                      evidence,
                    });
                    setAdmitted('등록 작업을 시작했습니다. 기체 관리의 작업 기록에서 확인하세요.');
                  } catch (e) {
                    setError(String(e));
                  }
                }}
              >
                분석 구간 {i + 1}을 준비된 비행으로 등록
              </button>
            </div>
          ))}
          {state.result.analyses.length > 0 && (
            <details>
              <summary>수신 기록을 학습·보정·평가 자료로 등록</summary>
              <p>
                같은 비행을 여러 역할에 재사용할 수 없습니다. 건강 상태와 비행 구간은 직접 확인해야
                합니다.
              </p>
              <label className="form-label">
                자료 용도
                <select
                  className="text-input"
                  value={admitRole}
                  onChange={(e) => setAdmitRole(e.target.value)}
                >
                  <option value="calibration">임계값 보정</option>
                  <option value="adaptation">적응 학습</option>
                  <option value="development">모델 선택</option>
                  <option value="evaluation">독립 평가</option>
                </select>
              </label>
              <label className="form-label">
                확인한 건강 상태
                <select
                  className="text-input"
                  value={admitHealth}
                  onChange={(e) => setAdmitHealth(e.target.value)}
                >
                  <option value="unknown">모름</option>
                  <option value="confirmed_normal">정상 확인</option>
                  <option value="abnormal">이상 확인</option>
                </select>
              </label>
              <label className="form-label">
                비행 구간·건강 상태 확인 근거
                <input
                  className="text-input"
                  value={evidence}
                  onChange={(e) => setEvidence(e.target.value)}
                />
              </label>
              <p>{admitted}</p>
            </details>
          )}
        </div>
      )}
      <p className="muted">
        ‘비행 종료 · 결과 분석’은 수신 기록을 마치고 보존 구간을 분석합니다. 착륙·시동 해제 명령은
        전송하지 않습니다. 정상 범위·이상 의심은 보정된 잔차 기준이며 실기체 탐지 성능 검증과
        구분합니다.
      </p>
    </section>
  );
}
