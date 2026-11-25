import LocalRunResult from './LocalRunResult';
import { lazy, Suspense, useState } from 'react';
import type { LocalState as State } from './useLocalAI';
import { localStatus as statusNames } from './local-insights';
import { api, post } from './api';
import type { Profile } from './local-flight-types';
import { DecisionResult } from './LocalPreparation';
import { numeric } from './types';
import WholeRunLive, { WholeRunCards } from './WholeRunLive';
import { FlightAnalysisPanel, FlightPostflight } from './FlightAnalysisViews';
import {
  CandidatePicker,
  CandidateReplayControls,
  type CandidateReplay,
} from './V02FlightCandidates';
import './local-flight.css';

const BufferedFlightChart = lazy(() => import('./BufferedFlightChart'));

type Model = { id: string; task: string; label: string };
type LiveProps = {
  runId: string;
  live: boolean;
  manage: () => void;
  state: State;
  onState: (state: State) => void;
  recordedDurationMs?: number;
};

export default function LocalLive(props: LiveProps) {
  if (props.state.mode === 'px4_whole_run') return <WholeRunLive key={props.runId} {...props} />;
  return <LegacyLocalLive {...props} />;
}

function LegacyLocalLive({ runId, live, manage, state, onState, recordedDurationMs }: LiveProps) {
  const [models, setModels] = useState<Model[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [version, setVersion] = useState('');
  const [candidate, setCandidate] = useState('');
  const [comparison, setComparison] = useState<CandidateReplay | null>(null);
  const [forecast, setForecast] = useState('forecast_lstm_h64_l2');
  const [postModel, setPostModel] = useState('run_health_lstm_h64_l2');
  const [system, setSystem] = useState(1),
    [component, setComponent] = useState(1);
  const [scope, setScope] = useState('real_aircraft');
  const [imu, setImu] = useState('HIGHRES_IMU'),
    [imuId, setImuId] = useState(0);
  const [autoFinish, setAutoFinish] = useState(false);
  const [buffered, setBuffered] = useState(false);
  const parallel = state.buffered_parallel === true || (state.forecast_models?.length ?? 0) > 1;
  const useBuffered = parallel || buffered;
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
      parallel ? Promise.resolve({ models: [] }) : api<{ models: Model[] }>('/local-ai/catalog'),
      api<{ profiles: Profile[] }>('/local-ai/profiles'),
    ]);
    setModels(c.models);
    setProfiles(p.profiles);
  };
  async function start() {
    setBusy(true);
    setError('');
    try {
      onState(
        await post<State>(`/sessions/${runId}/local-ai`, {
          profile_version: version,
          ...(parallel && candidate ? { candidate_job: candidate } : {}),
          forecast_model: forecast,
          post_model: postModel,
          source_system: system,
          source_component: component,
          scope,
          imu_message: useBuffered ? 'HIGHRES_IMU' : imu,
          imu_id: imuId,
          mode: useBuffered ? 'px4_buffered' : undefined,
          use_active: active,
          auto_rates: autoRates,
          auto_finish_disarm: !useBuffered && autoFinish,
        }),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  const comparisonSummary = comparison?.result?.whole_summary;
  const shownState = comparison
    ? { ...state, whole_summary: comparisonSummary ?? undefined, whole_run_models: undefined }
    : state;
  const profileId = profiles.find((profile) => profile.version === version)?.profile.id;
  return (
    <FlightAnalysisPanel
      status={statusNames[state.status] ?? state.status}
      action={
        live && (
          <button className="button small" onClick={manage}>
            기체 프로파일 관리
          </button>
        )
      }
    >
      {error && (
        <p className="notice error" role="alert">
          {error}
        </p>
      )}
      {parallel && state.status !== 'unconfigured' && (
        <p>
          {(comparison?.candidate_job ?? state.candidate_job)
            ? `후보 ${(comparison?.candidate_job ?? state.candidate_job)?.slice(0, 8)} · 파인튜닝 모델 + 전용 보정 기준`
            : '원본 모델 · 기존 판정 기준'}
        </p>
      )}
      {parallel &&
        Object.values(state.buffered_model_states ?? {}).some(
          (model) => model.scored_frames > 0,
        ) && (
          <p>
            {live ? '현재 경보' : '비행 중 경보'} ·{' '}
            {
              Object.values(state.buffered_model_states ?? {}).filter((model) =>
                live ? model.alarm : model.alarm_frames > 0,
              ).length
            }
            개 모델
          </p>
        )}
      {!live && state.status === 'finished' && state.mode === 'px4_buffered' && parallel && (
        <CandidateReplayControls runId={runId} choose={setComparison} selection={comparison?.id} />
      )}
      {live && state.status === 'unconfigured' && (
        <details
          onToggle={(e) => {
            if (e.currentTarget.open) void refresh().catch((e) => setError(String(e)));
          }}
        >
          <summary>이 비행의 모델 선택</summary>
          {state.buffered_available && !parallel && (
            <label>
              <input
                type="checkbox"
                checked={buffered}
                onChange={(e) => setBuffered(e.target.checked)}
              />{' '}
              PX4 MAVLink · 10초 버퍼 모델 사용
            </label>
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
            {parallel && profileId && (
              <CandidatePicker
                profile={profileId}
                scope={scope}
                value={candidate}
                onChange={setCandidate}
              />
            )}
            {!useBuffered && (
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
            )}
            {!useBuffered && (
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
            {!useBuffered && (
              <label className="form-label">
                가속도 메시지
                <select className="text-input" value={imu} onChange={(e) => setImu(e.target.value)}>
                  {['HIGHRES_IMU', 'SCALED_IMU', 'SCALED_IMU2', 'SCALED_IMU3'].map((m) => (
                    <option key={m}>{m}</option>
                  ))}
                </select>
              </label>
            )}
            {(useBuffered || imu === 'HIGHRES_IMU') && (
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
          {!useBuffered && (
            <label>
              <input
                type="checkbox"
                checked={active}
                onChange={(e) => setActive(e.target.checked)}
              />{' '}
              이 기체에서 채택한 파인튜닝 모델 사용 · 없으면 원본
            </label>
          )}
          {!useBuffered && (
            <label>
              <input
                type="checkbox"
                checked={autoRates}
                onChange={(e) => setAutoRates(e.target.checked)}
              />{' '}
              필수 메시지 주기 자동 설정 · 기체로 주기 설정 요청 전송
            </label>
          )}
          {!useBuffered && (
            <label>
              <input
                type="checkbox"
                checked={autoFinish}
                onChange={(e) => setAutoFinish(e.target.checked)}
              />{' '}
              착륙 후 비행 데이터가 10초 끊기면 자동 종료
            </label>
          )}
          {!useBuffered && (
            <small>
              시동 해제를 확인한 뒤 유효한 비행 데이터가 10초 연속 들어오지 않으면 기록을
              종료합니다. 기체가 지상에서도 데이터를 계속 보내면 비행 종료 버튼으로 마감하세요.
            </small>
          )}
          {useBuffered && <p>100ms 간격으로 확정한 프레임을 10초 지연으로 평가합니다.</p>}
          <p>
            출력 port·배터리 ID·모터 채널과 출력 범위는 선택한 프로파일을 사용합니다. 필요한
            메시지가 없으면 예측을 보류합니다.
          </p>
          <button className="button primary" disabled={busy || !version} onClick={start}>
            {busy ? '연결 중…' : parallel ? '비행 중 모델 4종 시작' : '로컬 AI 시작'}
          </button>
        </details>
      )}
      {state.finish_reason === 'confirmed_disarm' && (
        <p>기체 시동 해제와 비행 데이터 수신 중단 10초를 확인해 기록을 종료했습니다.</p>
      )}
      <p role="status">{state.message}</p>
      {state.scope && (
        <p>
          {state.scope === 'simulation_test' ? '시뮬레이터 통신 시험' : '실기체 입력'} ·{' '}
          {parallel ? '비행 중 모델 4종' : state.forecast_model} ·{' '}
          {state.mode === 'px4_buffered'
            ? '10초 지연 판정'
            : state.adapted_sha256
              ? `파인튜닝 ${state.adapted_sha256.slice(0, 8)}`
              : '원본 모델'}
        </p>
      )}
      {!parallel && state.status !== 'finished' && state.status !== 'analyzing' && (
        <DecisionResult decision={state.decision} />
      )}
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
                  <td>
                    {r.verified
                      ? r.ack === null
                        ? '주기·수신 확인 · ACK 미확인'
                        : '확인됨'
                      : r.state === 'readback_confirmed'
                        ? '주기 확인 · 수신률 확인 중 · ACK 미확인'
                        : `${r.state} · ACK ${r.ack ?? '—'}`}
                  </td>
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
          <p>
            관측 시작 후 {(state.prediction.origin_us / 1e6).toFixed(1)}초 기준 ·{' '}
            {state.mode === 'px4_buffered' ? '1초 뒤 모델값 · 10초 지연 표시' : '1초 뒤 예측'}
          </p>
          <div className="local-live-values">
            {['Roll 각속도', 'Pitch 각속도', 'Yaw 각속도', '팩 전압'].map((name, i) => (
              <div key={name}>
                <span>{name}</span>
                <strong>
                  {numeric(state.prediction!.values[i], i === 3 ? 2 : 3)}{' '}
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
      {state.buffered_summary && !parallel && (
        <p>
          예측 {numeric(state.buffered_summary.forecasts, 0)}회 · 판정{' '}
          {numeric(state.buffered_summary.scored_frames, 0)}건 · 이상 표시{' '}
          {numeric(state.buffered_summary.alarm_frames, 0)}건
        </p>
      )}
      {(parallel ||
        Boolean(state.whole_run_models?.length) ||
        state.whole_available ||
        state.whole_summary) && (
        <FlightPostflight>
          {shownState.whole_summary ? (
            <WholeRunCards state={shownState} recordedDurationMs={recordedDurationMs} />
          ) : (
            <>
              {state.whole_status === 'running' && (
                <p role="status">비행 종료 후 모델을 계산하고 있습니다.</p>
              )}
              {state.whole_error && <p role="alert">{state.whole_error}</p>}
              {state.whole_available ? (
                <button
                  className="button small"
                  disabled={busy}
                  onClick={async () => {
                    setBusy(true);
                    setError('');
                    try {
                      onState(await post<State>(`/sessions/${runId}/local-ai/postflight`));
                    } catch (failure) {
                      setError(String(failure));
                    } finally {
                      setBusy(false);
                    }
                  }}
                >
                  {busy ? '비행 후 판정 계산 중…' : '이 비행의 종료 후 판정 계산'}
                </button>
              ) : (
                state.whole_status !== 'running' && (
                  <p>비행 종료 · 결과 분석을 누르면 전체 런 모델 4종의 판정이 표시됩니다.</p>
                )
              )}
            </>
          )}
        </FlightPostflight>
      )}
      {state.mode === 'px4_buffered' ? (
        <Suspense fallback={<p role="status">그래프 기능을 불러오는 중…</p>}>
          <BufferedFlightChart
            key={comparison?.id ?? runId}
            runId={runId}
            comparison={comparison?.id}
            live={!['finished', 'interrupted', 'error'].includes(state.status)}
          />
        </Suspense>
      ) : parallel ? (
        <section className="buffered-chart">
          <h3>실측·예측·이상 점수 그래프</h3>
          <p role="status">모델 연결 후 그래프가 표시됩니다.</p>
        </section>
      ) : null}
      {state.result && (
        <div>
          <h3>비행 후 분석 결과</h3>
          <p>{state.result.message}</p>
          <LocalRunResult
            summary={state.insights?.report?.run_summary ?? state.result.run_summary}
            analyses={
              state.insights?.report?.analyses ?? state.result.analyses.map((r) => r.analysis)
            }
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
    </FlightAnalysisPanel>
  );
}
