import { useEffect, useState } from 'react';
import { api } from './api';
import type { Job } from './local-flight-types';

type Calibration = {
  id: string;
  schema: string;
  model_id: string;
  profile_id: string;
  threshold: number;
  quantile: number | null;
  normal_flights: number;
  active: boolean;
  binding: { weight_sha256: string; clock: string };
};
export default function LocalCalibration({
  jobs,
  model,
  disabled,
  submit,
}: {
  jobs: Job[];
  model: string;
  disabled: boolean;
  submit: (path: string, body: unknown) => Promise<void>;
}) {
  const [selected, setSelected] = useState<string[]>([]),
    [references, setReferences] = useState<string[]>([]),
    [adapted, setAdapted] = useState(true);
  const [records, setRecords] = useState<Calibration[]>([]),
    [error, setError] = useState('');
  const revision = jobs.map((j) => j.id + j.status).join(',');
  useEffect(() => {
    let stopped = false;
    api<{ calibrations: Calibration[] }>('/local-ai/calibrations')
      .then((r) => {
        if (!stopped) setRecords(r.calibrations);
      })
      .catch((e) => {
        if (!stopped) setError(String(e));
      });
    return () => {
      stopped = true;
    };
  }, [revision]);
  return (
    <section className="panel">
      <h2>기체별 임계값 보정</h2>
      <p className="local-note">
        두 모델 파인튜닝에는 보정·비교가 포함됩니다. 이곳에서는 단일 모델의 판정 기준을 별도로
        만들거나 저장된 기준을 관리합니다.
      </p>
      <p>
        정상 기준용 개발 비행 1~20개와, 학습·개발·평가에 사용하지 않은 보정 비행 2~20개를
        사용합니다. 선택 모델 A: {model || '미선택'}
      </p>
      <p>
        기존 최종 평가의 점수·경보 후처리로 계산합니다. 기준값은 선택한 기체의 정상 자료에서
        생성하며 실제 오경보율을 보장하지 않습니다.
      </p>
      <h3>정상 오차 기준용 개발 비행</h3>
      {jobs
        .filter(
          (j) =>
            j.status === 'complete' &&
            j.action === 'prepare' &&
            j.result?.role === 'development' &&
            j.result?.health === 'confirmed_normal',
        )
        .map((j) => (
          <label className="local-segment" key={j.id}>
            <input
              type="checkbox"
              checked={references.includes(j.id)}
              onChange={() =>
                setReferences((old) =>
                  old.includes(j.id) ? old.filter((id) => id !== j.id) : [...old, j.id],
                )
              }
            />
            {j.id.slice(0, 8)} · {j.result?.frames}시점
          </label>
        ))}
      <h3>임계값 보정용 비행</h3>
      {jobs
        .filter(
          (j) =>
            j.status === 'complete' &&
            j.action === 'prepare' &&
            j.result?.role === 'calibration' &&
            j.result?.health === 'confirmed_normal',
        )
        .map((j) => (
          <label className="local-segment" key={j.id}>
            <input
              type="checkbox"
              checked={selected.includes(j.id)}
              onChange={() =>
                setSelected((s) =>
                  s.includes(j.id) ? s.filter((id) => id !== j.id) : [...s, j.id],
                )
              }
            />
            {j.id.slice(0, 8)} · {j.result?.frames}시점
          </label>
        ))}
      <label className="local-segment">
        <input type="checkbox" checked={adapted} onChange={(e) => setAdapted(e.target.checked)} />{' '}
        이 기체에서 채택한 파인튜닝 모델 보정 · 없으면 원본
      </label>
      <button
        className="button primary"
        disabled={
          disabled ||
          !model ||
          selected.length < 2 ||
          selected.length > 20 ||
          references.length < 1 ||
          references.length > 20
        }
        onClick={() =>
          submit('/local-ai/calibrations', {
            model_id: model,
            prepared_jobs: selected,
            reference_jobs: references,
            use_active: adapted,
          })
        }
      >
        새 보정값 만들기
      </button>
      <p>
        생성 후 아래에서 ‘판정에 사용’을 선택하세요. 기체·가중치·입력 방식이 일치하는 파일 분석과
        실시간 분석에 적용됩니다. 과거 버전을 다시 선택하거나 적용을 해제할 수 있습니다.
      </p>
      {error && <p role="alert">{error}</p>}
      {records.map((c) => (
        <article key={c.id} className="local-job">
          <strong>
            {c.profile_id} · {c.model_id}
          </strong>
          <p>
            가중치 {c.binding.weight_sha256.slice(0, 8)} ·{' '}
            {c.binding.clock === 'receipt_monotonic' ? '실시간 수신 기록' : '온보드 로그'}
            <br />
            정상 {c.normal_flights}회 · 기준 {c.threshold.toFixed(5)} ·{' '}
            {c.active ? '판정에 사용 중' : '미선택'}
            <br />
            {c.schema === 'aircraft-final-protocol/1'
              ? '공통 점수·후처리 적용'
              : '이전 단일 점수 방식 · 새 보정 권장'}
          </p>
          <button
            className="button small"
            disabled={disabled}
            onClick={() =>
              submit('/local-ai/calibrations/select', { id: c.id, enabled: !c.active })
            }
          >
            {c.active ? '적용 해제' : '판정에 사용'}
          </button>
        </article>
      ))}
    </section>
  );
}
