import { useState } from 'react';
import { jobFilename, type Job, type V02Model, type V02Role } from './local-flight-types';
import ModelSelection, { modelIds } from './V02ModelSelection';

const roles = [
  ['adaptation', '1. 학습 로그', '정상 확인 로그로 출력층을 적응합니다.'],
  ['calibration', '2. 보정 로그', '별도 정상 확인 로그로 판정 기준을 보정합니다.'],
  ['evaluation', '3. 평가 로그', '독립 로그로 예측 오차와 경보 결과를 비교합니다.'],
] as const;

type Choice = { job_id: string; date: string };
export type V02Selection = Record<V02Role, Choice[]>;

function label(job: Job) {
  return [
    jobFilename(job),
    job.request?.profile?.name,
    job.result?.health === 'abnormal' ? '이상 확인' : '정상 확인',
    job.id.slice(0, 8),
  ]
    .filter(Boolean)
    .join(' · ');
}

export default function V02Adaptation({
  jobs,
  models,
  disabled,
  goToLogs,
  submit,
}: {
  jobs: Job[];
  models: V02Model[];
  disabled: boolean;
  goToLogs: () => void;
  submit: (logs: V02Selection, models: string[]) => Promise<void>;
}) {
  const [selectedModel, setSelectedModel] = useState('inflight-lstm-128x4');
  const ids = modelIds(models, selectedModel);
  const [dates, setDates] = useState<Record<string, string>>({});
  const eligible = jobs.filter(
    (job) =>
      job.action === 'prepare' &&
      job.status === 'complete' &&
      Boolean(job.result?.source_sha256) &&
      (job.result?.health === 'confirmed_normal' ||
        (job.result?.role === 'evaluation' && job.result?.health === 'abnormal')),
  );
  const logs: V02Selection = { adaptation: [], calibration: [], evaluation: [] };
  const sources = new Map<string, string>();
  const days = new Map<string, V02Role>();
  const aircraft = new Set<string>();
  let conflict = '';
  for (const [role] of roles) {
    for (const job of eligible) {
      const source = job.result?.source_sha256;
      if (!source || job.result?.role !== role || !(job.id in dates)) continue;
      const day = dates[job.id];
      const grouping = `${role}/${day}`;
      if (sources.has(source) && sources.get(source) !== grouping) {
        conflict = '같은 원본에는 하나의 날짜와 역할만 지정하세요.';
      }
      if (day && days.has(day) && days.get(day) !== role) {
        conflict = '같은 수집 날짜의 자료는 한 역할에 묶으세요.';
      }
      sources.set(source, grouping);
      if (day) days.set(day, role);
      if (job.request?.profile?.id) aircraft.add(job.request.profile.id);
      logs[role].push({ job_id: job.id, date: day });
    }
  }
  if (!conflict && aircraft.size > 1) conflict = '한 작업에는 같은 기체의 로그만 선택하세요.';
  const ready = roles.every(([role]) => logs[role].length && logs[role].every((row) => row.date));

  return (
    <section className="panel v02-adaptation" aria-label="V0.2 기체별 파인튜닝과 보정">
      <h3>V0.2 기체별 파인튜닝과 보정</h3>
      <p>로컬 CPU 출력층 적응 · 정상 판정 점수 99백분위 보정</p>
      <p>같은 기체 프로파일의 구간을 선택하고 실제 수집 날짜를 입력하세요.</p>
      <button className="button small" onClick={goToLogs}>
        로그 준비 열기
      </button>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (ready && ids.length && !disabled && !conflict) void submit(logs, ids);
        }}
      >
        <ModelSelection models={models} selected={selectedModel} onChange={setSelectedModel} />
        <div className="v02-role-grid">
          {roles.map(([role, title, description]) => {
            const options = eligible.filter((job) => job.result?.role === role);
            return (
              <fieldset key={role}>
                <legend>{title}</legend>
                <p>{description}</p>
                {options.length === 0 && <small>이 역할로 준비한 구간이 없습니다.</small>}
                {options.map((job) => (
                  <div className="v02-log-choice" key={job.id}>
                    <label>
                      <input
                        type="checkbox"
                        checked={job.id in dates}
                        onChange={(event) => {
                          const checked = event.target.checked;
                          setDates((previous) => {
                            const next = { ...previous };
                            if (checked) next[job.id] = '';
                            else delete next[job.id];
                            return next;
                          });
                        }}
                      />{' '}
                      {label(job)}
                    </label>
                    {job.id in dates && (
                      <label className="form-label">
                        수집 날짜
                        <input
                          className="text-input"
                          type="date"
                          required
                          defaultValue={dates[job.id]}
                          onInput={(event) => {
                            const day = event.currentTarget.value;
                            setDates((previous) => ({
                              ...previous,
                              [job.id]: day,
                            }));
                          }}
                        />
                      </label>
                    )}
                  </div>
                ))}
              </fieldset>
            );
          })}
        </div>
        {conflict && (
          <p className="notice warning" role="alert">
            {conflict}
          </p>
        )}
        <p>
          새 모델과 판정 기준은 검증 대기 후보로 저장됩니다. 앱의 분석에는 자동 적용되지 않습니다.
        </p>
        <button
          className="button primary"
          disabled={disabled || !ready || !ids.length || Boolean(conflict)}
        >
          {ids.length > 1 ? `${ids.length}종 일괄 파인튜닝·보정` : '후보 만들고 독립 평가'}
        </button>
      </form>
    </section>
  );
}
