import { useEffect, useState } from 'react';
import { api, post } from './api';
import { FlightModelControls, FlightResultSelect } from './FlightAnalysisViews';
import type { LocalState } from './useLocalAI';

type Candidate = {
  id: string;
  profile_id: string;
  profile_name: string;
  models: string[];
  origin: string;
};

export type CandidateReplay = {
  id: string;
  candidate_job: string;
  status: string;
  error?: string;
  result?: { whole_summary: LocalState['whole_summary'] | null };
};

function useCandidates(active: boolean) {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    void api<{ candidates: Candidate[] }>('/local-ai/v02/candidates', {
      signal: controller.signal,
    }).then(
      (data) => {
        if (!controller.signal.aborted) {
          setCandidates(data.candidates);
          setError('');
        }
      },
      (failure) => {
        if (!controller.signal.aborted) setError(String(failure));
      },
    );
    return () => controller.abort();
  }, [active]);
  return { candidates, error };
}

export function CandidatePicker({
  active = true,
  profile,
  scope,
  models,
  value,
  onChange,
}: {
  active?: boolean;
  profile?: string;
  scope?: string;
  models?: string[];
  value: string;
  onChange: (id: string) => void;
}) {
  const { candidates, error } = useCandidates(active);
  const choices = candidates.filter(
    (candidate) =>
      (!profile || candidate.profile_id === profile) &&
      (!scope ||
        candidate.origin === (scope === 'real_aircraft' ? 'px4_real' : 'synthetic_unit_fixture')) &&
      (!models || models.some((model) => candidate.models.includes(model))),
  );
  const chosen = choices.find((item) => item.id === value);
  return (
    <label className="form-label">
      이 비행의 모델과 판정 기준
      <select
        className="text-input"
        value={chosen?.id ?? ''}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">원본 모델 · 기존 기준</option>
        {choices.map((item) => (
          <option key={item.id} value={item.id}>
            {item.profile_name} · 후보 {item.id.slice(0, 8)} · {item.models.length}종 · 보정 포함
          </option>
        ))}
      </select>
      {error ? (
        <small role="alert">{error}</small>
      ) : chosen ? (
        <small>
          모델과 전용 보정 기준이 함께 적용됩니다. 후보는 검증 대기입니다.
          {scope === 'real_aircraft' && ' ULog 학습 후보의 MAVLink 실기체 성능은 검증 대기입니다.'}
        </small>
      ) : (
        <small>
          {choices.length
            ? '후보를 선택하면 저장된 모델과 전용 보정 기준을 함께 사용합니다.'
            : '선택 조건에 맞는 파인튜닝 후보가 없습니다.'}
        </small>
      )}
    </label>
  );
}

type ReplayList = { profile_id: string; scope: string; replays: CandidateReplay[] };

export function CandidateReplayControls({
  runId,
  choose,
  selection,
}: {
  runId: string;
  choose: (row: CandidateReplay | null) => void;
  selection?: string;
}) {
  const endpoint = `/sessions/${runId}/local-ai/candidate-replays`;
  const [data, setData] = useState<ReplayList>();
  const [candidate, setCandidate] = useState('');
  const [pending, setPending] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function read() {
      try {
        const result = await api<ReplayList>(endpoint, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setData(result);
        const item = result.replays.find((row) => row.id === pending);
        if (item?.status === 'complete') {
          choose(item);
          setPending('');
        } else if (item && !['queued', 'running'].includes(item.status)) {
          setError(item.error ?? '후보 분석이 중단됐습니다.');
          setPending('');
        }
      } catch (failure) {
        if (!controller.signal.aborted) setError(String(failure));
      }
      if (pending && !controller.signal.aborted) timer = setTimeout(read, 2000);
    }
    void read();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [choose, endpoint, pending]);
  const completed = data?.replays.filter((row) => row.status === 'complete') ?? [];
  return (
    <FlightModelControls>
      {data && (
        <CandidatePicker
          profile={data.profile_id}
          scope={data.scope}
          value={candidate}
          onChange={setCandidate}
        />
      )}
      <button
        className="button small"
        disabled={!candidate || busy || !!pending}
        onClick={async () => {
          setBusy(true);
          setError('');
          try {
            const job = await post<{ id: string }>(endpoint, { candidate_job: candidate });
            setPending(job.id);
          } catch (failure) {
            setError(String(failure));
          } finally {
            setBusy(false);
          }
        }}
      >
        선택한 후보로 다시 분석
      </button>
      {pending && <p role="status">저장된 UDP 원문으로 다시 분석 중…</p>}
      {error && <p role="alert">{error}</p>}
      <FlightResultSelect
        value={selection ?? ''}
        onChange={(id) => choose(completed.find((row) => row.id === id) ?? null)}
        options={[
          { value: '', label: '비행 당시 기록' },
          ...completed.map((row) => ({
            value: row.id,
            label: `후보 ${row.candidate_job.slice(0, 8)} · 전용 보정 기준 · ${row.id.slice(0, 8)}`,
          })),
        ]}
      />
    </FlightModelControls>
  );
}
