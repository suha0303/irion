import type { Analysis, Decision } from './local-flight-types';
import type { LocalState } from './useLocalAI';

export type LocalAlarm = {
  id: string;
  model: string;
  time: number;
  score: number;
  threshold: number;
  source: 'live' | 'log';
  analysis?: Analysis;
  targetScores?: number[];
};

export function modelDecision(analyses: Analysis[]): Decision | undefined {
  if (!analyses.length) return;
  const decisions = analyses.map((a) => a.decision ?? { status: 'unavailable' as const });
  const status = decisions.some((d) => d.status === 'suspect')
    ? 'suspect'
    : decisions.some((d) => d.status === 'unavailable')
      ? 'unavailable'
      : 'normal';
  const thresholds = new Set(decisions.map((d) => d.threshold));
  return {
    status,
    threshold: thresholds.size === 1 ? decisions[0].threshold : undefined,
    observations: decisions.reduce((sum, d) => sum + (d.observations ?? 0), 0),
    exceedances: decisions.reduce((sum, d) => sum + (d.exceedances ?? 0), 0),
  };
}

export function localAlarms(state: LocalState): LocalAlarm[] {
  const rows: LocalAlarm[] = [];
  for (const row of state.insights?.alarms ?? []) {
    const d = row.decision;
    if (
      d.status !== 'suspect' ||
      d.time_us === undefined ||
      d.threshold === undefined ||
      !d.scores?.length
    )
      continue;
    rows.push({
      id: `live-${row.model_id}-${d.time_us}`,
      model: row.model_id,
      time: d.time_us,
      score: d.scores[0],
      threshold: d.threshold,
      source: 'live',
      targetScores: d.target_scores,
    });
  }
  for (const [index, analysis] of (state.insights?.report?.analyses ?? []).entries()) {
    const threshold = analysis.decision?.threshold;
    if (threshold === undefined || analysis.decision?.status === 'unavailable') continue;
    for (const [i, score] of (analysis.scoring?.scores ?? []).entries()) {
      const time = analysis.scoring?.time_us[i];
      if (time === undefined || score <= threshold) continue;
      rows.push({
        id: `log-${index}-${time}`,
        model: analysis.model_id,
        time,
        score,
        threshold,
        source: 'log',
        analysis,
      });
    }
  }
  return rows.sort((a, b) => a.time - b.time || a.id.localeCompare(b.id));
}

export const localStatus: Record<string, string> = {
  loading: '결과 확인 중',
  unconfigured: '모델 선택 대기',
  starting: '모델 준비',
  warming_up: '이력 수집',
  waiting: '입력 연결 대기',
  predicting: 'CPU 추론 중',
  stale: '새 입력 대기',
  analyzing: '종료 후 분석 중',
  finished: '비행 분석 완료',
  error: '작업 오류',
  interrupted: '작업 중단',
};
