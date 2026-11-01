import type { Detection } from './types';
export function compareModels(lstm: Detection|null|undefined, transformer: Detection|null|undefined) {
  if (!lstm || !transformer || ![lstm,transformer].every(e=>['normal','suspect'].includes(e.status))) {
    return {state:'waiting',label:'두 모델의 유효 판정을 기다리고 있습니다.'};
  }
  const a=lstm.input_ref,b=transformer.input_ref;
  const sameReference = a?.kind===b?.kind && a?.sha256===b?.sha256 && a?.session_id===b?.session_id && a?.segment_sha256===b?.segment_sha256 && lstm.input_sha256===transformer.input_sha256;
  const sameSource = lstm.run_id===transformer.run_id && lstm.origin===transformer.origin && sameReference;
  const sameWindow = lstm.epoch===transformer.epoch && lstm.target===transformer.target &&
    lstm.start_ms===transformer.start_ms && lstm.end_ms===transformer.end_ms;
  if (!sameSource || !sameWindow) return {state:'unmatched',label:'근거 구간·대상이 다릅니다. 각 판정을 확인하세요.'};
  return lstm.status===transformer.status
    ? {state:'agree',label:'같은 근거 구간 · 두 모델의 판정 일치'}
    : {state:'different',label:'판정 불일치 · 같은 근거 구간을 확인하세요.'};
}
