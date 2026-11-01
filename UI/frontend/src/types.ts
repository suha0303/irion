export interface Signal {
  value: number | null; unit: string; status: 'valid' | 'missing' | 'stale' | 'invalid';
  age_ms: number | null; label: string; message: string; reason: string;
}
export interface Detection {
  event_id: string; run_id: string; input_sha256?: string; epoch: number;
  provider_id?: string; kind?: 'detector' | 'explanation';
  input_ref?: {kind:string;sha256?:string;segment_sha256?:string;session_id?:string};
  start_ms: number; end_ms: number; decided_at_ms: number; valid_until_ms: number;
  model_version: string; status: 'normal' | 'suspect' | 'unavailable' | 'warming_up' | 'explanation';
  target: string; score: number | null; threshold: number | null; title: string; description: string;
  origin: 'model' | 'fixture';
  evidence: { signal: string; unit: string; observed: number | null; predicted: number | null; residual: number | null; note: string }[];
}
export interface AIState { state: string; url: string; message: string; fixture_only?: boolean; last_sync_utc_ms?: number }
export interface Frame {
  source_kind?: 'live'|'replay'; session?: SessionState | null; providers?: ProviderState[];
  active_by_provider?: Record<string,Detection|null>; live_series?: Point[]; manifest?: Manifest;
  input_contract?: {sample_period_ms:number|null;preprocess_version:string;signals:Record<string,string>};
  run_id: string; epoch: number; t_ms: number; values: Record<string, Signal>;
  armed: boolean | null; mode: string; flight_state: string; events: Detection[];
  ai: AIState; fixture_mode: boolean; active_result: Detection | null;
  streams: { name: string; age_ms: number | null; last_t_ms: number | null; status: string }[];
}
export interface Manifest {
  npz?: NpzState | null;
  rates?: {raw_receive_hz:number|null;processed_receipt_hz:number|null;processed_output_hz:number|null;display_max_hz:number;model_inference_period_ms:number|null};
  source_kind?: 'live'|'replay'; session?: SessionState; profile?: InputProfile;
  run_id: string; name: string; input_sha256: string | null; source: string; source_system: number; source_component: number;
  duration_ms: number; total_records: number; accepted_records: number; excluded_records: number; epochs: number;
  boundaries: { t_ms: number; epoch: number; reason: string; line: number }[];
  issues: { line: number; reason: string; severity: string }[];
  streams: { name: string; count: number; median_hz: number | null; target_hz: number | null; max_interval_ms: number | null }[];
}
export interface Point { t_ms: number; epoch: number; values: Record<string, number | null> }
export interface NPZReport {
  shape: number[]; windows: number; data_ready_windows: number; model_ready_windows: number;
  missing_cells: number; invalid_windows: number; provenance_matches: boolean; normalization: string;
  input_hashes: string[]; current_input_sha256: string; schema_version: string; note: string;
  features: { name: string; unit: string; message: string }[];
  config: { window_rows: number; stride_rows: number; tick_ns: number };
}
export interface Metrics { render_ack: { samples: number; p95_ms: number | null }; result_receipt_to_render_ack: { samples: number; p95_ms: number | null }; target_ms: number; method: string }
export function detectionKey(event: Detection) { return `${event.provider_id ?? "legacy"}:${event.origin}:${event.model_version}:${event.event_id}`; }

export function timeLabel(ms: number, decimals = false) {
  const ticks = decimals ? Math.round(Math.max(0, ms) / 100) : Math.floor(Math.max(0, ms) / 1000);
  const perMinute = decimals ? 600 : 60;
  const tail = decimals ? ((ticks % perMinute) / 10).toFixed(1) : (ticks % perMinute).toString();
  return `${Math.floor(ticks / perMinute).toString().padStart(2, '0')}:${tail.padStart(decimals ? 4 : 2, '0')}`;
}
export function numeric(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('ko-KR', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
export const targetLabel: Record<string, string> = { battery: '전원 계통', powertrain: '구동계', motor_1: '모터 1', motor_2: '모터 2', motor_3: '모터 3', motor_4: '모터 4' };
export const statusLabel: Record<string, string> = { valid: '유효', missing: '미수신', stale: '만료', invalid: '유효하지 않음', normal: '정상 범위', suspect: '이상 의심', unavailable: '판정 불가', warming_up: '준비 중', explanation: '설명', disconnected: '미연결', connecting: '연결 중', ready: '결과 대기', saved: '저장 결과', error: '연결 오류', incompatible: '입력 호환 안 됨', waiting: '수신 대기', connected: '수신 중', finished: '기록 종료' };

export interface ProviderState {
  provider_id: string; name: string; kind: 'detector' | 'explanation'; model_version: string;
  state: string; message: string; url: string; input_compatible: boolean; fixture_only?: boolean;
  poll_interval_ms: number;
}
export interface SessionState {
  session_id: string; state: string; message: string; kind: string; recording: boolean;
  buffer_start_ms: number; last_sequence: number; dropped_sequences: number;
}
export interface InputProfile {
  feature_order?: string[];
  profile_id: string; input_format: 'mavlink'|'normalized'; preprocess_version: string;
  sample_period_ms: number | null; signals: Record<string,{label:string;unit:string;ttl_ms:number;message:string;display_key?:string|null}>;
}

export interface NpzState { state:string; error:string; received_rows:number; saved_rows:number; pending_rows:number; chunk_count:number; finished:boolean; schema_version:string; chunk_seconds:number }
export interface NpzChunk { filename:string; rows:number; start_sequence:number; end_sequence:number; start_ms:number; end_ms:number; epoch:number; sha256:string }
export interface ExportJob { job_id:string; run_id:string; state:'queued'|'running'|'ready'|'error'; error:string; download_url:string|null; rows?:number; sha256?:string }
