export type Aircraft = {
  schema: 'drone.real-aircraft/1';
  id: string;
  name: string;
  autopilot: string;
  provenance?: {
    extraction_revision?: string;
    source_sha256?: string;
    fields?: Record<string, ProfileEvidence>;
  };
  firmware: string;
  rotor_count: number | null;
  series_cells: number | null;
  capacity_ah: number | null;
  battery_id: string;
  motor_channels: Record<string, number>;
  output_min: number | null;
  output_max: number | null;
  mapping_evidence: string;
  notes: string;
  imu_instance: number | null;
  battery_instance: number | null;
  actuator_instance: number | null;
};
export type ProfileEvidence = {
  value: string | number | Record<string, number>;
  source: string;
  source_sha256: string;
  reference?: string | null;
  status: string;
};
export type ProfileSuggestion = {
  profile: Partial<Aircraft>;
  source_sha256: string;
  evidence: Record<string, ProfileEvidence>;
  questions: { field: string; label: string; reason: string; candidates: number[] }[];
  health: string;
  note: string;
  input_fields_present: boolean;
};
export type Profile = {
  profile: Aircraft;
  version: string;
  eligibility: {
    profile_complete: boolean;
    missing_fields: string[];
    log_missing_fields?: string[];
    timeline_ready?: boolean;
  };
};
export type Timing = {
  streams?: Record<string, { reversals: number; duplicates: number; max_gap_us: number }>;
  recorded_intersections?: { start_us: number; end_us: number; duration_s: number }[];
  candidate_segments?: { start_us: number; end_us: number; frames: number; usable: boolean }[];
  requested_frames?: number;
  retained_frames?: number;
  coverage?: number;
  missing_or_stale_frames?: number;
  clock_corrections?: Record<
    string,
    { drift_ppm: number; offset_us: number; residual_max_us: number }
  >;
};
export type FlightEvents = {
  supported: boolean;
  health: string;
  notes: string[];
  intervals: {
    index: number;
    flight: number;
    start_us: number;
    end_us: number;
    imu_instance: number | null;
    usable: boolean;
    evidence: string;
  }[];
  error_counts: { subsystem: number; code: number; count: number; label: string }[];
};
export type ForecastSeries = {
  segment: number;
  horizon_us: number;
  origin_us: number[];
  time_us: number[];
  observed: number[][];
  predicted: number[][];
  residual: number[][];
};
export type Decision = {
  status: 'normal' | 'suspect' | 'unavailable';
  reason?: string;
  threshold?: number;
  calibration_id?: string;
  observations?: number;
  exceedances?: number;
  scores?: number[];
  alarms?: boolean[];
  detector_policy?: string;
  active_domains?: string[];
  time_us?: number;
  target_scores?: number[];
};
export type Analysis = {
  scoring?: { scores: number[]; time_us: number[]; unit: string };
  segment_results?: SegmentAnalysis[];
  decision?: Decision;
  normal_alarm_rate?: number | null;
  model_id: string;
  adapted?: boolean;
  adapted_sha256?: string | null;
  metrics: {
    targets?: string[];
    units?: string[];
    rmse?: number[];
    persistence_rmse?: number[];
    segments?: unknown[];
    series?: ForecastSeries[];
  };
  retained_segments: number;
  elapsed_seconds?: number;
  coverage?: number;
};
export type Job = {
  id: string;
  action: string;
  status: string;
  error?: string;
  progress?: { epoch?: number; train_loss?: number; stage?: string; elapsed_seconds?: number };
  result?: {
    counts?: Record<string, number>;
    note?: string;
    missing?: string[];
    elapsed_seconds?: number;
    within_budget?: boolean;
    tuned_models?: {
      model_id: string;
      review_status?: string;
      review?: {
        promotion_check: {
          status: string;
          reasons: string[];
          missing: string[];
          flights?: {
            health: string;
            error_before?: number[];
            error_after?: number[];
            decisions?: { original: Decision; adapted: Decision };
            events?: {
              original: { covered: boolean; detected: boolean };
              adapted: { covered: boolean; detected: boolean };
            }[];
          }[];
        };
      };
    }[];
    run_summary?: RunSummary;
    source_sha256?: string;
    profile_suggestion?: ProfileSuggestion;
    flight_events?: FlightEvents;
    interval_evidence?: string;
    imu_instance?: number;
    filename?: string;
    autopilot?: string;
    missing_candidate_fields?: Record<string, string[]>;
    parser_warnings?: string;
    parameters?: Record<string, number>;
    messages?: string[];
    models?: { id: string; p95_ms: number }[];
    timing?: Timing;
    role?: string;
    health?: string;
    frames?: number;
    analyses?: Analysis[];
    original?: Analysis;
    adapted?: Analysis;
  };
};

export type SegmentMetrics = {
  targets?: string[];
  units?: string[];
  rmse?: number[];
  persistence_rmse?: number[];
  windows?: number;
  samples?: number;
  short_cell_v?: number;
  long_cell_v?: number;
  pack_voltage_rmse?: number;
};
export type SegmentAnalysis = {
  start_us: number;
  end_us: number;
  frames: number;
  metrics: SegmentMetrics;
  decision: Decision;
  series?: ForecastSeries[];
};
export type RunInterval = {
  start_us: number;
  end_us: number;
  reason: string;
  missing_features?: string[];
};
export type RunSummary = {
  status: 'normal' | 'suspect' | 'incomplete' | 'unavailable';
  partial: boolean;
  context: {
    start_us: number;
    end_us: number;
    extent_known: boolean;
    clock: string;
    filename?: string;
    notes: string[];
  };
  duration_us: number;
  analyzed_us: number;
  excluded_us: number;
  coverage: number | null;
  note: string;
  timeline: RunInterval[];
  models: (SegmentMetrics & {
    model_id: string;
    segment_counts: { normal: number; suspect: number; unavailable: number };
    observations: number;
    exceedances: number;
    adapted_variants: number;
  })[];
};
