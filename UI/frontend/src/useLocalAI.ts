import { useEffect, useState } from 'react';
import { api } from './api';
import type { Analysis, Decision, RunSummary } from './local-flight-types';

type BufferedCounts = { forecasts: number; scored_frames: number; alarm_frames: number };

export type LocalState = {
  status: string;
  message?: string;
  scope?: string;
  finish_reason?: string;
  mode?: string;
  hold_ms?: number;
  candidate_job?: string;
  candidate_models?: string[];
  whole_run_models?: string[];
  whole_available?: boolean;
  whole_status?: 'running' | 'complete' | 'failed';
  whole_error?: string;
  whole_summary?: {
    frames: number;
    valid_frames: number;
    duration_s: number;
    models: {
      model_id: string;
      alarm: boolean | null;
      score: number | null;
      threshold: number;
      coverage?: number;
      candidate_job?: string;
    }[];
  };
  buffered_available?: boolean;
  buffered_parallel?: boolean;
  forecast_models?: string[];
  buffered_model_states?: Record<
    string,
    BufferedCounts & { publishing: boolean; alarm: boolean | null }
  > | null;
  buffered_summary?: Partial<BufferedCounts> & { model_summaries?: Record<string, BufferedCounts> };
  forecast_model?: string;
  post_model?: string;
  adapted_sha256?: string;
  post_adapted_sha256?: string;
  decision?: Decision;
  missing_features?: string[];
  prediction?: { origin_us: number; values: number[]; units: string[] } | null;
  rates?: {
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
  result?: {
    message: string;
    run_summary?: RunSummary;
    analyses: { manifest: string; analysis: Analysis }[];
  };
  insights?: {
    history_available: boolean;
    alarm_count: number;
    alarms: { model_id: string; adapted_sha256?: string; decision: Decision }[];
    report?: { analyses: Analysis[]; run_summary?: RunSummary };
    report_source?: string;
    report_job?: string;
  };
};

export function useLocalAI(runId: string) {
  const [state, setState] = useState<LocalState>({ status: 'loading' });
  const [error, setError] = useState('');
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    setState({ status: 'loading' });
    setError('');
    if (!runId.startsWith('live-')) return;
    const poll = async () => {
      try {
        const next = await api<LocalState>(`/sessions/${runId}/local-ai`);
        if (!cancelled) {
          setState(next);
          setError('');
        }
      } catch (failure) {
        if (!cancelled) setError(String(failure));
      }
      if (!cancelled) timer = setTimeout(poll, 1000);
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [runId]);
  return { state, setState, error };
}
