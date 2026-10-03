"""Real directory tail, immutable forecast payloads, bounded PX4 issue/decision times."""

import copy
import gc
import json
import os
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .cli import live_evidence_verified, output
from .clock_probe import ClockProbe, deadline_bounds
from .dataset import feature_transform
from .inference import PendingForecasts, load_pending, save_pending
from .mapping import TARGETS
from .preprocessing import Preprocessor
from .source import DirectoryTail, dump, read_json, sha
from .training import Predictor


def write_durable(file, record):
    file.write(json.dumps(record, allow_nan=False) + "\n")
    file.flush()
    os.fsync(file.fileno())


@contextmanager
def live_gc_scope(defer):
    """Defer cyclic scans in the finite worker; reference counting remains active."""
    enabled = gc.isenabled()
    if defer:
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def live_forecast(
    run,
    bundle,
    out,
    instance,
    timeout=600,
    resume=False,
    stop_after=None,
    runtime_config=None,
    process_timing=None,
    prepared_predictor=None,
    prepared_tail=None,
):
    startup = {"function_entry_ns": time.monotonic_ns(), **(process_timing or {})}
    startup["cyclic_gc_enabled"] = gc.isenabled()
    run, bundle, out = Path(run).resolve(), Path(bundle).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError("live output exists; explicit resume or new output required")
    out.mkdir(parents=True, exist_ok=True)
    runtime_config = runtime_config or {"poll_interval_ms": 20, "trace": False}
    trace_file = (out / "timing.jsonl").open("a", buffering=65536)

    def trace(record):
        if runtime_config.get("trace"):
            trace_file.write(json.dumps({"pid": os.getpid(), **record}) + "\n")

    torch.set_num_threads(2)
    startup["model_load_start_ns"] = time.monotonic_ns()
    p = prepared_predictor or Predictor(bundle)
    if prepared_predictor is not None and p.folder.resolve() != bundle:
        raise ValueError("prepared predictor bundle mismatch")
    startup["model_load_done_ns"] = time.monotonic_ns()
    config = read_json(bundle / "preprocessing.json")
    cal = read_json(bundle / "calibration.json")
    metadata = read_json(run / "observations/metadata.json")
    engine = Preprocessor(metadata, config)
    tail = DirectoryTail(run, engine)
    if runtime_config.get("trace"):
        tail.trace = trace
    bundle_hash = sha(bundle / "integrity.json")
    initial = read_json(run / "run.json")
    start_size = (run / "observations/streams.jsonl").stat().st_size
    evidence = {
        "initial_status": initial["status"],
        "source_grew_while_attached": False,
        "live_verified": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "actual_model_live": True,
        "resume_count": 0,
    }
    ledger = PendingForecasts(cal["error_scale"], cal["high"])
    journal_names = ["payloads.jsonl", "forecasts.jsonl", "decisions.jsonl", "clock.jsonl"]
    written = set()
    clock_consistent = True
    consumed_frames = 0
    last_segment = None
    if resume:
        startup["restore_start_ns"] = time.monotonic_ns()
        checkpoint = read_json(out / "checkpoint-live.json")
        if checkpoint["bundle_hash"] != bundle_hash or checkpoint["run_id"] != metadata["run_id"]:
            raise ValueError("live restart bundle/run mismatch")
        for name, expected in checkpoint["journal_hashes"].items():
            if sha(out / name) != expected:
                raise ValueError("live checkpoint journal mismatch")
        if runtime_config.get("warm_shadow"):
            from .resume_state import adopt_shadow

            if prepared_tail is None:
                raise ValueError("live shadow is required for this handover")
            if sha(out / "handover-state.bin") != checkpoint["handover_state_sha256"]:
                raise ValueError("live handover state hash mismatch")
            engine, tail = adopt_shadow(prepared_tail, config, metadata, checkpoint["source"])
            tail.trace = trace if runtime_config.get("trace") else None
            consumed = checkpoint["consumed_frames"]
            anchor = engine.frames[consumed - 1]
            if [anchor[k] for k in ["clock_id", "segment_id", "time_us"]] != checkpoint[
                "consumed_frame"
            ]:
                raise ValueError("shadow consumed-frame prefix differs")
        elif runtime_config.get("fast_resume"):
            from .resume_state import restore_runtime

            engine, tail = restore_runtime(
                run,
                out / "runtime-state.bin",
                checkpoint["runtime_state_sha256"],
                config,
                metadata,
                checkpoint["source"],
            )
            if runtime_config.get("trace"):
                tail.trace = trace
            trace(
                {
                    "kind": "state_restored",
                    "host_ns": time.monotonic_ns(),
                    "frames": len(engine.frames),
                    "watermark": tail.last_mark,
                }
            )
        else:
            tail.verify_checkpoint(checkpoint["source"])
        if sha(out / "pending.json") != checkpoint["pending_sha256"]:
            raise ValueError("live pending state hash mismatch")
        ledger = load_pending(out / "pending.json")
        consumed_frames = checkpoint["consumed_frames"]
        last_segment = tuple(checkpoint["last_segment"]) if checkpoint["last_segment"] else None
        evidence = checkpoint["evidence"]
        evidence["resume_count"] += 1
        written = {
            json.loads(line)["forecast_id"]
            for line in (out / "decisions.jsonl").read_text().splitlines()
        }
        startup["restore_done_ns"] = time.monotonic_ns()
    files = {n: (out / n).open("a") for n in journal_names}
    probe = ClockProbe(instance, metadata["run_id"], files["clock.jsonl"])
    if "clock_timeout_s" in runtime_config:
        probe.timeout = runtime_config["clock_timeout_s"]
        probe.socket.settimeout(probe.timeout)
    waiter = None
    if runtime_config.get("wake_on_write"):
        from .file_wait import FileWaiter

        waiter = FileWaiter(run)
    started = time.monotonic()
    new_forecasts = 0
    paused = False
    startup["ready_ns"] = time.monotonic_ns()
    trace({"kind": "startup", "resume": resume, "runtime_config": runtime_config, **startup})
    rebuild_done = not resume or bool(prepared_tail) or runtime_config.get("fast_resume", False)

    def checkpoint_state():
        checkpoint_start = time.monotonic_ns()
        for file in files.values():
            file.flush()
            os.fsync(file.fileno())
        journals_done = time.monotonic_ns()
        save_pending(ledger, out / "pending.json")
        pending_done = time.monotonic_ns()
        state_hash = None
        handover_hash = None
        if paused and runtime_config.get("warm_shadow"):
            from .resume_state import save_handover_state

            handover_hash = save_handover_state(tail, out / "handover-state.bin")
        if runtime_config.get("fast_resume") and not (paused and runtime_config.get("warm_shadow")):
            from .resume_state import save_runtime

            state_hash = save_runtime(tail, out / "runtime-state.bin")
        runtime_done = time.monotonic_ns()
        source_checkpoint = tail.checkpoint()
        source_done = time.monotonic_ns()
        dump(
            out / "checkpoint-live.json",
            {
                "run_id": metadata["run_id"],
                "bundle_hash": bundle_hash,
                "source": source_checkpoint,
                "pending_sha256": sha(out / "pending.json"),
                "evidence": evidence,
                "consumed_frames": consumed_frames,
                "consumed_frame": [
                    engine.frames[consumed_frames - 1][k]
                    for k in ["clock_id", "segment_id", "time_us"]
                ]
                if consumed_frames
                else None,
                "last_segment": last_segment,
                "runtime_state_sha256": state_hash,
                "handover_state_sha256": handover_hash,
                "journal_hashes": {name: sha(out / name) for name in journal_names},
            },
        )
        trace(
            {
                "kind": "checkpoint",
                "start_ns": checkpoint_start,
                "journals_done_ns": journals_done,
                "pending_done_ns": pending_done,
                "runtime_done_ns": runtime_done,
                "source_done_ns": source_done,
                "done_ns": time.monotonic_ns(),
                "consumed_frames": consumed_frames,
            }
        )

    def persist_decisions():
        nonlocal clock_consistent
        new = [k for k in ledger.decisions if k not in written]
        if not new:
            return
        done = time.monotonic_ns()
        after = probe.query()
        for key in new:
            d = ledger.decisions[key]
            ready = d["decision_time_us"]
            d["targets_ready_through_us"] = ready
            d["host_scoring_done_ns"] = done
            if (
                after is not None
                and after["host_send_ns"] >= done
                and after["sim_time_us"] >= ready
            ):
                d["decision_time_us"] = after["sim_time_us"]
                d["decision_clock_nonce"] = after["nonce"]
                d["decision_time_semantic"] = "conservative_post_scoring_upper_bound"
            else:
                d["decision_time_us"] = None
                d["unverified_candidate_scores"] = d["scores"]
                d["scores"] = None
                d["state"] = "decision_timing_unverified"
                d["episode_start"] = False
                clock_consistent &= after is None or after["sim_time_us"] >= ready
            write_durable(files["decisions.jsonl"], d)
            written.add(key)

    try:
        if probe.query() is None:
            raise RuntimeError("PX4 TIMESYNC unavailable; no clock substitution permitted")
        last_clock_query_ns = time.monotonic_ns()
        while not tail.closed:
            progress = tail.poll()
            # Drain telemetry and keep a current same-run clock response during warmup.
            interval = runtime_config.get("clock_refresh_ms")
            if interval and time.monotonic_ns() - last_clock_query_ns >= interval * 1000000:
                probe.query()
                last_clock_query_ns = time.monotonic_ns()
            evidence["source_grew_while_attached"] |= (
                run / "observations/streams.jsonl"
            ).stat().st_size > start_size
            windows = {(w["clock_id"], w["segment_id"], w["origin_us"]): w for w in engine.windows}
            # Rebuild filters from the source prefix, but never replay that prefix
            # into the restored matcher/alarm or prematurely close pending forecasts.
            for frame in engine.frames[consumed_frames:]:
                frame_start_ns = time.monotonic_ns()
                if not rebuild_done:
                    trace(
                        {
                            "kind": "rebuild_done",
                            "host_ns": frame_start_ns,
                            "consumed_frames": consumed_frames,
                        }
                    )
                    rebuild_done = True
                consumed_frames += 1
                t = frame["time_us"]
                segment = (frame["clock_id"], frame["segment_id"])
                if last_segment is not None and last_segment != segment:
                    ledger.advance(t, closed=True)
                    persist_decisions()
                    ledger.alarm = type(ledger.alarm)(cal["high"])
                last_segment = segment
                valid = all(frame["quality"][n] == "valid" for n in TARGETS)
                ledger.observe(
                    metadata["run_id"],
                    frame["clock_id"],
                    frame["segment_id"],
                    t,
                    t,
                    [frame["values"][n] for n in TARGETS] if valid else None,
                )
                previous_alarm = copy.deepcopy(ledger.alarm)
                ledger.advance(t)
                persist_decisions()
                if any(
                    d["state"] == "decision_timing_unverified"
                    for d in list(ledger.decisions.values())[-1:]
                ):
                    ledger.alarm = previous_alarm
                    ledger.alarm.update(None, t)
                window = windows.get((*segment, t))
                key = f"{metadata['run_id']}:{frame['clock_id']}:{frame['segment_id']}:{t}"
                if window is None or key in ledger.forecasts:
                    continue
                before = probe.query()
                preclock_done_ns = time.monotonic_ns()
                scale = p.scaler
                x = (feature_transform([window["X"]], config) - scale["x_mean"]) / scale["x_scale"]
                c = (np.asarray([window["C"]]) - scale["c_mean"]) / scale["c_scale"]
                compute_start = time.monotonic_ns()
                prediction = p.physical(p.predict_scaled(x, c))[0]
                compute_done = time.monotonic_ns()
                # Commit the prediction before querying the post-completion clock or reading future targets.
                payload = {
                    "forecast_id": key,
                    "run_id": metadata["run_id"],
                    "clock_id": frame["clock_id"],
                    "segment_id": frame["segment_id"],
                    "origin_time_us": t,
                    "prediction": prediction.tolist(),
                    "host_compute_start_ns": compute_start,
                    "host_compute_done_ns": compute_done,
                    "input_cutoff_us": t,
                    "bundle_hash": bundle_hash,
                }
                write_durable(files["payloads.jsonl"], payload)
                persist_done = time.monotonic_ns()
                after = probe.query()
                bounds = deadline_bounds(before, after, persist_done, t + 100000)
                trace(
                    {
                        "kind": "forecast_timing",
                        "forecast_id": key,
                        "origin_us": t,
                        "frame_start_ns": frame_start_ns,
                        "preclock_done_ns": preclock_done_ns,
                        "compute_start_ns": compute_start,
                        "compute_done_ns": compute_done,
                        "payload_committed_ns": persist_done,
                        "postclock_done_ns": time.monotonic_ns(),
                        "certificate": bounds,
                        "tail": getattr(tail, "frame_ready", {}).get((*segment, t)),
                        "resume": resume,
                    }
                )
                status = {
                    "on_time": "awaiting_targets",
                    "late": "forecast_late",
                    "unverified": "forecast_timing_unverified",
                }[bounds["status"]]
                key = ledger.issue(
                    metadata["run_id"],
                    frame["clock_id"],
                    frame["segment_id"],
                    t,
                    bounds["upper_us"],
                    prediction,
                    status,
                )
                write_durable(
                    files["forecasts.jsonl"],
                    {
                        **ledger.forecasts[key],
                        "timing_certificate": bounds,
                        "issued_time_semantic": "upper_bound_after_immutable_payload_commit",
                        "host_payload_committed_ns": persist_done,
                    },
                )
                if after is not None and tail.last_mark is not None:
                    clock_consistent &= (
                        after["sim_time_us"] >= tail.last_mark["through_available_time_us"]
                    )
                new_forecasts += 1
                if stop_after is not None and new_forecasts >= stop_after:
                    paused = True
                    break
            if paused:
                break
            if not progress:
                if time.monotonic() - started >= timeout:
                    break
                wait_start = time.monotonic_ns()
                if waiter is None:
                    time.sleep(runtime_config["poll_interval_ms"] / 1000)
                    notified = False
                else:
                    notified = waiter.wait(runtime_config["poll_interval_ms"] / 1000)
                trace(
                    {
                        "kind": "consumer_wait",
                        "start_ns": wait_start,
                        "done_ns": time.monotonic_ns(),
                        "file_notification": notified,
                    }
                )
        if tail.closed and not paused:
            ledger.advance(engine.frames[-1]["time_us"], closed=True)
            persist_decisions()
        evidence["live_verified"] = live_evidence_verified(
            "live", evidence["initial_status"], evidence["source_grew_while_attached"], tail.closed
        )
        checkpoint_state()
        if tail.closed and not paused:
            output(engine, tail, out / "preprocessing", "live", evidence)
    finally:
        probe.close()
        if waiter is not None:
            waiter.close()
        for file in files.values():
            file.close()
        trace({"kind": "process_complete", "host_ns": time.monotonic_ns(), "paused": paused})
        trace_file.close()
    forecasts = [json.loads(s) for s in (out / "forecasts.jsonl").read_text().splitlines()]
    decisions = [json.loads(s) for s in (out / "decisions.jsonl").read_text().splitlines()]
    counts = Counter(f["timing_certificate"]["status"] for f in forecasts)
    result = {
        "run": str(run),
        "run_id": metadata["run_id"],
        "bundle_hash": bundle_hash,
        "closed": tail.closed,
        "paused_for_restart": paused,
        "evidence": evidence,
        "forecasts": len(forecasts),
        "unique_forecasts": len({f["forecast_id"] for f in forecasts}),
        "decisions": len(decisions),
        "unique_decisions": len({d["forecast_id"] for d in decisions}),
        "deadline_counts": dict(counts),
        "valid_scored_decisions": sum(d["scores"] is not None for d in decisions),
        "clock_queries_this_process": probe.count,
        "clock_timeouts_this_process": probe.errors,
        "clock_consistent_with_raw_watermarks": bool(clock_consistent),
        "all_forecasts_met_deadline": bool(forecasts) and counts["on_time"] == len(forecasts),
        "restart_policy": "verify source identity/prefix and journal hashes; rebuild common preprocessor; retain immutable forecast/decision identities",
        "hashes": {n: sha(out / n) for n in journal_names},
        "timing_sha256": sha(out / "timing.jsonl"),
        "runtime_config": runtime_config,
    }
    dump(out / "summary-live.json", result)
    return result
