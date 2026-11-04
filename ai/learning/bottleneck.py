"""One finite real-data diagnostic; disposable weights, never a main training run."""

import argparse
import gc
import gzip
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from capstone_ai.training import loss, seed_all
from execution.common import ROOT, Progress, memory_snapshot, read, sha, write
from pipeline.runtime import check

from . import comparison, study
from .benchmark import measure
from .diagnostic import same, stop_signal
from .models import make_model
from .optimization import make_optimizer, policy_for
from .telemetry import Telemetry
from .training import Batches, backend, clip_gradients, project_lock, safe_step

PHASE = "drone.phase."


def profiled_step(model, optimizer, x, c, y):
    """Mirror safe_step with ranges; equivalence is checked before each GPU profile."""
    with record_function(PHASE + "zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    with record_function(PHASE + "forward_loss"):
        value = loss(model(x, c), y)
    with record_function(PHASE + "loss_guard"):
        if not bool(torch.isfinite(value)):
            raise ValueError("Nonfinite loss; optimizer not executed")
    with record_function(PHASE + "backward"):
        value.backward()
    with record_function(PHASE + "gradient_guard_clip"):
        clip_gradients(model)
    with record_function(PHASE + "optimizer_step"):
        optimizer.step()
    return value.detach()


def event_row(event):
    def device_time(name, legacy):
        return float(getattr(event, name) if hasattr(event, name) else getattr(event, legacy, 0))

    return dict(name=event.key, calls=event.count,
                cpu_total_us=float(event.cpu_time_total), self_cpu_us=float(event.self_cpu_time_total),
                device_total_us=device_time("device_time_total", "cuda_time_total"),
                self_device_us=device_time("self_device_time_total", "self_cuda_time_total"))


def profile_model(entry, arrays, output, progress, config):
    output.mkdir(parents=True, exist_ok=False)
    spec = study.spec_for(entry)
    policy = policy_for(entry["optimizer"])
    seed_all(0)
    batches = Batches(arrays, 256, "cuda")
    with project_lock("cuda"), backend("cuda"):
        model = make_model(spec).to("cuda").train()
        reference = make_model(spec).to("cuda").train()
        reference.load_state_dict(model.state_dict())
        optimizer = make_optimizer(model.parameters(), .001, policy)
        other_optimizer = make_optimizer(reference.parameters(), .001, policy)
        x, c, y = batches.get(torch.arange(256))
        original = safe_step(reference, other_optimizer, x, c, y)
        instrumented = profiled_step(model, optimizer, x, c, y)
        equivalent = (same(original, instrumented)
                      and same(reference.state_dict(), model.state_dict())
                      and same(other_optimizer.state_dict(), optimizer.state_dict()))
        write(output / "step-equivalence.json", dict(passed=equivalent, device="cuda"))
        if not equivalent:
            raise ValueError("Instrumented step differs from actual safe_step")
        del reference, other_optimizer
        for _ in range(config["profile_warmup_steps"]):
            progress.check()
            safe_step(model, optimizer, x, c, y)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False, with_stack=False) as capture:
            for index in range(config["profile_steps"]):
                progress.check()
                offset = (index * 256) % (batches.count - 255)
                with record_function(PHASE + "loader_transfer"):
                    x, c, y = batches.get(torch.arange(offset, offset + 256))
                profiled_step(model, optimizer, x, c, y)
                capture.step()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        events = capture.key_averages()
        rows = [event_row(e) for e in events]
        phases = [r for r in rows if r["name"].startswith(PHASE)]
        cuda_events = sum(e.device_type == torch.autograd.DeviceType.CUDA for e in capture.events())
        result = dict(model=entry["id"], phase_times=phases, profiler_wall_seconds=elapsed,
                      steps=config["profile_steps"], cuda_events=cuda_events,
                      top_self_cpu=sorted(rows, key=lambda r: r["self_cpu_us"], reverse=True)[:25],
                      top_self_device=sorted(rows, key=lambda r: r["self_device_us"], reverse=True)[:25],
                      interpretation="Profiler timings include instrumentation overhead; CPU and GPU totals overlap and must not be added",
                      equivalent_to_actual_safe_step=equivalent,
                      throughput_source="Separate unprofiled500-step actual loader/model/optimizer measurement")
        trace = output / "trace.json"
        capture.export_chrome_trace(str(trace))
        if trace.stat().st_size > 200 * 1024**2:
            raise OSError("Profile trace exceeded200MiB diagnostic bound")
        with trace.open("rb") as source, gzip.open(output / "trace.json.gz", "wb") as target:
            shutil.copyfileobj(source, target)
        trace.unlink()
        write(output / "profile.json", result)
        if not cuda_events:
            raise RuntimeError("CUDA profiler events unavailable; do not claim a GPU bottleneck assessment")
        # Match the main trainer's checkpoint contents and temp-file/rename path.
        torch.cuda.synchronize()
        started = time.perf_counter()
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        state_copy_seconds = time.perf_counter() - started
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          best_state=best_state, model_spec=spec.to_dict(), diagnostic_only=True)
        timings = {}
        for name in ("last", "best"):
            temporary = output / (name + ".tmp")
            started = time.perf_counter()
            torch.save(checkpoint, temporary)
            temporary.replace(output / (name + ".pt"))
            timings[name + "_save_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        restored = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        timings["load_to_cpu_seconds"] = time.perf_counter() - started
        passed = same(restored["best_state"], best_state)
        checkpoint_bytes = (output / "last.pt").stat().st_size
        write(output / "checkpoint-io.json", dict(
            **timings, copy_best_state_to_cpu_seconds=state_copy_seconds, bytes=checkpoint_bytes,
            restored_best_state_exact=passed, device_path=str(output),
            persistence="torch.save close/rename, no fsync; single warm-cache sample, not worst-case durable NFS latency",
            weights_disposable=True,
        ))
        if not passed:
            raise ValueError("Diagnostic checkpoint reload mismatch")
        # Retain all failure evidence; delete only successful disposable checkpoint weights.
        for name in ("last.pt", "best.pt"):
            (output / name).unlink()
        del model, optimizer, checkpoint, restored, best_state, x, c, y
    del batches, capture, events
    gc.collect()
    torch.cuda.empty_cache()
    return result


def worker(root):
    root = Path(root)
    config = read(root / "input/config.json")
    if (config.get("execution_scope") != "bottleneck_diagnostic_only"
            or config.get("models") != study.MODELS
            or config.get("training_authorized") is not False
            or config.get("main_epochs") != 0):
        raise ValueError("Only the frozen disposable diagnostic is authorized")
    if sha(root / "input/config.json") != read(root / "authorization.json")["config_sha256"]:
        raise ValueError("Diagnostic authorization changed")
    out = root / "compute"
    out.mkdir(exist_ok=False)
    progress = Progress(out / "progress.json", hours=config["wall_hours"] - .1)
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    telemetry = None
    try:
        environment = check(ROOT / "ai", out, "benchmark", job_prefix="drone-ai-")
        if environment["job"].get("NumCPUs") != "4":
            raise ValueError("Expected four allocated CPUs for the diagnostic")
        initial_memory = memory_snapshot()
        write(out / "initial-memory.json", initial_memory)
        if initial_memory["available_bytes"] < 12 * 1024**3:
            raise MemoryError("Less than12GiB actual host/cgroup headroom; diagnostic stopped")
        upstream = Path(config["upstream"])
        if sha(upstream / "dataset/manifest.json") != config["manifest_sha256"]:
            raise ValueError("Frozen dataset manifest changed")
        progress.update(status="RUNNING", stage="input_admission", models_completed=[])
        started = time.perf_counter()
        _, _, headers = comparison.admission(upstream)
        admission_seconds = time.perf_counter() - started
        memory = comparison.memory_plan(headers)
        snapshot = memory_snapshot()
        write(out / "host-memory-admission.json", dict(plan=memory, snapshot=snapshot))
        if snapshot["available_bytes"] < memory["estimated_host_peak_bytes"] + 1024**3:
            raise MemoryError("Full-study estimated memory plus1GiB reserve unavailable")
        telemetry = Telemetry(out)
        telemetry.start()
        scaler = dict(np.load(upstream / "dataset/scaler.npz", allow_pickle=False))
        normalized, preparation = {}, {}
        for role in ("normal_train", "normal_validation"):
            started = time.perf_counter()
            normalized[role] = comparison.normalized_mmap(
                upstream / "dataset", role, scaler, out / "normalized" / role, progress)
            preparation[role] = dict(seconds=time.perf_counter() - started,
                                     windows=len(normalized[role]["X"]))
        rows = []
        for index, entry in enumerate(study.VARIANTS):
            progress.check()
            name = entry["id"]
            started_at = time.time()
            progress.update(stage="timing_train_validation", model=name, done=index,
                            total=len(study.MODELS), models_completed=[r["model"] for r in rows])
            timing = measure(study.spec_for(entry), normalized["normal_train"], [256],
                out / "measurements" / name, device="cuda", warmup=50, steps=500,
                gpu_authorized=True, progress=progress, seed=0, runtime_stage="benchmark",
                validation=normalized["normal_validation"], learning_rate=.001,
                optimization=policy_for(entry["optimizer"]))[0]
            progress.update(stage="operator_profile_and_checkpoint", model=name)
            profile_model(entry, normalized["normal_train"], out / "profiles" / name,
                          progress, config)
            rows.append(dict(model=name, timing=timing, started=started_at, finished=time.time()))
            write(out / "measurements-summary.json", rows)
        telemetry.stop()
        telemetry = None
        del normalized, scaler
        gc.collect()
        shutil.rmtree(out / "normalized")
        result = dict(status="DIAGNOSTIC_COMPLETE_PENDING_RELEASE", models=rows,
            source_config_sha256=sha(root / "input/config.json"), environment=environment,
            input_admission_seconds=admission_seconds, normalization=preparation,
            all16_measured=len(rows) == 16, batch_size=256, seed=0, precision="float32",
            diagnostic_peak_learning_rate=.001, learning_rate_selected=False,
            main_training_started=False, calibration_started=False, evaluation_started=False,
            final_test_used=False, automatic_retry=False, llm_calls=0,
            limitations=["Integrity reads and finite-input scans warm filesystem caches; cold-cache disk latency not isolated",
                         "Diagnostic rates are fixed0.001 for timing, not selected main-training rates",
                         "Profiler measurements carry overhead; use separate unprofiled500-step timings",
                         "Checkpoint close/rename timing is not fsync durability latency",
                         "Other users' CPU/network/storage activity may affect this run"])
        write(out / "result.json", result)
        progress.update(status="DIAGNOSTIC_COMPLETE_PENDING_RELEASE", stage="stopped",
                        models_completed=study.MODELS, done=16, total=16)
    except BaseException as exc:
        progress.update(status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}",
                        automatic_retry=False)
        raise
    finally:
        if telemetry is not None:
            telemetry.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    worker(parser.parse_args().root)
