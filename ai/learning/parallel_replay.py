"""Bounded GPU producer / CPU replay workers. No fitting, job control or retries."""

import multiprocessing
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError
from pathlib import Path

from capstone_ai.detection_replay import (
    prediction_stream,
    prepare_replay,
    records,
    replay_prepared,
)
from capstone_ai.source import read_json


def read_development(canonical, bundle):
    canonical = Path(canonical)
    frames = records(canonical / 'frames.jsonl')
    airborne = any(f['operating_state'] == 'airborne' for f in frames)
    if airborne:
        result = prepare_replay(canonical, bundle, frames=frames)
    else:
        result = dict(frames=frames, windows={}, ordered_windows=[],
                      summary=read_json(canonical / 'summary.json'))
    result['airborne'] = airborne
    return result


def finish_development(canonical, bundle, calibration, target, prepared, predictions,
                       batch_seconds, batch_size):
    """Executed only by CPU children; never construct a Predictor or touch CUDA."""
    if not prepared['airborne']:
        from pipeline.pipeline import seal_development

        return seal_development(Path(canonical), Path(bundle), calibration, Path(target),
                                prediction_batch_size=batch_size)
    count = sum((f['clock_id'], f['segment_id'], f['time_us']) in prepared['windows']
                for f in prepared['frames'])
    if len(predictions) != count:
        raise ValueError('Precomputed forecast count differs from causal frame origins')
    return replay_prepared(prepared, bundle, calibration, target, predictions, batch_seconds,
                           batch_size)


class ParallelDevelopment:
    """Two CPU outputs at most plus one input prefetch; all GPU calls stay in parent.

    Batch boundaries, model order and causal ledger math are unchanged. A model
    cannot finish until all its runs are sealed. Progress is parent-owned only.
    """

    def __init__(self, workers, progress):
        if type(workers) is not int or workers not in {2, 4, 8}:
            raise ValueError('Validated parallel replay uses2/4/8 CPU workers')
        self.workers, self.progress = workers, progress
        self.cpu = self.reader = None

    def __enter__(self):
        self.cpu = ProcessPoolExecutor(self.workers,
                                       mp_context=multiprocessing.get_context('spawn'))
        self.reader = ThreadPoolExecutor(1, thread_name_prefix='canonical-input')
        return self

    def __exit__(self, kind, value, traceback):
        self.reader.shutdown(wait=True, cancel_futures=True)
        self.cpu.shutdown(wait=True, cancel_futures=True)

    def run(self, dataset, bundle, calibration, folder, predictor, rows, batch_size, model):
        if batch_size != 256:
            raise ValueError('Parallel replay must retain the frozen inference batch256')
        ids = [r['run_id'] for r in rows]
        if len(ids) != len(set(ids)) or any(Path(n).name != n or n in {'.', '..'} for n in ids):
            raise ValueError('Unique safe run identifiers required')
        if any((Path(folder) / name).exists() for name in ids):
            raise FileExistsError('Parallel replay never reuses or overwrites run output')
        pending = deque()
        completed = 0
        started = time.monotonic()
        self.progress.update(stage='development_input', model=model, done=0, total=len(rows),
                             cpu_workers=self.workers, cpu_pending=0, phase_eta_s=None)

        def check():
            self.progress.check()
            for _, future in pending:
                if future.done():
                    future.result()  # Propagate even an out-of-order failure before more GPU work.

        def drain_one():
            nonlocal completed
            run_id, future = pending[0]
            while True:
                check()
                try:
                    future.result(timeout=2)
                    break
                except TimeoutError:
                    self.progress.update(force=False, stage='development_cpu_finish', model=model,
                                         run_id=run_id, done=completed, total=len(rows),
                                         cpu_workers=self.workers, cpu_pending=len(pending))
            pending.popleft()
            completed += 1
            self.progress.update(force=False, stage='development_inference', model=model,
                                 done=completed, total=len(rows), run_id=run_id,
                                 cpu_workers=self.workers, cpu_pending=len(pending),
                                 phase_eta_s=(time.monotonic() - started) / completed * (len(rows) - completed))

        def read(index):
            return self.reader.submit(read_development,
                                      Path(dataset) / 'canonical' / ids[index], bundle)

        next_input = read(0) if rows else None
        for index, run_id in enumerate(ids):
            check()
            while len(pending) >= self.workers:
                drain_one()
            prepared = next_input.result()
            next_input = read(index + 1) if index + 1 < len(rows) else None
            self.progress.update(force=False, stage='development_gpu_forward', model=model,
                                 run_id=run_id, done=completed, total=len(rows),
                                 cpu_workers=self.workers, cpu_pending=len(pending))
            batch_seconds = []
            predictions = list(prediction_stream(prepared['ordered_windows'], predictor,
                               prepared.get('config', {}), batch_size, batch_seconds, check))
            # CPU replay needs only issuance keys, observations, and physical forecasts.
            # Drop the 100x19 input windows before IPC to avoid copying them to children.
            compact = dict(airborne=prepared['airborne'], frames=prepared['frames'],
                           windows=dict.fromkeys(prepared['windows'], True), summary=prepared['summary'])
            check()
            pending.append((run_id, self.cpu.submit(finish_development,
                            str(Path(dataset) / 'canonical' / run_id), str(bundle), calibration,
                            str(Path(folder) / run_id), compact, predictions, batch_seconds, batch_size)))
        while pending:
            drain_one()
        self.progress.update(stage='development_inference', model=model, done=len(rows),
                             total=len(rows), cpu_workers=self.workers, cpu_pending=0,
                             phase_eta_s=0)


def validate_benchmark_plan(plan):
    fixed = dict(schema='drone-ai.parallel-replay-benchmark/1', candidates=[2, 4, 8],
                 warmup_runs=8, measured_runs=8, repeats=2, batch_size=256,
                 role='normal_validation', selection='lowest mean measured seconds; declared-order ties')
    if (not isinstance(plan, dict) or set(plan) != set(fixed) | {'run_id', 'model'}
            or any(type(plan.get(k)) is not type(v) or plan[k] != v for k, v in fixed.items())
            or any(type(c) is not int for c in plan['candidates'])
            or any(not isinstance(plan[k], str) or not plan[k] or Path(plan[k]).name != plan[k]
                   or plan[k] in {'.', '..'} for k in ('run_id', 'model'))):
        raise ValueError('One fixed normal-only2/4/8-worker benchmark plan required')


def build_benchmark_plan(upstream, source_cfg):
    dataset = Path(upstream) / 'dataset'
    runs = sorted((r for r in read_json(dataset / 'manifest.json')['runs']
                   if r['role'] == 'normal_validation'), key=lambda r: r['run_id'])
    run = next(r for r in runs if all((dataset / 'canonical' / r['run_id'] / n).is_file()
                                    for n in ('frames.jsonl', 'windows.jsonl', 'summary.json')))
    inputs = {}
    for name in ('frames.jsonl', 'windows.jsonl', 'summary.json'):
        relative = 'canonical/' + run['run_id'] + '/' + name
        inputs['dataset/' + relative] = source_cfg['dataset_files'][relative]
    plan = dict(schema='drone-ai.parallel-replay-benchmark/1', candidates=[2, 4, 8],
                warmup_runs=8, measured_runs=8, repeats=2, batch_size=256,
                role='normal_validation', run_id=run['run_id'], model=source_cfg['models'][0],
                selection='lowest mean measured seconds; declared-order ties')
    validate_benchmark_plan(plan)
    return plan, inputs


def semantic_replay(path):
    """Compare every forecast/decision/metric; only measured latency fields may differ."""
    import hashlib
    import json

    path = Path(path)
    forecasts = [{k: v for k, v in row.items() if k != 'compute_seconds'}
                 for row in records(path / 'forecasts.jsonl')]
    summary = {k: v for k, v in read_json(path / 'summary.json').items()
               if 'compute_seconds' not in k}
    from capstone_ai.source import sha

    return dict(forecasts=hashlib.sha256(json.dumps(forecasts, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                decisions=sha(path / 'decisions.jsonl'),
                summary=hashlib.sha256(json.dumps(summary, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                forecast_count=len(forecasts))


def benchmark_workers(upstream, out, cfg, progress):
    """One finite in-allocation benchmark, then one uniform choice for all models.

    Uses a frozen real normal-validation run and actual CUDA Predictor, with the
    same batch256/scaler/backend. Eight repeats of one cached run are a throughput
    proxy, not independent flights, cold-I/O measurements or detector evaluation.
    """
    import os
    import shutil
    import statistics

    import numpy as np

    from capstone_ai.source import sha
    from capstone_ai.training import Predictor
    from execution.common import write
    from pipeline.pipeline import seal_development

    plan = cfg['parallel_replay_benchmark']
    validate_benchmark_plan(plan)
    if plan['model'] != cfg['models'][0]:
        raise ValueError('Benchmark model differs from frozen first model')
    root = Path(out) / 'parallel-benchmark'
    root.mkdir(exist_ok=False)
    work = root / 'work'
    source = Path(upstream) / 'dataset/canonical' / plan['run_id']
    summary = read_json(Path(upstream) / 'dataset/manifest.json')
    if not any(r['run_id'] == plan['run_id'] and r['role'] == plan['role'] for r in summary['runs']):
        raise ValueError('Benchmark input is not the frozen normal-validation member')
    template = work / 'owned-source-copy'
    template.mkdir(parents=True)
    for name in ('frames.jsonl', 'windows.jsonl', 'summary.json'):
        progress.check()
        shutil.copyfile(source / name, template / name)
        key = 'dataset/canonical/' + plan['run_id'] + '/' + name
        if sha(template / name) != cfg['evaluation_input_files'][key]:
            raise ValueError('Benchmark copied input hash differs')
    dataset = work / 'dataset'
    rows = [{'run_id': f'normal-{i}'} for i in range(plan['measured_runs'])]
    for row in rows:
        target = dataset / 'canonical' / row['run_id']
        target.mkdir(parents=True)
        for name in ('frames.jsonl', 'windows.jsonl', 'summary.json'):
            # Link only the new owned copies, never the preserved source files.
            os.link(template / name, target / name)
    bundle = Path(out) / 'models' / plan['model']
    predictor = Predictor(bundle, device='cuda')
    calibration = dict(error_scale_physical=np.ones((10, 4)).tolist(), high=[1, 1])
    evidence = dict(status='RUNNING', plan=plan, candidates=[], input_role='normal_validation',
                    truth_used=False, diagnostic_fixed_unit_thresholds=True,
                    full_calibration_reused=False, gpu_device='cuda', batch_size=256,
                    limitation='Eight repeats of one warm-cached normal run; actual GPU Predictor and CPU writers. Not cold-I/O/full18model timing or detection performance.')
    write(root / 'summary.json', evidence)
    reference = work / 'serial'
    progress.update(stage='parallel_reference', model=plan['model'], done=0, total=len(rows),
                    parallel_benchmark_active=True)
    started = time.monotonic()
    signatures = []
    for index, row in enumerate(rows):
        progress.check()
        target = reference / row['run_id']
        seal_development(dataset / 'canonical' / row['run_id'], bundle, calibration,
                         target, predictor=predictor, prediction_batch_size=256)
        signatures.append(semantic_replay(target))
        progress.update(force=False, done=index + 1)
    evidence['serial_seconds'] = time.monotonic() - started
    evidence['reference_signatures'] = signatures
    for count in plan['candidates']:
        candidate = dict(workers=count, warmup_runs=8, measured_seconds=[], signatures=[])
        with ParallelDevelopment(count, progress) as pool:
            pool.run(dataset, bundle, calibration, work / f'warm-{count}', predictor,
                     rows, 256, plan['model'])
            for repeat in range(plan['repeats']):
                progress.update(stage='parallel_benchmark', benchmark_workers=count,
                                benchmark_repeat=repeat + 1)
                destination = work / f'measured-{count}-{repeat}'
                started = time.monotonic()
                interval_start = time.time()
                pool.run(dataset, bundle, calibration, destination, predictor,
                         rows, 256, plan['model'])
                candidate['measured_seconds'].append(time.monotonic() - started)
                candidate.setdefault('intervals', []).append(dict(start=interval_start, end=time.time()))
                current = [semantic_replay(destination / r['run_id']) for r in rows]
                if current != signatures:
                    raise ValueError('Parallel forecast/decision/metric mismatch; no full calibration')
                candidate['signatures'].append(current)
        candidate.update(mean_seconds=statistics.mean(candidate['measured_seconds']),
                         passed=True)
        evidence['candidates'].append(candidate)
        write(root / 'summary.json', evidence)
    selected = min(evidence['candidates'], key=lambda c: c['mean_seconds'])['workers']
    evidence.update(status='PASSED', selected_workers=selected, all_outputs_exact=True,
                    selection=plan['selection'], fresh_full_calibration_started=False)
    shutil.copytree(reference / rows[0]['run_id'], root / 'reference-run')
    write(root / 'summary.json', evidence)
    # Keep one complete reference plus all comparison digests and timings.
    # Failed runs never reach this success-only cleanup.
    shutil.rmtree(work)
    write(Path(out) / 'parallel-selection.json', evidence)
    progress.update(stage='parallel_selection_complete', selected_cpu_workers=selected,
                    parallel_benchmark_active=False)
    return evidence
