"""Real spawn-worker replay equivalence, bounded queues, failure and input safety."""

import json
import shutil

import numpy as np
import pytest
import torch
from test_post_training_sensitivity import config as sensitivity_config
from test_replay_optimized import FixturePredictor, case

from capstone_ai.detection_replay import records
from capstone_ai.source import read_json, sha
from learning import post_training as post
from learning.models import ModelSpec, make_model
from learning.parallel_replay import ParallelDevelopment, finish_development, read_development
from pipeline.pipeline import seal_development


class Progress:
    def __init__(self):
        self.updates = []

    def check(self):
        pass

    def update(self, **values):
        self.updates.append(values)


def inputs(tmp_path):
    canonical, bundle, _, _ = case(tmp_path)
    dataset = tmp_path / 'dataset'
    rows = []
    for i in range(3):
        name = f'run-{i}'
        path = dataset / 'canonical' / name
        shutil.copytree(canonical, path)
        rows.append({'run_id': name})
    # The third case must stay explicitly unavailable, not be dropped or inferred.
    path = dataset / 'canonical/run-2/frames.jsonl'
    ground = records(path)
    for f in ground:
        f['operating_state'] = 'ground'
    path.write_text(''.join(json.dumps(f) + '\n' for f in ground))
    (dataset / 'canonical/run-2/windows.jsonl').unlink()
    return dataset, bundle, rows


@pytest.mark.parametrize('name', [None, 'lstm', 'transformer'])
def test_real_two_process_replay_matches_serial_with_same_batch(tmp_path, name):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = make_model(ModelSpec(name, hidden=64, layers=2)).eval() if name else None
    dataset, bundle, rows = inputs(tmp_path)
    cal = {'error_scale_physical': np.ones((10, 4)), 'high': [1, 1]}
    serial, parallel = tmp_path / 'serial', tmp_path / 'parallel'
    old = FixturePredictor(bundle, model)
    for row in rows:
        seal_development(dataset / 'canonical' / row['run_id'], bundle, cal,
                         serial / row['run_id'], predictor=old, prediction_batch_size=256)
    predictor = FixturePredictor(bundle, model)
    progress = Progress()
    with ParallelDevelopment(2, progress) as pool:
        pool.run(dataset, bundle, cal, parallel, predictor, rows, 256, 'fixture')
    assert predictor.calls == old.calls
    assert max(p.get('cpu_pending', 0) for p in progress.updates) <= 2
    assert progress.updates[-1]['done'] == 3
    for row in rows:
        a, b = [p / row['run_id'] for p in (serial, parallel)]
        assert (a / 'decisions.jsonl').read_bytes() == (b / 'decisions.jsonl').read_bytes()
        forecasts = []
        for root in (a, b):
            forecasts.append([{k: v for k, v in r.items() if k != 'compute_seconds'}
                              for r in records(root / 'forecasts.jsonl')])
        assert forecasts[0] == forecasts[1]
        summary = []
        for root in (a, b):
            summary.append({k: v for k, v in read_json(root / 'summary.json').items()
                            if 'compute_seconds' not in k})
        assert summary[0] == summary[1]
        for filename, digest in read_json(b / 'seal.json')['files'].items():
            assert sha(b / filename) == digest
        for f in forecasts[1]:
            assert f['issued_time_us'] < f['target_times_us'][0]


def test_cpu_worker_failure_propagates_without_success_seal_or_retry(tmp_path):
    dataset, bundle, rows = inputs(tmp_path)
    path = dataset / 'canonical/run-0/frames.jsonl'
    frames = records(path)
    frames[0].pop('quality')
    path.write_text(''.join(json.dumps(f) + '\n' for f in frames))
    target = tmp_path / 'output'
    progress = Progress()
    predictor = FixturePredictor(bundle)
    with ParallelDevelopment(2, progress) as pool, pytest.raises(KeyError, match='quality'):
        pool.run(dataset, bundle, {'error_scale_physical': np.ones((10, 4)), 'high': [1, 1]},
                 target, predictor, rows[:1], 256, 'fixture')
    assert not (target / 'run-0/seal.json').exists()
    assert not any(p.get('done') == 1 for p in progress.updates)
    assert predictor.calls == [18]


def test_precomputed_forecast_count_must_match_causal_origins(tmp_path):
    dataset, bundle, rows = inputs(tmp_path)
    data = read_development(dataset / 'canonical/run-0', bundle)
    with pytest.raises(ValueError, match='count'):
        finish_development(dataset / 'canonical/run-0', bundle, {}, tmp_path / 'out',
                           data, [], [], 256)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('workers', [0, 1, 3, 16, True])
def test_concurrency_cannot_expand_silently(workers):
    with pytest.raises(ValueError):
        ParallelDevelopment(workers, Progress())


@pytest.mark.parametrize('change', [
    {'parallel_replay_workers': 4}, {'parallel_replay_workers': True},
    {'all_model_calibration_fresh': False}, {'prediction_batch_size': 1},
    {'sensitivity_policy': None}, {'continuation': {'reuse': True}},
])
def test_post_training_parallel_scope_is_fixed(change):
    cfg = dict(sensitivity_config(), parallel_replay_workers=2)
    post.config_check(cfg)
    cfg.update(change)
    with pytest.raises(ValueError):
        post.config_check(cfg)


def test_shared_normal_inputs_keep_fresh_calibration_identical(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from execution import engine
    from execution.common import read, write

    dataset = tmp_path / 'upstream/dataset'
    write(dataset / 'manifest.json', {'runs': [dict(run_id='r', paired_case_id='g', role='normal_calibration')]})
    (dataset / 'scaler.npz').write_bytes(b'same frozen train-only scaler')
    arrays = dict(X=np.zeros((20, 100, 19)), C=np.zeros((20, 6)), Y=np.ones((20, 10, 4)),
                  run_id=np.array(['r'] * 20), segment=np.zeros(20, dtype=np.int64),
                  origin_us=np.arange(20, dtype=np.int64) * 1000000)
    loads = []
    active_backend = [False]

    @contextmanager
    def backend(device):
        active_backend[0] = True
        try:
            yield
        finally:
            active_backend[0] = False

    def normalized(dataset, role, scaler):
        loads.append(role)
        return {k: v.copy() for k, v in arrays.items()}

    class Predictor:
        def __init__(self, folder, device):
            self.scaler = {'y_scale': np.ones(4)}

        def predict_scaled(self, x, c):
            return np.zeros((len(x), 10, 4))

    class Executor:
        def run(self, *args):
            assert args[5] == []  # No development cases in this calibration fixture.
            assert active_backend[0] is True

    monkeypatch.setattr(engine, 'normalized_role', normalized)
    monkeypatch.setattr(engine, 'Predictor', Predictor)
    outputs = []
    for label, executor in [('serial', None), ('parallel', Executor())]:
        out = tmp_path / label
        for name in ['a', 'b']:
            write(out / 'models' / name / 'integrity.json', {})
            (out / 'models' / name / 'scaler.npz').write_bytes((dataset / 'scaler.npz').read_bytes())
        engine.calibrate(tmp_path / 'upstream', out, Progress(), model_names=['a', 'b'],
                         backend_factory=backend, development_executor=executor,
                         prediction_batch_size=256)
        outputs.append([read(out / 'calibration' / name / 'calibration.json') for name in ['a', 'b']])
    assert outputs[0] == outputs[1]
    assert loads == ['normal_validation', 'normal_calibration'] * 3


@pytest.mark.parametrize('settings', [{'prediction_batch_size': 1},
                                    {'prediction_batch_size': 256, 'reuse_calibration': {'old': {}}}])
def test_parallel_engine_cannot_reuse_calibration_or_change_batch(tmp_path, settings):
    from execution.engine import calibrate

    with pytest.raises(ValueError, match='fresh calibration'):
        calibrate(tmp_path, tmp_path, Progress(), development_executor=object(), **settings)


@pytest.mark.parametrize('workers', [4, 8])
def test_larger_pool_budgets_require_matching_cpu_profile(workers):
    cfg = dict(sensitivity_config(), parallel_replay_workers=workers, cpus=workers + 2)
    post.config_check(cfg)
    cfg['cpus'] = 4
    with pytest.raises(ValueError):
        post.config_check(cfg)


def test_benchmark_real_spawn_candidates_and_exact_outputs(tmp_path, monkeypatch):
    from capstone_ai import training
    from capstone_ai.source import dump
    from learning.parallel_replay import benchmark_workers, build_benchmark_plan

    canonical, bundle, _, _ = case(tmp_path)
    upstream = tmp_path / 'upstream'
    destination = upstream / 'dataset/canonical/normal-validation'
    shutil.copytree(canonical, destination)
    dump(upstream / 'dataset/manifest.json', {'runs': [{'run_id': 'normal-validation', 'role': 'normal_validation'}]})
    hashes = {'canonical/normal-validation/' + p.name: sha(p) for p in destination.iterdir()}
    plan, inputs = build_benchmark_plan(upstream, dict(dataset_files=hashes, models=['fixture']))
    out = tmp_path / 'compute'
    shutil.copytree(bundle, out / 'models/fixture')
    monkeypatch.setattr(training, 'Predictor', lambda folder, device: FixturePredictor(folder))
    before = {p.name: sha(p) for p in destination.iterdir()}
    result = benchmark_workers(upstream, out,
                               dict(parallel_replay_benchmark=plan, models=['fixture'],
                                    evaluation_input_files=inputs), Progress())
    assert result['status'] == 'PASSED'
    assert result['selected_workers'] in [2, 4, 8]
    assert [c['workers'] for c in result['candidates']] == [2, 4, 8]
    assert all(c['passed'] and len(c['measured_seconds']) == 2 for c in result['candidates'])
    assert result['all_outputs_exact'] is True
    assert not (out / 'parallel-benchmark/work').exists()
    assert (out / 'parallel-benchmark/reference-run/seal.json').exists()
    assert {p.name: sha(p) for p in destination.iterdir()} == before


@pytest.mark.parametrize('change', [{'role': 'anomaly_dev'}, {'candidates': [2, 4]},
                                   {'batch_size': 512}, {'measured_runs': 16}])
def test_benchmark_plan_cannot_expand_or_use_development_truth(change):
    from learning.parallel_replay import validate_benchmark_plan

    plan = dict(schema='drone-ai.parallel-replay-benchmark/1', candidates=[2, 4, 8],
                warmup_runs=8, measured_runs=8, repeats=2, batch_size=256,
                role='normal_validation', run_id='normal', model='fixture',
                selection='lowest mean measured seconds; declared-order ties')
    validate_benchmark_plan(plan)
    plan.update(change)
    with pytest.raises(ValueError):
        validate_benchmark_plan(plan)
