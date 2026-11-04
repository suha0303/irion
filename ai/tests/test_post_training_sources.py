"""Preserved signed-runner and baseline provenance; CPU-only, no new fits."""
import copy

import pytest
from test_post_training import fresh_config

from execution.common import sha, write
from learning import comparison, study
from learning import post_training as post


def baseline_reference():
    return dict(source='/preserved/baselines', models=list(post.BASELINES),
                files={'input/config.json': 'cfg', 'compute/result-seal.json': 'seal'},
                config_sha256='cfg', seal_sha256='seal')


def test_study_baselines_require_explicit_provenance():
    cfg = dict(fresh_config(), source_comparison_revision=study.REVISION,
               models=list(study.MODELS) + list(post.BASELINES))
    with pytest.raises(ValueError):
        post.config_check(cfg)
    cfg['baseline_reference'] = baseline_reference()
    post.config_check(cfg)
    cfg['models'].append(comparison.VARIANTS[0]['id'])
    with pytest.raises(ValueError):
        post.config_check(cfg)


@pytest.mark.parametrize('change', [
    {'models': ['persistence']}, {'config_sha256': 'changed'},
    {'seal_sha256': 'changed'}, {'files': {}}, {'source': '../unsafe'},
])
def test_baseline_reference_cannot_silently_change(change):
    ref = dict(baseline_reference(), **change)
    cfg = dict(fresh_config(), source_comparison_revision=study.REVISION,
               models=list(study.MODELS) + list(post.BASELINES), baseline_reference=ref)
    with pytest.raises(ValueError):
        post.config_check(cfg)


@pytest.fixture
def signed(tmp_path):
    source = tmp_path / 'trained'
    write(source / 'input/ai/frozen.py', {'source': 'frozen'})
    cfg = dict(revision=study.REVISION, source=str(source / 'input'), main_training_authorized=True,
               learning_rate_selection_sha256='lr', seed=0, batch_size=256,
               precision='float32', epochs=60, early_stopping_patience=10,
               binding={'sources': {'ai/frozen.py': sha(source / 'input/ai/frozen.py')}})
    write(source / 'input/config.json', cfg)
    write(source / 'input/inventory.json', {'config.json': sha(source / 'input/config.json')})
    for n in ('worker.py', 'control.py'):
        (source / n).write_text('# preserved runner\n')
    write(source / 'server-profile.json', dict(runtime_profile='mamba25', slurm=dict(
        partition='p02', node='DIS01', nodes=1, ntasks=1, gpus=1, cpus=4, memory_gib=64)))
    auth = dict(main_training_authorized=True, old_lr_selection_sha256='lr',
                seed=0, batch_size=256, precision='float32', epochs_max=60, patience=10,
                worker_sha256=sha(source / 'worker.py'), controller_sha256=sha(source / 'control.py'),
                source_profile_sha256=sha(source / 'server-profile.json'))
    for key in ('learning_rate_search_authorized', 'calibration_authorized', 'evaluation_authorized',
                'final_test_authorized', 'automatic_retry', 'next_round_authorized',
                'llm_monitoring', 'llm_auto_resume'):
        auth[key] = False
    return source, cfg, auth


def test_signed_preparation_without_fabricating_evidence(signed):
    source, cfg, auth = signed
    before = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    assert post.training_preparation_proof(source, cfg, auth) == ['worker.py', 'control.py']
    assert not (source / 'prepared.json').exists()
    assert before == {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('name', ['worker.py', 'control.py', 'server-profile.json', 'input/ai/frozen.py', 'input/config.json'])
def test_signed_preparation_rejects_mutated_files(signed, name):
    source, cfg, auth = signed
    with (source / name).open('a') as f:
        f.write('\nchanged')
    with pytest.raises(ValueError):
        post.training_preparation_proof(source, cfg, auth)


@pytest.mark.parametrize('key', ['main_training_authorized', 'old_lr_selection_sha256', 'seed', 'batch_size',
                                'precision', 'epochs_max', 'patience', 'final_test_authorized', 'automatic_retry'])
def test_signed_preparation_rejects_changed_authorization(signed, key):
    source, cfg, auth = signed
    auth[key] = 'changed'
    with pytest.raises(ValueError):
        post.training_preparation_proof(source, cfg, auth)


@pytest.mark.parametrize('field', ['upstream', 'source_dataset_sha256', 'dataset_files', 'handoff_sha256', 'scaler', 'data', None])
def test_baselines_require_same_frozen_inputs(monkeypatch, field):
    current = dict(revision=study.REVISION, upstream='/upstream', source_dataset_sha256='dataset',
                   dataset_files={'file': 'digest'}, handoff_sha256='handoff',
                   binding={'data': {'train': 'train'}, 'scaler': 'scaler'})
    old = copy.deepcopy(current)
    old.update(revision=comparison.REVISION, models=list(comparison.MODELS))
    if field in {'scaler', 'data'}:
        old['binding'][field] = 'changed'
    elif field is not None:
        old[field] = 'changed'
    seal = {'files': {f'models/{n}/scaler.npz': 'scaler' for n in post.BASELINES}}
    monkeypatch.setattr(post, 'source_proof', lambda *a: (old, seal, {'proof': 'hash'}))
    if field is None:
        assert post.baseline_source_proof(current, '/baseline')[1] == seal
    else:
        with pytest.raises(ValueError, match='binding differs'):
            post.baseline_source_proof(current, '/baseline')


def test_post_training_watch_counts_models_and_release(tmp_path):
    from learning import operator

    cfg = dict(revision=study.REVISION, execution_scope=post.SCOPE,
               models=list(study.MODELS) + list(post.BASELINES), seed=0)
    write(tmp_path / 'input/config.json', cfg)
    write(tmp_path / 'controller.json', dict(status='REPORT_READY_STOPPED', stage='stopped'))
    write(tmp_path / 'completion.json', dict(status='REPORT_READY_STOPPED',
          calibration_release={'released': True}, evaluation_release={'released': True}))
    before = {str(p.relative_to(tmp_path)): sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    screen = operator.render(tmp_path)
    assert '신경망16 · 전체18' in screen
    assert '18종 공통 수치 검사' in screen
    assert '실제 할당 반납·평가 완료 확인됨' in screen
    assert before == {str(p.relative_to(tmp_path)): sha(p) for p in tmp_path.rglob('*') if p.is_file()}


def test_cross_node_baseline_reference_does_not_replay_login_storage_device(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    write(baseline / 'storage-policy.json', dict(schema='x500.storage/1', root=str(baseline),
                                                mount='/abr', device=119))
    ref = dict(baseline_reference(), source=str(baseline))
    cfg = dict(fresh_config(), source_comparison_revision=study.REVISION,
               models=list(study.MODELS) + list(post.BASELINES), baseline_reference=ref)

    def incompatible_login_policy(*args):
        raise ValueError('Selected storage filesystem unavailable/changed; no fallback')

    monkeypatch.setattr(post, 'external_root', incompatible_login_policy)
    post.config_check(cfg)
    assert (baseline / 'storage-policy.json').is_file()


def test_cross_node_reference_still_rejects_symlinks(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    link = tmp_path / 'baseline-link'
    link.symlink_to(original, target_is_directory=True)
    cfg = dict(fresh_config(), source_comparison_revision=study.REVISION,
               models=list(study.MODELS) + list(post.BASELINES),
               baseline_reference=dict(baseline_reference(), source=str(link)))
    with pytest.raises(ValueError, match='symlink'):
        post.config_check(cfg)


def test_checked_still_verifies_original_baseline_provenance(tmp_path, monkeypatch):
    root, source, baseline = (tmp_path / n for n in ('evaluation', 'neural', 'baseline'))
    source.mkdir()
    write(baseline / 'input/config.json', {'frozen': 'baseline config'})
    write(baseline / 'compute/result-seal.json', {'frozen': 'baseline seal'})
    ref = dict(baseline_reference(), source=str(baseline),
               files={n: sha(baseline / n) for n in ('input/config.json', 'compute/result-seal.json')},
               config_sha256=sha(baseline / 'input/config.json'),
               seal_sha256=sha(baseline / 'compute/result-seal.json'))
    cfg = dict(fresh_config(), source_comparison_revision=study.REVISION,
               models=list(study.MODELS) + list(post.BASELINES), baseline_reference=ref,
               source_training=str(source), source_training_files={}, model_files={})
    write(root / 'input/config.json', cfg)
    write(root / 'input/inventory.json', {'config.json': sha(root / 'input/config.json')})
    write(root / 'server-profile.json', {})
    write(root / 'authorization.json', dict(config_sha256=sha(root / 'input/config.json'),
                                          execution_scope=post.SCOPE))
    write(root / 'prepared.json', dict(input_inventory_sha256=sha(root / 'input/inventory.json'),
          server_profile_sha256=sha(root / 'server-profile.json'),
          authorization_sha256=sha(root / 'authorization.json')))
    monkeypatch.setattr(post, 'runtime_profile', lambda *a, **k: {})
    monkeypatch.setattr(post, 'profile_check', lambda *a: None)
    assert post.checked(root)[1] == cfg
    write(baseline / 'input/config.json', {'changed': True})
    with pytest.raises(ValueError, match='Input hash changed'):
        post.checked(root)
