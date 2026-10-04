"""CPU-only synthetic admission, export, cap and cleanup checks; no SITL."""
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('collector', Path(__file__).parents[2] / 'Data/generation/tools/collection_support.py')
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def fixture_run(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    for name in ('observations', 'reference', 'evaluation', 'px4'):
        (run / name).mkdir()
    (run / 'run.json').write_text('{"status":"completed"}')
    (run / 'observations/streams.jsonl').write_text('{"x":1}\n')
    (run / 'reference/profile.json').write_text('{}')
    (run / 'evaluation/scenario.json').write_text('{}')
    (run / 'px4/native.ulg').write_bytes(b'native bytes')
    inventory = {str(p.relative_to(run)): {'size': p.stat().st_size, 'sha256': c.sha(p)}
                 for p in run.rglob('*') if p.is_file()}
    (run / 'integrity.json').write_text(json.dumps(inventory))
    return run


def test_export_hashes_and_never_follows_native_symlink(tmp_path):
    run = fixture_run(tmp_path)
    outside = tmp_path / 'outside-dir'
    outside.mkdir()
    secret = outside / 'outside'
    secret.write_text('outside bytes must not be archived')
    (run / 'px4/external').symlink_to(outside)
    before = c.sha(secret)
    result = c.export_run(run, tmp_path / 'export', lambda: None)
    with tarfile.open(result['archive']) as tf:
        member = tf.getmember('px4/external')
        assert member.issym() and member.linkname == str(outside)
    assert not (Path(result['ai_run']) / 'px4/external').exists()
    assert (Path(result['ai_run']) / 'px4/native.ulg').read_bytes() == b'native bytes'
    assert c.sha(Path(result['ai_run']) / 'observations/streams.jsonl') == c.sha(run / 'observations/streams.jsonl')
    assert c.sha(secret) == before and run.exists()
    with pytest.raises(FileExistsError):
        c.export_run(run, tmp_path / 'export', lambda: None)


def test_ai_boundary_symlink_rejected_partial_preserved(tmp_path):
    run = fixture_run(tmp_path)
    (run / 'reference/link').symlink_to(run / 'run.json')
    with pytest.raises(c.Stop, match='symlink'):
        c.export_run(run, tmp_path / 'export', lambda: None)
    assert (tmp_path / '.export-evidence/export').exists()
    assert not (tmp_path / 'export').exists()
    assert run.exists()


def test_export_detects_source_mutation(tmp_path):
    run = fixture_run(tmp_path)
    changed = False
    def mutate():
        nonlocal changed
        if not changed:
            (run / 'observations/streams.jsonl').write_text('changed')
            changed = True
    with pytest.raises(c.Stop, match='mismatch|changed'):
        c.export_run(run, tmp_path / 'export', mutate)
    assert (tmp_path / '.export-evidence/export/run.tar').exists()


def test_corrupt_external_export_not_published(tmp_path):
    run = fixture_run(tmp_path)
    def corrupt(path):
        (path / 'observations/streams.jsonl').write_text('corrupt')
    with pytest.raises(c.Stop, match='changed before publication'):
        c.export_run(run, tmp_path / 'export', lambda: None, corrupt)
    assert not (tmp_path / 'export').exists()
    assert (tmp_path / '.export-evidence/export').exists()


def test_original_integrity_mismatch_rejected(tmp_path):
    run = fixture_run(tmp_path)
    (run / 'px4/native.ulg').write_bytes(b'tampered')
    with pytest.raises(c.Stop, match='Original integrity mismatch'):
        c.export_run(run, tmp_path / 'export', lambda: None)
    assert not (tmp_path / 'export').exists()


@pytest.mark.parametrize('args', [(100, 0, 1000, 1000, 0), (0, 100, 1000, 1000, 0),
                                  (0, 0, 49, 1000, 0), (0, 0, 1000, 49, 0),
                                  (0, 0, 1000, 1000, 10)])
def test_limits_stop(args):
    budget = dict(local_added_bytes_cap=100, external_added_bytes_cap=100,
                  min_free_bytes=50, campaign_wall_s=10)
    with pytest.raises(c.Stop):
        c.check_limits(*args, budget)


def test_missing_start_rejected_before_output(tmp_path):
    p, m, r = (tmp_path / n for n in ('plan', 'cases', 'receipt'))
    for file in (p, m, r):
        file.write_text('{}')
    with pytest.raises(c.Stop, match='START'):
        c.validate(p, m, r)
    assert len(list(tmp_path.iterdir())) == 3


def test_hash_bound_start_accepts_fixed_plan_then_rejects_drift(tmp_path):
    base = Path(__file__).parent / 'fixtures/collector-1gb-contract'
    plan = tmp_path / 'plan.json'
    cases = tmp_path / 'case_manifest.json'
    coordinator = tmp_path / 'coordinator.yaml'
    coordinator.write_bytes((base / 'coordinator-plan.yaml').read_bytes())
    fixture_plan = json.loads((base / 'plan.json').read_text())
    fixture_plan['coordinator_plan'] = dict(path=str(coordinator), sha256=c.sha(coordinator))
    plan.write_text(json.dumps(fixture_plan))
    cases.write_bytes((base / 'case_manifest.json').read_bytes())
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps(dict(action='START', revision='R2_1GB', authorized=True,
        plan_sha256=c.sha(plan), case_manifest_sha256=c.sha(cases), wrapper_sha256=c.sha(c.__file__))))
    actual, selected = c.validate(plan, cases, receipt)
    assert len(selected['cases']) == 30
    assert actual['budget']['retries'] == 0
    original = coordinator.read_bytes()
    coordinator.write_bytes(original + b'\n')
    with pytest.raises(c.Stop, match='Coordinator plan drift'):
        c.validate(plan, cases, receipt)
    coordinator.write_bytes(original)
    cases.write_text(cases.read_text() + '\n')
    with pytest.raises(c.Stop, match='START'):
        c.validate(plan, cases, receipt)


def test_command_stop_and_owned_cleanup(tmp_path):
    log = tmp_path / 'command.log'
    with pytest.raises(c.Stop, match='wall cap'):
        c.command([sys.executable, '-c', 'import subprocess,time; subprocess.Popen(["sleep","60"],start_new_session=True); time.sleep(60)'],
                  tmp_path, log, lambda: None, 0.8)
    evidence = json.loads(Path(str(log) + '.processes.json').read_text())
    assert evidence['remaining'] == []
    assert len(evidence['owned']) >= 2


def test_pid_identity_mismatch_not_signalled(monkeypatch):
    monkeypatch.setattr(c, 'identity', lambda pid: dict(pid=pid, uid=1, start=2))
    monkeypatch.setattr(c.os, 'pidfd_open', lambda pid: pytest.fail('must not signal reused pid'))
    c.send_owned(dict(pid=123, uid=1, start=1), 15)


def test_first_failure_stops_preserves_local_partial(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    (root / 'runs/old').mkdir(parents=True)
    (root / 'runs/old/keep').write_text('unchanged')
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'raw').mkdir()  # coordinator pre-created empty reservation
    monkeypatch.setattr(c, 'check_mount', lambda *args, **kwargs: 123)
    work = tmp_path / 'work'
    plan = dict(root=str(root), external_root=str(external), local_work=str(work),
                producer_files_sha256={}, budget=dict(campaign_wall_s=100,
                local_added_bytes_cap=10**8, external_added_bytes_cap=10**8, min_free_bytes=0))
    calls = []
    def fail(*args):
        calls.append(args)
        (root / 'runs/newpartial').mkdir()
        (root / 'runs/newpartial/partial').write_text('preserve')
        raise c.Stop('synthetic first failure')
    monkeypatch.setattr(c, 'command', fail)
    case = dict(scenario='hover_normal', seed=31011, role='normal_train')
    with pytest.raises(c.Stop, match='first failure'):
        c.collect(plan, {'cases': [case, case]})
    assert len(calls) == 1
    assert (root / 'runs/old/keep').read_text() == 'unchanged'
    assert (root / 'runs/newpartial/partial').read_text() == 'preserve'
    assert json.loads((work / 'ledger.json').read_text())['status'] == 'FAILED_STOPPED'


def test_empty_raw_reservation_and_nonempty_rejection(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    c.reserve_raw(raw)
    (raw / 'keep').write_text('preserve')
    with pytest.raises(c.Stop, match='empty'):
        c.reserve_raw(raw)
    assert (raw / 'keep').read_text() == 'preserve'


def test_mount_loss_and_symlink_path_stop(tmp_path, monkeypatch):
    mount = tmp_path / 'disk'
    external = mount / 'Data/X500'
    external.mkdir(parents=True)
    monkeypatch.setattr(c.os.path, 'ismount', lambda p: True)
    device = c.check_mount(external, mount)
    monkeypatch.setattr(c.os.path, 'ismount', lambda p: False)
    with pytest.raises(c.Stop, match='mount absent'):
        c.check_mount(external, mount, device)
    monkeypatch.setattr(c.os.path, 'ismount', lambda p: True)
    alias = mount / 'alias'
    alias.symlink_to(mount / 'Data', target_is_directory=True)
    with pytest.raises(c.Stop, match='symlink'):
        c.check_mount(alias / 'X500', mount)
