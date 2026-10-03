"""Finite R2 collection/export. No execution without a hash-bound START receipt."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import tarfile
import time
from pathlib import Path


class Stop(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    # Keep failed/partial evidence; never overwrite raw data.
    with Path(path).open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def entries(root):
    """lstat traversal: symlink directories/files are never followed."""
    for entry in sorted(Path(root).iterdir()):
        mode = entry.lstat().st_mode
        yield entry, mode
        if stat.S_ISDIR(mode):
            yield from entries(entry)


def size(root):
    if not Path(root).exists():
        return 0
    return sum(p.lstat().st_size for p, mode in entries(root)
               if stat.S_ISREG(mode))


def snapshot(run):
    result = {}
    for p, mode in entries(run):
        rel = str(p.relative_to(run))
        if stat.S_ISREG(mode):
            result[rel] = {'kind': 'file', 'bytes': p.stat().st_size, 'sha256': sha(p)}
        elif stat.S_ISLNK(mode):
            if not p.is_dir():
                raise Stop(f'File/broken symlink requires review: {p}')
            result[rel] = {'kind': 'symlink', 'target': os.readlink(p)}
        elif stat.S_ISDIR(mode):
            result[rel] = {'kind': 'directory'}
        else:
            raise Stop(f'Unsupported special file: {p}')
    return result


def export_run(run, dest, check, verify_export=None):
    """Archive links as links; AI export is regular files only. Never extract tar."""
    run, dest = Path(run), Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    evidence_root = dest.parent / '.export-evidence'
    evidence_root.mkdir(exist_ok=True)
    staging = evidence_root / dest.name
    staging.mkdir()  # exclusive; failed stages remain, no resume/overwrite
    before = snapshot(run)
    integrity = json.loads((run / 'integrity.json').read_text())
    for rel, expected in integrity.items():
        info = before.get(rel)
        if not info or info.get('kind') != 'file' or (info['bytes'], info['sha256']) != (expected['size'], expected['sha256']):
            raise Stop(f'Original integrity mismatch: {rel}')
    archive = staging / 'run.tar'
    class CheckedReader:
        def __init__(self, stream):
            self.stream = stream

        def read(self, length):
            check()
            return self.stream.read(length)

    with tarfile.open(archive, 'x', dereference=False) as tf:
        for p, mode in entries(run):
            check()
            member = tf.gettarinfo(p, arcname=str(p.relative_to(run)))
            if member.isfile():
                with p.open('rb') as stream:
                    tf.addfile(member, CheckedReader(stream))
            else:
                tf.addfile(member)
    with archive.open('rb') as f:
        os.fsync(f.fileno())
    # Hash every archive member against local bytes; no path extraction.
    archived = {}
    with tarfile.open(archive, 'r:') as tf:
        for member in tf:
            check()
            if member.isfile():
                stream = tf.extractfile(member)
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    check()
                    h.update(block)
                archived[member.name] = {'kind': 'file', 'bytes': member.size,
                                         'sha256': h.hexdigest()}
            elif member.issym():
                archived[member.name] = {'kind': 'symlink', 'target': member.linkname}
            elif member.isdir():
                archived[member.name] = {'kind': 'directory'}
            else:
                raise Stop('Unexpected archive member type')
    if archived != before:
        raise Stop('Archive content/hash mismatch')
    ai = staging / 'regular' / run.name
    ai.mkdir(parents=True)
    exported = {}
    for rel, info in before.items():
        if info['kind'] != 'file':
            continue  # Directory symlinks are archived only, never traversed.
        check()
        src, dst = run / rel, ai / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as inp, dst.open('xb') as output:
            while block := inp.read(1024 * 1024):
                check()
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if dst.stat().st_size != info['bytes'] or sha(dst) != info['sha256']:
            raise Stop(f'Export hash mismatch: {rel}')
        exported[rel] = info
    if snapshot(run) != before:
        raise Stop('Local stage changed during export')
    if sha(ai / 'integrity.json') != sha(run / 'integrity.json'):
        raise Stop('Original integrity file changed')
    if verify_export:
        verify_export(ai)
    for rel, info in exported.items():
        check()
        if sha(ai / rel) != info['sha256']:
            raise Stop(f'Export changed before publication: {rel}')
    final_run = dest
    result = {'local_run': str(run), 'archive': str(archive),
              'archive_sha256': sha(archive), 'ai_run': str(final_run),
              'observations': str(final_run / 'observations'),
              'evaluation_truth_restricted': str(final_run / 'evaluation'),
              'regular_files': exported, 'archive_entries': before,
              'truth_policy': 'evaluation is evaluator-only; never neural input',
              'local_preserved': True, 'status': 'HASH_VERIFIED'}
    save(staging / 'export-manifest.json', result)
    check()
    if dest.exists():
        raise Stop('Publish destination appeared; no overwrite')
    ai.rename(dest)  # same external filesystem; never local-to-external rename
    return result


def identity(pid):
    try:
        path = Path('/proc') / str(pid)
        tail = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'ppid': int(tail[1]), 'start': int(tail[19]),
                'uid': path.stat().st_uid, 'state': tail[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(saved):
    current = identity(saved['pid'])
    return current and all(current[k] == saved[k] for k in ('pid', 'start', 'uid'))


def track(proc, owned):
    allp = [identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    allp = [p for p in allp if p and p['uid'] == os.getuid()]
    ancestors = {p['pid'] for p in owned.values() if same_process(p)}
    changed = True
    while changed:
        changed = False
        for p in allp:
            if p['ppid'] in ancestors and p['pid'] not in ancestors:
                ancestors.add(p['pid'])
                owned[p['pid']] = p
                changed = True


def send_owned(saved, sig):
    # pidfd binds the signal to the verified process, not a reused PID/group.
    if not same_process(saved):
        return
    try:
        fd = os.pidfd_open(saved['pid'])
        try:
            if same_process(saved) and saved['uid'] == os.getuid():
                signal.pidfd_send_signal(fd, sig)
        finally:
            os.close(fd)
    except ProcessLookupError:
        pass


def live_owned(owned):
    live = []
    for p in owned.values():
        current = identity(p['pid'])
        if current and all(current[k] == p[k] for k in ('pid', 'start', 'uid')) and current['state'] != 'Z':
            live.append(p)
    return live


def command(args, cwd, log, check, seconds):
    proc = None
    owned = {}
    started = time.monotonic()
    try:
        with Path(log).open('xb') as output:
            proc = subprocess.Popen(args, cwd=cwd, stdout=output,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            initial = identity(proc.pid)
            if initial:
                owned[proc.pid] = initial
            while proc.poll() is None:
                track(proc, owned)
                check()
                if time.monotonic() - started > seconds:
                    raise Stop('Per-command wall cap')
                time.sleep(0.25)
            track(proc, owned)
            if proc.returncode:
                raise Stop(f'Command exit {proc.returncode}: {args[0]}')
            if live_owned(owned):
                raise Stop('Producer left live owned descendants')
    finally:
        if proc:
            # SIGINT lets the existing Python runner execute its finally cleanup.
            if proc.poll() is None and proc.pid in owned:
                send_owned(owned[proc.pid], signal.SIGINT)
                deadline = time.monotonic() + 12
                while proc.poll() is None and time.monotonic() < deadline:
                    track(proc, owned)
                    time.sleep(0.1)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for p in reversed(list(owned.values())):
                    send_owned(p, sig)
                time.sleep(0.1)
            proc.wait(timeout=5)
            save(str(log) + '.processes.json', {'owned': list(owned.values()),
                 'exit_code': proc.returncode, 'remaining': live_owned(owned)})
            if live_owned(owned):
                raise Stop('Owned process cleanup incomplete')


def check_limits(local_bytes, external_bytes, local_free, external_free, elapsed, budget):
    if elapsed >= budget['campaign_wall_s']:
        raise Stop('Campaign wall cap')
    if local_bytes >= budget['local_added_bytes_cap'] or external_bytes >= budget['external_added_bytes_cap']:
        raise Stop('Output cap')
    if min(local_free, external_free) < budget['min_free_bytes']:
        raise Stop('Free-space floor')


def check_mount(external, mount=Path('/media/sa/X31'), expected_device=None):
    external, mount = Path(external), Path(mount)
    if not external.is_absolute() or mount not in external.parents:
        raise Stop('External path outside designated mount')
    # Check every component, including /media and the mount itself.
    for part in [*reversed(external.parents), external]:
        if part.is_symlink() or not part.is_dir():
            raise Stop(f'External path missing/not directory/symlink: {part}')
    if not os.path.ismount(mount):
        raise Stop('External mount absent; no internal-disk fallback')
    device = mount.stat().st_dev
    if external.stat().st_dev != device or (expected_device is not None and device != expected_device):
        raise Stop('External device changed')
    return device


def reserve_raw(path):
    path = Path(path)
    if path.is_symlink():
        raise Stop('raw reservation is symlink')
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise Stop('raw reservation must be empty')
    else:
        path.mkdir()


def validate(plan_path, cases_path, receipt_path):
    plan = json.loads(Path(plan_path).read_text())
    cases = json.loads(Path(cases_path).read_text())
    receipt = json.loads(Path(receipt_path).read_text())
    expected = {'action': 'START', 'revision': 'R2_1GB', 'authorized': True,
                'plan_sha256': sha(plan_path), 'case_manifest_sha256': sha(cases_path),
                'wrapper_sha256': sha(__file__)}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise Stop('Missing/mismatched hash-bound coordinator START receipt')
    if plan['case_manifest_sha256'] != sha(cases_path):
        raise Stop('Case manifest drift')
    if plan.get('coordinator_plan'):
        if sha(plan['coordinator_plan']['path']) != plan['coordinator_plan']['sha256']:
            raise Stop('Coordinator plan drift')
    counts = {}
    groups = {}
    for c in cases['cases']:
        counts[c['role']] = counts.get(c['role'], 0) + 1
        groups.setdefault(c['group_id'], set()).add(c['split'])
        if c['mission_seconds'] != 180 or c['retries'] != 0:
            raise Stop('Case duration/retry drift')
    if counts != dict(normal_train=9, normal_validation=3, normal_calibration=3,
                      normal_dev_eval=3, anomaly_dev=12) or any(len(g) != 1 for g in groups.values()):
        raise Stop('Role/lineage/count drift')
    if len({c['case_id'] for c in cases['cases']}) != 30:
        raise Stop('Duplicate case ID')
    if [c['generation_index'] for c in cases['cases']] != list(range(1, 31)):
        raise Stop('Case order drift')
    if any(c['role'] == 'anomaly_dev' for c in cases['cases'][:18]) or any(c['role'] != 'anomaly_dev' for c in cases['cases'][18:]):
        raise Stop('Normal18/anomaly12 phase drift')
    if len(groups) != 18:
        raise Stop('Independent group count drift')
    return plan, cases


def collect(plan, cases):
    root = Path(plan['root'])
    external = Path(plan['external_root'])
    work = Path(plan['local_work'])
    if external.is_symlink() or not external.is_dir():
        raise Stop('External root missing or symlink')
    external_device = check_mount(external)
    baseline = {p.name for p in (root / 'runs').iterdir() if p.is_dir()}
    if 'baseline_run_ids' in plan and baseline != set(plan['baseline_run_ids']):
        raise Stop('Run inventory changed since preparation')
    requested_seeds = {c['seed'] for c in cases['cases']}
    for name in baseline:
        source = root / 'runs' / name / 'evaluation/scenario.json'
        if source.exists() and json.loads(source.read_text()).get('seed') in requested_seeds:
            raise Stop('Proposed seed already used')
    work.mkdir()  # no retries/resume of a partial campaign
    ledger = {'status': 'RUNNING', 'baseline_run_ids': sorted(baseline), 'attempts': []}
    start = time.monotonic()
    case_start = None
    external_out = external / 'raw'
    def new_runs():
        return [p for p in (root / 'runs').iterdir() if p.is_dir() and p.name not in baseline]
    def check():
        check_mount(external, expected_device=external_device)
        if external_out.is_symlink():
            raise Stop('External raw path became symlink')
        if case_start is not None and time.monotonic() - case_start >= 360:
            raise Stop('Per-case total wall cap including verification/export')
        check_limits(size(work) + sum(size(p) for p in new_runs()), size(external_out),
                     shutil.disk_usage(root).free, shutil.disk_usage(external).free,
                     time.monotonic() - start, plan['budget'])
    def locks():
        for rel, expected in plan['producer_files_sha256'].items():
            if sha(root / rel) != expected:
                raise Stop(f'Source lock drift: {rel}')
    try:
        check()
        reserve_raw(external_out)
        check()
        locks()
        save(work / 'baseline.json', ledger)
        for index, case in enumerate(cases['cases']):
            check()
            locks()
            if index >= 30:
                raise Stop('Attempt cap')
            before = {p.name for p in new_runs()}
            case_start = time.monotonic()
            attempt = {'case': case, 'status': 'STARTED'}
            ledger['attempts'].append(attempt)
            save(work / 'ledger.json', ledger)
            args = [str(root / 'capstone-sim'), 'run', '--scenario', case['scenario'],
                    '--seed', str(case['seed']), '--duration', '180', '--instance', '3',
                    '--timing', '--capture-interval-ms', '5']
            if case.get('fault_at_s') is not None:
                args += ['--fault-at-s', str(case['fault_at_s'])]
            command(args, root, work / f'{index:02d}-producer.log', check, 360)
            locks()
            added = [p for p in new_runs() if p.name not in before]
            if len(added) != 1:
                raise Stop('Expected exactly one new run; concurrent/unknown activity')
            run = added[0]
            state = json.loads((run / 'run.json').read_text())
            scenario = json.loads((run / 'evaluation/scenario.json').read_text())
            if state['status'] != 'completed' or state.get('parent_attempt_id'):
                raise Stop('Producer incomplete or unexpected ancestry')
            if (scenario['name'], scenario['seed']) != (case['scenario'], case['seed']):
                raise Stop('Case identity mismatch')
            mission = case['scenario_parameters_snapshot'].get('mission', 'hover')
            if scenario.get('paired_case_id') != f'x500-{mission}-seed{case["seed"]}':
                raise Stop('Paired identity mismatch')
            if scenario.get('continuity') != 'independent_reset':
                raise Stop('Unexpected continuation')
            if case['role'] != 'anomaly_dev' and scenario['health_eligibility'] != 'confirmed_healthy':
                raise Stop('Required normal health eligibility failed')
            command([str(root / 'capstone-sim'), 'verify', '--run', str(run)], root,
                    work / f'{index:02d}-verify.log', check, 360)
            def verify_external(path):
                command([str(root / 'capstone-sim'), 'verify', '--run', str(path)], root,
                        work / f'{index:02d}-external-verify.log', check, 360)
            attempt.update(run_id=run.name, export=export_run(run, external_out / run.name, check, verify_external),
                           status='COLLECTED_PENDING_AI_AUDIT')
            save(work / 'ledger.json', ledger)
            check()
            save(external_out / 'path-manifest.json', ledger)
            if index == 17:
                # This is producer/transport admission, not AI feature quality.
                gate = {'status': 'NORMAL18_PRODUCER_EXPORT_CHECKS_PASSED',
                        'checks': ['completed', 'confirmed_healthy', 'case/seed/paired/reset',
                                   'local verify', 'archive/regular hash readback', 'external verify'],
                        'ai_feature_quality_checked': False,
                        'context_windows_monitoring_seconds_checked': False,
                        'note': 'Full AI quality/lineage audit required before DATA_FROZEN/training.'}
                save(work / 'normal18-gate.json', gate)
                save(external_out / 'normal18-gate.json', gate)
            case_start = None
        locks()
        check()
        ledger['status'] = 'COLLECTED_PENDING_AI_AUDIT_STOPPED'
    except BaseException as exc:
        ledger.update(status='FAILED_STOPPED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        ledger['elapsed_s'] = time.monotonic() - start
        ledger['preserved_new_local_runs'] = [str(p) for p in new_runs()]
        save(work / 'ledger.json', ledger)
        # Local failure evidence survives external disconnect/read-only/full errors.
        try:
            check_mount(external, expected_device=external_device)
            if external_out.is_dir():
                save(external_out / 'path-manifest.json', ledger)
        except (OSError, Stop):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--start-receipt', required=True)
    args = parser.parse_args()
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise Stop('Verified process cleanup requires Linux pidfd support')
    def terminate(_signum, _frame):
        raise KeyboardInterrupt('Wrapper terminated; preserve partial and stop')
    signal.signal(signal.SIGTERM, terminate)
    plan, cases = validate(args.plan, args.cases, args.start_receipt)
    lock_path = Path(plan['root']) / 'ai/experiments/ROUND_001/data-1gb-01/collection.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect(plan, cases)


if __name__ == '__main__':
    main()
