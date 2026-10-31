"""Build a new portable probe without modifying the frozen GPU package."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / 'x500-gpu-probe-v9.tar.gz'
    if archive.exists():
        raise FileExistsError('Existing release is not overwritten')
    frozen = ROOT / 'ai/artifacts/gpu-handoff-v6/release/x500-gpu-smoke-v1.tar.gz'
    expected = '99b5a495d9d17b75daeb0681b45ee0f649c1cb5f096ad067eb649250a69014e9'
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen v6 archive changed')
    entries = {'AGENTS.md': (ROOT / 'AGENTS.md').read_bytes(),
               frozen.name: frozen.read_bytes(),
               frozen.name + '.sha256': (expected + '  ' + frozen.name + '\n').encode()}
    for file in sorted((ROOT / 'ai/remote').iterdir()):
        if file.is_file() and file.suffix in {'.py', '.sh', '.md', '.json'}:
            entries['remote/' + file.name] = file.read_bytes()
    for file in sorted((ROOT / 'ai/src/capstone_ai').glob('*.py')):
        entries['current-ai/src/capstone_ai/' + file.name] = file.read_bytes()
    for name in ['pyproject.toml', 'requirements.lock']:
        entries['current-ai/' + name] = (ROOT / 'ai' / name).read_bytes()
    for name in ['AGENTS.md', '00_SCOPE_AND_REVIEW.md', '00_SKILL_LINKS.md',
                 '00_SHARED_DATA_CONTRACT.md', '00_SOURCE_NOTES.md', '02_AI_PREPROCESSING_CODEX_SPEC.md']:
        entries['CODEX_REBUILD_PACKAGE_V4_2/' + name] = (ROOT / 'CODEX_REBUILD_PACKAGE_V4_2' / name).read_bytes()
    unchanged = []
    for file in (frozen.parent / 'x500-gpu-smoke-v1/ai/src/capstone_ai').glob('*.py'):
        if file.read_bytes() != (ROOT / 'ai/src/capstone_ai' / file.name).read_bytes():
            raise ValueError('Current neural/shared source differs: ' + file.name)
        unchanged.append(file.name)
    entries['MANIFEST.json'] = (json.dumps({
        'purpose': 'single allocated-GPU arithmetic and frozen LSTM one-batch probe',
        'resource_limits': {'partition': 'p02', 'node': 'DIS01', 'nodes': 1, 'tasks': 1, 'gpus': 1,
                            'concurrent_project_gpu_jobs': 1},
        'node_change': 'User explicitly replaced DIS05 with DIS01 in this task',
        'runtime_change': 'User explicitly approved adding mamba25 without changing frozen torch21 evidence',
        'shared_source_unchanged': sorted(unchanged),
        'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                  for name, data in sorted(entries.items())}}, indent=2) + '\n').encode()
    entries['SHA256SUMS'] = ''.join(hashlib.sha256(data).hexdigest() + '  ' + name + '\n'
                                  for name, data in sorted(entries.items())).encode()
    with archive.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for name, data in sorted(entries.items()):
                info = tarfile.TarInfo('x500-gpu-probe-v9/' + name)
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + '.sha256').write_text(digest + '  ' + archive.name + '\n')
    print(json.dumps({'archive': str(archive), 'sha256': digest, 'bytes': archive.stat().st_size,
                      'files': len(entries), 'shared_source_unchanged': len(unchanged)}))


if __name__ == '__main__':
    main()
