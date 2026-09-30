"""Install/check/export a source checkout. Never collect data or submit a trainer."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENV = ROOT / "Data/generation/.venv"
SOURCE_TREES = (
    "Data/generation", "simulation/sim", "simulation/profiles", "simulation/contracts",
    "simulation/config", "simulation/data/public", "ai", "docs", ".github",
)
EXCLUDED = {
    ".git", ".venv", ".python", ".bootstrap", "__pycache__", ".cache", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "old", "archive", "experiments", "reports", "artifacts",
    "long-run-claims", "node_modules", "vendor", "build", "runtime",
}
AI_RUNTIME_DIRECTORIES = {".local", "operations", "outputs", "runs"}
ROOT_FILES = ("README.md", "AGENTS.md", ".gitignore", "simulation/README.md",
              "simulation/capstone-sim", "simulation/requirements.lock", "simulation/pyproject.toml",
              "Data/README.md")
STATIC_ARTIFACTS = ("battery-fit.json", "propulsion-comparison.json", "mass-inertia-ledger.json")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, *, timeout=None, env=None):
    return subprocess.run(command, check=True, cwd=ROOT, timeout=timeout, env=env)


def host_supported():
    release = platform.freedesktop_os_release() if sys.platform == "linux" else {}
    return (sys.platform == "linux" and platform.machine() == "x86_64"
            and release.get("ID") == "ubuntu" and release.get("VERSION_ID") == "24.04")


@contextmanager
def setup_lock():
    """Never replace a data environment while an independent pipeline owns it."""
    import fcntl

    path = ROOT / "Data/operations/pretrain/pretrain.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Data pipeline is active; setup did not change its environment") from exc
        yield


def setup(args):
    if not host_supported():
        raise ValueError("Supported host: Ubuntu 24.04 x86_64 with a local systemd user session")
    with setup_lock():
        return install_environment(args)


def install_environment(args):
    python = shutil.which("python3.12")
    if python is None:
        raise ValueError("python3.12 + python3.12-venv are required; see docs/data/PORTABLE_SETUP.md")
    if ENV.exists() and not (ENV / "bin/python").is_file():
        raise ValueError("Incomplete data environment; preserve/rename it before setup: " + str(ENV))
    if not ENV.exists():
        run([python, "-m", "venv", str(ENV)])
    executable = str(ENV / "bin/python")
    run([executable, "-c", "import sys; assert sys.version_info[:2] == (3,12), sys.version"])
    requirements = ROOT / "Data/generation/requirements.lock"
    run([executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)])
    if args.tests:
        run([executable, "-m", "pip", "install", "--disable-pip-version-check", "-r",
             str(ROOT / "Data/generation/requirements-test.lock")])
    if args.simulation:
        run(["bash", str(ROOT / "simulation/sim/tools/bootstrap.sh")])
    print("설치 종료. 수집·학습을 시작하지 않았습니다.")
    return doctor(argparse.Namespace(processing_only=not args.simulation, json=False))


def probe(command):
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
        return dict(passed=result.returncode == 0, exit_code=result.returncode,
                    detail=(result.stdout + result.stderr).strip()[-4000:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(passed=False, detail=str(exc))


def doctor(args):
    checks = dict(supported_host=dict(passed=host_supported(),
                                     detail="Ubuntu24.04 x86_64; native Linux/systemd/cgroupv2"))
    data_python = ENV / "bin/python"
    code = r'''
import sys, importlib.metadata as md
from pathlib import Path
assert sys.version_info[:2] == (3,12), sys.version
root=Path(sys.argv[1])
for line in (root/'Data/generation/requirements.lock').read_text().splitlines():
    if line.strip() and not line.startswith('#'):
        name, version=line.strip().split('==')
        assert md.version(name)==version, (name, md.version(name), version)
sys.path[:0]=[str(root/'Data/generation'),str(root/'ai')]
from pretrain.common import PACKAGE, package_imports, verify, read
verify({str(PACKAGE/name):digest for name,digest in read(PACKAGE/'PACKAGE.json')['files'].items()})
package_imports()
from capstone_ai.preprocessing import Preprocessor
from pretrain import assembly, worker, streaming, cli
assert 'torch' not in sys.modules, 'Data path must not initialize training dependencies'
print('Python3.12; pinned CPU dependencies; sealed preprocessing; data imports OK; no torch')
'''
    checks["processing_environment"] = probe([str(data_python), "-B", "-c", code, str(ROOT)])
    if not args.processing_only:
        checks["cgroupv2"] = dict(passed=Path("/sys/fs/cgroup/cgroup.controllers").is_file(),
                                   detail="/sys/fs/cgroup/cgroup.controllers")
        checks["user_service_manager"] = probe(["systemctl", "--user", "show-environment"])
        # Environment variable values can contain secrets; expose only the exit status.
        checks["user_service_manager"]["detail"] = "systemctl --user show-environment (values omitted)"
        checks["native_runtime"] = probe([
            str(ROOT / "simulation/.venv/bin/python"), "-B", "-c",
            "import sys;sys.path.insert(0,sys.argv[1]);from capstone_sim import collection_config;"
            "collection_config.assert_frozen();print(collection_config.producer_lock_path())",
            str(ROOT / "simulation/sim/src"),
        ])
        checks["gazebo"] = probe(["pkg-config", "--modversion", "gz-sim8", "gz-transport13"])
    passed = all(row["passed"] for row in checks.values())
    result = dict(status="READY" if passed else "NOT_READY", checks=checks,
                  scope="processing only" if args.processing_only else "collection and processing",
                  flight_started=False, training_started=False)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for name, row in checks.items():
            print(f"{'OK' if row['passed'] else 'FAIL'}  {name}: {row['detail']}")
        print(result["status"])
    return 0 if passed else 3


def source_files(root=ROOT):
    """An explicit source export, independent of Git and all host/job/model state."""
    result = {}

    def add(path):
        relative = path.relative_to(root)
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
            raise ValueError("Symlink in source export: " + str(relative))
        if path.is_file():
            if (path.name.startswith(".env")
                    or path.name in {"id_rsa", "id_ed25519", "known_hosts", "installation.json", "setup-report.json"}
                    or path.suffix in {".pem", ".key", ".pt", ".pth", ".safetensors", ".pyc", ".log"}):
                return
            result[relative.as_posix()] = path

    for name in ROOT_FILES:
        add(root / name)
    for tree in SOURCE_TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for directory, dirs, files in os.walk(base, followlinks=False):
            # Editable/install metadata is host-generated, not cloneable source.
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.startswith(".")
                             and not d.endswith((".egg-info", ".dist-info")))
            if Path(directory) == root / "ai":
                dirs[:] = [d for d in dirs if d not in AI_RUNTIME_DIRECTORIES]
            for name in dirs:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("Symlink directory in source export: " + str(Path(directory) / name))
            for name in sorted(files):
                add(Path(directory) / name)
    for name in STATIC_ARTIFACTS:
        add(root / "simulation/artifacts" / name)
    for path in (root / "simulation/artifacts/sources").glob("*"):
        add(path)
    for path in (root / "simulation/artifacts/fixtures").rglob("*"):
        if path.is_file():
            add(path)
    return dict(sorted(result.items()))


def export(args):
    destination = Path(args.output).expanduser().absolute()
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("Symlink export destination forbidden")
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(ROOT):
        raise ValueError("Export requires a NEW path outside this checkout")
    if destination.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("Export output must end in .tar.gz")
    files = source_files()
    manifest = dict(schema="x500.source-export/1", created_utc=datetime.now(timezone.utc).isoformat(),
                    files={name: sha(path) for name, path in files.items()},
                    includes_user_data=False, includes_models=False, includes_host_runtime=False)
    # Never place job state in the new source tree. Output is exclusive, never overwrites a user file.
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as output, tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name, path in files.items():
                if sha(path) != manifest["files"][name]:
                    raise ValueError("Source changed while exporting: " + name)
                archive.add(path, arcname="Capstone/" + name, recursive=False)
                if sha(path) != manifest["files"][name]:
                    raise ValueError("Source changed while exporting: " + name)
            with tempfile.TemporaryFile() as body:
                content = (json.dumps(manifest, indent=2) + "\n").encode()
                body.write(content)
                body.seek(0)
                info = tarfile.TarInfo("Capstone/SOURCES.json")
                info.size, info.mode = len(content), 0o644
                archive.addfile(info, body)
    except BaseException:
        # Preserve an incomplete export as evidence rather than presenting it as a usable package.
        if destination.exists():
            failed = Path(tempfile.mkdtemp(prefix=destination.name + ".failed-", dir=destination.parent))
            destination.rename(failed / (destination.name + ".incomplete"))
            print("Incomplete export preserved:", failed, file=sys.stderr)
        raise
    print(f"{destination}\n소스 {len(files)}개 · 데이터/모델/작업 상태/환경 제외")
    return 0


def main():
    parser = argparse.ArgumentParser(description="드론 수집·정제 설치/점검/소스 배포 (기본 X500). 수집·학습 자동 시작 없음")
    sub = parser.add_subparsers(dest="action", required=True)
    install = sub.add_parser("setup", help="CPU 전용 데이터 환경 설치")
    install.add_argument("--simulation", action="store_true", help="PX4/Gazebo producer도 설치·빌드")
    install.add_argument("--tests", action="store_true", help="데이터 CPU 시험 의존성도 설치")
    check = sub.add_parser("doctor", help="일회성 사전검사; 비행/학습 없음")
    check.add_argument("--processing-only", action="store_true")
    check.add_argument("--json", action="store_true")
    pack = sub.add_parser("export", help="호스트 상태를 제외한 독립 소스 압축본 작성")
    pack.add_argument("--output", required=True, help="체크아웃 밖의 새 .tar.gz 경로")
    args = parser.parse_args()
    try:
        return {"setup": setup, "doctor": doctor, "export": export}[args.action](args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print("STOP:", exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
