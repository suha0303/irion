"""Prepare a pinned, project-local PX4/Gazebo runtime; never run a flight."""

import argparse
import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim/src"))

from capstone_sim.runtime_manifest import (  # noqa: E402
    PLUGIN,
    PROFILE,
    PX4_BINARY,
    RUNTIME_MANIFEST,
    SOURCE_MANIFEST,
    TRANSPORT,
    checked_files,
    sha,
    verify_runtime,
    verify_sources,
)

APT_PACKAGES = (
    "build-essential ccache cmake ninja-build git patch curl python3.12-dev "
    "python3.12-venv pkg-config libssl-dev libxml2-dev libxml2-utils zip unzip rsync "
    "gz-harmonic libgz-sim8-dev libgz-transport13-dev libgz-cmake3-dev libgz-msgs10-dev "
    "libgz-utils2-dev libeigen3-dev libzmq3-dev cppzmq-dev libsqlite3-dev uuid-dev "
    "libprotobuf-dev protobuf-compiler"
)
COMMANDS = (
    "git",
    "cmake",
    "ninja",
    "gcc",
    "g++",
    "make",
    "pkg-config",
    "patch",
    "xz",
    "gz",
    "ldd",
    "protoc",
    "ccache",
    "zip",
    "unzip",
    "rsync",
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def environment():
    env = dict(os.environ)
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "sim/src")
    env["CCACHE_DIR"] = str(ROOT / ".cache/ccache")
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return env


def prerequisites():
    info = platform.freedesktop_os_release()
    errors = []
    if (
        sys.version_info[:2] != (3, 12)
        or sys.platform != "linux"
        or platform.machine() != "x86_64"
        or info.get("ID") != "ubuntu"
        or info.get("VERSION_ID") != "24.04"
    ):
        errors.append("Supported native target: Ubuntu 24.04 x86_64 / Python 3.12")
    missing = [name for name in COMMANDS if shutil.which(name) is None]
    if missing:
        errors.append("Missing commands: " + ", ".join(missing))
    versions = {}
    if shutil.which("pkg-config"):
        for package, prefix in (("gz-sim8", "8."), ("gz-transport13", "13.")):
            p = subprocess.run(
                ["pkg-config", "--modversion", package], capture_output=True, text=True, check=False
            )
            versions[package] = p.stdout.strip()
            if p.returncode or not p.stdout.startswith(prefix):
                errors.append("Missing Gazebo Harmonic development dependency: " + package)
    return dict(
        passed=not errors,
        errors=errors,
        gazebo=versions,
        system_install_automatic=False,
        system_install_command="sudo apt-get install --no-install-recommends " + APT_PACKAGES,
        gazebo_repository="https://gazebosim.org/docs/harmonic/install_ubuntu/",
    )


class Setup:
    def __init__(self, jobs):
        self.jobs = jobs
        self.started = time.monotonic()
        self.rows = []
        self.status_path = ROOT / "artifacts/setup-status.json"
        self.log_path = ROOT / "artifacts/setup.log"
        self.environment = environment()
        self.first_record = True

    def record(self, status, step, **extra):
        if self.first_record and self.status_path.exists():
            previous = ROOT / "artifacts/setup-history" / (sha(self.status_path) + ".json")
            previous.parent.mkdir(parents=True, exist_ok=True)
            if not previous.exists():
                shutil.copy2(self.status_path, previous)
        self.first_record = False
        write(
            self.status_path,
            dict(
                schema="x500.native-setup/1",
                status=status,
                step=step,
                elapsed_s=time.monotonic() - self.started,
                rows=self.rows,
                flight_started=False,
                training_started=False,
                **extra,
            ),
        )
        print(f"{status}: {step}", flush=True)

    def run(self, step, args, env=None):
        self.record("RUNNING", step)
        started = time.monotonic()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as log:
            log.write("\n" + json.dumps(dict(step=step, argv=list(map(str, args)))) + "\n")
            log.flush()
            result = subprocess.run(
                list(map(str, args)),
                cwd=ROOT,
                env=env or self.environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        self.rows.append(
            dict(step=step, exit_code=result.returncode, elapsed_s=time.monotonic() - started)
        )
        if result.returncode:
            raise RuntimeError(f"{step} failed ({result.returncode}); see {self.log_path}")

    def python(self):
        python = ROOT / ".venv/bin/python"
        if not python.is_file():
            self.run("native Python environment", [sys.executable, "-m", "venv", ROOT / ".venv"])
        version = subprocess.check_output(
            [str(python), "-c", "import sys;print(sys.version_info[:2])"], text=True
        ).strip()
        if version != "(3, 12)":
            raise ValueError("Existing simulation/.venv is not Python 3.12; preserved")
        self.run(
            "native Python dependencies",
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--cache-dir",
                ROOT / ".cache/pip",
                "-r",
                ROOT / "requirements.lock",
            ],
        )
        return python

    def px4(self, source, python):
        path = ROOT / "vendor/PX4-Autopilot"
        upstream = source["upstream"]["px4"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.run(
                "PX4 source checkout",
                [
                    "git",
                    "clone",
                    "--branch",
                    upstream["release"],
                    "--depth",
                    "1",
                    upstream["url"],
                    path,
                ],
            )
        if (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            != upstream["commit"]
        ):
            raise ValueError("Existing PX4 revision differs from pinned commit; preserved")
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        )
        allowed = {name.removeprefix("vendor/PX4-Autopilot/") for name in source["vendor_files"]}
        for line in status.splitlines():
            if line[3:] not in allowed:
                raise ValueError("Unexpected local PX4 changes preserved: " + line)
        self.run(
            "PX4 pinned submodules",
            [
                "git",
                "-C",
                path,
                "submodule",
                "update",
                "--init",
                "--recursive",
                "--depth",
                "1",
                *upstream["submodules"],
            ],
        )
        for name, commit in upstream["submodule_commits"].items():
            actual = subprocess.check_output(
                ["git", "-C", str(path / name), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != commit:
                raise ValueError("PX4 submodule revision mismatch: " + name)
        self.run("PX4 X500 coupling patch", [python, "sim/tools/patch_px4.py"])
        checked_files(ROOT, source["vendor_files"])
        self.run("PX4 build", ["make", "-C", path, "px4_sitl_default", f"-j{self.jobs}"])

    def plugins(self, python):
        self.run(
            "validate reference header",
            [python, "-c", "from capstone_sim.profile import assert_generated;assert_generated()"],
        )
        self.run(
            "validate current header",
            [
                python,
                "-c",
                "from capstone_sim.profile import assert_generated;from capstone_sim import collection_config;"
                "assert_generated(collection_config.DIRECTORY/'vehicle.yaml',collection_config.HEADER)",
            ],
        )
        for name, extra in (
            ("reference", []),
            (
                "current",
                [
                    "-DDRONE_PROFILE_HEADER="
                    + str(ROOT / "profiles/collection/vehicle_generated.hpp")
                ],
            ),
        ):
            directory = ROOT / "build" / name
            self.run(
                f"{name} physics configure",
                [
                    "cmake",
                    "-S",
                    "sim",
                    "-B",
                    directory,
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    *extra,
                ],
            )
            self.run(
                f"{name} physics build", ["cmake", "--build", directory, "--parallel", self.jobs]
            )
            self.run(
                f"{name} physics tests", ["ctest", "--test-dir", directory, "--output-on-failure"]
            )
        self.run(
            "current numerical checks",
            [
                ROOT / "build/current/collection_numeric",
                ROOT / "profiles/collection/battery_runtime.txt",
                ROOT / "profiles/propulsion_runtime.txt",
            ],
        )
        for name in (
            "libDronePowertrain.so",
            "powertrain_test",
            "boundaries_test",
            "powertrain_bench",
            "rc_bench",
            "capture_timing_fixture",
        ):
            shutil.copy2(ROOT / "build/reference" / name, ROOT / "build" / name)

    def transport(self, source):
        upstream = source["upstream"]["transport"]
        archive = ROOT / ".cache/downloads/gazebo-transport.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            self.record("RUNNING", "download pinned private transport source")
            temporary = archive.with_suffix(".download")
            with (
                urllib.request.urlopen(upstream["url"], timeout=300) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
            if sha(temporary) != upstream["sha256"]:
                raise ValueError(
                    "Private transport source download hash mismatch; evidence preserved"
                )
            temporary.replace(archive)
        if sha(archive) != upstream["sha256"]:
            raise ValueError("Cached private transport source hash mismatch; preserved")
        directory = ROOT / ".cache/sources/gazebo-transport"
        if not directory.exists():
            directory.parent.mkdir(parents=True, exist_ok=True)
            staging = directory.with_name("gazebo-transport.extracting")
            staging.mkdir(exist_ok=False)
            with tarfile.open(archive) as tar:
                tar.extractall(staging, filter="data")
            children = list(staging.iterdir())
            if len(children) != 1 or not children[0].is_dir():
                raise ValueError("Unexpected private transport source archive layout")
            children[0].rename(directory)
            staging.rmdir()
            self.run(
                "private transport startup fix",
                ["patch", "-d", directory, "-p1", "--fuzz=0", "--input", ROOT / upstream["patch"]],
            )
        header = directory / "include/gz/transport/Discovery.hh"
        if sha(header) != upstream["patched_header_sha256"]:
            raise ValueError("Private transport patched source differs; preserved")
        build = ROOT / "build/transport"
        prefix = ROOT / "runtime/gazebo-transport"
        self.run(
            "private transport configure",
            [
                "cmake",
                "-S",
                directory,
                "-B",
                build,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_TESTING=OFF",
                "-DSKIP_PYBIND11=ON",
                "-DCMAKE_CXX_FLAGS=-fno-gnu-unique -fstack-protector-strong",
                "-DCMAKE_INSTALL_PREFIX=" + str(prefix),
            ],
        )
        self.run("private transport build", ["cmake", "--build", build, "--parallel", self.jobs])
        self.run("private transport project-local install", ["cmake", "--install", build])

    def adopt(self, source):
        """Migrate only the exact reviewed local binaries, never seal arbitrary files."""
        for target, item in source["reviewed_local_runtime"].items():
            path = ROOT / target
            if not path.exists():
                original = ROOT / item["source"]
                if not original.is_file() or sha(original) != item["sha256"]:
                    raise ValueError(
                        "Reviewed existing binary missing or changed: " + str(original)
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, path)
            if sha(path) != item["sha256"]:
                raise ValueError("Existing stable runtime differs from reviewed binary; preserved")
        alias = (ROOT / TRANSPORT).parent / "libgz-transport13.so.13"
        if not alias.exists():
            alias.symlink_to(Path(TRANSPORT).name)
        checked_files(ROOT, source["vendor_files"])

    def seal(self, source):
        verify_sources(ROOT)
        checked_files(ROOT, source["vendor_files"])
        upstream = source["upstream"]["px4"]
        checkout = ROOT / "vendor/PX4-Autopilot"
        for relative, commit in {".": upstream["commit"], **upstream["submodule_commits"]}.items():
            actual = subprocess.check_output(
                ["git", "-C", str(checkout / relative), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != commit:
                raise ValueError("Native upstream revision differs from pinned source: " + relative)
        runtime_files = dict(source["files"] | source["vendor_files"])
        runtime_files[SOURCE_MANIFEST] = sha(ROOT / SOURCE_MANIFEST)
        required = [
            PX4_BINARY,
            PLUGIN,
            "build/libDronePowertrain.so",
            TRANSPORT,
            "runtime/gazebo-transport/lib/libgz-transport13-log.so.13.6.0",
            "runtime/gazebo-transport/lib/libgz-transport13-parameters.so.13.6.0",
        ]
        etc = ROOT / "vendor/PX4-Autopilot/build/px4_sitl_default/etc"
        if not (etc / "init.d-posix/airframes/4001_gz_x500").is_file():
            raise ValueError("PX4 private startup tree is incomplete")
        required += [str(path.relative_to(ROOT)) for path in etc.rglob("*") if path.is_file()]
        for relative in required:
            runtime_files[relative] = sha(ROOT / relative)
        private_env = dict(self.environment)
        private_env["LD_LIBRARY_PATH"] = str((ROOT / TRANSPORT).parent)
        library_directory = Path(
            subprocess.check_output(
                ["pkg-config", "--variable=libdir", "gz-sim8"], text=True
            ).strip()
        )
        gazebo_consumers = {path.resolve() for path in library_directory.glob("libgz-sim8-gz.so*")}
        if not gazebo_consumers:
            raise ValueError("Gazebo native server CLI library missing")
        for target in (
            PX4_BINARY,
            PLUGIN,
            "build/libDronePowertrain.so",
            TRANSPORT,
            *sorted(gazebo_consumers),
        ):
            result = subprocess.run(
                ["ldd", "-r", str(ROOT / target)],
                env=private_env,
                capture_output=True,
                text=True,
                check=False,
            )
            if (
                result.returncode
                or "not found" in result.stdout
                or "undefined symbol:" in result.stdout + result.stderr
            ):
                raise ValueError(
                    "Native relocation check failed: "
                    + str(target)
                    + "\n"
                    + result.stdout
                    + result.stderr
                )
        self.run(
            "record native environment and source revision",
            [
                ROOT / ".venv/bin/python",
                "-B",
                "-c",
                "from pathlib import Path;from capstone_sim.cli import doctor;"
                "doctor(Path('artifacts/environment.json'),record_source_lock=True)",
            ],
        )
        runtime_files["artifacts/source_lock.json"] = sha(ROOT / "artifacts/source_lock.json")
        payload = dict(
            schema="x500.producer-runtime/1",
            revision="X500_CURRENT",
            configuration="collection",
            profile_id="vtc6_4s4p",
            source_manifest=SOURCE_MANIFEST,
            source_manifest_sha256=sha(ROOT / SOURCE_MANIFEST),
            files=runtime_files,
            px4_binary_sha256=runtime_files[PX4_BINARY],
            physics_plugin_sha256=runtime_files[PLUGIN],
            x500_profile_sha256=runtime_files[PROFILE],
            status="PREPARED_OFFLINE_NO_NEW_FLIGHT",
            platform=platform.platform(),
            flight_started=False,
            source_revision_sha256=sha(ROOT / "artifacts/source_lock.json"),
        )
        if "fc_parameters" in source:
            payload["fc_parameters"] = dict(source["fc_parameters"])
        path = ROOT / RUNTIME_MANIFEST
        if path.exists():
            previous = ROOT / "artifacts/runtime-history" / (sha(path) + ".json")
            if not previous.exists():
                previous.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, previous)
        write(path, payload)
        verify_runtime(ROOT)
        self.record(
            "PREPARED_STOPPED",
            "native runtime prepared; no flight started",
            runtime_manifest=str(path),
            log=str(self.log_path),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="read-only prerequisite and source verification"
    )
    parser.add_argument("--jobs", type=int, default=4, help="native build jobs (positive integer)")
    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help="adopt only the exactly pinned reviewed workstation binaries",
    )
    args = parser.parse_args(argv)
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    setup = Setup(args.jobs)
    locks = []
    try:
        source = verify_sources(ROOT)
        checks = prerequisites()
        if args.check:
            print(json.dumps(checks, indent=2))
            return 0 if checks["passed"] else 3
        if not checks["passed"]:
            raise ValueError(json.dumps(checks, indent=2))
        paths = [
            ROOT / ".cache/setup.lock",
            ROOT.parent / "Data/operations/collection.lock",
        ]
        paths += [ROOT / ".cache/instances" / f"{i}.lock" for i in range(16)]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a")
            locks.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.adopt_existing:
            setup.adopt(source)
        else:
            python = setup.python()
            setup.px4(source, python)
            setup.plugins(python)
            setup.transport(source)
        setup.seal(source)
        return 0
    except BlockingIOError as error:
        print(
            "Native setup/collection already owns a required lock: " + str(error), file=sys.stderr
        )
        return 3
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        if not args.check:
            setup.record("FAILED_STOPPED", "setup failed; no automatic retry", reason=str(error))
        print(str(error), file=sys.stderr)
        return 3
    finally:
        for stream in reversed(locks):
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
