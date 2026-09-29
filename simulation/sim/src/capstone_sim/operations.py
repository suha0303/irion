import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

from .recording import dump, sha
from .runner import PX4, ROOT


def fetch(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    url = "https://api.figshare.com/v2/articles/14226830/versions/3"
    with urllib.request.urlopen(url, timeout=60) as r:
        meta = json.load(r)
    dump(out / "metadata.json", meta)
    result = []
    for name in ["README.txt", "VAH07.csv"]:
        entry = next(f for f in meta["files"] if f["name"] == name)
        p = out / name
        if (
            not p.exists()
            or p.stat().st_size != entry["size"]
            or hashlib.md5(p.read_bytes()).hexdigest() != entry["computed_md5"]
        ):
            tmp = p.with_suffix(p.suffix + ".download")
            with (
                urllib.request.urlopen(entry["download_url"], timeout=300) as r,
                tmp.open("wb") as w,
            ):
                while block := r.read(1024 * 1024):
                    w.write(block)
            if (
                tmp.stat().st_size != entry["size"]
                or hashlib.md5(tmp.read_bytes()).hexdigest() != entry["computed_md5"]
            ):
                raise ValueError("public file integrity mismatch")
            tmp.replace(p)
        result.append(
            {
                "path": str(p),
                "sha256": sha(p),
                "metadata": entry,
                "license": meta["license"],
            }
        )
    dump(out / "provenance.json", result)
    return result


def profile_validate(path):
    from .profile import load_profile

    profile = load_profile(path)
    result = {"status": "passed", "profile": str(path), "sha256": sha(path)}
    if profile["id"] != "x500_reference":
        from .vehicle_runtime import inspection

        result["simulation_profile"] = inspection(path)
    return result


def profile_operation(operation, path=None, *, jobs=4, retry=False):
    """No selection means the original X500 preset, never an example vehicle."""
    from . import collection_config, vehicle_runtime
    from .profile import PROFILE, assert_generated, load_profile

    path = Path(path) if path is not None else PROFILE
    try:
        profile = load_profile(path)
    except (KeyError, yaml.YAMLError) as error:
        raise ValueError("Invalid vehicle profile: " + str(error)) from error
    if operation == "validate":
        return profile_validate(path)
    if profile["id"] != "x500_reference":
        return (
            vehicle_runtime.inspection(path)
            if operation == "inspect"
            else vehicle_runtime.prepare(path, jobs=jobs, retry=retry)
        )
    current = path.absolute() == (collection_config.DIRECTORY / "vehicle.yaml").absolute()
    if not current and path.absolute() != PROFILE.absolute():
        raise ValueError("The reserved X500 ID requires a built-in X500 preset")
    result = dict(
        status="PROFILE_VALIDATED_NO_FLIGHT",
        vehicle="x500",
        profile_id="x500_reference",
        profile=str(path.absolute()),
        configuration="collection" if current else "reference",
        rotor_count=4,
        schema="drone.observation.metadata/2.0",
        actual_flight_started=False,
        training_started=False,
    )
    if operation == "prepare":
        if type(jobs) is not int or jobs < 1:
            raise ValueError("Positive build job count required")
        from .runtime_manifest import verify_runtime

        runtime = verify_runtime(ROOT)
        assert_generated(path, collection_config.HEADER if current else None)
        plugin = collection_config.PLUGIN if current else ROOT / "build/libDronePowertrain.so"
        if runtime["files"].get(str(plugin.relative_to(ROOT))) != sha(plugin):
            raise ValueError("X500 preset not prepared; run drone-data setup --simulation")
        result.update(
            status="PREPARED_NO_FLIGHT",
            plugin=str(plugin),
            note="Built-in X500 runtime checked; no flight or new compilation",
        )
    return result


def native_build(jobs=4):
    """Use the installer's shared locks instead of mutating native builds directly."""
    if type(jobs) is not int or jobs < 1:
        raise ValueError("Positive build job count required")
    log = ROOT / "artifacts/build-command.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "sim/tools/install.py"), "--jobs", str(jobs)],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return dict(
        status="PREPARED_NO_FLIGHT" if result.returncode == 0 else "FAILED_STOPPED",
        exit_code=result.returncode,
        log=str(log),
        flight_started=False,
    )


def bootstrap():
    # Never invokes apt, sudo, usermod or global configuration.
    commands = [
        ["git", "--version"],
        ["cmake", "--version"],
        ["pkg-config", "--modversion", "gz-sim8"],
        ["pkg-config", "--modversion", "gz-transport13"],
    ]
    checks = []
    for c in commands:
        p = subprocess.run(c, capture_output=True, text=True, check=False)
        checks.append(
            {
                "command": c,
                "exit_code": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
        )
    if any(c["exit_code"] for c in checks):
        return {"status": "needs_environment", "checks": checks}
    return {
        "status": "ready",
        "checks": checks,
        "python": sys.executable,
        "px4_present": (PX4 / ".git").exists(),
        "note": "Existing project-local environment checked. Fresh installation: sim/tools/bootstrap.sh; no system installation.",
    }


def bench(config):
    c = yaml.safe_load(Path(config).read_text())
    out = ROOT / "artifacts" / c.get("output", "bench.csv")
    with out.open("w") as f:
        p = subprocess.run(
            [
                str(ROOT / "build/powertrain_bench"),
                str(c.get("duration_s", 10)),
                str(c.get("soc", 0.9)),
                str(c.get("duty", 0.55)),
                str(c.get("capacity_ah", 6)),
            ],
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    return {
        "exit_code": p.returncode,
        "output": str(out),
        "stderr": p.stderr,
        "origin": "physics_bench",
        "profile": "analytic_prior",
        "closed_loop": False,
    }
