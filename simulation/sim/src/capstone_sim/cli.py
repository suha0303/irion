import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .acceptance import acceptance
from .campaign import campaign, plan
from .control import control
from .operations import bench, bootstrap, fetch, native_build, profile_operation
from .recording import dump, sha, verify
from .recovery import recover
from .runner import PX4, ROOT, SCENARIOS, run_flight


def command(args, log):
    env = os.environ.copy()
    env["PATH"] = str(ROOT / ".venv/bin") + ":" + env["PATH"]
    env["CCACHE_DIR"] = str(ROOT / ".cache/ccache")
    with (ROOT / log).open("w") as f:
        p = subprocess.run(args, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    return {"command": args, "exit_code": p.returncode, "log": str(ROOT / log)}


def doctor(path, *, record_source_lock=False):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Doctor output cannot use symlinks")
    path = path.resolve()
    # Diagnostic output must not overwrite inputs or the runtime seal.
    protected = {
        ROOT / name
        for name in (
            "config/producer-sources.json",
            "artifacts/runtime-manifest.json",
            "artifacts/source_lock.json",
        )
    }
    for name in ("config/producer-sources.json", "artifacts/runtime-manifest.json"):
        manifest = ROOT / name
        if manifest.is_file():
            protected.update(
                ROOT / key for key in json.loads(manifest.read_text()).get("files", {})
            )
    if path in protected or path.is_symlink():
        raise ValueError("Doctor output cannot overwrite a source input or runtime seal")
    spec_root = ROOT.parent / "docs/specifications/CODEX_REBUILD_PACKAGE_V4_2"
    spec_names = [
        "00_SCOPE_AND_REVIEW.md",
        "00_SKILL_LINKS.md",
        "00_SOURCE_NOTES.md",
        "00_SHARED_DATA_CONTRACT.md",
        "01_SIMULATION_CODEX_SPEC.md",
    ]
    spec_hashes = {}
    for name in spec_names:
        source = spec_root / name
        if not source.is_file():
            raise ValueError(f"required specification missing: {source}")
        spec_hashes[name] = sha(source)
    commands = [
        ["uname", "-a"],
        ["gz", "sim", "--versions"],
        ["cmake", "--version"],
        ["gcc", "--version"],
        ["free", "-h"],
        ["df", "-h", str(ROOT)],
    ]
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "root": str(ROOT),
        "tools": [],
    }
    for c in commands:
        p = subprocess.run(c, capture_output=True, text=True, check=False)
        result["tools"].append(
            {
                "command": c,
                "exit_code": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
        )
    dump(path, result)
    if not record_source_lock:
        return {"status": "recorded", "report": str(path), "source_lock_modified": False}
    lock = {
        "px4_release": "v1.16.0",
        "px4_commit": subprocess.check_output(
            ["git", "-C", str(PX4), "rev-parse", "HEAD"], text=True
        ).strip(),
        "submodules": subprocess.check_output(
            ["git", "-C", str(PX4), "submodule", "status"], text=True
        ),
        "python_packages": subprocess.check_output(
            [sys.executable, "-m", "pip", "--no-cache-dir", "freeze"], text=True
        ),
        "gazebo": subprocess.check_output(
            ["pkg-config", "--modversion", "gz-sim8"], text=True
        ).strip(),
        "spec_sha256": spec_hashes,
        "spec_directory": str(spec_root),
        "local_sources": {
            str(p.relative_to(ROOT)): sha(p)
            for base in ["sim", "profiles", "contracts"]
            for p in (ROOT / base).rglob("*")
            if p.is_file() and "__pycache__" not in str(p)
        },
        "licenses": {"PX4": "BSD-3-Clause", "Gazebo": "Apache-2.0", "CMU": "CC-BY-4.0"},
    }
    lock_path = ROOT / "artifacts/source_lock.json"
    if lock_path.exists():
        archive = ROOT / "artifacts/source-lock-history"
        archive.mkdir(exist_ok=True)
        previous = archive / (sha(lock_path) + ".json")
        if not previous.exists():
            previous.write_bytes(lock_path.read_bytes())
    dump(lock_path, lock)
    return {"status": "recorded", "report": str(path)}


def main():
    p = argparse.ArgumentParser(
        description="X500 and explicit multirotor profiles. Exit 0 success, 2 input, 3 environment, 4 run/verification failure."
    )
    sub = p.add_subparsers(dest="action", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--report", default="artifacts/environment.json")
    build = sub.add_parser(
        "build", help="Build under the native installer's shared locks; no flight"
    )
    build.add_argument("--jobs", type=int, default=4)
    c = sub.add_parser("campaign")
    c.add_argument("operation", choices=["plan", "run"])
    c.add_argument("--config", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--profile", choices=["development"], default="development")
    pr = sub.add_parser("profile")
    pr.add_argument("operation", choices=["validate", "inspect", "prepare"])
    pr.add_argument("path", nargs="?", help="Omit to use the built-in X500 preset")
    pr.add_argument("--jobs", type=int, default=4)
    pr.add_argument(
        "--retry",
        action="store_true",
        help="Preserve a failed preparation and explicitly try again",
    )
    dt = sub.add_parser("data")
    dt.add_argument("operation", choices=["fetch"])
    dt.add_argument("--dataset", choices=["cmu-evtol"], required=True)
    dt.add_argument("--out", required=True)
    fit = sub.add_parser("fit-battery")
    fit.add_argument("--config", default="profiles/fitting/cmu.yaml")
    be = sub.add_parser("bench")
    be.add_argument("--config", default="profiles/bench/reference.yaml")
    ex = sub.add_parser("export")
    ex.add_argument("--run", required=True)
    ex.add_argument("--format", choices=["drone-raw-2", "drone-raw-3"], required=True)
    r = sub.add_parser("run")
    selection = r.add_mutually_exclusive_group()
    selection.add_argument(
        "--vehicle",
        choices=["x500", "x500_reference"],
        default="x500",
        help="Built-in X500 (default); x500_reference is its contract ID",
    )
    selection.add_argument(
        "--vehicle-profile", help="Prepared custom multirotor vehicle YAML; simulator-only contract"
    )
    r.add_argument("--configuration", choices=["collection"], default=None)
    r.add_argument("--scenario", choices=list(SCENARIOS), default="hover_normal")
    r.add_argument("--seed", type=int, default=17)
    r.add_argument("--normal-variation-json", help="Frozen healthy conditions and smooth path JSON")
    r.add_argument("--operation-variation", help="Frozen health-independent conditions/path JSON for collection")
    r.add_argument("--fault-after-mission-s", type=float, help="Dynamic onset seconds after airborne path zero")
    r.add_argument("--real-time-factor", type=int, choices=(2, 3, 4, 8), default=2)
    r.add_argument("--duration", type=float, default=90)
    r.add_argument("--instance", type=int, default=0)
    r.add_argument("--timing", action="store_true", help="Separate native/producer timing sidecars")
    r.add_argument("--capture-interval-ms", type=float, default=5.0)
    r.add_argument(
        "--fault-at-s",
        type=float,
        help="Physical step/ramp onset in PX4 boot seconds; omitted preserves scenario default",
    )
    preview = sub.add_parser("preview", help="Plan or explicitly fly one random GUI review; no collection")
    preview.add_argument("operation", choices=["plan", "fly"])
    preview.add_argument("--seed", type=int, default=20261030)
    preview.add_argument("--duration", type=int, choices=[180, 900], default=900)
    preview.add_argument("--plan", default="artifacts/random-continuous-v3/preview.json")
    v = sub.add_parser("verify")
    v.add_argument("--suite", choices=["unit", "integration", "end-to-end"], default="unit")
    v.add_argument("--run")
    re = sub.add_parser("recover")
    re.add_argument("--run", required=True)
    re.add_argument("--out", required=True)
    ctl = sub.add_parser("control")
    ctl.add_argument("--run", required=True)
    ctl.add_argument(
        "--operation",
        choices=["status", "set_scenario", "apply_parameter_change", "reset"],
        required=True,
    )
    ctl.add_argument("--request-id", type=int, default=1)
    ctl.add_argument("--target", default="battery_r0")
    ctl.add_argument("--value", type=float, default=0.0)
    ctl.add_argument("--requested-us", type=int, default=0)
    ctl.add_argument("--scenario", choices=list(SCENARIOS))
    a = p.parse_args()
    code = 0
    try:
        if a.action == "doctor":
            result = doctor(ROOT / a.report)
        elif a.action == "build":
            result = native_build(a.jobs)
            code = 3 if result["exit_code"] else 0
        elif a.action == "bootstrap":
            result = bootstrap()
            code = 3 if result["status"] == "needs_environment" else 0
        elif a.action == "profile":
            result = profile_operation(a.operation, a.path, jobs=a.jobs, retry=a.retry)
        elif a.action == "data":
            result = fetch(a.out)
        elif a.action == "fit-battery":
            if a.config != "profiles/fitting/cmu.yaml":
                raise ValueError("only the supplied fixed CMU protocol is implemented")
            result = command(
                [sys.executable, "sim/tools/fit_battery.py"],
                "artifacts/battery-fit.stdout.log",
            )
            code = 4 if result["exit_code"] else 0
        elif a.action == "bench":
            result = bench(a.config)
            code = 4 if result["exit_code"] else 0
        elif a.action == "export":
            metadata = json.loads((Path(a.run) / "observations/metadata.json").read_text())
            expected = (
                "drone.observation.metadata/3.0"
                if a.format == "drone-raw-3"
                else "drone.observation.metadata/2.0"
            )
            if metadata.get("schema") != expected:
                raise ValueError("Export format must match the observation metadata contract")
            result = verify(a.run)
            result["path"] = str((Path(a.run) / "observations").resolve())
            code = 4 if result["status"] == "failed" else 0
        elif a.action == "campaign":
            result = plan(a.config) if a.operation == "plan" else campaign(a.config)
            code = 4 if result.get("status") == "partial" else 0
        elif a.action == "recover":
            result = recover(a.run, a.out)
        elif a.action == "control":
            result = control(
                a.run, a.operation, a.request_id, a.target, a.value, a.requested_us, a.scenario
            )
            code = 4 if result.get("exit_code", 0) else 0
        elif a.action == "preview":
            from .preview import fly, prepare

            result = (
                prepare(ROOT / a.plan, a.seed, a.duration)
                if a.operation == "plan"
                else fly(ROOT / a.plan)
            )
            code = 4 if result.get("status") == "failed" else 0
        elif a.action == "run":
            result = run_flight(
                a.scenario,
                a.seed,
                a.duration,
                a.instance,
                timing=a.timing,
                capture_interval_ms=a.capture_interval_ms,
                fault_at_s=a.fault_at_s,
                configuration=a.configuration,
                operation_variation=json.loads(a.operation_variation) if a.operation_variation else None,
                fault_after_mission_s=a.fault_after_mission_s,
                normal_variation=json.loads(a.normal_variation_json)
                if a.normal_variation_json
                else None,
                real_time_factor=a.real_time_factor,
                vehicle_profile=a.vehicle_profile,
            )
            code = 0 if result["status"] == "completed" else 4
        elif a.action == "verify":
            if a.run:
                result = verify(a.run)
                code = 0 if result["status"] == "passed" else 4
            elif a.suite == "unit":
                result = []
                for args, log in [
                    (
                        ["ctest", "--test-dir", "build/reference", "--output-on-failure"],
                        "artifacts/unit.log",
                    ),
                    (
                        [sys.executable, "-m", "unittest", "discover", "-s", "sim/tests", "-v"],
                        "artifacts/python-tests.log",
                    ),
                    (["ruff", "check", "sim"], "artifacts/lint.log"),
                    (["mypy", "sim/src"], "artifacts/types.log"),
                ]:
                    result.append(command(args, log))
                code = 4 if any(r["exit_code"] for r in result) else 0
            else:
                result = acceptance(a.suite)
                code = 0 if result["status"] == "passed" else 4
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, yaml.YAMLError) as e:
        result = {"status": "failed", "error": str(e)}
        code = 2 if isinstance(e, (ValueError, yaml.YAMLError)) else 3
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
