"""Explicit, once-only R3 operators. No scheduler, retry, watcher or next stage."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/home/sa/CapstoneV3/ai/experiments/ROUND_001/long-runs-02")
PACKAGE = BASE / "ai/release/x500-round001-r3-long15"
PACKAGE_SHA = None
AUDITOR = PACKAGE / "wrapper/preprocessing_tools/audit.py"
AUDITOR_SHA = None
PROTECTED_SHA = None
STORAGE = Path("/media/sa/X31/Data/X500/ROUND_001/R3_LONG15")
PREPARED = STORAGE / "processed/attempt-01"
MODELS = STORAGE / "models/attempt-01"
CALIBRATION = STORAGE / "calibration/attempt-01"
META = STORAGE / "metadata"
REMOTE_PACKAGE = None
REMOTE = None
REMOTE_PYTHON = "/abr/coss43/miniconda3/envs/mamba/bin/python"
HOST = "coss43@155.230.81.91"
DATASET_FILES = {
    "dataset/normal_train.npz",
    "dataset/normal_validation.npz",
    "dataset/scaler.npz",
    "dataset/manifest.json",
    "dataset/canonical-seal.json",
    "ridge-memory.json",
}
START = "User explicitly invoked operator_stages.py for this single R3_LONG15 stage; fixed one-round approval, no retry or automatic next stage."


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("traversal/symlink forbidden: " + str(path))
    return path


def read(path):
    return json.loads(safe(path).read_text())


def write(path, data):
    path = safe(path)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def link(path):
    return {"path": str(safe(path)), "sha256": sha(safe(path))}


def environment(package=None):
    package = PACKAGE if package is None else package
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(package / p) for p in ("wrapper", "ai", "ai/src"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def control():
    # Load only verified, frozen control code; never pipeline/model/Torch here.
    if sha(PACKAGE / "PACKAGE.json") != PACKAGE_SHA:
        raise ValueError("contract-fixed package digest mismatch")
    for name, digest in read(PACKAGE / "PACKAGE.json")["files"].items():
        if Path(name).is_absolute() or sha(safe(PACKAGE / name)) != digest:
            raise ValueError("package source/inventory mismatch before import: " + name)
    for part in reversed(("wrapper", "ai", "ai/src")):
        sys.path.insert(0, str(PACKAGE / part))
    from preprocessing_tools import control as module

    module.package_check(PACKAGE)
    return module


def completion(folder, stage, freeze_sha, ctl):
    folder = safe(folder)
    ref = {"path": str(folder), "seal_sha256": sha(folder / "artifact.json")}
    _, artifact = ctl.artifact(ref, stage, freeze_sha)
    release, ended = read(folder / "release.json"), read(folder / "exit.json")
    if (
        artifact.get("package_sha256") != PACKAGE_SHA
        or ended.get("exit_code") != 0
        or release.get("passed") is not True
        or release.get("released") is not True
        or release.get("artifact_sha256") != ref["seal_sha256"]
    ):
        raise ValueError("successful exit/release and same-package artifact required")
    ref.update(exit_sha256=sha(folder / "exit.json"), release_sha256=sha(folder / "release.json"))
    if stage == "main":
        ctl.returned_model(ref, freeze_sha, PACKAGE_SHA)
    return ref, artifact


def approvals(folder, stage, out, frozen, protected, ctl, inputs=None, audit=None):
    common = dict(
        round_id="ROUND_001",
        revision="R3_LONG15",
        stage=stage,
        execution_authorized=True,
        output=str(out),
        package_sha256=PACKAGE_SHA,
        plan_sha256=read(PACKAGE / "PACKAGE.json")["plan_sha256"],
        profile=ctl.PROFILE,
    )
    lock = dict(
        common,
        status="DATA_FROZEN",
        frozen_manifest=frozen,
        protected_inventory=protected,
        inputs=inputs or {},
    )
    if audit is not None:
        lock["derived_audit"] = audit
    approval = dict(common, exact_message=START, allowed_stages=[stage], lock_content=lock)
    ap, lp = folder / "approval.json", folder / "lock.json"
    write(ap, approval)
    write(lp, dict(lock, approval_sha256=sha(ap)))
    return ap, lp


class CommandFailure(RuntimeError):
    def __init__(self, code, label):
        self.code = code if code > 0 else 128 - code
        super().__init__(f"{label} exited {code}; preserve evidence and stop")


class Runner:
    def __init__(self, folder):
        self.folder, self.index = folder, 0

    def run(self, argv, *, env=None, check=True, stream=False):
        self.index += 1
        stem = self.folder / f"command-{self.index:02d}"
        write(
            stem.with_suffix(".json"),
            {"argv": argv, "started_utc": datetime.now(timezone.utc).isoformat()},
        )
        with stem.with_suffix(".log").open("x") as log:
            if stream:
                child = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
                )
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = child.wait()
            else:
                code = subprocess.run(
                    argv, stdout=log, stderr=subprocess.STDOUT, env=env
                ).returncode
        write(stem.with_suffix(".exit.json"), {"exit_code": code})
        if check and code:
            raise CommandFailure(code, str(stem))
        return code


def launch(stage, out, ap, lp, runner):
    runner.run(
        [
            sys.executable,
            "-B",
            "-m",
            "preprocessing_tools.launch",
            "--root",
            str(PACKAGE),
            "--stage",
            stage,
            "--out",
            str(out),
            "--approval",
            str(ap),
            "--lock",
            str(lp),
        ],
        env=environment(),
    )


def ram_check(path):
    evidence = read(path)
    observed = datetime.fromisoformat(evidence["observed_utc"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - observed).total_seconds()
    if (
        evidence.get("kind") != "coordinator_verified_cpu_RAM_budget"
        or evidence.get("hostname") != "DIS01"
        or not evidence.get("source")
        or not -60 <= age <= 900
        or any(
            type(evidence.get(k)) is not int or evidence[k] <= 0
            for k in ("allocation_budget_bytes", "available_bytes")
        )
    ):
        raise ValueError("fresh <=900s verified DIS01 CPU RAM/budget evidence required")
    return link(path)


def memory_from_slurm(record):
    """Fail closed unless Slurm explicitly supplies the default memory per GPU."""

    def fields(text):
        return dict(re.findall(r"([A-Za-z_]+)\s*=\s*([^\s]+)", text))

    node = fields(record["node"])
    partition = fields(record["partition"])
    config = fields(record["config"])
    if node.get("NodeName") != "DIS01" or partition.get("PartitionName") != "p02":
        raise ValueError("RAM source node/partition mismatch")
    # No extrapolation from GPU VRAM or the earlier job19008 memory request.
    defaults = fields(partition.get("JobDefaults", "").replace(",", " "))
    value = defaults.get(
        "DefMemPerGPU", partition.get("DefMemPerGPU", config.get("DefMemPerGPU", ""))
    )
    if not value.isdigit() or int(value) <= 0:
        raise ValueError("Slurm DefMemPerGPU not explicit; verified RAM evidence required")
    real, allocated, free = (int(node[k]) for k in ("RealMemory", "AllocMem", "FreeMem"))
    available = min(free, real - allocated)
    if available <= 0 or allocated < 0:
        raise ValueError("no verified available DIS01 CPU memory")
    return dict(
        kind="coordinator_verified_cpu_RAM_budget",
        hostname="DIS01",
        observed_utc=record["observed_utc"],
        allocation_budget_bytes=int(value) * 1024**2,
        available_bytes=available * 1024**2,
        source="read-only Slurm node FreeMem/RealMemory/AllocMem and explicit DefMemPerGPU; one GPU; no allocation submitted",
    )


def obtain_ram(folder, runner):
    probe = """
import subprocess,json
from datetime import datetime,timezone
def call(*a): return subprocess.check_output(['scontrol',*a],text=True)
r={'node':call('show','node','-o','DIS01'),'partition':call('show','partition','-o','p02'),'config':call('show','config'),'observed_utc':datetime.now(timezone.utc).isoformat()}
print(json.dumps(r))
"""
    runner.run(ssh([REMOTE_PYTHON, "-B", "-c", probe], ssh_options()))
    raw = runner.folder / f"command-{runner.index:02d}.log"
    evidence = memory_from_slurm(read(raw))
    evidence["source"] += "; raw=" + str(raw) + "; sha256=" + sha(raw)
    path = folder / "ram-evidence.json"
    write(path, evidence)
    ram_check(path)
    return path


def audit_inputs(folder, frozen, protected, ram, runner):
    if sha(AUDITOR) != AUDITOR_SHA:
        raise ValueError("packaged auditor changed")
    ram_check(ram)
    output = folder / "derived-audit.json"
    argv = [
        sys.executable,
        "-B",
        "-m",
        "preprocessing_tools.audit",
        "--root",
        str(PACKAGE),
        "--prepared",
        str(PREPARED),
        "--freeze",
        frozen["path"],
        "--protected",
        protected["path"],
        "--ram",
        str(ram),
        "--out",
        str(output),
    ]
    runner.run(argv, env=environment())
    report = read(output)
    if report.get("status") != "DERIVED_AUDIT_PASSED" or report.get("passed") is not True:
        raise ValueError("derived/window/scaler/RAM gate failed; no training")
    return link(output)


def ssh_options():
    socket = safe(BASE.parent / "data-1gb-01/ssh-session/control")
    if not stat.S_ISSOCK(socket.stat().st_mode):
        raise ValueError("existing user-authenticated SSH control socket required")
    # If the existing multiplex connection disappears, forbid direct-auth fallback.
    return [
        "-o",
        "ControlPath=" + str(socket),
        "-o",
        "ControlMaster=no",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ProxyCommand=false",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]


def ssh(argv, options):
    return ["ssh", "-p", "10000", *options, HOST, shlex.join(argv)]


def copy_to(source, dest, runner, options):
    runner.run(["scp", "-P", "10000", *options, str(safe(source)), HOST + ":" + dest])


def remote_train(folder, frozen, protected, dataset, artifact, audit, ram, runner):
    if set(artifact["files"]) != DATASET_FILES:
        raise ValueError("prepared inventory must be exactly six approved files")
    options = ssh_options()
    # New staging is exclusive; no remote deletion, retries or package edits.
    reserve = (
        "from pathlib import Path; import os; os.umask(0o077); "
        f"p=Path({REMOTE!r}); p.mkdir(); "
        "[(p/n).mkdir(parents=True) for n in ('prepared/dataset','metadata')]"
    )
    runner.run(ssh([REMOTE_PYTHON, "-B", "-c", reserve], options))
    for name in sorted(DATASET_FILES | {"artifact.json", "exit.json", "release.json"}):
        copy_to(PREPARED / name, REMOTE + "/prepared/" + name, runner, options)
    remote_meta = {}
    for name, ref in (
        ("frozen-manifest.json", frozen),
        ("protected-inventory.json", protected),
        ("derived-audit.json", audit),
        ("ram-evidence.json", ram_check(ram)),
    ):
        destination = REMOTE + "/metadata/" + name
        copy_to(ref["path"], destination, runner, options)
        remote_meta[name] = dict(path=destination, sha256=ref["sha256"])
    ctl = control()
    ap, lp = approvals(
        folder,
        "main",
        REMOTE + "/main",
        remote_meta["frozen-manifest.json"],
        remote_meta["protected-inventory.json"],
        ctl,
        inputs={"dataset": dict(dataset, path=REMOTE + "/prepared")},
        audit=remote_meta["derived-audit.json"],
    )
    for p in (ap, lp):
        copy_to(p, REMOTE + "/metadata/" + p.name, runner, options)
    # Remote verify transferred input inventory and completion before existing launcher.
    # No scientific code runs on the login node; the launcher alone submits main.
    bridge = f"""
import sys, os, subprocess, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
root=Path({REMOTE_PACKAGE!r})
def require(ok, message):
    if not ok: raise ValueError(message)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
require(digest(root/'PACKAGE.json') == {PACKAGE_SHA!r}, 'package hash')
for name, expected in json.loads((root/'PACKAGE.json').read_text())['files'].items():
    p=root/name
    require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'inventory path')
    require(not any(q.is_symlink() for q in (p,*p.parents)) and digest(p)==expected, 'source pre-import hash')
sys.path[:0]=[str(root/p) for p in ('wrapper','ai','ai/src')]
from preprocessing_tools.control import admission, artifact, linked, sha, read
ap=Path({(REMOTE + "/metadata/approval.json")!r})
lp=Path({(REMOTE + "/metadata/lock.json")!r})
require(sha(ap)=={sha(ap)!r} and sha(lp)=={sha(lp)!r}, 'approval transfer hash')
lock, _, _=admission(root,ap,lp,'main',Path({(REMOTE + "/main")!r}))
ref=lock['inputs']['dataset']
p, info=artifact(ref,'prepare',lock['frozen_manifest']['sha256'])
require(set(info['files'])=={DATASET_FILES!r}, 'dataset inventory')
for name,key in (('exit.json','exit_sha256'),('release.json','release_sha256')):
    require(sha(p/name)==ref[key], 'completion transfer hash')
require(read(p/'exit.json')['exit_code']==0, 'prepare exit')
r=read(p/'release.json')
require(r['passed'] is True and r['released'] is True and r['artifact_sha256']==ref['seal_sha256'], 'prepare release')
ram=linked({remote_meta["ram-evidence.json"]!r})
age=(datetime.now(timezone.utc)-datetime.fromisoformat(ram['observed_utc'].replace('Z','+00:00'))).total_seconds()
require(-60<=age<=900 and ram['hostname']=='DIS01', 'stale RAM evidence')
env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join(str(root/p) for p in ('wrapper','ai','ai/src'))
env['PYTHONDONTWRITEBYTECODE']='1'
child=subprocess.Popen([sys.executable,'-B','-m','preprocessing_tools.launch','--root',str(root),'--stage','main','--out',{(REMOTE + "/main")!r},'--approval',str(ap),'--lock',str(lp)],env=env)
worker=Path({(REMOTE + "/main/worker.log")!r})
offset=0
def emit_epochs():
    global offset
    if not worker.is_file() or worker.is_symlink(): return
    with worker.open() as f:
        f.seek(offset)
        for _ in range(256):
            start=f.tell()
            line=f.readline(65536)
            if not line or not line.endswith('\\n'):
                f.seek(start)
                break
            if line.startswith('lstm '): print(line.rstrip(),flush=True)
        offset=f.tell()
# Finite metadata forwarding for this launcher only; no extra SSH/allocation/LLM.
# Epoch counts never imply completion percentages. No existing program is killed.
for _ in range(1180):
    try:
        code=child.wait(timeout=10)
        emit_epochs()
        break
    except subprocess.TimeoutExpired:
        emit_epochs()
else:
    print('Metadata forwarding budget exhausted; launcher left untouched.',flush=True)
    code=child.wait()
sys.exit(code)
"""
    write(folder / "remote-bridge.json", {"code": bridge, "package_sha256": PACKAGE_SHA})
    code = runner.run(ssh([REMOTE_PYTHON, "-B", "-c", bridge], options), check=False, stream=True)
    # Fetch once even on failure to retain partial evidence; never resubmit or cancel.
    if safe(MODELS).exists():
        raise ValueError("returned model destination already exists")
    fetch = runner.run(
        ["scp", "-r", "-P", "10000", *options, HOST + ":" + REMOTE + "/main", str(MODELS)],
        check=False,
    )
    if code:
        raise CommandFailure(code, "remote main; state may be unknown, inspect evidence manually")
    if fetch:
        raise CommandFailure(fetch, "model evidence retrieval")
    completion(MODELS, "main", frozen["sha256"], ctl)


def run_stage(stage, folder, ram, runner):
    ctl = control()
    frozen = link(BASE / "frozen-manifest.json")
    protected = link(BASE / "protected-inventory.json")
    if protected["sha256"] != PROTECTED_SHA:
        raise ValueError("protected inventory changed")
    if stage == "prepare":
        ap, lp = approvals(folder, "prepare", PREPARED, frozen, protected, ctl)
        ctl.admission(PACKAGE, ap, lp, "prepare", PREPARED)
        launch("prepare", PREPARED, ap, lp, runner)
        completion(PREPARED, "prepare", frozen["sha256"], ctl)
        return
    dataset, artifact = completion(PREPARED, "prepare", frozen["sha256"], ctl)
    if set(artifact["files"]) != DATASET_FILES:
        raise ValueError("prepared dataset inventory mismatch")
    if stage == "train":
        if safe(MODELS).exists():
            raise ValueError("model destination already exists")
        ram = ram or obtain_ram(folder, runner)
        audit = audit_inputs(folder, frozen, protected, ram, runner)
        remote_train(folder, frozen, protected, dataset, artifact, audit, ram, runner)
        return
    # Revalidate the already completed train audit, rather than inventing a new RAM observation.
    audit = link(META / "operator-train/derived-audit.json")
    quality = read(audit["path"])
    if quality.get("package_sha256") != PACKAGE_SHA:
        raise ValueError("verified derived-audit package required")
    trained, _ = completion(MODELS, "main", frozen["sha256"], ctl)
    ap, lp = approvals(
        folder,
        "calibrate",
        CALIBRATION,
        frozen,
        protected,
        ctl,
        inputs={"dataset": dataset, "trained": trained},
        audit=audit,
    )
    ctl.admission(PACKAGE, ap, lp, "calibrate", CALIBRATION)
    launch("calibrate", CALIBRATION, ap, lp, runner)
    completion(CALIBRATION, "calibrate", frozen["sha256"], ctl)


def execute(stage, ram):
    if sys.version_info[:2] != (3, 10) or os.environ.get("SLURM_JOB_ID"):
        raise ValueError("use existing local Python3.10 outside Slurm")
    if not Path("/media/sa/X31").is_mount() or not safe(META).is_dir():
        raise ValueError("external storage unavailable; no fallback")
    for category in ("processed", "models", "calibration"):
        if not safe(STORAGE / category).is_dir():
            raise ValueError("external output category unavailable")
    # Prerequisite absence is not an attempt; once a claim is made, any failure is terminal.
    safe(BASE / "frozen-manifest.json").stat()
    if stage != "prepare":
        safe(PREPARED / "release.json").stat()
    if stage == "train" and ram is not None:
        ram_check(ram)
    folder = META / ("operator-" + stage)
    with safe(META / "operator.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(
            META / ("operator-" + stage + ".claimed"),
            dict(
                stage=stage,
                pid=os.getpid(),
                uid=os.getuid(),
                source_sha256=sha(__file__),
                exact_message=START,
                retries=0,
                next_stage_authorized=False,
            ),
        )
        safe(folder).mkdir()
        try:
            run_stage(stage, folder, ram, Runner(folder))
        except Exception as error:
            code = error.code if isinstance(error, CommandFailure) else 1
            write(
                folder / "result.json",
                dict(
                    passed=False,
                    exit_code=code,
                    error=str(error),
                    retry=False,
                    next_stage_authorized=False,
                ),
            )
            return code
        write(
            folder / "result.json",
            dict(passed=True, exit_code=0, stage=stage, retry=False, next_stage_authorized=False),
        )
    return 0


def configure(path, stage):
    global PACKAGE, PACKAGE_SHA, AUDITOR, AUDITOR_SHA, PROTECTED_SHA, REMOTE_PACKAGE, REMOTE
    request = read(path)
    if (
        request.get("action") != "START"
        or request.get("authorized") is not True
        or request.get("revision") != "R3_LONG15"
        or request.get("stage") != stage
        or not request.get("exact_message")
    ):
        raise ValueError("explicit same-stage R3 START request required")
    PACKAGE = safe(request["package"])
    PACKAGE_SHA = request["package_sha256"]
    if sha(PACKAGE / "PACKAGE.json") != PACKAGE_SHA:
        raise ValueError("START package hash differs")
    for name in ("frozen-manifest.json", "protected-inventory.json"):
        if sha(safe(BASE / name)) != request["inputs_sha256"][name]:
            raise ValueError("START manifest hash differs")
    PROTECTED_SHA = request["inputs_sha256"]["protected-inventory.json"]
    AUDITOR = PACKAGE / "wrapper/preprocessing_tools/audit.py"
    AUDITOR_SHA = read(PACKAGE / "PACKAGE.json")["files"]["wrapper/preprocessing_tools/audit.py"]
    REMOTE_PACKAGE, REMOTE = request.get("remote_package"), request.get("remote_stage")
    if stage == "train":
        for destination in (REMOTE_PACKAGE, REMOTE):
            if (
                not isinstance(destination, str)
                or not destination.startswith("/abr/coss43/")
                or ".." in Path(destination).parts
                or not all(c.isalnum() or c in "/_.-" for c in destination)
            ):
                raise ValueError("explicit safe user-owned remote paths required")
        if Path(REMOTE).is_relative_to(Path(REMOTE_PACKAGE)) or Path(REMOTE_PACKAGE).is_relative_to(
            Path(REMOTE)
        ):
            raise ValueError("remote package and mutable staging must be disjoint")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "calibrate"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument(
        "--ram-evidence",
        type=Path,
        default=None,
        help="train only: optional verified DIS01 CPU RAM JSON age <=900s; default read-only Slurm inventory, no allocation",
    )
    args = parser.parse_args(argv)
    try:
        configure(args.request, args.stage)
        return execute(args.stage, args.ram_evidence)
    except Exception as error:
        print(f"STOP: {error}. No retry or next stage.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
