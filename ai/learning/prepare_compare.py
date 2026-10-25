"""Human-operated preparation/transfer/start CLI. No automatic GPU submission."""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from execution.common import ROOT, Progress, python_for, read, runtime_profile, sha, verify, write

from . import comparison

HOME = ROOT / "ai/configs"
DEFAULT_BUNDLE = Path(ROOT / "Data/processed/pretrain/merge-1200-20261017-01/prepared-portable")


def inventory(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in root.rglob("*")
        if p.is_file() and p != root / "inventory.json"
    }


def remote_path(path):
    p = Path(path)
    if (
        not p.is_absolute()
        or ".." in p.parts
        or p == Path("/")
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", p.name)
    ):
        raise ValueError("A new absolute attempt directory with a safe name is required")
    return p


def external_root(path):
    from x500_storage import guard

    candidate = Path(path).expanduser().absolute()
    if ".." in candidate.parts:
        raise ValueError("Output path traversal forbidden")
    policy_file = candidate / "storage-policy.json"
    # Validate the path before opening the local policy (no symlink traversal).
    guard(candidate)
    return guard(candidate, read(policy_file) if policy_file.exists() else None)


class PreparationProgress(Progress):
    """Check the selected filesystem while restoring/writing preparation files."""

    def __init__(self, path, *, storage_root, hours):
        from x500_storage import capture

        self.storage_root = storage_root
        self.storage_policy = capture(storage_root)
        super().__init__(path, hours=hours)

    def check(self):
        external_root(self.storage_root)
        from x500_storage import guard

        guard(self.storage_root, self.storage_policy)
        super().check()

    def update(self, **fields):
        external_root(self.storage_root)
        from x500_storage import guard

        guard(self.storage_root, self.storage_policy)
        super().update(**fields)


def restore_ref(bundle, ref, destination, progress):
    from execution.common import child
    from pretrain.storage import verify_bundle

    progress.check()
    source = child(bundle, ref["path"])
    if source.stat().st_size != ref["bytes"] or sha(source) != ref["sha256"]:
        raise ValueError("Portable archive changed")
    progress.check()
    destination.mkdir(parents=True, exist_ok=False)
    verify_bundle(
        source, ref["members"], progress, destination, progress=lambda **kw: progress.update(**kw)
    )


def prepare(args):
    from console.profile import load

    profile = load(args.profile)
    if any(
        type(value) is not int or not 1 <= value <= 168
        for value in (args.wall_hours, args.evaluation_hours)
    ):
        raise ValueError("Explicit finite fit/evaluation hours within1..168 required")
    root, bundle = external_root(args.root), args.bundle.resolve()
    if root.is_relative_to(bundle) or bundle.is_relative_to(root):
        raise ValueError("Preparation must use a separate new directory")
    remote = remote_path(args.remote_root)
    plan = read(HOME / "comparison-plan.json")
    receipt = read(bundle / "receipt.json")
    if (
        sha(bundle / "receipt.json") != plan["dataset"]["receipt_sha256"]
        or receipt.get("status") != "PORTABLE_DATASET_VERIFIED"
        or receipt.get("final_test_included") is not False
        or receipt["dataset"]["sha256"] != plan["dataset"]["archive_sha256"]
    ):
        raise ValueError("Only the frozen merged1200 portable dataset is admitted")
    need = receipt["uncompressed_dataset_bytes"] + 24 * 1024**3
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < need:
        raise OSError("Insufficient selected storage: restored data plus24GiB reserve required")
    root.mkdir(exist_ok=False)
    from x500_storage import capture

    write(root / "storage-policy.json", capture(root))
    write(root / "server-profile.json", profile)
    progress = PreparationProgress(root / "prepare-progress.json", storage_root=root, hours=12)
    try:
        upstream = root / "upstream"
        progress.update(
            stage="restore_verify_dataset",
            round=comparison.REVISION,
            done=0,
            total=len(receipt["dataset"]["members"]),
        )
        restore_ref(bundle, receipt["dataset"], upstream, progress)
        rows, seal, headers = comparison.admission(upstream)
        preprocess = ROOT / "Data/generation/pretrain/resources/preprocessing/preprocessing.json"
        frozen = read(upstream / "evidence/frozen-manifest.json")
        if sha(preprocess) != frozen["revision_lock"]["preprocessing_sha256"]:
            raise ValueError("Preprocessing differs from frozen dataset revision")
        progress.check()
        shutil.copyfile(preprocess, upstream / "preprocessing.json")
        progress.update(
            stage="prepare_evaluator_only_truth",
            done=0,
            total=len(receipt["evaluator_truth"]["members"]),
        )
        restore_ref(bundle, receipt["evaluator_truth"], root / "truth", progress)
        index = {}
        for row in rows:
            if row["role"] not in {"normal_dev_eval", "anomaly_dev"}:
                continue
            name = row["run_id"] + ".jsonl"
            expected = row["archive"]["members"]["evaluation/events.jsonl"]
            if receipt["evaluator_truth"]["members"].get(name) != {
                k: expected[k] for k in ("bytes", "sha256")
            }:
                raise ValueError("Development truth differs from frozen source")
            index[row["run_id"]] = dict(
                sha256=expected["sha256"], bytes=expected["bytes"], role=row["role"]
            )
        if set(receipt["evaluator_truth"]["members"]) != {k + ".jsonl" for k in index}:
            raise ValueError("Unexpected evaluator truth membership")
        write(root / "truth/index.json", index)
        inputs = root / "input"
        sources = {}
        for folder in ("execution", "learning", "pipeline", "src/capstone_ai"):
            for source in (ROOT / "ai" / folder).rglob("*.py"):
                progress.check()
                target = inputs / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                sources[str(source.relative_to(ROOT))] = sha(target)
        for source in (ROOT / "ai/console").rglob("*.py"):
            target = inputs / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sources[str(source.relative_to(ROOT))] = sha(target)
        probe = inputs / "ai/remote/gpu_probe.py"
        probe.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "ai/remote/gpu_probe.py", probe)
        sources["ai/remote/gpu_probe.py"] = sha(probe)
        profile = inputs / "ai/remote/environment-mamba25.json"
        shutil.copy2(ROOT / "ai/remote/environment-mamba25.json", profile)
        sources["ai/remote/environment-mamba25.json"] = sha(profile)
        memory = comparison.memory_plan(headers)
        cfg = dict(
            revision=comparison.REVISION,
            models=comparison.MODELS,
            variants=comparison.VARIANTS,
            seed=0,
            batch_size=256,
            epochs=60,
            early_stopping_patience=10,
            learning_rate=0.001,
            dropout=0.1,
            weight_decay=0.00001,
            model_revision="explicit_dropout_v1",
            precision="float32",
            amp=False,
            retries=0,
            final_test_used=False,
            gate_epochs=4,
            gate_windows=512,
            production_training_authorized=False,
            wall_hours=args.wall_hours,
            evaluation_hours=args.evaluation_hours,
            queue_hours=24,
            memory_gib=64,
            cpus=4,
            memory=memory,
            training_stride_us=1000000,
            history=100,
            future=10,
            step_us=100000,
            models_sequential=True,
            gpu_jobs_max=1,
            next_round_authorized=False,
            development_counts={
                k: comparison.COUNTS[k] for k in ("normal_dev_eval", "anomaly_dev")
            },
            upstream=str(remote / "upstream"),
            upstream_files={
                **{"dataset/" + k: v for k, v in seal.items()},
                **{"evidence/" + p.name: sha(p) for p in (upstream / "evidence").iterdir()},
                "preprocessing.json": sha(preprocess),
            },
            dataset_files=seal,
            handoff_sha256=sha(upstream / "evidence/handoff.json"),
            truth_index_sha256=sha(root / "truth/index.json"),
            binding=dict(
                data={k: seal[k] for k in ("normal_train.npz", "normal_validation.npz")},
                scaler=seal["scaler.npz"],
                sources=sources,
            ),
            source_dataset_sha256=receipt["dataset"]["sha256"],
            failure_policy="Preserve evidence and stop, no retry or automatic tuning",
        )
        comparison.config_check(cfg, require_approval=False)
        progress.check()
        write(inputs / "config.json", cfg)
        write(inputs / "inventory.json", inventory(inputs))
        write(
            root / "prepared.json",
            dict(
                status="PREPARED_PENDING_TRAINING_APPROVAL",
                remote_root=str(remote),
                local_root=str(root),
                source_bundle=str(bundle),
                source_receipt_sha256=sha(bundle / "receipt.json"),
                input_inventory_sha256=sha(inputs / "inventory.json"),
                source_ai_sha256=sources,
                server_profile_sha256=sha(root / "server-profile.json"),
                submitted_jobs=[],
                llm_calls=0,
                memory=memory,
                fit_budget_hours=args.wall_hours,
                evaluation_budget_hours=args.evaluation_hours,
            ),
        )
        progress.update(status="PREPARED_PENDING_TRAINING_APPROVAL", stage="stopped")
    except BaseException as exc:
        progress.update(status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}")
        raise


def authorize(root, confirmed):
    if not confirmed:
        raise ValueError("Explicit --approve-one-comparison is required; no GPU work is started")
    root = external_root(root)
    if (root / "controller.claim").exists() or (root / "authorization.json").exists():
        raise ValueError("Already claimed/authorized; never rewrite or automatically retry")
    verify(root / "input", read(root / "input/inventory.json"))
    cfg = read(root / "input/config.json")
    comparison.config_check(cfg, require_approval=False)
    write(root / "config-before-authorization.json", cfg)
    cfg["production_training_authorized"] = True
    comparison.config_check(cfg)
    write(root / "input/config.json", cfg)
    write(root / "input/inventory.json", inventory(root / "input"))
    receipt = read(root / "prepared.json")
    receipt.update(
        input_inventory_sha256=sha(root / "input/inventory.json"), status="AUTHORIZED_NOT_STARTED"
    )
    write(root / "prepared.json", receipt)
    write(
        root / "authorization.json",
        dict(
            approved=time.time(),
            models=cfg["models"],
            seed=0,
            config_sha256=sha(root / "input/config.json"),
            no_next_round=True,
        ),
    )


def deploy(root):
    root = external_root(root)
    receipt = read(root / "prepared.json")
    if sha(root / "input/inventory.json") != receipt["input_inventory_sha256"]:
        raise ValueError("Prepared inventory changed")
    verify(root / "input", read(root / "input/inventory.json"))
    if sha(root / "server-profile.json") != receipt["server_profile_sha256"]:
        raise ValueError("Selected server profile changed")
    target = remote_path(receipt["remote_root"])
    profile = runtime_profile(root, required=True)
    connection = profile.get("ssh")
    if not connection:
        raise ValueError("Deployment requires explicit SSH connection details in the profile")
    destination = connection["host"]
    if connection.get("user"):
        destination = connection["user"] + "@" + destination
    ssh_options = [
        "-p",
        str(connection["port"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
    ]
    ssh = ["ssh", *ssh_options, destination]
    # Exclusive directory prevents overwriting another attempt or an active job.
    subprocess.run([*ssh, "mkdir -- " + shlex.quote(str(target))], check=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--info=progress2",
            "-e",
            shlex.join(["ssh", *ssh_options]),
            str(root) + "/",
            destination + ":" + str(target) + "/",
        ],
        check=True,
    )
    print("Transferred; no GPU job submitted. Remote start: " + str(target / "start.sh"))


def start(root):
    remote_path(root)
    approved_python = python_for(root, required=True)
    if Path(os.sys.executable).resolve() != Path(approved_python).resolve():
        raise ValueError("Start on the selected server with its profile Python interpreter")
    cfg = read(root / "input/config.json")
    comparison.config_check(cfg)
    verify(root / "input", read(root / "input/inventory.json"))
    if (root / "controller.claim").exists():
        raise ValueError("Attempt already claimed; no automatic retry")
    name = "drone-ai-" + root.name
    command = (
        "export PYTHONPATH="
        + shlex.quote(str(root / "input/ai") + ":" + str(root / "input/ai/src"))
        + "; export DRONE_AI_PROFILE="
        + shlex.quote(str(root / "server-profile.json"))
        + "; exec "
        + shlex.join(
            [approved_python, "-B", "-m", "learning.operator", "control", "--root", str(root)]
        )
        + " >> "
        + shlex.quote(str(root / "controller.log"))
        + " 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", name, command], check=True)
    print("Finite controller started in server tmux: " + name)


def launchers(root, remote):
    approved_python = python_for(root, required=True)
    root = external_root(root)
    prefix = str(remote / "input/ai") + ":" + str(remote / "input/ai/src")
    for name, module, action in (
        ("start.sh", "learning.prepare_compare", "start"),
        ("watch.sh", "learning.operator", "status"),
    ):
        body = (
            "#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH="
            + shlex.quote(prefix)
            + "\nexport DRONE_AI_PROFILE="
            + shlex.quote(str(remote / "server-profile.json"))
            + "\nexport PYTHONDONTWRITEBYTECODE=1\nexec "
            + shlex.join([approved_python, "-B", "-m", module, action, "--root", str(remote)])
            + "\n"
        )
        (root / name).write_text(body)
        (root / name).chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare", help="선택한 로컬 저장소에서 복원·검증; 학습 없음")
    p.add_argument(
        "--root",
        type=Path,
        default=ROOT / "Data/learning/comparison",
        help="학습 준비 출력 폴더; 기본 프로젝트/Data/learning/comparison",
    )
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--remote-root", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True, help="직접 입력한 서버·Python 프로필")
    p.add_argument("--wall-hours", type=int, required=True)
    p.add_argument("--evaluation-hours", type=int, required=True)
    for action in ("authorize", "deploy", "start", "status"):
        p = sub.add_parser(action)
        p.add_argument("--root", type=Path, required=True)
        if action == "authorize":
            p.add_argument("--approve-one-comparison", action="store_true")
        if action == "status":
            p.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
        launchers(args.root, remote_path(args.remote_root))
    elif args.action == "authorize":
        authorize(args.root, args.approve_one_comparison)
    elif args.action == "deploy":
        deploy(args.root)
    elif args.action == "start":
        start(args.root.resolve())
    else:
        from .operator import display

        display(args.root.resolve(), args.once)


if __name__ == "__main__":
    main()
