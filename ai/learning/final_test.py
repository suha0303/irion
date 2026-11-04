"""One explicitly approved final inference/evaluation, with frozen models and thresholds.

The finite controller runs in server tmux. No training, calibration, model selection,
automatic retries, result interpretation, or LLM calls are implemented here.
"""

import argparse
import csv
import fcntl
import gc
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from execution.common import (
    SUBMIT_LOCK,
    Progress,
    child,
    memory_snapshot,
    python_for,
    read,
    runtime_profile,
    sha,
    verify,
    write,
)

from . import operator, study

SCOPE = "final_test_evaluation"
MODELS = [*study.MODELS, "persistence", "ridge_forecast"]
COUNTS = {"normal_test": 216, "anomaly_test": 108}


def validate_manifest(manifest):
    from collections import Counter, defaultdict

    fixed = dict(
        schema="drone.final-test-dataset/1",
        status="FINAL_DATASET_SEALED_STOPPED",
        scaler_fit_performed=False,
        training_authorized=False,
        calibration_authorized=False,
        final_test_scoring_authorized=False,
    )
    if any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k, v in fixed.items()):
        raise ValueError("Sealed, unscaled, never-scored final dataset required")
    rows = manifest["runs"]
    if (
        len(rows) != 324
        or len({r["run_id"] for r in rows}) != 324
        or Counter(r["role"] for r in rows) != COUNTS
        or Counter(r["final_cohort"] for r in rows)
        != dict(standalone_normal=108, paired_control=108, anomaly=108)
    ):
        raise ValueError("Exact324-run final membership required")
    pairs = defaultdict(list)
    for row in rows:
        if Path(row["run_id"]).name != row["run_id"] or row["run_id"] in {".", ".."}:
            raise ValueError("Unsafe final run identifier")
        if row.get("pair_id"):
            pairs[row["pair_id"]].append(row)
        if row["role"] != ("anomaly_test" if row["final_cohort"] == "anomaly" else "normal_test"):
            raise ValueError("Final cohort/role mismatch")
    if len(pairs) != 108:
        raise ValueError("Exactly108 normal/anomaly pairs required")
    for pair, rr in pairs.items():
        if (
            len(rr) != 2
            or {r["final_cohort"] for r in rr} != {"paired_control", "anomaly"}
            or any(
                rr[0][k] != rr[1][k]
                for k in (
                    "seed",
                    "group_id",
                    "motion",
                    "environment",
                    "mission_seconds",
                    "operation_variation",
                )
            )
            or set(manifest["pairs"][pair]) != {r["run_id"] for r in rr}
        ):
            raise ValueError("Final paired conditions changed")
    if manifest["zero_target_runs"]:
        raise ValueError("Final run without evaluable targets")


def config_check(cfg):
    fixed = dict(
        schema="drone-ai.final-test/1",
        execution_scope=SCOPE,
        models=MODELS,
        seed=0,
        prediction_batch_size=256,
        precision="float32",
        amp=False,
        production_training_authorized=False,
        calibration_authorized=False,
        final_test_authorized=True,
        final_test_used=True,
        analysis_authorized=False,
        automatic_retry=False,
        retries=0,
        next_round_authorized=False,
        cpu_workers=8,
        cpus=10,
        memory_gib=64,
        gpu_jobs_max=1,
        counts=COUNTS,
        wall_hours=24,
        evaluation_hours=12,
        queue_hours=24,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("One fixed18-model final evaluation without fitting or analysis required")
    from execution.evaluate import final_protocol

    if cfg["protocol"] != final_protocol(cfg["source_protocol"]):
        raise ValueError("Fixed final protocol differs from source")


def checked(root, *, full=False):
    from console.profile import absolute_path

    root = absolute_path(root)
    cfg = read(root / "input/config.json")
    config_check(cfg)
    auth = read(root / "authorization.json")
    if (
        auth.get("config_sha256") != sha(root / "input/config.json")
        or auth.get("final_test_authorized") is not True
        or auth.get("analysis_authorized") is not False
        or auth.get("inventory_sha256") != sha(root / "input/inventory.json")
    ):
        raise ValueError("Final authorization binding changed")
    verify(root / "input", read(root / "input/inventory.json"))
    profile = runtime_profile(root, required=True)
    fixed = dict(partition="p02", node="DIS01", nodes=1, ntasks=1, gpus=1, cpus=10, memory_gib=64)
    if profile["slurm"] != fixed or sha(root / "server-profile.json") != cfg["profile_sha256"]:
        raise ValueError("Only selected p02/DIS01 single-GPU profile allowed")
    if Path(profile["output_root"]) != root or Path(profile["dataset_root"]) != Path(
        cfg["dataset"]
    ):
        raise ValueError("Selected profile paths differ")
    source = Path(cfg["source_evaluation"])
    verify(source, cfg["source_files"])
    frozen = read(root / "input/dataset/provenance/sealed-manifest.json")
    validate_manifest(frozen)
    if sha(root / "input/dataset/provenance/sealed-manifest.json") != cfg["final_manifest_sha256"]:
        raise ValueError("Final dataset seal changed")
    if full:
        verify(root / "input/dataset", cfg["dataset_files"])
        verify(root / "input/truth", cfg["truth_files"])
    for name in MODELS:
        folder = root / "compute/models" / name
        verify(folder, read(folder / "integrity.json"))
        if sha(folder / "integrity.json") != cfg["bundle_integrities"][name]:
            raise ValueError("Frozen model integrity changed")
        if sha(folder / "scaler.npz") != cfg["common_scaler_sha256"]:
            raise ValueError("Final inference must use the common training scaler")
        cal = root / "compute/calibration" / name / "calibration.json"
        if (
            sha(cal) != cfg["calibration_hashes"][name]
            or read(cal)["bundle_integrity_sha256"] != cfg["bundle_integrities"][name]
        ):
            raise ValueError("Frozen normal-only calibration changed")
    return root, cfg


def prepare(args):
    from console.profile import absolute_path, load
    from execution.evaluate import final_protocol, protocol_hash

    if not args.approve_final_test:
        raise ValueError("Explicit final-test approval required")
    root, source, data = map(absolute_path, (args.root, args.source, args.data))
    if root.exists() or any(
        root == p or root in p.parents or p in root.parents for p in (source, data)
    ):
        raise ValueError("New separate output required; preserve every existing input")
    completion = read(source / "completion.json")
    if completion["status"] != "REPORT_READY_STOPPED" or completion["final_test_used"] is not False:
        raise ValueError("Completed development calibration source required")
    for key in ("calibration_release", "evaluation_release"):
        operator.require_success(completion[key])
    source_protocol = read(source / "evaluation/protocol.json")
    report = read(source / "evaluation/report.json")
    seal = read(source / "compute/result-seal.json")
    if (
        sha(source / "compute/result-seal.json") != completion["compute_seal_sha256"]
        or sha(source / "evaluation/report.json") != completion["evaluation_report_sha256"]
        or protocol_hash(source_protocol) != report["protocol_sha256"]
        or seal["models"] != MODELS
        or seal["final_test_used"] is not False
        or seal.get("all_model_sensitivity_passed") is not True
    ):
        raise ValueError("Preserved18-model development proof changed")
    transfer = read(data / "transfer.json")
    verify(data, transfer["files"])
    frozen = read(data / "dataset/provenance/sealed-manifest.json")
    validate_manifest(frozen)
    if (
        sha(data / "dataset/provenance/sealed-manifest.json")
        != read(data / "dataset/provenance/seal.json")["manifest_sha256"]
    ):
        raise ValueError("Final manifest seal mismatch")
    protected_path = data / "dataset/provenance/final-protection.json"
    if sha(protected_path) != frozen["protection"]["sha256"]:
        raise ValueError("Frozen train/development protection inventory changed")
    protected = read(protected_path)
    for field, values in (
        ("seeds", {r["seed"] for r in frozen["runs"]}),
        ("group_ids", {r["group_id"] for r in frozen["runs"]}),
        (
            "raw_ids",
            {r["run_id"] for r in frozen["runs"]} | {r["raw_flight_id"] for r in frozen["runs"]},
        ),
        ("raw_sha256", {r["raw_sha256"] for r in frozen["runs"]}),
    ):
        if values & set(protected[field]):
            raise ValueError("Final data overlap the frozen train/development inventory")
    profile = load(args.profile)
    if (
        Path(profile["output_root"]) != root
        or Path(profile["dataset_root"]) != root / "input/dataset"
    ):
        raise ValueError("Final profile paths differ")
    root.mkdir()
    (root / "input").mkdir()
    runtime = Path(__file__).resolve().parents[1]
    runtime_inventory = read(runtime.parent / "SOURCES.json")["files"]
    for rel, digest in runtime_inventory.items():
        if not rel.startswith("ai/") or sha(runtime.parent / rel) != digest:
            raise ValueError("Frozen runtime source changed")
        target = child(root / "input", rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runtime.parent / rel, target)
    # The uploaded staging directory is exclusively owned by this deployment.
    # Moving avoids another11.9GB copy and cannot alter the preserved local source.
    for name in ("dataset", "truth"):
        shutil.move(str(data / name), str(root / "input" / name))
    shutil.copyfile(args.profile, root / "server-profile.json")
    source_files = {
        n: sha(source / n)
        for n in (
            "completion.json",
            "compute/result-seal.json",
            "evaluation/report.json",
            "evaluation/protocol.json",
        )
    }
    bundles, calibrations = {}, {}
    for name in MODELS:
        folder = source / "compute/models" / name
        entries = read(folder / "integrity.json")
        entries["integrity.json"] = sha(folder / "integrity.json")
        for filename, digest in entries.items():
            rel = f"models/{name}/{filename}"
            if seal["files"].get(rel) != digest or sha(child(folder, filename)) != digest:
                raise ValueError("Model export differs from completed development seal")
            target = child(root / "compute", rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(child(folder, filename), target)
        rel = f"calibration/{name}/calibration.json"
        calibration = child(source / "compute", rel)
        if sha(calibration) != seal["files"].get(rel):
            raise ValueError("Calibration differs from completed development seal")
        target = child(root / "compute", rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(calibration, target)
        bundles[name] = entries["integrity.json"]
        calibrations[name] = sha(target)
    common_preprocessing = read(root / "compute/models" / MODELS[0] / "preprocessing.json")
    if any(
        read(root / "compute/models" / name / "preprocessing.json") != common_preprocessing
        for name in MODELS
    ):
        raise ValueError("Every model must use the same frozen preprocessing contract")
    for row in frozen["runs"]:
        summary = read(root / "input/dataset/canonical" / row["run_id"] / "summary.json")
        if summary["config"] != common_preprocessing or summary["run_id"] != row["run_id"]:
            raise ValueError("Final canonical preprocessing differs from the trained models")
    dataset_files = {
        n.removeprefix("dataset/"): h
        for n, h in transfer["files"].items()
        if n.startswith("dataset/")
    }
    truth_files = {
        n.removeprefix("truth/"): h for n, h in transfer["files"].items() if n.startswith("truth/")
    }
    cfg = dict(
        schema="drone-ai.final-test/1",
        execution_scope=SCOPE,
        models=MODELS,
        seed=0,
        prediction_batch_size=256,
        precision="float32",
        amp=False,
        production_training_authorized=False,
        calibration_authorized=False,
        final_test_authorized=True,
        final_test_used=True,
        analysis_authorized=False,
        automatic_retry=False,
        retries=0,
        next_round_authorized=False,
        cpu_workers=8,
        cpus=10,
        memory_gib=64,
        gpu_jobs_max=1,
        counts=COUNTS,
        wall_hours=24,
        evaluation_hours=12,
        queue_hours=24,
        source_evaluation=str(source),
        source_files=source_files,
        source_protocol=source_protocol,
        protocol=final_protocol(source_protocol),
        dataset=str(root / "input/dataset"),
        dataset_files=dataset_files,
        truth_files=truth_files,
        final_manifest_sha256=sha(root / "input/dataset/provenance/sealed-manifest.json"),
        bundle_integrities=bundles,
        calibration_hashes=calibrations,
        common_scaler_sha256=sha(root / "compute/models" / MODELS[0] / "scaler.npz"),
        profile_sha256=sha(root / "server-profile.json"),
        reports_only=True,
    )
    write(root / "input/config.json", cfg)
    inventory = {n: h for n, h in runtime_inventory.items()}
    inventory.update({"config.json": sha(root / "input/config.json")})
    # Bulk canonical data are verified once in each allocated phase, not on every UI read.
    inventory["dataset/manifest.json"] = sha(root / "input/dataset/manifest.json")
    inventory["dataset/provenance/sealed-manifest.json"] = cfg["final_manifest_sha256"]
    write(root / "input/inventory.json", inventory)
    write(
        root / "authorization.json",
        dict(
            final_test_authorized=True,
            analysis_authorized=False,
            production_training_authorized=False,
            calibration_authorized=False,
            config_sha256=sha(root / "input/config.json"),
            inventory_sha256=sha(root / "input/inventory.json"),
            automatic_retry=False,
            next_round_authorized=False,
            user_request=args.approval_text,
        ),
    )
    checked(root, full=True)
    write(
        root / "prepared.json",
        dict(
            status="FINAL_EVALUATION_PREPARED",
            models=MODELS,
            final_runs=324,
            config_sha256=sha(root / "input/config.json"),
            training=False,
            calibration=False,
        ),
    )


def inference_seal(root, cfg):
    seal = read(root / "compute/result-seal.json")
    if (
        seal.get("status") != "FINAL_INFERENCE_COMPLETE"
        or seal.get("models") != MODELS
        or seal.get("truth_join") is not False
        or seal.get("final_test_used") is not True
        or seal.get("config_sha256") != sha(root / "input/config.json")
        or seal.get("calibration_hashes") != cfg["calibration_hashes"]
    ):
        raise ValueError("Complete final inference seal required before evaluator truth join")
    return seal


def submit(root, phase, cfg):
    if phase not in {"infer", "evaluate"}:
        raise ValueError("Final inference and evaluation only; no training/calibration")
    checked(root)
    with SUBMIT_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if operator.own_queue():
            raise ValueError("Own pending/running job exists; no concurrent submission")
        if phase == "evaluate":
            operator.require_success(read(root / "release.json"))
            inference_seal(root, cfg)
        with (root / (phase + "-submitted.claim")).open("x") as f:
            f.write("One submission. Unknown outcomes must never be retried.\n")
        script = root / (phase + "-job.sh")
        script.write_text(
            "#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH="
            + shlex.quote(str(root / "input/ai") + ":" + str(root / "input/ai/src"))
            + "\nexport DRONE_AI_PROFILE="
            + shlex.quote(str(root / "server-profile.json"))
            + "\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
            + "export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2\nexec "
            + shlex.join(
                [
                    python_for(root, required=True),
                    "-B",
                    "-m",
                    "learning.final_test",
                    phase,
                    "--root",
                    str(root),
                ]
            )
            + "\n"
        )
        hours = cfg["wall_hours"] if phase == "infer" else cfg["evaluation_hours"]
        argv = [
            "sbatch",
            "--parsable",
            "--partition=p02",
            "--nodelist=DIS01",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            "--cpus-per-task=10",
            "--mem=64G",
            f"--time={hours}:00:00",
            "--no-requeue",
            "--job-name=drone-ai-final-" + phase,
            "--output=" + str(root / (phase + "-slurm.log")),
            "--error=" + str(root / (phase + "-slurm.log")),
            "--chdir=" + str(root),
            str(script),
        ]
        write(
            root / (phase + "-submission-intent.json"),
            dict(argv=argv, time=time.time(), automatic_retry=False),
        )
        identifier = operator.command(argv).split(";")[0]
        if not identifier.isdigit():
            raise ValueError("Unknown submission outcome; no retry")
        record = dict(job_id=identifier, uid=os.getuid(), phase=phase, submitted=time.time())
        write(root / (phase + "-submission.json"), record)
        return record


def wait_files(root, phase):
    limit = time.monotonic() + 120
    paths = (
        ["compute/result-seal.json"]
        if phase == "infer"
        else ["evaluation/report.json", "evaluation/metrics.json", "evaluation/evidence.json"]
    )
    while True:
        if all((root / n).is_file() for n in paths):
            return
        if time.monotonic() > limit:
            raise RuntimeError("Completed job output not visible; no retry")
        time.sleep(5)


def control(root):
    root, cfg = checked(root)
    if os.environ.get("SLURM_JOB_ID") or not os.environ.get("TMUX"):
        raise ValueError("Final controller requires server login-node tmux")
    with (root / "controller.claim").open("x") as f:
        f.write("One final evaluation; no retry or next round.\n")
    state = dict(
        status="RUNNING",
        stage="infer",
        execution_scope=SCOPE,
        pid=os.getpid(),
        started=time.time(),
        llm_calls=0,
        retries=0,
        analysis_authorized=False,
        next_round_authorized=False,
    )
    record = None
    try:
        for phase in ("infer", "evaluate"):
            record = None
            state.update(stage=phase, allocation_released=False)
            write(root / "controller.json", state)
            record = submit(root, phase, cfg)
            budget = {
                **cfg,
                "wall_hours": cfg["wall_hours"] if phase == "infer" else cfg["evaluation_hours"],
            }
            release = operator.wait_release(root, record, state, budget)
            write(
                root / ("release.json" if phase == "infer" else "evaluation-release.json"), release
            )
            operator.require_success(release)
            state.update(allocation_released=True)
            wait_files(root, phase)
        verify(root / "compute", inference_seal(root, cfg)["files"])
        report = read(root / "evaluation/report.json")
        if (
            sha(root / "evaluation/metrics.json") != report["metrics_sha256"]
            or sha(root / "evaluation/evidence.json") != report["evidence_sha256"]
        ):
            raise ValueError("Final evaluation result binding changed")
        checked(root)
        if operator.own_queue():
            raise ValueError("Own queue not empty at completion")
        write(
            root / "completion.json",
            dict(
                status="REPORT_READY_STOPPED",
                finished=time.time(),
                execution_scope=SCOPE,
                final_test_used=True,
                analysis_performed=False,
                training_repeated=False,
                calibration_refitted=False,
                models=MODELS,
                inference_release=read(root / "release.json"),
                evaluation_release=read(root / "evaluation-release.json"),
                compute_seal_sha256=sha(root / "compute/result-seal.json"),
                metrics_sha256=report["metrics_sha256"],
                evaluation_report_sha256=sha(root / "evaluation/report.json"),
                records_sha256=sha(root / "evaluation/model-results.csv"),
                automatic_retry=False,
                llm_calls=0,
                next_round_authorized=False,
            ),
        )
        state.update(
            status="REPORT_READY_STOPPED",
            stage="stopped",
            finished=time.time(),
            allocation_released=True,
        )
        write(root / "controller.json", state)
    except BaseException as exc:
        released = False
        if record is not None:
            try:
                released = operator.snapshot(record).get("released", False)
            except Exception:
                pass
        state.update(
            status="FAILED_STOPPED",
            reason=f"{type(exc).__name__}: {exc}",
            allocation_released=released,
            finished=time.time(),
            automatic_retry=False,
        )
        write(root / "controller.json", state)
        raise


def start(root):
    root, _ = checked(root)
    if (
        os.environ.get("SLURM_JOB_ID")
        or Path(sys.executable).resolve() != Path(python_for(root, required=True)).resolve()
    ):
        raise ValueError("Start with selected server Python on login node")
    with (root / "start.claim").open("x") as f:
        f.write("One final evaluation controller launch; no retry.\n")
    name = "drone-ai-final-" + root.name
    command = (
        "export PYTHONPATH="
        + shlex.quote(str(root / "input/ai") + ":" + str(root / "input/ai/src"))
        + "; export DRONE_AI_PROFILE="
        + shlex.quote(str(root / "server-profile.json"))
        + "; export PYTHONDONTWRITEBYTECODE=1; exec "
        + shlex.join(
            [
                python_for(root, required=True),
                "-B",
                "-m",
                "learning.final_test",
                "control",
                "--root",
                str(root),
            ]
        )
        + " >> "
        + shlex.quote(str(root / "controller.log"))
        + " 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", name, command], check=True)
    print("Finite final evaluation handed to server tmux: " + name)


def worker(root, phase):
    from pipeline.runtime import check

    from .diagnostic import stop_signal
    from .training import backend, project_lock

    root, cfg = checked(root, full=True)
    path = root / ("compute/progress.json" if phase == "infer" else "evaluation-progress.json")
    hours = cfg["wall_hours"] if phase == "infer" else cfg["evaluation_hours"]
    progress = Progress(path, hours=hours - 0.1)
    progress.update(final_test_used=True, training_authorized=False, calibration_authorized=False)
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    telemetry = None
    try:
        runtime_out = root / ("compute/runtime" if phase == "infer" else "evaluation-runtime")
        runtime_out.mkdir(exist_ok=False)
        environment = check(root / "input/ai", runtime_out, phase, job_prefix="drone-ai-final-")
        memory = memory_snapshot()
        write(
            runtime_out / "memory-admission.json",
            dict(snapshot=memory, required_bytes=16 * 1024**3),
        )
        if (
            int(environment["job"]["NumCPUs"]) < 10
            or memory["errors"]
            or memory["available_bytes"] < 16 * 1024**3
        ):
            raise MemoryError("Final evaluation allocation/RAM admission failed")
        if phase == "infer":
            from capstone_ai.training import Predictor

            from .parallel_replay import ParallelDevelopment
            from .telemetry import Telemetry

            telemetry = Telemetry(root / "compute")
            telemetry.start()
            rows = read(Path(cfg["dataset"]) / "manifest.json")["runs"]
            with project_lock("cuda"), ParallelDevelopment(8, progress) as executor:
                for index, name in enumerate(MODELS):
                    progress.update(
                        stage="final_inference", model=name, models_completed=index, models_total=18
                    )
                    bundle = root / "compute/models" / name
                    predictor = Predictor(bundle, device="cuda")
                    calibration = read(root / "compute/calibration" / name / "calibration.json")
                    with backend("cuda"):
                        executor.run(
                            cfg["dataset"],
                            bundle,
                            calibration,
                            root / "compute/calibration" / name,
                            predictor,
                            rows,
                            256,
                            name,
                        )
                    del predictor
                    gc.collect()
                    progress.update(models_completed=index + 1, models_total=18)
            telemetry.stop()
            telemetry = None
            checked(root)
            out = root / "compute"
            files = {
                str(p.relative_to(out)): sha(p)
                for p in out.rglob("*")
                if p.is_file() and p.name not in {"progress.json", "result-seal.json"}
            }
            write(
                out / "result-seal.json",
                dict(
                    status="FINAL_INFERENCE_COMPLETE",
                    models=MODELS,
                    files=files,
                    seed=0,
                    config_sha256=sha(root / "input/config.json"),
                    final_test_used=True,
                    truth_join=False,
                    training_repeated=False,
                    calibration_refitted=False,
                    calibration_hashes=cfg["calibration_hashes"],
                    prediction_batch_size=256,
                    cpu_workers=8,
                    automatic_retry=False,
                    next_round_authorized=False,
                ),
            )
            progress.update(status="FINAL_INFERENCE_COMPLETE_PENDING_RELEASE", stage="stopped")
        else:
            from execution.evaluate import evaluate

            operator.require_success(read(root / "release.json"))
            verify(root / "compute", inference_seal(root, cfg)["files"])

            def truth_provider(row, target, budget):
                budget.check()
                name = row["run_id"] + ".jsonl"
                source = child(root / "input/truth", name)
                if sha(source) != cfg["truth_files"][name]:
                    raise ValueError("Final evaluator truth binding changed")
                shutil.copyfile(source, target)

            with project_lock("cuda"):
                metrics = evaluate(
                    cfg,
                    root / "compute",
                    root / "evaluation",
                    progress,
                    truth_provider=truth_provider,
                    model_names=MODELS,
                    development_counts=COUNTS,
                    evaluation_mode="final_test",
                    completion_status="EVALUATION_COMPLETE_PENDING_RELEASE",
                )
            # Record fixed metrics in model declaration order. Never rank, select or interpret.
            with (root / "evaluation/model-results.csv").open("x") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "model",
                        "events",
                        "detected",
                        "unavailable",
                        "recall",
                        "normal_scored_hours",
                        "normal_union_alarms",
                        "normal_union_alarms_per_hour",
                    ]
                )
                for name in MODELS:
                    m = metrics["models"][name]
                    writer.writerow(
                        [
                            name,
                            m["events"]["events"],
                            m["events"]["detected"],
                            m["events"]["unavailable"],
                            m["events"]["recall_all"],
                            m["normal_scored_hours"],
                            m["normal_episodes"]["union"],
                            m["normal_episodes_per_hour"]["union"],
                        ]
                    )
            progress.update(status="EVALUATION_COMPLETE_PENDING_RELEASE", stage="stopped")
    except BaseException as exc:
        progress.update(
            status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}", automatic_retry=False
        )
        raise
    finally:
        if telemetry is not None:
            telemetry.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    for key in ("root", "source", "data", "profile", "approval-text"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--approve-final-test", action="store_true")
    for action in ("start", "control", "infer", "evaluate"):
        sub.add_parser(action).add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "start":
        start(args.root)
    elif args.action == "control":
        control(args.root)
    else:
        worker(args.root, args.action)


if __name__ == "__main__":
    main()
