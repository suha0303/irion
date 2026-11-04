"""One finite, separately authorized calibration/development-evaluation continuation.

The completed training attempt and its upstream are immutable inputs. This module
has no fit entry point, does not resume training, and never retries a submitted job.
"""

import argparse
import fcntl
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from execution.common import (
    ROOT,
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

from . import comparison, operator
from .prepare_compare import external_root, inventory, remote_path

SCHEMA = "drone-ai.post-training/1"
SCOPE = "post_training_evaluation"
COMPLETE = "TRAINING_READY_STOPPED_PENDING_EVALUATION_APPROVAL"
BASELINES = ("persistence", "ridge_forecast")


def config_check(cfg):
    from . import study

    source_revision = cfg.get("source_comparison_revision", comparison.REVISION)
    if source_revision not in {comparison.REVISION, study.REVISION}:
        raise ValueError("Unknown preserved comparison revision")
    expected_models = list(study.MODELS if source_revision == study.REVISION else comparison.MODELS)
    baseline = cfg.get("baseline_reference")
    if baseline is not None:
        if (source_revision != study.REVISION or not isinstance(baseline, dict)
                or set(baseline) != {"source", "files", "models", "config_sha256", "seal_sha256"}
                or baseline["models"] != list(BASELINES)
                or not isinstance(baseline["files"], dict) or not baseline["files"]
                or baseline["files"].get("input/config.json") != baseline["config_sha256"]
                or baseline["files"].get("compute/result-seal.json") != baseline["seal_sha256"]):
            raise ValueError("Two explicitly bound preserved baselines required")
        # This is a read-only cross-node provenance path, not selected output storage.
        # A source's login-node st_dev policy is not portable to the compute node.
        # Preparation admits the original storage; checked() verifies frozen hashes.
        from console.profile import absolute_path

        absolute_path(baseline["source"])
        expected_models.extend(BASELINES)
    diagnostic = cfg.get("numerical_diagnostic_only", False)
    if type(diagnostic) is not bool:
        raise ValueError("Explicit diagnostic-only boolean required")
    fixed = dict(
        schema=SCHEMA,
        execution_scope=SCOPE,
        models=expected_models,
        seed=0,
        production_training_authorized=False,
        calibration_authorized=not diagnostic,
        evaluation_authorized=not diagnostic,
        final_test_used=False,
        next_round_authorized=False,
        retries=0,
        memory_gib=64,
        cpus=cfg.get("parallel_replay_workers", 2) + 2,
        gpu_jobs_max=1,
        models_sequential=True,
        development_counts=dict(normal_dev_eval=128, anomaly_dev=144),
        normal_counts=dict(normal_validation=104, normal_calibration=104),
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("Post-training-only authorization/configuration mismatch")
    if diagnostic and (
        cfg.get("all_model_calibration_fresh") is not True or "numerical_policy" in cfg
    ):
        raise ValueError("Diagnostic-only requires fresh inputs without an acceptance policy")
    if "numerical_policy" in cfg:
        from .replay_validation import _validate_policy

        _validate_policy(cfg["numerical_policy"])
        if (
            cfg.get("all_model_calibration_fresh") is not True
            or cfg["numerical_policy"]["batch_size"] != cfg.get("prediction_batch_size")
            or cfg["numerical_policy"]["require_same_batch_exact"] is not True
        ):
            raise ValueError("Common numerical policy requires fresh calibration for every model")
    if "sensitivity_policy" in cfg or "sensitivity_plan" in cfg:
        from .sensitivity_metrics import validate_policy

        validate_policy(cfg.get("sensitivity_policy"))
        if (
            diagnostic
            or "numerical_policy" in cfg
            or cfg.get("all_model_calibration_fresh") is not True
            or cfg.get("prediction_batch_size") != 256
            or not isinstance(cfg.get("sensitivity_plan"), dict)
        ):
            raise ValueError("Sensitivity gate requires one uniform fresh full comparison")
    if "parallel_replay_workers" in cfg and (
        type(cfg["parallel_replay_workers"]) is not int or cfg["parallel_replay_workers"] not in {2, 4, 8}
        or cfg.get("all_model_calibration_fresh") is not True
        or cfg.get("prediction_batch_size") != 256 or "sensitivity_policy" not in cfg
        or diagnostic or cfg.get("continuation")
    ):
        raise ValueError("Parallel replay requires2/4/8 CPU workers and uniform fresh sensitivity-gated inputs")
    if "parallel_replay_benchmark" in cfg:
        from .parallel_replay import validate_benchmark_plan

        validate_benchmark_plan(cfg["parallel_replay_benchmark"])
        if cfg.get("parallel_replay_workers") != 8:
            raise ValueError("2/4/8 comparison requires the frozen maximum8-worker budget")
    for key in ("wall_hours", "evaluation_hours", "queue_hours"):
        if type(cfg.get(key)) is not int or not 1 <= cfg[key] <= 168:
            raise ValueError("Explicit finite hours required: " + key)
    required = cfg.get("memory_required_bytes")
    if type(required) is not int or not 1024**3 < required <= 64 * 1024**3:
        raise ValueError("Invalid bounded calibration memory requirement")
    if "prediction_batch_size" in cfg and (
        type(cfg["prediction_batch_size"]) is not int
        or cfg["prediction_batch_size"] not in {1, 256}
        or cfg.get("all_model_replay_fresh") is not True
        or cfg.get("verify_prediction_batch") is not True
    ):
        raise ValueError("Explicit uniform fresh replay configuration required")
    reuse = cfg.get("reuse_calibration", {})
    if not isinstance(reuse, dict) or set(reuse) - set(cfg["models"]):
        raise ValueError("Unexpected normal calibration reuse model")
    if "all_model_calibration_fresh" in cfg and (
        cfg["all_model_calibration_fresh"] is not True
        or cfg.get("prediction_batch_size") != 256
        or cfg.get("all_model_replay_fresh") is not True
        or cfg.get("verify_prediction_batch") is not True
        or cfg.get("reuse_calibration") != {}
        or "continuation" in cfg
    ):
        raise ValueError("Fresh calibration requires all models from scratch without reuse")
    for value in reuse.values():
        if set(value.get("files", {})) != {
            "calibration.json",
            "normal-residuals.json",
            "normal-residuals.npz",
        }:
            raise ValueError("Complete normal-only calibration evidence required")


def profile_check(profile):
    if any(
        profile["slurm"].get(k) != v
        for k, v in dict(
            partition="p02", node="DIS01", nodes=1, ntasks=1, gpus=1, memory_gib=64
        ).items()
    ):
        raise ValueError("Only p02/DIS01, one GPU and64GiB are authorized")
    if profile["slurm"].get("cpus") not in {4, 6, 10}:
        raise ValueError("CPU budget must match2/4/8 replay workers plus2 producer slots")
    if profile["runtime_profile"] != "mamba25":
        raise ValueError("Explicit existing mamba25 environment required")


def training_preparation_proof(source, cfg, authorization):
    """Accept either sealed preparation or the separately signed training-only runner.

    The latter deliberately has no prepared.json. Its worker/controller/profile and
    exact LR-bound sources are verified instead; never manufacture missing evidence.
    """
    if (source / "prepared.json").is_file():
        if authorization.get("execution_scope") != "training_only":
            raise ValueError("Source training authorization scope changed")
        prepared = read(source / "prepared.json")
        if (sha(source / "input/inventory.json") != prepared["input_inventory_sha256"]
                or sha(source / "server-profile.json") != prepared["server_profile_sha256"]):
            raise ValueError("Source preparation binding changed")
        return ["prepared.json"]
    from . import study

    forbidden = ("learning_rate_search_authorized", "calibration_authorized",
                 "evaluation_authorized", "final_test_authorized", "automatic_retry",
                 "next_round_authorized", "llm_monitoring", "llm_auto_resume")
    if (cfg.get("revision") != study.REVISION
            or cfg.get("source") != str(source / "input")
            or cfg.get("main_training_authorized") is not True
            or authorization.get("main_training_authorized") is not True
            or any(authorization.get(k) is not False for k in forbidden)
            or authorization.get("old_lr_selection_sha256") != cfg.get("learning_rate_selection_sha256")
            or any(authorization.get(k) != cfg.get(k) for k in ("seed", "batch_size", "precision"))
            or authorization.get("epochs_max") != cfg.get("epochs")
            or authorization.get("patience") != cfg.get("early_stopping_patience")):
        raise ValueError("Signed training-only preparation binding changed")
    for name, key in (("worker.py", "worker_sha256"), ("control.py", "controller_sha256"),
                      ("server-profile.json", "source_profile_sha256")):
        if sha(child(source, name)) != authorization.get(key):
            raise ValueError("Signed training runner changed: " + name)
    profile_check(read(source / "server-profile.json"))
    source_binding = cfg.get("binding", {}).get("sources", {})
    if not source_binding or read(source / "input/inventory.json").get("config.json") != sha(source / "input/config.json"):
        raise ValueError("Missing signed config/source inventory")
    verify(source / "input", source_binding)
    return ["worker.py", "control.py"]


def source_proof(source):
    """Verify immutable completion and exported training outputs, without loading models."""
    source = external_root(source)
    complete = read(source / "completion.json")
    release = read(source / "release.json")
    submission = read(source / "fit-submission.json")
    state = read(source / "controller.json")
    operator.require_success(release)
    if (
        complete.get("status") != COMPLETE
        or state.get("status") != COMPLETE
        or complete.get("execution_scope") != "training_only"
        or complete.get("fit_release") != release
        or submission.get("uid") != os.getuid()
        or submission.get("phase") != "fit"
        or submission.get("job_id") != release.get("job_id")
        or any(
            complete.get(k) is not False
            for k in ("calibration_started", "evaluation_started", "next_round_authorized")
        )
    ):
        raise ValueError("A successfully released own training-only completion is required")
    cfg = read(source / "input/config.json")
    comparison.config_check(cfg)
    if comparison.execution_scope(cfg) != "training_only":
        raise ValueError("Source must be the independently completed training-only attempt")
    authorization = read(source / "authorization.json")
    if (
        authorization.get("config_sha256") != sha(source / "input/config.json")
        or authorization.get("calibration_authorized") is not False
        or authorization.get("evaluation_authorized") is not False
    ):
        raise ValueError("Source training authorization binding changed")
    preparation_files = training_preparation_proof(source, cfg, authorization)
    verify(source / "input", read(source / "input/inventory.json"))
    seal = read(source / "compute/result-seal.json")
    if (
        sha(source / "compute/result-seal.json") != complete["compute_seal_sha256"]
        or sha(source / "training-report.txt") != complete["report_sha256"]
        or seal.get("config_sha256") != sha(source / "input/config.json")
        or seal.get("status") != "TRAINING_COMPLETE"
        or seal.get("models") != cfg["models"]
        or any(
            seal.get(k) is not False
            for k in (
                "calibration_started",
                "inference_started",
                "evaluation_started",
                "truth_join",
                "final_test_used",
            )
        )
    ):
        raise ValueError("Source training result seal differs from completed training-only scope")
    verify(source / "compute", seal["files"])
    names = [
        "completion.json",
        "release.json",
        "fit-submission.json",
        "controller.json",
        "server-profile.json",
        "authorization.json",
        "input/config.json",
        "input/inventory.json",
        "compute/result-seal.json",
        "training-report.txt",
        *preparation_files,
    ]
    return cfg, seal, {name: sha(source / name) for name in names}


def baseline_source_proof(source_cfg, baseline):
    """Admit only two existing baselines trained on the identical frozen dataset."""
    from . import study

    cfg, seal, files = source_proof(baseline)
    if (source_cfg.get("revision") != study.REVISION
            or cfg.get("revision") != comparison.REVISION
            or any(name not in cfg["models"] for name in BASELINES)
            or any(cfg.get(k) != source_cfg.get(k) for k in
                   ("upstream", "source_dataset_sha256", "dataset_files", "handoff_sha256"))
            or any(cfg.get("binding", {}).get(k) != source_cfg.get("binding", {}).get(k)
                   for k in ("data", "scaler"))):
        raise ValueError("Baseline dataset/train-scaler binding differs from neural training")
    for name in BASELINES:
        scaler = seal["files"].get("models/" + name + "/scaler.npz")
        if not scaler or scaler != source_cfg["binding"]["scaler"]:
            raise ValueError("Baseline bundle train scaler differs")
    return cfg, seal, files


def dataset_binding(upstream, cfg, receipt):
    """Bind only the already admitted normal calibration and development inputs."""
    seal = read(upstream / "evidence/dataset-seal.json")
    if seal != cfg["dataset_files"] or cfg["source_dataset_sha256"] != receipt["dataset"]["sha256"]:
        raise ValueError("Source dataset differs from portable receipt")
    if (
        receipt.get("status") != "PORTABLE_DATASET_VERIFIED"
        or receipt.get("final_test_included") is not False
    ):
        raise ValueError("Only verified development-only portable input is allowed")
    verify(
        upstream,
        {
            k: v
            for k, v in cfg["upstream_files"].items()
            if k.startswith("evidence/") or k == "preprocessing.json"
        },
    )
    if sha(upstream / "dataset/manifest.json") != seal["manifest.json"]:
        raise ValueError("Manifest binding changed")
    rows = read(upstream / "dataset/manifest.json")["runs"]
    if dict(Counter(r["role"] for r in rows)) != comparison.COUNTS:
        raise ValueError("Expected frozen1200-run role membership")
    dev = [r for r in rows if r["role"] in {"normal_dev_eval", "anomaly_dev"}]
    dev_ids = {r["run_id"] for r in dev}
    if len(dev_ids) != 272:
        raise ValueError("Development membership is duplicated")
    expected_truth = {}
    for row in dev:
        value = row["archive"]["members"]["evaluation/events.jsonl"]
        expected_truth[row["run_id"] + ".jsonl"] = {k: value[k] for k in ("bytes", "sha256")}
    if receipt["evaluator_truth"]["members"] != expected_truth:
        raise ValueError("Development truth index differs from sealed data")
    chosen = {}
    for name, digest in seal.items():
        parts = Path(name).parts
        if name in {
            "manifest.json",
            "scaler.npz",
            "normal_validation.npz",
            "normal_calibration.npz",
        } or (len(parts) >= 3 and parts[0] == "canonical" and parts[1] in dev_ids):
            chosen["dataset/" + name] = digest
    # Full input hashes are checked once by the calibration worker before inference.
    return chosen


def calibration_continuation(previous, cfg):
    """Read only stopped prior evidence; every development replay remains fresh."""
    previous, old = checked(previous)
    forbidden = (
        "evaluate-submitted.claim",
        "evaluate-submission-intent.json",
        "evaluate-submission.json",
        "evaluation-submission.json",
        "evaluation",
        "truth",
        "completion.json",
    )
    if any((previous / name).exists() for name in forbidden):
        raise ValueError("Prior evaluation/truth exposure forbids this calibration continuation")
    controller = read(previous / "controller.json")
    record = read(previous / "calibrate-submission.json")
    release = read(previous / "release.json")
    if (
        controller.get("status") != "FAILED_STOPPED"
        or record.get("uid") != os.getuid()
        or record.get("phase") != "calibrate"
        or record.get("job_id") != release.get("job_id")
        or release.get("released") is not True
        or release.get("slurm_state", "").partition(" ")[0] not in operator.TERMINAL - {"COMPLETED"}
    ):
        raise ValueError("A stopped, released own calibration attempt is required")
    actual = operator.snapshot(record)
    if (
        actual.get("released") is not True
        or actual.get("slurm_state") != release.get("slurm_state")
        or actual.get("exit_code") != release.get("exit_code")
        or operator.own_queue()
    ):
        raise ValueError("Actual prior allocation release and empty own queue required")
    for key in (
        "source_training",
        "source_training_config_sha256",
        "source_training_seal_sha256",
        "upstream",
        "source_dataset_sha256",
        "dataset_files",
        "model_files",
        "models",
        "seed",
        "receipt_sha256",
        "truth_archive",
        "evaluation_plan_sha256",
    ):
        if old.get(key) != cfg.get(key):
            raise ValueError("Continuation changes frozen input: " + key)

    def semantic(value):
        return {k: v for k, v in value.items() if k != "implementation_sha256"}

    if semantic(old["protocol"]) != semantic(cfg["protocol"]):
        raise ValueError("Continuation must retain the original detection/evaluation criteria")
    proof_names = (
        "input/config.json",
        "input/inventory.json",
        "prepared.json",
        "server-profile.json",
        "authorization.json",
        "controller.json",
        "calibrate-submission.json",
        "release.json",
    )
    proof_files = {name: sha(previous / name) for name in proof_names}
    reused, files, incomplete = {}, {}, []
    required = ("calibration.json", "normal-residuals.json", "normal-residuals.npz")
    for name in cfg["models"]:
        folder = previous / "compute/calibration" / name
        present = [child(folder, part).is_file() for part in required]
        if not all(present):
            if any(present):
                incomplete.append(name)
            continue
        evidence = read(folder / "normal-residuals.json")
        fitted = read(folder / "calibration.json")
        bundle_sha = sha(previous / "compute/models" / name / "integrity.json")
        if (
            evidence.get("schema") != "drone-ai.normal-residual-evidence/1"
            or evidence.get("model") != name
            or evidence.get("file") != "normal-residuals.npz"
            or evidence.get("roles") != ["normal_validation", "normal_calibration"]
            or evidence.get("truth_used") is not False
            or evidence.get("sha256") != sha(folder / "normal-residuals.npz")
            or evidence.get("calibration_sha256") != sha(folder / "calibration.json")
            or evidence.get("bundle_integrity_sha256") != bundle_sha
            or fitted.get("bundle_integrity_sha256") != bundle_sha
            or evidence.get("dataset_manifest_sha256") != cfg["dataset_files"]["manifest.json"]
        ):
            raise ValueError("Normal calibration evidence binding changed: " + name)
        hashes = {part: sha(folder / part) for part in required}
        reused[name] = dict(files=hashes, source_attempt=str(previous))
        files.update(
            {"calibration/" + name + "/" + part: digest for part, digest in hashes.items()}
        )
    return dict(
        source_attempt=str(previous),
        source_files=proof_files,
        source_calibration_files=files,
        released_job=actual,
        prior_protocol=old["protocol"],
        reuse_calibration=reused,
        incomplete_calibration_preserved=incomplete,
        development_inference_reused=False,
        all_model_replay_fresh=True,
        user_requested_continuation=True,
    )


def copy_normal_calibration(previous, root, continuation):
    """Copy only verified normal evidence, never previous development run directories."""
    for name, digest in continuation["source_calibration_files"].items():
        source, target = child(previous / "compute", name), child(root / "compute", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        if sha(target) != digest:
            raise ValueError("Copied normal calibration changed: " + name)


def read_policy_file(path, validate):
    """Read a regular JSON file without weakening storage or symlink checks."""
    path = Path(path).expanduser().absolute()
    if ".." in path.parts:
        raise ValueError("Policy path traversal forbidden")
    path = child(external_root(path.parent), path.name)
    if not path.is_file():
        raise ValueError("Regular policy JSON file required: " + str(path))
    policy = read(path)
    validate(policy)
    return policy


def prepare(args):
    """Server-side preparation only; explicit approval does not submit any job."""
    from console.profile import load

    if not args.approve_post_training:
        raise ValueError("Explicit --approve-post-training required")
    fresh_calibration = getattr(args, "fresh_calibration", False)
    diagnostic = getattr(args, "diagnostic_only", False)
    policy_path = getattr(args, "numerical_policy", None)
    sensitivity_path = getattr(args, "sensitivity_policy", None)
    if sensitivity_path is not None and (
        not fresh_calibration or diagnostic or policy_path is not None
    ):
        raise ValueError(
            "--sensitivity-policy requires fresh calibration and excludes other numerical modes"
        )
    if (diagnostic or policy_path is not None) and not fresh_calibration:
        raise ValueError("Diagnostic/common policy requires --fresh-calibration")
    if diagnostic and policy_path is not None:
        raise ValueError("Diagnostic-only records evidence without selecting a policy")
    if fresh_calibration and getattr(args, "continue_from", None) is not None:
        raise ValueError("--fresh-calibration and --continue-from are mutually exclusive")
    # Validate file inputs before copying any model or creating an attempt directory.
    if policy_path is not None:
        from .replay_validation import _validate_policy

        numerical_policy = read_policy_file(policy_path, _validate_policy)
    if sensitivity_path is not None:
        from .sensitivity_metrics import validate_policy

        sensitivity_policy = read_policy_file(sensitivity_path, validate_policy)
    source, root = external_root(args.source), remote_path(external_root(args.root))
    bundle = external_root(args.bundle)
    previous = getattr(args, "continue_from", None)
    previous = external_root(previous) if previous is not None else None
    if root.exists():
        raise FileExistsError("New output required; existing evidence is never overwritten")
    profile = load(args.profile)
    profile_check(profile)
    source_cfg, source_seal, source_files = source_proof(source)
    baseline_source = getattr(args, "baseline_source", None)
    baseline_source = external_root(baseline_source) if baseline_source is not None else None
    model_sources = [(source, source_cfg["models"], source_seal)]
    model_names = list(source_cfg["models"])
    baseline_reference = None
    if baseline_source is not None:
        _, baseline_seal, baseline_files = baseline_source_proof(source_cfg, baseline_source)
        model_sources.append((baseline_source, BASELINES, baseline_seal))
        model_names.extend(BASELINES)
        baseline_reference = dict(source=str(baseline_source), files=baseline_files,
                                  models=list(BASELINES),
                                  config_sha256=baseline_files["input/config.json"],
                                  seal_sha256=baseline_files["compute/result-seal.json"])
    upstream = external_root(source_cfg["upstream"])
    protected_inputs = ((source, upstream, bundle) + ((previous,) if previous else ())
                        + ((baseline_source,) if baseline_source else ()))
    for protected in protected_inputs:
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError("Post-training output must be separate from all preserved inputs")
    if Path(profile["dataset_root"]) != upstream or Path(profile["output_root"]) != root:
        raise ValueError("Profile must bind the preserved upstream and new output root")
    receipt = read(bundle / "receipt.json")
    needed_inputs = dataset_binding(upstream, source_cfg, receipt)
    from execution.prepare import npz_headers

    raw_bytes = sum(
        value["bytes"]
        for role in ("normal_validation", "normal_calibration")
        for value in npz_headers(upstream / "dataset" / (role + ".npz")).values()
    )
    # Historical normalization uses float64. Bound raw arrays, normalized copies,
    # residuals/BLAS/replay, then retain the runtime1GiB reserve inside the request.
    required_bytes = max(16 * 1024**3, 4 * raw_bytes + 8 * 1024**3)
    if required_bytes > 64 * 1024**3:
        raise MemoryError("Calibration arrays plus reserve exceed fixed64GiB allocation")
    truth_ref = receipt["evaluator_truth"]
    truth_archive = child(bundle, truth_ref["path"])
    if (
        truth_archive.stat().st_size != truth_ref["bytes"]
        or sha(truth_archive) != truth_ref["sha256"]
    ):
        raise ValueError("Evaluator truth archive changed")
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 24 * 1024**3:
        raise OSError("24GiB output reserve required")
    root.mkdir()
    progress = Progress(root / "prepare-progress.json", hours=2)
    try:
        progress.update(status="RUNNING", stage="freeze_post_training_inputs")
        write(root / "server-profile.json", profile)
        inputs = root / "input"
        for folder in ("execution", "learning", "pipeline", "src/capstone_ai", "console"):
            for path in (ROOT / "ai" / folder).rglob("*.py"):
                target = inputs / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        for name in ("gpu_probe.py", "environment-mamba25.json"):
            target = inputs / "ai/remote" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "ai/remote" / name, target)
        shutil.copy2(ROOT / "ai/x500_storage.py", inputs / "ai/x500_storage.py")
        protocol_file = ROOT / "ai/configs/development-evaluation.json"
        shutil.copy2(protocol_file, inputs / "development-evaluation.json")
        models = {}
        for model_source, names, model_seal in model_sources:
            for name in names:
                prefix = "models/" + name + "/"
                expected = {
                    k[len(prefix) :]: v for k, v in model_seal["files"].items() if k.startswith(prefix)
                }
                if not expected or "integrity.json" not in expected:
                    raise ValueError("Missing sealed exported bundle: " + name)
                # Keep bundles intact; integrity.json may bind best/last checkpoints.
                for filename, digest in expected.items():
                    target = child(root / "compute/models" / name, filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child(model_source / "compute/models" / name, filename), target)
                    if sha(target) != digest:
                        raise ValueError("Copied model bundle changed")
                    models["models/" + name + "/" + filename] = digest
        from execution.evaluate import default_protocol_manifest

        cfg = dict(
            schema=SCHEMA,
            revision=source_cfg["revision"],
            source_comparison_revision=source_cfg["revision"],
            execution_scope=SCOPE,
            models=model_names,
            seed=0,
            production_training_authorized=False,
            calibration_authorized=True,
            evaluation_authorized=True,
            final_test_used=False,
            next_round_authorized=False,
            retries=0,
            memory_gib=64,
            cpus=profile["slurm"]["cpus"],
            gpu_jobs_max=1,
            models_sequential=True,
            development_counts=dict(normal_dev_eval=128, anomaly_dev=144),
            normal_counts=dict(normal_validation=104, normal_calibration=104),
            wall_hours=args.calibration_hours,
            evaluation_hours=args.evaluation_hours,
            queue_hours=args.queue_hours,
            memory_required_bytes=required_bytes,
            memory=dict(
                raw_normal_validation_calibration_bytes=raw_bytes,
                required_bytes_including_reserve=required_bytes,
                requested_bytes=64 * 1024**3,
                estimate="four times raw normal arrays plus8GiB; minimum16GiB",
            ),
            source_training=str(source),
            source_training_files=source_files,
            source_training_config_sha256=source_files["input/config.json"],
            source_training_seal_sha256=source_files["compute/result-seal.json"],
            upstream=str(upstream),
            evaluation_input_files=needed_inputs,
            dataset_files=source_cfg["dataset_files"],
            handoff_sha256=source_cfg["handoff_sha256"],
            source_dataset_sha256=source_cfg["source_dataset_sha256"],
            model_files=models,
            source_bundle=str(bundle),
            receipt_sha256=sha(bundle / "receipt.json"),
            truth_archive=truth_ref,
            protocol=default_protocol_manifest(),
            evaluation_plan_sha256=sha(inputs / "development-evaluation.json"),
            failure_policy="Preserve evidence and stop; no retry, refit or automatic tuning",
        )
        if baseline_reference is not None:
            cfg["baseline_reference"] = baseline_reference
        if previous is not None:
            cfg.update(
                prediction_batch_size=256, verify_prediction_batch=True, all_model_replay_fresh=True
            )
            continuation = calibration_continuation(previous, cfg)
            copy_normal_calibration(previous, root, continuation)
            cfg.update(
                continuation=continuation, reuse_calibration=continuation["reuse_calibration"]
            )
            write(inputs / "continuation.json", continuation)
        if fresh_calibration:
            cfg.update(
                prediction_batch_size=256,
                verify_prediction_batch=True,
                all_model_replay_fresh=True,
                all_model_calibration_fresh=True,
                reuse_calibration={},
            )
            require_fresh_calibration_inputs(root, cfg)
        if diagnostic:
            cfg.update(
                numerical_diagnostic_only=True,
                calibration_authorized=False,
                evaluation_authorized=False,
            )
        if policy_path is not None:
            cfg["numerical_policy"] = numerical_policy
            write(inputs / "numerical-policy.json", cfg["numerical_policy"])
            cfg["numerical_policy_sha256"] = sha(inputs / "numerical-policy.json")
        if sensitivity_path is not None:
            from .sensitivity_data import build_plan, verify_selection_layout

            cfg["sensitivity_policy"] = sensitivity_policy
            cfg["sensitivity_plan"] = build_plan(upstream / "dataset/manifest.json")
            write(
                inputs / "sensitivity-layout.json",
                verify_selection_layout(cfg["sensitivity_plan"], upstream / "dataset"),
            )
            write(inputs / "sensitivity-policy.json", cfg["sensitivity_policy"])
            write(inputs / "sensitivity-plan.json", cfg["sensitivity_plan"])
            cfg["sensitivity_policy_sha256"] = sha(inputs / "sensitivity-policy.json")
            cfg["sensitivity_plan_sha256"] = sha(inputs / "sensitivity-plan.json")
        if getattr(args, "parallel_replay_workers", None) is not None:
            cfg["parallel_replay_workers"] = args.parallel_replay_workers
        if getattr(args, "benchmark_parallel", False):
            from .parallel_replay import build_benchmark_plan

            plan, benchmark_inputs = build_benchmark_plan(upstream, source_cfg)
            cfg["parallel_replay_benchmark"] = plan
            cfg["evaluation_input_files"].update(benchmark_inputs)
        config_check(cfg)
        write(inputs / "config.json", cfg)
        write(inputs / "protocol.json", cfg["protocol"])
        write(inputs / "inventory.json", inventory(inputs))
        write(
            root / "authorization.json",
            dict(
                approved=time.time(),
                execution_scope=SCOPE,
                config_sha256=sha(inputs / "config.json"),
                models=model_names,
                training_authorized=False,
                calibration_authorized=not diagnostic,
                evaluation_authorized=not diagnostic,
                numerical_diagnostic_only=diagnostic,
                final_test_authorized=False,
                next_round_authorized=False,
            ),
        )
        write(
            root / "prepared.json",
            dict(
                status="AUTHORIZED_NOT_STARTED",
                execution_scope=SCOPE,
                source_training=str(source),
                input_inventory_sha256=sha(inputs / "inventory.json"),
                server_profile_sha256=sha(root / "server-profile.json"),
                authorization_sha256=sha(root / "authorization.json"),
                source_data_modified=False,
                training_repeated=False,
                llm_calls=0,
            ),
        )
        progress.update(status="AUTHORIZED_NOT_STARTED", stage="stopped")
    except BaseException as exc:
        progress.update(status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}")
        raise


def checked(root):
    root = external_root(root)
    cfg = read(root / "input/config.json")
    config_check(cfg)
    prepared = read(root / "prepared.json")
    for filename, key in (
        ("input/inventory.json", "input_inventory_sha256"),
        ("server-profile.json", "server_profile_sha256"),
        ("authorization.json", "authorization_sha256"),
    ):
        if sha(root / filename) != prepared[key]:
            raise ValueError("Post-training preparation changed: " + filename)
    verify(root / "input", read(root / "input/inventory.json"))
    authorization = read(root / "authorization.json")
    if (
        authorization.get("config_sha256") != sha(root / "input/config.json")
        or authorization.get("execution_scope") != SCOPE
    ):
        raise ValueError("Post-training authorization binding changed")
    profile_check(runtime_profile(root, required=True))
    if runtime_profile(root, required=True).get("slurm", {}).get("cpus", 4) != cfg["cpus"]:
        raise ValueError("Frozen CPU allocation/profile differs")
    verify(Path(cfg["source_training"]), cfg["source_training_files"])
    if cfg.get("baseline_reference"):
        baseline = cfg["baseline_reference"]
        verify(Path(baseline["source"]), baseline["files"])
    verify(root / "compute", cfg["model_files"])
    if "numerical_policy" in cfg and (
        sha(root / "input/numerical-policy.json") != cfg["numerical_policy_sha256"]
        or read(root / "input/numerical-policy.json") != cfg["numerical_policy"]
    ):
        raise ValueError("Frozen common numerical policy changed")
    if "sensitivity_policy" in cfg:
        for key in ("sensitivity_policy", "sensitivity_plan"):
            path = root / "input" / (key.replace("_", "-") + ".json")
            if sha(path) != cfg[key + "_sha256"] or read(path) != cfg[key]:
                raise ValueError("Frozen common sensitivity inputs changed")
    if cfg.get("continuation"):
        continuation = cfg["continuation"]
        verify(Path(continuation["source_attempt"]), continuation["source_files"])
        verify(root / "compute", continuation["source_calibration_files"])
    return root, cfg


def start(root):
    root, cfg = checked(root)
    approved_python = python_for(root, required=True)
    if Path(sys.executable).resolve() != Path(approved_python).resolve() or os.environ.get(
        "SLURM_JOB_ID"
    ):
        raise ValueError("Start with the selected server Python on the login node")
    if (root / "controller.claim").exists():
        raise ValueError("Attempt already claimed; no automatic retry")
    with (root / "start.claim").open("x") as stream:
        stream.write("Single controller launch; never retry an unknown launch outcome.\n")
    name = "drone-ai-" + root.name
    command = (
        "export PYTHONPATH="
        + shlex.quote(str(root / "input/ai") + ":" + str(root / "input/ai/src"))
        + "; export DRONE_AI_PROFILE="
        + shlex.quote(str(root / "server-profile.json"))
        + "; export PYTHONDONTWRITEBYTECODE=1; exec "
        + shlex.join(
            [approved_python, "-B", "-m", "learning.post_training", "control", "--root", str(root)]
        )
        + " >> "
        + shlex.quote(str(root / "controller.log"))
        + " 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", name, command], check=True)
    print("Finite post-training controller started in server tmux: " + name)


def submit(root, phase, cfg):
    root, frozen = checked(root)
    if cfg != frozen:
        raise ValueError("Submission configuration differs from frozen authorization")
    permitted = {"diagnose"} if cfg.get("numerical_diagnostic_only") else {"calibrate", "evaluate"}
    if phase not in permitted:
        raise ValueError("Only calibration and evaluation are authorized")
    with SUBMIT_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if operator.own_queue():
            raise ValueError("Own pending/running job exists; no concurrent GPU submission")
        if phase == "evaluate":
            operator.require_success(read(root / "release.json"))
            seal_check(root, cfg)
        with (root / (phase + "-submitted.claim")).open("x") as stream:
            stream.write("One submission; unknown outcome must never be retried.\n")
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
                    "learning.post_training",
                    phase,
                    "--root",
                    str(root),
                ]
            )
            + "\n"
        )
        hours = cfg["wall_hours"] if phase != "evaluate" else cfg["evaluation_hours"]
        argv = [
            "sbatch",
            "--parsable",
            "--partition=p02",
            "--nodelist=DIS01",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            f"--cpus-per-task={cfg['cpus']}",
            "--mem=64G",
            f"--time={hours}:00:00",
            "--no-requeue",
            "--job-name=drone-ai-" + phase,
            "--output=" + str(root / (phase + "-slurm.log")),
            "--error=" + str(root / (phase + "-slurm.log")),
            "--chdir=" + str(root),
            str(script),
        ]
        write(
            root / (phase + "-submission-intent.json"),
            dict(argv=argv, time=time.time(), automatic_retry=False),
        )
        response = operator.command(argv)
        identifier = response.split(";")[0]
        if not identifier.isdigit():
            raise ValueError("Unknown submission outcome; no retry: " + response)
        record = dict(job_id=identifier, uid=os.getuid(), phase=phase, submitted=time.time())
        write(root / (phase + "-submission.json"), record)
        return record


def seal_check(root, cfg):
    seal = read(root / "compute/result-seal.json")
    if (
        seal.get("status") != "TRAIN_CALIBRATION_INFERENCE_COMPLETE"
        or seal.get("models") != cfg["models"]
        or seal.get("seed") != cfg["seed"]
        or seal.get("config_sha256") != sha(root / "input/config.json")
        or seal.get("training_repeated") is not False
        or seal.get("truth_join") is not False
        or seal.get("final_test_used") is not False
        or seal.get("source_training_seal_sha256") != cfg["source_training_seal_sha256"]
    ):
        raise ValueError("Complete calibration/inference seal required before truth join")
    return seal


def diagnostic_seal_check(root, cfg):
    seal = read(root / "compute/diagnostic-seal.json")
    if (
        seal.get("status") != "DIAGNOSTIC_COMPLETE_PENDING_RELEASE"
        or seal.get("models") != cfg["models"]
        or seal.get("config_sha256") != sha(root / "input/config.json")
        or any(
            seal.get(key) is not False
            for key in (
                "training_repeated",
                "calibration_started",
                "inference_started",
                "truth_join",
            )
        )
    ):
        raise ValueError("Complete normal-only diagnostic seal required")
    verify(root / "compute", seal["files"])
    return seal


def wait_files(root, phase, state, cfg):
    """Bounded NFS visibility wait only; never recompute or resubmit."""
    started = time.monotonic()
    deadline = started + operator.FILE_VISIBILITY_SECONDS
    state.update(
        file_visibility="CHECKING",
        file_visibility_phase=phase,
        file_visibility_waited_s=0,
        file_visibility_limit_s=operator.FILE_VISIBILITY_SECONDS,
    )
    write(root / "controller.json", state)
    while True:
        try:
            if phase == "diagnose":
                diagnostic_seal_check(root, cfg)
            elif phase == "calibrate":
                seal = seal_check(root, cfg)
                for name in seal["files"]:
                    with child(root / "compute", name).open("rb"):
                        pass
            else:
                result = read(root / "evaluation/report.json")
                if result["status"] != "EVALUATION_COMPLETE_PENDING_RELEASE":
                    raise ValueError("Unexpected evaluator completion status")
                read(root / "evaluation/metrics.json")
            state.update(
                file_visibility="VISIBLE", file_visibility_waited_s=time.monotonic() - started
            )
            write(root / "controller.json", state)
            return
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise RuntimeError("Released output not visible within120seconds; no retry")
            state.update(
                file_visibility="WAITING",
                file_visibility_waited_s=time.monotonic() - started,
                updated=time.time(),
            )
            write(root / "controller.json", state)
            time.sleep(operator.FILE_VISIBILITY_INTERVAL_SECONDS)


def control(root):
    root, cfg = checked(root)
    if os.environ.get("SLURM_JOB_ID") or not os.environ.get("TMUX"):
        raise ValueError("Finite controller must run in server login-node tmux")
    if Path(sys.executable).resolve() != Path(python_for(root, required=True)).resolve():
        raise ValueError("Selected server Python required")
    with (root / "controller.claim").open("x") as stream:
        stream.write("One approved post-training continuation, no automatic retry.\n")
    state = dict(
        status="RUNNING",
        stage="submission",
        round=cfg["revision"],
        execution_scope=SCOPE,
        pid=os.getpid(),
        started=time.time(),
        llm_calls=0,
        retries=0,
        allocation_released=False,
        next_round_authorized=False,
    )
    record = None
    phase = None
    try:
        phases = (
            ("diagnose",) if cfg.get("numerical_diagnostic_only") else ("calibrate", "evaluate")
        )
        for phase in phases:
            # Never let a preceding released job stand in for a new, unknown
            # sbatch outcome. An unacknowledged intent is not proof of no job.
            record = None
            state.update(stage=phase, allocation_released=False)
            write(root / "controller.json", state)
            record = submit(root, phase, cfg)
            state.update(stage=phase, job_id=record["job_id"], allocation_released=False)
            write(root / "controller.json", state)
            budget = {
                **cfg,
                "wall_hours": cfg["wall_hours"] if phase != "evaluate" else cfg["evaluation_hours"],
            }
            release = operator.wait_release(root, record, state, budget)
            write(
                root / ("evaluation-release.json" if phase == "evaluate" else "release.json"),
                release,
            )
            operator.require_success(release)
            state.update(allocation_released=True)
            wait_files(root, phase, state, cfg)
        if cfg.get("numerical_diagnostic_only"):
            diagnostic_seal_check(root, cfg)
            checked(root)
            if operator.own_queue():
                raise ValueError("Own queue remains nonempty at diagnostic completion")
            write(
                root / "completion.json",
                dict(
                    status="DIAGNOSTIC_READY_STOPPED",
                    finished=time.time(),
                    diagnostic_release=read(root / "release.json"),
                    diagnostic_seal_sha256=sha(root / "compute/diagnostic-seal.json"),
                    training_repeated=False,
                    calibration_started=False,
                    inference_started=False,
                    final_test_used=False,
                    next_round_authorized=False,
                    automatic_retry=False,
                    llm_calls=0,
                ),
            )
            state.update(status="DIAGNOSTIC_READY_STOPPED", stage="stopped", finished=time.time())
            write(root / "controller.json", state)
            return
        verify(root / "compute", seal_check(root, cfg)["files"])
        result = read(root / "evaluation/report.json")
        if sha(root / "evaluation/metrics.json") != result["metrics_sha256"]:
            raise ValueError("Evaluation metrics hash mismatch")
        if sha(root / "evaluation/evidence.json") != result["evidence_sha256"]:
            raise ValueError("Evaluation evidence index hash mismatch")
        checked(root)
        if operator.own_queue():
            raise ValueError("Own queue remains nonempty at final completion")
        operator.report(root, cfg)
        write(
            root / "completion.json",
            dict(
                status="REPORT_READY_STOPPED",
                finished=time.time(),
                execution_scope=SCOPE,
                calibration_release=read(root / "release.json"),
                evaluation_release=read(root / "evaluation-release.json"),
                source_training=str(cfg["source_training"]),
                training_repeated=False,
                compute_seal_sha256=sha(root / "compute/result-seal.json"),
                report_sha256=sha(root / "report.txt"),
                metrics_sha256=result["metrics_sha256"],
                evaluation_report_sha256=sha(root / "evaluation/report.json"),
                final_test_used=False,
                next_round_authorized=False,
                automatic_retry=False,
                llm_calls=0,
            ),
        )
        state.update(
            status="REPORT_READY_STOPPED",
            stage="stopped",
            finished=time.time(),
            report=str(root / "report.txt"),
        )
        write(root / "controller.json", state)
    except BaseException as exc:
        released = False
        if record is not None:
            try:
                released = operator.snapshot(record).get("released", False)
            except Exception:
                pass
        try:
            uncertain = bool(operator.own_queue())
        except Exception:
            uncertain = True
        unacknowledged = bool(
            phase
            and (root / (phase + "-submission-intent.json")).exists()
            and not (root / (phase + "-submission.json")).exists()
        )
        uncertain = uncertain or unacknowledged
        state.update(
            status="FAILED_STOPPED"
            if (record is None or released) and not uncertain
            else "CONTROLLER_STOPPED_JOB_STATE_UNCONFIRMED",
            reason=f"{type(exc).__name__}: {exc}",
            automatic_retry=False,
            submission_outcome_unconfirmed=unacknowledged,
            allocation_released=released,
            updated=time.time(),
        )
        write(root / "controller.json", state)
        raise


def require_fresh_calibration_inputs(root, cfg):
    """Fresh attempts may contain model exports, never prior calibration evidence."""
    if cfg.get("all_model_calibration_fresh") is not True:
        return
    artifacts = (root / "input/continuation.json", root / "compute/calibration")
    if (
        cfg.get("reuse_calibration") != {}
        or "continuation" in cfg
        or any(path.exists() or path.is_symlink() for path in artifacts)
    ):
        raise ValueError("Fresh calibration must start without calibration or reuse artifacts")


def calibrate_worker(root, *, diagnostic_only=False):
    from execution.engine import calibrate, preflight_models
    from pipeline.runtime import check

    from .diagnostic import stop_signal
    from .telemetry import Telemetry
    from .training import backend, project_lock

    root, cfg = checked(root)
    if diagnostic_only != cfg.get("numerical_diagnostic_only", False):
        raise ValueError("Worker phase differs from the frozen authorization")
    phase = "diagnose" if diagnostic_only else "calibrate"
    out, upstream = root / "compute", Path(cfg["upstream"])
    with (out / (phase + "-started.claim")).open("x") as stream:
        stream.write("No automatic calibration retry.\n")
    progress = Progress(out / "progress.json", hours=cfg["wall_hours"] - 0.1)
    telemetry = None
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        require_fresh_calibration_inputs(root, cfg)
        progress.update(stage="resource_admission", round=cfg["revision"])
        environment = check(root / "input/ai", out, phase, job_prefix="drone-ai-")
        if cfg.get("parallel_replay_workers") and int(environment["job"]["NumCPUs"]) < cfg["cpus"]:
            raise RuntimeError("Allocated CPU count differs from parallel replay budget")
        snapshot = memory_snapshot()
        required = cfg["memory_required_bytes"]
        admitted = not snapshot.get("errors") and snapshot["available_bytes"] >= required
        write(
            out / "host-memory-admission.json",
            dict(
                available_bytes=snapshot["available_bytes"],
                required_bytes=required,
                passed=admitted,
                snapshot=snapshot,
                environment=environment,
            ),
        )
        if not admitted:
            raise MemoryError("Calibration memory admission failed; see host-memory-admission.json")
        progress.update(stage="verify_calibration_development_inputs")
        verify(
            upstream,
            {
                key: value
                for key, value in cfg["evaluation_input_files"].items()
                if not diagnostic_only
                or key
                in {"dataset/normal_validation.npz", "dataset/scaler.npz", "dataset/manifest.json"}
            },
        )
        telemetry = Telemetry(out)
        telemetry.start()
        development_context = nullcontext(None)
        if cfg.get("parallel_replay_workers"):
            from .parallel_replay import ParallelDevelopment

            selected_workers = cfg["parallel_replay_workers"]
            if cfg.get("parallel_replay_benchmark"):
                from .parallel_replay import benchmark_workers

                with project_lock("cuda"), backend("cuda"):
                    selection = benchmark_workers(upstream, out, cfg, progress)
                selected_workers = selection["selected_workers"]
            development_context = ParallelDevelopment(selected_workers, progress)
        with project_lock("cuda"), development_context as development_executor:
            if diagnostic_only:
                preflight_models(
                    upstream,
                    out,
                    progress,
                    predict_device="cuda",
                    backend_factory=backend,
                    model_names=cfg["models"],
                    prediction_batch_size=cfg["prediction_batch_size"],
                    diagnostic_only=True,
                )
            else:
                calibrate(
                    upstream,
                    out,
                    progress,
                    predict_device="cuda",
                    backend_factory=backend,
                    model_names=cfg["models"],
                    preserve_normal_residuals=True,
                    reuse_calibration=cfg.get("reuse_calibration", {}),
                    prediction_batch_size=cfg.get("prediction_batch_size", 1),
                    verify_prediction_batch=cfg.get("verify_prediction_batch", False),
                    numerical_policy=cfg.get("numerical_policy"),
                    sensitivity_plan=cfg.get("sensitivity_plan"),
                    sensitivity_policy=cfg.get("sensitivity_policy"),
                    development_executor=development_executor,
                )
        telemetry.stop()
        telemetry = None
        checked(root)
        files = {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file() and p.name not in {"progress.json", "result-seal.json"}
        }
        if diagnostic_only:
            write(
                out / "diagnostic-seal.json",
                dict(
                    status="DIAGNOSTIC_COMPLETE_PENDING_RELEASE",
                    files=files,
                    models=cfg["models"],
                    config_sha256=sha(root / "input/config.json"),
                    training_repeated=False,
                    calibration_started=False,
                    inference_started=False,
                    truth_join=False,
                ),
            )
            progress.update(status="DIAGNOSTIC_COMPLETE_PENDING_RELEASE", stage="stopped")
            return
        write(
            out / "result-seal.json",
            dict(
                files=files,
                status="TRAIN_CALIBRATION_INFERENCE_COMPLETE",
                models=cfg["models"],
                seed=cfg["seed"],
                config_sha256=sha(root / "input/config.json"),
                execution_scope=SCOPE,
                source_training_seal_sha256=cfg["source_training_seal_sha256"],
                training_repeated=False,
                calibration_started=True,
                inference_started=True,
                evaluation_started=False,
                truth_join=False,
                final_test_used=False,
                next_round_authorized=False,
                prediction_batch_size=cfg.get("prediction_batch_size", 1),
                all_model_replay_fresh=cfg.get("all_model_replay_fresh", True),
                all_model_calibration_fresh=cfg.get("all_model_calibration_fresh", False),
                normal_calibration_reused=list(cfg.get("reuse_calibration", {})),
                prior_inference_reused=False,
                all_model_sensitivity_passed=bool(cfg.get("sensitivity_policy")),
                diagnostic_calibration_reused=False,
            ),
        )
        progress.update(status="CALIBRATION_INFERENCE_COMPLETE_PENDING_RELEASE", stage="stopped")
    except BaseException as exc:
        progress.update(
            status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}", automatic_retry=False
        )
        raise
    finally:
        if telemetry is not None:
            telemetry.stop()


def extract_truth(root, cfg, progress):
    """Evaluator-only extraction after successful inference release and seal."""
    operator.require_success(read(root / "release.json"))
    seal_check(root, cfg)
    bundle, ref = Path(cfg["source_bundle"]), cfg["truth_archive"]
    if sha(bundle / "receipt.json") != cfg["receipt_sha256"]:
        raise ValueError("Evaluator receipt changed")
    archive = child(bundle, ref["path"])
    if archive.stat().st_size != ref["bytes"] or sha(archive) != ref["sha256"]:
        raise ValueError("Evaluator archive changed")
    target = root / "truth"
    target.mkdir(exist_ok=False)
    seen = set()
    with tarfile.open(archive, "r|*") as stream:
        for member in stream:
            progress.check()
            if (
                member.name not in ref["members"]
                or member.name in seen
                or not member.isfile()
                or member.sparse
            ):
                raise ValueError("Unexpected, duplicate or nonregular truth member")
            expected = ref["members"][member.name]
            if member.size != expected["bytes"] or not 0 <= member.size <= 10_000_000:
                raise ValueError("Truth member size mismatch")
            path = child(target, member.name)
            with stream.extractfile(member) as source, path.open("xb") as dest:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
            if sha(path) != expected["sha256"]:
                raise ValueError("Truth member hash mismatch")
            seen.add(member.name)
    if seen != set(ref["members"]):
        raise ValueError("Missing evaluator truth members")
    write(target / "index.json", ref["members"])


def evaluate_worker(root):
    from execution.evaluate import evaluate
    from pipeline.runtime import check

    from .diagnostic import stop_signal
    from .training import project_lock

    root, cfg = checked(root)
    if cfg.get("numerical_diagnostic_only"):
        raise ValueError("Diagnostic-only attempt cannot evaluate")
    progress = Progress(root / "evaluation-progress.json", hours=cfg["evaluation_hours"] - 0.1)
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        operator.require_success(read(root / "release.json"))
        seal = seal_check(root, cfg)
        verify(root / "compute", seal["files"])
        runtime_out = root / "evaluation-runtime"
        runtime_out.mkdir(exist_ok=False)
        check(root / "input/ai", runtime_out, "evaluate", job_prefix="drone-ai-")
        extract_truth(root, cfg, progress)

        def truth_provider(row, target, budget):
            budget.check()
            if row["role"] not in cfg["development_counts"]:
                raise ValueError("Development truth only")
            expected = row["archive"]["members"]["evaluation/events.jsonl"]
            source = child(root / "truth", row["run_id"] + ".jsonl")
            if source.stat().st_size != expected["bytes"] or sha(source) != expected["sha256"]:
                raise ValueError("Truth binding changed")
            shutil.copyfile(source, target)

        with project_lock("cuda"):
            evaluate(
                {**cfg, "dataset": str(Path(cfg["upstream"]) / "dataset")},
                root / "compute",
                root / "evaluation",
                progress,
                truth_provider=truth_provider,
                model_names=cfg["models"],
                development_counts=cfg["development_counts"],
                completion_status="EVALUATION_COMPLETE_PENDING_RELEASE",
            )
        progress.update(status="EVALUATION_COMPLETE_PENDING_RELEASE", stage="stopped")
    except BaseException as exc:
        progress.update(
            status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}", automatic_retry=False
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--baseline-source", type=Path,
                   help="Preserved training-only comparison supplying persistence/ridge; no refit")
    p.add_argument("--parallel-replay-workers", type=int, choices=[2, 4, 8],
                   help="Bounded GPU producer plus2/4/8 CPU replay workers; batch256 unchanged")
    p.add_argument("--benchmark-parallel", action="store_true",
                   help="Use actual GPU/normal data to select one uniform2/4/8-worker setting")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--calibration-hours", type=int, default=24)
    p.add_argument("--evaluation-hours", type=int, default=12)
    p.add_argument("--queue-hours", type=int, default=24)
    p.add_argument("--approve-post-training", action="store_true")
    p.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="All-model normal-only batch diagnostics, release and stop; no calibration",
    )
    p.add_argument(
        "--numerical-policy",
        type=Path,
        help="One frozen common numerical criterion for all models before calibration",
    )
    p.add_argument(
        "--sensitivity-policy",
        type=Path,
        help="Frozen all-model contiguous normal numerical/alarm gate before fresh calibration",
    )
    p.add_argument(
        "--fresh-calibration",
        action="store_true",
        help="Recompute normal calibration and replay every selected model at batch256; no reuse",
    )
    p.add_argument(
        "--continue-from",
        type=Path,
        help="Stopped prior evaluation; reuse normal calibration only, replay all models fresh",
    )
    for action in ("start", "control", "calibrate", "evaluate", "diagnose"):
        sub.add_parser(action).add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        {
            "start": start,
            "control": control,
            "calibrate": calibrate_worker,
            "evaluate": evaluate_worker,
            "diagnose": lambda root: calibrate_worker(root, diagnostic_only=True),
        }[args.action](args.root)


if __name__ == "__main__":
    main()
