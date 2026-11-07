"""Finite explicit SGD suite; normal-only LR probes and unchanged resource/resume gates."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from execution.common import ROOT, Progress, read, sha, write

from learning.diagnostic import same
from learning.optimization import CANDIDATES, POLICY, policy_for
from learning.study import PROBE

from .benchmark import measure
from .data import load
from .models import ModelSpec, make_model
from .predict import Predictor, seal_export
from .training import backend, run_fit

CONFIG = ROOT / "ai/configs/history19-sgd.json"


def config():
    cfg = read(CONFIG)
    required = dict(
        revision="history19_sgd_v1",
        feature_profile="common19_ctx0",
        context_features=0,
        input_shape=[100, 19],
        target_shape=[10, 4],
        step_us=100000,
        cadence_us=1000000,
    )
    if any(
        type(cfg.get(k)) is not type(v) or cfg[k] != v
        for k, v in required.items()
    ):
        raise ValueError("Fixed history19 input/target contract required")
    if (
        cfg["optimization"] != POLICY
        or cfg["learning_rate_probe"] != PROBE
        or cfg["learning_rate_candidates"] != {"sgd": CANDIDATES["sgd"]}
        or len(cfg["variants"]) != 8
        or any(e["optimizer"] != "sgd" for e in cfg["variants"])
    ):
        raise ValueError("Preserved SGD/norm/probe policy required")
    previous = read(ROOT / "ai/configs/normalization-study.json")
    for field in ("normalization", "initialization", "main_fit", "gpu"):
        if cfg[field] != previous[field]:
            raise ValueError(
                "Other implementation conditions must remain fixed: " + field
            )
    expected = [
        dict(
            id=f"{name}_h{hidden}_l{layers}_sgd_ctx0_s0",
            name=name,
            hidden=hidden,
            layers=layers,
            heads=4,
            optimizer="sgd",
        )
        for hidden, layers in ((64, 2), (64, 4), (128, 2), (128, 4))
        for name in ("lstm", "transformer")
    ]
    if cfg["variants"] != expected:
        raise ValueError(
            "Fixed history19 SGD variants required: h64/h128, layers2/4"
        )
    return cfg


def spec(entry):
    return ModelSpec(
        entry["name"], entry["hidden"], entry["layers"], entry["heads"]
    )


def settings(cfg, **overrides):
    values = cfg["main_fit"]
    return dict(
        seed=values["seed"],
        batch_size=values["batch_size"],
        epochs=values["epochs_max"],
        patience=values["early_stopping_patience"],
        optimization=policy_for("sgd"),
        **overrides,
    )


def select_rates(cfg, out, train, validation, binding, progress):
    samples = {}
    for role, arrays, maximum in (
        ("train", train, PROBE["train_windows_max"]),
        ("validation", validation, PROBE["validation_windows_max"]),
    ):
        ids = np.linspace(
            0,
            len(arrays["X"]) - 1,
            min(maximum, len(arrays["X"])),
            dtype=np.int64,
        )
        samples[role] = {k: np.asarray(v[ids]) for k, v in arrays.items()}
    record = dict(
        candidate_grid={"sgd": CANDIDATES["sgd"]},
        protocol=PROBE,
        binding=binding,
        models={},
        development_data_used=False,
        event_truth_used=False,
        final_test_used=False,
        probe_weights_reused_for_main=False,
    )
    for entry in cfg["variants"]:
        trials, initial = [], None
        for index, rate in enumerate(CANDIDATES["sgd"]):
            folder = (
                out
                / "learning-rate-probe"
                / entry["id"]
                / f"candidate-{index + 1}"
            )
            args = settings(
                cfg,
                device="cuda",
                gpu_authorized=True,
                progress=progress,
                learning_rate=rate,
            )
            args["epochs"] = PROBE["epochs"]
            history = run_fit(
                spec(entry),
                samples["train"],
                samples["validation"],
                folder,
                binding=binding,
                **args,
            )
            current = read(folder / "initialization.json")["state_sha256"]
            if initial is not None and initial != current:
                raise ValueError(
                    "Rate candidates have different initial weights"
                )
            initial = current
            trials.append(
                dict(
                    rate=rate,
                    validation_loss=history[-1]["validation_loss"],
                    history_sha256=sha(folder / "history.json"),
                )
            )
        best = min(trials, key=lambda trial: trial["validation_loss"])
        record["models"][entry["id"]] = dict(
            rate=best["rate"], trials=trials, initial_state_sha256=initial
        )
    write(out / "learning-rate-selection.json", record)
    return record


def gate_child(request):
    request = read(request)
    if request.get("phase") not in {"full", "partial", "resume"}:
        raise ValueError("Unknown explicit gate phase")
    cfg = config()
    with np.load(request["arrays"], allow_pickle=False) as source:
        train = {k: source["train_" + k] for k in ("X", "Y")}
        val = {k: source["validation_" + k] for k in ("X", "Y")}
    args = settings(
        cfg,
        device="cuda",
        gpu_authorized=True,
        learning_rate=request["rate"],
        progress=Progress(Path(request["out"] + "-progress.json"), hours=0.5),
    )
    args.update(
        epochs=4,
        patience=10,
        stop_after_epoch=2 if request["phase"] == "partial" else None,
        resume=request["resume"] if request["phase"] == "resume" else None,
    )
    run_fit(
        spec(request["entry"]),
        train,
        val,
        request["out"],
        binding=request["binding"],
        **args,
    )


def gates(cfg, out, train, val, binding, selected, progress):
    if min(len(train["X"]), len(val["X"])) < 512:
        raise ValueError(
            "Production preflight requires512 windows per normal role"
        )
    folder = out / "gates"
    folder.mkdir()
    sample = folder / "inputs.npz"
    np.savez(
        sample,
        **{
            role + "_" + k: v[:512]
            for role, arrays in (("train", train), ("validation", val))
            for k, v in arrays.items()
        },
    )
    proof = {}
    for entry in cfg["variants"]:
        name = entry["id"]
        rate = selected["models"][name]["rate"]
        benchmark = measure(
            spec(entry),
            train,
            [256],
            folder / (name + "-memory"),
            device="cuda",
            gpu_authorized=True,
            runtime_stage="fit",
            progress=progress,
            seed=0,
            warmup=50,
            steps=500,
            validation=val,
            learning_rate=rate,
            optimization=policy_for("sgd"),
        )
        for phase in ("full", "partial", "resume"):
            request = folder / f"{name}-{phase}.json"
            write(
                request,
                dict(
                    arrays=str(sample),
                    entry=entry,
                    rate=rate,
                    binding=binding,
                    phase=phase,
                    out=str(folder / (name + "-" + phase)),
                    resume=str(folder / (name + "-partial/last.pt")),
                ),
            )
            with request.with_suffix(".log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "learning.history19.runner",
                        "--gate-request",
                        str(request),
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
        full = torch.load(
            folder / (name + "-full/last.pt"),
            map_location="cpu",
            weights_only=False,
        )
        resumed = torch.load(
            folder / (name + "-resume/last.pt"),
            map_location="cpu",
            weights_only=False,
        )
        for key in (
            "model",
            "optimizer",
            "rng",
            "completed_optimizer_steps",
            "early_stop_bad_epochs",
        ):
            if not same(full[key], resumed[key]):
                raise ValueError("Separate-process restore mismatch: " + key)
        for key in ("train_loss", "validation_loss"):
            if [r[key] for r in full["history"]] != [
                r[key] for r in resumed["history"]
            ]:
                raise ValueError("Resume loss mismatch")
        if sha(folder / (name + "-full/model.safetensors")) != sha(
            folder / (name + "-resume/model.safetensors")
        ):
            raise ValueError("Resume exported weights differ")
        proof[name] = dict(memory=benchmark, resume_exact=True)
    write(out / "gates.json", proof)


def fit(manifest, out, *, approved=False):
    if not approved:
        raise ValueError(
            "Explicit --approve-training required; code changes do not start training"
        )
    out = Path(out).absolute()
    manifest = Path(manifest).absolute()
    if any(p.is_symlink() for p in (manifest, *manifest.parents)):
        raise ValueError("Dataset symlink input forbidden")
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("Symlink output forbidden")
    if out.is_relative_to(manifest.parent) or manifest.parent.is_relative_to(
        out
    ):
        raise ValueError("Model output must be separate from sealed dataset")
    out.mkdir(parents=True, exist_ok=False)
    cfg = config()
    progress = Progress(
        out / "progress.json",
        hours=cfg["gpu"]["proposed_fit_including_preflight_hours"],
    )
    try:
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError(
                "A verified existing DIS01 allocation is required before fit"
            )
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
            raise ValueError("Fixed CUBLAS workspace required")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from pipeline.runtime import check

        check(ROOT / "ai", out, "fit", job_prefix="drone-ai-")
        from execution.common import child, memory_snapshot
        from execution.prepare import npz_headers

        from learning.comparison import memory_plan

        manifest_path = Path(manifest).absolute()
        membership = read(manifest_path)
        headers = {
            role: npz_headers(
                child(manifest_path.parent, membership["files"][role]["path"])
            )
            for role in ("normal_train", "normal_validation")
        }
        headers["scaler_frames"] = npz_headers(
            child(
                manifest_path.parent,
                membership["files"]["normal_train_frames"]["path"],
            )
        )
        plan = memory_plan(headers)
        available = memory_snapshot()["available_bytes"]
        if plan["estimated_host_peak_bytes"] + 1024**3 > min(
            64 * 1024**3, available
        ):
            raise MemoryError(
                "Fixed64GiB host-memory plan rejected; no smaller batch or omission"
            )
        import shutil

        if shutil.disk_usage(out).free < plan["array_bytes"] + 24 * 1024**3:
            raise OSError("Insufficient disk for new mmap cache and evidence")
        write(
            out / "memory-admission.json",
            dict(plan=plan, available_bytes=available, passed=True),
        )
        arrays, scaler, binding = load(
            manifest, cache=out / "normalized", progress=progress
        )
        sources = {
            str(p.relative_to(ROOT)): sha(p)
            for p in Path(__file__).parent.glob("*.py")
        }
        sources["ai/learning/optimization.py"] = sha(
            ROOT / "ai/learning/optimization.py"
        )
        binding["sources"] = sources
        write(
            out / "request.json",
            dict(
                manifest=str(Path(manifest).absolute()),
                binding=binding,
                config=cfg,
                automatic_retry=False,
                automatic_resume=False,
            ),
        )
        selected = select_rates(
            cfg,
            out,
            arrays["normal_train"],
            arrays["normal_validation"],
            binding,
            progress,
        )
        gates(
            cfg,
            out,
            arrays["normal_train"],
            arrays["normal_validation"],
            binding,
            selected,
            progress,
        )
        for entry in cfg["variants"]:
            progress.check()
            target = out / "models" / entry["id"]
            choice = selected["models"][entry["id"]]
            run_fit(
                spec(entry),
                arrays["normal_train"],
                arrays["normal_validation"],
                target,
                binding=binding,
                **settings(
                    cfg,
                    device="cuda",
                    gpu_authorized=True,
                    progress=progress,
                    learning_rate=choice["rate"],
                ),
            )
            if (
                read(target / "initialization.json")["state_sha256"]
                != choice["initial_state_sha256"]
            ):
                raise ValueError("Main model did not start fresh")
            from safetensors.torch import load_file

            reference = make_model(spec(entry)).to("cuda").eval()
            exported = make_model(spec(entry)).to("cuda").eval()
            checkpoint = torch.load(
                target / "best.pt", map_location="cpu", weights_only=False
            )
            reference.load_state_dict(checkpoint["best_state"])
            exported.load_state_dict(
                load_file(str(target / "model.safetensors"))
            )
            sample = torch.as_tensor(
                arrays["normal_validation"]["X"][:2],
                dtype=torch.float32,
                device="cuda",
            )
            with backend("cuda"), torch.no_grad():
                difference = float(
                    (reference(sample) - exported(sample)).abs().max()
                )
            if not np.isfinite(difference) or difference > 1e-6:
                raise ValueError(
                    "Same-device best-checkpoint/export restoration differs"
                )
            write(
                target / "export-check.json",
                dict(passed=True, max_abs=difference, atol=1e-6),
            )
            del reference, exported, checkpoint, sample
            seal_export(target, scaler)
            loaded = Predictor(target)
            if not np.isfinite(
                loaded.predict_scaled(arrays["normal_validation"]["X"][:2])
            ).all():
                raise ValueError("Restored export nonfinite")
        write(
            out / "result.json",
            dict(
                status="FIT_ONLY_STOPPED",
                models=[v["id"] for v in cfg["variants"]],
                calibration_complete=False,
                evaluation_complete=False,
                automatic_retry=False,
                next_stage_authorized=False,
            ),
        )
    except BaseException as error:
        write(
            out / "failure.json",
            dict(
                status="FAILED_STOPPED", error=str(error), automatic_retry=False
            ),
        )
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Internal finite history19 gate worker"
    )
    parser.add_argument("--gate-request", type=Path, required=True)
    gate_child(parser.parse_args().gate_request)
