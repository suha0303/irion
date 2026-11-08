"""Bounded CPU development fit. No Slurm submission, server access or
full-run launch.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import safetensors
import torch
from safetensors.torch import save_file

from capstone_ai.training import seed_all
from learning.history19.training import backend, clip_gradients
from learning.optimization import WarmupCosine, make_optimizer, policy_for

from .data import Scaler, batch, fingerprint, validate_splits
from .models import ModelSpec, make_model, normal_loss


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def fit_cpu(
    spec,
    train,
    validation,
    folder,
    *,
    epochs=4,
    batch_size=2,
    seed=0,
    learning_rate=0.01,
    resume=None,
    stop_after_epoch=None,
):
    """Epoch-boundary checkpoint test on small inputs, normal-only, FP32/SGD.

    Total epochs fixes the schedule before the first step. A stopped run resumes
    into a new directory, preserving every previous result/checkpoint.
    """
    spec.validate()
    validate_splits([*train, *validation])
    if (
        not train
        or not validation
        or any(r.role != "normal_train" for r in train)
        or any(r.role != "normal_validation" for r in validation)
    ):
        raise ValueError(
            "Fit and selection require separate normal_train/normal_validation"
        )
    if any(len(r.values) <= 100 for r in [*train, *validation]):
        raise ValueError(
            "Each fit/validation run needs a post-initial scoring interval"
        )
    if (
        len(train) + len(validation) > 16
        or sum(len(r.values) for r in [*train, *validation]) > 20000
    ):
        raise ValueError(
            "This is a bounded local development fit, not production training"
        )
    if (
        any(type(v) is not int or v < 1 for v in (epochs, batch_size))
        or type(seed) is not int
    ):
        raise ValueError(
            "Positive integer epochs/batch size and integer seed required"
        )
    total_steps = math.ceil(len(train) / batch_size) * epochs
    optimization = policy_for("sgd")
    schedule = WarmupCosine(total_steps, learning_rate, optimization)
    end = epochs if stop_after_epoch is None else stop_after_epoch
    if type(end) is not int or not 1 <= end <= epochs:
        raise ValueError("Stop epoch must be inside the fixed plan")
    scaler = Scaler.fit(train)
    plan = dict(
        model=spec.to_dict(),
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        learning_rate=learning_rate,
        optimization=optimization,
        device="cpu",
        precision="fp32",
        train_hash=fingerprint(train),
        validation_hash=fingerprint(validation),
        scaler=scaler.to_dict(),
    )
    # Bind resume to implementation as well as data/scaler/hyperparameters.
    learning = Path(__file__).parent.parent
    sources = [
        Path(__file__).with_name(n)
        for n in ("models.py", "data.py", "training.py")
    ]
    sources += [
        learning / n
        for n in (
            "history19/models.py",
            "history19/training.py",
            "optimization.py",
        )
    ]
    sources.append(learning.parent / "src/capstone_ai/training.py")
    plan["source_hashes"] = {
        str(p.relative_to(learning.parent)): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sources
    }
    plan["environment"] = dict(
        python=sys.version,
        torch=str(torch.__version__),
        numpy=np.__version__,
        safetensors=safetensors.__version__,
        threads=2,
    )
    seed_all(seed)
    model = make_model(spec).cpu()
    optimizer = make_optimizer(model.parameters(), learning_rate, optimization)
    start, step, records, best_loss, best_epoch, best_state = (
        0,
        0,
        [],
        None,
        None,
        None,
    )
    if resume is not None:
        resume = Path(resume)
        if resume.name != "last.pt" or any(
            p.is_symlink() for p in (resume, *resume.parents)
        ):
            raise ValueError(
                "Resume requires the saved, regular last.pt checkpoint"
            )
        integrity = json.loads(
            (resume.parent / "integrity.json").read_text(encoding="utf-8")
        )
        if (
            hashlib.sha256(resume.read_bytes()).hexdigest()
            != integrity["last.pt"]
        ):
            raise ValueError("Resume checkpoint integrity mismatch")
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["plan"] != plan:
            raise ValueError(
                "Resume data/model/scaler/source/environment/plan changed"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, step, records = saved["epoch"], saved["step"], saved["records"]
        best_loss, best_epoch, best_state = (
            saved["best_loss"],
            saved["best_epoch"],
            saved["best_model"],
        )
        torch.set_rng_state(saved["rng"])
    if not start < end:
        raise ValueError("No remaining epochs in requested interval")
    folder = Path(folder)
    if any(p.is_symlink() for p in (folder, *folder.parents)):
        raise ValueError("Symlink output unsupported")
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "plan.json", plan)
    with backend("cpu"):
        for epoch in range(start, end):
            model.train()
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(train), generator=generator).tolist()
            train_sum = 0.0
            for offset in range(0, len(order), batch_size):
                selected = [
                    train[i] for i in order[offset : offset + batch_size]
                ]
                x, lengths = batch(selected, scaler)
                for group in optimizer.param_groups:
                    group["lr"] = schedule(step)
                optimizer.zero_grad(set_to_none=True)
                loss = normal_loss(model(x, lengths), x[:, :, 17], lengths)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite loss; no optimizer step")
                loss.backward()
                clip_gradients(model)
                optimizer.step()
                step += 1
                train_sum += float(loss.detach()) * len(selected)
            model.eval()
            validation_sum = 0.0
            with torch.no_grad():
                for offset in range(0, len(validation), batch_size):
                    selected = validation[offset : offset + batch_size]
                    x, lengths = batch(selected, scaler)
                    value = normal_loss(model(x, lengths), x[:, :, 17], lengths)
                    if not bool(torch.isfinite(value)):
                        raise ValueError("Nonfinite validation loss")
                    validation_sum += float(value) * len(selected)
            value = validation_sum / len(validation)
            records.append(
                dict(
                    epoch=epoch + 1,
                    train_loss=train_sum / len(train),
                    validation_loss=value,
                )
            )
            if best_loss is None or value < best_loss:
                best_loss, best_epoch = value, epoch + 1
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            saved = dict(
                plan=plan,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch + 1,
                step=step,
                records=records,
                rng=torch.get_rng_state(),
                best_loss=best_loss,
                best_epoch=best_epoch,
                best_model=best_state,
            )
            torch.save(saved, folder / "last.pt")
    save_file(best_state, folder / "model.safetensors")
    write(folder / "model-spec.json", spec.to_dict())
    write(folder / "scaler.json", scaler.to_dict())
    result = dict(
        status="completed" if end == epochs else "stopped_at_epoch_boundary",
        model=spec.to_dict(),
        device="cpu",
        epochs=end,
        optimizer_steps=step,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        records=records,
        origins=sorted({r.origin for r in [*train, *validation]}),
        detection_performance_verified=False,
    )
    write(folder / "result.json", result)
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.iterdir()
        if p.is_file()
    }
    write(folder / "integrity.json", manifest)
    return result


def restore(folder):
    from safetensors.torch import load_file

    folder = Path(folder)
    if any(p.is_symlink() for p in (folder, *folder.parents)):
        raise ValueError("Symlink model folder unsupported")
    manifest = json.loads(
        (folder / "integrity.json").read_text(encoding="utf-8")
    )
    for name in (
        "plan.json",
        "model-spec.json",
        "scaler.json",
        "model.safetensors",
    ):
        if (folder / name).is_symlink():
            raise ValueError("Symlink model file unsupported")
        if (
            hashlib.sha256((folder / name).read_bytes()).hexdigest()
            != manifest[name]
        ):
            raise ValueError("Saved model integrity mismatch")
    model = make_model(
        ModelSpec(
            **json.loads(
                (folder / "model-spec.json").read_text(encoding="utf-8")
            )
        )
    )
    model.load_state_dict(load_file(folder / "model.safetensors"))
    return model.eval(), Scaler.from_dict(
        json.loads((folder / "scaler.json").read_text(encoding="utf-8"))
    )
