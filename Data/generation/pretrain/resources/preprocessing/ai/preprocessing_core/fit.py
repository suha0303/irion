"""R1 training transport; the existing model, loss, optimizer math and scaler are reused."""

import copy
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import capstone_ai.models as model_source
import capstone_ai.training as training_source
from capstone_ai.models import device_for, make_model
from capstone_ai.training import loss, seed_all

from . import backend as backend_source
from .backend import (
    SCHEMA,
    backend_scope,
    check_model,
    execution_environment,
    resume_guard,
    trace_batch,
)
from .control import best_update, sha, write


def batch(data, indices, device):
    """Slice host arrays first. Never convert a whole split to a device tensor."""
    return [
        torch.as_tensor(data[k][indices], dtype=torch.float32, device=device)
        for k in ("X", "C", "Y")
    ]


def data_signature(train, validation):
    digest = hashlib.sha256()
    for data in (train, validation):
        for key in ("X", "C", "Y"):
            array = data[key]
            digest.update(str((key, array.shape, array.dtype.str)).encode())
            for i in range(0, len(array), 32):
                digest.update(np.ascontiguousarray(array[i : i + 32]).tobytes())
    return digest.hexdigest()


def training_state(model, optimizer, device):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def neural_fit(
    config, train, validation, folder, budget, resume=None, stop_after=None, diagnostic_prefix=None
):
    # Covers construction/.to, train, validation, export, and restored-model inference.
    with backend_scope(config):
        return _fit(
            config, train, validation, folder, budget, resume, stop_after, diagnostic_prefix
        )


def _fit(config, train, validation, folder, budget, resume, stop_after, diagnostic_prefix):
    """Only LSTM; research caller fixes config, lifecycle caller fixes its separate fixture."""
    folder = Path(folder)
    device = device_for(config["device"])
    if not len(train["X"]) or not len(validation["X"]):
        raise ValueError("empty train/validation")
    seed_all(config["seed"])
    model = make_model("lstm", config["hidden"]).to(device)
    check_model(model, device)
    environment = execution_environment(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    signature = data_signature(train, validation)
    sources = {
        "fit.py": sha(__file__),
        "models.py": sha(model_source.__file__),
        "backend.py": sha(backend_source.__file__),
        "training.py": sha(training_source.__file__),
    }
    start, best, best_epoch, history = 0, float("inf"), -1, []
    best_state = copy.deepcopy(model.state_dict())
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        resume_guard(checkpoint, config, signature, sources, environment)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            if len(checkpoint["cuda_rng"]) != 1:
                raise ValueError("one CUDA RNG state required")
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        start, best, best_epoch, history, best_state = [
            checkpoint[k] for k in ("next_epoch", "best", "best_epoch", "history", "best_state")
        ]
        if diagnostic_prefix:
            # Re-read actual restored objects and generators, not a copy of the input checkpoint.
            restored_boundary = {**checkpoint, **training_state(model, optimizer, device)}
            torch.save(restored_boundary, folder / (diagnostic_prefix + "-restored.pt"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, config["epochs"]):
        budget.check()
        model.train()
        order = torch.randperm(len(train["X"]))
        total = 0.0
        for batch_index, indices in enumerate(order.split(config["batch_size"])):
            budget.check()
            x, c, y = batch(train, indices.numpy(), device)
            check_model(model, device)
            if diagnostic_prefix and epoch == 2 and batch_index == 0:
                torch.save(
                    {
                        **training_state(model, optimizer, device),
                        "indices": indices,
                        "epoch": epoch,
                    },
                    folder / (diagnostic_prefix + "-epoch2-pre.pt"),
                )
            with trace_batch(
                folder,
                str(diagnostic_prefix) + "-train",
                device,
                True,
                bool(diagnostic_prefix) and epoch == start and batch_index == 0,
            ):
                optimizer.zero_grad()
                value = loss(model(x, c), y)
                if not torch.isfinite(value):
                    raise ValueError("nonfinite training loss")
                value.backward()
                if not all(
                    p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
                ):
                    raise ValueError("nonfinite gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if diagnostic_prefix and epoch == 2 and batch_index == 0:
                torch.save(
                    {
                        **training_state(model, optimizer, device),
                        "indices": indices,
                        "epoch": epoch,
                        "loss": value.item(),
                    },
                    folder / (diagnostic_prefix + "-epoch2-post.pt"),
                )
            total += value.item() * len(indices)
            del x, c, y, value
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for i in range(0, len(validation["X"]), config["batch_size"]):
                budget.check()
                x, c, y = batch(validation, slice(i, i + config["batch_size"]), device)
                check_model(model, device)
                with trace_batch(
                    folder,
                    str(diagnostic_prefix) + "-validation",
                    device,
                    False,
                    bool(diagnostic_prefix) and epoch == start and i == 0,
                ):
                    vtotal += loss(model(x, c), y).item() * len(x)
                del x, c, y
        vloss = vtotal / len(validation["X"])
        if not np.isfinite(vloss):
            raise ValueError("nonfinite validation loss")
        history.append(
            {"epoch": epoch, "train_loss": total / len(train["X"]), "validation_loss": vloss}
        )
        improved = best_update(vloss, best)
        if improved:
            best, best_epoch, best_state = vloss, epoch, copy.deepcopy(model.state_dict())
        checkpoint = {
            "schema": SCHEMA,
            "environment": environment,
            "lstm_backend": config["lstm_backend"],
            "dropout": 0.1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "best": best,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "history": history,
            "config": config,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "data_order": order.tolist(),
            "data_signature": signature,
            "source_hashes": sources,
        }
        torch.save(checkpoint, folder / "last.pt")
        if diagnostic_prefix and epoch == 1:
            torch.save(checkpoint, folder / "boundary-epoch1.pt")
        if improved:
            torch.save(checkpoint, folder / "best.pt")
        write(folder / "epoch-history.json", history)
        print("lstm", history[-1], flush=True)
        budget.check()
        if stop_after is not None and epoch + 1 >= stop_after:
            break
        if epoch - best_epoch >= config.get("early_stopping_patience", config["epochs"]):
            break
    model.load_state_dict(best_state)
    model.eval()
    save_file(
        {k: v.cpu().contiguous() for k, v in model.state_dict().items()},
        str(folder / "model.safetensors"),
    )
    restored = make_model("lstm", config["hidden"]).to(device)
    restored.load_state_dict(load_file(str(folder / "model.safetensors")))
    restored.eval()
    check_model(model, device)
    check_model(restored, device)
    with (
        torch.no_grad(),
        trace_batch(
            folder, str(diagnostic_prefix) + "-save-load", device, False, bool(diagnostic_prefix)
        ),
    ):
        x, c, _ = batch(validation, slice(0, 2), device)
        difference = (model(x, c) - restored(x, c)).abs().max().item()
    if not np.isfinite(difference) or difference > 1e-6:
        raise ValueError("save/restore mismatch")
    result = {
        "lstm_backend": config["lstm_backend"],
        "environment": environment,
        "history": history,
        "selected_epoch": best_epoch,
        "save_load_max_abs": difference,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
    }
    write(folder / "fit.json", result)
    return result
