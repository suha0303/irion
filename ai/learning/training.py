"""Next-version training building blocks. CUDA requires a verified existing DIS01 allocation.

No launcher, submission, automatic tuning, retries, or production input selection.
"""

import fcntl
import hashlib
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from capstone_ai.training import loss, seed_all
from execution.common import write

from .models import STUDY_REVISION, make_model


@contextmanager
def backend(device):
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    before = {
        k: getattr(torch.backends.cudnn, k)
        for k in ("enabled", "benchmark", "benchmark_limit", "deterministic", "allow_tf32")
    }
    # Explicit reproducibility policy; restore global flags even after a failed step.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(
            **{
                **before,
                "enabled": device == "cuda",
                "benchmark": False,
                "deterministic": True,
                "allow_tf32": False,
            }
        ):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


class Batches:
    """Float32 conversion once; bounded reusable pinned buffers, ordered transfer events."""

    def __init__(self, arrays, batch_size, device):
        self.device = torch.device(device)
        self.data = {
            k: torch.from_numpy(np.ascontiguousarray(arrays[k], dtype=np.float32))
            for k in ("X", "C", "Y")
        }
        n = len(self.data["X"])
        if n == 0 or any(
            len(v) != n or any(not torch.isfinite(v[i : i + 512]).all() for i in range(0, n, 512))
            for v in self.data.values()
        ):
            raise ValueError("Empty, mismatched or nonfinite fit input")
        if (
            self.data["X"].shape[1:] != (100, 19)
            or self.data["C"].shape[1:] != (6,)
            or self.data["Y"].shape[1:] != (10, 4)
        ):
            raise ValueError("Forecast input shape mismatch")
        self.count, self.capacity, self.slot = n, batch_size, 0
        self.buffers = (
            [
                {
                    k: torch.empty((batch_size, *v.shape[1:]), dtype=torch.float32, pin_memory=True)
                    for k, v in self.data.items()
                }
                for _ in range(2)
            ]
            if device == "cuda"
            else []
        )
        self.events = [None, None]

    def get(self, indices):
        ids = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        if not 0 < len(ids) <= self.capacity:
            raise ValueError("Batch exceeds declared capacity")
        if self.device.type == "cpu":
            return [v.index_select(0, ids) for v in self.data.values()]
        slot = self.slot
        self.slot = (slot + 1) % 2
        if self.events[slot] is not None:
            self.events[slot].synchronize()  # Buffer lifetime, not a per-tensor device-wide sync.
        for k, v in self.data.items():
            torch.index_select(v, 0, ids, out=self.buffers[slot][k][: len(ids)])
        result = [
            v[: len(ids)].to(self.device, non_blocking=True) for v in self.buffers[slot].values()
        ]
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.events[slot] = event
        return result


def clip_gradients(model):
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise ValueError("Missing gradients")
    # The pre-clip L2 norm includes every gradient: NaN, Inf and overflow propagate.
    # Reuse this existing reduction instead of launching isfinite/all per parameter.
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=True)
    if not bool(torch.isfinite(norm)):
        raise ValueError("Nonfinite gradients/norm; optimizer not executed")


def safe_step(model, optimizer, x, c, y):
    optimizer.zero_grad(set_to_none=True)
    value = loss(model(x, c), y)
    if not bool(torch.isfinite(value)):
        raise ValueError("Nonfinite loss; optimizer not executed")
    value.backward()
    clip_gradients(model)
    optimizer.step()
    return value.detach()


def state_rng():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    )


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if device == "cuda":
        if not state["cuda"]:
            raise ValueError("Missing CUDA RNG")
        torch.cuda.set_rng_state_all(state["cuda"])


def _run_fit(
    spec,
    train,
    validation,
    folder,
    *,
    binding,
    device="cpu",
    seed=17,
    batch_size=128,
    epochs=20,
    patience=5,
    learning_rate=0.001,
    resume=None,
    stop_after_epoch=None,
    progress=None,
    gpu_authorized=False,
    runtime_stage="fit",
    model_id=None,
    optimization=None,
):
    """Epoch-boundary resume only. Caller supplies sealed data/config/source binding.

    CPU mode is for small synthetic tests. Main-round approval/packaging remains separate.
    """
    spec.validate()
    study = spec.revision == STUDY_REVISION
    if study != (optimization is not None):
        raise ValueError("The normalization study requires its explicit optimization policy")
    if study:
        from .optimization import WarmupCosine, make_optimizer, validate

        validate(optimization)
    if device not in {"cpu", "cuda"} or min(batch_size, epochs, patience) < 1:
        raise ValueError("Invalid training settings")
    if not isinstance(binding, dict) or not all(
        binding.get(k) for k in ("data", "scaler", "sources")
    ):
        raise ValueError("Sealed data/scaler/source binding required")
    folder = Path(folder)
    environment = {"device": device, "torch": str(torch.__version__), "numpy": np.__version__}
    if device == "cuda":
        if not gpu_authorized:
            raise ValueError("Separate explicit GPU training approval required")
        from pipeline.runtime import check

        if progress is None or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUDA requires an explicit budget and fixed CUBLAS workspace policy")
        if runtime_stage not in {"fit", "benchmark"}:
            raise ValueError("Unknown CUDA purpose")
        verified = check(
            Path(__file__).resolve().parents[1], folder, runtime_stage, job_prefix="drone-ai-"
        )
        # Job IDs change between fresh-process resume tests; bind numerical environment instead.
        environment.update(
            {
                k: verified[k]
                for k in ("versions", "cuda", "cudnn_version", "gpu_name", "gpu_memory")
            }
        )
    cfg = dict(
        spec=asdict(spec),
        seed=seed,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        precision="float32",
        amp=False,
        epoch_boundary_only=True,
    )
    if study:
        cfg["optimization"] = optimization
    seed_all(seed)
    tb, vb = Batches(train, batch_size, device), Batches(validation, batch_size, device)
    steps = (tb.count + batch_size - 1) // batch_size
    total_steps = epochs * steps
    schedule = WarmupCosine(total_steps, learning_rate, optimization) if study else None
    warmup = schedule.warmup if study else 0
    with backend(device):
        model = make_model(spec)
        if study:
            digest = hashlib.sha256()
            for name, parameter in model.state_dict().items():
                digest.update(name.encode())
                digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
                digest.update(memoryview(parameter.detach().contiguous().numpy()).cast("B"))
            initial_hash = digest.hexdigest()
            write(folder / "initialization.json", dict(
                seed=seed, state_sha256=initial_hash, model_spec=asdict(spec),
                weights_from_previous_training=False,
            ))
        model = model.to(device)
        optimizer = (
            make_optimizer(model.parameters(), learning_rate, optimization)
            if study else torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        )
        completed_steps, stale_epochs = 0, 0
        history = []
        best = float("inf")
        best_epoch = -1
        best_state = None
        start = 0
        if resume is not None:
            # Only local trusted project checkpoints, never untrusted uploads.
            ck = torch.load(resume, map_location="cpu", weights_only=False)
            if (
                ck.get("schema") != "x500-next-fit-v1"
                or ck["config"] != cfg
                or ck["binding"] != binding
                or ck["environment"] != environment
            ):
                raise ValueError("Resume binding/config/environment mismatch")
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            history, best, best_epoch, best_state = (
                ck["history"],
                ck["best"],
                ck["best_epoch"],
                ck["best_state"],
            )
            start = ck["next_epoch"]
            if study:
                completed_steps = ck["completed_optimizer_steps"]
                stale_epochs = ck["early_stop_bad_epochs"]
                if ck["initial_state_sha256"] != initial_hash or completed_steps != start * steps:
                    raise ValueError("Initial weights or optimizer-step position changed")
            finished_patience = stale_epochs >= patience if study else start - 1 - best_epoch >= patience
            if start >= epochs or finished_patience:
                raise ValueError("Checkpoint already finished its approved epochs/early stopping")
            restore_rng(ck["rng"], device)
        for epoch in range(start, epochs):
            if progress is not None:
                progress.update(
                    stage="training",
                    model=model_id or spec.name,
                    epoch=epoch + 1,
                    epochs=epochs,
                    batch=0,
                    batches=steps,
                    done=epoch * steps,
                    total=epochs * steps,
                )
            order = torch.randperm(tb.count)
            model.train()
            total = torch.zeros((), device=device, dtype=torch.float64)
            started = time.monotonic()
            last_report = started
            for i, indices in enumerate(order.split(batch_size)):
                if progress is not None:
                    progress.check()
                x, c, y = tb.get(indices)
                if study:
                    current_rate = schedule(completed_steps)
                    for group in optimizer.param_groups:
                        group["lr"] = current_rate
                value = safe_step(model, optimizer, x, c, y)
                completed_steps += 1
                total.add_(value.to(torch.float64) * len(indices))
                if progress is not None and time.monotonic() - last_report >= 2:
                    # Scalar read only at display cadence, not on every batch.
                    progress.update(
                        stage="training",
                        model=model_id or spec.name,
                        epoch=epoch + 1,
                        batch=i + 1,
                        batches=steps,
                        epochs=epochs,
                        done=epoch * steps + i + 1,
                        total=epochs * steps,
                        phase_eta_s=(time.monotonic() - started) / (i + 1) * (steps - i - 1),
                        train_loss=float(total) / min((i + 1) * batch_size, tb.count),
                    )
                    last_report = time.monotonic()
            train_loss = float(total) / tb.count
            train_seconds = time.monotonic() - started
            model.eval()
            val_total = torch.zeros((), device=device, dtype=torch.float64)
            valid_flags = torch.ones((), device=device, dtype=torch.bool)
            vt = time.monotonic()
            if progress is not None:
                progress.update(stage="validation", done=0, total=vb.count, phase_eta_s=None)
            with torch.no_grad():
                for offset in range(0, vb.count, batch_size):
                    if progress is not None:
                        progress.check()
                    x, c, y = vb.get(torch.arange(offset, min(offset + batch_size, vb.count)))
                    value = loss(model(x, c), y)
                    val_total.add_(value.to(torch.float64) * len(x))
                    valid_flags.logical_and_(torch.isfinite(value))
                    if progress is not None:
                        progress.update(force=False, done=min(offset + batch_size, vb.count))
            if not bool(valid_flags):
                raise ValueError("Nonfinite validation loss")
            score = float(val_total) / vb.count
            improved = score < best
            if improved:
                best, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if study:
                stale_epochs = 0 if improved or epoch * steps < warmup else stale_epochs + 1
            history.append(
                dict(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    validation_loss=score,
                    train_seconds=train_seconds,
                    validation_seconds=time.monotonic() - vt,
                )
            )
            if study:
                history[-1].update(
                    learning_rate_last=current_rate,
                    optimizer_steps=completed_steps,
                    early_stop_bad_epochs=stale_epochs,
                    warmup_complete=completed_steps >= warmup,
                )
            ck = dict(
                schema="x500-next-fit-v1",
                config=cfg,
                binding=binding,
                environment=environment,
                next_epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                history=history,
                best=best,
                best_epoch=best_epoch,
                best_state=best_state,
                rng=state_rng(),
            )
            if study:
                ck.update(completed_optimizer_steps=completed_steps,
                          early_stop_bad_epochs=stale_epochs, initial_state_sha256=initial_hash)
            temp = folder / "last.tmp"
            torch.save(ck, temp)
            temp.replace(folder / "last.pt")
            if improved:
                torch.save(ck, folder / "best.pt")
            write(folder / "history.json", history)
            if progress is not None:
                progress.update(
                    stage="checkpoint",
                    train_loss=train_loss,
                    validation_loss=score,
                    best_epoch=best_epoch + 1,
                    done=epoch + 1,
                    total=epochs,
                    phase_eta_s=None,
                    **(dict(learning_rate=current_rate, optimizer=optimization["optimizer"],
                            optimizer_steps=completed_steps, schedule_steps=total_steps)
                       if study else {}),
                )
            if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
                break
            if (stale_epochs >= patience if study else epoch - best_epoch >= patience):
                break
        if best_state is None:
            raise ValueError("No trained model")
        save_file(
            {k: v.contiguous() for k, v in best_state.items()}, str(folder / "model.safetensors")
        )
        write(folder / "model-spec.json", asdict(spec))
        write(folder / "config.json", cfg)
        write(
            folder / "result.json",
            dict(
                status="FIT_ONLY_STOPPED",
                calibration_complete=False,
                evaluation_complete=False,
                automatic_retry=False,
                gpu_resume_verified=False,
                epochs=len(history),
            ),
        )
    return history


@contextmanager
def project_lock(device):
    if device != "cuda":
        yield
        return
    from execution.remote import REMOTE_LOCK

    with REMOTE_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run_fit(spec, train, validation, folder, **kwargs):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        if kwargs.get("device") == "cuda" and not kwargs.get("gpu_authorized"):
            raise ValueError("Separate explicit GPU training approval required")
        with project_lock(kwargs.get("device", "cpu")):
            return _run_fit(spec, train, validation, folder, **kwargs)
    except BaseException as exc:
        write(
            folder / "failure.json",
            dict(
                status="FAILED_STOPPED",
                reason=f"{type(exc).__name__}: {exc}",
                automatic_retry=False,
            ),
        )
        raise
