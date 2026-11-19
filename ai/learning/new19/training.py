"""AdamW fitting, independent LR sweeps, and epoch-boundary resume."""

import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from .data import (
    TRAIN_ORIGIN_INTERVAL_US,
    Batches,
    digest,
    fit_scaler,
    load_split,
    write,
)
from .models import VARIANTS, make_model

WEIGHT_DECAY = 1e-4
SEED = 0
SCHEMA = "drone-ai.new19.fit/1"


def objective(model, batch):
    x, y, mask = batch
    if mask is None:
        error = F.smooth_l1_loss(model(x), y, reduction="none", beta=1)
        return 0.5 * error[..., :3].mean() + 0.5 * error[..., 3].mean()
    predicted = model(x, y).squeeze(-1)
    observed = torch.where(mask, x[..., 17], predicted.detach())
    error = F.smooth_l1_loss(predicted, observed, reduction="none", beta=1)
    return ((error * mask).sum(1) / mask.sum(1)).mean()


def lr_at(step, total, peak, architecture):
    fraction = 0.02 if architecture == "lstm" else 0.07
    up = min(total - 1, math.floor(total * fraction))
    if step < up:
        return peak * (0.1 + 0.9 * step / up)
    progress = (step - up) / max(1, total - 1 - up)
    return peak * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)


def adamw(model, rate):
    return torch.optim.AdamW(
        model.parameters(), lr=rate, weight_decay=WEIGHT_DECAY
    )


def to_device(batch, device):
    return tuple(x.to(device) if x is not None else None for x in batch)


def update(model, optimizer, batch, rate):
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    loss = objective(model, to_device(batch, next(model.parameters()).device))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return float(loss.detach())


def validation(model, data, batch_size):
    model.eval()
    losses = 0.0
    with torch.inference_mode():
        for start in range(0, len(data.items), batch_size):
            ids = range(start, min(len(data.items), start + batch_size))
            batch = to_device(data.batch(ids), next(model.parameters()).device)
            losses += len(ids) * float(objective(model, batch))
    value = losses / len(data.items)
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite validation loss")
    return value


def training_batches(runs, task):
    scaler = fit_scaler(runs, task, origin_interval_us=TRAIN_ORIGIN_INTERVAL_US)
    return Batches(
        runs, scaler, task, origin_interval_us=TRAIN_ORIGIN_INTERVAL_US
    )


def datasets(manifest, name):
    task = VARIANTS[name].task
    roles = load_split(manifest, ("normal_train", "normal_validation"))
    train = training_batches(roles["normal_train"], task)
    return (
        train,
        Batches(roles["normal_validation"], train.scaler, task),
        train.scaler,
    )


def fit(
    manifest,
    name,
    output,
    *,
    rate,
    epochs,
    batch_size,
    resume=False,
    until_epoch=None,
    device="cpu",
    patience=None,
):
    if not math.isfinite(rate) or rate <= 0 or epochs < 1 or batch_size < 1:
        raise ValueError("Provide a positive rate, epoch count, and batch size")
    stop = epochs if until_epoch is None else until_epoch
    if not 1 <= stop <= epochs:
        raise ValueError("Epoch boundary outside the fixed plan")
    if patience is not None and (type(patience) is not int or patience < 1):
        raise ValueError("Patience must be a positive integer")
    manifest, output = Path(manifest), Path(output)
    train, val, scaler = datasets(manifest, name)
    n_steps = math.ceil(len(train.items) / batch_size)
    binding = dict(
        schema=SCHEMA,
        variant=name,
        manifest_sha256=digest(manifest),
        scaler=scaler,
        train_origin_interval_us=train.origin_interval_us,
        validation_origin_interval_us=val.origin_interval_us,
        epochs=epochs,
        batch_size=batch_size,
        rate=rate,
        seed=SEED,
        device=str(device),
        patience=patience,
        torch=str(torch.__version__),
        sources={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    )
    torch.manual_seed(SEED)
    model = make_model(name).to(device)
    optimizer = adamw(model, rate)
    checkpoint = output / "last.pt"
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["binding"] != binding:
            raise ValueError(
                "Cannot resume with a changed fit plan, source or data"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, best_model, logs = (
            state[k] for k in ("epoch", "best", "best_model", "logs")
        )
        if start > stop:
            raise ValueError("Resume target predates the stored epoch")
    else:
        output.mkdir(parents=True, exist_ok=False)
        start, best, best_model, logs = 0, math.inf, None, []
        write(output / "fit-plan.json", binding)
    fraction = 0.02 if model.variant.architecture == "lstm" else 0.07
    warmup = min(epochs * n_steps - 1, math.floor(epochs * n_steps * fraction))
    bad = logs[-1].get("bad_epochs", 0) if logs else 0
    early_stopped = patience is not None and bad >= patience
    for epoch in range(start, stop):
        if early_stopped:
            break
        torch.manual_seed(SEED + epoch)
        order = np.random.default_rng(SEED + epoch).permutation(
            len(train.items)
        )
        model.train()
        total = 0.0
        for index in range(n_steps):
            ids = order[index * batch_size : (index + 1) * batch_size]
            peak = lr_at(
                epoch * n_steps + index,
                epochs * n_steps,
                rate,
                model.variant.architecture,
            )
            total += len(ids) * update(model, optimizer, train.batch(ids), peak)
        val_loss = validation(model, val, batch_size)
        if val_loss < best:
            best = val_loss
            best_model = {
                key: value.detach().to(device="cpu", copy=True)
                for key, value in model.state_dict().items()
            }
            bad = 0
        elif (epoch + 1) * n_steps > warmup:
            bad += 1
        early_stopped = patience is not None and bad >= patience
        logs.append(
            dict(
                epoch=epoch + 1,
                train_loss=total / len(train.items),
                validation_loss=val_loss,
                validation_items=len(val.items),
                bad_epochs=bad,
            )
        )
        temp = checkpoint.with_suffix(".tmp")
        torch.save(
            dict(
                binding=binding,
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                best=best,
                best_model=best_model,
                logs=logs,
            ),
            temp,
        )
        temp.replace(checkpoint)
        write(output / "progress.json", logs[-1])
    weights = output / "model.safetensors"
    temp = weights.with_suffix(".tmp")
    save_file(best_model, str(temp))
    temp.replace(weights)
    write(
        output / "bundle.json",
        dict(
            schema="drone-ai.new19.bundle/1",
            variant=name,
            scaler=scaler,
            model_sha256=digest(weights),
            checkpoint_sha256=digest(checkpoint),
        ),
    )
    result = dict(
        variant=name,
        status="early_stopped"
        if early_stopped
        else "complete"
        if stop == epochs
        else "paused_at_epoch",
        train_items=len(train.items),
        train_origin_interval_us=train.origin_interval_us,
        optimizer_steps=len(logs) * n_steps,
        best_epoch=min(logs, key=lambda row: row["validation_loss"])["epoch"],
        epochs=logs,
    )
    write(output / "result.json", result)
    return result


def lr_range(
    manifest, name, output, *, start_lr, end_lr, steps, batch_size, device="cpu"
):
    if not (0 < start_lr < end_lr < math.inf and steps >= 8 and batch_size > 0):
        raise ValueError(
            "Provide a finite ordered LR range, batch size and at least eight steps"
        )
    roles = load_split(manifest, ("normal_train",))
    task = VARIANTS[name].task
    train = training_batches(roles["normal_train"], task)
    torch.manual_seed(SEED)
    model = make_model(name).to(device)
    optimizer = adamw(model, start_lr)
    rng = np.random.default_rng(SEED)
    rows, ema, best = [], 0.0, math.inf
    status = "range_complete"
    for i, rate in enumerate(np.geomspace(start_lr, end_lr, steps)):
        ids = rng.choice(
            len(train.items),
            size=min(batch_size, len(train.items)),
            replace=False,
        )
        try:
            value = update(model, optimizer, train.batch(ids), float(rate))
        except FloatingPointError:
            status = "diverged"
            break
        ema = 0.9 * ema + 0.1 * value
        smooth = ema / (1 - 0.9 ** (i + 1))
        rows.append(dict(rate=float(rate), loss=value, ema=smooth))
        if i >= 7 and smooth > 4 * best:
            status = "diverged"
            break
        best = min(best, smooth)
    # The first batches stabilize the EMA and optimizer. Their slope is
    # dominated by initialization and must not nominate the peak rate.
    eligible = range(
        max(3, math.ceil(steps * 0.1)), len(rows) - 1 - (status == "diverged")
    )
    downhill = [i for i in eligible if rows[i]["ema"] < rows[i - 2]["ema"]]
    index = min(
        downhill,
        key=lambda i: rows[i]["ema"] - rows[i - 2]["ema"],
        default=None,
    )
    result = dict(
        variant=name,
        manifest_sha256=digest(manifest),
        status=status,
        train_items=len(train.items),
        train_origin_interval_us=train.origin_interval_us,
        rows=rows,
        candidate=None if index is None else rows[index]["rate"],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "lr-range.json", result)
    return result
