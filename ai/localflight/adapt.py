"""Bounded local adaptation; no threshold reuse or automatic deployment."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from capstone_ai.training import loss as objective
from learning.run_health19 import models

from .adapt_support import (
    FrozenHeadCache,
    anchor_loss,
    cpu_state,
    device_details,
    parameter_names,
    peak_memory,
    set_trainable,
)
from .catalog import Catalog, digest
from .prepared import distinct_flights, load, segment_ranges, windows
from .runtime import Runtime


def write(path, row):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def health_segments(flight, scaler):
    metadata, values, _ = flight
    physical = values.copy()
    physical[:, 17] /= metadata["profile"]["series_cells"]
    physical[:, 18] /= metadata["profile"]["capacity_ah"]
    x = ((physical - scaler["x_mean"]) / scaler["x_scale"]).astype("f4")
    return [x[a:b] for a, b in segment_ranges(metadata, len(x))]


def health_objective(model, x, cache=None, key=None):
    device = next(model.parameters()).device
    values = torch.from_numpy(x[None]).to(device)
    lengths = torch.tensor([len(x)], dtype=torch.int64, device=device)
    predicted = (
        model(values, lengths)
        if cache is None
        else cache.prediction(values, [key], lengths)
    )
    return models.normal_loss(predicted, values[:, :, 17], lengths)


def fit(
    model_id,
    adaptation,
    development,
    output,
    *,
    bundle,
    epochs=5,
    learning_rate=0.001,
    cancelled=lambda: False,
    progress=lambda row: None,
    stop_after_epoch=None,
    resume=None,
    device="cpu",
    strategy="head",
    time_limit_seconds=1800,
    patience=5,
    anchor_weight=1.0,
):
    started = perf_counter()
    if (
        type(epochs) is not int
        or not 1 <= epochs <= 200
        or not np.isfinite(learning_rate)
        or not 0 < learning_rate <= 0.01
    ):
        raise ValueError("Adaptation is bounded to 1..200 epochs and LR <= .01")
    if (
        type(time_limit_seconds) not in (int, float)
        or not np.isfinite(time_limit_seconds)
        or not 0 < time_limit_seconds <= 3600
        or type(patience) is not int
        or not 1 <= patience <= 50
        or type(anchor_weight) not in (int, float)
        or not np.isfinite(anchor_weight)
        or not 0 <= anchor_weight <= 100
    ):
        raise ValueError("Invalid time budget, patience or anchor weight")
    if not 1 <= len(adaptation) <= 3 or len(development) > 1:
        raise ValueError(
            "Use 1..3 adaptation flights and at most one development flight"
        )
    runtime = Runtime(model_id, catalog=Catalog(bundle))
    permitted = parameter_names(runtime.model, strategy)
    if strategy != "head" and (not development or epochs < 3):
        raise ValueError(
            "Partial adaptation requires development flight and >=3 epochs"
        )
    hardware = device_details(device)
    health_task = runtime.entry["task"] == "run_health"
    flights = [load(p, normal=True) for p in [*adaptation, *development]]
    metadata = [f[0] for f in flights]
    distinct_flights(metadata)
    if any(
        m["role"] != "adaptation" for m in metadata[: len(adaptation)]
    ) or any(m["role"] != "development" for m in metadata[len(adaptation) :]):
        raise ValueError(
            "Evaluation flights may not be used for fitting or model selection"
        )
    arrays = (
        [health_segments(f, runtime.scaler) for f in flights]
        if health_task
        else [windows(*f, runtime.scaler) for f in flights]
    )
    train_x = (
        arrays[: len(adaptation)]
        if health_task
        else np.concatenate([a[0] for a in arrays[: len(adaptation)]])
    )
    train_y = (
        None
        if health_task
        else np.concatenate([a[1] for a in arrays[: len(adaptation)]])
    )
    if len(train_x) > 3000:
        raise ValueError("Adaptation limit is 3000 windows")
    validation = arrays[len(adaptation) :]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    binding = dict(
        base_model_id=model_id,
        base_sha256=runtime.entry["weight_sha256"],
        input_manifests=[digest(p) for p in [*adaptation, *development]],
        source_sha256=[m["source_sha256"] for m in metadata],
        profile=metadata[0]["profile"],
        origin=metadata[0]["origin"],
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        torch=str(torch.__version__),
        threads=2,
        trainable=strategy,
        trainable_names=sorted(permitted),
        time_limit_seconds=time_limit_seconds,
        patience=patience,
        anchor_weight=anchor_weight,
        promotion_policy="normal-and-anomaly-regression/1",
        task=runtime.entry["task"],
        loss_policy="equal_flight_equal_segment_after_initial_100"
        if health_task
        else "forecast_objective",
        optimizer="SGD momentum=0 weight_decay=0",
        seed=0,
        code_sha256=digest(__file__),
        support_sha256=digest(Path(__file__).with_name("adapt_support.py")),
        preprocessing_sha256=digest(Path(__file__).with_name("prepared.py")),
    )
    write(out / "plan.json", binding)
    model = runtime.model.to(device)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    trainable = set_trainable(model, strategy)
    optimizer = torch.optim.SGD(trainable, lr=learning_rate)
    history, start = [], 0
    best_loss = float("inf")
    if resume:
        folder = Path(resume)
        old = json.loads(
            (folder / "checkpoint.json").read_text(encoding="utf-8")
        )
        if (
            old["binding"] != binding
            or digest(folder / "last.safetensors") != old["last_sha256"]
        ):
            raise ValueError("Checkpoint/input/plan mismatch")
        if digest(folder / "best.safetensors") != old["best_sha256"]:
            raise ValueError("Best checkpoint changed")
        model.load_state_dict(
            load_file(str(folder / "last.safetensors"), device="cpu")
        )
        best = load_file(str(folder / "best.safetensors"), device="cpu")
        start, history, best_loss = (
            old["epoch"],
            old["history"],
            old["best_loss"],
        )
    else:
        best = cpu_state(model)
    end = epochs if stop_after_epoch is None else stop_after_epoch
    if type(end) is not int or not 1 <= end <= epochs or start > end:
        raise ValueError("No valid remaining epoch range")
    stale = old.get("stale_epochs", 0) if resume else 0
    stop_reason = "epochs_complete"
    completed_epoch = start
    head_cache = FrozenHeadCache(model)
    for epoch in range(start, end):
        if perf_counter() - started >= time_limit_seconds:
            stop_reason = "time_budget"
            break
        phase = "head" if strategy == "head" or epoch < 2 else strategy
        if phase != "head":
            head_cache.features.clear()
        trainable = set_trainable(model, phase)
        optimizer.param_groups[0]["lr"] = learning_rate * (
            1 if phase == "head" else 0.1
        )
        # These frozen architectures contain no dropout. cuDNN LSTM backward
        # requires training mode when backbone parameters are unfrozen.
        model.train(device == "cuda" and phase != "head")
        ids = np.random.default_rng(epoch).permutation(len(train_x))
        total = 0.0
        batch_size = 1 if health_task else 16
        for offset in range(0, len(ids), batch_size):
            if perf_counter() - started >= time_limit_seconds:
                stop_reason = "time_budget"
                break
            if cancelled():
                write(
                    out / "status.json",
                    dict(status="cancelled", last_complete_epoch=epoch),
                )
                return dict(status="cancelled", output=str(out))
            batch = ids[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            if health_task:
                # One optimizer step per flight, independent of segment
                # count/duration.
                segments = train_x[int(batch[0])]
                flight_loss = 0.0
                for segment_index, x in enumerate(segments):
                    if perf_counter() - started >= time_limit_seconds:
                        stop_reason = "time_budget"
                        break
                    if cancelled():
                        write(
                            out / "status.json",
                            dict(status="cancelled", last_complete_epoch=epoch),
                        )
                        return dict(status="cancelled", output=str(out))
                    loss = health_objective(
                        model,
                        x,
                        head_cache if phase == "head" else None,
                        (int(batch[0]), segment_index),
                    ) / len(segments)
                    loss = loss + anchor_weight * anchor_loss(
                        model, original
                    ) / len(segments)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite adaptation loss")
                    loss.backward()
                    flight_loss += float(loss.detach())
                if stop_reason == "time_budget":
                    break
                total += flight_loss
            else:
                values = torch.from_numpy(train_x[batch]).to(device)
                predicted = (
                    head_cache.prediction(values, batch.tolist())
                    if phase == "head"
                    else model(values)
                )
                loss = objective(
                    predicted,
                    torch.from_numpy(train_y[batch]).to(device),
                )
                loss = loss + anchor_weight * anchor_loss(model, original)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite adaptation loss")
                loss.backward()
                total += float(loss.detach()) * len(batch)
            torch.nn.utils.clip_grad_norm_(
                trainable, 1.0, error_if_nonfinite=True
            )
            optimizer.step()
        if stop_reason == "time_budget":
            # Never publish an incomplete epoch; best and last checkpoint
            # still refer to completed work.
            break
        validation_loss = None
        model.eval()
        if validation:
            with torch.inference_mode():
                if health_task:
                    segments = validation[0]
                    validation_loss = 0.0
                    for x in segments:
                        if perf_counter() - started >= time_limit_seconds:
                            stop_reason = "time_budget"
                            break
                        validation_loss += float(
                            health_objective(model, x)
                        ) / len(segments)
                else:
                    vx, vy = validation[0][:2]
                    validation_loss = 0.0
                    for i in range(0, len(vx), 16):
                        if perf_counter() - started >= time_limit_seconds:
                            stop_reason = "time_budget"
                            break
                        validation_loss += (
                            float(
                                objective(
                                    model(
                                        torch.from_numpy(vx[i : i + 16]).to(
                                            device
                                        )
                                    ),
                                    torch.from_numpy(vy[i : i + 16]).to(device),
                                )
                            )
                            * len(vx[i : i + 16])
                            / len(vx)
                        )
            if stop_reason == "time_budget":
                break
            if not np.isfinite(validation_loss):
                raise ValueError("Nonfinite development loss")
        criterion = validation_loss if validation_loss is not None else 0.0
        if not validation or criterion < best_loss:
            best_loss = criterion
            best = cpu_state(model)
            stale = 0
        else:
            stale += 1
        row = dict(
            epoch=epoch + 1,
            train_loss=total / len(train_x),
            development_loss=validation_loss,
            phase=phase,
            elapsed_seconds=perf_counter() - started,
            peak_memory_bytes=peak_memory(device),
        )
        history.append(row)
        for filename, state in (
            ("last.safetensors", cpu_state(model)),
            ("best.safetensors", best),
        ):
            temporary = out / (filename + ".tmp")
            save_file(state, str(temporary))
            temporary.replace(out / filename)
        write(
            out / "checkpoint.json",
            dict(
                binding=binding,
                epoch=epoch + 1,
                history=history,
                best_loss=best_loss,
                stale_epochs=stale,
                last_sha256=digest(out / "last.safetensors"),
                best_sha256=digest(out / "best.safetensors"),
            ),
        )
        progress(row)
        completed_epoch = epoch + 1
        if validation and stale >= patience and epoch >= 2:
            stop_reason = "early_stopping"
            break
    # Calibration belongs to the new model; the base model's threshold is not
    # carried over.
    save_file(best, str(out / "model.safetensors"))
    np.savez(out / "scaler.npz", **runtime.scaler)
    result = dict(
        status="experimental_adaptation_complete"
        if completed_epoch > 0
        and (end == epochs or stop_reason != "epochs_complete")
        else "paused_at_epoch",
        base_model_id=model_id,
        model_sha256=digest(out / "model.safetensors"),
        calibration_status="not_calibrated",
        health_verdict_enabled=False,
        independent_evaluation_completed=False,
        origin=metadata[0]["origin"],
        history=history,
        output=str(out),
        binding=binding,
        hardware=hardware,
        elapsed_seconds=perf_counter() - started,
        peak_memory_bytes=peak_memory(device),
        completed_epochs=completed_epoch,
        stop_reason=stop_reason,
        promotion_status="requires_normal_and_anomaly_review",
    )
    if completed_epoch == 0:
        result["status"] = "budget_exhausted_without_checkpoint"
    write(out / "result.json", result)
    return result
