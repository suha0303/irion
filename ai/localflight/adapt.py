"""Finite CPU output-head adaptation; no threshold reuse or automatic deployment."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from capstone_ai.training import loss as objective

from .catalog import Catalog, digest
from .prepared import distinct_flights, load, segment_ranges, windows
from .runtime import Runtime


def write(path, row):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def health_segments(flight, scaler):
    metadata, values, _ = flight
    physical = values.copy()
    physical[:, 17] /= metadata["profile"]["series_cells"]
    physical[:, 18] /= metadata["profile"]["capacity_ah"]
    x = ((physical - scaler["x_mean"]) / scaler["x_scale"]).astype("f4")
    return [x[a:b] for a, b in segment_ranges(metadata, len(x))]


def health_objective(model, x):
    from learning.run_health19.models import normal_loss

    values = torch.from_numpy(x[None])
    lengths = torch.tensor([len(x)], dtype=torch.int64)
    return normal_loss(model(values, lengths), values[:, :, 17], lengths)


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
):
    if type(epochs) is not int or not 1 <= epochs <= 20 or not 0 < learning_rate <= 0.01:
        raise ValueError("Adaptation is bounded to 1..20 epochs and LR <= .01")
    if not 1 <= len(adaptation) <= 3 or len(development) > 1:
        raise ValueError("Use 1..3 adaptation flights and at most one development flight")
    runtime = Runtime(model_id, catalog=Catalog(bundle))
    health_task = runtime.entry["task"] == "run_health"
    flights = [load(p, normal=True) for p in [*adaptation, *development]]
    metadata = [f[0] for f in flights]
    distinct_flights(metadata)
    if any(m["role"] != "adaptation" for m in metadata[: len(adaptation)]) or any(
        m["role"] != "development" for m in metadata[len(adaptation) :]
    ):
        raise ValueError("Evaluation flights may not be used for fitting or model selection")
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
    train_y = None if health_task else np.concatenate([a[1] for a in arrays[: len(adaptation)]])
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
        device="cpu",
        torch=str(torch.__version__),
        threads=2,
        trainable="head",
        task=runtime.entry["task"],
        loss_policy="equal_flight_equal_segment_after_initial_100"
        if health_task
        else "forecast_objective",
        optimizer="SGD momentum=0 weight_decay=0",
        seed=0,
        code_sha256=digest(__file__),
        preprocessing_sha256=digest(Path(__file__).with_name("prepared.py")),
    )
    write(out / "plan.json", binding)
    model = runtime.model
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(model.head.parameters(), lr=learning_rate)
    history, start = [], 0
    best_loss = float("inf")
    if resume:
        folder = Path(resume)
        old = json.loads((folder / "checkpoint.json").read_text())
        if old["binding"] != binding or digest(folder / "last.safetensors") != old["last_sha256"]:
            raise ValueError("Checkpoint/input/plan mismatch")
        if digest(folder / "best.safetensors") != old["best_sha256"]:
            raise ValueError("Best checkpoint changed")
        model.load_state_dict(load_file(str(folder / "last.safetensors"), device="cpu"))
        best = load_file(str(folder / "best.safetensors"), device="cpu")
        start, history, best_loss = old["epoch"], old["history"], old["best_loss"]
    else:
        best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    end = epochs if stop_after_epoch is None else stop_after_epoch
    if type(end) is not int or not start < end <= epochs:
        raise ValueError("No valid remaining epoch range")
    for epoch in range(start, end):
        # The only trainable layers have no dropout; frozen encoders stay in inference behavior.
        model.eval()
        ids = np.random.default_rng(epoch).permutation(len(train_x))
        total = 0.0
        batch_size = 1 if health_task else 16
        for offset in range(0, len(ids), batch_size):
            if cancelled():
                write(out / "status.json", dict(status="cancelled", last_complete_epoch=epoch))
                return dict(status="cancelled", output=str(out))
            batch = ids[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            if health_task:
                # One optimizer step per flight, independent of segment count/duration.
                segments = train_x[int(batch[0])]
                flight_loss = 0.0
                for x in segments:
                    if cancelled():
                        write(
                            out / "status.json", dict(status="cancelled", last_complete_epoch=epoch)
                        )
                        return dict(status="cancelled", output=str(out))
                    loss = health_objective(model, x) / len(segments)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite adaptation loss")
                    loss.backward()
                    flight_loss += float(loss.detach())
                total += flight_loss
            else:
                loss = objective(
                    model(torch.from_numpy(train_x[batch])), torch.from_numpy(train_y[batch])
                )
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite adaptation loss")
                loss.backward()
                total += float(loss.detach()) * len(batch)
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        validation_loss = None
        if validation:
            with torch.inference_mode():
                if health_task:
                    segments = validation[0]
                    validation_loss = sum(
                        float(health_objective(model, x)) for x in segments
                    ) / len(segments)
                else:
                    vx, vy = validation[0][:2]
                    validation_loss = sum(
                        float(
                            objective(
                                model(torch.from_numpy(vx[i : i + 16])),
                                torch.from_numpy(vy[i : i + 16]),
                            )
                        )
                        * len(vx[i : i + 16])
                        for i in range(0, len(vx), 16)
                    ) / len(vx)
            if not np.isfinite(validation_loss):
                raise ValueError("Nonfinite development loss")
        criterion = validation_loss if validation_loss is not None else 0.0
        if not validation or criterion < best_loss:
            best_loss = criterion
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        row = dict(
            epoch=epoch + 1, train_loss=total / len(train_x), development_loss=validation_loss
        )
        history.append(row)
        for filename, state in (
            ("last.safetensors", model.state_dict()),
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
                last_sha256=digest(out / "last.safetensors"),
                best_sha256=digest(out / "best.safetensors"),
            ),
        )
        progress(row)
    # Calibration belongs to the new model; the base model's threshold is not carried over.
    save_file(best, str(out / "model.safetensors"))
    np.savez(out / "scaler.npz", **runtime.scaler)
    result = dict(
        status="experimental_adaptation_complete" if end == epochs else "paused_at_epoch",
        base_model_id=model_id,
        model_sha256=digest(out / "model.safetensors"),
        calibration_status="not_calibrated",
        health_verdict_enabled=False,
        independent_evaluation_completed=False,
        origin=metadata[0]["origin"],
        history=history,
        output=str(out),
        binding=binding,
    )
    write(out / "result.json", result)
    return result
