"""Adapt the final voltage layer and assess whole-run scores offline."""

from collections import Counter

import numpy as np
import torch
from safetensors.torch import save_file

from .data import (
    STEP_US,
    digest,
    physical_features,
    read,
    whole_admitted,
    write,
)
from .models import PATCH, STRIDE
from .scoring import RunReference, run_components


def fit_head(runtime, bundle, logs, output, ridge):
    """Fit the final layer on post-initial observed voltage with one weight per source."""
    model = runtime.model.requires_grad_(False)
    head = model.head[-1]
    sources = {log.binding["source_sha256"] for log in logs}
    runs = [run for log in logs for run in log.runs if whole_admitted(run)]
    counts = Counter()
    for run in runs:
        counts[run.run_id] += int(run.observed[PATCH:, 3].sum())
    if set(counts) != sources:
        raise ValueError(
            "종료 후 모델에 필요한 초기 10초와 후속 전압 관측이 없는 학습 원본이 있습니다."
        )
    width = head.in_features + 1
    gram, rhs = np.zeros((width, width)), np.zeros(width)
    hidden = []
    hook = head.register_forward_pre_hook(
        lambda _module, args: hidden.append(args[0])
    )
    try:
        with torch.inference_mode():
            for run in runs:
                values = runtime.normalized(physical_features(run.values))
                size = len(values)
                prediction = model(values[None], torch.tensor([size]))[
                    0, :, 0
                ].numpy()
                features = np.zeros((size, width))
                overlap = np.zeros(size)
                for index, patch in enumerate(hidden.pop().numpy()):
                    start = index * STRIDE
                    count = min(PATCH, size - start)
                    features[start : start + count, :-1] += patch[:count]
                    overlap[start : start + count] += 1
                features[:, :-1] /= overlap[:, None]
                features[:, -1] = 1
                observed = run.observed[:, 3].copy()
                observed[:PATCH] = False
                z = features[observed]
                error = (values[:, 17].numpy() - prediction)[observed]
                gram += z.T @ z / counts[run.run_id]
                rhs += z.T @ error / counts[run.run_id]
    finally:
        hook.remove()
    update = np.linalg.solve(
        gram / len(sources) + ridge * np.eye(width), rhs / len(sources)
    )
    if not np.isfinite(update).all():
        raise ValueError("종료 후 모델의 출력층 보정값이 유한하지 않습니다.")
    with torch.no_grad():
        head.weight.add_(torch.from_numpy(update[:-1][None]))
        head.bias.add_(torch.from_numpy(update[-1:]))
    output.mkdir()
    weights = output / "model.safetensors"
    save_file(model.state_dict(), str(weights))
    meta = read(bundle / "bundle.json")
    meta.pop("checkpoint_sha256", None)
    meta.update(
        model_sha256=digest(weights),
        adaptation=dict(
            parent_bundle_sha256=digest(bundle / "bundle.json"),
            ridge=ridge,
            observed_training_frames=dict(counts),
        ),
    )
    write(output / "bundle.json", meta)


def calibrate(runtime, logs, quantile):
    """Use one maximum segment score per independent normal source."""
    by_source = {}
    for log in logs:
        records = by_source.setdefault(log.binding["source_sha256"], [])
        for run in log.runs:
            prediction = runtime.voltage(run)
            record = run_components(run, prediction)
            if record is not None:
                records.append(record[0])
    if any(not rows for rows in by_source.values()):
        raise ValueError("종료 후 점수를 산출할 수 없는 보정 원본이 있습니다.")
    means = [np.mean(np.square(rows), axis=0) for rows in by_source.values()]
    scale = np.maximum(np.sqrt(np.mean(means, axis=0)), 1e-8)
    maxima = [
        max(np.linalg.norm(rows / scale, axis=1)) for rows in by_source.values()
    ]
    return dict(
        scale=scale.tolist(),
        threshold=float(np.quantile(maxima, quantile, method="higher")),
        quantile=quantile,
        source_count=len(maxima),
    )


def evaluate(log, original, baseline, *, candidate=None, calibrated=None):
    """Compare ended-segment decisions and retain voltage series in pack volts."""
    models = {"original": original}
    params = {"original": ("original", baseline)}
    if candidate is not None:
        models["adapted"] = candidate
        params.update(
            x500=("adapted", baseline), calibrated=("adapted", calibrated)
        )
    rules = {
        key: (model, RunReference(np.asarray(p["scale"]), p["threshold"]))
        for key, (model, p) in params.items()
    }
    totals = dict.fromkeys([*models, "persistence"], 0.0)
    flags = {key: dict(available_segments=0, alarm_segments=0) for key in rules}
    compared = 0
    cells = log.metadata["profile"]["series_cells"]
    segments, series = [], []
    for run in log.runs:
        volts = run.values[:, 17] / 4
        mask = run.observed[:, 3].copy()
        mask[:PATCH] = False
        if not whole_admitted(run):
            mask[:] = False
        scored = int(mask.sum())
        compared += scored
        predictions = {
            name: runtime.voltage(run) for name, runtime in models.items()
        }
        predictions["persistence"] = (
            np.full(len(volts), volts[min(PATCH, len(volts)) - 1])
            if scored
            else None
        )
        observed = np.where(run.observed[:, 3], volts * cells, np.nan)
        value_series, residual_series = {}, {}
        for name, prediction in predictions.items():
            predicted = (
                np.where(mask, prediction * cells, np.nan)
                if prediction is not None
                else np.full(len(volts), np.nan)
            )
            residual = observed - predicted
            totals[name] += float(np.nansum(residual**2))
            value_series[name] = np.where(mask, predicted, None).tolist()
            residual_series[name] = np.where(mask, residual, None).tolist()
        decisions = {}
        for key, (name, reference) in rules.items():
            result = reference.decide(run_components(run, predictions[name]))
            decisions[key] = dict(**result, threshold=reference.threshold)
            flags[key]["available_segments"] += result["alarm"] is not None
            flags[key]["alarm_segments"] += result["alarm"] is True
        segments.append(
            dict(
                start_us=int(run.time_us[0]),
                end_us=int(run.time_us[-1]) + STEP_US,
                retained_frames=len(volts),
                scored_frames=scored,
                whole_decisions=decisions,
            )
        )
        series.append(
            dict(
                time_us=run.time_us.tolist(),
                observed=np.where(run.observed[:, 3], observed, None).tolist(),
                prediction=value_series,
                residual=residual_series,
            )
        )
    if not compared:
        raise ValueError("종료 후 점수를 산출할 수 있는 구간이 없습니다.")
    report = dict(
        **log.binding,
        scored_frames=compared,
        segments=segments,
        rmse={
            key: [float(np.sqrt(value / compared))]
            for key, value in totals.items()
        },
        whole_decisions=flags,
    )
    return report, dict(task="whole-run", segments=series)
