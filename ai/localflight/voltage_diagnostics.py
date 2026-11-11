"""Read-only voltage error and input sensitivity diagnostics for frozen models.

Perturbations replace selected normalized inputs with their training means.
They identify model sensitivity, not valid aircraft signals or corrected
predictions. This module never fits, calibrates, selects, or saves a model.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from capstone_ai.mapping import FEATURES

from .analysis import predict
from .catalog import DEFAULT_BUNDLE, Catalog, digest
from .prepared import load, segment_ranges, windows
from .runtime import Runtime

GROUPS = {
    "original": (),
    "motors": (0, 1, 2, 3),
    "current": (18,),
    "rate_error": (7, 8, 9),
    "other_motion": tuple(range(4, 17)),
    "non_voltage": tuple(i for i in range(19) if i != 17),
}


def error_metrics(predicted, observed, cells):
    """Separate signed bias and centered variation without correcting output."""
    error = (np.asarray(predicted, dtype="f8") - observed) * cells
    bias = float(error.mean())
    return dict(
        samples=int(error.size),
        rmse_pack_v=float(np.sqrt(np.mean(error**2))),
        bias_predicted_minus_observed_pack_v=bias,
        centered_rmse_pack_v=float(np.sqrt(np.mean((error - bias) ** 2))),
    )


def inputs(values, profile, scaler):
    """Return cell-V/C-rate inputs and per-feature training-scale distances."""
    physical = values.astype("f8", copy=True)
    physical[:, 17] /= profile["series_cells"]
    physical[:, 18] /= profile["capacity_ah"]
    normalized = ((physical - scaler["x_mean"]) / scaler["x_scale"]).astype(
        "f4"
    )
    return (
        physical,
        normalized,
        [
            dict(
                feature=name,
                median=float(np.median(physical[:, i])),
                train_mean=float(scaler["x_mean"][i]),
                train_scale=float(scaler["x_scale"][i]),
                abs_z_p95=float(np.quantile(np.abs(normalized[:, i]), 0.95)),
            )
            for i, name in enumerate(FEATURES)
        ],
    )


def _package(runtime, metadata, values, times):
    scaler, profile = runtime.scaler, metadata["profile"]
    cells = profile["series_cells"]
    physical, normalized, stats = inputs(values, profile, scaler)
    forecast = runtime.entry["task"] == "forecast"
    spans = segment_ranges(metadata, len(values))
    if forecast:
        x, y, hold, origins = windows(metadata, values, times, scaler)
        scale, mean = scaler["y_scale"][3], scaler["y_mean"][3]
        observed = y[:, :, 3] * scale + mean
        predicted = predict(runtime, x)[:, :, 3] * scale + mean
        baseline = np.broadcast_to(
            (hold[:, 3] * scale + mean)[:, None], observed.shape
        )
        # Select by time coverage, before inspecting any prediction error.
        selected = np.linspace(0, len(x) - 1, min(32, len(x)), dtype=int)
        blocks = [(x[selected], observed[selected])]
        indices = np.searchsorted(times, origins)
        expected = np.stack([physical[i + 1 : i + 11, 17] for i in indices])
        reconstruction_error = float(np.max(np.abs(expected - observed)))
        baseline_kind = "last_observation_at_forecast_origin"
    else:
        scale, mean = scaler["x_scale"][17], scaler["x_mean"][17]
        blocks = [
            (normalized[None, a:b], physical[a + 100 : b, 17]) for a, b in spans
        ]
        observed = np.concatenate([target for _, target in blocks])
        predicted = np.concatenate(
            [
                runtime.predict_scaled(x, np.array([x.shape[1]], dtype="i8"))[
                    0, 100:
                ]
                * scale
                + mean
                for x, _ in blocks
            ]
        )
        baseline = np.concatenate(
            [
                np.full(b - a - 100, np.median(physical[a : a + 100, 17]))
                for a, b in spans
            ]
        )
        reconstruction_error = 0.0
        baseline_kind = "initial_ten_second_median_hold_reference_only"

    variants = {}
    for label, columns in GROUPS.items():
        predictions, targets = [], []
        for x, target in blocks:
            changed = x.copy()
            changed[:, :, columns] = 0
            if forecast:
                output = runtime.predict_scaled(changed)[:, :, 3]
            else:
                output = runtime.predict_scaled(
                    changed, np.array([changed.shape[1]], dtype="i8")
                )[0, 100:]
            predictions.append((output * scale + mean).reshape(-1))
            targets.append(target.reshape(-1))
        variants[label] = error_metrics(
            np.concatenate(predictions), np.concatenate(targets), cells
        )
    return dict(
        full_error=error_metrics(predicted, observed, cells),
        baseline=error_metrics(baseline, observed, cells),
        baseline_kind=baseline_kind,
        sensitivity=variants,
        sensitivity_scope="32 evenly spaced windows maximum"
        if forecast
        else "all scored frames of each contiguous segment",
        target_reconstruction_max_abs_cell_v=reconstruction_error,
        input_statistics=stats,
    )


def diagnose(manifests, *, bundle=DEFAULT_BUNDLE):
    """Inspect up to 64 packages/72,000 frames without changing their roles."""
    if not 1 <= len(manifests) <= 64:
        raise ValueError("Select 1..64 prepared packages")
    if len({Path(path).resolve() for path in manifests}) != len(manifests):
        raise ValueError("Duplicate prepared package")
    packages = [(Path(path), load(path)) for path in manifests]
    if sum(len(values) for _, (_, values, _) in packages) > 72000:
        raise ValueError("Diagnostic input exceeds two hours")
    catalog, results = Catalog(bundle), []
    for entry in catalog.list():
        runtime = Runtime(entry["id"], catalog=catalog)
        for path, (metadata, values, times) in packages:
            results.append(
                dict(
                    model_id=entry["id"],
                    weight_sha256=runtime.entry["weight_sha256"],
                    scaler_sha256=digest(runtime.entry["files"]["scaler.npz"]),
                    package=path.parent.name,
                    manifest_sha256=digest(path),
                    source_sha256=metadata["source_sha256"],
                    health=metadata["health"],
                    role=metadata["role"],
                    series_cells=metadata["profile"]["series_cells"],
                    capacity_ah=metadata["profile"]["capacity_ah"],
                    frames=len(values),
                    **_package(runtime, metadata, values, times),
                )
            )
        del runtime
    return dict(
        schema="drone.voltage-diagnostics/1",
        scope="read-only; sensitivity is not a correction or causal proof",
        perturbation="selected normalized features replaced by training means",
        groups=GROUPS,
        fitting=False,
        healthy_labels_assumed=False,
        real_aircraft_validated=False,
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output must be a new evidence file")
    result = diagnose(args.manifests, bundle=args.bundle)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
