"""CPU prediction and independent before/after comparison on admitted flights."""

from pathlib import Path
from time import perf_counter

import numpy as np
from safetensors.torch import load_file

from .catalog import Catalog, digest, read_json
from .prepared import load, segment_ranges, windows
from .registry import fingerprint
from .runtime import Runtime


def predict(runtime, x):
    return np.concatenate(
        [runtime.predict_scaled(x[i : i + 16]) for i in range(0, len(x), 16)]
    )


def load_runtime(model_id, metadata, *, bundle, adapted=None):
    """Bind the same immutable weights/scaler to offline and live inputs."""
    runtime = Runtime(model_id, catalog=Catalog(bundle))
    adaptation = None
    if adapted:
        folder = Path(adapted)
        adaptation = read_json(folder / "result.json")
        if (
            adaptation["status"] != "experimental_adaptation_complete"
            or adaptation["base_model_id"] != model_id
        ):
            raise ValueError("Incomplete or mismatched adapted model")
        if digest(folder / "model.safetensors") != adaptation["model_sha256"]:
            raise ValueError("Adapted weights changed")
        if adaptation["binding"]["profile"] != metadata["profile"]:
            raise ValueError(
                "Adapted model is bound to a different profile version"
            )
        if adaptation["binding"]["origin"] != metadata["origin"]:
            raise ValueError("Adapted model and input flight origins differ")
        if (
            adaptation["binding"]["base_sha256"]
            != runtime.entry["weight_sha256"]
        ):
            raise ValueError("Base model changed")
        with np.load(folder / "scaler.npz", allow_pickle=False) as scaler:
            if set(scaler.files) != set(runtime.scaler) or any(
                not np.array_equal(runtime.scaler[k], scaler[k])
                for k in runtime.scaler
            ):
                raise ValueError("Adapted scaler changed")
        runtime.model.load_state_dict(
            load_file(str(folder / "model.safetensors"), device="cpu")
        )
    return runtime, adaptation


def analyze(model_id, manifest, *, bundle, adapted=None, data_root=None):
    started = perf_counter()
    metadata, values, times = load(manifest)
    runtime, adaptation = load_runtime(
        model_id, metadata, bundle=bundle, adapted=adapted
    )
    from .calibration import annotate, binding, forecast_scores, health_score

    s = runtime.scaler
    ranges = segment_ranges(metadata, len(values))
    segment_results = []
    if runtime.entry["task"] == "forecast":
        x, y, persistence, origins = windows(metadata, values, times, s)
        predicted = predict(runtime, x)
        scale = s["y_scale"].copy()
        scale[3] *= metadata["profile"]["series_cells"]
        errors = (y - predicted) * scale
        baseline_errors = (y - persistence[:, None, :]) * scale
        predicted_physical = predicted * s["y_scale"] + s["y_mean"]
        observed_physical = y * s["y_scale"] + s["y_mean"]
        predicted_physical[:, :, 3] *= metadata["profile"]["series_cells"]
        observed_physical[:, :, 3] *= metadata["profile"]["series_cells"]
        series = []
        for index, (a, b) in enumerate(segment_ranges(metadata, len(values))):
            mask = (origins >= times[a]) & (origins <= times[b - 1])
            series.append(
                dict(
                    segment=index,
                    horizon_us=1000000,
                    origin_us=origins[mask].tolist(),
                    time_us=(origins[mask] + 1000000).tolist(),
                    observed=observed_physical[mask, -1].tolist(),
                    predicted=predicted_physical[mask, -1].tolist(),
                    residual=errors[mask, -1].tolist(),
                )
            )
            segment_results.append(
                dict(
                    start_us=int(times[a]),
                    end_us=int(times[b - 1]) + 100000,
                    frames=b - a,
                    metrics=dict(
                        targets=[
                            "body_rate_p",
                            "body_rate_q",
                            "body_rate_r",
                            "pack_voltage",
                        ],
                        units=["rad/s", "rad/s", "rad/s", "V"],
                        windows=int(mask.sum()),
                        rmse=np.sqrt(
                            np.mean(errors[mask] ** 2, axis=(0, 1))
                        ).tolist(),
                        persistence_rmse=np.sqrt(
                            np.mean(baseline_errors[mask] ** 2, axis=(0, 1))
                        ).tolist(),
                    ),
                    scores=forecast_scores(
                        y[mask, -1], predicted[mask, -1], s
                    ).tolist(),
                    series=[series[-1]],
                )
            )
        scoring = dict(
            scores=forecast_scores(y[:, -1], predicted[:, -1], s).tolist(),
            time_us=(origins + 1000000).tolist(),
            unit="one_second_max_abs_standardized_residual",
        )
        result = dict(
            targets=[
                "body_rate_p",
                "body_rate_q",
                "body_rate_r",
                "pack_voltage",
            ],
            units=["rad/s", "rad/s", "rad/s", "V"],
            windows=len(x),
            rmse=np.sqrt(np.mean(errors**2, axis=(0, 1))).tolist(),
            persistence_rmse=np.sqrt(
                np.mean(baseline_errors**2, axis=(0, 1))
            ).tolist(),
            first_origin_us=int(origins[0]),
            last_origin_us=int(origins[-1]),
            series=series,
        )
    else:
        physical = values.copy()
        physical[:, 17] /= metadata["profile"]["series_cells"]
        physical[:, 18] /= metadata["profile"]["capacity_ah"]
        x = ((physical - s["x_mean"]) / s["x_scale"]).astype("f4")
        from learning.run_health19.scoring import components

        segments = []
        for a, b in segment_ranges(metadata, len(x)):
            predicted = runtime.predict_scaled(
                x[None, a:b], np.array([b - a], dtype="i8")
            )[0]
            segments.append(
                dict(
                    start_us=int(times[a]),
                    end_us=int(times[b - 1]) + 100000,
                    frames=b - a,
                    components=components(
                        predicted,
                        x[a:b, 17],
                        cell_voltage_scale=float(s["x_scale"][17]),
                    ),
                )
            )
        scoring = dict(
            scores=[
                health_score(row["components"], float(s["x_scale"][17]))
                for row in segments
            ],
            time_us=[row["end_us"] for row in segments],
            unit="segment_voltage_standardized_rmse",
        )
        result = dict(
            segments=segments, frames=len(x), whole_flight_verdict=False
        )
        for row, score in zip(segments, scoring["scores"]):
            c = row["components"]
            segment_results.append(
                dict(
                    start_us=row["start_us"],
                    end_us=row["end_us"],
                    frames=row["frames"],
                    metrics=dict(
                        samples=c["samples"],
                        short_cell_v=c["short_cell_v"],
                        long_cell_v=c["long_cell_v"],
                        pack_voltage_rmse=float(
                            np.hypot(c["short_cell_v"], c["long_cell_v"])
                            * metadata["profile"]["series_cells"]
                        ),
                    ),
                    scores=[score],
                )
            )
    from .run_summary import evidence

    output = dict(
        run_evidence=evidence(metadata, times, ranges),
        segment_results=segment_results,
        calibration_binding=binding(runtime, metadata, adaptation),
        scoring=scoring,
        model_id=model_id,
        adapted=adaptation is not None,
        profile_id=metadata["profile"]["id"],
        profile_version=fingerprint(metadata["profile"]),
        adapted_sha256=adaptation["model_sha256"] if adaptation else None,
        source_sha256=metadata["source_sha256"],
        role=metadata["role"],
        origin=metadata["origin"],
        metrics=result,
        health_verdict=None,
        calibration_status="not_calibrated_for_this_aircraft",
        timing=metadata.get("timing"),
        retained_segments=len(segment_ranges(metadata, len(values))),
        real_aircraft_validated=False,
        elapsed_seconds=perf_counter() - started,
        coverage=metadata.get("timing", {}).get("coverage", 1.0),
        normal_alarm_rate=None,
        alarm_rate_status="unavailable_without_aircraft_calibration",
    )

    if data_root:
        return annotate(output, metadata, data_root)
    for row in segment_results:
        row["decision"] = dict(
            status="unavailable", reason="기체별 보정값 없음"
        )
    return output


def analyze_run(models, manifests, *, bundle, data_root=None, use_active=False):
    """One source run, possibly multiple sensor-specific prepared packages."""
    from .registry import active_folder
    from .run_summary import aggregate

    if not models or len(set(models)) != len(models):
        raise ValueError("Select distinct models")
    if not 1 <= len(manifests) <= 64 or len(set(manifests)) != len(manifests):
        raise ValueError("Select 1..64 distinct prepared packages")
    loaded = [load(path) for path in manifests]
    identities = {
        (m["source_sha256"], m["profile"]["id"], m["origin"])
        for m, _, _ in loaded
    }
    if len(identities) != 1:
        raise ValueError("One source flight and aircraft identity required")
    corrections = [
        m.get("timing", {}).get("clock_corrections", {}) for m, _, _ in loaded
    ]
    if any(c != corrections[0] for c in corrections):
        raise ValueError("Use the same clock corrections for a combined run")
    spans = sorted(
        (int(t[a]), int(t[b - 1]) + 100000)
        for m, v, t in loaded
        for a, b in segment_ranges(m, len(v))
    )
    if any(b > c for (_, b), (c, _) in zip(spans, spans[1:])):
        raise ValueError(
            "Prepared intervals overlap; select each part of the run once"
        )
    if sum(len(v) for _, v, _ in loaded) > 72000:
        raise ValueError("Run summary input exceeds two hours")
    analyses = []
    for path, (m, _, _) in sorted(
        zip(manifests, loaded), key=lambda row: int(row[1][2][0])
    ):
        for mid in models:
            adapted = (
                active_folder(data_root, m["profile"], mid, bundle=bundle)
                if use_active
                else None
            )
            analyses.append(
                analyze(
                    mid,
                    path,
                    bundle=bundle,
                    data_root=data_root,
                    adapted=adapted,
                )
            )
    return dict(analyses=analyses, run_summary=aggregate(analyses))


def compare(adapted, manifest, *, bundle):
    adapted = Path(adapted)
    result = read_json(adapted / "result.json")
    metadata, _, _ = load(manifest)
    if metadata["role"] != "evaluation":
        raise ValueError("An independent evaluation flight is required")
    if metadata["source_sha256"] in result["binding"]["source_sha256"]:
        raise ValueError(
            "Evaluation flight was already used for adaptation or selection"
        )
    if metadata["origin"] != result["binding"]["origin"]:
        raise ValueError("Synthetic and real evaluation cannot be mixed")
    original = analyze(result["base_model_id"], manifest, bundle=bundle)
    updated = analyze(
        result["base_model_id"], manifest, bundle=bundle, adapted=adapted
    )
    return dict(
        original=original,
        adapted=updated,
        independent=True,
        detection_performance_verified=False,
        scope="Prediction comparison on one held-out flight; not anomaly sensitivity or health certification",
    )
