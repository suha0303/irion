"""Compare one preserved normal-validation flight with delivered GPU outputs.

This is a portability diagnostic, never a model-selection or final-test rerun.
User policy 2026-11-14: cross-backend numerical deltas are informational, not a
release gate. Retain the historical allclose statistic for traceability.
"""

import json
from pathlib import Path

import numpy as np

from capstone_ai.detection_protocol import score_sequence
from learning.run_health19.scoring import Calibration, components

from .catalog import Catalog, digest, read_json
from .runtime import Runtime


def check(bundle, dataset):
    catalog, dataset = Catalog(bundle), Path(dataset)
    with (dataset / "online-index.jsonl").open(encoding="utf-8") as f:
        reference = next(
            r for r in map(json.loads, f) if r["role"] == "normal_validation"
        )
    manifest_path = Path(reference["manifest"]["path"])
    if digest(manifest_path) != reference["manifest"]["sha256"]:
        raise ValueError("Validation metadata changed")
    metadata = read_json(manifest_path)
    source = manifest_path.parent / metadata["file"]["path"]
    if digest(source) != metadata["file"]["sha256"]:
        raise ValueError("Validation input changed")
    rid = metadata["run"]["run_id"]
    with np.load(source, allow_pickle=False) as z:
        online = dict(z)
    whole_ref = None
    for p in sorted(dataset.glob("run-health19-[0-9][0-9][0-9].json")):
        for r in read_json(p)["runs"]:
            if r["run_id"] == rid:
                whole_ref = r
    if whole_ref is None:
        raise ValueError("Missing matching full-run validation input")
    source = dataset / whole_ref["path"]
    if digest(source) != whole_ref["sha256"]:
        raise ValueError("Full-run validation input changed")
    with np.load(source, allow_pickle=False) as z:
        values = z["values"].copy()
    rows = []
    for entry in catalog.list():
        runtime = Runtime(entry["id"], catalog=catalog)
        s, params = runtime.scaler, runtime.entry["calibration"]
        expected_path = catalog.checked(
            f"models/{entry['id']}/evaluation/predictions/{rid}.npz"
        )
        with np.load(expected_path, allow_pickle=False) as z:
            reference_values = dict(z)
        if entry["task"] == "forecast":
            x = ((online["X"] - s["x_mean"]) / s["x_scale"]).astype("f4")
            actual = np.concatenate(
                [
                    runtime.predict_scaled(x[i : i + 32])
                    for i in range(0, len(x), 32)
                ]
            )
            expected = reference_values["prediction_scaled"]
            y = ((online["Y"] - s["y_mean"]) / s["y_scale"]).astype("f4")

            def decisions(pred, target, params):
                errors = (target - pred) ** 2 / np.array(
                    params["reference_error_scale_scaled"]
                )
                scores = np.stack(
                    [errors[:, :, :3].mean((1, 2)), errors[:, :, 3].mean(1)],
                    axis=1,
                )
                return score_sequence(
                    scores,
                    np.repeat(rid, len(scores)),
                    online["segment_id"],
                    online["origin_us"],
                    params["high"],
                )

            a = decisions(actual, y, params)
            b = decisions(expected, y, params)
            alarms_equal = all(
                (i["alarm_active"], i["episode_start"])
                == (j["alarm_active"], j["episode_start"])
                for i, j in zip(a, b)
            )
        else:
            physical = values.copy()
            physical[:, 17] /= 4
            physical[:, 18] /= 12
            x = ((physical - s["x_mean"]) / s["x_scale"]).astype("f4")
            pred = runtime.predict_scaled(
                x[None], np.array([len(x)], dtype="i8")
            )[0]
            actual = pred * s["x_scale"][17] + s["x_mean"][17]
            expected = reference_values["predicted_cell_v"]
            threshold = Calibration(**params)

            def alarm(p, threshold, cell_voltage):
                return threshold.decide(
                    components(p, cell_voltage, cell_voltage_scale=1)
                )["alarm"]

            alarms_equal = alarm(actual, threshold, physical[:, 17]) == alarm(
                expected, threshold, physical[:, 17]
            )
        rows.append(
            dict(
                id=entry["id"],
                max_abs=float(np.max(np.abs(actual - expected))),
                allclose=bool(
                    np.allclose(actual, expected, atol=1e-6, rtol=1e-5)
                ),
                alarm_equal=alarms_equal,
                outputs=int(actual.size),
            )
        )
    return dict(
        scope=(
            "one preserved normal-validation flight; CPU portability "
            "diagnostics only"
        ),
        run_id=rid,
        atol=1e-6,
        rtol=1e-5,
        models=rows,
        final_test_used=False,
        numeric_tolerance_is_gate=False,
        policy="user_20261114_numeric_deltas_informational",
        status="CPU_EXECUTION_COMPLETE_WITH_NUMERIC_DIAGNOSTICS",
    )
