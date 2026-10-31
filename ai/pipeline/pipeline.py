"""Explicit R1 role wrappers. No test.npz reader and no event-truth evaluator."""

import json
import shutil
from pathlib import Path

import numpy as np

from capstone_ai.cli import process
from capstone_ai.dataset import feature_transform, target_transform
from capstone_ai.detection_protocol import fit_calibration
from capstone_ai.detection_replay import replay
from capstone_ai.mapping import FEATURES, TARGETS
from capstone_ai.training import Predictor, flat

from .control import CONFIG, FIT_ROLES, MODELS, read, sha, validate_config, write
from .fit import neural_fit


def scaler_from_train(frames, contexts, preprocess=None):
    """Caller supplies unique valid normal_train frames and once-per-origin contexts only."""
    x = feature_transform(frames, preprocess)
    c = np.asarray(contexts)
    if len(x) == 0 or len(c) == 0 or not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError("empty/nonfinite normal_train scaler input")
    scaler = {
        "x_mean": x.mean(0),
        "x_scale": np.maximum(x.std(0), 1e-5),
        "c_mean": c.mean(0),
        "c_scale": np.maximum(c.std(0), 1e-5),
    }
    indices = [FEATURES.index(n) for n in TARGETS]
    scaler.update(y_mean=scaler["x_mean"][indices], y_scale=scaler["x_scale"][indices])
    return scaler


def prepare(frozen, out, preprocess, budget, *, process_run=None):
    """Bounded 49-case pilot only; not a 100GB loader."""
    out = Path(out)
    out.mkdir(exist_ok=False)
    samples = {role: [] for role in FIT_ROLES}
    unique_frames, report = [], []
    for row in frozen["runs"]:
        budget.check()
        canonical = out / "canonical" / row["run_id"]
        result = (process_run or process)(row["path"], canonical, preprocess)
        if not result["closed"]:
            raise ValueError("unusable run retained; stop entire round: " + row["run_id"])
        role = row["role"]
        item = {
            "run_id": row["run_id"],
            "role": role,
            "summary": result,
            "excluded_targets": None,
            "accepted_targets": None,
        }
        if role in FIT_ROLES:
            if result["valid_frames"] == 0 or result["windows"] == 0:
                raise ValueError("required normal fit run lacks valid frames/windows")
            frames = [json.loads(s) for s in (canonical / "frames.jsonl").read_text().splitlines()]
            windows = [
                json.loads(s) for s in (canonical / "windows.jsonl").read_text().splitlines()
            ]
            lookup = {(f["clock_id"], f["segment_id"], f["time_us"]): f for f in frames}
            accepted = 0
            for w in windows:
                targets = [
                    lookup.get((w["clock_id"], w["segment_id"], t)) for t in w["target_times_us"]
                ]
                if any(
                    f is None or any(f["quality"][n] != "valid" for n in TARGETS) for f in targets
                ):
                    continue
                y = [[f["values"][n] for n in TARGETS] for f in targets]
                samples[role].append(
                    (w["X"], w["C"], y, row["run_id"], w["origin_us"], w["segment_id"])
                )
                accepted += 1
            if accepted == 0:
                raise ValueError("normal fit run has no arrived valid targets")
            item.update(excluded_targets=len(windows) - accepted, accepted_targets=accepted)
            if role == "normal_train":
                unique_frames.extend(
                    [
                        [f["values"][n] for n in FEATURES]
                        for f in frames
                        if f["operating_state"] == "airborne"
                        and all(f["quality"][n] == "valid" for n in FEATURES)
                    ]
                )
        report.append(item)
    for role, values in samples.items():
        x, c, y, ids, origins, segments = zip(*values)
        np.savez_compressed(
            out / (role + ".npz"),
            X=feature_transform(x, preprocess),
            C=np.array(c),
            Y=target_transform(y),
            run_id=np.array(ids),
            origin_us=np.array(origins),
            segment=np.array(segments),
        )
    scaler = scaler_from_train(unique_frames, [row[1] for row in samples["normal_train"]], preprocess)
    np.savez(out / "scaler.npz", **scaler)
    write(
        out / "manifest.json",
        {
            "runs": frozen["runs"],
            "audit": report,
            "scaler_fit": "normal_train unique valid airborne frames; context once per train origin",
            "unique_train_frames": len(unique_frames),
            "hashes": {p.name: sha(p) for p in out.glob("*.npz")},
        },
    )
    budget.check()
    return out


def normalized_role(dataset, role, scaler):
    if role not in FIT_ROLES:
        raise ValueError("only explicit normal fit roles may load fit arrays")
    manifest = read(Path(dataset) / "manifest.json")
    path = Path(dataset) / (role + ".npz")
    if sha(path) != manifest["hashes"][path.name]:
        raise ValueError("fit data changed")
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in ("run_id", "origin_us", "segment")}
        for key, scale in (("X", "x"), ("C", "c"), ("Y", "y")):
            result[key] = (data[key] - scaler[scale + "_mean"]) / scaler[scale + "_scale"]
            if not np.isfinite(result[key]).all():
                raise ValueError("nonfinite fit input")
    return result


def bundles(dataset, out, preprocess, budget, *, lstm_backend=None):
    validate_config(CONFIG)
    if lstm_backend != "aten_cuda":
        raise ValueError("explicit aten_cuda main policy required")
    neural_config = {**CONFIG, "lstm_backend": lstm_backend}
    out.mkdir(exist_ok=False)
    with np.load(dataset / "scaler.npz", allow_pickle=False) as archive:
        scaler = dict(archive)
    train, val = [
        normalized_role(dataset, role, scaler) for role in ("normal_train", "normal_validation")
    ]
    for name in MODELS:
        budget.check()
        folder = out / name
        folder.mkdir()
        history = []
        if name == "lstm":
            result = neural_fit(neural_config, train, val, folder, budget)
        elif name == "ridge_forecast":
            # Same predeclared dual-ridge solver and validation selection as training.py, on CPU.
            x, v = flat(train), flat(val)
            y = train["Y"].reshape(len(x), -1)
            best = float("inf")
            for alpha in (1.0, 10.0, 100.0):
                budget.check()
                coefficients = x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y)
                error = ((v @ coefficients).reshape(val["Y"].shape) - val["Y"]) ** 2
                score = 0.5 * error[..., :3].mean() + 0.5 * error[..., 3].mean()
                if not np.isfinite(score):
                    raise ValueError("nonfinite ridge validation")
                history.append({"alpha": alpha, "validation_loss": float(score)})
                if score < best:
                    best, selected, chosen = score, coefficients, alpha
            np.savez(folder / "ridge.npz", coefficients=selected, alpha=chosen)
            result = {"history": history, "device": "cpu"}
        else:
            result = {"history": [], "device": "cpu", "fit": "none"}
        shutil.copyfile(dataset / "scaler.npz", folder / "scaler.npz")
        shutil.copyfile(dataset / "manifest.json", folder / "training_manifest.json")
        write(
            folder / "model.json",
            {
                "name": name,
                "hidden": 32,
                "task": "forecast",
                "target_order": TARGETS,
                "feature_order": FEATURES,
                "config": neural_config if name == "lstm" else CONFIG,
            },
        )
        write(folder / "preprocessing.json", preprocess)
        write(
            folder / "forecast_contract.json",
            {
                "schema": "drone.forecast/3.0",
                "history": 100,
                "horizon": 10,
                "step_us": 100000,
                "cadence_us": 1000000,
                "require_issued_before_first_target": True,
                "teacher_forcing": False,
            },
        )
        write(folder / "training.json", result)
        write(folder / "integrity.json", {p.name: sha(p) for p in folder.iterdir() if p.is_file()})


def calibrate_and_seal(dataset, models, out, budget):
    """One normal calibration per model, then development inference; never joins truth."""
    out.mkdir(exist_ok=False)
    rows = read(dataset / "manifest.json")["runs"]
    groups = {r["run_id"]: r["paired_case_id"] for r in rows}
    seals = {}
    for name in MODELS:
        budget.check()
        predictor = Predictor(models / name, device="cpu")
        val, cal = [
            normalized_role(dataset, role, predictor.scaler)
            for role in ("normal_validation", "normal_calibration")
        ]
        vr = val["Y"] - predictor.predict_scaled(val["X"], val["C"])
        cr = cal["Y"] - predictor.predict_scaled(cal["X"], cal["C"])
        physical = predictor.scaler["y_scale"].copy()
        physical[-1] *= 4
        fitted = fit_calibration(
            vr, cr, cal["run_id"], cal["segment"], cal["origin_us"], physical, groups
        )
        fitted.update(bundle_integrity_sha256=sha(models / name / "integrity.json"))
        folder = out / name
        folder.mkdir()
        write(folder / "calibration.json", fitted)
        for row in rows:
            if row["role"] not in {"normal_dev_eval", "anomaly_dev"}:
                continue
            budget.check()
            target = folder / row["run_id"]
            seal_development(dataset / "canonical" / row["run_id"], models / name, fitted, target)
            seals[str(target.relative_to(out) / "seal.json")] = sha(target / "seal.json")
    write(out / "all-forecast-seals.json", seals)
    budget.check()


def seal_development(canonical, bundle, calibration, target, *, predictor=None):
    """Zero airborne exposure is an unavailable case, never a dropped case or healthy zero."""
    frames = [json.loads(s) for s in (canonical / "frames.jsonl").read_text().splitlines()]
    if any(f["operating_state"] == "airborne" for f in frames):
        # Existing replay handles zero windows with positive airborne exposure.
        if predictor is None:
            return replay(canonical, bundle, calibration, target)
        return replay(canonical, bundle, calibration, target, predictor=predictor)
    summary = read(canonical / "summary.json")
    target.mkdir(parents=True, exist_ok=False)
    for name in ("forecasts.jsonl", "decisions.jsonl"):
        (target / name).write_text("")
    result = {
        "status": "insufficient_data",
        "reason": "no_observed_airborne_exposure",
        "operational_seconds": 0,
        "valid_decisions": 0,
        "forecast_count": 0,
        "scored_seconds": 0,
        "coverage": None,
        "episodes_per_hour": None,
        "excluded_case": False,
        "truth_join": "coordinator_pending",
        "coordinator_action": "retain case/event denominators as unavailable; do not call positive-exposure evaluation_metrics",
        "preprocessing_exclusions": summary["exclusions"],
    }
    write(target / "summary.json", result)
    write(
        target / "seal.json",
        {
            "files": {n: sha(target / n) for n in ("forecasts.jsonl", "decisions.jsonl")},
            "canonical": summary["output_hashes"],
            "bundle_integrity_sha256": sha(bundle / "integrity.json"),
            "execution": "sealed_unavailable_before_evaluation_truth_join",
        },
    )
    return result
