"""Train and inspect the eight new common19 models in separate bundles."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from capstone_ai.mapping import FEATURES

from . import (
    buffered,
    calibration,
    data,
    inference,
    scoring,
    training,
    transport,
)
from .models import VARIANTS


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--threads", type=int, default=2)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List the eight fresh model variants")
    prep = commands.add_parser(
        "prepare", help="Prepare one original SITL UDP flight"
    )
    for field in ("capture", "profile", "execution", "output"):
        prep.add_argument("--" + field, type=Path, required=True)
    for field in ("source-system", "source-component", "imu-id"):
        prep.add_argument("--" + field, type=int, required=True)
    prep.add_argument(
        "--buffered",
        action="store_true",
        help="Align PX4 samples with a 10-second hold before 10 Hz publication",
    )
    for command in ("fit", "lr-range"):
        operation = commands.add_parser(command)
        operation.add_argument("--model", choices=VARIANTS, required=True)
        operation.add_argument("--manifest", type=Path, required=True)
        operation.add_argument("--output", type=Path, required=True)
        operation.add_argument("--batch-size", type=int)
        if command == "fit":
            operation.add_argument("--lr", type=float, required=True)
            operation.add_argument("--epochs", type=int, required=True)
            operation.add_argument("--until-epoch", type=int)
            operation.add_argument("--resume", action="store_true")
        else:
            operation.add_argument("--start-lr", type=float, default=1e-6)
            operation.add_argument("--end-lr", type=float, default=1e-1)
            operation.add_argument("--steps", type=int, default=100)
    pred = commands.add_parser(
        "predict", help="Replay one prepared flight offline"
    )
    for field in ("bundle", "run", "output"):
        pred.add_argument("--" + field, type=Path, required=True)
    pred.add_argument("--calibration", type=Path)
    calibrate = commands.add_parser(
        "calibrate", help="Fit thresholds using named normal roles"
    )
    for field in ("bundle", "manifest", "output"):
        calibrate.add_argument("--" + field, type=Path, required=True)
    smoke = commands.add_parser(
        "smoke", help="Synthetic CPU fit/export/reload for all eight"
    )
    smoke.add_argument("--output", type=Path, required=True)
    return cli


def predict(args):
    runtime = inference.Runtime(args.bundle)
    run = data.load_run(args.run, args.run.stem)
    fitted = (
        None
        if args.calibration is None
        else calibration.load_calibration(args.bundle, args.calibration)
    )
    args.output.mkdir(parents=True, exist_ok=False)
    if runtime.model.variant.task == "inflight":
        predictions, count, residuals = inference.replay_forecasts(runtime, run)
        np.savez_compressed(
            args.output / "predictions.npz",
            time_us=run.time_us,
            predictions=predictions,
            residuals=residuals,
            contributors=count,
        )
        result = dict(
            status="available"
            if np.isfinite(residuals).any()
            else "unavailable",
            predictions=int((count > 0).sum()),
            matched=int(np.isfinite(residuals).all(1).sum()),
            timing="offline_replay",
        )
        if fitted is not None:
            reference = scoring.FlightReference(
                np.array(fitted["center"]),
                np.array(fitted["precision"]),
                fitted["voltage_scale"],
            )
            decisions = list(
                calibration.flight_decisions(
                    run, residuals, reference, fitted["selected"]["threshold"]
                )
            )
            data.write(args.output / "decisions.json", decisions)
            score_count = sum(row["status"] == "available" for row in decisions)
            result.update(
                alarm=any(row["alarm"] is True for row in decisions)
                if score_count
                else None,
                episodes=sum(row["episode_start"] for row in decisions),
                scored_targets=score_count,
            )
    else:
        predictions = runtime.voltage(run)
        record = (
            None
            if predictions is None
            else scoring.run_components(run, predictions)
        )
        result = {"status": "unavailable"}
        if record is not None:
            np.savez_compressed(
                args.output / "predictions.npz",
                time_us=run.time_us,
                predicted_cell_v=predictions,
                observed_voltage=run.observed[:, 3],
            )
            result = dict(
                status="available",
                short_cell_v=float(record[0][0]),
                long_cell_v=float(record[0][1]),
                coverage=record[1],
            )
        if fitted is not None:
            reference = scoring.RunReference(
                np.array(fitted["scale"]), fitted["threshold"]
            )
            result.update(reference.decide(record))
    data.write(args.output / "result.json", result)
    return result


def smoke(output):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    rows = []
    for n, role in enumerate(("normal_train", "normal_validation")):
        length = 112 + 9 * n
        values = rng.normal(0, 0.1, (length, 19)).astype(np.float32)
        values[:, 17] += 15
        path = output / f"{role}.npz"
        np.savez(
            path,
            values=values,
            time_us=np.arange(length, dtype=np.int64) * data.STEP_US,
            segment=np.zeros(length, dtype=np.int64),
            observed=np.ones((length, 4), dtype=bool),
        )
        sha = data.digest(path)
        rows.append(
            dict(
                run_id=role,
                role=role,
                path=path.name,
                sha256=sha,
                raw_sha256=f"synthetic-{sha}",
                group_id=role,
                retry_group=role,
                pair_id=None,
            )
        )
    manifest = output / "manifest.json"
    data.write(
        manifest,
        dict(
            schema="drone-ai.new19.dataset/1",
            features=FEATURES,
            fixture="synthetic",
            runs=rows,
        ),
    )
    results = {}
    run = data.load_run(output / "normal_validation.npz", "normal_validation")
    for name, variant in VARIANTS.items():
        bundle = output / name
        opts = dict(
            rate=1e-3,
            epochs=2,
            batch_size=16 if variant.task == "inflight" else 2,
        )
        training.fit(manifest, name, bundle, until_epoch=1, **opts)
        result = training.fit(manifest, name, bundle, resume=True, **opts)
        runtime = inference.Runtime(bundle)
        y = (
            runtime.forecast(run.values[None, :100])
            if variant.task == "inflight"
            else runtime.voltage(run)
        )
        if not np.isfinite(y).all():
            raise ValueError(f"Nonfinite restored prediction: {name}")
        results[name] = dict(
            shape=list(y.shape),
            optimizer_steps=result["optimizer_steps"],
            validation=result["epochs"][-1]["validation_loss"],
        )
    summary = dict(
        scope="synthetic_cpu", models=results, torch=str(torch.__version__)
    )
    data.write(output / "smoke.json", summary)
    return summary


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.threads < 1:
        cli.error("Thread count must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.command == "models":
        result = {name: vars(variant) for name, variant in VARIANTS.items()}
    elif args.command == "prepare":
        preprocessor = buffered if args.buffered else transport
        result = preprocessor.prepare(
            args.capture,
            args.profile,
            args.execution,
            args.output,
            source_system=args.source_system,
            source_component=args.source_component,
            imu_id=args.imu_id,
        )
    elif args.command == "predict":
        result = predict(args)
    elif args.command == "calibrate":
        result = calibration.calibrate(
            args.bundle,
            inference.Runtime(args.bundle),
            args.manifest,
            args.output,
        )
    elif args.command == "smoke":
        result = smoke(args.output)
    else:
        task = VARIANTS[args.model].task
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else (64 if task == "inflight" else 2)
        )
        if args.command == "fit":
            result = training.fit(
                args.manifest,
                args.model,
                args.output,
                rate=args.lr,
                epochs=args.epochs,
                batch_size=batch_size,
                resume=args.resume,
                until_epoch=args.until_epoch,
            )
        else:
            result = training.lr_range(
                args.manifest,
                args.model,
                args.output,
                start_lr=args.start_lr,
                end_lr=args.end_lr,
                steps=args.steps,
                batch_size=batch_size,
            )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
