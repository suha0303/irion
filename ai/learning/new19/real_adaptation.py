"""Evaluate prepared flight logs and build isolated V0.2 candidates."""

from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from capstone_ai.mapping import TARGETS
from localflight.prepared import load, observed_targets, segment_ranges
from localflight.run_summary import evidence, partition

from . import real_whole
from .app_candidates import PROFILE_KEYS, selected_paths
from .calibration import decisions_from_scores, load_calibration
from .data import STEP_US, TARGET_INDICES, Batches, Run, digest, read, write
from .inference import Runtime, replay_forecasts
from .models import VARIANTS
from .scoring import FlightReference

MODEL = "inflight-lstm-128x4"
ROLES = ("adaptation", "calibration", "evaluation")
RIDGE = 0.01
QUANTILE = 0.99


@dataclass
class InputLog:
    """One selected prepared package and its source-bound continuous runs."""

    metadata: dict
    binding: dict
    runs: list[Run]
    coverage: dict


def read_prepared(path, *, normal=False, day=None):
    """Load recorded segments in the model's voltage and current convention."""
    manifest = Path(path)
    before = digest(manifest)
    meta, values, times = load(manifest, normal=normal)
    raw = meta["source_sha256"]
    profile = meta["profile"]
    values[:, 17] *= 4 / profile["series_cells"]
    values[:, 18] *= 12 / profile["capacity_ah"]
    mask = np.column_stack(
        [observed_targets(meta, times, (target,)) for target in TARGETS]
    )
    ranges = segment_ranges(meta, len(times))
    runs = [
        Run(
            raw,
            values[start:end],
            times[start:end],
            np.zeros(end - start, dtype="i8"),
            mask[start:end],
        )
        for start, end in ranges
    ]
    if digest(manifest) != before:
        raise ValueError("작업 준비 중 비행 매니페스트가 바뀌었습니다.")
    binding = dict(
        filename=meta.get("run_context", {}).get("filename") or raw[:8],
        date=day,
        role=meta["role"],
        source_sha256=raw,
        prepared_sha256=before,
        health=meta["health"],
    )
    bounds = evidence(meta, times, ranges)
    timeline = partition(**bounds)
    coverage = dict(
        context=bounds["context"],
        analyzed_us=sum(
            row["end_us"] - row["start_us"]
            for row in timeline
            if row["reason"] == "analyzed"
        ),
        retained_frames=len(times),
        timeline=timeline,
    )
    return InputLog(meta, binding, runs, coverage)


def load_inputs(selections):
    """Check role, health, date, profile and original recording overlap."""
    if set(selections) != set(ROLES):
        raise ValueError("학습·보정·평가 자료를 각각 선택하세요.")
    sources, days, spans = {}, {}, {}
    expected = None
    by_role = {}
    for role in ROLES:
        choice = selections[role]
        if not choice:
            raise ValueError("학습·보정·평가 자료를 각각 선택하세요.")
        inputs = by_role[role] = []
        for item in choice:
            day = item["date"]
            if date.fromisoformat(day).isoformat() != day:
                raise ValueError("비행 날짜를 YYYY-MM-DD 형식으로 입력하세요.")
            log = read_prepared(
                item["manifest"], normal=role != "evaluation", day=day
            )
            meta = log.metadata
            if meta["role"] != role:
                raise ValueError(
                    "준비 구간에 기록된 역할과 선택한 역할이 다릅니다."
                )
            allowed = {"confirmed_normal", "abnormal"}
            if (
                meta["health"] not in allowed
                or not meta.get("health_evidence", "").strip()
            ):
                raise ValueError(
                    "평가에도 정상 또는 이상 확인 근거를 기록하세요."
                )
            signature = (
                tuple(meta["profile"].get(key) for key in PROFILE_KEYS),
                meta["origin"],
                meta.get("live_input"),
                meta["preprocessing_revision"],
                meta.get("timing", {}).get("mode"),
                meta.get("timing", {}).get("fill_policy"),
            )
            if expected is None:
                expected = signature
            if signature != expected:
                raise ValueError(
                    "한 작업에서는 같은 기체 프로파일과 입력 방식만 선택하세요."
                )
            raw = meta["source_sha256"]
            if sources.setdefault(raw, (role, day)) != (role, day):
                raise ValueError(
                    "같은 원본에는 하나의 날짜와 역할만 지정하세요."
                )
            if days.setdefault(day, role) != role:
                raise ValueError("같은 비행 날짜의 자료는 한 역할에 묶으세요.")

            prior = spans.setdefault(raw, [])
            for run in log.runs:
                lo, hi = int(run.time_us[0]), int(run.time_us[-1])
                if any(lo <= stop and begin <= hi for begin, stop in prior):
                    raise ValueError(
                        "같은 원본에서 겹치는 준비 구간을 선택했습니다."
                    )
                prior.append((lo, hi))
            inputs.append(log)
    return by_role


def load_original(root, model_id):
    base = Path(root) / model_id
    bundle = base / "fit"
    baseline_path = base / "calibration/calibration.json"
    baseline = load_calibration(bundle, baseline_path)
    runtime = Runtime(bundle)
    if runtime.model.variant.name != model_id:
        raise ValueError("선택한 모델과 번들의 모델이 다릅니다.")
    return runtime, bundle, baseline_path, baseline


def fit_head(bundle, logs, output):
    """Fit a file-balanced ridge correction to the frozen model's linear head."""
    runtime = Runtime(bundle)
    model = runtime.model.requires_grad_(False)
    runs = [run for log in logs for run in log.runs]
    windows = Batches(
        runs, runtime.scaler, "inflight", origin_interval_us=1_000_000
    )
    owners = [runs[index].run_id for index, _ in windows.items]
    counts = Counter(owners)
    if set(counts) != {run.run_id for run in runs}:
        raise ValueError("학습에 필요한 연속 관측 창이 없는 원본이 있습니다.")
    width = model.head.in_features + 1
    gram, rhs, weight_total = (
        np.zeros((width, width)),
        np.zeros((width, 4)),
        0.0,
    )
    hidden = []
    hook = model.head.register_forward_pre_hook(
        lambda _module, args: hidden.append(args[0])
    )
    try:
        with torch.inference_mode():
            for start in range(0, len(owners), 128):
                stop = min(start + 128, len(owners))
                x, target, _ = windows.batch(range(start, stop))
                error = (target - model(x)).numpy().reshape(-1, 4)
                z = np.column_stack(
                    (
                        hidden.pop().numpy().reshape(-1, width - 1),
                        np.ones(len(error)),
                    )
                )
                weight = np.repeat(
                    [1 / counts[label] for label in owners[start:stop]],
                    target.shape[1],
                )
                scaled = z.T * weight
                gram += scaled @ z
                rhs += scaled @ error
                weight_total += weight.sum()
    finally:
        hook.remove()
    update = np.linalg.solve(
        gram / weight_total + RIDGE * np.eye(width), rhs / weight_total
    )
    if not np.isfinite(update).all():
        raise ValueError("출력층 보정값이 유한하지 않습니다.")
    with torch.no_grad():
        model.head.weight.add_(torch.from_numpy(update[:-1].T))
        model.head.bias.add_(torch.from_numpy(update[-1]))
    output.mkdir()
    weights = output / "model.safetensors"
    save_file(model.state_dict(), str(weights))
    original = read(Path(bundle) / "bundle.json")
    original.pop("checkpoint_sha256", None)
    original.update(
        model_sha256=digest(weights),
        adaptation=dict(
            parent_bundle_sha256=digest(Path(bundle) / "bundle.json"),
            ridge=RIDGE,
            train_windows=dict(counts),
        ),
    )
    write(output / "bundle.json", original)
    return Runtime(output)


def reference(parameters):
    return FlightReference(
        np.asarray(parameters["center"]),
        np.asarray(parameters["precision"]),
        parameters["voltage_scale"],
    )


def calibration(runtime, logs):
    errors = []
    for log in logs:
        for run in log.runs:
            residuals = replay_forecasts(runtime, run, batch_size=256)[2]
            valid = residuals[np.isfinite(residuals).all(axis=1)]
            if not len(valid):
                raise ValueError("보정 구간에 유효한 예측·실측 쌍이 없습니다.")
            errors.append(valid)
    residuals = np.concatenate(errors)
    ref = FlightReference.fit(residuals)
    threshold = np.maximum(
        np.quantile(
            ref.score_series(residuals), QUANTILE, axis=0, method="higher"
        ),
        1e-10,
    )
    return dict(
        center=ref.center.tolist(),
        precision=ref.precision.tolist(),
        voltage_scale=ref.voltage_scale,
        scored_frames=len(residuals),
        selected=dict(quantile=QUANTILE, threshold=threshold.tolist()),
    )


class Persistence:
    """Predict the last observed target at each future lead."""

    @staticmethod
    def forecast(values):
        return np.repeat(values[:, -1:, TARGET_INDICES], 10, axis=1)


def nullable(values):
    return np.where(np.isfinite(values), values, None).tolist()


def evaluate(log, original, baseline, *, candidate=None, calibrated=None):
    """Collect comparable metrics and graph data without crossing a segment."""
    runtimes = {"original": original, "persistence": Persistence()}
    rules = {"original": ("original", baseline)}
    if candidate is not None:
        runtimes["adapted"] = candidate
        rules.update(
            x500=("adapted", baseline), calibrated=("adapted", calibrated)
        )
    totals = {key: np.zeros(4) for key in runtimes}
    flags = {key: dict(episode_starts_us=[], alarm_frames=0) for key in rules}
    compared = 0
    physical = np.array([1, 1, 1, log.metadata["profile"]["series_cells"] / 4])
    series, segments = [], []
    for run in log.runs:
        forecasts = {
            name: replay_forecasts(runtime, run, batch_size=128)
            for name, runtime in runtimes.items()
        }
        mask = np.isfinite(forecasts["original"][2]).all(axis=1)
        scored = int(mask.sum())
        compared += scored
        for key, (_, _, loss) in forecasts.items():
            if not np.array_equal(np.isfinite(loss).all(axis=1), mask):
                raise ValueError(
                    "모델 간 판정 대상 프레임이 일치하지 않습니다."
                )
            totals[key] += ((loss[mask] * physical) ** 2).sum(axis=0)
        decisions, segment_alarms = {}, {}
        for key, (name, params) in rules.items():
            threshold = params["selected"]["threshold"]
            scores = reference(params).score_series(forecasts[name][2])
            rows = list(decisions_from_scores(run, scores, threshold))
            alarm_frames = sum(row["alarm"] is True for row in rows)
            flags[key]["alarm_frames"] += alarm_frames
            flags[key]["episode_starts_us"].extend(
                row["time_us"] for row in rows if row["episode_start"]
            )
            decisions[key] = dict(
                threshold=threshold,
                raw_score=nullable(scores),
                ema=[row.get("ema", [None, None]) for row in rows],
                domains=[row.get("domains", [None, None]) for row in rows],
            )
            segment_alarms[key] = alarm_frames
        segments.append(
            dict(
                start_us=int(run.time_us[0]),
                end_us=int(run.time_us[-1]) + STEP_US,
                retained_frames=len(run.time_us),
                scored_frames=scored,
                alarm_frames=segment_alarms,
            )
        )
        series.append(
            dict(
                time_us=run.time_us.tolist(),
                observed=nullable(
                    np.where(
                        run.observed,
                        run.values[:, TARGET_INDICES] * physical,
                        np.nan,
                    )
                ),
                prediction={
                    key: nullable(values * physical)
                    for key, (values, _, _) in forecasts.items()
                },
                residual={
                    key: nullable(loss * physical)
                    for key, (_, _, loss) in forecasts.items()
                },
                rules=decisions,
            )
        )
    if not compared:
        raise ValueError("평가 구간에 유효한 예측·실측 쌍이 없습니다.")
    report = dict(
        **log.binding,
        scored_frames=compared,
        segments=segments,
        rmse={
            name: np.sqrt(total / compared).tolist()
            for name, total in totals.items()
        },
        decisions={
            key: dict(
                value,
                episodes=len(value["episode_starts_us"]),
                alarm_seconds=value["alarm_frames"] * STEP_US / 1_000_000,
                alarm_percent=100 * value["alarm_frames"] / compared,
            )
            for key, value in flags.items()
        },
    )
    return report, dict(task="inflight", segments=series)


def train_candidate(model_root, model_id, inputs, output):
    """Write one candidate and its evaluation for the same split."""
    original, bundle, baseline_path, baseline = load_original(
        model_root, model_id
    )
    selected = {
        role: [log.binding for log in logs] for role, logs in inputs.items()
    }
    profile = inputs["adaptation"][0].metadata["profile"]
    output.mkdir(parents=True, exist_ok=False)
    split = output / "split.json"
    write(
        split,
        dict(
            profile=profile,
            input_origin=inputs["adaptation"][0].metadata["origin"],
            logs=selected,
            original_bundle_sha256=digest(bundle / "bundle.json"),
            original_calibration_sha256=digest(baseline_path),
        ),
    )
    candidate_dir = output / "candidate"
    task = original.model.variant.task
    if task == "whole-run":
        real_whole.fit_head(
            Runtime(bundle), bundle, inputs["adaptation"], candidate_dir, RIDGE
        )
        candidate = Runtime(candidate_dir)
        calibrated = real_whole.calibrate(
            candidate, inputs["calibration"], QUANTILE
        )
        assess = real_whole.evaluate
    else:
        candidate = fit_head(bundle, inputs["adaptation"], candidate_dir)
        calibrated = calibration(candidate, inputs["calibration"])
        assess = evaluate
    write(
        output / "calibration.json",
        dict(
            schema="drone-ai.new19.calibration/1",
            bundle_sha256=digest(candidate_dir / "bundle.json"),
            manifest_sha256=digest(split),
            parameters=calibrated,
        ),
    )
    reports = []
    for index, log in enumerate(inputs["evaluation"]):
        report, chart = assess(
            log, original, baseline, candidate=candidate, calibrated=calibrated
        )
        write(output / f"{index}.json", chart)
        reports.append(report)
    state = dict(
        model_id=model_id,
        task=task,
        profile_id=profile["id"],
        profile_name=profile["name"],
        input_origin=inputs["adaptation"][0].metadata["origin"],
        output=str(output),
        active=False,
        review_status="pending",
        anomaly_evidence=any(r["health"] == "abnormal" for r in reports),
        targets=TARGETS,
        counts={
            key: len({r["source_sha256"] for r in rows})
            for key, rows in selected.items()
        },
        evaluation=reports,
    )
    state["thresholds"] = (
        [calibrated["threshold"]]
        if task == "whole-run"
        else calibrated["selected"]["threshold"]
    )
    write(output / "result.json", state)
    return state


def run(request):
    """Read the historical single-model worker contract."""
    torch.set_num_threads(2)
    return train_candidate(
        request["model_root"],
        MODEL,
        load_inputs(request["logs"]),
        Path(request["output"]),
    )


def execute_v02(request):
    """Process the chosen bundles serially and isolate outputs by model."""
    models = request["models"]
    if (
        not models
        or len(models) != len(set(models))
        or any(model not in VARIANTS for model in models)
    ):
        raise ValueError("사용 가능한 V0.2 모델을 선택하세요.")
    torch.set_num_threads(2)
    analysis = request["action"] == "v02_analyze"
    if analysis:
        if request.get("candidate_job") and not set(models).issubset(
            request.get("candidate_models", {})
        ):
            raise ValueError(
                "선택한 후보에 모델과 전용 보정 기준이 모두 필요합니다."
            )
        log = read_prepared(request["manifest"])
    else:
        inputs = load_inputs(request["logs"])
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    results, failures = [], {}
    for model_id in models:
        write(
            output / "progress.json",
            dict(
                model_id=model_id,
                finished=len(results) + len(failures),
                total=len(models),
            ),
        )
        model_output = output / model_id
        try:
            model_root = request["model_roots"][model_id]
            task = VARIANTS[model_id].task
            if analysis:
                original, bundle, baseline_path, baseline = load_original(
                    model_root, model_id
                )
                assess = (
                    real_whole.evaluate if task == "whole-run" else evaluate
                )
                candidate_options = {}
                if model_id in request.get("candidate_models", {}):
                    chosen, calibration_path = selected_paths(
                        dict(
                            request,
                            profile=log.metadata["profile"],
                            log_origin=log.metadata["origin"],
                        ),
                        model_id,
                        bundle,
                        baseline_path,
                    )
                    candidate_options = dict(
                        candidate=Runtime(chosen),
                        calibrated=load_calibration(chosen, calibration_path),
                    )
                report, chart = assess(
                    log, original, baseline, **candidate_options
                )
                model_output.mkdir()
                write(model_output / "0.json", chart)
                results.append(
                    dict(model_id=model_id, task=task, evaluation=[report])
                )
            else:
                results.append(
                    train_candidate(model_root, model_id, inputs, model_output)
                )
        except (OSError, ValueError, RuntimeError) as error:
            failures[model_id] = str(error)
    state = dict(
        analyses=results if analysis else [],
        candidates=results if not analysis else [],
        failures=failures,
    )
    if analysis:
        state.update(
            filename=log.binding["filename"],
            coverage=log.coverage,
            candidate_job=request.get("candidate_job"),
        )
    write(output / "result.json", state)
    if not results:
        raise ValueError(
            "\n".join(f"{model}: {error}" for model, error in failures.items())
        )
    return state
