"""Coordinator-only truth join after all four models have sealed their development forecasts."""

import math
import tarfile
from pathlib import Path

import numpy as np

from capstone_ai.detection_protocol import event_result, grouped_rate_interval, point_metrics
from capstone_ai.detection_replay import evaluate_sealed, records

from .common import MODELS, child, read, sha, verify, write


def forecast_errors(decisions):
    valid = [d for d in decisions if d["scores"] is not None]
    if not valid:
        return {"scored_forecasts": 0, "mae": None, "rmse": None, "signed_bias": None}
    signed = np.asarray([d["signed_residual"] for d in valid])
    if signed.shape != (len(valid), 10, 4) or not np.isfinite(signed).all():
        raise ValueError("invalid sealed residuals")
    return {
        "scored_forecasts": len(valid),
        "target_samples_per_target": len(valid) * 10,
        "mae": np.abs(signed).mean((0, 1)).tolist(),
        "rmse": np.sqrt((signed**2).mean((0, 1))).tolist(),
        "signed_bias": signed.mean((0, 1)).tolist(),
        "signed_definition": "actual minus forecast",
        "units": ["rad/s"] * 3 + ["V"],
    }


def event_counts(events):
    total = len(events)
    available = sum(not e["insufficient_data"] for e in events)
    detected = sum(e["detected"] for e in events)
    return {
        "events": total,
        "detected": detected,
        "missed_or_unavailable": total - detected,
        "with_opportunity": available,
        "unavailable": total - available,
        "recall_all": detected / total if total else None,
        "recall_conditional": detected / available if available else None,
        "new_episode_recall_all": sum(e["new_episode_detected"] for e in events) / total
        if total
        else None,
    }


def truth_member(row, target, progress):
    archive = row["archive"]
    path = Path(archive["path"])
    if sha(path) != archive["sha256"] or path.stat().st_size != archive["bytes"]:
        raise ValueError("Development archive changed")
    expected = archive["members"]["evaluation/events.jsonl"]
    if expected["kind"] != "file" or not 0 <= expected["bytes"] <= 10_000_000:
        raise ValueError("Invalid development event inventory")
    found = False
    with tarfile.open(path, "r|gz") as stream:
        for member in stream:
            progress.check()
            if member.name != "evaluation/events.jsonl":
                continue
            if found or not member.isfile() or member.sparse or member.size != expected["bytes"]:
                raise ValueError("Duplicate/invalid event member")
            found = True
            with stream.extractfile(member) as source, target.open("xb") as output:
                output.write(source.read(10_000_001))
    if not found or target.stat().st_size != expected["bytes"] or sha(target) != expected["sha256"]:
        raise ValueError("Development truth hash mismatch")


def evaluate(
    config,
    download,
    output,
    progress,
    *,
    truth_provider=truth_member,
    model_names=MODELS,
    development_counts=None,
):
    model_names = tuple(model_names)
    counts = development_counts or dict(normal_dev_eval=11, anomaly_dev=12)
    expected_total = sum(counts.values())
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("Explicit unique model identities required")
    download = Path(download)
    output = Path(output)
    release = read(output.parent / "release.json")
    if (
        not release.get("released")
        or release.get("slurm_state") != "COMPLETED"
        or release.get("exit_code") != "0:0"
    ):
        raise ValueError("Successful Slurm release required before truth join")
    seal = read(download / "result-seal.json")
    if (
        seal.get("status") != "TRAIN_CALIBRATION_INFERENCE_COMPLETE"
        or seal.get("models") != list(model_names)
        or seal.get("final_test_used") is not False
        or seal.get("truth_join") is not False
    ):
        raise ValueError("All configured sealed inference results required")
    verify(download, seal["files"])
    dataset = Path(config["dataset"])
    if sha(dataset / "manifest.json") != config["dataset_files"]["manifest.json"]:
        raise ValueError("Dataset binding changed")
    rows = [
        r
        for r in read(dataset / "manifest.json")["runs"]
        if r["role"] in {"normal_dev_eval", "anomaly_dev"}
    ]
    if any(sum(r["role"] == role for r in rows) != count for role, count in counts.items()):
        raise ValueError("Development membership/count mismatch")
    output.mkdir(exist_ok=False)
    (output / "truth").mkdir()
    per_model = {name: [] for name in model_names}
    common_residuals = {name: [] for name in model_names}
    all_residuals = {name: [] for name in model_names}
    common_count = 0
    for index, row in enumerate(rows):
        progress.check()
        progress.update(
            stage="development_evaluation",
            done=index,
            total=expected_total,
            model=None,
            run_id=row["run_id"],
        )
        truth = output / "truth" / (row["run_id"] + ".jsonl")
        truth_provider(row, truth, progress)
        frame_path = child(dataset, "canonical/" + row["run_id"] + "/frames.jsonl")
        if sha(frame_path) != config["dataset_files"][str(frame_path.relative_to(dataset))]:
            raise ValueError("Evaluation canonical changed")
        with frame_path.open() as stream:
            first = __import__("json").loads(next(stream))["time_us"]
        decisions_by_model = {}
        for name in model_names:
            replay = download / "calibration" / name / row["run_id"]
            summary = read(replay / "summary.json")
            ds = records(replay / "decisions.jsonl")
            if summary["operational_seconds"] > 0:
                report = evaluate_sealed(replay, truth, first)
            else:
                verify(replay, read(replay / "seal.json")["files"])
                if ds:
                    raise ValueError("Decisions without operational exposure")
                report = dict(
                    events=[event_result(e, [], first) for e in records(truth)],
                    metrics=dict(
                        operational_seconds=0,
                        scored_seconds=0,
                        coverage=None,
                        episodes={k: [] for k in ("union", "battery", "propulsion")},
                        reason="no_observed_airborne_exposure",
                    ),
                    point_metrics=point_metrics(records(truth), [], first),
                )
            if row["role"] == "normal_dev_eval" and report["events"]:
                raise ValueError("Normal development control has anomaly events")
            for event in report["events"]:
                event["pre_event_30_opportunities_met"] = (
                    None
                    if event["onset_kind"] == "pre_existing"
                    else event["pre_onset_valid_decisions"] >= 30
                )
            per_model[name].append(
                dict(
                    run_id=row["run_id"],
                    role=row["role"],
                    scenario=row["scenario"],
                    group=row["group_id"],
                    **report,
                    forecast_errors=forecast_errors(ds),
                    replay_summary=summary,
                )
            )
            keyed = {}
            seen = set()
            for d in ds:
                key = (d["clock_id"], d["segment_id"], d["origin_time_us"])
                if key in seen:
                    raise ValueError("Duplicate forecast origin")
                seen.add(key)
                if d["scores"] is not None:
                    keyed[key] = d
            decisions_by_model[name] = keyed
            all_residuals[name].extend([d["signed_residual"] for d in keyed.values()])
        common = set.intersection(*(set(v) for v in decisions_by_model.values()))
        common_count += len(common)
        for name in model_names:
            common_residuals[name].extend(
                [decisions_by_model[name][k]["signed_residual"] for k in sorted(common)]
            )
    results = {}
    for name, items in per_model.items():
        events = [e for item in items for e in item["events"]]
        normal = [r for r in items if r["role"] == "normal_dev_eval"]
        hours = sum(r["metrics"]["scored_seconds"] for r in normal) / 3600
        alarms = {
            k: sum(len(r["metrics"]["episodes"][k]) for r in normal)
            for k in ("union", "battery", "propulsion")
        }

        def errors(residuals):
            return forecast_errors([dict(scores=[0, 0], signed_residual=r) for r in residuals])

        results[name] = dict(
            per_run=items,
            events=event_counts(events),
            by_onset_kind={
                k: event_counts([e for e in events if e["onset_kind"] == k])
                for k in sorted({e["onset_kind"] for e in events})
            },
            normal_scored_hours=hours,
            normal_episodes=alarms,
            normal_episodes_per_hour={k: v / hours if hours else None for k, v in alarms.items()},
            zero_union_poisson95_upper_per_hour=-math.log(0.05) / hours
            if hours and not alarms["union"]
            else None,
            normal_group_interval=grouped_rate_interval(
                [
                    dict(
                        group=r["group"],
                        episodes=len(r["metrics"]["episodes"]["union"]),
                        hours=r["metrics"]["scored_seconds"] / 3600,
                    )
                    for r in normal
                ]
            ),
            forecast_errors_all_valid=errors(all_residuals[name]),
            forecast_errors_common_origins=errors(common_residuals[name]),
        )
        # Recount saved per-run results; failure is never hidden by a partial report.
        if len(items) != expected_total or results[name]["events"]["detected"] != sum(
            e["detected"] for r in items for e in r["events"]
        ):
            raise ValueError("Result cross-check failed")
    metrics = dict(
        evaluation_mode="development",
        final_test_used=False,
        seed=config.get("seed", 17),
        models=results,
        common_valid_origins=common_count,
        point_adjustment=False,
        limitations=[
            f"single model seed{config.get('seed', 17)}",
            "grouped development results; simulation only",
            "development evaluation; no locked final test",
        ],
    )
    write(output / "metrics.json", metrics)
    write(
        output / "report.json",
        dict(
            status="REPORT_READY_STOPPED",
            evaluation_mode="development",
            final_test_used=False,
            models=list(model_names),
            seed=config.get("seed", 17),
            metrics_sha256=sha(output / "metrics.json"),
            release=release,
            next_round_authorized=False,
            dataset_handoff_sha256=config.get("handoff_sha256"),
            config=config,
            comparison_criteria=[
                "normal false alarm episodes/hour",
                "event recall including unavailable cases",
                "onset/recognition delay",
                "forecast errors on common valid origins",
            ],
            retries=0,
            llm_calls=0,
            poor_performance_action="report only",
        ),
    )
    progress.update(
        status="REPORT_READY_STOPPED",
        stage="stopped",
        done=expected_total,
        total=expected_total,
        report=str(output / "report.json"),
        phase_eta_s=None,
    )
    return metrics
