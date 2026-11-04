"""Coordinator-only truth join after configured models seal their development forecasts."""

import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np

from capstone_ai.detection_protocol import event_result, grouped_rate_interval, point_metrics
from capstone_ai.detection_replay import evaluate_sealed, records

from .common import MODELS, child, read, sha, verify, write


class ResidualSummary:
    """Accumulate error statistics without retaining the full evaluation pool."""

    def __init__(self):
        self.count = 0
        self.absolute = np.zeros(4, dtype=np.float64)
        self.squared = np.zeros(4, dtype=np.float64)
        self.signed = np.zeros(4, dtype=np.float64)

    def add(self, residuals):
        if not len(residuals):
            return
        signed = np.asarray(residuals, dtype=np.float64)
        if signed.shape != (len(residuals), 10, 4) or not np.isfinite(signed).all():
            raise ValueError("invalid sealed residuals")
        self.count += len(signed)
        self.absolute += np.abs(signed).sum((0, 1))
        self.squared += (signed**2).sum((0, 1))
        self.signed += signed.sum((0, 1))

    def result(self):
        if not self.count:
            return {"scored_forecasts": 0, "mae": None, "rmse": None, "signed_bias": None}
        n = self.count * 10
        return {
            "scored_forecasts": self.count,
            "target_samples_per_target": n,
            "mae": (self.absolute / n).tolist(),
            "rmse": np.sqrt(self.squared / n).tolist(),
            "signed_bias": (self.signed / n).tolist(),
            "signed_definition": "actual minus forecast",
            "units": ["rad/s"] * 3 + ["V"],
        }


def forecast_errors(decisions):
    summary = ResidualSummary()
    summary.add([d["signed_residual"] for d in decisions if d["scores"] is not None])
    return summary.result()


def default_protocol_manifest():
    """Record the existing implementation; this does not change detector settings."""
    import capstone_ai.detection as detection
    import capstone_ai.detection_protocol as protocol
    import capstone_ai.detection_replay as replay
    import capstone_ai.inference as inference

    return {
        "schema": "drone.evaluation-protocol/1",
        "evaluation_mode": "development",
        "normal_validation_scale": "q95 squared scaled residual per lead/target, floor1e-8",
        "normal_calibration_quantiles": protocol.QUANTILES,
        "normal_calibration_union_target_per_hour": 1.0,
        "detector": {
            "ewma_alpha": 0.3,
            "enter_blocks": 3,
            "exit_blocks": 10,
            "exit_threshold_ratio": 0.7,
            "block_seconds": 1,
        },
        "event_success": "awareness",
        "max_detection_delay_s": None,
        "threshold_multiplier": [1.0, 1.0],
        "threshold_domain_order": ["propulsion", "battery"],
        "matching": "applied onset; pre-existing recognition; simultaneous events may share alarm",
        "point_adjustment": False,
        "normal_group_bootstrap": {"seed": 17, "draws": 1000},
        "implementation_sha256": {
            **{m.__name__: sha(m.__file__) for m in (detection, protocol, replay, inference)},
            "execution.evaluate": sha(__file__),
        },
    }


def protocol_hash(protocol):
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


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


def summarize_model(items, all_errors, common_errors, expected_total, *,
                    normal_role="normal_dev_eval", anomaly_role="anomaly_dev"):
    events = [e for item in items for e in item["events"]]
    normal = [r for r in items if r["role"] == normal_role]
    hours = sum(r["metrics"]["scored_seconds"] for r in normal) / 3600
    alarms = {
        k: sum(len(r["metrics"]["episodes"][k]) for r in normal)
        for k in ("union", "battery", "propulsion")
    }
    delay = {}
    operational = sum(r["metrics"]["operational_seconds"] for r in items)
    scored = sum(r["metrics"]["scored_seconds"] for r in items)
    for key in (
        "onset_delay_s",
        "recognition_from_observation_s",
        "recognition_from_first_valid_s",
        "boundary_delay_s",
    ):
        values = [e[key] for e in events if e.get(key) is not None]
        delay[key] = {
            "observed_events": len(values),
            "p50_p95_s": np.quantile(values, [0.5, 0.95]).tolist() if values else None,
            "interpretation": "detected events only; misses are reported separately",
        }
    result = dict(
        per_run=items,
        events=event_counts(events),
        by_onset_kind={
            k: event_counts([e for e in events if e["onset_kind"] == k])
            for k in sorted({e["onset_kind"] for e in events})
        },
        by_scenario={
            k: event_counts([e for r in items if r["scenario"] == k for e in r["events"]])
            for k in sorted({r["scenario"] for r in items})
        },
        anomaly_independent_groups=len({r["group"] for r in items if r["role"] == anomaly_role}),
        delay_summary=delay,
        exposure=dict(
            operational_seconds=operational,
            scored_seconds=scored,
            coverage=scored / operational if operational else None,
            valid_decisions=sum(r["metrics"].get("valid_decisions", 0) for r in items),
            unscored_forecasts=sum(r["metrics"].get("unscored_forecasts", 0) for r in items),
        ),
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
        forecast_errors_all_valid=all_errors,
        forecast_errors_common_origins=common_errors,
    )
    if len(items) != expected_total or result["events"]["detected"] != sum(
        e["detected"] for r in items for e in r["events"]
    ):
        raise ValueError("Result cross-check failed")
    return result


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


def final_protocol(source):
    """Retain the completed development protocol; only identify the final scope."""
    current = default_protocol_manifest()
    old, new = dict(source), dict(current)
    old_impl, new_impl = dict(old.pop("implementation_sha256")), dict(new.pop("implementation_sha256"))
    # This module adds role/scope admission. Detector and event matching are unchanged.
    old_impl.pop("execution.evaluate")
    new_impl.pop("execution.evaluate")
    if old != new or old_impl != new_impl:
        raise ValueError("Final test must retain the completed development protocol")
    return {**current, "evaluation_mode": "final_test",
            "calibration_policy": "reuse frozen normal-only calibration; never refit on final test",
            "source_development_protocol_sha256": protocol_hash(source)}


def evaluate(
    config,
    download,
    output,
    progress,
    *,
    truth_provider=truth_member,
    model_names=MODELS,
    development_counts=None,
    completion_status="REPORT_READY_STOPPED",
    evaluation_mode="development",
):
    model_names = tuple(model_names)
    final = evaluation_mode == "final_test"
    if evaluation_mode not in {"development", "final_test"}:
        raise ValueError("Unknown evaluation mode")
    if final and (config.get("execution_scope") != "final_test_evaluation"
                  or config.get("final_test_authorized") is not True
                  or config.get("calibration_authorized") is not False
                  or config.get("production_training_authorized") is not False):
        raise ValueError("Explicit read-only final evaluation authorization required")
    normal_role, anomaly_role = ("normal_test", "anomaly_test") if final else ("normal_dev_eval", "anomaly_dev")
    counts = development_counts or ({normal_role: 216, anomaly_role: 108} if final
                                   else dict(normal_dev_eval=11, anomaly_dev=12))
    if set(counts) != {normal_role, anomaly_role}:
        raise ValueError("Evaluation roles differ from authorized scope")
    expected_total = sum(counts.values())
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("Explicit unique model identities required")
    if completion_status not in ("REPORT_READY_STOPPED", "EVALUATION_COMPLETE_PENDING_RELEASE"):
        raise ValueError("Unknown evaluation completion status")
    protocol = final_protocol(config["source_protocol"]) if final else default_protocol_manifest()
    if "protocol" in config and config["protocol"] != protocol:
        raise ValueError("Frozen evaluation protocol or implementation changed")
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
        seal.get("status") != ("FINAL_INFERENCE_COMPLETE" if final else "TRAIN_CALIBRATION_INFERENCE_COMPLETE")
        or seal.get("models") != list(model_names)
        or seal.get("final_test_used") is not final
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
        if r["role"] in {normal_role, anomaly_role}
    ]
    if (len(rows) != expected_total or len({r['run_id'] for r in rows}) != expected_total
            or any(sum(r["role"] == role for r in rows) != count for role, count in counts.items())):
        raise ValueError("Development membership/count mismatch")
    output.mkdir(exist_ok=False)
    (output / "truth").mkdir()
    per_model = {name: [] for name in model_names}
    common_residuals = {name: ResidualSummary() for name in model_names}
    all_residuals = {name: ResidualSummary() for name in model_names}
    evidence_rows = []
    common_count = 0
    for index, row in enumerate(rows):
        progress.check()
        progress.update(
            stage="final_evaluation" if final else "development_evaluation",
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
        evidence_rows.append(
            {
                **{k: row[k] for k in ("run_id", "role", "scenario", "group_id")},
                **{
                    k: row[k]
                    for k in ("scenario_parameters_snapshot", "normal_variation", "paired_case_id",
                              "final_cohort", "pair_id", "motion", "environment", "seed")
                    if k in row
                },
                "observation_start_us": first,
                "truth_file": str(truth.relative_to(output)),
                "truth_sha256": sha(truth),
            }
        )
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
            if row["role"] == normal_role and report["events"]:
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
                    group=row.get("paired_case_id") or row["group_id"],
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
            all_residuals[name].add([d["signed_residual"] for d in keyed.values()])
        common = set.intersection(*(set(v) for v in decisions_by_model.values()))
        common_count += len(common)
        for name in model_names:
            common_residuals[name].add(
                [decisions_by_model[name][k]["signed_residual"] for k in sorted(common)]
            )
    results = {
        name: summarize_model(
            items, all_residuals[name].result(), common_residuals[name].result(), expected_total,
            normal_role=normal_role, anomaly_role=anomaly_role,
        )
        for name, items in per_model.items()
    }
    write(output / "protocol.json", protocol)
    write(
        output / "evidence.json",
        {
            "schema": "drone.reusable-evaluation/1",
            "models": list(model_names),
            "inference_root": str(download.resolve()),
            "inference_seal_sha256": sha(download / "result-seal.json"),
            "protocol_sha256": protocol_hash(protocol),
            "runs": evidence_rows,
            "final_test_used": final,
            "truth_join": True,
        },
    )
    metrics = dict(
        evaluation_mode=evaluation_mode,
        final_test_used=final,
        seed=config.get("seed", 17),
        models=results,
        common_valid_origins=common_count,
        point_adjustment=False,
        protocol_sha256=protocol_hash(protocol),
        limitations=[
            f"single model seed{config.get('seed', 17)}",
            "grouped final-test results; simulation only" if final else "grouped development results; simulation only",
            "locked final test; no fitting or threshold tuning" if final else "development evaluation; no locked final test",
        ],
    )
    write(output / "metrics.json", metrics)
    write(
        output / "report.json",
        dict(
            status=completion_status,
            evaluation_mode=evaluation_mode,
            final_test_used=final,
            models=list(model_names),
            seed=config.get("seed", 17),
            metrics_sha256=sha(output / "metrics.json"),
            evidence_sha256=sha(output / "evidence.json"),
            protocol_sha256=protocol_hash(protocol),
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
        status=completion_status,
        stage="stopped",
        done=expected_total,
        total=expected_total,
        report=str(output / "report.json"),
        phase_eta_s=None,
    )
    return metrics
