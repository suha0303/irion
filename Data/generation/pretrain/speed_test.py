"""Finite matched-seed SITL speed pilot. No production collection or models."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .common import ROOT, STORE, external, package_imports, read, runs_root, sha, write
from .speed_metrics import PROTOCOL
from .speed_metrics import metrics as matched_metrics


def metrics(canonical, record):
    frames = [json.loads(line) for line in (canonical / "frames.jsonl").read_text().splitlines()]
    summary = read(canonical / "summary.json")
    names = summary["feature_order"]
    airborne = [f for f in frames if f["operating_state"] == "airborne"]
    origin = airborne[0]["time_us"]
    valid = [f for f in airborne if all(f["quality"][name] == "valid" for name in names)]
    time = np.asarray([(f["time_us"] - origin) / 1e6 for f in valid])
    grid = np.arange(20.0, 110.01, 0.1)
    if len(time) < 2 or time[0] > grid[0] or time[-1] < grid[-1]:
        raise ValueError("Insufficient common matched-flight comparison interval")
    values = np.asarray([[f["values"][name] for name in names] for f in valid])
    aligned = np.stack([np.interp(grid, time, values[:, i]) for i in range(len(names))], axis=1)
    if any(b["time_us"] - a["time_us"] != 100000 for a, b in zip(frames, frames[1:])):
        raise ValueError("Canonical10Hz information grid changed")
    return dict(
        features=names,
        config=summary["config"],
        aligned=aligned.tolist(),
        valid_airborne_fraction=len(valid) / len(airborne),
        windows=summary["windows"],
        topic_quality=record["assessment"]["topic_quality"],
        effective_rtf=record["assessment"].get("effective_rtf"),
        raw_run=record["run_id"],
        canonical=str(canonical),
    )


def compare(base, repeat, candidate):
    for trial in (repeat, candidate):
        if any(
            base.get(key) != trial.get(key)
            for key in ("features", "config", "protocol", "comparison_intervals")
        ):
            raise ValueError("Feature/transform/comparison interval contract changed with speed")
        if set(base["topic_quality"]) != set(trial["topic_quality"]):
            raise ValueError("Comparison topic set changed")
    a, b, c = [np.asarray(x["aligned"]) for x in (base, repeat, candidate)]
    baseline_rmse = np.sqrt(np.mean((a - b) ** 2, axis=0))
    candidate_rmse = np.sqrt(np.mean((a - c) ** 2, axis=0))
    floors = []
    for name in base["features"]:
        floors.append(
            0.005
            if name.startswith("motor_request")
            else 0.03
            if name == "pack_voltage"
            else 0.3
            if name == "pack_current"
            else 0.1
            if name.startswith("specific_force")
            else 0.05
            if name.startswith("climb")
            else 0.02
            if "rate" in name
            else 0.01
        )
    allowed = np.maximum(2 * baseline_rmse, floors)
    raw = {}
    for topic, stats in candidate["topic_quality"].items():
        refs = [x["topic_quality"][topic] for x in (base, repeat)]

        def rate(s):
            duration = s["duration_us"] if "duration_us" in s else s["last"] - s["first"]
            return s["count"] / (duration / 1e6)

        low, high = min(map(rate, refs)) * 0.95, max(map(rate, refs)) * 1.05
        raw[topic] = dict(
            rate_hz=rate(stats),
            reference_rate_hz=[rate(r) for r in refs],
            covered=stats.get("covered", True) and all(r.get("covered", True) for r in refs),
            max_gap_us=stats["max_gap_us"],
            reference_max_gap_us=[r["max_gap_us"] for r in refs],
            rate_passed=low <= rate(stats) <= high,
            gap_passed=stats["max_gap_us"] <= max(r["max_gap_us"] for r in refs) + 1000,
        )
    coverage = (
        candidate["valid_airborne_fraction"]
        >= min(base["valid_airborne_fraction"], repeat["valid_airborne_fraction"]) - 0.01
    )
    windows = (
        min(base["windows"], repeat["windows"]) * 0.98
        <= candidate["windows"]
        <= max(base["windows"], repeat["windows"]) * 1.02
    )
    return dict(
        passed=bool(
            np.all(candidate_rmse <= allowed)
            and coverage
            and windows
            and all(v["rate_passed"] and v["gap_passed"] and v["covered"] for v in raw.values())
        ),
        bitwise_equal=bool(np.array_equal(a, c)),
        coverage_passed=coverage,
        windows_passed=windows,
        baseline_repeat_rmse=dict(zip(base["features"], baseline_rmse.tolist())),
        candidate_rmse=dict(zip(base["features"], candidate_rmse.tolist())),
        allowed_rmse=dict(zip(base["features"], allowed.tolist())),
        topic_quality=raw,
        protocol=base.get("protocol", "information-v1"),
        comparison_intervals=base.get("comparison_intervals"),
        scope="single hover120s, one lane, one matched seed; no16-lane/maneuver-wide or bitwise-equivalence claim",
    )


def _run(runner):
    package_imports()
    from preprocessing_tools.control import RAW_FILES

    output = external(STORE / "speed-tests" / runner.task.name)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    protocol = runner.config["speed_test"].get("protocol", "information-v1")
    single = runner.config["speed_test"].get("candidate_only", False)
    baseline = runner.c.candidate_baseline(runner.config)[0] if single else None
    basis = dict(
        candidate_only=single, baseline_review=runner.config["speed_test"].get("baseline_review")
    )
    runner.update(
        planned_stage_count=3 * len(runner.config["speed_test"]["factors"]),
        speed_protocol=protocol,
        speed_test_output=str(output),
    )
    for index, factor in enumerate(runner.config["speed_test"]["factors"]):
        runner.update(speed_factor=factor, speed_trial=index + 1)
        label = f"speed-{index + 1:02}"
        row = dict(
            case_id=label,
            role="benchmark",
            scenario="hover_normal",
            seed=runner.config["seed_base"] + 2000,
            group_id="speed-comparison-only",
            mission_seconds=120,
            retries=0,
        )
        folder = runner.flight("benchmark", 1, [row], label)
        record = next(iter(read(folder / "execution/ledger.json")["cases"].values()))
        raw = runs_root(ROOT) / record["run_id"]
        world = ET.parse(raw / "reference/world.sdf")
        if (
            float(world.find(".//physics/real_time_factor").text) != factor
            or float(world.find(".//physics/max_step_size").text) != 0.001
        ):
            raise ValueError("Physics step/speed request was not honored")
        row.update(
            run_id=record["run_id"],
            path=str(raw),
            files={name: record["export"]["members"][name]["sha256"] for name in RAW_FILES},
        )
        runner.batch(label + "-preprocess", "preprocess", [row], 1, output / label / "canonical")
        runner.batch(
            label + "-audit",
            "audit",
            [row],
            1,
            output / label / "audit",
            canonical=output / label / "canonical",
        )
        measure = matched_metrics if protocol == PROTOCOL else metrics
        measured = measure(output / label / "canonical" / row["run_id"], record)
        result = dict(
            requested_factor=factor,
            measured=measured,
            training_eligible=False,
            capture_host_interval_ms=10 / factor,
        )
        if single or index >= 2:
            refs = baseline["results"] if single else results
            result["comparison"] = compare(refs[0]["measured"], refs[1]["measured"], measured)
        results.append(result)
        write(
            output / "results.json",
            dict(status="RUNNING", results=results, production_speed_changed=False, **basis),
        )
        if (single or index >= 2) and not result["comparison"]["passed"]:
            write(
                output / "results.json",
                dict(
                    status="FAILED_INFORMATION_GATE_STOPPED",
                    results=results,
                    production_speed_changed=False,
                    automatic_retry=False,
                    **basis,
                ),
            )
            raise RuntimeError(
                f"{factor}x information gate failed; remaining speed candidates not run"
            )
    write(
        output / "results.json",
        dict(
            status="SPEED_CANDIDATE_COMPLETE_STOPPED" if single else "SPEED_PILOT_COMPLETE_STOPPED",
            results=results,
            production_speed_changed=False,
            production_default=2,
            automatic_retry=False,
            limitations="Pilot only; exact bit equality separately recorded. Single lane, hover, one seed.",
            **basis,
        ),
    )
    runner.update(result=str(output / "results.json"))


def run(runner):
    try:
        _run(runner)
    except BaseException as exc:
        path = STORE / "speed-tests" / runner.task.name / "results.json"
        state = read(path) if path.exists() else dict(results=[])
        state.update(
            status="FAILED_STOPPED",
            reason=f"{type(exc).__name__}: {exc}",
            automatic_retry=False,
            production_speed_changed=False,
        )
        write(path, state)
        raise


def review(task, output):
    """Recompute only preserved completed trials; never generate or resume flights."""
    task, output = Path(task).resolve(), Path(output).resolve()
    cfg = read(task / "config.json")
    state = read(task / "operator/state.json")
    if (
        state["status"] not in {"FAILED_STOPPED", "TEST_COMPLETE_STOPPED"}
        or cfg.get("mode") != "test"
        or cfg.get("speed_test", {}).get("factors") != [2, 2, 4, 8]
    ):
        raise ValueError("Stopped finite speed-test source required")
    if sha(task / "config.json") != (task / "config.sha256").read_text().strip():
        raise ValueError("Original configuration changed")
    from x500_storage import task_root

    old = task_root(read(task / "config.json")) / "speed-tests" / task.name / "results.json"
    old_hash = sha(old)
    prior = read(old)
    if len(prior["results"]) < 3:
        raise ValueError("Two baseline repetitions and at least one candidate required")
    if output.is_relative_to(task) or output.is_relative_to(old.parent):
        raise ValueError("Review output must be separate from original evidence")
    output.mkdir(parents=True, exist_ok=False)
    from .speed_metrics import INTERVALS

    write(
        output / "protocol.json",
        dict(
            protocol=PROTOCOL,
            comparison_intervals=INTERVALS,
            rationale="Fixed steady mission window;60s context entirely within mission. Selected before review computation; no error tolerances widened.",
            implementation_sha256={
                p.name: sha(p)
                for p in (Path(__file__), Path(__file__).with_name("speed_metrics.py"))
            },
            new_flights_authorized=False,
        ),
    )
    results = []
    try:
        for index, trial in enumerate(prior["results"]):
            ledger_path = task / f"speed-{index + 1:02}/execution/ledger.json"
            ledger = read(ledger_path)
            if ledger["status"] != "COLLECTED_PENDING_QUALITY_STOPPED" or len(ledger["cases"]) != 1:
                raise ValueError("Complete single-flight evidence required")
            record = next(iter(ledger["cases"].values()))
            if (
                record["run_id"] != trial["measured"]["raw_run"]
                or record["status"] != "ARCHIVED_PENDING_QUALITY"
            ):
                raise ValueError("Original trial identity/status changed")
            measured = matched_metrics(Path(trial["measured"]["canonical"]), record)
            result = dict(
                requested_factor=trial["requested_factor"],
                measured=measured,
                source_ledger_sha256=sha(ledger_path),
                training_eligible=False,
            )
            if index >= 2:
                result["comparison"] = compare(
                    results[0]["measured"], results[1]["measured"], measured
                )
            results.append(result)
        passed = all(r["comparison"]["passed"] for r in results[2:])
        status = (
            "REVIEW_COMPLETE_STOPPED"
            if passed and len(results) == 4
            else "PARTIAL_REVIEW_STOPPED_8X_NOT_RUN"
            if passed
            else "FAILED_INFORMATION_GATE_STOPPED"
        )
        report = dict(
            status=status,
            protocol=PROTOCOL,
            source_task=str(task),
            original_results_sha256=old_hash,
            original_status=prior["status"],
            results=results,
            evidence_reinterpreted_post_failure=True,
            thresholds_changed=False,
            original_evidence_unchanged=True,
            new_flights_started=False,
            automatic_retry=False,
            production_speed_changed=False,
            missing_factors=[8] if len(results) < 4 else [],
            limitation="Post-failure correction of comparison scope on existing trials; independent confirmation and8x remain untested if absent.",
        )
        if sha(old) != old_hash:
            raise ValueError("Original results changed during review")
        write(output / "results.json", report)
    except BaseException as exc:
        write(
            output / "results.json",
            dict(
                status="REVIEW_FAILED_STOPPED",
                reason=f"{type(exc).__name__}: {exc}",
                results=results,
                new_flights_started=False,
            ),
        )
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate a stopped speed task, no new flights")
    parser.add_argument("task", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = review(args.task, args.output)
    print(json.dumps(dict(status=report["status"], result=str(args.output / "results.json"))))
