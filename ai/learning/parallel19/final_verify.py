"""One frozen-model normal calibration and offline final evaluation per model.

Offline results do not certify live timing, real-aircraft health, SOH or RUL.
Final labels never enter model.forward, calibration, or model/LR selection.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from .control import read, sha, write


def selection_binding(plan, request, cohort, model_id):
    root = Path(plan["output"])
    per_model = cohort == "final" and plan.get(
        "final_per_model_when_search_stopped", False
    )
    selection = (
        Path(request["selection_manifest"])
        if cohort == "development" or per_model
        else root / "selected-models.json"
    )
    if cohort == "development":
        allowed = [root, *[Path(p) for p in plan.get("preserved_outputs", [])]]
        if not any(
            selection.resolve().is_relative_to(
                (p / "development-selections").resolve()
            )
            for p in allowed
        ):
            raise ValueError("Invalid development binding")
    elif per_model and not any(
        selection.resolve()
        == (p / "final-selections" / f"{model_id}.json").resolve()
        for p in [root, *[Path(x) for x in plan.get("preserved_outputs", [])]]
    ):
        raise ValueError("Invalid per-model final binding")
    if sha(selection) != request["selected_manifest_sha256"]:
        raise ValueError("Unfrozen models")
    receipt = read(selection)
    bound = (
        receipt["selected"]
        if cohort == "development" or per_model
        else receipt["models"][model_id]
    )
    if bound != request["selected"]:
        raise ValueError("Selected model differs from frozen receipt")
    if per_model and (
        receipt.get("final_test_used_for_selection") is not False
        or not receipt.get("search_status", "").startswith("LOCAL_")
        or receipt.get("model") != model_id
    ):
        raise ValueError("Model search must be frozen before final")
    return selection


def events(row):
    if row["role"] != "anomaly_test":
        return []
    evidence = row["source_evidence"]
    scenario = evidence["scenario_name"]
    mission = row["mission_interval"]
    existing = scenario.endswith("_existing")
    transitions = evidence["transition_evidence"]
    if not existing and not transitions:
        raise ValueError("Missing actual fault onset")
    onset = (
        None
        if existing
        else min(t["actual_onset_us"] for t in transitions.values())
    )
    boundary = (
        None
        if existing
        else max(t["actual_endpoint_us"] for t in transitions.values())
    )
    return [
        dict(
            event_id=row["run_id"],
            onset_kind="pre_existing"
            if existing
            else "ramp"
            if scenario.endswith("_ramp")
            else "step",
            physical_onset_time_us=onset,
            applied_time_us=onset,
            degradation_boundary_time_us=boundary,
            observation_start_time_us=mission["start_us"],
            end_time_us=mission["end_us_exclusive"] - 1,
        )
    ]


def summarize(rows, expected_runs=729):
    normal = [r for r in rows if r["role"] == "normal_test"]
    anomaly = [r for r in rows if r["role"] == "anomaly_test"]

    def part(items):
        return {
            "runs": len(items),
            "alarm_runs": sum(r["alarm"] for r in items),
            "alarm_rate": sum(r["alarm"] for r in items) / len(items)
            if items
            else None,
        }

    result = {
        "normal": part(normal),
        "anomaly": part(anomaly),
        "by_scenario": {},
        "all_required_runs_scored": len(rows) == expected_runs,
    }
    for name in sorted(set(r["scenario"] for r in rows)):
        result["by_scenario"][name] = part(
            [r for r in rows if r["scenario"] == name]
        )
    for label in ("standalone", "paired"):
        result["normal_" + label] = part(
            [r for r in normal if r["normal_kind"] == label]
        )
    event_rows = [r["event"] for r in anomaly if "event" in r]
    if event_rows:
        result["event_recall"] = sum(e["detected"] for e in event_rows) / len(
            event_rows
        )
        result["new_episode_recall"] = sum(
            e["new_episode_detected"] for e in event_rows
        ) / len(event_rows)
        delays = [
            e["onset_delay_s"]
            for e in event_rows
            if e["onset_delay_s"] is not None
        ]
        result["detected_onset_delay_seconds"] = {
            "n": len(delays),
            "median": float(np.median(delays)) if delays else None,
            "p95": float(np.quantile(delays, 0.95)) if delays else None,
        }
        hours = sum(r["scored_seconds"] for r in normal) / 3600
        result["normal_episodes_per_hour"] = (
            sum(r["episodes"] for r in normal) / hours
        )
        from capstone_ai.detection_protocol import grouped_rate_interval

        result["normal_episode_rate_interval"] = grouped_rate_interval(
            [
                {
                    "group": r["group"],
                    "episodes": r["episodes"],
                    "hours": r["scored_seconds"] / 3600,
                }
                for r in normal
            ]
        )
    return result


def predict(m, entry, root, rows, scaler, out, phase):
    import torch

    from learning.run_health19.data import physical_features

    from .fit import progress

    batch = entry["batch"]
    result = {}
    for start in range(0, len(rows), 8):
        selected = rows[start : start + 8]
        arrays = []
        for row in selected:
            p = root / row["path"]
            if sha(p) != row["sha256"]:
                raise ValueError("Final input changed")
            with np.load(p, allow_pickle=False) as z:
                arrays.append(dict(z))
        with torch.inference_mode():
            if entry["task"] == "forecast":
                for row, a in zip(selected, arrays):
                    chunks = [
                        m(torch.from_numpy(a["X"][j : j + batch]).cuda())
                        .cpu()
                        .numpy()
                        for j in range(0, len(a["X"]), batch)
                    ]
                    predicted = np.concatenate(chunks)
                    residual = a["Y"] - predicted
                    result[row["run_id"]] = {
                        "residual": residual,
                        "origin_us": a["origin_us"],
                        "segment_id": a["segment_id"],
                    }
                    np.savez_compressed(
                        out / (row["run_id"] + ".npz"),
                        prediction_scaled=predicted,
                        residual_scaled=residual,
                        origin_us=a["origin_us"],
                        segment_id=a["segment_id"],
                    )
            else:
                normalized = [
                    (
                        (physical_features(a["values"]) - scaler["x_mean"])
                        / scaler["x_scale"]
                    ).astype("f4")
                    for a in arrays
                ]
                lengths = np.array([len(x) for x in normalized], dtype="i8")
                x = np.zeros((len(lengths), int(lengths.max()), 19), dtype="f4")
                for i, a in enumerate(normalized):
                    x[i, : len(a)] = a
                predicted = (
                    m(
                        torch.from_numpy(x).cuda(),
                        torch.from_numpy(lengths).cuda(),
                    )
                    .cpu()
                    .numpy()
                )
                from learning.run_health19.scoring import components

                for i, (row, a) in enumerate(zip(selected, arrays)):
                    n = int(lengths[i])
                    pred = predicted[i, :n]
                    observed = normalized[i][:, 17]
                    c = components(
                        pred,
                        observed,
                        cell_voltage_scale=float(scaler["x_scale"][17]),
                    )
                    if c["status"] != "available":
                        raise ValueError("Unavailable final run")
                    result[row["run_id"]] = c
                    np.savez_compressed(
                        out / (row["run_id"] + ".npz"),
                        predicted_cell_v=pred * scaler["x_scale"][17]
                        + scaler["x_mean"][17],
                        observed_cell_v=observed * scaler["x_scale"][17]
                        + scaler["x_mean"][17],
                        time_us=a["time_us"],
                    )
        if any(
            not np.isfinite(v["residual"]).all()
            for v in (result[row["run_id"]] for row in selected)
            if "residual" in v
        ):
            raise ValueError("Nonfinite prediction")
        progress(
            out.parent,
            stage=phase,
            completed_runs=min(start + 8, len(rows)),
            total_runs=len(rows),
        )
    return result


def run(args):
    import torch
    from capstone_ai.detection_protocol import (
        event_result,
        fit_calibration,
        point_metrics,
        score_sequence,
    )
    from safetensors.torch import load_file

    from learning.run_health19.data import Run
    from learning.run_health19.scoring import Calibration

    from .fit import backend, model, runtime

    plan = read(args.plan)
    entry = next(e for e in plan["models"] if e["id"] == args.entry)
    request = read(args.request)
    entry = {**entry, "node": request["node"]}
    out = Path(request["output"])
    out.mkdir(parents=True, exist_ok=False)
    runtime(plan, entry, out)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cohort = request.get("cohort", "final")
    if cohort not in {"final", "development"}:
        raise ValueError("Unknown cohort")
    expected_runs = 729 if cohort == "final" else 243
    cache_key = cohort + "_cache_sha256"
    root = Path(plan[cohort + "_inputs"])
    cache = read(root / "cache.json")
    if sha(root / "cache.json") != plan[cache_key]:
        raise ValueError("Final input inventory changed")
    selection = selection_binding(plan, request, cohort, args.entry)
    selected = request["selected"]
    folder = Path(selected["folder"])
    if sha(folder / "model.safetensors") != selected["model_sha256"]:
        raise ValueError("Selected weights changed")
    if sha(root / "scaler.npz") != sha(
        Path(plan["inputs"]) / "main/scaler.npz"
    ):
        raise ValueError("Train scaler mismatch")
    scaler = dict(np.load(root / "scaler.npz", allow_pickle=False))
    rows = cache["runs"]
    validation = [r for r in rows if r["role"] == "normal_validation"]
    calibration = [r for r in rows if r["role"] == "normal_calibration"]
    final = [r for r in rows if r["cohort"] == cohort]
    if (len(validation), len(calibration), len(final)) != (
        81,
        81,
        expected_runs,
    ):
        raise ValueError("Wrong evaluation cohorts")
    if any(
        r["role"] not in {"normal_validation", "normal_calibration"}
        and r["cohort"] != cohort
        for r in rows
    ):
        raise ValueError("Mixed evaluation cohorts")
    m = model(entry)
    m.load_state_dict(load_file(str(folder / "model.safetensors")))
    m = m.cuda().eval()
    saved = out / "predictions"
    saved.mkdir()
    with backend("cuda"):
        v = predict(
            m, entry, root, validation, scaler, saved, "normal_validation"
        )
        c = predict(
            m, entry, root, calibration, scaler, saved, "normal_calibration"
        )
        if entry["task"] == "forecast":

            def concat(rows, records, key):
                return np.concatenate([records[r["run_id"]][key] for r in rows])

            ids = np.concatenate(
                [
                    np.repeat(r["run_id"], len(c[r["run_id"]]["residual"]))
                    for r in calibration
                ]
            )
            scale = scaler["y_scale"].copy()
            scale[3] *= 4
            calibrated = fit_calibration(
                concat(validation, v, "residual"),
                concat(calibration, c, "residual"),
                ids,
                concat(calibration, c, "segment_id"),
                concat(calibration, c, "origin_us"),
                scale,
                {r["run_id"]: r["group_id"] for r in calibration},
            )
        else:

            def pairs(rows, records):
                result = []
                for row in rows:
                    with np.load(root / row["path"], allow_pickle=False) as a:
                        run = Run(
                            row["run_id"],
                            row["role"],
                            tuple(row["lineage"]),
                            row["health_eligibility"],
                            row["origin"],
                            a["values"],
                            a["time_us"],
                            row["cohort"],
                        )
                    result.append((run, records[row["run_id"]]))
                return result

            threshold = Calibration.fit(
                pairs(validation, v), pairs(calibration, c), alpha=0.05
            )
            calibrated = dataclasses.asdict(threshold)
        write(
            out / "calibration.json",
            {
                "task": entry["task"],
                "parameters": calibrated,
                "normal_validation_runs": 81,
                "normal_calibration_runs": 81,
                "selected_manifest_sha256": request["selected_manifest_sha256"],
                "final_used": False,
            },
        )
        calibration_hash = sha(out / "calibration.json")
        # The complete threshold record is frozen before the first final forward.
        f = predict(m, entry, root, final, scaler, saved, cohort + "_inference")
    records = []
    for row in final:
        r = f[row["run_id"]]
        record = {
            "run_id": row["run_id"],
            "role": row["role"],
            "group": row["group_id"],
            "scenario": (
                "combined_"
                + row["source_evidence"]["scenario_name"].rsplit("_", 1)[1]
            )
            if "-combined-" in row["case_id"]
            else row["source_evidence"]["scenario_name"],
            "normal_kind": "paired"
            if row["case_id"].endswith("-normal")
            else "standalone",
        }
        if entry["task"] == "forecast":
            contribution = r["residual"] ** 2 / np.array(
                calibrated["reference_error_scale_scaled"]
            )
            scores = np.stack(
                [
                    contribution[:, :, :3].mean((1, 2)),
                    contribution[:, :, 3].mean(1),
                ],
                axis=1,
            )
            decisions = score_sequence(
                scores,
                np.repeat(row["run_id"], len(scores)),
                r["segment_id"],
                r["origin_us"],
                calibrated["high"],
            )
            for d in decisions:
                d.update(
                    clock_id=row["mission_interval"]["clock_id"],
                    unscored_reasons=[],
                )
            truth = events(row)
            if truth:
                record["event"] = event_result(
                    truth[0], decisions, row["mission_interval"]["start_us"]
                )
            record.update(
                alarm=any(d["alarm_active"] for d in decisions),
                episodes=sum(d["episode_start"] for d in decisions),
                scored_seconds=len(decisions),
                operational_seconds=(
                    row["mission_interval"]["end_us_exclusive"]
                    - row["mission_interval"]["start_us"]
                )
                / 1e6,
                point_metrics=point_metrics(
                    truth, decisions, row["mission_interval"]["start_us"]
                ),
                rmse_physical=np.sqrt(
                    np.mean((r["residual"] * scale) ** 2, axis=(0, 1))
                ).tolist(),
            )
            write(out / (row["run_id"] + "-decisions.json"), decisions)
        else:
            record.update(components=r, **threshold.decide(r))
        records.append(record)
    if sha(out / "calibration.json") != calibration_hash:
        raise ValueError("Threshold mutated during final")
    summary = summarize(records, expected_runs)
    write(out / "per-run.json", records)
    write(
        out / "result.json",
        {
            "status": "FINAL_VERIFICATION_COMPLETE"
            if cohort == "final"
            else "DEVELOPMENT_EVALUATION_COMPLETE",
            "model": entry,
            "selected_lr": selected["lr"],
            "model_sha256": selected["model_sha256"],
            "calibration_sha256": calibration_hash,
            "evaluation_cohort": cohort,
            "evaluation_cache_sha256": plan[cache_key],
            "final_cache_sha256": plan[cache_key]
            if cohort == "final"
            else None,
            "summary": summary,
            "finished_at": time.time(),
            "scope": "offline held-out X500 simulation; not real-time or hardware certification",
            "final_used_for_selection": False,
            "optimizer_steps": 0,
        },
    )
    write(
        out / "integrity.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file()
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--entry", required=True)
    p.add_argument("--request", required=True)
    run(p.parse_args())
