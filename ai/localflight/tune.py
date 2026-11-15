"""Budget one forecast/post-flight pair; never automatically activate models."""

import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

from .adapt import fit, write
from .adapt_review import review
from .catalog import Catalog, digest, read_json
from .registry import register


def select_reviewed(root, model_id, comparison, *, bundle):
    """Activate an explicitly chosen, regression-checked model and threshold."""
    from . import calibration, registry

    descriptor = registry.read_model(root, model_id, bundle=bundle)
    check = comparison.get("promotion_check", {})
    if (
        check.get("status") != "passed"
        or check.get("model_sha256") != descriptor["model_sha256"]
    ):
        raise ValueError("Normal/anomaly regression review has not passed")
    stored = check["calibrations"]["adapted"]
    record = {k: v for k, v in stored.items() if k not in {"id", "active"}}
    calibration_id = registry.fingerprint(record)
    if (
        calibration_id != stored["id"]
        or record["binding"]["weight_sha256"] != descriptor["model_sha256"]
    ):
        raise ValueError("Reviewed calibration/model mismatch")
    path = Path(root) / "calibrations" / (calibration_id + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write(path, record)
    calibration.read(root, calibration_id)
    event = registry.select_model(root, model_id, comparison, bundle=bundle)
    try:
        selected = calibration.select(root, calibration_id)
    except (ValueError, OSError):
        registry.rollback(root, event["id"], bundle=bundle)
        raise
    return dict(model=event, calibration=selected)


def run(request, *, bundle):
    """Train selected models sequentially, reserving time for CPU review."""
    models = request["models"]
    catalog = Catalog(bundle)
    if (
        not isinstance(models, list)
        or len(models) != 2
        or {catalog.entry(m)["task"] for m in models}
        != {"forecast", "run_health"}
    ):
        raise ValueError(
            "Select exactly one forecast and one post-flight model"
        )
    budget = request.get("budget_seconds", 3600)
    if type(budget) is not int or not 60 <= budget <= 3600:
        raise ValueError("Pair budget must be 60..3600 seconds")
    if not request.get("development"):
        raise ValueError("A separate normal development flight is required")
    out = Path(request["output"])
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    training_deadline = started + budget * 0.7
    write(out / "plan.json", request)
    resume = Path(request["resume"]) if request.get("resume") else None
    if resume:
        previous = read_json(resume / "plan.json")

        def stable(value):
            return {
                k: v
                for k, v in value.items()
                if k
                not in {"output", "resume", "cpu_python", "cpu_python_home"}
            }

        if stable(previous) != stable(request):
            raise ValueError("Pair resume plan/input mismatch")
    rows = []
    for index, model in enumerate(models):
        remaining = training_deadline - perf_counter()
        if remaining <= 0:
            break
        allocation = budget * 0.35
        folder = out / model
        previous_folder = resume / model if resume else None
        if previous_folder and (previous_folder / "result.json").is_file():
            previous_result = read_json(previous_folder / "result.json")
            if previous_result["status"] == "experimental_adaptation_complete":
                folder.mkdir()
                write(folder / "result.json", previous_result)
                rows.append(dict(model_id=model, adaptation=previous_result))
                continue

        def progress(row):
            write(
                out / "progress.json",
                {
                    **row,
                    "stage": "training",
                    "model_id": model,
                    "elapsed_seconds": perf_counter() - started,
                },
            )

        candidate = fit(
            model,
            request["adaptation"],
            request["development"],
            folder,
            bundle=bundle,
            epochs=request.get("epochs", 30),
            device=request.get("device", "cpu"),
            strategy="staged_last_blocks",
            time_limit_seconds=min(3600, allocation),
            progress=progress,
            resume=previous_folder
            if previous_folder
            and (previous_folder / "checkpoint.json").is_file()
            else None,
        )
        rows.append(dict(model_id=model, adaptation=candidate))
    for row in rows:
        if perf_counter() - started >= budget:
            row["review_status"] = "time_budget"
            continue
        candidate = row["adaptation"]
        if candidate["status"] != "experimental_adaptation_complete":
            row["review_status"] = "incomplete_training"
            continue
        if request.get("cpu_python"):
            try:
                env = dict(os.environ)
                env.pop("PYTHONHOME", None)
                if request.get("cpu_python_home"):
                    env["PYTHONHOME"] = request["cpu_python_home"]
                restored = subprocess.run(
                    [
                        request["cpu_python"],
                        "-B",
                        "-m",
                        "localflight.cli",
                        "worker",
                    ],
                    input=json.dumps(
                        dict(
                            action="verify_adapted",
                            source=candidate["output"],
                            bundle=str(bundle),
                        )
                    ),
                    text=True,
                    capture_output=True,
                    check=True,
                    env=env,
                    timeout=max(0.01, started + budget - perf_counter()),
                )
                row["cpu_restore"] = json.loads(restored.stdout)
            except subprocess.TimeoutExpired:
                row["review_status"] = "time_budget"
                continue
            except subprocess.CalledProcessError as error:
                row["review_status"] = "cpu_restore_failed"
                row["error"] = error.stderr[-4000:]
                continue
        if not request.get("calibrations") or not request.get("evaluations"):
            row["review_status"] = (
                "requires_calibration_and_labelled_evaluation"
            )
            continue
        write(
            out / "progress.json",
            dict(
                stage="review",
                model_id=row["model_id"],
                elapsed_seconds=perf_counter() - started,
            ),
        )
        try:
            report = review(
                candidate["output"],
                request["calibrations"],
                request["evaluations"],
                out / (row["model_id"] + "-review"),
                bundle=bundle,
                events=request.get("events"),
                normal_references=request["development"],
                deadline=started + budget,
            )
        except TimeoutError:
            row["review_status"] = "time_budget"
            continue
        row["review_status"] = report["promotion_check"]["status"]
        row["review"] = report
        review_path = out / (row["model_id"] + "-review") / "review.json"
        row["review_file"] = str(review_path)
        row["review_sha256"] = digest(review_path)
        # Registration preserves a candidate; activation always needs a
        # separate reviewed choice. Rejected candidates remain inspectable.
        if request.get("data_root"):
            row["registered"] = register(
                request["data_root"], candidate["output"], bundle=bundle
            )
    result = dict(
        status="pair_reviewed"
        if len(rows) == 2 and all("review" in r for r in rows)
        else "pair_incomplete",
        tuned_models=rows,
        elapsed_seconds=perf_counter() - started,
        budget_seconds=budget,
        within_budget=perf_counter() - started <= budget,
        automatically_activated=False,
        output=str(out),
    )
    write(out / "result.json", result)
    write(
        out / "progress.json",
        dict(stage="finished", elapsed_seconds=result["elapsed_seconds"]),
    )
    return result
