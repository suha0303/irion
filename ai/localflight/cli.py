"""Local worker protocol; optional CUDA training, CPU analysis, no server."""

import argparse
import json
import sys
from pathlib import Path

from localflight import prepare as localflight_prepare
from localflight import prepared

from .catalog import DEFAULT_BUNDLE, Catalog, digest, read_json
from .logs import inspect_log
from .profiles import list_profiles, save_profile


def comparison_evidence(request):
    if not request.get("comparison_file"):
        return request["comparison"]
    path = Path(request["comparison_file"])
    if digest(path) != request["comparison_sha256"]:
        raise ValueError("Review evidence changed")
    value = read_json(path)
    return value["result"] if request.get("comparison_is_job") else value


def execute(request):
    action = request["action"]
    bundle = request.get("bundle", str(DEFAULT_BUNDLE))
    if action == "normal_plan":
        from .normal_plan import create

        return create(request["manifests"], request["output"])
    if action == "device_info":
        from .adapt_support import device_details

        return device_details(request.get("device", "cpu"))
    if action == "tune":
        from .tune import run

        return run(request, bundle=bundle)
    if action == "verify_adapted":
        import torch

        from .registry import validate

        result = validate(request["source"], bundle)
        return dict(
            status="cpu_restore_passed",
            torch=str(torch.__version__),
            model_sha256=result["model_sha256"],
        )
    if action == "tune_select":
        from .tune import select_reviewed

        return select_reviewed(
            request["data_root"],
            request["model_id"],
            comparison_evidence(request),
            bundle=bundle,
        )
    if action == "adapt_review":
        from .adapt_review import review

        return review(
            request["adapted"],
            request["calibrations"],
            request["evaluations"],
            request["output"],
            bundle=bundle,
            events=request.get("events"),
            normal_references=request.get("normal_references"),
        )
    if action.startswith("calibration_"):
        from . import calibration
        from .registry import active_folder

        root = request["data_root"]
        if action == "calibration_list":
            return calibration.list_calibrations(root)
        if action == "calibration_select":
            return calibration.select(
                root, request["id"], request.get("enabled", True)
            )
        if action == "calibration_fit":
            metadata = prepared.load(request["manifests"][0], normal=True)[0]
            adapted = (
                active_folder(
                    root,
                    metadata["profile"],
                    request["model_id"],
                    bundle=bundle,
                )
                if request.get("use_active")
                else None
            )
            if request.get("normal_references"):
                return calibration.fit_protocol(
                    root,
                    request["model_id"],
                    request["normal_references"],
                    request["manifests"],
                    bundle=bundle,
                    adapted=adapted,
                )
            return calibration.fit(
                root,
                request["model_id"],
                request["manifests"],
                bundle=bundle,
                adapted=adapted,
                quantile=request.get("quantile", 0.99),
            )
    if action == "admit_live":
        from .calibration import admit_completed

        return admit_completed(
            request["manifest"],
            request["output"],
            role=request["role"],
            health=request["health"],
            evidence=request["evidence"],
        )
    if action == "live_validate":
        from .live import validate

        return validate(request)
    if action == "catalog":
        return dict(
            models=Catalog(bundle).list(), device="cpu", external_server=False
        )
    if action == "inspect":
        return inspect_log(request["path"])
    if action == "profiles":
        return dict(profiles=list_profiles(request["data_root"]))
    if action == "save_profile":
        return save_profile(request["data_root"], request["profile"])
    if action == "benchmark":
        from .benchmark import benchmark

        return benchmark(bundle, repeats=request.get("repeats", 5))
    if action in {"prepare", "timeline"}:
        return localflight_prepare.prepare(
            request["path"],
            request["profile"],
            request.get("output"),
            preview=action == "timeline",
            **request["settings"],
        )
    if action == "analyze":
        from .analysis import analyze_run

        if not 1 <= len(request["models"]) <= 2:
            raise ValueError("Select one or two models")
        manifests = (
            request["manifests"]
            if "manifests" in request
            else [request["manifest"]]
        )
        return analyze_run(
            request["models"],
            manifests,
            bundle=bundle,
            data_root=request.get("data_root"),
            use_active=request.get("use_active", False),
        )
    if action.startswith("model_"):
        from . import registry

        root = request["data_root"]
        if action == "model_list":
            return registry.list_models(root, bundle=bundle)
        if action == "model_register":
            return registry.register(root, request["source"], bundle=bundle)
        if action == "model_import":
            return registry.import_model(root, request["source"], bundle=bundle)
        if action == "model_export":
            return registry.export_model(
                root, request["model_id"], request["output"], bundle=bundle
            )
        if action == "model_select":
            return registry.select_model(
                root,
                request["model_id"],
                comparison_evidence(request),
                bundle=bundle,
            )
        if action == "model_rollback":
            return registry.rollback(root, request["event_id"], bundle=bundle)
    if action == "adapt":
        from .adapt import fit, write

        output = Path(request["output"])
        return fit(
            request["model_id"],
            request["adaptation"],
            request["development"],
            output,
            bundle=bundle,
            epochs=request.get("epochs", 5),
            resume=request.get("resume"),
            device=request.get("device", "cpu"),
            strategy=request.get("strategy", "head"),
            time_limit_seconds=request.get("time_limit_seconds", 1800),
            progress=lambda row: write(output / "progress.json", row),
        )
    if action == "compare":
        from .analysis import compare

        return compare(request["adapted"], request["manifest"], bundle=bundle)
    raise ValueError("Unknown local operation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["catalog", "inspect", "benchmark", "worker"]
    )
    parser.add_argument("--path", type=Path)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "worker":
            body = sys.stdin.read(65537)
            if len(body) > 65536:
                raise ValueError("Worker request exceeds 64 KiB")
            request = json.loads(body)
        else:
            request = dict(action=args.action, bundle=str(args.bundle))
            if args.path:
                request["path"] = str(args.path)
        result = execute(request)
        encoded = (
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as f:
                f.write(encoded)
        print(encoded, end="")
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(
            json.dumps({"error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
