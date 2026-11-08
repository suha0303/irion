"""Publish a read-only final artifact inventory after verified allocation
release.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + ".tmp")
    q.write_text(json.dumps(v, indent=2) + "\n", encoding="utf-8")
    q.replace(p)


def safe(root, relative):
    p = Path(relative)
    if p.is_absolute() or not p.parts or ".." in p.parts:
        raise ValueError("Unsafe relative artifact path")
    target = Path(root) / p
    if any(x.is_symlink() for x in (target, *target.parents)):
        raise ValueError("Artifact symlink forbidden")
    return target


def export(base):
    base = Path(base).absolute()
    output = base / "output"
    plan = read(output / "plan.json")
    state = read(output / "state.json")
    if (
        state["status"] != "REPORT_READY_STOPPED"
        or not state.get("allocation_released")
        or state.get("next_allocation_queued") is not False
    ):
        raise ValueError("Final verification and allocation release required")
    queue = subprocess.check_output(
        ["squeue", "-u", plan["owner"], "-h", "-o", "%i"], text=True
    )
    ids = {
        str(state["allocation"]["leader"]),
        *(str(i) for i in state["allocation"]["components"].values()),
    }
    if any(x.split("+")[0] in ids for x in queue.split()):
        raise ValueError("Own allocation still present")
    selected = read(output / "selected-models.json")
    results = read(output / "final-results.json")
    expected = {e["id"] for e in plan["models"]}
    if (
        len(expected) != 16
        or set(selected["models"]) != expected
        or set(results) != expected
    ):
        raise ValueError("All16 models required")
    if selected["final_test_used_for_selection"]:
        raise ValueError("Final data used for selection")
    delivery = base / "delivery"
    delivery.mkdir(exist_ok=True)
    lock = (delivery / "export.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return {"status": "EXPORT_IN_PROGRESS"}
    bundle = delivery / "final"
    bundle.mkdir(exist_ok=True)
    if (bundle / "MANIFEST.json").exists():
        existing = read(bundle / "MANIFEST.json")
        if existing["selected_manifest_sha256"] != sha(
            output / "selected-models.json"
        ):
            raise ValueError("Published selection differs")
        return {
            "status": "FINAL_EXPORT_READY",
            "path": str(bundle),
            "manifest_sha256": sha(bundle / "MANIFEST.json"),
            "files": len(existing["files"]),
        }
    input_hashes = read(Path(plan["inputs"]) / "cache.json")["files"]
    inventory = {}
    index = {}

    def add(source, destination, expected_sha=None):
        source = Path(source)
        if (
            any(p.is_symlink() for p in (source, *source.parents))
            or not source.is_file()
        ):
            raise ValueError("Regular artifact required")
        digest = sha(source)
        if expected_sha is not None and digest != expected_sha:
            raise ValueError("Changed artifact: " + str(source))
        target = safe(bundle, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != digest:
                raise ValueError("Previous delivery differs")
        else:
            os.link(source, target)
        inventory[destination] = digest

    for name in sorted(expected):
        if not re.fullmatch("[a-zA-Z0-9_-]+", name):
            raise ValueError("Invalid model ID")
        chosen = selected["models"][name]
        training = Path(chosen["folder"])
        integrity = read(training / "integrity.json")
        allowed = [
            Path(plan["baseline"]),
            output,
            *[Path(p) for p in plan.get("preserved_outputs", [])],
        ]
        if not any(
            training.resolve().is_relative_to(p.resolve()) for p in allowed
        ):
            raise ValueError("Unexpected model origin")
        if integrity["model.safetensors"] != chosen["model_sha256"]:
            raise ValueError("Selected weight binding mismatch")
        for filename in ("model.safetensors", "model-spec.json", "scaler.json"):
            add(
                training / filename,
                f"models/{name}/{filename}",
                integrity[filename],
            )
        add(
            Path(plan["inputs"]) / "main/scaler.npz",
            f"models/{name}/scaler.npz",
            input_hashes["main/scaler.npz"],
        )
        for filename in ("history.json", "best.json", "result.json"):
            add(
                training / filename,
                f"models/{name}/training/{filename}",
                integrity[filename],
            )
        add(
            training / "integrity.json",
            f"models/{name}/training/source-integrity.json",
        )
        request = read(state["final_requests"][name]["request"])
        final = Path(request["output"])
        if not any(
            final.resolve().is_relative_to(p.resolve())
            for p in [
                output,
                *[Path(p) for p in plan.get("preserved_outputs", [])],
            ]
        ):
            raise ValueError("Unexpected evaluation path")
        result = read(final / "result.json")
        if (
            result["status"] != "FINAL_VERIFICATION_COMPLETE"
            or result["model_sha256"] != chosen["model_sha256"]
            or result["final_used_for_selection"]
        ):
            raise ValueError("Final evaluation binding mismatch")
        if sha(final / "calibration.json") != result["calibration_sha256"]:
            raise ValueError("Calibration changed")
        for relative, digest in read(final / "integrity.json").items():
            add(
                safe(final, relative),
                f"models/{name}/evaluation/{relative}",
                digest,
            )
        add(
            final / "integrity.json", f"models/{name}/evaluation/integrity.json"
        )
        if plan.get("development_required"):
            development = state["development"][str(training)]
            if development["status"] != "DEVELOPMENT_EVALUATION_COMPLETE":
                raise ValueError(
                    "Selected model development evaluation missing"
                )
            dev = Path(development["result"]["folder"])
            if not any(
                dev.resolve().is_relative_to(p.resolve())
                for p in [
                    output,
                    *[Path(p) for p in plan.get("preserved_outputs", [])],
                ]
            ):
                raise ValueError("Unexpected development folder")
            for relative, digest in read(dev / "integrity.json").items():
                add(
                    safe(dev, relative),
                    f"models/{name}/development/{relative}",
                    digest,
                )
            add(
                dev / "integrity.json",
                f"models/{name}/development/integrity.json",
            )
        index[name] = {
            "folder": f"models/{name}",
            "selected_lr": chosen["lr"],
            "weights_sha256": chosen["model_sha256"],
            "source_training": str(training),
        }
        if plan.get("final_per_model_when_search_stopped"):
            selection = output / "final-selections" / f"{name}.json"
            receipt = read(selection)
            if (
                receipt["selected"] != chosen
                or sha(selection) != request["selected_manifest_sha256"]
            ):
                raise ValueError("Per-model final selection changed")
            add(selection, f"models/{name}/evaluation/frozen-selection.json")
    for name in (
        "plan.json",
        "state.json",
        "selected-models.json",
        "final-results.json",
    ):
        add(output / name, "provenance/" + name)
    if plan.get("development_required"):
        add(
            output / "development-results.json",
            "provenance/development-results.json",
        )
        add(
            Path(plan["development_inputs"]) / "cache.json",
            "provenance/development-cache.json",
            plan["development_cache_sha256"],
        )
    for key in ("cache_sha256", "final_cache_sha256"):
        folder = Path(
            plan["inputs"] if key == "cache_sha256" else plan["final_inputs"]
        )
        add(folder / "cache.json", "provenance/" + key + ".json", plan[key])
    add(
        Path(plan["inputs"]) / "main/fit-manifest.json",
        "provenance/fit-manifest.json",
    )
    for relative, digest in plan["source_sha256"].items():
        add(safe(plan["code"], relative), "code/" + relative, digest)
    write(bundle / "model-index.json", index)
    inventory["model-index.json"] = sha(bundle / "model-index.json")
    write(
        bundle / "MANIFEST.json",
        {
            "schema": "drone-ai.final-delivery/1",
            "source": str(base),
            "files": inventory,
            "models": sorted(expected),
            "selected_manifest_sha256": sha(output / "selected-models.json"),
            "no_training_data_copy": True,
            "no_optimizer_checkpoint_copy": True,
        },
    )
    return {
        "status": "FINAL_EXPORT_READY",
        "path": str(bundle),
        "manifest_sha256": sha(bundle / "MANIFEST.json"),
        "files": len(inventory),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base")
    a = p.parse_args()
    print(json.dumps(export(a.base)))
