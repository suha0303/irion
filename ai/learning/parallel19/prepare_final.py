"""Package audited validation/calibration/final inputs; no model or scoring."""

import argparse
import collections
import json
import shutil
from pathlib import Path

import numpy as np

from learning.admission19 import (
    check_producer,
    check_protection,
    source_evidence,
    validate_lineages,
)
from learning.collection19.aggregate import whole_manifest_paths

from .control import read, sha, write


def prepare(task, output, cohort="final"):
    if cohort not in {"final", "development"}:
        raise ValueError("Unknown evaluation cohort")
    normal_count, anomaly_count = (486, 243) if cohort == "final" else (162, 81)
    task, output = Path(task), Path(output)
    completion = read(task / "completion.json")
    if (
        completion["status"] != "MAIN19_INPUTS_PASSED_STOPPED"
        or completion["done"] != 1674
        or completion["failed"]
    ):
        raise ValueError("Incomplete dataset")
    checkpoint = read(task / "operator/aggregate-main-complete.json")
    root = Path(checkpoint["output"])
    audit = checkpoint["aggregate"]
    for key in ("receipt", "independent_audit"):
        ref = audit[key]
        if sha(ref["path"]) != ref["sha256"] or not read(ref["path"])["passed"]:
            raise ValueError("Audit changed")
    output.mkdir(parents=True, exist_ok=False)
    protection = read(root / "protection.json")
    producer = read(task / "expected-producer.json")
    scaler = dict(np.load(root / "scaler.npz", allow_pickle=False))
    whole = {}
    for p in whole_manifest_paths(root):
        for row in read(p)["runs"]:
            whole[row["run_id"]] = (row, p.parent)
    rows = [
        json.loads(l)
        for l in (root / "online-index.jsonl").read_text().splitlines()
    ]
    selected = [
        r
        for r in rows
        if r["role"] in {"normal_validation", "normal_calibration"}
        or r["cohort"] == cohort
    ]
    counts = collections.Counter((r["role"], r["cohort"]) for r in selected)
    if counts != {
        ("normal_validation", "normal_validation"): 81,
        ("normal_calibration", "normal_calibration"): 81,
        ("normal_test", cohort): normal_count,
        ("anomaly_test", cohort): anomaly_count,
    }:
        raise ValueError("Final inventory changed")
    record = {
        "runs": [],
        "audit": audit["independent_audit"],
        "counts": {
            "validation": 81,
            "calibration": 81,
            cohort + "_normal": normal_count,
            cohort + "_anomaly": anomaly_count,
        },
        "model_scoring_performed": False,
        "files": {},
    }
    for i, ref in enumerate(selected):
        path = Path(ref["manifest"]["path"])
        if sha(path) != ref["manifest"]["sha256"]:
            raise ValueError("Online manifest changed")
        manifest = read(path)
        row = manifest["run"]
        rid = row["run_id"]
        w, parent = whole[rid]
        check_producer(manifest["producer"], producer)
        source_evidence(row, manifest["producer"])
        check_protection(
            [row],
            protection,
            usage="final_test"
            if row.get("cohort") == "final"
            else "input_check",
        )
        online = path.parent / manifest["file"]["path"]
        physical = parent / w["path"]
        if (
            sha(online) != manifest["file"]["sha256"]
            or sha(physical) != w["sha256"]
        ):
            raise ValueError("Audited array changed")
        with (
            np.load(online, allow_pickle=False) as a,
            np.load(physical, allow_pickle=False) as b,
        ):
            x = ((a["X"] - scaler["x_mean"]) / scaler["x_scale"]).astype("f4")
            y = ((a["Y"] - scaler["y_mean"]) / scaler["y_scale"]).astype("f4")
            values = b["values"]
            times = b["time_us"]
            if (
                not np.isfinite(x).all()
                or not np.isfinite(y).all()
                or not np.isfinite(values).all()
            ):
                raise ValueError("Nonfinite input")
            if (
                x.shape[1:] != (100, 19)
                or y.shape[1:] != (10, 4)
                or values.shape[1:] != (19,)
                or np.any(np.diff(times) != 100000)
            ):
                raise ValueError("Shape/time contract")
            dest = output / (rid + ".npz")
            np.savez(
                dest,
                X=x,
                Y=y,
                values=values,
                time_us=times,
                origin_us=a["origin_us"],
                segment_id=a["segment_id"],
            )
        metadata = {
            **row,
            "path": dest.name,
            "sha256": sha(dest),
            "original_online_sha256": manifest["file"]["sha256"],
            "original_whole_sha256": w["sha256"],
        }
        record["runs"].append(metadata)
        if (i + 1) % 100 == 0:
            print("prepared", i + 1, len(selected), flush=True)
    fit = read(root / "history19-manifest.json")["runs"]
    train = [r for r in fit if r["role"] == "normal_train"]
    validate_lineages(train + record["runs"])
    shutil.copy2(root / "scaler.npz", output / "scaler.npz")
    shutil.copy2(root / "whole-scaler.json", output / "scaler.json")
    record["files"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write(output / "cache.json", record)
    print("FINAL_INPUTS_READY", len(selected), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("output")
    p.add_argument(
        "--cohort", choices=["final", "development"], default="final"
    )
    a = p.parse_args()
    prepare(a.task, a.output, a.cohort)
