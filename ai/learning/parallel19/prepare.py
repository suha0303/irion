"""Build portable normalized normal-only caches from independently audited
aggregates.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from execution.common import read, sha, write
from learning.collection19.aggregate import (
    load_shared_scaler,
    whole_manifest_paths,
)
from learning.history19.data import load
from learning.run_health19.data import iter_manifests


def prepare(task, output):
    task, output = Path(task), Path(output)
    completion = read(task / "completion.json")
    if (
        completion["status"] != "MAIN19_INPUTS_PASSED_STOPPED"
        or completion["done"] != 1674
        or completion["failed"]
    ):
        raise ValueError("Complete1674 and actual termination required")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "drone-ai.parallel19.cache/1",
        "scopes": {},
        "files": {},
        "source_completion": {
            "path": str(task / "completion.json"),
            "sha256": sha(task / "completion.json"),
        },
        "no_calibration_development_final_inputs": True,
    }
    expected = read(task / "expected-producer.json")
    for scope, counts in [("lr108", (81, 27)), ("main", (540, 81))]:
        checkpoint = read(task / f"operator/aggregate-{scope}-complete.json")
        result, root = checkpoint["aggregate"], Path(checkpoint["output"])
        if not result["passed"] or not result["actual_data_verified"]:
            raise ValueError("Independent input audit required")
        for key in ("receipt", "independent_audit"):
            ref = result[key]
            if (
                sha(ref["path"]) != ref["sha256"]
                or not read(ref["path"])["passed"]
            ):
                raise ValueError("Aggregate receipt changed")
        protection = read(root / "protection.json")
        dest = output / scope
        dest.mkdir()
        arrays, scaler, binding = load(
            root / "history19-manifest.json",
            cache=dest / "forecast",
            expected_producer=expected,
            protection=protection,
        )
        fit_manifest = read(root / "history19-manifest.json")
        rows = fit_manifest["runs"]
        ids = {
            role: sorted(r["run_id"] for r in rows if r["role"] == role)
            for role in ("normal_train", "normal_validation")
        }
        if tuple(map(len, ids.values())) != counts:
            raise ValueError("Frozen role counts changed")
        shared = load_shared_scaler(root)
        if not np.array_equal(
            shared.mean, scaler["x_mean"]
        ) or not np.array_equal(shared.scale, scaler["x_scale"]):
            raise ValueError("Representations must share train-only scaler")
        # Shards can cross role boundaries. Select metadata first, never decode
        # calibration/test arrays. Temporary regular hardlinks preserve the
        # bytes.
        with tempfile.TemporaryDirectory(
            dir=output, prefix="fit-selection-"
        ) as work:
            work = Path(work)
            manifests = []
            for path in whole_manifest_paths(root):
                content = read(path)
                content["runs"] = [
                    r for r in content["runs"] if r["role"] in ids
                ]
                if not content["runs"]:
                    continue
                for row in content["runs"]:
                    source = (path.parent / row["path"]).resolve()
                    if not source.is_relative_to(root.resolve()):
                        raise ValueError("Run escapes aggregate")
                    name = row["run_id"] + ".npz"
                    os.link(source, work / name)
                    row["path"] = name
                selected = work / path.name
                write(selected, content)
                manifests.append(selected)
            runs = list(
                iter_manifests(
                    manifests,
                    expected_producer=expected,
                    protection=protection,
                    usage="fit",
                )
            )
        for role in ids:
            selected = sorted(
                (r for r in runs if r.role == role), key=lambda r: r.run_id
            )
            if [r.run_id for r in selected] != ids[role]:
                raise ValueError("Online and whole-run membership differs")
            folder = dest / "run_health" / role
            folder.mkdir(parents=True)
            lengths = np.array(
                [len(r.values) for r in selected], dtype=np.int64
            )
            values = np.lib.format.open_memmap(
                folder / "X.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(selected), int(lengths.max()), 19),
            )
            values[:] = 0
            for i, run in enumerate(selected):
                values[i, : len(run.values)] = shared.transform(run)
            values.flush()
            np.save(folder / "lengths.npy", lengths)
            with np.load(root / f"{role}.npz", allow_pickle=False) as z:
                run_ids = z["run_id"]
            np.save(dest / "forecast" / role / "run_id.npy", run_ids)
        write(dest / "scaler.json", shared.to_dict())
        shutil.copy2(root / "scaler.npz", dest / "scaler.npz")
        write(dest / "fit-manifest.json", fit_manifest)
        record["scopes"][scope] = {
            "binding": binding,
            "roles": ids,
            "counts": dict(zip(ids, counts)),
            "windows": {role: len(a["X"]) for role, a in arrays.items()},
            "audit": result["independent_audit"],
        }
        print("prepared", scope, record["scopes"][scope]["windows"], flush=True)
        del arrays, runs
    for role in ("normal_train", "normal_validation"):
        if not set(record["scopes"]["lr108"]["roles"][role]) <= set(
            record["scopes"]["main"]["roles"][role]
        ):
            raise ValueError("LR role must be same-role main subset")
    record["files"] = {
        str(p.relative_to(output)): sha(p)
        for p in output.rglob("*")
        if p.is_file()
    }
    write(output / "cache.json", record)
    print("CACHE_READY", output, flush=True)


if __name__ == "__main__":
    import sys

    prepare(sys.argv[1], sys.argv[2])
