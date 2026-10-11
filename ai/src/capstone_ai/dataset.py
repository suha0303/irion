"""Selection metadata is consumed here, never passed into a neural forward call."""

import json
from pathlib import Path

import numpy as np

from .cli import process
from .mapping import FEATURES, TARGETS, capacity_ah
from .source import dump, read_json, sha


def validate_split(rows):
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        parent[find(a)] = find(b)

    for r in rows:
        keys = ["run:" + r["run_id"]]
        keys.extend("run:" + attempt for attempt in r.get("previous_attempts", []))
        for field in [
            "paired_case_id",
            "parent_attempt_id",
            "raw_sha256",
            "raw_flight_id",
            "continuation_id",
        ]:
            if r.get(field):
                keys.append(("run:" if field == "parent_attempt_id" else field + ":") + r[field])
        for k in keys[1:]:
            union(keys[0], k)
        if (
            r["split"] in {"train", "validation", "calibration"}
            and r["health_eligibility"] != "confirmed_healthy"
        ):
            raise ValueError("nonhealthy fit membership")
    groups: dict[str, str] = {}
    for r in rows:
        key = find("run:" + r["run_id"])
        if key in groups and groups[key] != r["split"]:
            raise ValueError("raw/paired/retry/continuation split leakage")
        groups[key] = r["split"]
    if set(groups.values()) != {"train", "validation", "calibration", "test"}:
        raise ValueError("insufficient_independent_groups")
    return groups


def feature_transform(x, config=None):
    x = np.asarray(x, dtype=np.float64).copy()
    x[..., FEATURES.index("pack_voltage")] /= 4
    x[..., FEATURES.index("pack_current")] /= capacity_ah(config)
    return x


def target_transform(y):
    y = np.asarray(y, dtype=np.float64).copy()
    y[..., -1] /= 4
    return y


def verify_input(path, selection):
    integrity = read_json(path / "integrity.json")
    for field, reference_name in [
        ("integrity_sha256", "integrity.json"),
        ("source_lock_sha256", "reference/source_lock.json"),
    ]:
        if selection.get(field) and sha(path / reference_name) != selection[field]:
            raise ValueError("handoff hash mismatch: " + reference_name)
    for name, entry in integrity.items():
        file = (path / name).resolve()
        if not file.is_relative_to(path):
            raise ValueError("integrity path escapes run")
        if (
            not file.is_file()
            or file.stat().st_size != entry["size"]
            or sha(file) != entry["sha256"]
        ):
            raise ValueError("run integrity mismatch: " + name)
    for required in [
        "run.json",
        "observations/metadata.json",
        "observations/streams.jsonl",
        "observations/watermarks.jsonl",
        "evaluation/scenario.json",
    ]:
        if required not in integrity:
            raise ValueError("required integrity entry missing: " + required)
    return sha(path / "integrity.json")


def build(config_path, out):
    cfg = read_json(config_path)
    out = Path(out)
    if (out / "locked_selection.json").exists():
        raise ValueError("dataset already locked; use a new output revision")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for selection in cfg["runs"]:
        path = Path(selection["path"]).resolve()
        run = read_json(path / "run.json")
        scenario = read_json(path / "evaluation/scenario.json")
        if run["status"] != "completed" or run["origin"] != "px4_sitl":
            raise ValueError("dataset requires completed real SITL run")
        integrity_hash = verify_input(path, selection)
        if selection.get("run_id", run["run_id"]) != run["run_id"]:
            raise ValueError("selection run identity mismatch")
        for name in ["health_eligibility", "paired_case_id", "continuity"]:
            if name in selection and selection[name] != scenario.get(name):
                raise ValueError("selection metadata mismatch: " + name)
        row = {
            **selection,
            "run_id": run["run_id"],
            "parent_attempt_id": run["parent_attempt_id"],
            "paired_case_id": scenario["paired_case_id"],
            "health_eligibility": scenario["health_eligibility"],
            "continuity": scenario.get("continuity", "unknown"),
            "raw_sha256": sha(path / "observations/streams.jsonl"),
            "selection_evidence_sha256": sha(path / "evaluation/scenario.json"),
            "integrity_verified_sha256": integrity_hash,
        }
        if row["continuity"] != "independent_reset":
            raise ValueError("continuity requires explicit protocol")
        rows.append(row)
    groups = validate_split(rows)
    dump(
        out / "locked_selection.json",
        {
            "protocol": "x500_run_holdout",
            "runs": rows,
            "groups": groups,
            "configuration_sha256": sha(config_path),
        },
    )
    arrays: dict[str, list[tuple]] = {s: [] for s in ["train", "validation", "calibration", "test"]}
    train_frames = []
    preprocess = read_json(cfg["preprocessing"])
    for row in rows:
        folder = out / "canonical" / row["run_id"]
        report = process(row["path"], folder, preprocess)
        if not report["closed"] or report["valid_frames"] == 0 or report["windows"] == 0:
            raise ValueError("incomplete or unusable selected run: " + row["run_id"])
        frames = [json.loads(s) for s in (folder / "frames.jsonl").read_text().splitlines()]
        windows = [json.loads(s) for s in (folder / "windows.jsonl").read_text().splitlines()]
        lookup = {(f["clock_id"], f["segment_id"], f["time_us"]): f for f in frames}
        accepted = 0
        for w in windows:
            target = [lookup.get((w["clock_id"], w["segment_id"], t)) for t in w["target_times_us"]]
            if any(f is None or any(f["quality"][n] != "valid" for n in TARGETS) for f in target):
                continue
            target_values = [[f["values"][n] for n in TARGETS] for f in target if f is not None]
            arrays[row["split"]].append(
                (w["X"], w["C"], target_values, row["run_id"], w["origin_us"], w["segment_id"])
            )
            accepted += 1
        if row["split"] == "train":
            # Count each valid, airborne frame once, regardless of overlapping windows.
            train_frames.extend(
                [
                    [f["values"][n] for n in FEATURES]
                    for f in frames
                    if f["operating_state"] == "airborne"
                    and all(f["quality"][n] == "valid" for n in FEATURES)
                ]
            )
        row.update(
            windows=accepted,
            excluded_targets=len(windows) - accepted,
            preprocessing=report["output_hashes"],
        )
        print(row["run_id"], row["split"], accepted, flush=True)
    for split, samples in arrays.items():
        if not samples:
            raise ValueError("no valid windows for " + split)
        x, c, y, ids, times, segments = zip(*samples)
        np.savez_compressed(
            out / (split + ".npz"),
            X=feature_transform(x, preprocess),
            C=np.array(c),
            Y=target_transform(y),
            run_id=np.array(ids),
            origin_us=np.array(times),
            segment=np.array(segments),
        )
    unique_x = feature_transform(train_frames, preprocess)
    train = np.load(out / "train.npz")
    target_indices = [FEATURES.index(n) for n in TARGETS]
    scaler = {
        "x_mean": unique_x.mean(0),
        "x_scale": np.maximum(unique_x.std(0), 1e-5),
        "c_mean": train["C"].mean(0),
        "c_scale": np.maximum(train["C"].std(0), 1e-5),
    }
    scaler["y_mean"] = scaler["x_mean"][target_indices]
    scaler["y_scale"] = scaler["x_scale"][target_indices]
    np.savez(out / "scaler.npz", **scaler)
    dump(
        out / "manifest.json",
        {
            "runs": rows,
            "groups": groups,
            "profile": "common19_ctx6",
            "target_order": TARGETS,
            "scaler_fit": "unique valid airborne normal train frames; context once per train origin",
            "unique_train_frames": len(unique_x),
            "counts": {s: len(a) for s, a in arrays.items()},
            "hashes": {p.name: sha(p) for p in out.glob("*.npz")},
        },
    )
