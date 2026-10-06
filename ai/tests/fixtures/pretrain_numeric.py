"""Analytical canonical fixtures only. No source observations, training or evaluator."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from preprocessing_tools import derived as audit


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def jsonlines(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_dataset(root, rows=None):
    if rows is None:
        rows = [{"run_id": "synthetic-" + r, "role": r} for r in audit.FIT]
    root.mkdir(parents=True)
    samples = {r: [] for r in audit.FIT}
    selected, contexts, reports = [], [], []
    for row in rows:
        role = row["role"]
        folder = root / "canonical" / row["run_id"]
        folder.mkdir(parents=True)
        frames = []
        factor = 1 if role == "normal_train" else 10
        for j in range(621):
            value = {name: float(factor) for name in audit.FEATURES}
            value.update(
                body_rate_p=1.0 * factor,
                body_rate_q=2.0 * factor,
                body_rate_r=3.0 * factor,
                pack_voltage=(8.0 if j < 600 else 12.0) * factor,
                pack_current=6.0 * factor,
            )
            q = dict.fromkeys(audit.FEATURES, "valid")
            if j == 0:
                value["pack_voltage"] = 1000.0
            if j == 1:
                value["pack_voltage"] = 999.0
                q["pack_voltage"] = "invalid"
            c = None if j < 600 else [4.0, 0.0, 1.0, 0.0, 0.2, 0.3]
            if j in (600, 610, 620):
                c = {
                    600: [3.0, 0.0, 1.0, 0.0, 0.1, 0.2],
                    610: [5.0, 0.0, 1.0, 0.0, 0.3, 0.4],
                    620: [999.0, 0.0, 1.0, 0.0, 0.5, 0.6],
                }[j]
            frames.append(
                dict(
                    run_id=row["run_id"],
                    clock_id="synthetic-clock",
                    segment_id=0,
                    time_us=j * 100000,
                    feature_order=audit.FEATURES,
                    values=value,
                    quality=q,
                    warmup=j < 600,
                    context=c,
                    operating_state="ground" if j == 0 else "airborne",
                    lineage={},
                )
            )
        windows = []
        for j in (600, 610, 620):
            w = dict(
                run_id=row["run_id"],
                clock_id="synthetic-clock",
                segment_id=0,
                origin_us=j * 100000,
                feature_order=audit.FEATURES,
                history_times_us=[i * 100000 for i in range(j - 99, j + 1)],
                target_times_us=[i * 100000 for i in range(j + 1, j + 11)],
                context_start_us=(j - 600) * 100000,
                X=[[f["values"][n] for n in audit.FEATURES] for f in frames[j - 99 : j + 1]],
                C=frames[j]["context"],
            )
            windows.append(w)
            if role in audit.FIT and j < 620:
                x = np.asarray(w["X"], dtype=np.float64)
                x[:, 17] /= 4
                x[:, 18] /= 12
                y = np.asarray(
                    [[f["values"][n] for n in audit.TARGETS] for f in frames[j + 1 : j + 11]]
                )
                y[:, -1] /= 4
                samples[role].append((x, np.asarray(w["C"]), y, row["run_id"], w["origin_us"], 0))
                if role == "normal_train":
                    contexts.append(w["C"])
        exclusions = [
            dict(time_us=j * 100000, segment_id=0, reasons=["context_warmup"])
            for j in range(0, 600, 10)
        ]
        for name, values in (("frames", frames), ("windows", windows), ("exclusions", exclusions)):
            jsonlines(folder / (name + ".jsonl"), values)
        summary = dict(
            run_id=row["run_id"],
            origin="synthetic_fixture",
            closed=True,
            frames=621,
            valid_frames=620,
            windows=3,
            feature_order=audit.FEATURES,
            exclusions={"context_warmup": 60},
        )
        write(folder / "summary.json", summary)
        reports.append(
            dict(
                run_id=row["run_id"],
                role=role,
                summary=summary,
                accepted_targets=2 if role in audit.FIT else None,
                excluded_targets=1 if role in audit.FIT else None,
            )
        )
        if role == "normal_train":
            a = np.asarray(
                [[f["values"][n] for n in audit.FEATURES] for f in frames[2:]], dtype=np.float64
            )
            a[:, 17] /= 4
            a[:, 18] /= 12
            selected.extend(a)
    for role, values in samples.items():
        columns = list(zip(*values))
        np.savez(
            root / (role + ".npz"),
            **{
                k: np.asarray(v)
                for k, v in zip(("X", "C", "Y", "run_id", "origin_us", "segment"), columns)
            },
        )
    x, c = np.asarray(selected), np.asarray(contexts)
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-5)
    indices = [audit.FEATURES.index(n) for n in audit.TARGETS]
    np.savez(
        root / "scaler.npz",
        x_mean=xm,
        x_scale=xs,
        c_mean=c.mean(0),
        c_scale=np.maximum(c.std(0), 1e-5),
        y_mean=xm[indices],
        y_scale=xs[indices],
    )
    manifest = dict(
        runs=rows,
        audit=reports,
        unique_train_frames=len(x),
        hashes={p.name: audit.sha(p) for p in root.glob("*.npz")},
    )
    write(root / "manifest.json", manifest)
    return {"runs": rows}


def refresh_hash(dataset, name):
    m = audit.read(dataset / "manifest.json")
    m["hashes"][name] = audit.sha(dataset / name)
    write(dataset / "manifest.json", m)


def test_independent_scaler_and_accounting(tmp_path):
    data = tmp_path / "dataset"
    frozen = make_dataset(data)
    result = audit.inspect_dataset(data, frozen)
    train = result["by_role"]["normal_train"]
    assert (
        train["windows"] == 3 and train["accepted_targets"] == 2 and train["excluded_targets"] == 1
    )
    assert train["warmup_frames"] == 600 and train["valid_airborne_frames"] == 619
    assert train["observed_airborne_seconds"] == 62 and train["valid_airborne_seconds"] == 61.9
    assert train["quality_counts"]["pack_voltage"] == {"valid": 620, "invalid": 1}
    assert train["target_drop_detail"]["missing_target_frames"] == 10
    assert result["scaler_observations"]["frame_rows"] == 619
    assert result["scaler_observations"]["context_rows"] == 2
    with np.load(data / "scaler.npz") as scaler:
        assert scaler["x_mean"][17] == pytest.approx((598 * 2 + 21 * 3) / 619)
        assert scaler["x_scale"][17] == pytest.approx((598 * 21) ** 0.5 / 619)
        assert scaler["c_mean"][0] == 4 and scaler["c_scale"][0] == 1
        assert scaler["y_mean"][-1] != 3  # Future Y statistics must not fit the target scaler.
    assert result["npz_max_abs_error"] == 0


@pytest.mark.parametrize(
    "kind", ["history_weighted", "excluded_context", "future_target", "validation_contamination"]
)
def test_wrong_scaler_rejected_even_after_rehash(tmp_path, kind):
    data = tmp_path / "dataset"
    frozen = make_dataset(data)
    with np.load(data / "scaler.npz") as f:
        values = dict(f)
    if kind == "history_weighted":
        values["x_mean"][17] = 3
    elif kind == "excluded_context":
        values["c_mean"][0] = (3 + 5 + 999) / 3
    elif kind == "future_target":
        values["y_mean"][-1] = 3
    else:
        values["x_mean"][0] = 5.5
    np.savez(data / "scaler.npz", **values)
    refresh_hash(data, "scaler.npz")
    with pytest.raises(ValueError, match="scaler/"):
        audit.inspect_dataset(data, frozen)


def test_duplicate_equal_frame_is_not_silently_deduplicated(tmp_path):
    data = tmp_path / "dataset"
    frozen = make_dataset(data)
    path = data / "canonical" / frozen["runs"][0]["run_id"] / "frames.jsonl"
    frames = audit.records(path)
    frames.insert(2, copy.deepcopy(frames[2]))
    jsonlines(path, frames)
    with pytest.raises(ValueError, match="duplicate canonical"):
        audit.inspect_dataset(data, frozen)


@pytest.mark.parametrize("change", ["future_history", "duplicate_origin"])
def test_window_causality_and_once_per_origin(tmp_path, change):
    data = tmp_path / "dataset"
    frozen = make_dataset(data)
    folder = data / "canonical" / frozen["runs"][0]["run_id"]
    windows = audit.records(folder / "windows.jsonl")
    if change == "future_history":
        windows[0]["history_times_us"][-1] += 100000
    else:
        windows.insert(1, copy.deepcopy(windows[0]))
    jsonlines(folder / "windows.jsonl", windows)
    with pytest.raises(ValueError, match="grid|duplicate"):
        audit.inspect_dataset(data, frozen)


@pytest.mark.parametrize("key", ["X", "Y", "origin_us"])
def test_npz_content_and_identity_recomputed(tmp_path, key):
    data = tmp_path / "dataset"
    frozen = make_dataset(data)
    file = data / "normal_train.npz"
    with np.load(file) as f:
        values = dict(f)
    values[key].flat[0] += 1
    np.savez(file, **values)
    refresh_hash(data, file.name)
    with pytest.raises(ValueError):
        audit.inspect_dataset(data, frozen)


def test_nonvalid_target_drop_is_distinct_from_missing(tmp_path):
    data = tmp_path / "dataset"
    frozen = make_dataset(data)
    row = frozen["runs"][0]
    folder = data / "canonical" / row["run_id"]
    frames = audit.records(folder / "frames.jsonl")
    frames[615]["quality"]["pack_voltage"] = "invalid"
    jsonlines(folder / "frames.jsonl", frames)
    windows = audit.records(folder / "windows.jsonl")[:2]
    jsonlines(folder / "windows.jsonl", windows)
    summary = audit.read(folder / "summary.json")
    summary.update(valid_frames=619, windows=2)
    write(folder / "summary.json", summary)
    report, samples, _ = audit.inspect_run(folder, row)
    assert len(samples) == 1 and report["target_drop_detail"]["nonvalid_target_frames"] == 1
    assert report["target_drop_detail"]["missing_target_frames"] == 0


def test_no_airborne_dev_is_retained(tmp_path):
    data = tmp_path / "dataset"
    make_dataset(data)
    row = {"run_id": "synthetic-no-exposure", "role": "anomaly_dev"}
    folder = data / "canonical" / row["run_id"]
    folder.mkdir()
    for name in ("frames", "windows", "exclusions"):
        jsonlines(folder / (name + ".jsonl"), [])
    write(
        folder / "summary.json",
        dict(
            run_id=row["run_id"],
            closed=True,
            frames=0,
            valid_frames=0,
            windows=0,
            exclusions={},
            feature_order=audit.FEATURES,
        ),
    )
    report, accepted, selected = audit.inspect_run(folder, row)
    assert report["airborne_envelope_seconds"] == 0 and report["accepted_targets"] == 0
    assert accepted == selected == []


def test_ridge_actual_dimensions_and_cpu_headroom(tmp_path):
    data = tmp_path / "dataset"
    frozen = make_dataset(data)
    dims = audit.inspect_dataset(data, frozen)["dimensions"]
    pending = audit.memory_assessment(dims, None)
    assert pending["D"] == 1907 and pending["K"] == 40 and pending["one_gram_bytes"] == 32
    assert not pending["memory_audit_passed"]
    evidence = dict(
        hostname="DIS01",
        kind="coordinator_verified_cpu_RAM_budget",
        source="synthetic metadata",
        observed_utc="2026-10-14T13:00:00Z",
        allocation_budget_bytes=8 * 1024**3,
        available_bytes=8 * 1024**3,
    )
    result = audit.memory_assessment(dims, evidence)
    assert result["memory_audit_passed"] and not result["peak_rss_measured"]
    assert not audit.memory_assessment(dims, {**evidence, "available_bytes": 1024**3})[
        "memory_audit_passed"
    ]
    with pytest.raises(ValueError):
        audit.memory_assessment(dims, {**evidence, "hostname": "ABRM02"})
