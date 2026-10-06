"""Exercise the real prepare/scaler and independent numeric audit with tiny canonical data."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pretrain import common


def test_pooled_scaler_uses_all_train_and_no_holdout_values(tmp_path):
    common.package_imports()
    fixtures = common.load("pooled_numeric_fixture", common.ROOT / "ai/tests/fixtures/pretrain_numeric.py")
    pipeline = common.load("pipeline.pooled_pipeline", common.ROOT / "ai/pipeline/pipeline.py")
    audit = common.load("pooled_audit", common.RESOURCES / "admission/derived.py")
    # First collection contributes one train run, second contributes two.
    # This deliberately detects an incorrect average of per-collection means.
    rows = [
        dict(run_id=name, role=role)
        for name, role in (
            ("set1-train", "normal_train"),
            ("set2-train-a", "normal_train"),
            ("set2-train-b", "normal_train"),
            ("validation", "normal_validation"),
            ("calibration", "normal_calibration"),
            ("normal-dev", "normal_dev_eval"),
            ("anomaly-dev", "anomaly_dev"),
        )
    ]
    source = tmp_path / "source"
    fixtures.make_dataset(source, rows)
    offsets = {"set1-train": 0, "set2-train-a": 20, "set2-train-b": 40}
    feature_index = audit.FEATURES.index("body_rate_p")
    for row in rows:
        folder = source / "canonical" / row["run_id"]
        offset = offsets.get(row["run_id"], 100000)
        frames = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
        for frame in frames:
            frame["values"]["body_rate_p"] += offset
            if frame["context"] is not None:
                frame["context"][0] += offset
        windows = [json.loads(line) for line in (folder / "windows.jsonl").read_text().splitlines()]
        for window in windows:
            for sample in window["X"]:
                sample[feature_index] += offset
            window["C"][0] += offset
        fixtures.jsonlines(folder / "frames.jsonl", frames)
        fixtures.jsonlines(folder / "windows.jsonl", windows)
        row["path"] = str(folder)

    def canonical(raw, target, config):
        shutil.copytree(raw, target)
        return common.read(Path(target) / "summary.json")

    frozen = dict(runs=rows)
    output = pipeline.prepare(
        frozen,
        tmp_path / "combined",
        common.read(common.PACKAGE / "preprocessing.json"),
        common.Budget(),
        process_run=canonical,
    )
    checked = audit.inspect_dataset(output, frozen)
    assert checked["npz_max_abs_error"] == 0
    assert max(checked["scaler_max_abs_errors"].values()) < 1e-10
    with np.load(output / "scaler.npz") as scaler:
        assert scaler["x_mean"][feature_index] == pytest.approx(21)
        assert scaler["x_scale"][feature_index] == pytest.approx(np.std([1, 21, 41]))
        assert scaler["c_mean"][0] == pytest.approx(24)
        # Average of set means would be (1 + 31)/2 =16, which is incorrect.
        assert scaler["x_mean"][feature_index] != 16
        pooled = dict(scaler)
    manifest = common.read(output / "manifest.json")
    assert manifest["unique_train_frames"] == 3 * 619
    assert len(list(output.glob("scaler*.npz"))) == 1
    train = pipeline.normalized_role(output, "normal_train", pooled)
    validation = pipeline.normalized_role(output, "normal_validation", pooled)
    assert set(train["run_id"]) == set(offsets)
    assert validation["X"][0, 0, feature_index] > 1000  # Holdout extreme cannot fit its own scaler.
