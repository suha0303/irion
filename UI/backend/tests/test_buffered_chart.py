import json

import pytest

from arion.buffered_chart import chart
from arion.file_io import hash_file


def make_worker(folder, model, wire="capture.mavlink.bin"):
    folder.mkdir()
    calibration = folder / "calibration.json"
    calibration.write_text(
        json.dumps({"parameters": {"selected": {"threshold": [2, 3]}}})
    )
    (folder / "request.json").write_text(
        json.dumps(
            {
                "mode": "px4_buffered",
                "wire": wire,
                "binding": {
                    "model": model,
                    "hold_us": 10_000_000,
                    "calibration_sha256": hash_file(calibration),
                },
                "new19_calibration": str(calibration),
            }
        )
    )
    (folder / "state.json").write_text('{"status":"finished"}')
    (folder / "buffered-summary.json").write_text(
        '{"wire_sha256":"capture-hash"}'
    )
    row = {
        "origin_us": 20_000_000,
        "segment": 0,
        "features": [10.0] * 19,
        "observed": [True, True, True, True],
        "forecast": [[77.0] * 4] * 10,
        "residual": [2, 1, 0, -1],
        "decision": {
            "scores": [4, 1],
            "ema": [3, 2],
            "domains": [True, False],
        },
    }
    gap = {
        **row,
        "origin_us": 20_100_000,
        "segment": -1,
        "observed": [False] * 4,
        "residual": None,
        "decision": {"scores": None},
    }
    (folder / "buffered-results.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(gap) + '\n{"partial":'
    )
    return folder


def test_chart_uses_target_residual_scores_and_frozen_thresholds(tmp_path):
    folder = make_worker(tmp_path / "primary", "inflight-lstm-64x2")
    result = chart(folder)
    assert result["thresholds"] == [2, 3]
    assert result["hold_ms"] == 10000
    assert result["frames"] == [
        {
            "time_us": 20_000_000,
            "segment": 0,
            "observed": [10] * 4,
            "predicted": [8, 9, 10, 11],
            "residual": [2, 1, 0, -1],
            "raw_score": [4, 1],
            "score": [3, 2],
            "alarm": [True, False],
        },
        {
            "time_us": 20_100_000,
            "segment": -1,
            "observed": [None] * 4,
            "predicted": [None] * 4,
            "residual": [None] * 4,
            "raw_score": [None, None],
            "score": [None, None],
            "alarm": [False, False],
        },
    ]


def test_chart_binds_comparison_worker_to_same_recording(tmp_path):
    folder = make_worker(tmp_path / "primary", "inflight-lstm-64x2")
    model = "inflight-transformer-128x4"
    worker = make_worker(tmp_path / "worker", model)
    comparison = {model: worker}
    result = chart(folder, model, comparison, wire_sha="capture-hash")
    assert result["models"] == ["inflight-lstm-64x2", model]
    assert result["model"] == model
    with pytest.raises(ValueError, match="UDP 원문 해시"):
        chart(folder, model, comparison, wire_sha="other")
    with pytest.raises(ValueError, match="기록된 모델"):
        chart(folder, "unknown", comparison, wire_sha="capture-hash")


def test_chart_switches_models_in_one_growing_journal(tmp_path):
    models = [
        "inflight-lstm-64x2",
        "inflight-lstm-128x4",
        "inflight-transformer-64x2",
        "inflight-transformer-128x4",
    ]
    folder = make_worker(tmp_path / "primary", models[0])
    bindings = {}
    for i, model in enumerate(models):
        calibration = tmp_path / model / "calibration/calibration.json"
        calibration.parent.mkdir(parents=True)
        calibration.write_text(
            json.dumps({"parameters": {"selected": {"threshold": [2, 3 + i]}}})
        )
        bindings[model] = {"calibration_sha256": hash_file(calibration)}
    (folder / "request.json").write_text(
        json.dumps(
            {
                "mode": "px4_buffered",
                "inflight_root": str(tmp_path),
                "binding": {"hold_us": 10_000_000, "models": bindings},
            }
        )
    )
    (folder / "state.json").write_text('{"status":"predicting"}')
    journal = folder / "buffered-results.jsonl"
    row = json.loads(journal.read_text().splitlines()[0])
    row["models"] = {
        model: {"residual": [0, 0, 0, i], "decision": row["decision"]}
        for i, model in enumerate(models)
    }
    journal.write_text(json.dumps(row) + "\n")
    for i, model in enumerate(models):
        payload = chart(folder, model)
        assert payload["models"] == models
        assert payload["live"] is True
        assert payload["frames"][0]["observed"][3] == 10
        assert payload["frames"][0]["predicted"][3] == 10 - i
        assert payload["thresholds"][1] == 3 + i
