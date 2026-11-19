"""Exercise New19 adaptation with independent synthetic flight packages."""

from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file
from test_localflight_adapt import flight

from learning.new19.data import digest, read, write
from learning.new19.models import VARIANTS, make_model
from learning.new19.real_adaptation import MODEL, load_inputs
from localflight.cli import execute


def baseline(root, model_id=MODEL):
    bundle = root / model_id / "fit"
    bundle.mkdir(parents=True)
    model = make_model(model_id)
    weights = bundle / "model.safetensors"
    save_file(model.state_dict(), str(weights))
    write(
        bundle / "bundle.json",
        dict(
            schema="drone-ai.new19.bundle/1",
            variant=model_id,
            model_sha256=digest(weights),
            scaler=dict(
                mean=[0.0] * 19,
                scale=[1.0] * 19,
                delta_mean=[0.0] * 4,
                delta_scale=[1.0] * 4,
            ),
        ),
    )
    calibration = bundle.parent / "calibration"
    calibration.mkdir()
    write(
        calibration / "calibration.json",
        dict(
            schema="drone-ai.new19.calibration/1",
            bundle_sha256=digest(bundle / "bundle.json"),
            parameters=dict(scale=[1.0, 1.0], threshold=3.0)
            if model_id.startswith("whole-run-")
            else dict(
                center=[0.0, 0.0, 0.0],
                precision=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                voltage_scale=1.0,
                selected=dict(threshold=[3.0, 3.0]),
            ),
        ),
    )
    return weights


def prepared_logs(root):
    root.mkdir()
    return {
        role: [dict(manifest=str(flight(root, role, role)), date=day)]
        for role, day in (
            ("adaptation", "2026-01-01"),
            ("calibration", "2026-01-02"),
            ("evaluation", "2026-01-03"),
        )
    }


@pytest.mark.parametrize("overlap", ["date", "raw", "segment"])
def test_rejects_split_leakage(tmp_path, overlap):
    logs = prepared_logs(tmp_path / "prepared")
    train = logs["adaptation"][0]
    test = logs["evaluation"][0]
    if overlap == "date":
        test["date"] = train["date"]
        reason = "같은 비행 날짜"
    elif overlap == "raw":
        manifest = Path(test["manifest"])
        meta = read(manifest)
        meta["source_sha256"] = read(train["manifest"])["source_sha256"]
        write(manifest, meta)
        reason = "같은 원본"
    else:
        logs["adaptation"].append(train)
        reason = "겹치는 준비 구간"
    with pytest.raises(ValueError, match=reason):
        load_inputs(logs)


def test_synthetic_worker_saves_only_inactive_candidate(tmp_path):
    model_root = tmp_path / "models"
    weights = baseline(model_root)
    watched = {
        path: digest(path) for path in model_root.rglob("*") if path.is_file()
    }
    logs = prepared_logs(tmp_path / "prepared")
    output = tmp_path / "new19-candidates" / "example"
    evaluation = execute(
        dict(
            action="new19_adapt",
            model_root=str(model_root),
            logs=logs,
            output=str(output),
        )
    )["new19_candidate"]
    assert evaluation == read(output / "result.json")
    assert evaluation["review_status"] == "pending"
    assert evaluation["active"] is False
    assert evaluation["anomaly_evidence"] is False
    assert evaluation["counts"] == {role: 1 for role in logs}
    report = evaluation["evaluation"][0]
    assert report["scored_frames"] > 0
    assert set(report["rmse"]) == {"original", "adapted", "persistence"}
    assert set(report["decisions"]) == {"original", "x500", "calibrated"}
    candidate = output / "candidate"
    calibration = read(output / "calibration.json")
    assert calibration["bundle_sha256"] == digest(candidate / "bundle.json")
    assert calibration["manifest_sha256"] == digest(output / "split.json")
    assert calibration["parameters"]["selected"]["quantile"] == 0.99
    assert list(tmp_path.glob("**/active.json")) == []
    assert all(digest(path) == expected for path, expected in watched.items())
    base = load_file(str(weights))
    fitted = load_file(str(candidate / "model.safetensors"))
    changed = {
        name for name, tensor in base.items() if not tensor.equal(fitted[name])
    }
    assert changed == {"head.weight", "head.bias"}


def test_v02_batch_and_whole_run_final_layer(tmp_path):
    root = tmp_path / "models"
    weights = {model_id: baseline(root, model_id) for model_id in VARIANTS}
    logs = prepared_logs(tmp_path / "prepared")
    output = tmp_path / "batch"
    options = dict(
        models=list(VARIANTS),
        model_roots=dict.fromkeys(VARIANTS, str(root)),
    )
    result = execute(
        dict(action="v02_adapt", logs=logs, output=str(output), **options)
    )["v02"]
    assert result["failures"] == {}
    assert len(result["candidates"]) == 8
    for item in result["candidates"]:
        model_id = item["model_id"]
        assert item["active"] is False and item["review_status"] == "pending"
        assert item["evaluation"][0]["scored_frames"] == 150
        folder = output / model_id
        before = load_file(str(weights[model_id]))
        after = load_file(str(folder / "candidate/model.safetensors"))
        changed = {key for key in before if not before[key].equal(after[key])}
        last_layer = "head.2" if item["task"] == "whole-run" else "head"
        assert changed == {f"{last_layer}.weight", f"{last_layer}.bias"}
        graph = read(folder / "0.json")
        segment = graph["segments"][0]
        assert graph["task"] == item["task"]
        assert len(segment["time_us"]) == 250
        assert {"original", "adapted", "persistence"} == segment[
            "prediction"
        ].keys()
        if item["task"] == "whole-run":
            assert segment["prediction"]["adapted"][:100] == [None] * 100
            assert (
                item["evaluation"][0]["whole_decisions"]["calibrated"][
                    "available_segments"
                ]
                == 1
            )
            params = read(folder / "calibration.json")["parameters"]
            assert params["quantile"] == 0.99 and params["source_count"] == 1
        else:
            assert set(segment["rules"]) == {"original", "x500", "calibrated"}
    assert not list(output.glob("**/active.json"))

    model_id = "whole-run-transformer-64x2"
    one = tmp_path / "single"
    report = execute(
        dict(
            action="v02_analyze",
            manifest=logs["evaluation"][0]["manifest"],
            output=str(one),
            models=[model_id],
            model_roots=options["model_roots"],
        )
    )["v02"]
    assert report["failures"] == {} and report["candidates"] == []
    assert report["analyses"][0]["model_id"] == model_id
    assert report["coverage"]["retained_frames"] == 250
    assert (
        "adapted"
        not in read(one / model_id / "0.json")["segments"][0]["prediction"]
    )
