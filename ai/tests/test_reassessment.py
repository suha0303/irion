import json

import numpy as np
import pytest

from capstone_ai.detection import DomainAlarms
from execution.common import read, sha, write
from execution.evaluate import ResidualSummary, evaluate, forecast_errors
from learning.reassess import checked_criteria, reassess


class Progress:
    def check(self):
        pass

    def update(self, **values):
        pass


def fixture_evaluation(tmp_path):
    dataset = tmp_path / "dataset"
    inference = tmp_path / "inference"
    dataset.mkdir()
    inference.mkdir()
    names = ["fixture_a", "fixture_b"]
    rows = []
    for index in range(3):
        row = dict(
            run_id=f"r{index}",
            role="anomaly_dev" if index else "normal_dev_eval",
            group_id=f"scenario-group-{index}",
            paired_case_id="paired" if index else "normal",
            scenario=f"scenario{index}",
            normal_variation={"wind_mps": 2},
        )
        rows.append(row)
        folder = dataset / "canonical" / row["run_id"]
        folder.mkdir(parents=True)
        (folder / "frames.jsonl").write_text('{"time_us":0}\n')
        for name in names:
            detector = DomainAlarms([1, 1])
            decisions = []
            for j in range(6):
                origin = j * 1_000_000
                scores = [2.0, 2.0] if index else [0.0, 0.0]
                decisions.append(
                    dict(
                        run_id=row["run_id"],
                        clock_id="clock",
                        segment_id=0,
                        origin_time_us=origin,
                        decision_time_us=origin + 1_000_000,
                        scores=scores,
                        signed_residual=np.full((10, 4), index + j / 10).tolist(),
                        unscored_reasons=[],
                        **detector.update(scores, origin),
                    )
                )
            replay = inference / "calibration" / name / row["run_id"]
            replay.mkdir(parents=True)
            (replay / "decisions.jsonl").write_text(
                "".join(json.dumps(d) + "\n" for d in decisions)
            )
            (replay / "forecasts.jsonl").write_text('{"fixture":"frozen"}\n')
            write(
                replay / "summary.json",
                dict(operational_seconds=6, operational_spans_us={"clock": [0, 6_000_000]}),
            )
            write(replay / "seal.json", {"files": {p.name: sha(p) for p in replay.glob("*.jsonl")}})
    for name in names:
        write(inference / "calibration" / name / "calibration.json", {"high": [1, 1]})
    write(dataset / "manifest.json", {"runs": rows})
    write(
        inference / "result-seal.json",
        dict(
            models=names,
            final_test_used=False,
            truth_join=False,
            status="TRAIN_CALIBRATION_INFERENCE_COMPLETE",
            files={
                str(p.relative_to(inference)): sha(p) for p in inference.rglob("*") if p.is_file()
            },
        ),
    )
    write(tmp_path / "release.json", dict(released=True, slurm_state="COMPLETED", exit_code="0:0"))
    config = dict(
        dataset=str(dataset),
        seed=0,
        dataset_files={
            str(p.relative_to(dataset)): sha(p) for p in dataset.rglob("*") if p.is_file()
        },
    )

    def truth_provider(row, path, progress):
        if row["run_id"] == "r0":
            value = ""
        elif row["run_id"] == "r1":
            value = (
                json.dumps(
                    dict(event_id="step", onset_kind="in_flight_step", applied_time_us=4_000_000)
                )
                + "\n"
            )
        else:
            value = json.dumps(dict(event_id="existing", onset_kind="pre_existing")) + "\n"
        path.write_text(value)

    output = tmp_path / "evaluation"
    metrics = evaluate(
        config,
        inference,
        output,
        Progress(),
        truth_provider=truth_provider,
        model_names=names,
        development_counts=dict(normal_dev_eval=1, anomaly_dev=2),
        completion_status="EVALUATION_COMPLETE_PENDING_RELEASE",
    )
    return output, inference, metrics


def test_accumulator_matches_concatenated_residuals_without_retaining_pool():
    residuals = np.random.default_rng(17).normal(size=(71, 10, 4))
    summary = ResidualSummary()
    for part in np.array_split(residuals, 9):
        summary.add(part)
    actual = summary.result()
    for key, expected in [
        ("mae", np.abs(residuals).mean((0, 1))),
        ("rmse", np.sqrt((residuals**2).mean((0, 1)))),
        ("signed_bias", residuals.mean((0, 1))),
    ]:
        np.testing.assert_allclose(actual[key], expected, rtol=1e-13, atol=1e-14)
    assert actual["scored_forecasts"] == 71
    assert all(not isinstance(value, list) for value in vars(summary).values())
    assert forecast_errors([])["mae"] is None
    with pytest.raises(ValueError, match="residuals"):
        summary.add([np.zeros((9, 4))])


def test_default_reassessment_preserves_fixed_results_and_original_files(tmp_path):
    evaluation, inference, original = fixture_evaluation(tmp_path)
    before = {
        str(p): sha(p) for root in (evaluation, inference) for p in root.rglob("*") if p.is_file()
    }
    metrics = reassess(evaluation, tmp_path / "reassessed", {})
    assert metrics["common_valid_origins"] == original["common_valid_origins"] == 18
    for name in metrics["models"]:
        old, new = original["models"][name], metrics["models"][name]
        assert new["anomaly_independent_groups"] == 1
        for key in old.keys() - {"per_run"}:
            assert new[key] == old[key]
        for old_run, new_run in zip(old["per_run"], new["per_run"]):
            for key in ("events", "metrics", "point_metrics", "forecast_errors"):
                assert new_run[key] == old_run[key]
    assert {path: sha(path) for path in before} == before
    assert read(evaluation / "report.json")["status"] == "EVALUATION_COMPLETE_PENDING_RELEASE"
    assert read(evaluation / "evidence.json")["runs"][0]["normal_variation"] == {"wind_mps": 2}
    with pytest.raises(ValueError, match="fresh output"):
        reassess(evaluation, tmp_path / "reassessed", {})


def test_explicit_matching_and_threshold_variants_do_not_refit_models(tmp_path, monkeypatch):
    evaluation, _, _ = fixture_evaluation(tmp_path)
    import capstone_ai.detection_protocol as protocol
    import capstone_ai.training as training

    def forbidden(*args, **kwargs):
        pytest.fail("Reassessment must not execute models or fit thresholds")

    monkeypatch.setattr(protocol, "fit_calibration", forbidden)
    monkeypatch.setattr(training.Predictor, "__init__", forbidden)
    new_episode = reassess(evaluation, tmp_path / "episode", {"event_success": "new_episode"})
    window = reassess(evaluation, tmp_path / "window", {"max_detection_delay_s": 0})
    high = reassess(evaluation, tmp_path / "high", {"threshold_multiplier": [3, 3]})
    for name in new_episode["models"]:
        assert new_episode["models"][name]["events"]["detected"] == 1
        assert window["models"][name]["events"]["detected"] == 1
        assert high["models"][name]["events"]["detected"] == 0
        assert high["models"][name]["forecast_errors_all_valid"]["scored_forecasts"] == 18
    assert len({r["protocol_sha256"] for r in (new_episode, window, high)}) == 3


@pytest.mark.parametrize("tamper", ["truth", "decisions", "protocol", "evidence"])
def test_changed_immutable_evidence_rejected(tmp_path, tamper):
    evaluation, inference, _ = fixture_evaluation(tmp_path)
    target = {
        "truth": evaluation / "truth/r1.jsonl",
        "decisions": inference / "calibration/fixture_a/r1/decisions.jsonl",
        "protocol": evaluation / "protocol.json",
        "evidence": evaluation / "evidence.json",
    }[tamper]
    target.write_text(target.read_text() + "\n")
    # A whitespace-only protocol change preserves its canonical semantic hash;
    # change a value to test protocol protection.
    if tamper == "protocol":
        value = read(target)
        value["event_success"] = "new_episode"
        write(target, value)
    with pytest.raises(ValueError, match="changed|mismatch"):
        reassess(evaluation, tmp_path / "reassessed", {})
    assert not (tmp_path / "reassessed/report.json").exists()


@pytest.mark.parametrize(
    "criteria",
    [
        {"unknown": 1},
        {"event_success": "anything"},
        {"max_detection_delay_s": -1},
        {"max_detection_delay_s": True},
        {"threshold_multiplier": [1]},
        {"threshold_multiplier": [0, 1]},
        {"threshold_multiplier": [float("nan"), 1]},
    ],
)
def test_invalid_or_unimplemented_criteria_are_rejected(criteria):
    with pytest.raises(ValueError):
        checked_criteria(criteria)
