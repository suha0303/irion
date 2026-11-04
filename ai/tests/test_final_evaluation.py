"""Final holdout admission, unchanged scoring and finite submission boundaries."""

import copy
import os
import sys

import pytest
from test_reassessment import Progress, fixture_evaluation

from execution.common import read, sha, write
from execution.evaluate import default_protocol_manifest, evaluate, final_protocol
from learning import final_test as final
from learning import operator


def final_fixture(tmp_path):
    old, inference, expected = fixture_evaluation(tmp_path)
    dataset = tmp_path / "dataset"
    manifest = read(dataset / "manifest.json")
    for row in manifest["runs"]:
        row["role"] = "normal_test" if row["role"] == "normal_dev_eval" else "anomaly_test"
    write(dataset / "manifest.json", manifest)
    seal = read(inference / "result-seal.json")
    seal.update(status="FINAL_INFERENCE_COMPLETE", final_test_used=True)
    write(inference / "result-seal.json", seal)
    cfg = dict(
        dataset=str(dataset),
        seed=0,
        execution_scope=final.SCOPE,
        final_test_authorized=True,
        production_training_authorized=False,
        calibration_authorized=False,
        source_protocol=default_protocol_manifest(),
        dataset_files={
            str(p.relative_to(dataset)): sha(p) for p in dataset.rglob("*") if p.is_file()
        },
    )
    cfg["protocol"] = final_protocol(cfg["source_protocol"])

    def truth(row, target, progress):
        target.write_bytes((old / "truth" / (row["run_id"] + ".jsonl")).read_bytes())

    return cfg, inference, expected, truth


def test_final_scoring_matches_development_math_without_fitting(tmp_path, monkeypatch):
    cfg, inference, expected, truth = final_fixture(tmp_path)
    import capstone_ai.detection_protocol as protocol
    import capstone_ai.training as training

    def forbidden(*args, **kwargs):
        pytest.fail("Final evaluator cannot load weights or fit calibration")

    monkeypatch.setattr(protocol, "fit_calibration", forbidden)
    monkeypatch.setattr(training.Predictor, "__init__", forbidden)
    before = {str(p): sha(p) for p in inference.rglob("*") if p.is_file()}
    output = tmp_path / "final"
    actual = evaluate(
        cfg,
        inference,
        output,
        Progress(),
        truth_provider=truth,
        model_names=list(expected["models"]),
        evaluation_mode="final_test",
        development_counts=dict(normal_test=1, anomaly_test=2),
    )
    assert actual["final_test_used"] is True
    assert actual["evaluation_mode"] == "final_test"
    assert actual["common_valid_origins"] == expected["common_valid_origins"]
    for name, model in actual["models"].items():
        for key in expected["models"][name].keys() - {"per_run"}:
            assert model[key] == expected["models"][name][key]
        assert {r["role"] for r in model["per_run"]} == {"normal_test", "anomaly_test"}
    assert {path: sha(path) for path in before} == before
    assert read(output / "evidence.json")["final_test_used"] is True
    assert read(output / "report.json")["final_test_used"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"final_test_authorized": False},
        {"calibration_authorized": True},
        {"production_training_authorized": True},
        {"execution_scope": "post_training_evaluation"},
    ],
)
def test_final_requires_explicit_readonly_authorization(tmp_path, change):
    cfg, inference, _, truth = final_fixture(tmp_path)
    cfg.update(change)
    with pytest.raises(ValueError, match="authorization"):
        evaluate(
            cfg,
            inference,
            tmp_path / "final",
            Progress(),
            truth_provider=truth,
            model_names=["fixture_a", "fixture_b"],
            evaluation_mode="final_test",
            development_counts=dict(normal_test=1, anomaly_test=2),
        )
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_success", "new_episode"),
        ("threshold_multiplier", [2, 2]),
        ("max_detection_delay_s", 30),
        ("point_adjustment", True),
    ],
)
def test_final_cannot_change_completed_protocol(field, value):
    protocol = default_protocol_manifest()
    protocol[field] = value
    with pytest.raises(ValueError, match="retain"):
        final_protocol(protocol)


def test_final_requires_final_inference_seal_and_correct_roles(tmp_path):
    cfg, inference, _, truth = final_fixture(tmp_path)
    seal = read(inference / "result-seal.json")
    seal["final_test_used"] = False
    write(inference / "result-seal.json", seal)
    with pytest.raises(ValueError, match="sealed"):
        evaluate(
            cfg,
            inference,
            tmp_path / "final",
            Progress(),
            truth_provider=truth,
            model_names=["fixture_a", "fixture_b"],
            evaluation_mode="final_test",
            development_counts=dict(normal_test=1, anomaly_test=2),
        )
    with pytest.raises(ValueError, match="roles"):
        evaluate(
            cfg,
            inference,
            tmp_path / "final",
            Progress(),
            evaluation_mode="final_test",
            development_counts=dict(normal_dev_eval=1, anomaly_dev=2),
        )


def test_failed_final_inference_never_submits_evaluator(tmp_path, monkeypatch):
    cfg = dict(queue_hours=24, wall_hours=24, evaluation_hours=12)
    monkeypatch.setattr(final, "checked", lambda root: (root, cfg))
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    submitted = []

    def submit(root, phase, cfg):
        submitted.append(phase)
        return dict(job_id="123", uid=os.getuid(), phase=phase)

    monkeypatch.setattr(final, "submit", submit)
    failed = dict(released=True, slurm_state="FAILED", exit_code="1:0", queue=[])
    monkeypatch.setattr(operator, "wait_release", lambda *a: failed)
    monkeypatch.setattr(operator, "snapshot", lambda *a: failed)
    with pytest.raises(RuntimeError):
        final.control(tmp_path)
    assert submitted == ["infer"]
    assert read(tmp_path / "controller.json")["status"] == "FAILED_STOPPED"
    assert not (tmp_path / "completion.json").exists()
    with pytest.raises(FileExistsError):
        final.control(tmp_path)
    assert submitted == ["infer"]


def test_final_submit_is_pinned_single_gpu_and_never_retries(tmp_path, monkeypatch):
    cfg = dict(wall_hours=24, evaluation_hours=12)
    monkeypatch.setattr(final, "checked", lambda root: (root, cfg))
    monkeypatch.setattr(final, "SUBMIT_LOCK", tmp_path / "lock")
    monkeypatch.setattr(final, "python_for", lambda *a, **k: sys.executable)
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    commands = []
    monkeypatch.setattr(operator, "command", lambda argv: commands.append(argv) or "123;cluster")
    with pytest.raises(ValueError, match="no training"):
        final.submit(tmp_path, "fit", cfg)
    final.submit(tmp_path, "infer", cfg)
    for flag in [
        "--partition=p02",
        "--nodelist=DIS01",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:1",
        "--cpus-per-task=10",
        "--mem=64G",
        "--no-requeue",
    ]:
        assert flag in commands[0]
    assert "CUDA_VISIBLE_DEVICES" not in (tmp_path / "infer-job.sh").read_text()
    with pytest.raises(FileExistsError):
        final.submit(tmp_path, "infer", cfg)
    assert len(commands) == 1


def test_final_submit_refuses_overlap_and_failed_release(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "checked", lambda root: (root, {}))
    monkeypatch.setattr(final, "SUBMIT_LOCK", tmp_path / "lock")
    monkeypatch.setattr(operator, "own_queue", lambda: "123|existing|RUNNING|DIS01")
    with pytest.raises(ValueError, match="concurrent"):
        final.submit(tmp_path, "infer", {})
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    write(tmp_path / "release.json", dict(released=True, slurm_state="FAILED", exit_code="1:0"))
    with pytest.raises(RuntimeError):
        final.submit(tmp_path, "evaluate", {})
    assert not (tmp_path / "evaluate-submitted.claim").exists()


def manifest_fixture():
    manifest = dict(
        schema="drone.final-test-dataset/1",
        status="FINAL_DATASET_SEALED_STOPPED",
        scaler_fit_performed=False,
        training_authorized=False,
        calibration_authorized=False,
        final_test_scoring_authorized=False,
        zero_target_runs=[],
        runs=[],
        pairs={},
    )
    for i in range(108):
        for cohort in ["standalone_normal", "paired_control", "anomaly"]:
            row = dict(
                run_id=f"{cohort}-{i}",
                role="anomaly_test" if cohort == "anomaly" else "normal_test",
                final_cohort=cohort,
                seed=i,
                group_id=f"g{i}",
                motion="hover",
                environment="cold",
                mission_seconds=480,
                operation_variation={"wind": 0},
            )
            if cohort != "standalone_normal":
                row["pair_id"] = f"p{i}"
                manifest["pairs"].setdefault(f"p{i}", []).append(row["run_id"])
            manifest["runs"].append(row)
    return manifest


def test_manifest_rejects_mismatched_pair_and_unscaled_source():
    manifest = manifest_fixture()
    final.validate_manifest(manifest)
    bad = copy.deepcopy(manifest)
    bad["runs"][2]["operation_variation"]["wind"] = 1
    with pytest.raises(ValueError, match="paired conditions"):
        final.validate_manifest(bad)
    manifest["scaler_fit_performed"] = True
    with pytest.raises(ValueError, match="unscaled"):
        final.validate_manifest(manifest)


def test_final_console_does_not_claim_training_or_recalibration(tmp_path):
    write(tmp_path / "input/config.json", dict(execution_scope=final.SCOPE, models=final.MODELS))
    write(tmp_path / "controller.json", dict(status="RUNNING", stage="evaluate", job_id="123"))
    write(tmp_path / "compute/progress.json", dict(models_completed=18))
    write(tmp_path / "evaluation-progress.json", dict(done=23, total=324, stage="final_evaluation"))
    screen = operator.render(tmp_path)
    assert "학습 완료 신경망16 + baseline2" in screen
    assert "학습·보정·분석 없음" in screen
    assert "최종 추론 완료 모델: 18/18" in screen
    assert "23/324" in screen
    assert "새 정상 보정" not in screen
