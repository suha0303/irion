"""Read-only UI: distinguish queued, sealed runs and actual released
completion.
"""

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reports/final-evaluation-20261103/attempt-02/operator/detailed.py"
)
spec = importlib.util.spec_from_file_location("final_readonly_display", SCRIPT)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def data():
    names = [
        f"{family}_h{hidden}_l{layers}_{optimizer}_s0"
        for hidden, layers in [(64, 2), (128, 2), (128, 4), (256, 4)]
        for family in ["lstm", "transformer"]
        for optimizer in ["sgd", "adam"]
    ]
    return dict(
        config=dict(models=[*names, "persistence", "ridge_forecast"]),
        controller=dict(
            status="RUNNING", stage="infer", started=100, job_id="123", pid=44
        ),
        inference={},
        evaluation={},
        completion={},
        infer_release={},
        eval_release={},
        gpu={},
        environment={},
        memory={},
        prepared={},
        scheduler=dict(state="PENDING", reason="Resources"),
        root="/server/final/attempt-02",
    )


def test_pending_is_not_computing_or_completed_and_lists_all_models():
    value = data()
    text = screen.render(value, now=200)
    assert "자원 할당 대기" in text and "Resources" in text
    assert "아직 GPU 추론 시작 전" in text
    assert "0/5,832" in text
    assert text.count("WAIT") == 19  # Eighteen models plus the status legend.
    assert "전체 완료 확인 NO" in text
    assert "학습·재보정·분석 없음" in text


def test_completed_current_model_is_not_double_counted():
    value = data()
    value["scheduler"] = {"state": "RUNNING"}
    value["inference"] = dict(
        models_completed=1,
        model=value["config"]["models"][0],
        done=324,
        total=324,
        stage="development_inference",
        started=100,
        updated=198,
    )
    text = screen.render(value, now=200)
    assert "324/5,832" in text and "648/5,832" not in text
    value["inference"].update(
        model=value["config"]["models"][1], done=100, cpu_pending=3
    )
    text = screen.render(value, now=200)
    assert "424/5,832" in text
    assert "CPU 미완료 3" in text
    assert "이 모델 100/324" in text


def test_model_switch_does_not_count_previous_model_done_value():
    value = data()
    value["scheduler"] = {"state": "RUNNING"}
    value["inference"] = dict(
        models_completed=1,
        model=value["config"]["models"][1],
        done=324,
        stage="final_inference",
    )
    assert "324/5,832" in screen.render(value, now=200)


def test_live_scheduler_overrides_stale_controller_pending():
    value = data()
    value["controller"]["queue"] = ["123|drone-ai-final-infer|PENDING|"]
    value["scheduler"] = {"state": "RUNNING"}
    assert "자원 할당 대기 | RUNNING" not in screen.render(value, now=200)


@pytest.mark.parametrize(
    "release",
    [
        dict(released=False),
        dict(released=True, slurm_state="FAILED", exit_code="1:0"),
    ],
)
def test_evaluation_requires_successful_releases_for_completion(
    release,
):
    value = data()
    value["controller"]["stage"] = "evaluate"
    value["inference"]["models_completed"] = 18
    value["evaluation"] = dict(done=324, total=324, stage="stopped")
    value["completion"] = dict(
        status="REPORT_READY_STOPPED", analysis_performed=False
    )
    value["infer_release"] = copy.deepcopy(release)
    value["eval_release"] = copy.deepcopy(release)
    text = screen.render(value, now=200)
    assert "324/324" in text and "5,832/5,832" in text
    assert "전체 완료 확인 NO" in text


def test_release_display_does_not_read_performance_results():
    value = data()
    value["inference"]["models_completed"] = 18
    value["controller"].update(status="REPORT_READY_STOPPED", stage="stopped")
    value["completion"] = dict(
        status="REPORT_READY_STOPPED", analysis_performed=False
    )
    for key in ["infer_release", "eval_release"]:
        value[key] = dict(
            released=True, slurm_state="COMPLETED", exit_code="0:0"
        )
    text = screen.render(value, now=200)
    assert "전체 완료 확인 YES" in text
    assert "완료 · 반납·정지 확인" in text
    source = SCRIPT.read_text(encoding="utf-8")
    assert "metrics.json" not in source and "model-results.csv" not in source
    assert (
        "scancel" not in source
        and "sbatch" not in source
        and "os.kill" not in source
    )


def test_gpu_old_sample_and_corrupt_progress_are_explicit():
    value = data()
    value["controller"]["stage"] = "evaluate"
    value["gpu"] = dict(
        status="SAMPLED",
        timestamp=100,
        used_mib=1024,
        total_mib=49152,
        utilization_pct=0,
        power_w=70,
        temperature_c=40,
    )
    value["evaluation"] = {"_read_error": "temporary invalid JSON"}
    text = screen.render(value, now=200)
    assert "마지막 추론 표본(100초 전)" in text
    assert "상태 읽기 미확인" in text
    assert "전체 완료 확인 NO" in text
