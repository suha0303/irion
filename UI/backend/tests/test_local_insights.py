"""Keep local UI results bound to the exact source and selected variants."""

import json

from arion.local_insights import insights


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def analysis(model, weight, source="raw-flight"):
    return dict(
        model_id=model,
        adapted_sha256=weight,
        source_sha256=source,
        profile_version="profile",
        calibration_binding=dict(
            clock="receipt_monotonic",
            live_input=dict(imu_message="HIGHRES_IMU", imu_id=0),
            fast_max_age_ms=200,
        ),
    )


def state():
    return dict(
        status="finished",
        forecast_model="forecast",
        post_model="health",
        adapted_sha256="forecast-weight",
        post_adapted_sha256="health-weight",
        result=dict(
            source_sha256="raw-flight",
            analyses=[
                dict(
                    manifest="preserved",
                    analysis=analysis("health", "health-weight"),
                )
            ],
        ),
    )


def test_saved_analysis_requires_source_profile_and_selected_weights(tmp_path):
    row = state()
    candidates = [
        analysis("forecast", "forecast-weight"),
        analysis("health", "health-weight"),
    ]
    path = tmp_path / "jobs/job.json"
    job = dict(
        id="job",
        action="analyze",
        status="complete",
        result=dict(analyses=candidates),
    )
    for key, wrong in [
        ("source_sha256", "another-flight"),
        ("adapted_sha256", "other-model"),
        ("profile_version", "other-profile"),
    ]:
        original = candidates[0][key]
        candidates[0][key] = wrong
        save(path, job)
        assert (
            insights(tmp_path, "live-example", row)["report_source"]
            == "flight_completion"
        )
        candidates[0][key] = original
    for key, wrong in [
        ("clock", "boot"),
        ("live_input", dict(imu_message="SCALED_IMU2", imu_id=1)),
        ("fast_max_age_ms", 500),
    ]:
        binding = candidates[0]["calibration_binding"]
        original = binding[key]
        binding[key] = wrong
        save(path, job)
        assert (
            insights(tmp_path, "live-example", row)["report_source"]
            == "flight_completion"
        )
        binding[key] = original
    save(path, job)
    result = insights(tmp_path, "live-example", row)
    assert result["report_source"] == "saved_log_analysis"
    assert result["report_job"] == "job"
    assert not result["history_available"]
    assert result["alarms"] == []


def test_live_alarms_survive_restart_and_ignore_incomplete_last_line(tmp_path):
    folder = tmp_path / "live/live-example"
    folder.mkdir(parents=True)
    path = folder / "alarms.jsonl"
    path.write_text(
        json.dumps(
            dict(
                model_id="forecast", decision=dict(status="suspect", time_us=42)
            )
        )
        + '\n{"unfinished":'
    )
    result = insights(tmp_path, "live-example", dict(status="predicting"))
    assert result["alarm_count"] == 1
    assert result["alarms"][0]["decision"]["time_us"] == 42
    assert result["history_available"]


def test_completed_forecast_is_used_without_replacing_with_later_job(tmp_path):
    row = state()
    row["result"]["forecast_analyses"] = [
        dict(analysis=analysis("forecast", "forecast-weight"))
    ]
    result = insights(tmp_path, "live-example", row)
    assert {a["model_id"] for a in result["report"]["analyses"]} == {
        "forecast",
        "health",
    }
    assert result["report_source"] == "flight_completion"
