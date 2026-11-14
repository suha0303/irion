"""Check local CPU jobs, immutable profiles and API validation."""

import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arion.data_transfer import inventory, migrate
from arion.file_io import atomic_json
from arion.local_flight import LocalFlight
from arion.main import create_app


def test_real_cpu_catalog_and_immutable_profile(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        result = client.get("/api/local-ai/catalog")
        assert result.status_code == 200
        rows = result.json()["models"]
        assert len(rows) == 16
        assert {r["task"] for r in rows} == {"forecast", "run_health"}
        profile = dict(
            schema="drone.real-aircraft/1",
            id="pending",
            name="Unknown aircraft",
            autopilot="ardupilot",
            rotor_count=None,
            series_cells=None,
            capacity_ah=None,
            motor_channels={},
        )
        saved = client.post("/api/local-ai/profiles", json=profile)
        assert saved.status_code == 200
        assert not saved.json()["eligibility"]["profile_complete"]
        assert not saved.json()["eligibility"]["real_aircraft_validated"]
        first = saved.json()["version"]
        profile["series_cells"] = 12
        second = client.post("/api/local-ai/profiles", json=profile).json()[
            "version"
        ]
        assert first != second
        assert len(client.get("/api/local-ai/profiles").json()["profiles"]) == 2
        profile["capacity_ah"] = -1
        assert (
            client.post("/api/local-ai/profiles", json=profile).status_code
            == 422
        )


def test_upload_type_and_cancel_ownership(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert (
            client.post(
                "/api/local-ai/logs", files={"file": ("x.txt", b"no")}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/local-ai/jobs/" + "a" * 32 + "/cancel"
            ).status_code
            == 409
        )


def test_cancel_before_worker_starts_and_restart(tmp_path):
    async def run():
        service = LocalFlight(tmp_path)
        job = service.create_job(dict(action="benchmark"))
        await service.close()
        assert service.active is None
        assert (
            json.loads(service.job_path(job["id"]).read_text(encoding="utf-8"))[
                "status"
            ]
            == "cancelled"
        )
        with pytest.raises(ValueError):
            service.job_path("../../other")

    asyncio.run(run())
    path = tmp_path / "jobs" / ("b" * 32 + ".json")
    path.write_text(
        json.dumps(dict(id="b" * 32, status="running", created_at=0)),
        encoding="utf-8",
    )
    LocalFlight(tmp_path)
    assert (
        json.loads(path.read_text(encoding="utf-8"))["status"] == "interrupted"
    )


def test_new_jobs_require_completed_sources_and_reject_unknown_resume(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        for endpoint, body in (
            (
                "timeline",
                dict(
                    log_job="a" * 32,
                    profile_id="missing",
                    profile_version="b" * 64,
                ),
            ),
            ("prepare", dict(timeline_job="a" * 32)),
            (
                "analyze",
                dict(prepared_job="a" * 32, models=["forecast_lstm_h64_l2"]),
            ),
            ("adapt", dict(adaptation_jobs=[], development_jobs=[])),
            ("compare", dict(adapted_job="a" * 32, prepared_job="b" * 32)),
        ):
            assert (
                client.post("/api/local-ai/" + endpoint, json=body).status_code
                == 422
            )
        assert (
            client.post(
                "/api/local-ai/jobs/" + "a" * 32 + "/resume"
            ).status_code
            == 422
        )


def test_real_cpu_worker_analyzes_adapts_and_compares_synthetic_segments(
    tmp_path,
):
    project = Path(__file__).resolve().parents[4]
    # Generate explicit synthetic fixtures in the AI environment; no real log is
    # asserted healthy, and the UI interpreter never imports PyTorch.
    code = """
import json, sys
from pathlib import Path
from test_localflight_calibration import make_flight
root=Path(sys.argv[1])
roles = {'adaptation':'adaptation','evaluation':'evaluation','cal-a':'calibration','cal-b':'calibration'}
print(json.dumps({name:str(make_flight(root,name,role)) for name,role in roles.items()}))
"""

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            str(project / p)
            for p in ("ai", "ai/src", "ai/tests", "Data/generation")
        ),
    }
    made = subprocess.run(
        [str(project / "ai/.venv/bin/python"), "-c", code, str(tmp_path)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    fixtures = json.loads(made.stdout)
    with TestClient(create_app(tmp_path)) as client:
        root = tmp_path / "local-ai/jobs"
        root.mkdir(exist_ok=True)
        ids = {}
        for role, path in fixtures.items():
            job_id = hashlib.md5(role.encode()).hexdigest()
            ids[role] = job_id
            (root / (job_id + ".json")).write_text(
                json.dumps(
                    dict(
                        id=job_id,
                        action="prepare",
                        status="complete",
                        created_at=0,
                        result=dict(
                            manifest=path, role=role, health="confirmed_normal"
                        ),
                    )
                ),
                encoding="utf-8",
            )

        def complete(endpoint, body, expected_status="complete"):
            response = client.post("/api/local-ai/" + endpoint, json=body)
            assert response.status_code == 200, response.text
            job_id = response.json()["id"]
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                row = next(
                    j
                    for j in client.get("/api/local-ai/jobs").json()["jobs"]
                    if j["id"] == job_id
                )
                if row["status"] not in {"queued", "running"}:
                    assert row["status"] == expected_status, row
                    return row
                time.sleep(0.05)
            pytest.fail("CPU worker timeout")

        analyzed = complete(
            "analyze",
            dict(
                prepared_job=ids["evaluation"],
                models=["forecast_lstm_h64_l2", "run_health_lstm_h64_l2"],
            ),
        )
        assert len(analyzed["result"]["analyses"]) == 2
        assert (
            analyzed["result"]["run_summary"]["coverage"] is None
        )  # legacy synthetic fixture has no full source extent
        assert analyzed["result"]["run_summary"]["status"] == "unavailable"
        for invalid in ([], [ids["evaluation"], ids["evaluation"]]):
            assert (
                client.post(
                    "/api/local-ai/analyze",
                    json=dict(
                        prepared_jobs=invalid, models=["forecast_lstm_h64_l2"]
                    ),
                ).status_code
                == 422
            )
        fitted = complete(
            "adapt",
            dict(
                adaptation_jobs=[ids["adaptation"]],
                development_jobs=[],
                model_id="forecast_lstm_h64_l2",
                epochs=1,
            ),
        )
        comparison = complete(
            "compare",
            dict(adapted_job=fitted["id"], prepared_job=ids["evaluation"]),
        )
        assert comparison["result"]["independent"]
        assert not comparison["result"]["detection_performance_verified"]
        registered = complete("models/register", dict(adapted_job=fitted["id"]))
        model_id = registered["result"]["id"]
        selected = complete(
            "models/" + model_id + "/select",
            dict(comparison_job=comparison["id"]),
            expected_status="failed",
        )
        assert "퇴행" in selected["error"]
        active = complete(
            "analyze",
            dict(
                prepared_job=ids["evaluation"],
                models=["forecast_lstm_h64_l2"],
                use_active=True,
            ),
        )
        assert active["result"]["analyses"][0]["adapted"] is False
        original = complete(
            "analyze",
            dict(
                prepared_job=ids["evaluation"],
                models=["forecast_lstm_h64_l2"],
                use_active=True,
            ),
        )
        assert original["result"]["analyses"][0]["adapted"] is False
        exported = complete("models/" + model_id + "/export", {})
        download = client.get("/api/local-ai/model-exports/" + exported["id"])
        assert download.status_code == 200 and download.content.startswith(
            b"PK"
        )
        assert len(client.get("/api/local-ai/models").json()["models"]) == 1

        calibrated = complete(
            "calibrations",
            dict(
                prepared_jobs=[ids["cal-a"], ids["cal-b"]],
                model_id="forecast_lstm_h64_l2",
                use_active=False,
            ),
        )
        calibration_id = calibrated["result"]["id"]
        assert not calibrated["result"]["active"]
        complete("calibrations/select", dict(id=calibration_id, enabled=True))
        assert client.get("/api/local-ai/calibrations").json()["calibrations"][
            0
        ]["active"]
        result = complete(
            "analyze",
            dict(
                prepared_job=ids["evaluation"], models=["forecast_lstm_h64_l2"]
            ),
        )["result"]["analyses"][0]
        assert (
            result["decision"]["status"] == "normal"
            and result["normal_alarm_rate"] == 0
        )
        reused = complete(
            "analyze",
            dict(prepared_job=ids["cal-a"], models=["forecast_lstm_h64_l2"]),
        )["result"]["analyses"][0]
        assert reused["decision"]["status"] == "unavailable"
        complete("calibrations/select", dict(id=calibration_id, enabled=False))
        assert not client.get("/api/local-ai/calibrations").json()[
            "calibrations"
        ][0]["active"]

    # Model/calibration/manifest hashes must survive a data-directory move.
    # Pause a real two-epoch fit after its first completed epoch, then compare
    # relocation/resume against the uninterrupted two-epoch control.
    pause_code = """
import json, sys
from pathlib import Path
from localflight.adapt import fit
from localflight.catalog import DEFAULT_BUNDLE
root, flight = Path(sys.argv[1]), sys.argv[2]
folder = root / 'local-ai/adapted/interrupted-fixture'
args = ('forecast_lstm_h64_l2', [flight], [])
paused = fit(*args, folder, bundle=DEFAULT_BUNDLE, epochs=2, stop_after_epoch=1)
control = fit(*args, root / 'uninterrupted-control', bundle=DEFAULT_BUNDLE, epochs=2)
print(json.dumps(dict(paused=paused, expected_sha256=control['model_sha256'])))
"""
    paused = json.loads(
        subprocess.run(
            [
                str(project / "ai/.venv/bin/python"),
                "-c",
                pause_code,
                str(tmp_path),
                fixtures["adaptation"],
            ],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    interrupted_id = "f" * 32
    atomic_json(
        tmp_path / "local-ai/jobs" / (interrupted_id + ".json"),
        dict(
            id=interrupted_id,
            status="interrupted",
            action="adapt",
            created_at=0,
            request=dict(
                action="adapt",
                model_id="forecast_lstm_h64_l2",
                adaptation=[fixtures["adaptation"]],
                development=[],
                epochs=2,
                output=paused["paused"]["output"],
            ),
        ),
    )
    before = inventory(tmp_path)
    moved = tmp_path.with_name(tmp_path.name + "-moved")
    migrate(tmp_path, moved)
    assert inventory(tmp_path) == before
    hidden = tmp_path.with_name(tmp_path.name + "-preserved")
    tmp_path.rename(hidden)
    try:
        with TestClient(create_app(moved)) as client:
            assert (
                client.get("/api/local-ai/models").json()["models"][0]["id"]
                == model_id
            )
            assert (
                client.get(
                    "/api/local-ai/model-exports/" + exported["id"]
                ).content
                == download.content
            )
            complete(
                "calibrations/select", dict(id=calibration_id, enabled=True)
            )
            moved_analysis = complete(
                "analyze",
                dict(
                    prepared_job=ids["evaluation"],
                    models=["forecast_lstm_h64_l2"],
                ),
            )
            # Wall-clock duration is not part of the numerical result.
            assert {
                k: v
                for k, v in moved_analysis["result"]["analyses"][0].items()
                if k != "elapsed_seconds"
            } == {k: v for k, v in result.items() if k != "elapsed_seconds"}
            resumed = complete("jobs/" + interrupted_id + "/resume", {})
            assert (
                resumed["result"]["model_sha256"] == paused["expected_sha256"]
            )
            assert resumed["result"]["binding"] == paused["paused"]["binding"]
    finally:
        hidden.rename(tmp_path)


def test_record_only_profile_fields_optional_and_legacy_log_reinspection(
    tmp_path,
):
    with TestClient(create_app(tmp_path)) as client:
        p = dict(
            schema="drone.real-aircraft/1",
            autopilot="ardupilot",
            rotor_count=4,
            series_cells=12,
            capacity_ah=44,
            motor_channels=dict(
                front_right=1, front_left=3, rear_right=4, rear_left=2
            ),
            output_min=1000,
            output_max=2000,
            imu_instance=1,
            battery_instance=0,
            actuator_instance=0,
        )
        response = client.post("/api/local-ai/profiles", json=p)
        assert response.status_code == 200, response.text
        saved = response.json()
        assert saved["profile"]["id"].startswith("aircraft-")
        assert saved["eligibility"]["profile_complete"]
        assert saved["eligibility"]["log_missing_fields"] == []
        assert (
            client.post(
                "/api/local-ai/logs/" + "a" * 32 + "/reinspect"
            ).status_code
            == 422
        )


def test_auto_interval_keeps_saved_profile_and_binds_source(
    tmp_path, monkeypatch
):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        profile = dict(
            schema="drone.real-aircraft/1",
            autopilot="ardupilot",
            rotor_count=4,
            series_cells=12,
            capacity_ah=44,
            motor_channels=dict(
                front_right=1, front_left=3, rear_right=4, rear_left=2
            ),
            output_min=1000,
            output_max=2000,
            imu_instance=1,
            battery_instance=0,
            actuator_instance=0,
        )
        saved = client.post("/api/local-ai/profiles", json=profile).json()
        root = tmp_path / "local-ai"
        job_id = "a" * 32
        (root / "jobs").mkdir(exist_ok=True)
        (root / "jobs" / f"{job_id}.json").write_text(
            json.dumps(
                dict(
                    id=job_id,
                    action="inspect",
                    status="complete",
                    request=dict(path=str(tmp_path / "fixture.BIN")),
                    result=dict(
                        sha256="b" * 64,
                        flight_events=dict(
                            intervals=[
                                dict(
                                    index=0,
                                    start_us=1_000_000,
                                    end_us=20_000_000,
                                    imu_instance=0,
                                    usable=True,
                                    evidence="test events",
                                )
                            ]
                        ),
                    ),
                )
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            LocalFlight, "create_job", lambda self, request: request
        )
        body = dict(
            log_job=job_id,
            profile_id=saved["profile"]["id"],
            profile_version=saved["version"],
            flight_interval=0,
        )
        result = client.post("/api/local-ai/timeline", json=body)
        assert result.status_code == 200, result.text
        assert result.json()["profile"]["imu_instance"] == 0
        assert result.json()["settings"]["expected_source_sha256"] == "b" * 64
        assert result.json()["settings"]["start_us"] == 1_000_000
        assert (
            client.get("/api/local-ai/profiles").json()["profiles"][0][
                "profile"
            ]["imu_instance"]
            == 1
        )
        for bad in (True, -1, 3, "0"):
            assert (
                client.post(
                    "/api/local-ai/timeline",
                    json={**body, "flight_interval": bad},
                ).status_code
                == 422
            )
