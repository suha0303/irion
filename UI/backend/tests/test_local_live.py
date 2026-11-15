"""Check live CPU analysis, staleness and end-of-flight results."""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from arion.contracts import SourceConfig
from arion.main import create_app


def test_udp_bind_address_accepts_lan_not_arbitrary_hostname():
    assert SourceConfig(host="0.0.0.0").host == "0.0.0.0"
    assert SourceConfig(host="192.168.1.10").host == "192.168.1.10"
    with pytest.raises(ValueError):
        SourceConfig(host="example.com")


@pytest.mark.parametrize("automatic", [False, True])
def test_cpu_live_prediction_staleness_and_finish(tmp_path, automatic):
    project = Path(
        os.environ.get("ARION_AI_ROOT", Path(__file__).resolve().parents[4])
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            str(project / p)
            for p in ("ai", "ai/src", "ai/tests", "Data/generation")
        ),
    )
    fixture = subprocess.run(
        [
            str(project / "ai/.venv/bin/python"),
            "-c",
            (
                "import json; from test_localflight_live import "
                "request,samples; "
                "print(json.dumps(dict(request=request(),"
                "rows=list(samples()))))"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(fixture.stdout)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = create_app(tmp_path)
    with TestClient(app) as client:
        version = client.post(
            "/api/local-ai/profiles", json=data["request"]["profile"]
        ).json()["version"]
        response = client.post(
            "/api/sessions",
            json=dict(kind="udp", port=port, name="synthetic CPU integration"),
        )
        assert response.status_code == 200
        run_id = response.json()["run_id"]
        request = {k: v for k, v in data["request"].items() if k != "profile"}
        request["profile_version"] = version
        request["auto_finish_disarm"] = automatic
        url = f"/api/sessions/{run_id}/local-ai"
        assert client.post(url, json=request).status_code == 200
        assert client.post(url, json=request).status_code == 409
        assert (
            client.post("/api/local-ai/benchmark", json={}).status_code == 409
        )
        run = app.state.store.get(run_id)
        offset = (
            time.monotonic_ns()
            - data["rows"][-1]["raw"]["receive_monotonic_ns"]
        )
        for item in data["rows"]:
            item["raw"]["receive_monotonic_ns"] += offset
            run.append_raw(item["raw"])
        deadline = time.monotonic() + 30
        next_boot_ms = 13000
        # Continue receiving while the CPU initializes; old journal rows alone
        # must never be advertised as current predictions.
        while time.monotonic() < deadline:
            next_sample_at = time.monotonic() + 0.1
            for original in data["rows"][-8:]:
                item = json.loads(json.dumps(original))
                raw = item["raw"]
                raw["receive_monotonic_ns"] = time.monotonic_ns()
                fields = raw["fields"]
                if "time_boot_ms" in fields:
                    fields["time_boot_ms"] = next_boot_ms
                if "time_usec" in fields:
                    fields["time_usec"] = next_boot_ms * 1000
                run.append_raw(raw)
            next_boot_ms += 100
            state = client.get(url).json()
            if state["status"] in ("predicting", "error"):
                break
            # Journal fsync is part of the 10 Hz interval, not extra delay.
            time.sleep(max(0.0, next_sample_at - time.monotonic()))
        assert state["status"] == "predicting", state
        assert len(state["prediction"]["values"]) == 4
        assert state["health_verdict"] is None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = client.get(url).json()
            if state["status"] == "stale":
                break
            time.sleep(0.1)
        assert state["status"] == "stale" and state["prediction"] is None
        if automatic:
            assert (
                run_id in app.state.store.active
            )  # initial disarmed and stale did not stop capture
            for mode in (128, 0, 0, 0):
                run.append_raw(
                    dict(
                        source_system=1,
                        source_component=1,
                        receive_monotonic_ns=time.monotonic_ns(),
                        message_name="HEARTBEAT",
                        fields=dict(autopilot=12, base_mode=mode),
                    )
                )
                time.sleep(1.01)
        else:
            assert (
                client.post(f"/api/sessions/{run_id}/finish").status_code == 200
            )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            state = client.get(url).json()
            if state["status"] in ("finished", "error"):
                break
            time.sleep(0.1)
        assert state["status"] == "finished", state
        assert state["finish_reason"] == (
            "confirmed_disarm" if automatic else "manual"
        )
        assert run_id not in app.state.store.active
        assert state["decision"]["status"] == "unavailable"
        assert "분석 중" not in state["decision"]["reason"]
        assert state["result"]["status"] == "complete"
        assert state["insights"]["history_available"]
        assert state["insights"]["report_source"] == "flight_completion"
        assert {
            row["model_id"] for row in state["insights"]["report"]["analyses"]
        } == {request["forecast_model"], request["post_model"]}
        summary = state["result"]["run_summary"]
        assert summary["status"] == "unavailable"
        assert summary["analyzed_us"] > 0
        assert (
            summary["duration_us"]
            == summary["analyzed_us"] + summary["excluded_us"]
        )
        assert state["result"]["analyses"][0]["analysis"]["segment_results"]
        assert (
            state["result"]["analyses"][0]["analysis"]["health_verdict"] is None
        )
        assert client.post(url, json=request).status_code == 422
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(url).json()["status"] == "finished"


def test_configured_ui_port_websocket_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("ARION_PORT", "8879")
    with TestClient(create_app(tmp_path)) as client:
        m = client.post(
            "/api/sessions",
            json=dict(kind="normalized", profile_id="normalized_flight_v1"),
        ).json()
        url = f"/api/ws/runs/{m['run_id']}"
        with client.websocket_connect(
            url, headers={"origin": "http://127.0.0.1:8879"}
        ) as ws:
            assert ws.receive_json()["frame"]["run_id"] == m["run_id"]
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                url, headers={"origin": "http://example.com:8879"}
            ):
                pass
