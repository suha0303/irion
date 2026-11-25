"""Run an installed ARION offline analysis and UDP smoke with synthetic inputs."""

import argparse
import hashlib
import json
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from struct import pack

from httpx import Client, ConnectError

PROFILE = {
    "schema": "drone.real-aircraft/1",
    "id": "release-smoke-px4",
    "name": "Release smoke PX4",
    "autopilot": "px4",
    "rotor_count": 4,
    "series_cells": 4,
    "capacity_ah": 12,
    "motor_channels": {
        "front_right": 1,
        "front_left": 2,
        "rear_right": 3,
        "rear_left": 4,
    },
    "output_min": 1000,
    "output_max": 2000,
    "imu_instance": 0,
    "battery_instance": 0,
    "actuator_instance": 0,
}
HEARTBEAT = bytes.fromhex("fd09000000010100000000000000020c000303b331")
TOPICS = (
    ("actuator_outputs", "uint32_t noutputs;float[4] output;", "I4f"),
    ("vehicle_angular_velocity", "float[3] xyz;", "3f"),
    ("vehicle_rates_setpoint", "float roll;float pitch;float yaw;", "3f"),
    ("vehicle_attitude", "float[4] q;", "4f"),
    ("vehicle_acceleration", "float[3] xyz;", "3f"),
    (
        "vehicle_local_position",
        "float x;float y;float z;float vz;uint8_t v_z_valid;",
        "4fB",
    ),
    ("vehicle_local_position_setpoint", "float vz;", "f"),
    ("battery_status", "float voltage_v;float current_a;", "2f"),
    ("vehicle_status", "uint8_t arming_state;uint8_t nav_state;", "BB"),
    ("vehicle_land_detected", "uint8_t landed;", "B"),
)


def write_ulog(path):
    """Write a little-endian ULog fixture with a continuous airborne interval."""
    with path.open("wb") as stream:

        def message(tag, body):
            stream.write(pack("<Hc", len(body), tag) + body)

        stream.write(b"ULog\x01\x12\x35\x01" + pack("<Q", 1_000_000))
        message(b"B", bytes(40))
        for name, fields, _ in TOPICS:
            message(b"F", f"{name}:uint64_t timestamp;{fields}".encode())
        for index, (name, _, _) in enumerate(TOPICS):
            message(b"A", pack("<BH", 0, index) + name.encode())
        for index in range(410):
            phase = index / 13
            moving = 10 <= index < 390
            rows = (
                (4, *(1470 + 22 * math.sin(phase + offset) for offset in range(4))),
                (
                    0.04 * math.sin(phase),
                    0.03 * math.cos(phase),
                    0.01 * math.sin(phase * 0.2),
                ),
                (0.02 * math.sin(phase), 0.02 * math.cos(phase), 0.0),
                (1.0, 0, 0, 0),
                (0.1 * math.sin(phase), 0.1 * math.cos(phase), -9.81),
                (
                    2 * math.sin(phase),
                    2 * math.cos(phase),
                    -4.0 if moving else 0.0,
                    0.0,
                    1,
                ),
                (0.0,),
                (16.4 - index * 0.0015, 8 + math.sin(phase)),
                (2 if moving else 1, 4),
                (0 if moving else 1,),
            )
            for topic, ((_, _, fields), values) in enumerate(
                zip(TOPICS, rows, strict=True)
            ):
                message(
                    b"D",
                    pack("<HQ" + fields, topic, 1_000_000 + index * 100_000, *values),
                )


def wait_for(fn, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        result = fn()
        if result is not None:
            return result
        time.sleep(0.15)
    raise TimeoutError("Timed out waiting for installed ARION")


def verify_egress_block():
    """Require that a public TCP connection cannot leave the test process."""
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=2):
            pass
    except OSError as error:
        return str(error)
    raise RuntimeError("Offline verification requires blocked external egress")


def analyze(client, fixture):
    def get(path):
        response = client.get("/api" + path)
        response.raise_for_status()
        return response.json()

    def post(path, data=None, **kwargs):
        response = client.post("/api" + path, json=data, **kwargs)
        if response.is_error:
            raise RuntimeError(f"{path}: {response.status_code} {response.text}")
        return response.json()

    def done(job):
        def poll():
            row = next(
                row for row in get("/local-ai/jobs")["jobs"] if row["id"] == job["id"]
            )
            if row["status"] in {"queued", "running"}:
                return None
            if row["status"] != "complete":
                raise RuntimeError(row.get("error") or row["status"])
            return row

        return wait_for(poll, 90)

    def health():
        try:
            return get("/health")
        except ConnectError:
            return None

    assert wait_for(health, 12)["edition"] == "v0.2"
    models = [model["id"] for model in get("/local-ai/catalog")["models"]]
    assert len(models) == 8
    with fixture.open("rb") as stream:
        log = done(post("/local-ai/logs", files={"file": (fixture.name, stream)}))
    assert log["result"]["autopilot"] == "px4"
    assert get(f"/local-ai/logs/{log['id']}/monitor")["points"]
    profile = post("/local-ai/profiles", PROFILE)
    preview = done(
        post(
            "/local-ai/timeline",
            {
                "log_job": log["id"],
                "profile_id": PROFILE["id"],
                "profile_version": profile["version"],
                "start_us": 3_000_000,
                "end_us": 38_000_000,
            },
        )
    )
    segments = [
        i
        for i, row in enumerate(preview["result"]["timing"]["candidate_segments"])
        if row["usable"]
    ]
    assert segments
    prepared = done(
        post(
            "/local-ai/prepare",
            {
                "timeline_job": preview["id"],
                "interval_evidence": "Synthetic fixture airborne interval",
                "role": "evaluation",
                "health": "unknown",
                "health_evidence": "",
                "selected_segments": segments,
            },
        )
    )
    result = done(
        post(
            "/local-ai/v02/analyze", {"prepared_job": prepared["id"], "models": models}
        )
    )
    model_result = result["result"]["v02"]
    assert not model_result["failures"] and len(model_result["analyses"]) == 8
    assert all(
        model["evaluation"][0]["scored_frames"] > 0
        for model in model_result["analyses"]
    )
    graph = get(f"/local-ai/v02/jobs/{result['id']}/graphs/{models[0]}/0")
    assert graph["segments"][0]["time_us"]

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.bind(("127.0.0.1", 0))
        port = sender.getsockname()[1]
    session = post(
        "/sessions",
        {
            "kind": "udp",
            "name": "Release smoke MAVLink",
            "host": "127.0.0.1",
            "port": port,
            "source_system": 1,
            "source_component": 1,
        },
    )
    path = (
        Path(os.environ["ARION_DATA_DIR"])
        / "sessions"
        / (session["run_id"] + ".mavlink.bin")
    )
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:

        def receive():
            sender.sendto(HEARTBEAT, ("127.0.0.1", port))
            return True if path.exists() and HEARTBEAT in path.read_bytes() else None

        wait_for(receive, 10)
    finished = post(f"/sessions/{session['run_id']}/finish", {})
    assert finished["storage"] == "udp-wire/1"
    assert finished["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "models": len(models),
        "frames": model_result["coverage"]["retained_frames"],
        "udp_original": True,
    }


def run(core, output):
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "release-smoke.ulg"
    write_ulog(fixture)
    (output / "release-smoke-profile.json").write_text(
        json.dumps(PROFILE, indent=2) + "\n", encoding="utf-8"
    )
    offline = verify_egress_block()
    runtime = core / "runtime"
    backend = core / "irion-upload/UI/backend"
    data = output / "data"
    env = dict(
        os.environ,
        PYTHONHOME=str(runtime),
        PYTHONNOUSERSITE="1",
        PYTHONUTF8="1",
        PYTHONPATH=os.pathsep.join(map(str, (core / "ai", core / "ai/src", backend))),
        ARION_AI_ROOT=str(core),
        ARION_AI_PYTHON=sys.executable,
        ARION_DATA_DIR=str(data),
        ARION_V02_INFLIGHT_ROOT=str(core / "models/v02"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    previous = os.environ.get("ARION_DATA_DIR")
    os.environ["ARION_DATA_DIR"] = str(data)
    try:
        with (output / "backend.log").open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-s", "-B", "-m", "arion.desktop"],
                cwd=backend,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
            )
            try:
                ready = queue.Queue()

                def read_port():
                    ready.put(process.stdout.readline())

                threading.Thread(target=read_port, daemon=True).start()
                try:
                    port = json.loads(ready.get(timeout=20))["port"]
                except queue.Empty as error:
                    raise TimeoutError(
                        "ARION did not announce its loopback port"
                    ) from error
                with Client(
                    base_url=f"http://127.0.0.1:{port}", timeout=25, trust_env=False
                ) as client:
                    result = analyze(client, fixture)
            finally:
                process.stdin.close()
                try:
                    code = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise
            assert code == 0
    finally:
        if previous is None:
            del os.environ["ARION_DATA_DIR"]
        else:
            os.environ["ARION_DATA_DIR"] = previous
    result.update(offline_egress=offline, platform=sys.platform, backend_exit=code)
    (output / "release-check.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.core.resolve(), args.output.resolve())
