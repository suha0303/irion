"""Exercise sustained UDP input, slow storage, durable shutdown and I/O errors."""

import asyncio
import json
import os
import socket
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from pymavlink.dialects.v20 import common

from arion.contracts import SourceConfig
from arion.main import create_app


def test_busy_udp_slow_disk_keeps_api_responsive_and_preserves_input(
    tmp_path, monkeypatch
):
    async def check():
        app = create_app(tmp_path)
        sources = app.state.sources
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        run = app.state.store.create_session(SourceConfig(port=port))
        await sources.connect(run)
        address = sources.handles[run.run_id].getsockname()
        fsync = os.fsync

        def slow_fsync(fd):
            time.sleep(0.03)
            fsync(fd)

        monkeypatch.setattr(os, "fsync", slow_fsync)
        mav = common.MAVLink(None, srcSystem=1, srcComponent=1)
        delays = []
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                started = time.monotonic()
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                    for i in range(60):
                        sender.sendto(
                            mav.attitude_encode(
                                i * 50, i / 100, 0, 0, 0, 0, 0
                            ).pack(mav),
                            address,
                        )
                for _ in range(20):
                    response = await client.get("/api/health")
                    assert response.status_code == 200
                    await asyncio.sleep(0.01)
                    now = time.monotonic()
                    delays.append(now - started)
                    started = now
                assert max(delays) < 0.2, delays
                assert run.accepted == 60
            await sources.disconnect(run)
            assert run.session_status()["journal"]["pending_bytes"] == 0
            app.state.store.finish_session(run.run_id)
            rows = [
                json.loads(line) for line in run.path.read_bytes().splitlines()
            ]
            raw = [row["raw"] for row in rows if row.get("type") == "mavlink"]
            assert [r["fields"]["time_boot_ms"] for r in raw] == list(
                range(0, 3000, 50)
            )
            assert len([r for r in rows if r.get("type") == "sample"]) == 60
        finally:
            await sources.close()
            app.state.store.close()

    asyncio.run(check())


def test_receipt_clock_survives_dashboard_pause_and_reader_is_reaped(tmp_path):
    async def check():
        app = create_app(tmp_path)
        sources = app.state.sources
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        run = app.state.store.create_session(SourceConfig(port=port))
        await sources.connect(run)
        receiver = sources.receivers[run.run_id]
        mav = common.MAVLink(None, srcSystem=1, srcComponent=1)

        def aircraft():
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                for i in range(20):
                    sender.sendto(
                        mav.attitude_encode(i * 20, 0, 0, 0, 0, 0, 0).pack(mav),
                        ("127.0.0.1", port),
                    )
                    time.sleep(0.02)

        try:
            # Wait for interpreter startup before testing steady receipt timing.
            await asyncio.sleep(0.2)
            sender = threading.Thread(target=aircraft)
            sender.start()
            time.sleep(0.45)  # Deliberately stall the entire API event loop.
            sender.join()
            for _ in range(100):
                if run.accepted == 20:
                    break
                await asyncio.sleep(0.01)
            assert run.accepted == 20
            await sources.disconnect(run)
            assert receiver.returncode == 0
            rows = [
                json.loads(line) for line in run.path.read_bytes().splitlines()
            ]
            receipt = [
                r["raw"]["receive_monotonic_ns"]
                for r in rows
                if r.get("type") == "mavlink"
            ]
            assert 300_000_000 < receipt[-1] - receipt[0] < 500_000_000
            assert (
                max(b - a for a, b in zip(receipt, receipt[1:])) < 100_000_000
            )
        finally:
            await sources.close()
            app.state.store.close()

    asyncio.run(check())


def test_writer_failure_is_reported_and_flush_does_not_hang(
    tmp_path, monkeypatch
):
    from arion.live_journal import LiveJournal

    def fail(unused_fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", fail)
    journal = LiveJournal((tmp_path / "journal").open("xb"))
    journal.write(b"preserved input\n")
    with pytest.raises(OSError, match="disk full"):
        journal.flush()
    with pytest.raises(OSError, match="disk full"):
        journal.write(b"must not be acknowledged\n")
    with pytest.raises(OSError, match="disk full"):
        journal.close()
    assert not journal.thread.is_alive()


def test_recording_overload_is_explicit_and_existing_bytes_are_preserved(
    tmp_path,
):
    from arion.live_journal import MAX_PENDING_BYTES, LiveJournal

    path = tmp_path / "journal"
    journal = LiveJournal(path.open("xb"))
    journal.write(b"first\n")
    with pytest.raises(OSError, match="버퍼 초과"):
        journal.write(b"x" * (MAX_PENDING_BYTES + 1))
    journal.close()
    assert path.read_bytes() == b"first\n"


def test_replay_loading_does_not_block_health_endpoint(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    store = app.state.store
    run = store.create_session(
        SourceConfig(kind="normalized", profile_id="normalized_flight_v1")
    )
    store.finish_session(run.run_id)
    original = store.get
    loading = threading.Event()
    errors = []

    def slow_load(run_id):
        if not loading.is_set():
            loading.set()
            time.sleep(0.5)
        return original(run_id)

    monkeypatch.setattr(store, "get", slow_load)
    with TestClient(app) as client:

        def open_replay():
            try:
                with client.websocket_connect(
                    f"/api/ws/runs/{run.run_id}"
                ) as ws:
                    assert ws.receive_json()["frame"]["run_id"] == run.run_id
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=open_replay)
        thread.start()
        try:
            assert loading.wait(timeout=2)
            before = time.monotonic()
            assert client.get("/api/health").status_code == 200
            assert time.monotonic() - before < 0.2
        finally:
            thread.join(timeout=3)
        assert not thread.is_alive()
        assert not errors


def test_received_esc_rpm_is_displayed_without_pwm_or_unit_conversion(tmp_path):
    async def check():
        app = create_app(tmp_path)
        sources = app.state.sources
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        run = app.state.store.create_session(SourceConfig(port=port))
        await sources.connect(run)
        # Independent PX4 C encoder fixture, MAVLink2 message 291, CRC extra 10.
        packet = bytes.fromhex(
            "fd38000000010123010040420f0000000000000000006810000034efffff"
            "30110000000070410000704100007041000070410000000000000040"
            "00004040000080404b52"
        )
        corrupted = bytearray(packet)
        corrupted[-1] ^= 1
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(corrupted, ("127.0.0.1", port))
                sender.sendto(packet, ("127.0.0.1", port))
            for _ in range(100):
                if run.accepted:
                    break
                await asyncio.sleep(0.01)
            values = run.frame(run.duration_ms)["values"]
            assert [values[f"motor_{i}_rpm"]["value"] for i in range(1, 5)] == [
                0,
                4200,
                -4300,
                4400,
            ]
            assert values["motor_1_rpm"]["status"] == "valid"
            assert run.accepted == 1 and run.excluded == 1
            assert (
                run.frame(run.duration_ms + 251)["values"]["motor_2_rpm"][
                    "value"
                ]
                is None
            )
        finally:
            await sources.close()
            app.state.store.close()

    asyncio.run(check())
