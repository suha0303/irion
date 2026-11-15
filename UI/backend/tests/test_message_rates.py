"""Check MAVLink rate negotiation, acknowledgments and fallback."""

import asyncio
import json
import os
import pty
import socket

import pytest
from pymavlink.dialects.v20 import common

from arion.contracts import SourceConfig
from arion.store_v2 import StoreV2
from arion.streaming import LiveSources


@pytest.mark.parametrize(
    "transport, startup_delay", [("udp", 10.3), ("serial", 0)]
)
def test_interval_protocol_ack_readback_observation_and_disconnect(
    tmp_path, transport, startup_delay
):
    async def check():
        store = StoreV2(tmp_path)
        sources = LiveSources(store)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        master, slave = pty.openpty() if transport == "serial" else (None, None)
        if master is not None:
            os.set_blocking(master, False)
        run = store.create_session(
            SourceConfig(
                kind=transport,
                port=port,
                device=os.ttyname(slave) if slave is not None else "",
                source_system=201,
                source_component=1,
            )
        )
        await sources.connect(run)
        fc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fc.bind(("127.0.0.1", 0))
        fc.setblocking(False)
        encoder = common.MAVLink(None, srcSystem=201, srcComponent=1)
        decoder = common.MAVLink(None)
        commands = []
        intervals = {}
        target = ("127.0.0.1", port)

        async def emit(msg):
            packet = msg.pack(encoder)
            if master is None:
                fc.sendto(packet, target)
            else:
                os.write(master, packet)
            encoder.seq = (encoder.seq + 1) % 256

        async def aircraft():
            await asyncio.sleep(startup_delay)
            stamp = 0
            while True:
                await emit(encoder.heartbeat_encode(2, 12, 0, 0, 3))
                await emit(encoder.attitude_encode(stamp, 0, 0, 0, 0, 0, 0))
                stamp += 8
                try:
                    data = (
                        fc.recvfrom(65535)[0]
                        if master is None
                        else os.read(master, 65535)
                    )
                except BlockingIOError:
                    data = b""
                for msg in decoder.parse_buffer(data) or []:
                    assert msg.get_type() == "COMMAND_LONG"
                    assert (msg.target_system, msg.target_component) == (201, 1)
                    commands.append(msg.command)
                    assert msg.command in (510, 511)
                    mid = int(msg.param1)
                    if msg.command == 511:
                        intervals[mid] = int(msg.param2)
                        await emit(
                            encoder.command_ack_encode(
                                511, 0, target_system=255, target_component=190
                            )
                        )
                    else:
                        await emit(
                            encoder.message_interval_encode(mid, intervals[mid])
                        )
                await asyncio.sleep(0.008)

        task = asyncio.create_task(aircraft())
        try:
            request = dict(
                source_system=201,
                source_component=1,
                imu_message="HIGHRES_IMU",
                imu_id=0,
                profile=dict(
                    autopilot="px4", battery_instance=0, actuator_instance=0
                ),
            )
            sources.rates.start(run, request)
            with pytest.raises(ValueError, match="이미"):
                sources.rates.start(run, request)
            await asyncio.wait_for(sources.rates.tasks[run.run_id], 30)
            status = sources.rates.status(run.run_id)
            assert status["state"] == "complete" and status["sent"] == 20
            assert len(commands) == 20
            fast_interval = 10_000 if transport == "udp" else 50_000
            assert all(
                intervals[mid] == fast_interval
                for mid in (30, 83, 32, 85, 36, 105)
            )
            assert intervals[245] == 200_000
            assert intervals[147] == 200_000
            esc = next(r for r in status["rows"] if r["name"] == "ESC_STATUS")
            assert esc["required"] is False and esc["verified"] is False
            assert intervals[291] == 50_000
            journal = [
                json.loads(line)
                for line in run.path.read_text(encoding="utf-8").splitlines()
            ]
            requests = [
                row
                for row in journal
                if row.get("type") == "message_rate_request"
            ]
            assert [row["command"] for row in requests] == commands
            assert all(row["target"] == [201, 1] for row in requests)
            assert all(
                r["ack"] == 0 and r["readback_us"] > 0 for r in status["rows"]
            )
            assert next(r for r in status["rows"] if r["name"] == "ATTITUDE")[
                "verified"
            ]
            assert not status[
                "all_verified"
            ]  # ACK without data is not success.
            assert run.path.with_suffix(".rates.json").exists()
            await sources.disconnect(run)
            assert sources.rates.status(run.run_id)["state"] == "disconnected"
            assert not sources.rates.tasks
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            fc.close()
            await sources.close()
            store.close()
            if master is not None:
                os.close(master)
                os.close(slave)

    asyncio.run(check())


def test_timeout_does_not_advance_to_ambiguous_next_command(tmp_path):
    async def check():
        store = StoreV2(tmp_path)
        sources = LiveSources(store)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        run = store.create_session(SourceConfig(port=port))
        await sources.connect(run)
        fc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fc.bind(("127.0.0.1", 0))
        fc.setblocking(False)
        request = dict(
            source_system=201,
            source_component=1,
            imu_message="HIGHRES_IMU",
            imu_id=0,
            profile=dict(
                autopilot="px4", battery_instance=0, actuator_instance=0
            ),
        )
        mav = common.MAVLink(None, srcSystem=201, srcComponent=1)
        # Wrong system acknowledgement cannot complete the request.
        wrong = common.MAVLink(None, srcSystem=202, srcComponent=1)

        async def aircraft():
            while True:
                fc.sendto(
                    mav.heartbeat_encode(2, 12, 0, 0, 3).pack(mav),
                    ("127.0.0.1", port),
                )
                fc.sendto(
                    wrong.command_ack_encode(511, 0).pack(wrong),
                    ("127.0.0.1", port),
                )
                await asyncio.sleep(0.1)

        task = asyncio.create_task(aircraft())
        try:
            sources.rates.start(run, request)
            await asyncio.wait_for(sources.rates.tasks[run.run_id], 5)
            state = sources.rates.status(run.run_id)
            assert state["state"] == "incomplete" and state["sent"] == 1
            assert state["rows"][0]["status"] == "ack_timeout"
            assert all(r["status"] == "pending" for r in state["rows"][1:])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            fc.close()
            await sources.close()
            store.close()

    asyncio.run(check())
