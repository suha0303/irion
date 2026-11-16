"""Exercise lost/delayed ACKs over UDP without accepting another row's ACK."""

import asyncio
import socket
import time
from types import SimpleNamespace

import pytest
from pymavlink.dialects.v20 import common

from arion.message_rates import MessageRates


@pytest.mark.parametrize(
    "failure", ["lost_ack", "lost_request", "late_ack", "wrong_source"]
)
def test_configuration_recovers_only_with_matching_readback(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(MessageRates, "RESPONSE_TIMEOUT", 0.04, raising=False)

    async def check():
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for handle in (client, fc):
            handle.bind(("127.0.0.1", 0))
            handle.setblocking(False)
        run = SimpleNamespace(
            run_id="recovery",
            live=True,
            path=tmp_path / "source.jsonl",
            config=SimpleNamespace(
                kind="udp", source_system=201, source_component=1
            ),
            record_message_rate_request=lambda *args: None,
        )
        rates = MessageRates(SimpleNamespace(handles={run.run_id: client}))
        encoder = common.MAVLink(None, srcSystem=201, srcComponent=1)
        wrong = common.MAVLink(None, srcSystem=202, srcComponent=1)
        decoder = common.MAVLink(None)
        requests = []
        intervals = {30: 50_000}
        set_counts = {}
        delayed = []
        target = client.getsockname()

        def emit(message, mav=encoder):
            fc.sendto(message.pack(mav), target)
            mav.seq = (mav.seq + 1) % 256

        async def aircraft():
            while True:
                emit(encoder.heartbeat_encode(2, 12, 0, 0, 3))
                emit(encoder.attitude_encode(0, 0, 0, 0, 0, 0, 0))
                for when, message in delayed[:]:
                    if time.monotonic() >= when:
                        emit(message)
                        delayed.remove((when, message))
                try:
                    data = fc.recvfrom(65535)[0]
                except BlockingIOError:
                    data = b""
                for msg in decoder.parse_buffer(data) or []:
                    assert msg.command in (510, 511)
                    mid = int(msg.param1)
                    requests.append((msg.command, mid, msg.confirmation))
                    if msg.command == 511:
                        set_counts[mid] = set_counts.get(mid, 0) + 1
                        lost = mid == 245 and (
                            set_counts[mid] == 1 or failure == "wrong_source"
                        )
                        blocked = failure == "late_ack" and mid == 30
                        if not blocked and not (
                            lost and failure == "lost_request"
                        ):
                            intervals[mid] = int(msg.param2)
                        ack = encoder.command_ack_encode(
                            511,
                            4 if blocked else 0,
                            target_system=255,
                            target_component=190,
                        )
                        if lost:
                            if failure == "late_ack":
                                delayed.append((time.monotonic() + 0.055, ack))
                        else:
                            emit(ack)
                    else:
                        if failure == "wrong_source" and mid == 245:
                            emit(
                                wrong.message_interval_encode(
                                    mid, intervals.get(mid, 0)
                                ),
                                wrong,
                            )
                        else:
                            emit(
                                encoder.message_interval_encode(
                                    mid, intervals.get(mid, 0)
                                )
                            )
                await asyncio.sleep(0.002)

        async def receive():
            decoder = common.MAVLink(None)
            while True:
                data, peer = await asyncio.get_running_loop().sock_recvfrom(
                    client, 65535
                )
                for msg in decoder.parse_buffer(data) or []:
                    rates.observe(run, msg, peer)

        tasks = [
            asyncio.create_task(aircraft()),
            asyncio.create_task(receive()),
        ]
        try:
            rates.start(
                run,
                dict(
                    source_system=201,
                    source_component=1,
                    imu_message="HIGHRES_IMU",
                    imu_id=0,
                    profile=dict(
                        autopilot="px4", battery_instance=0, actuator_instance=0
                    ),
                ),
            )
            await asyncio.wait_for(rates.tasks[run.run_id], 8)
            result = rates.status(run.run_id)
            rows = {r["id"]: r for r in result["rows"]}
            if failure in ("lost_ack", "lost_request"):
                assert result["state"] == "complete"
                assert rows[245]["ack"] is None
                assert rows[245]["readback_us"] == 200_000
                assert rows[245]["status"] == "readback_confirmed"
                assert rows[105]["readback_us"] == 10_000
                assert rows[30]["verified"]
                assert not rows[105][
                    "verified"
                ]  # Configuration is not data receipt.
                assert not result["all_verified"]
                if failure == "lost_request":
                    assert [
                        c
                        for command, mid, c in requests
                        if command == 511 and mid == 245
                    ] == [0, 1]
            else:
                assert result["state"] == "incomplete"
                blocked = 30 if failure == "late_ack" else 245
                assert rows[blocked]["attempts"] == 3
                assert rows[blocked]["ack"] is None
                assert not rows[blocked]["verified"]
                assert rows[105]["status"] == "pending"
        finally:
            await rates.disconnect(run.run_id)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            client.close()
            fc.close()

    asyncio.run(check())
