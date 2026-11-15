"""Bounded, opt-in MAVLink stream interval configuration; no flight commands."""

import asyncio
import time
from collections import deque

from pymavlink.dialects.v20 import common

from arion import local_flight

MESSAGES = [
    (0, "HEARTBEAT", 1),
    (245, "EXTENDED_SYS_STATE", 5),
    (30, "ATTITUDE", 20),
    (83, "ATTITUDE_TARGET", 20),
    (32, "LOCAL_POSITION_NED", 20),
    (85, "POSITION_TARGET_LOCAL_NED", 20),
    (36, "SERVO_OUTPUT_RAW", 20),
    (147, "BATTERY_STATUS", 5),
    (291, "ESC_STATUS", 20),
]
IMUS = {
    "HIGHRES_IMU": 105,
    "SCALED_IMU": 26,
    "SCALED_IMU2": 116,
    "SCALED_IMU3": 129,
}


def input_policy(run, request):
    """Pin requested acquisition rates without assuming the FC delivers them.

    UDP can carry the 100 Hz fast streams used by the native training capture.
    Serial keeps its existing bounded rates: increasing these without knowing
    the shared link budget can cause more packet loss. Neither is a detection
    performance guarantee; ACK, readback and observed rates remain separate.
    """
    fast_hz = 100 if run.config.kind == "udp" else 20
    fast = {
        "ATTITUDE",
        "ATTITUDE_TARGET",
        "LOCAL_POSITION_NED",
        "POSITION_TARGET_LOCAL_NED",
        "SERVO_OUTPUT_RAW",
    }
    rates = {name: fast_hz if name in fast else hz for _, name, hz in MESSAGES}
    rates[request["imu_message"]] = fast_hz
    return dict(
        schema="mavlink-acquisition/1",
        transport=run.config.kind,
        rates_hz=rates,
    )


class MessageRates:
    """Negotiate telemetry intervals and track acknowledgments per live
    session.
    """

    def __init__(self, sources):
        self.sources = sources
        self.heartbeats = {}
        self.sessions = {}
        self.tasks = {}

    def observe(self, run, msg, peer):
        now = time.monotonic()
        identity = (msg.get_srcSystem(), msg.get_srcComponent())
        fields = msg.to_dict()
        if msg.get_type() == "HEARTBEAT" and fields.get("autopilot") in (3, 12):
            cache = self.heartbeats.setdefault(run.run_id, {})
            if len(cache) < 8 or identity in cache:
                cache[identity] = dict(
                    peer=peer, at=now, autopilot=fields["autopilot"]
                )
        session = self.sessions.get(run.run_id)
        if (
            not session
            or identity != session["target"]
            or peer != session.get("peer")
        ):
            return
        if msg.get_msgId() in session["samples"]:
            selected = session["request"]
            correct_instance = (
                (
                    msg.get_type() != "BATTERY_STATUS"
                    or fields.get("id")
                    == selected["profile"]["battery_instance"]
                )
                and (
                    msg.get_type() != "SERVO_OUTPUT_RAW"
                    or fields.get("port")
                    == selected["profile"]["actuator_instance"]
                )
                and (
                    msg.get_type() != "HIGHRES_IMU"
                    or fields.get("id", 0) == selected["imu_id"]
                )
            )
            if correct_instance:
                session["samples"][msg.get_msgId()].append(now)
        pending = session.get("pending")
        if not pending or pending.done():
            return
        if (
            session["waiting"] == "ack"
            and msg.get_type() == "COMMAND_ACK"
            and fields.get("command") == 511
        ):
            if fields.get("target_system", 0) not in (0, 255) or fields.get(
                "target_component", 0
            ) not in (0, 190):
                return
            if (
                fields.get("result") != 5
            ):  # IN_PROGRESS is not a final acknowledgement.
                pending.set_result(fields)
        elif (
            session["waiting"] == "interval"
            and msg.get_type() == "MESSAGE_INTERVAL"
            and fields.get("message_id") == session["message_id"]
        ):
            pending.set_result(fields)

    def start(self, run, request):
        if run.run_id in self.tasks:
            raise ValueError("메시지 주기 요청이 이미 진행 중입니다.")
        if (
            not run.live
            or run.config.kind == "normalized"
            or run.run_id not in self.sources.handles
        ):
            raise ValueError("연결된 UDP/시리얼 세션이 필요합니다.")
        if request["imu_message"] not in IMUS:
            raise ValueError("Unsupported IMU message")
        target = (request["source_system"], request["source_component"])
        if any(type(v) is not int or not 1 <= v <= 255 for v in target):
            raise ValueError("명시적 system/component ID 필요")
        if target == (255, 190):
            raise ValueError("요청기와 대상 MAVLink ID가 겹칩니다.")
        if any(
            v is not None and v != target[i]
            for i, v in enumerate(
                (run.config.source_system, run.config.source_component)
            )
        ):
            raise ValueError("수신 필터와 AI 대상 ID가 다릅니다.")
        policy = input_policy(run, request)
        if request.get("telemetry_policy", policy) != policy:
            raise ValueError(
                "메시지 주기 설정과 고정된 AI 입력 설정이 다릅니다."
            )
        rows = [
            dict(
                id=i,
                name=n,
                hz=policy["rates_hz"][n],
                ack=None,
                readback_us=None,
                status="pending",
                required=n != "ESC_STATUS",
            )
            for i, n, _ in MESSAGES
        ]
        rows.insert(
            3,
            dict(
                id=IMUS[request["imu_message"]],
                name=request["imu_message"],
                hz=policy["rates_hz"][request["imu_message"]],
                ack=None,
                readback_us=None,
                status="pending",
                required=True,
            ),
        )
        session = dict(
            target=target,
            peer=None,
            request=request,
            policy=policy,
            rows=rows,
            state="waiting_heartbeat",
            samples={r["id"]: deque(maxlen=512) for r in rows},
            pending=None,
            encoder=common.MAVLink(None, srcSystem=255, srcComponent=190),
            sent=0,
        )
        self.sessions[run.run_id] = session
        self.tasks[run.run_id] = asyncio.create_task(
            self._configure(run, session)
        )
        return self.status(run.run_id)

    async def _send(self, run, session, command, message_id, interval=0):
        if command not in (510, 511):
            raise ValueError("Only GET/SET_MESSAGE_INTERVAL are allowed")
        if not run.live:
            raise ValueError("종료된 세션")
        identity = self.heartbeats.get(run.run_id, {}).get(session["target"])
        if (
            not identity
            or time.monotonic() - identity["at"] > 3
            or identity["peer"] != session["peer"]
        ):
            raise ValueError("HEARTBEAT 끊김/UDP peer 변경: 전송 중단")
        mav = session["encoder"]
        packet = mav.command_long_encode(
            *session["target"], command, 0, message_id, interval, 0, 0, 0, 0, 0
        ).pack(mav)
        mav.seq = (mav.seq + 1) % 256
        handle = self.sources.handles[run.run_id]
        if run.config.kind == "udp":
            # Nonblocking datagram write works on bundled Python 3.10 as well.
            # Backpressure/OSError is reported; never block the receiver.
            if handle.sendto(packet, session["peer"]) != len(packet):
                raise ValueError("UDP 설정 요청 일부만 전송됨")
        else:
            if handle.write(packet) != len(packet):
                raise ValueError("시리얼 설정 요청 일부만 전송됨")
        session["sent"] += 1
        run.record_message_rate_request(
            session["target"], command, message_id, interval
        )

    async def _exchange(self, run, session, row, command):
        session["waiting"] = "ack" if command == 511 else "interval"
        session["message_id"] = row["id"]
        future = asyncio.get_running_loop().create_future()
        session["pending"] = future
        try:
            await self._send(
                run,
                session,
                command,
                row["id"],
                round(1e6 / row["hz"]) if command == 511 else 0,
            )
            return await asyncio.wait_for(future, 2)
        finally:
            session["pending"] = None

    async def _configure(self, run, session):
        try:
            # Starting AI before the FC is normal. Wait until this explicitly
            # selected connection ends; never send to an unobserved peer.
            while run.live:
                hb = self.heartbeats.get(run.run_id, {}).get(session["target"])
                if hb and time.monotonic() - hb["at"] < 3:
                    expected = (
                        12
                        if session["request"]["profile"]["autopilot"] == "px4"
                        else 3
                    )
                    if hb["autopilot"] != expected:
                        raise ValueError(
                            "HEARTBEAT autopilot과 프로파일 불일치"
                        )
                    session["peer"] = hb["peer"]
                    break
                await asyncio.sleep(0.1)
            else:
                return
            session["state"] = "configuring"
            for row in session["rows"]:
                row["status"] = "requesting"
                try:
                    ack = await self._exchange(run, session, row, 511)
                except asyncio.TimeoutError:
                    row["status"] = "ack_timeout"
                    # ACK has no message ID. Stop to avoid attributing a late
                    # ACK
                    # to the next SET command. Manual retry starts a new
                    # attempt.
                    session["state"] = "incomplete"
                    return
                row["ack"] = ack["result"]
                if ack["result"] != 0:
                    row["status"] = "rejected"
                    continue
                try:
                    reply = await self._exchange(run, session, row, 510)
                    row["readback_us"] = reply["interval_us"]
                    row["status"] = "accepted"
                except asyncio.TimeoutError:
                    row["status"] = "accepted_no_readback"
            session["state"] = "observing"
            await asyncio.sleep(5)
            session["state"] = "complete"
        except asyncio.CancelledError:
            session["state"] = "interrupted"
            raise
        except (ValueError, OSError, KeyError) as error:
            session.update(state="error", error=str(error))
        finally:
            try:
                await asyncio.to_thread(
                    local_flight.write,
                    run.path.with_suffix(".rates.json"),
                    self.status(run.run_id),
                )
            finally:
                self.tasks.pop(run.run_id, None)

    def status(self, run_id):
        session = self.sessions.get(run_id)
        if not session:
            return dict(state="not_requested", rows=[], sent=0)
        now = time.monotonic()
        rows = []
        for row in session["rows"]:
            samples = [t for t in session["samples"][row["id"]] if now - t <= 5]
            hz = (
                (len(samples) - 1) / (samples[-1] - samples[0])
                if len(samples) >= 3 and samples[-1] > samples[0]
                else 0
            )
            fresh = bool(samples) and now - samples[-1] <= max(
                1.0, 3 / row["hz"]
            )
            readback = row["readback_us"]
            correct = (
                readback is not None
                and readback > 0
                and abs(readback - 1e6 / row["hz"]) <= 0.2 * (1e6 / row["hz"])
            )
            verified = (
                row["ack"] == 0 and correct and fresh and hz >= 0.7 * row["hz"]
            )
            rows.append(
                dict(
                    row,
                    observed_hz=hz if fresh else 0,
                    verified=verified,
                    state="verified" if verified else row["status"],
                )
            )
        return dict(
            state=session["state"],
            error=session.get("error"),
            rows=rows,
            sent=session["sent"],
            all_verified=all(r["verified"] for r in rows if r["required"]),
            target=list(session["target"]),
            policy=session["policy"],
            note="대상 HEARTBEAT까지 대기. ESC_STATUS는 선택 신호이며 FC/ESC 미지원 시 RPM을 추정하지 않습니다. ACK·조회·수신률과 AI 입력 적격성은 별도.",
        )

    async def disconnect(self, run_id):
        task = self.tasks.get(run_id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.heartbeats.pop(run_id, None)
        session = self.sessions.get(run_id)
        if session:
            session["state"] = "disconnected"
            for samples in session["samples"].values():
                samples.clear()
