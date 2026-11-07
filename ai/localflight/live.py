"""Passive MAVLink journal -> bounded CPU forecasts -> explicit end-of-flight report.

Receipt-clock sampling is deliberate: no inferred network latency correction. Source
clocks detect duplicates/reboots; gaps reset history. Never sends a vehicle command.
"""

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from capstone_ai.mapping import FEATURES

from .analysis import analyze, load_runtime
from .catalog import DEFAULT_BUNDLE, Catalog, digest, read_json
from .prepare import causal_frames
from .prepared import REVISION
from .profiles import eligibility
from .registry import active_folder


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def validate(request):
    if type(request.get("auto_finish_disarm", False)) is not bool:
        raise ValueError("자동 종료 설정은 참/거짓이어야 합니다.")
    profile = request["profile"]
    e = eligibility(profile)
    if not e["profile_complete"] or not e["model_structure_compatible"]:
        raise ValueError("완전한 4로터 기체 프로파일과 모터 매핑이 필요합니다.")
    if any(
        profile.get(k) is None
        for k in ("battery_instance", "actuator_instance")
    ):
        raise ValueError("배터리 ID와 출력 port를 지정해 주세요.")
    for key in ("source_system", "source_component"):
        if type(request.get(key)) is not int or not 1 <= request[key] <= 255:
            raise ValueError(
                "MAVLink system/component ID 1..255를 지정해 주세요."
            )
    if request.get("scope") not in ("real_aircraft", "simulation_test"):
        raise ValueError("실기체 또는 시뮬레이터 통신 시험을 명시하세요.")
    if request.get("imu_message") not in (
        "HIGHRES_IMU",
        "SCALED_IMU",
        "SCALED_IMU2",
        "SCALED_IMU3",
    ):
        raise ValueError("IMU 메시지를 선택하세요.")
    if (
        type(request.get("imu_id")) is not int
        or not 0 <= request["imu_id"] <= 255
    ):
        raise ValueError("MAVLink IMU ID가 필요합니다.")
    catalog = Catalog(request.get("bundle", DEFAULT_BUNDLE))
    for key, task in (
        ("forecast_model", "forecast"),
        ("post_model", "run_health"),
    ):
        if catalog.entry(request[key])["task"] != task:
            raise ValueError(
                "비행 중 모델과 종료 후 모델의 종류가 맞지 않습니다."
            )
    return dict(
        valid=True,
        transport="MAVLink telemetry with optional interval requests",
        clock="receipt_monotonic",
        origin="synthetic_unit_fixture"
        if request["scope"] == "simulation_test"
        else profile["autopilot"] + "_real",
    )


def number(value):
    return (
        float(value)
        if type(value) in (int, float) and math.isfinite(value)
        else None
    )


def decode(name, f, p, request):
    """common.xml SI/FRD/NED semantics, explicit channels/instances and masks."""
    if name == "ATTITUDE":
        return dict(
            zip(
                ("roll", "pitch", "body_rate_p", "body_rate_q", "body_rate_r"),
                (
                    number(f.get(k))
                    for k in (
                        "roll",
                        "pitch",
                        "rollspeed",
                        "pitchspeed",
                        "yawspeed",
                    )
                ),
            )
        )
    if name == "ATTITUDE_TARGET":
        mask = f.get("type_mask", 255)
        return {
            "rate_target_" + a: None if mask & (1 << i) else number(f.get(k))
            for i, (a, k) in enumerate(
                zip(
                    "pqr",
                    ("body_roll_rate", "body_pitch_rate", "body_yaw_rate"),
                )
            )
        }
    if name == request["imu_message"]:
        if name == "HIGHRES_IMU" and f.get("id", 0) != request["imu_id"]:
            return {}
        scale = 1 if name == "HIGHRES_IMU" else 9.80665 / 1000
        return {
            "specific_force_" + a: (
                number(f.get(a + "acc")) * scale
                if number(f.get(a + "acc")) is not None
                else None
            )
            for i, a in enumerate("xyz")
            if name != "HIGHRES_IMU" or f.get("fields_updated", 0) & (1 << i)
        }
    if name == "LOCAL_POSITION_NED":
        v = number(f.get("vz"))
        return dict(climb_rate=-v if v is not None else None)
    if name == "POSITION_TARGET_LOCAL_NED":
        v = number(f.get("vz"))
        # Only LOCAL_NED. Body/local-offset frames require a separate conversion.
        return dict(
            climb_target=-v
            if v is not None
            and f.get("coordinate_frame") == 1
            and not f.get("type_mask", 65535) & 32
            else None
        )
    if name == "SERVO_OUTPUT_RAW" and f.get("port") == p["actuator_instance"]:
        return {
            "motor_request_" + pos: (
                (number(f.get(f"servo{channel}_raw")) - p["output_min"])
                / (p["output_max"] - p["output_min"])
                if number(f.get(f"servo{channel}_raw")) is not None
                else None
            )
            for pos, channel in p["motor_channels"].items()
        }
    if name == "BATTERY_STATUS" and f.get("id") == p["battery_instance"]:
        # MAVLink uses summed mV cells, UINT16_MAX for unused primary slots,
        # and 0 for unused extension slots. Capacity is never inferred here.
        cells = f.get("voltages", [])
        valid = [number(v) for v in cells if v != 65535]
        extension = [
            number(v) for v in f.get("voltages_ext", []) if v not in (0, 65535)
        ]
        voltage = (
            sum(valid + extension) / 1000
            if valid and all(v is not None and v > 0 for v in valid + extension)
            else None
        )
        current = number(f.get("current_battery"))
        return dict(
            pack_voltage=voltage,
            pack_current=None
            if current is None or current == -1
            else current / 100,
        )
    return {}


class DisarmGuard:
    """Require an observed arming and three fresh disarmed heartbeats over 2 s."""

    def __init__(self):
        self.seen_armed = False
        self.first = self.last = None
        self.count = 0

    def accept(self, armed, timestamp):
        if armed is True:
            self.seen_armed = True
        if armed is not False or not self.seen_armed:
            self.first = self.last = None
            self.count = 0
            return
        if self.last is not None and timestamp <= self.last:
            return
        if self.last is None or timestamp - self.last > 3_000_000:
            self.first, self.count = timestamp, 0
        self.last = timestamp
        self.count += 1

    @property
    def ready(self):
        return self.count >= 3 and self.last - self.first >= 2_000_000


class Mapper:
    def __init__(self, request):
        self.request = request
        self.profile = request["profile"]
        self.base = None
        self.previous = {}
        self.heartbeat = None
        self.last_us = 0
        self.last_receipt = None
        self.messages = {}
        self.resets = 0
        self.duplicates = 0
        self.armed = None
        self.disarm = DisarmGuard()

    def accept(self, item):
        if item.get("type") != "mavlink":
            return []
        raw = item["raw"]
        if (raw["source_system"], raw["source_component"]) != (
            self.request["source_system"],
            self.request["source_component"],
        ):
            return []
        ns = raw["receive_monotonic_ns"]
        if type(ns) is not int or ns < 0:
            raise ValueError("Invalid receipt timestamp")
        if self.base is None:
            self.base = ns
        # A monotonic-clock reset is a hard stop, never fabricated elapsed time.
        if self.last_receipt is not None and ns < self.last_receipt:
            raise ValueError("수신 시계 역행: 새 세션이 필요합니다.")
        self.last_receipt = ns
        t = (ns - self.base) // 1000
        self.last_us = t
        name, fields = raw["message_name"], raw["fields"]
        self.messages[name] = self.messages.get(name, 0) + 1
        if name == "HEARTBEAT":
            expected = 12 if self.profile["autopilot"] == "px4" else 3
            if fields.get("autopilot") != expected:
                raise ValueError(
                    "HEARTBEAT autopilot과 기체 프로파일이 다릅니다."
                )
            self.heartbeat = t
            mode = fields.get("base_mode")
            self.armed = (
                bool(mode & 128)
                if type(mode) is int and 0 <= mode <= 255
                else None
            )
            self.disarm.accept(self.armed, t)
        events = []
        updates = decode(name, fields, self.profile, self.request)
        clock = fields.get("time_boot_ms", fields.get("time_usec"))
        key = (name, fields.get("id"), fields.get("port"))
        old = self.previous.get(key)
        if item.get("boundary") or (
            old is not None and clock is not None and clock < old
        ):
            events.append((t, t, {"__reset__": True}))
            self.previous.clear()
            self.resets += 1
            old = None
        if clock is not None:
            self.previous[key] = clock
        if clock is not None and old == clock:
            self.duplicates += 1
            return events
        if updates:
            events.append((t, t, updates))
        return events


def completed(events, journal, output, request, origin, *, end_us=None):
    """Archive retained segments in bounded packages; analyze each without gap bridging."""
    from .run_summary import aggregate

    end_us = (
        end_us
        if end_us is not None
        else events[-1][0] + 100000
        if events
        else 0
    )
    context = dict(
        start_us=0,
        end_us=end_us,
        extent_known=True,
        clock="receipt_monotonic",
        filename=journal.name,
        notes=[],
    )
    if end_us < 11_000_000:
        return dict(
            status="insufficient_data",
            message="11초 이상의 완전한 입력 구간이 없습니다.",
            analyses=[],
            run_summary=aggregate(
                [],
                context=context,
                spans=[],
                exclusions=[
                    dict(start_us=0, end_us=end_us, reason="too_short")
                ],
            ),
        )
    values, times, timing = causal_frames(events, 0, end_us)
    segments = [s for s in timing["candidate_segments"] if s["usable"]]
    output.mkdir(exist_ok=False)
    source_hash = digest(journal)
    shutil.copyfile(journal, output / "source.jsonl")
    if digest(output / "source.jsonl") != source_hash:
        raise ValueError("종료된 원본 저널이 변경되었습니다.")
    reports = []
    for i, segment in enumerate(segments):
        a, b = segment["start_index"], segment["end_index"]
        # Balanced chunks avoid discarding a short final remainder.
        chunks = np.array_split(np.arange(a, b), math.ceil((b - a) / 10000))
        for j, indices in enumerate(chunks):
            folder = output / f"segment-{i}-{j}"
            folder.mkdir()
            # Immutable local hard links avoid copying the entire flight per segment.
            os.link(output / "source.jsonl", folder / "source.jsonl")
            np.savez(
                folder / "frames.npz",
                values=values[indices],
                time_us=times[indices],
            )
            m = dict(
                schema="drone.real-common19/1",
                preprocessing_revision=REVISION,
                feature_order=FEATURES,
                units="physical_pack_V_A",
                profile=request["profile"],
                origin=origin,
                live_input=dict(
                    imu_message=request["imu_message"], imu_id=request["imu_id"]
                ),
                role="evaluation",
                health="unknown",
                health_evidence="",
                mapping_verified=True,
                mapping_evidence=request["profile"].get("mapping_evidence")
                or "사용자가 저장한 MAVLink 채널/인스턴스 설정",
                mapping_scope="Declared MAVLink receipt-clock adapter; real performance unverified",
                source_path="source.jsonl",
                source_sha256=source_hash,
                file=dict(
                    path="frames.npz", sha256=digest(folder / "frames.npz")
                ),
                timing=dict(
                    timing,
                    clock="receipt_monotonic",
                    coverage=timing["complete_frames"]
                    / timing["requested_frames"],
                ),
                run_context=context,
            )
            write(folder / "manifest.json", m)
            result = analyze(
                request["post_model"],
                folder / "manifest.json",
                bundle=request.get("bundle", DEFAULT_BUNDLE),
                adapted=request.get("post_adapted"),
                data_root=request.get("data_root"),
            )
            reports.append(
                dict(manifest=str(folder / "manifest.json"), analysis=result)
            )
    return dict(
        status="complete" if reports else "insufficient_data",
        analyses=reports,
        timing=timing,
        source_sha256=source_hash,
        health_verdict=None,
        run_summary=aggregate(
            [r["analysis"] for r in reports],
            context=context,
            spans=[
                s
                for r in reports
                for s in r["analysis"]["run_evidence"]["spans"]
            ],
            exclusions=[
                *timing["excluded_intervals"],
                *[
                    dict(
                        start_us=s["start_us"],
                        end_us=s["end_us"],
                        reason="too_short",
                    )
                    for s in timing["candidate_segments"]
                    if not s["usable"]
                ],
            ],
        ),
        message="보존된 연속 구간별 분석 완료"
        if reports
        else "19개 입력이 모두 보존된 11초 구간이 없습니다.",
    )


def arrived_decision(
    pending, values, times, segment_start, reset, cells, scaler, calibration
):
    """Score only an arrived exact target within the same uninterrupted segment."""
    from .calibration import decide, forecast_scores

    future = [p for p in pending if p["due"] > int(times[-1])]
    decision = None
    for prior in pending:
        if (
            prior["due"] > int(times[-1])
            or prior["reset"] != reset
            or prior["origin"] < segment_start
        ):
            continue
        index = np.searchsorted(times, prior["due"])
        if index >= len(times) or int(times[index]) != prior["due"]:
            continue
        observed = values[index, [4, 5, 6, 17]].copy()
        observed[3] /= cells
        scaled = (observed - scaler["y_mean"]) / scaler["y_scale"]
        score = forecast_scores(
            scaled[None], np.array(prior["scaled"])[None], scaler
        )
        decision = dict(decide(calibration, score), time_us=prior["due"])
    return future, decision


def run(request):
    admitted = validate(request)
    folder = Path(request["output"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "state.json"
    state = dict(
        status="starting",
        profile_id=request["profile"]["id"],
        forecast_model=request["forecast_model"],
        post_model=request["post_model"],
        scope=request["scope"],
        clock="receipt_monotonic",
        health_verdict=None,
        notice="전송 지연 자동 보정 없음 · 보정 기준 판정은 실기체 성능 검증과 구분",
    )
    write(path, state)
    metadata = dict(profile=request["profile"], origin=admitted["origin"])
    bundle = request.get("bundle", DEFAULT_BUNDLE)
    adapted = (
        active_folder(
            request["data_root"],
            request["profile"],
            request["forecast_model"],
            bundle=bundle,
        )
        if request.get("use_active")
        else None
    )
    runtime, adaptation = load_runtime(
        request["forecast_model"], metadata, bundle=bundle, adapted=adapted
    )
    post_adapted = (
        active_folder(
            request["data_root"],
            request["profile"],
            request["post_model"],
            bundle=bundle,
        )
        if request.get("use_active")
        else None
    )
    if post_adapted:
        # Validate and pin the selected immutable version when recording starts.
        post_runtime, post_adaptation = load_runtime(
            request["post_model"], metadata, bundle=bundle, adapted=post_adapted
        )
        del post_runtime
        request["post_adapted"] = str(post_adapted)
        state["post_adapted_sha256"] = post_adaptation["model_sha256"]
    state["adapted_sha256"] = adaptation["model_sha256"] if adaptation else None
    from .calibration import active, binding

    metadata["live_input"] = dict(
        imu_message=request["imu_message"], imu_id=request["imu_id"]
    )
    metadata["timing"] = dict(
        clock="receipt_monotonic", fast_max_age_ms=200, battery_max_age_ms=350
    )
    try:
        calibration = active(
            request["data_root"], binding(runtime, metadata, adaptation)
        )
        calibration_error = None
    except (ValueError, KeyError, OSError) as error:
        calibration, calibration_error = None, str(error)
    pending_predictions = []
    mapper = Mapper(request)
    events = []
    last_publish, last_data = 0.0, time.monotonic()
    journal = Path(request["journal"])
    pending = ""
    with journal.open(encoding="utf-8") as source:
        while True:
            line = source.readline(262145)
            if line:
                pending += line
                if len(pending) > 262144:
                    raise ValueError("MAVLink journal line exceeds limit")
                if not pending.endswith("\n"):
                    continue
                item = json.loads(pending)
                pending = ""
                added = mapper.accept(item)
                if added:
                    events.extend(added)
                    last_data = time.monotonic()
                if len(events) > 500000 or mapper.last_us > 7_199_000_000:
                    raise ValueError(
                        "실시간 AI 한도: 2시간 또는 500000개 변환 메시지. 원본 기록은 계속됩니다."
                    )
            if time.monotonic() - last_publish >= 1:
                state.update(
                    status="warming_up",
                    messages=mapper.messages,
                    reset_count=mapper.resets,
                    duplicate_count=mapper.duplicates,
                    armed=mapper.armed,
                    prediction=None,
                    decision=dict(
                        status="unavailable",
                        reason=calibration_error
                        or "유효한 미래 실측 도착 대기",
                    ),
                    missing_features=[],
                    updated_at=time.time(),
                )
                if (
                    mapper.heartbeat is None
                    or mapper.last_us - mapper.heartbeat > 3_000_000
                ):
                    state.update(
                        status="waiting", message="일치하는 HEARTBEAT 수신 대기"
                    )
                elif time.monotonic() - last_data > 1 or (
                    mapper.last_receipt is not None
                    and time.monotonic_ns() - mapper.last_receipt
                    > 1_000_000_000
                ):
                    state.update(
                        status="stale", message="수신 중단 · 예측 보류"
                    )
                elif mapper.last_us >= 11_000_000:
                    start = max(0, mapper.last_us - 14_000_000)
                    recent = [e for e in events if e[0] >= start]
                    values, times, report = causal_frames(
                        recent, start, mapper.last_us + 1
                    )
                    spans = report["candidate_segments"]
                    last = spans[-1] if spans else None
                    state["missing_features"] = [
                        name
                        for name, value in zip(FEATURES, values[-1])
                        if not np.isfinite(value)
                    ]
                    if (
                        last
                        and last["end_index"] == len(times)
                        and last["frames"] >= 100
                    ):
                        x = values[-100:].copy()
                        x[:, 17] /= request["profile"]["series_cells"]
                        x[:, 18] /= request["profile"]["capacity_ah"]
                        s = runtime.scaler
                        y = runtime.predict_scaled(
                            ((x - s["x_mean"]) / s["x_scale"]).astype("f4")[
                                None
                            ]
                        )[0]
                        pending_predictions, decision = arrived_decision(
                            pending_predictions,
                            values,
                            times,
                            last["start_us"],
                            mapper.resets,
                            request["profile"]["series_cells"],
                            s,
                            calibration,
                        )
                        if decision is not None:
                            state["decision"] = decision
                        pending_predictions.append(
                            dict(
                                origin=int(times[-1]),
                                due=int(times[-1]) + 1000000,
                                scaled=y[-1].tolist(),
                                reset=mapper.resets,
                            )
                        )
                        pending_predictions = pending_predictions[-4:]
                        y = y * s["y_scale"] + s["y_mean"]
                        y[:, 3] *= request["profile"]["series_cells"]
                        state.update(
                            status="predicting",
                            message="CPU 예측 중 · 실측 도착 후 보정 기준 판정",
                            missing_features=[],
                            prediction=dict(
                                origin_us=int(times[-1]),
                                horizon_us=1000000,
                                targets=[
                                    "body_rate_p",
                                    "body_rate_q",
                                    "body_rate_r",
                                    "pack_voltage",
                                ],
                                units=["rad/s", "rad/s", "rad/s", "V"],
                                values=y[-1].tolist(),
                            ),
                        )
                    else:
                        state["message"] = "완전한 19입력의 연속 10초 이력 대기"
                else:
                    state["message"] = "초기 입력 이력 수집 중"
                write(path, state)
                last_publish = time.monotonic()
            if not line:
                if (
                    request.get("auto_finish_disarm")
                    and mapper.disarm.ready
                    and not (folder / "auto-finish.json").exists()
                ):
                    write(
                        folder / "auto-finish.json",
                        dict(
                            reason="confirmed_disarm", receipt_us=mapper.last_us
                        ),
                    )
                if (folder / "finish").exists():
                    if pending:
                        raise ValueError(
                            "종료 저널의 마지막 행이 완전하지 않습니다."
                        )
                    state.update(
                        status="analyzing",
                        prediction=None,
                        decision=dict(
                            status="unavailable",
                            reason="실시간 판정 종료 · 비행 후 분석 중",
                        ),
                        message="기록 종료 · 구간별 비행 후 분석 중",
                    )
                    write(path, state)
                    result = completed(
                        events,
                        journal,
                        folder / "completed",
                        request,
                        admitted["origin"],
                        end_us=mapper.last_us + 100000
                        if mapper.base is not None
                        else 0,
                    )
                    write(folder / "result.json", result)
                    state.update(
                        status="finished",
                        message=result["message"],
                        result=result,
                        finish_reason="confirmed_disarm"
                        if (folder / "auto-finish.json").exists()
                        else "manual",
                        decision=dict(
                            status="unavailable",
                            reason="실시간 판정 종료 · 아래 비행 후 결과 확인",
                        ),
                    )
                    write(path, state)
                    return
                if (folder / "stop").exists():
                    state.update(
                        status="interrupted",
                        prediction=None,
                        message="앱 종료 · 기록 보존, 분석 중단",
                        decision=dict(status="unavailable", reason="분석 중단"),
                    )
                    write(path, state)
                    return
                time.sleep(0.05)


def main():
    request = json.loads(sys.stdin.readline(65537))
    try:
        run(request)
    except Exception as error:
        path = Path(request["output"]) / "state.json"
        state = read_json(path) if path.exists() else {}
        state.update(
            status="error",
            prediction=None,
            decision=dict(status="unavailable", reason=str(error)),
            message=str(error),
            updated_at=time.time(),
        )
        write(path, state)
        raise


if __name__ == "__main__":
    main()
