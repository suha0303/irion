"""Run the app's delayed PX4 prediction and replay the same saved datagrams."""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from capstone_ai.mapping import TARGETS
from localflight.live import write
from localflight.profiles import eligibility

from .app_candidates import selected_paths
from .buffered import HOLD_US, REVISION, BufferedPreprocessor
from .calibration import load_calibration
from .data import digest, read
from .inference import FlightSession, Runtime
from .models import VARIANTS
from .scoring import FlightAlarm, FlightReference
from .transport import HEADER, MAGIC, datagrams

UNITS = ["rad/s", "rad/s", "rad/s", "V"]
FRAME_POLICY = "valid-flight-edges/1"


def model_paths(request):
    if "inflight_root" not in request:
        return [
            (
                None,
                Path(request["new19_bundle"]),
                Path(request["new19_calibration"]),
            )
        ]
    root = Path(request["inflight_root"])
    return [
        (
            name,
            *selected_paths(
                request,
                name,
                root / name / "fit",
                root / name / "calibration/calibration.json",
            ),
        )
        for name, variant in VARIANTS.items()
        if variant.task == "inflight"
    ]


def validate(request):
    profile = request["profile"]
    if not eligibility(profile)["profile_complete"] or any(
        profile.get(key) is None
        for key in ("battery_instance", "actuator_instance")
    ):
        raise ValueError("PX4 기체 프로파일의 모터·배터리 매핑을 확인하세요.")
    if (
        request.get("scope") not in {"simulation_test", "real_aircraft"}
        or request.get("imu_message") != "HIGHRES_IMU"
    ):
        raise ValueError("연결 대상과 HIGHRES_IMU를 지정해 주세요.")
    preprocessor = processor(request)
    binding = dict(
        preprocessing_revision=preprocessor.revision,
        hold_us=HOLD_US,
    )
    models = {}
    for expected, bundle, calibration in model_paths(request):
        variant = Runtime(bundle).model.variant
        if variant.task != "inflight" or (
            expected and variant.name != expected
        ):
            raise ValueError("비행 중 예측용 new19 번들을 확인해 주세요.")
        load_calibration(bundle, calibration)
        models[variant.name] = dict(
            model=variant.name,
            bundle_sha256=digest(bundle / "bundle.json"),
            weights_sha256=digest(bundle / "model.safetensors"),
            calibration_sha256=digest(calibration),
        )
    if "inflight_root" in request:
        binding["models"] = models
    else:
        binding.update(next(iter(models.values())))
    # Replays without this binding keep their original journal format.
    if "binding" not in request or "frame_policy" in request["binding"]:
        binding["frame_policy"] = FRAME_POLICY
    if request.get("whole_run_root"):
        from .app_whole import load_models

        binding["whole_models"] = [
            identity
            for identity, _, _ in load_models(
                request["whole_run_root"], request
            )
        ]
    return binding


def processor(request):
    return BufferedPreprocessor(
        request["profile"],
        request["source_system"],
        request["source_component"],
        request["imu_id"],
        revision=request.get("binding", {}).get(
            "preprocessing_revision", REVISION
        ),
    )


class Detector:
    def __init__(self, bundle, calibration, clock_us):
        cal = load_calibration(bundle, calibration)
        self.session = FlightSession(
            Runtime(bundle), clock_us=clock_us, hold_us=HOLD_US
        )
        self.reference = FlightReference(
            cal["center"], cal["precision"], cal["voltage_scale"]
        )
        self.alarm = FlightAlarm(cal["selected"]["threshold"])

    def commit(self, frame):
        if frame.segment != self.session.segment:
            self.alarm.reset()
        forecast, residual, status = self.session.step(
            frame.time_us,
            frame.segment,
            frame.values,
            frame.observed,
            last_input_available_us=frame.time_us + HOLD_US,
        )
        score = None if residual is None else self.reference.score(residual)
        judgment = self.alarm.update(frame.time_us, score)
        judgment["scores"] = None if score is None else score.tolist()
        result = dict(
            publication=status,
            forecast=forecast.values if forecast else None,
            residual=None if residual is None else residual.tolist(),
            decision=judgment,
        )
        return result, None if forecast is None else forecast.issued_us


def growing_capture(path, finish):
    """Tail the receiver-owned binary log using the original arrival stamps."""
    while not path.is_file() or path.stat().st_size < len(MAGIC):
        if finish.exists():
            return
        time.sleep(0.01)
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError("Expected original MLUDP001 capture")
        while True:
            offset = stream.tell()
            header = stream.read(HEADER.size)
            if len(header) == HEADER.size:
                stamp, _, address, port, size = HEADER.unpack(header)
                payload = stream.read(size)
                if len(payload) == size:
                    yield stamp, (address, port), payload
                    continue
            if finish.exists():
                if header:
                    raise ValueError("Truncated final UDP record")
                return
            stream.seek(offset)
            time.sleep(0.01)


def analyze(request, output, *, replay=False):
    torch.set_num_threads(1)
    output = Path(output)
    path = Path(request["wire"])
    state_path = output / "state.json"
    binding = validate(request)
    if binding != request["binding"]:
        raise ValueError(
            "학습 모델·보정값·전처리 버전이 기록의 설정과 다릅니다."
        )
    parallel = "models" in binding
    source = processor(request)
    frame_time = write_clock = frames = 0
    replay_clock = replay
    trim_edges = binding.get("frame_policy") == FRAME_POLICY
    last_valid_end = last_valid_count = 0

    def clock():
        if replay_clock:
            return frame_time + HOLD_US
        return (time.monotonic_ns() - source.origin_ns) // 1000

    detectors = {
        model or binding["model"]: Detector(bundle, calibration, clock)
        for model, bundle, calibration in model_paths(request)
    }
    totals = {
        model: dict(forecasts=0, scored_frames=0, alarm_frames=0)
        for model in detectors
    }
    state = dict(
        status="warming_up",
        mode="px4_buffered",
        scope=request["scope"],
        hold_ms=HOLD_US // 1000,
        message="PX4 MAVLink 수신 및 10초 버퍼 대기 중",
    )
    if request.get("candidate_job"):
        state["candidate_job"] = request["candidate_job"]
        state["candidate_models"] = list(request["candidate_models"])
    if parallel:
        state["forecast_models"] = list(detectors)
    else:
        state["forecast_model"] = binding["model"]
    if binding.get("whole_models"):
        state["whole_run_models"] = [
            identity["model_id"] for identity in binding["whole_models"]
        ]
    if not replay:
        write(state_path, state)

    incoming = (
        datagrams(path) if replay else growing_capture(path, output / "finish")
    )
    with ExitStack() as stack:
        journal = stack.enter_context(
            (output / "buffered-results.jsonl").open("xb")
        )
        pool = (
            stack.enter_context(ThreadPoolExecutor(max_workers=len(detectors)))
            if parallel
            else None
        )

        def record(ready):
            nonlocal frame_time, write_clock, frames
            nonlocal last_valid_end, last_valid_count
            for frame in ready:
                frame_time = frame.time_us
                computed = (
                    pool.map(
                        lambda detector: detector.commit(frame),
                        detectors.values(),
                    )
                    if pool
                    else (
                        detector.commit(frame)
                        for detector in detectors.values()
                    )
                )
                row = dict(
                    origin_us=frame_time,
                    segment=frame.segment,
                    observed=frame.observed,
                    features=[
                        x if math.isfinite(x) else None for x in frame.values
                    ],
                )
                results = {}
                for model, (result, issued) in zip(
                    detectors, computed, strict=True
                ):
                    stats = totals[model]
                    stats["forecasts"] += result["forecast"] is not None
                    alarm = result["decision"]["alarm"]
                    stats["scored_frames"] += alarm is not None
                    stats["alarm_frames"] += alarm is True
                    if not replay:
                        result["issued_us"] = (
                            issued if not replay_clock else None
                        )
                    results[model] = result
                if parallel:
                    row["models"] = results
                else:
                    row.update(results[binding["model"]])
                valid = frame.segment >= 0
                if not trim_edges or valid or frames:
                    line = json.dumps(
                        row, allow_nan=False, separators=(",", ":")
                    )
                    journal.write((line + "\n").encode("utf-8"))
                    frames += 1
                    if trim_edges and valid:
                        last_valid_end = journal.tell()
                        last_valid_count = frames
                if not replay_clock and time.monotonic() - write_clock >= 0.25:
                    journal.flush()
                    update_state(state, results, totals, frame_time, parallel)
                    write(state_path, state)
                    write_clock = time.monotonic()

        for stamp, peer, payload in incoming:
            record(source.accept(stamp, peer, payload))
        if source.last_receipt_ns is not None:
            replay_clock = True
            record(source.advance(source.last_receipt_ns + HOLD_US * 1000))
        if trim_edges:
            # Keep an invalid stretch if valid flight data resumed after it.
            journal.truncate(last_valid_end)
            frames = last_valid_count

    counts = (
        {"model_summaries": totals} if parallel else totals[binding["model"]]
    )
    summary = dict(
        **binding,
        **counts,
        wire_sha256=digest(path) if path.is_file() else None,
        frames=frames,
        late_samples=source.late,
    )
    write(output / "buffered-summary.json", summary)
    if not replay:
        message = (
            f"MAVLink 기록 종료 · 비행 중 모델 {len(totals)}종"
            if parallel
            else f"MAVLink 기록 종료 · 판정 {counts['scored_frames']}프레임 · 이상 표시 {counts['alarm_frames']}프레임"
        )
        state.update(
            prediction=None,
            decision=None,
            buffered_summary=summary,
            message=message,
        )
        if binding.get("whole_models"):
            from .app_whole import score_buffered

            state.update(status="analyzing", whole_status="running")
            write(state_path, state)
            try:
                state["whole_summary"] = score_buffered(
                    output,
                    request["whole_run_root"],
                    expected=binding["whole_models"],
                )
                state["whole_status"] = "complete"
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                state.update(whole_status="failed", whole_error=str(error))
        state["status"] = "finished"
        write(state_path, state)
    return summary


def replay_candidate(request):
    """Record a separate candidate replay using the sealed UDP input."""
    output = Path(request["output"])
    if digest(request["wire"]) != request["wire_sha256"]:
        raise ValueError("저장된 MAVLink 원문 해시가 일치하지 않습니다.")
    request["binding"] = validate(request)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "request.json", request)
    summary = analyze(request, output, replay=True)
    if summary["wire_sha256"] != request["wire_sha256"]:
        raise ValueError("분석 중 MAVLink 원문이 바뀌었습니다.")
    whole = None
    if request["binding"].get("whole_models"):
        from .app_whole import score_buffered

        whole = score_buffered(
            output,
            request["whole_run_root"],
            expected=request["binding"]["whole_models"],
        )
    write(output / "state.json", dict(status="finished"))
    return dict(
        candidate_job=request["candidate_job"],
        candidate_models=list(request["candidate_models"]),
        buffered_summary=summary,
        whole_summary=whole,
    )


def update_state(state, results, totals, frame_time, parallel):
    active = sum(result["issued_us"] is not None for result in results.values())
    state.update(
        status="predicting" if active else "warming_up",
        updated_at=time.time(),
        message="10초 지연 예측 · 실측 비교"
        if active
        else "연속된 19입력의 최근 10초 이력 대기",
    )
    if parallel:
        state["buffered_model_states"] = {
            model: dict(
                **totals[model],
                publishing=result["issued_us"] is not None,
                alarm=result["decision"]["alarm"],
            )
            for model, result in results.items()
        }
        return
    result = next(iter(results.values()))
    alarm = result["decision"]["alarm"]
    state.update(
        prediction=(
            dict(
                origin_us=frame_time,
                values=result["forecast"][-1],
                units=UNITS,
                targets=TARGETS,
            )
            if active
            else None
        ),
        decision=dict(
            status="suspect"
            if alarm
            else "normal"
            if alarm is False
            else "unavailable",
            time_us=frame_time,
            reason="PX4 10초 버퍼 · 보정 기준",
        ),
        scored_frames=totals[state["forecast_model"]]["scored_frames"],
    )


def compare_replay(folder):
    folder = Path(folder)
    request = read(folder / "request.json")
    live = read(folder / "buffered-summary.json")
    if live["wire_sha256"] != digest(request["wire"]):
        raise ValueError("저장한 MAVLink UDP 파일이 바뀌었습니다.")
    mismatches, frame_count, live_predictions = 0, 0, 0
    max_delay_us = None
    with TemporaryDirectory(prefix="replay-", dir=folder) as replay_dir:
        replay = analyze(request, replay_dir, replay=True)
        with (
            (folder / "buffered-results.jsonl").open(
                encoding="utf-8"
            ) as original,
            (Path(replay_dir) / "buffered-results.jsonl").open(
                encoding="utf-8"
            ) as regenerated,
        ):
            for live_line, replay_line in zip_longest(original, regenerated):
                frame_count += 1
                if live_line is None or replay_line is None:
                    mismatches += 1
                    continue
                row = json.loads(live_line)
                for result in row.get("models", {"single": row}).values():
                    issued = result.pop("issued_us")
                    if issued is not None:
                        live_predictions += 1
                        delay = issued - row["origin_us"]
                        max_delay_us = (
                            delay
                            if max_delay_us is None
                            else max(max_delay_us, delay)
                        )
                mismatches += row != json.loads(replay_line)
    result = dict(
        equal=frame_count > 0 and mismatches == 0 and live == replay,
        frames=frame_count,
        differing_frames=mismatches,
        live_predictions=live_predictions,
        max_live_delay_us=max_delay_us,
        wire_sha256=live["wire_sha256"],
    )
    keys = (
        ("models", "model_summaries")
        if "models" in live
        else (
            "forecasts",
            "scored_frames",
            "alarm_frames",
            "weights_sha256",
            "calibration_sha256",
        )
    )
    result.update({key: live[key] for key in keys})
    write(folder / "replay-comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, metavar="REQUEST")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.replay:
        if args.output is None:
            parser.error("--replay requires --output")
        request = read(args.replay)
        args.output.mkdir(parents=True, exist_ok=False)
        print(json.dumps(analyze(request, args.output, replay=True)))
    else:
        request = json.load(sys.stdin)
        output = Path(request["output"])
        try:
            analyze(request, output)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            write(
                output / "state.json",
                dict(status="error", prediction=None, message=str(error)),
            )
            raise


if __name__ == "__main__":
    main()
