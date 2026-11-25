"""Score one complete PX4 MAVLink session with the four new whole-run models."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from localflight.live import write
from localflight.profiles import eligibility

from .app_candidates import selected_paths
from .app_live import FRAME_POLICY, growing_capture, processor
from .buffered import HOLD_US
from .calibration import load_calibration
from .data import Run, digest, read
from .inference import Runtime
from .models import VARIANTS
from .scoring import RunReference, run_components


def load_models(root, request=None):
    models = []
    for model_id, variant in VARIANTS.items():
        if variant.task != "whole-run":
            continue
        folder = Path(root) / model_id
        bundle, calibration = selected_paths(
            request or {},
            model_id,
            folder / "fit",
            folder / "calibration" / "calibration.json",
        )
        runtime = Runtime(bundle)
        parameters = load_calibration(bundle, calibration)
        scale = np.asarray(parameters["scale"], dtype=float)
        threshold = float(parameters["threshold"])
        if (
            runtime.model.variant.name != model_id
            or scale.shape != (2,)
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
            or not np.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError(
                f"Invalid whole-run model or calibration: {model_id}"
            )
        models.append(
            (
                dict(
                    model_id=model_id,
                    bundle_sha256=digest(bundle / "bundle.json"),
                    weights_sha256=digest(bundle / "model.safetensors"),
                    calibration_sha256=digest(calibration),
                    **(
                        {"candidate_job": request["candidate_job"]}
                        if request
                        and model_id in request.get("candidate_models", {})
                        else {}
                    ),
                ),
                runtime,
                RunReference(scale, threshold),
            )
        )
    return models


def binding(source, models):
    return dict(
        preprocessing_revision=source.revision,
        hold_us=HOLD_US,
        frame_policy=FRAME_POLICY,
        models=[identity for identity, _, _ in models],
    )


def validate(request):
    torch.set_num_threads(1)
    if (
        not eligibility(request["profile"])["profile_complete"]
        or request["scope"] not in {"simulation_test", "real_aircraft"}
        or request["imu_message"] != "HIGHRES_IMU"
    ):
        raise ValueError(
            "PX4 X500 프로파일과 HIGHRES_IMU 연결 설정을 확인하세요."
        )
    source = processor(request)
    return binding(source, load_models(request["whole_root"], request))


def analyze(request, update):
    """Wait for capture completion, then reuse the same buffered frame kernel."""
    torch.set_num_threads(1)
    source = processor(request)
    models = load_models(request["whole_root"], request)
    if request["binding"] != binding(source, models):
        raise ValueError(
            "세션에서 확정한 모델 또는 전처리 버전이 달라졌습니다."
        )
    wire, output = Path(request["wire"]), Path(request["output"])
    frames = []
    update(
        status="collecting",
        message="MAVLink 수신 중 · 비행 종료 후 전체 런 분석",
    )
    for stamp, peer, payload in growing_capture(wire, output / "finish"):
        frames.extend(source.accept(stamp, peer, payload))
    update(
        status="analyzing", message="수신 기록 확정 · 전체 런 모델 4종 분석 중"
    )
    if source.last_receipt_ns is not None:
        frames.extend(source.advance(source.last_receipt_ns + HOLD_US * 1000))
    valid = [i for i, frame in enumerate(frames) if frame.segment >= 0]
    selected = frames[valid[0] : valid[-1] + 1] if valid else []
    run = None
    if selected:
        run = Run(
            output.name,
            np.asarray([f.values for f in selected], dtype=np.float32),
            np.asarray([f.time_us for f in selected], dtype=np.int64),
            np.asarray([f.segment for f in selected], dtype=np.int64),
            np.asarray([f.observed for f in selected], dtype=bool),
        )
    result = score_run(
        output,
        run,
        models,
        source.late,
        digest(wire) if wire.is_file() else None,
        save_arrays=True,
    )
    update(
        status="finished",
        message="비행 종료 · 전체 런 결과 4종",
        whole_summary=result,
    )


def score_run(
    output,
    run,
    models,
    late_samples,
    wire_sha256,
    *,
    save_arrays=False,
    journal_sha256=None,
):
    """Apply the sealed whole-run models and preserve the input reference."""
    predictions, rows = {}, []
    for identity, runtime, reference in models:
        prediction = runtime.voltage(run) if run is not None else None
        record = None if prediction is None else run_components(run, prediction)
        if prediction is not None:
            predictions[identity["model_id"]] = prediction
        rows.append(
            dict(
                **identity,
                threshold=reference.threshold,
                **reference.decide(record),
            )
        )
    if save_arrays and run is not None:
        np.savez_compressed(
            output / "whole-run.npz",
            values=run.values,
            time_us=run.time_us,
            segment=run.segment,
            observed=run.observed,
        )
        if predictions:
            np.savez_compressed(
                output / "whole-predictions.npz",
                time_us=run.time_us,
                **predictions,
            )
    count = len(run.time_us) if run is not None else 0
    result = dict(
        frames=count,
        valid_frames=int(np.count_nonzero(run.segment >= 0)) if count else 0,
        duration_s=count / 10,
        late_samples=late_samples,
        wire_sha256=wire_sha256,
        models=rows,
    )
    if journal_sha256 is not None:
        result["journal_sha256"] = journal_sha256
    write(output / "whole-summary.json", result)
    return result


def score_buffered(folder, root, *, expected=None):
    """Score the trimmed observation journal without decoding UDP a second time."""
    folder = Path(folder)
    request = read(folder / "request.json")
    summary = read(folder / "buffered-summary.json")
    wire = Path(request["wire"])
    if not summary["wire_sha256"] or digest(wire) != summary["wire_sha256"]:
        raise ValueError("저장한 MAVLink UDP 원문을 확인하세요.")
    models = load_models(root, request)
    if expected is not None and expected != [
        identity for identity, _, _ in models
    ]:
        raise ValueError(
            "비행 종료 후 모델 또는 보정값이 기록 시작 이후 바뀌었습니다."
        )
    journal = folder / "buffered-results.jsonl"
    before = digest(journal)
    with journal.open(encoding="utf-8") as stream:
        frames = [
            (row["origin_us"], row["segment"], row["features"], row["observed"])
            for row in map(json.loads, stream)
        ]
    if len(frames) != summary["frames"] or digest(journal) != before:
        raise ValueError("확정 프레임 기록이 저장된 요약과 다릅니다.")
    valid = [i for i, row in enumerate(frames) if row[1] >= 0]
    run = None
    if valid:
        time_us, segment, values, observed = zip(
            *frames[valid[0] : valid[-1] + 1], strict=True
        )
        run = Run(
            folder.name,
            np.asarray(values, dtype=np.float32),
            np.asarray(time_us, dtype=np.int64),
            np.asarray(segment, dtype=np.int64),
            np.asarray(observed, dtype=bool),
        )
    return score_run(
        folder,
        run,
        models,
        summary["late_samples"],
        summary["wire_sha256"],
        journal_sha256=before,
    )


def main():
    request = json.load(sys.stdin)
    path = Path(request["output"]) / "state.json"
    state = dict(
        mode="px4_whole_run",
        scope=request["scope"],
        hold_ms=HOLD_US // 1000,
        whole_run_models=[m["model_id"] for m in request["binding"]["models"]],
    )
    if request.get("candidate_job"):
        state["candidate_job"] = request["candidate_job"]
        state["candidate_models"] = list(request["candidate_models"])

    def update(**changes):
        state.update(changes)
        write(path, state)

    try:
        analyze(request, update)
    except (OSError, ValueError, RuntimeError) as error:
        update(status="error", message=str(error))
        raise


if __name__ == "__main__":
    main()
