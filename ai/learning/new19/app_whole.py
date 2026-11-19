"""Score one complete PX4 MAVLink session with the four new whole-run models."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from localflight.live import write
from localflight.profiles import eligibility

from .app_live import FRAME_POLICY, growing_capture, processor
from .buffered import HOLD_US
from .calibration import load_calibration
from .data import Run, digest
from .inference import Runtime
from .models import VARIANTS
from .scoring import RunReference, run_components


def load_models(root):
    models = []
    for model_id, variant in VARIANTS.items():
        if variant.task != "whole-run":
            continue
        folder = Path(root) / model_id
        bundle = folder / "fit"
        calibration = folder / "calibration" / "calibration.json"
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
    return binding(source, load_models(request["whole_root"]))


def analyze(request, update):
    """Wait for capture completion, then reuse the same buffered frame kernel."""
    torch.set_num_threads(1)
    source = processor(request)
    models = load_models(request["whole_root"])
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
    if run is not None:
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
    result = dict(
        frames=len(selected),
        valid_frames=len(valid),
        duration_s=(len(selected) / 10),
        late_samples=source.late,
        wire_sha256=digest(wire) if wire.is_file() else None,
        models=rows,
    )
    write(output / "whole-summary.json", result)
    update(
        status="finished",
        message="비행 종료 · 전체 런 결과 4종",
        whole_summary=result,
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
