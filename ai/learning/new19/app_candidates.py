"""Bind selected V0.2 candidates to the aircraft and original model."""

from pathlib import Path

from localflight.profiles import ANALYSIS_FIELDS

from .data import digest, read

PROFILE_KEYS = (
    "id",
    "firmware",
    "battery_id",
    "battery_instance",
    "actuator_instance",
    *ANALYSIS_FIELDS,
)


def selected_paths(request, model_id, original, baseline):
    """Choose the aircraft-specific bundle and its own calibrated threshold."""
    directory = request.get("candidate_models", {}).get(model_id)
    if directory is None:
        return original, baseline
    folder = Path(directory)
    split = folder / "split.json"
    source = read(split)
    bundle = folder / "candidate"
    calibration = folder / "calibration.json"
    if any(
        source["profile"].get(key) != request["profile"].get(key)
        for key in PROFILE_KEYS
    ):
        raise ValueError(
            "후보를 학습한 기체의 입력 설정과 현재 프로파일이 다릅니다."
        )
    if (
        digest(original / "bundle.json") != source["original_bundle_sha256"]
        or digest(baseline) != source["original_calibration_sha256"]
        or read(calibration)["manifest_sha256"] != digest(split)
    ):
        raise ValueError(
            "후보의 원본 모델, 보정 또는 학습 이력이 달라졌습니다."
        )
    origin = request.get("candidate_origin")
    if source.get("input_origin", origin) != origin:
        raise ValueError("후보가 학습에 사용한 입력 형식과 기록이 다릅니다.")
    if request.get("scope") == "real_aircraft" and origin != "px4_real":
        raise ValueError(
            "실기체 UDP 시험에는 같은 PX4 기체의 후보를 선택하세요."
        )
    if (
        request.get("scope") == "simulation_test"
        and origin != "synthetic_unit_fixture"
    ):
        raise ValueError(
            "시뮬레이터 시험에는 시험용 데이터로 만든 후보를 선택하세요."
        )
    if (
        request.get("log_origin") is not None
        and origin != request["log_origin"]
    ):
        raise ValueError(
            "후보의 학습 로그와 분석할 로그의 입력 형식이 다릅니다."
        )
    return bundle, calibration
