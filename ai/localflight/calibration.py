"""Aircraft/model-specific empirical residual thresholds; no detection
guarantees.

Normal calibration flights are disjoint from fitting/development/evaluation. A
window count is never presented as a count of independent flights.
"""

import math
import re
import shutil
import time
from pathlib import Path

import numpy as np

from .catalog import digest, read_json
from .prepared import REVISION, distinct_flights, load
from .registry import fingerprint, write

POLICY = "real-residual-threshold/1"


def binding(runtime, metadata, adaptation=None):
    timing = metadata.get("timing", {})
    return dict(
        model_id=runtime.entry["id"],
        profile_version=fingerprint(metadata["profile"]),
        origin=metadata["origin"],
        weight_sha256=adaptation["model_sha256"]
        if adaptation
        else runtime.entry["weight_sha256"],
        scaler_sha256=digest(runtime.entry["files"]["scaler.npz"]),
        preprocessing=REVISION,
        live_input=metadata.get("live_input"),
        policy=POLICY,
        clock="receipt_monotonic"
        if timing.get("clock") == "receipt_monotonic"
        else "onboard",
        fast_max_age_ms=timing.get("fast_max_age_ms", 200),
        battery_max_age_ms=timing.get("battery_max_age_ms", 350),
    )


def forecast_scores(observed, predicted, scaler):
    a = np.asarray(observed, dtype="f8")
    b = np.asarray(predicted, dtype="f8")
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("One-second normalized-target pairs required")
    # Inputs are already normalized with the immutable y scaler.
    scores = np.max(np.abs(a - b), axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite residual score")
    return scores


def health_score(components, voltage_scale):
    if components.get("status") != "available":
        raise ValueError("No valid post-initial voltage observations")
    score = (
        math.hypot(components["short_cell_v"], components["long_cell_v"])
        / voltage_scale
    )
    if not math.isfinite(score) or score < 0:
        raise ValueError("Nonfinite run residual score")
    return score


def _path(root, calibration_id):
    if not re.fullmatch(r"[a-f0-9]{64}", calibration_id):
        raise ValueError("Invalid calibration ID")
    return Path(root) / "calibrations" / (calibration_id + ".json")


def read(root, calibration_id):
    path = _path(root, calibration_id)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink calibration not admitted")
    record = read_json(path)
    if fingerprint(record) != calibration_id or record.get("schema") != POLICY:
        raise ValueError("Calibration integrity mismatch")
    threshold = record.get("threshold")
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("Invalid calibration threshold")
    return record


def list_calibrations(root):
    index = Path(root) / "calibration-selection.json"
    selected = read_json(index) if index.exists() else {}
    rows = []
    for path in sorted((Path(root) / "calibrations").glob("*.json")):
        record = read(root, path.stem)
        rows.append(
            dict(
                record,
                id=path.stem,
                active=selected.get(fingerprint(record["binding"]))
                == path.stem,
            )
        )
    return dict(calibrations=rows)


def select(root, calibration_id, enabled=True):
    record = read(root, calibration_id)
    path = Path(root) / "calibration-selection.json"
    index = read_json(path) if path.exists() else {}
    key = fingerprint(record["binding"])
    if enabled:
        index[key] = calibration_id
    else:
        if index.get(key) != calibration_id:
            raise ValueError(
                "Only the currently selected calibration can be disabled"
            )
        index.pop(key)
    write(path, index)
    return dict(id=calibration_id, enabled=enabled)


def active(root, expected):
    path = Path(root) / "calibration-selection.json"
    if not path.exists():
        return None
    calibration_id = read_json(path).get(fingerprint(expected))
    if not calibration_id:
        return None
    record = read(root, calibration_id)
    if record["binding"] != expected:
        raise ValueError("Calibration/model/profile/input policy mismatch")
    return dict(record, id=calibration_id)


def fit(root, model_id, manifests, *, bundle, adapted=None, quantile=0.99):
    from .analysis import analyze

    if type(quantile) not in (int, float) or not 0.9 <= quantile <= 0.999:
        raise ValueError("Empirical quantile must be 0.9..0.999")
    if not 2 <= len(manifests) <= 20:
        raise ValueError(
            "보정 전용 정상 비행 2~20개가 필요합니다. 실기체 오경보율 보장은 아닙니다."
        )
    flights = [load(p, normal=True)[0] for p in manifests]
    distinct_flights(flights)
    if any(m["role"] != "calibration" for m in flights):
        raise ValueError(
            "학습·모델 선택·평가와 분리한 calibration 비행만 사용하세요."
        )
    training_sources = (
        read_json(Path(adapted) / "result.json")["binding"]["source_sha256"]
        if adapted
        else []
    )
    if set(training_sources) & {m["source_sha256"] for m in flights}:
        raise ValueError(
            "학습/모델 선택에 사용한 원본 비행은 보정에 재사용할 수 없습니다."
        )
    results = [
        analyze(model_id, p, bundle=bundle, adapted=adapted) for p in manifests
    ]
    bindings = [r["calibration_binding"] for r in results]
    if any(b != bindings[0] for b in bindings):
        raise ValueError(
            (
                "Calibration flights require identical clock and preprocessing "
                "policy"
            )
        )
    arrays = [r["scoring"]["scores"] for r in results]
    minimum = 20 if model_id.startswith("forecast_") else 1
    if any(len(a) < minimum for a in arrays):
        raise ValueError(
            (
                f"Each calibration flight needs at least {minimum} valid "
                f"scoring observations"
            )
        )
    per_flight = [
        float(np.quantile(a, quantile, method="higher")) for a in arrays
    ]
    threshold = float(np.quantile(per_flight, quantile, method="higher"))
    record = dict(
        schema=POLICY,
        binding=bindings[0],
        model_id=model_id,
        profile_id=flights[0]["profile"]["id"],
        created_at=time.time(),
        method=(
            "per-flight empirical quantile, then across-flight empirical "
            "quantile"
        ),
        quantile=quantile,
        threshold=threshold,
        source_sha256=[m["source_sha256"] for m in flights],
        training_source_sha256=training_sources,
        manifests_sha256=[digest(p) for p in manifests],
        observations_per_flight=[len(a) for a in arrays],
        per_flight_quantiles=per_flight,
        unit=results[0]["scoring"]["unit"],
        normal_flights=len(flights),
        fitted_exceedance_per_flight=[
            float(np.mean(np.asarray(a) > threshold)) for a in arrays
        ],
        validation_status="experimental_normal_only",
        false_alarm_guarantee=False,
        real_aircraft_validated=False,
        note="정상 보정자료의 경험적 기준. 독립 정상/이상 비행 성능 검증 필요.",
    )
    calibration_id = fingerprint(record)
    path = _path(root, calibration_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        import json

        json.dump(record, f, ensure_ascii=False, allow_nan=False, indent=2)
    return dict(record, id=calibration_id, active=False)


def decide(record, scores, *, source_sha256=None):
    a = np.asarray(scores, dtype="f8")
    if record is None:
        return dict(
            status="unavailable",
            reason="일치하는 기체·모델·입력 방식의 보정값 없음",
        )
    if source_sha256 in record["source_sha256"] + record.get(
        "training_source_sha256", []
    ):
        return dict(
            status="unavailable",
            reason="학습·보정에 사용한 비행: 독립 판정에서 제외",
            calibration_id=record["id"],
        )
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any(a < 0):
        return dict(status="unavailable", reason="유효한 예측/실측 쌍 없음")
    alarms = a > record["threshold"]
    return dict(
        status="suspect" if bool(alarms.any()) else "normal",
        threshold=record["threshold"],
        scores=a.tolist(),
        alarms=alarms.tolist(),
        exceedances=int(alarms.sum()),
        observations=len(a),
        calibration_id=record["id"],
        score_unit=record["unit"],
        scope="retained_scored_observations_only",
        validation_status=record["validation_status"],
        real_aircraft_validated=False,
        false_alarm_guarantee=False,
    )


def annotate(result, metadata, root):
    try:
        record = active(root, result["calibration_binding"])
        decision = decide(
            record,
            result["scoring"]["scores"],
            source_sha256=metadata["source_sha256"],
        )
        for row in result.get("segment_results", []):
            row["decision"] = decide(
                record, row["scores"], source_sha256=metadata["source_sha256"]
            )
    except (ValueError, OSError, KeyError) as error:
        decision = dict(
            status="unavailable", reason="보정값 검사 실패: " + str(error)
        )
        for row in result.get("segment_results", []):
            row["decision"] = decision
    result["decision"] = decision
    result["health_verdict"] = (
        decision["status"] if decision["status"] != "unavailable" else None
    )
    result["calibration_status"] = (
        "empirical_calibration_applied"
        if result["health_verdict"]
        else "unavailable"
    )
    if (
        result["health_verdict"]
        and metadata["role"] == "evaluation"
        and metadata["health"] == "confirmed_normal"
    ):
        result["normal_alarm_rate"] = (
            decision["exceedances"] / decision["observations"]
        )
        result["alarm_rate_status"] = (
            "independent_normal_observation_exceedance_rate_not_flight_FPR"
        )
    return result


def admit_completed(manifest, output, *, role, health, evidence):
    """New declared role/version; original recorded flight and hash stay
    immutable.
    """

    metadata, values, _ = load(manifest)
    if metadata.get("timing", {}).get("clock") != "receipt_monotonic":
        raise ValueError("Only completed live packages use this admission path")
    if role not in {"adaptation", "development", "evaluation", "calibration"}:
        raise ValueError("Invalid declared flight role")
    if health not in {"unknown", "confirmed_normal", "abnormal"}:
        raise ValueError("Invalid declared flight health")
    if not isinstance(evidence, str) or not 1 <= len(evidence.strip()) <= 2000:
        raise ValueError("비행 구간·건강 상태 확인 근거를 입력하세요.")
    if role == "calibration" and health != "confirmed_normal":
        raise ValueError("보정에는 정상 확인 비행이 필요합니다.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(manifest).parent
    for name in ("source.jsonl", "frames.npz"):
        shutil.copyfile(source / name, output / name)
    metadata.update(
        role=role,
        health=health,
        health_evidence=evidence,
        admission=dict(
            source_manifest_sha256=digest(manifest), evidence=evidence
        ),
    )
    write(output / "manifest.json", metadata)
    load(output / "manifest.json", normal=role == "calibration")
    return dict(
        manifest=str(output / "manifest.json"),
        role=role,
        health=health,
        frames=len(values),
        source_sha256=metadata["source_sha256"],
    )
