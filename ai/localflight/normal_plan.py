"""Allocate confirmed-normal flights before fitting; preserve original files."""

import shutil
from pathlib import Path

import numpy as np

from .catalog import digest
from .prepared import distinct_flights, load
from .registry import write


def counts(number):
    """Engineering allocation, not a statistical sample-size guarantee."""
    if not 1 <= number <= 50:
        raise ValueError("Select 1..50 distinct confirmed-normal flights")
    if number < 5:
        return dict(
            adaptation=1,
            development=int(number >= 2),
            calibration=int(number >= 3),
            evaluation=int(number >= 4),
        )
    train = min(3, number - 4)
    dev = max(1, min(5, number // 6))
    evaluation = max(1, int(number * 0.3))
    calibration = number - train - dev - evaluation
    if calibration > 20:
        dev += calibration - 20
        calibration = 20
    return dict(
        adaptation=train,
        development=dev,
        calibration=calibration,
        evaluation=evaluation,
    )


def create(manifests, output):
    """Use source hashes for stable assignment, never prediction/evaluation scores."""
    flights = [(Path(p), load(p, normal=True)) for p in manifests]
    distinct_flights([f[1][0] for f in flights])
    allocation = counts(len(flights))
    flights.sort(key=lambda row: row[1][0]["source_sha256"])
    roles = [role for role, count in allocation.items() for _ in range(count)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for (source, (metadata, values, times)), role in zip(flights, roles):
        target = output / metadata["source_sha256"]
        target.mkdir()
        shutil.copyfile(
            source.parent / metadata["file"]["path"], target / "frames.npz"
        )
        if metadata.get("source_path"):
            shutil.copyfile(
                source.parent / metadata["source_path"],
                target / Path(metadata["source_path"]).name,
            )
        revised = dict(
            metadata,
            role=role,
            file=dict(path="frames.npz", sha256=digest(target / "frames.npz")),
            normal_plan=dict(
                source_manifest_sha256=digest(source),
                allocation="source_hash_order_before_model_scoring",
            ),
        )
        if metadata.get("source_path"):
            revised["source_path"] = Path(metadata["source_path"]).name
        write(target / "manifest.json", revised)
        load(target / "manifest.json", normal=True)
        rows.append(
            dict(
                source_sha256=metadata["source_sha256"],
                role=role,
                manifest=str(target / "manifest.json"),
                frames=len(times),
                retained_seconds=len(times) / 10,
                feature_quantiles=np.quantile(
                    values, [0.01, 0.5, 0.99], axis=0
                ).tolist(),
            )
        )
    missing = [role for role, count in allocation.items() if not count]
    if allocation["calibration"] < 2:
        missing.append("at_least_two_calibration_flights")
    result = dict(
        schema="aircraft-normal-plan/1",
        flights=rows,
        counts=allocation,
        missing=missing + ["separate_labelled_anomaly_evaluation"],
        recommended_normal_flights=[10, 20],
        independent_false_alarm_guarantee=False,
        note="1–3회는 입력 확인·초기 적응용. 5회부터 역할 분리 가능. 10–20회는 공학적 권장 시작점이며 검증 보장이 아닙니다.",
    )
    write(output / "plan.json", result)
    return result
