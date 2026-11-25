"""Read model charts from the existing buffered prediction journal."""

import json
from functools import lru_cache
from pathlib import Path
from struct import Struct

from .file_io import hash_file, read_json

# common19 body_rate_p/q/r and pack_voltage, in the model's target order.
TARGET_COLUMNS = (4, 5, 6, 17)
FLOAT32 = Struct("=f")
LIVE_FRAMES = 1800
LIVE_READ_BYTES = 6 << 20


@lru_cache(maxsize=8)
def _thresholds(path, sha):
    if hash_file(path) != sha:
        raise ValueError("저장한 보정 파일이 분석 시점과 다릅니다.")
    return read_json(path)["parameters"]["selected"]["threshold"]


def _frames(path, live, model):
    with path.open("rb") as stream:
        if live:
            window = LIVE_READ_BYTES * (3 if model else 1)
            start = max(0, stream.seek(0, 2) - window)
            stream.seek(start)
            if start:
                stream.readline()
            lines = stream.readlines()[-LIVE_FRAMES:]
        else:
            lines = stream
        for line in lines:
            # The worker may still be appending the last record.
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            valid = row["segment"] >= 0
            observed = [
                FLOAT32.unpack(FLOAT32.pack(row["features"][column]))[0]
                if valid
                and row["observed"][index]
                and row["features"][column] is not None
                else None
                for index, column in enumerate(TARGET_COLUMNS)
            ]
            result = row["models"][model] if model else row
            residual = result["residual"] or [None] * len(TARGET_COLUMNS)
            # The scoring input blends previously issued forecasts for this
            # target time. It is not the 1-second-ahead value from this row.
            predicted = [
                actual - error
                if actual is not None and error is not None
                else None
                for actual, error in zip(observed, residual, strict=True)
            ]
            decision = result["decision"]
            yield dict(
                time_us=row["origin_us"],
                segment=row["segment"],
                observed=observed,
                predicted=predicted,
                residual=residual,
                raw_score=decision["scores"] or [None, None],
                score=decision.get("ema", [None, None]),
                alarm=decision.get("domains", [False, False]),
            )


@lru_cache(maxsize=8)
def _saved_frames(path, modified_ns, size, model):
    del modified_ns, size
    return list(_frames(path, False, model))


def chart(primary, model=None, additional=None, *, wire_sha=None):
    """Return charts for the session worker and explicitly registered workers."""
    primary_request = read_json(primary / "request.json")
    if primary_request["mode"] != "px4_buffered":
        raise ValueError("PX4 10초 버퍼 분석 기록을 선택하세요.")
    binding = primary_request["binding"]
    if "models" in binding:
        models = list(binding["models"])
        selected = model or models[0]
        if selected not in models:
            raise ValueError("이 비행에 기록된 모델을 선택하세요.")
        candidate = primary_request.get("candidate_models", {}).get(selected)
        calibration = (
            Path(candidate) / "calibration.json"
            if candidate
            else Path(primary_request["inflight_root"])
            / selected
            / "calibration/calibration.json"
        )
        return _chart_data(
            primary,
            selected,
            models,
            calibration,
            binding["models"][selected],
            binding["hold_us"],
            selector=selected,
            candidate_models=primary_request.get("candidate_models", {}),
        )
    primary_model = binding["model"]
    workers = (additional or {}) | {primary_model: primary}
    selected = model or primary_model
    if selected not in workers:
        raise ValueError("이 비행에 기록된 모델을 선택하세요.")
    folder = workers[selected]
    request = (
        primary_request
        if folder == primary
        else read_json(folder / "request.json")
    )
    binding = request["binding"]
    if (
        binding["model"] != selected
        or request["wire"] != primary_request["wire"]
    ):
        raise ValueError("모델의 수신 기록이 이 비행과 일치하지 않습니다.")
    if folder != primary and (
        not wire_sha
        or read_json(folder / "buffered-summary.json")["wire_sha256"]
        != wire_sha
    ):
        raise ValueError("모델 결과의 UDP 원문 해시가 이 비행과 다릅니다.")
    return _chart_data(
        folder,
        selected,
        [primary_model, *sorted(set(workers) - {primary_model})],
        Path(request["new19_calibration"]),
        binding,
        binding["hold_us"],
    )


def _chart_data(
    folder,
    selected,
    models,
    calibration,
    binding,
    hold_us,
    *,
    selector=None,
    candidate_models=(),
):
    thresholds = _thresholds(calibration, binding["calibration_sha256"])
    state = read_json(folder / "state.json")
    live = state["status"] not in {"finished", "error", "interrupted"}
    journal = folder / "buffered-results.jsonl"
    if not journal.is_file():
        frames = []
    elif live:
        frames = list(_frames(journal, True, selector))
    else:
        stat = journal.stat()
        frames = _saved_frames(
            journal, stat.st_mtime_ns, stat.st_size, selector
        )
    return dict(
        model=selected,
        models=models,
        candidate_models=list(candidate_models),
        live=live,
        hold_ms=hold_us // 1000,
        thresholds=thresholds,
        frames=frames,
    )
