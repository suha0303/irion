"""Version-bound flight boundaries and operational events, never health labels."""

import math
import re
from collections import Counter

REFERENCE = (
    "https://github.com/ArduPilot/ardupilot/blob/Copter-4.4.0/libraries/AP_Logger/AP_Logger.h"
)
LABELS = {
    2: "조종기 수신",
    5: "조종기 페일세이프",
    6: "배터리 페일세이프",
    8: "지상국 페일세이프",
    12: "충돌 검사",
    16: "EKF 검사",
    24: "EKF 주 센서 변경",
}


def summarize(audit):
    rows = audit.get("operational_events", [])
    firmware = [m for m in audit.get("messages", []) if m.startswith("ArduCopter V")]
    reviewed = bool(firmware) and all(re.match(r"ArduCopter V4\.4\.0(?:\s|$)", m) for m in firmware)
    errors = [
        r
        for r in rows
        if r["kind"] == "ERR" and type(r.get("Subsys")) is int and type(r.get("ECode")) is int
    ]
    counts = Counter((r.get("Subsys"), r.get("ECode")) for r in errors)
    result = dict(
        reference=REFERENCE,
        supported=reviewed,
        health="unknown",
        intervals=[],
        error_counts=[
            dict(
                subsystem=k[0],
                code=k[1],
                count=n,
                label=LABELS.get(k[0], "기타 기록") if reviewed else "의미 미확인",
            )
            for k, n in sorted(counts.items())
        ],
        events=rows,
        notes=[],
    )
    if not reviewed:
        result["notes"].append(
            "검토한 펌웨어 버전이 아니므로 이벤트 기반 구간을 자동 지정하지 않습니다."
        )
        return result
    if audit.get("operational_events_truncated") or any(
        s["reversals"] for s in audit.get("timing", {}).get("streams", {}).values()
    ):
        result["notes"].append("기록 시각 역행 또는 이벤트 한도 초과: 자동 구간 지정 보류")
        return result
    flights, start = [], None
    for row in rows:
        if row["kind"] != "EV":
            continue
        if row.get("Id") == 28:
            if start is not None:
                result["notes"].append("중복 이륙 이벤트: 불명확한 앞 구간 제외")
            start = row["TimeUS"]
        elif row.get("Id") == 18 and start is not None:
            if row["TimeUS"] > start:
                flights.append((start, row["TimeUS"]))
            start = None
    if start is not None:
        result["notes"].append("착륙 이벤트가 없는 마지막 비행은 수동 구간 검사가 필요합니다.")
    histories = audit.get("parameter_values", {})
    masks, ekf = histories.get("EK3_IMU_MASK", []), histories.get("AHRS_EKF_TYPE", [])
    mask = masks[0] if len(masks) == 1 else None
    core_imus = (
        [i for i in range(16) if int(mask) & (1 << i)]
        if type(mask) in (int, float)
        and math.isfinite(mask)
        and mask == int(mask)
        and 0 < mask < 65536
        and ekf == [3]
        else []
    )
    primaries = [r for r in rows if r["kind"] == "XKF4"]
    observed = audit.get("instances", {}).get("IMU", {})
    for flight, (a, b) in enumerate(flights, 1):
        cuts = sorted({a, b, *[r["TimeUS"] for r in primaries if a < r["TimeUS"] < b]})
        for left, right in zip(cuts, cuts[1:]):
            available = [r["PI"] for r in primaries if r["TimeUS"] <= left]
            core = available[-1] if available else None
            imu = core_imus[core] if type(core) is int and 0 <= core < len(core_imus) else None
            if not observed.get(str(imu)):
                imu = None
            # Stay within the existing per-package 1000 s CPU admission limit.
            for chunk in range(left, right, 1_000_000_000):
                end = min(right, chunk + 1_000_000_000)
                result["intervals"].append(
                    dict(
                        index=len(result["intervals"]),
                        flight=flight,
                        start_us=chunk,
                        end_us=end,
                        imu_instance=imu,
                        usable=end - chunk >= 11_000_000,
                        evidence=f"EV.NOT_LANDED={a} → EV.LAND_COMPLETE={b}; XKF4.PI={core}; 구간별 IMU={imu}",
                    )
                )
    result["notes"].append(
        "통신·제어기 이벤트와 AI 건강 판정은 별개입니다. 정상 비행으로 자동 지정하지 않습니다."
    )
    return result
