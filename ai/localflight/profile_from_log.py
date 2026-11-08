"""Logged profile extraction with explicitly labelled, editable cell-count
estimates.
"""

import math
import re

from .profiles import eligibility

REVISION = "logged-profile-v2"
ARDU_MATRIX = (
    "https://github.com/ArduPilot/ardupilot/blob/Copter-4.4.0/"
    "libraries/AP_Motors/AP_MotorsMatrix.cpp"
)
ARDU_PARAMS = (
    "https://ardupilot.org/copter/docs/parameters-Copter-stable-V4.4.0.html"
)
PX4_PARAMS = "https://docs.px4.io/main/en/advanced_config/parameter_reference"
LABELS = {
    "autopilot": "비행제어기 종류",
    "rotor_count": "로터 수",
    "series_cells": "배터리 직렬 셀 수",
    "capacity_ah": "배터리 정격 용량 (Ah)",
    "motor_channels": "모터 위치별 출력 채널",
    "output_min": "모터 출력 최솟값",
    "output_max": "모터 출력 최댓값",
    "imu_instance": "분석할 IMU",
    "battery_instance": "분석할 배터리",
    "actuator_instance": "출력 인스턴스",
}


def suggest(audit):
    profile = dict(
        schema="drone.real-aircraft/1",
        id="log-" + audit["sha256"][:12],
        name=audit["filename"],
        autopilot=audit["autopilot"],
        motor_channels={},
    )
    evidence, reasons, candidates = {}, {}, {}
    histories = audit.get("parameter_values", {})

    # Legacy inventories lost changed-parameter information; never use them as
    # proof.
    def parameter(name):
        values = histories.get(name, [])
        if (
            len(values) == 1
            and type(values[0]) in (int, float)
            and math.isfinite(values[0])
        ):
            return values[0]
        return None

    def put(key, value, source, reference=None):
        profile[key] = value
        evidence[key] = dict(
            value=value,
            source=source,
            source_sha256=audit["sha256"],
            reference=reference,
            status="logged_configuration",
        )

    def instances(topic):
        return sorted(
            int(i)
            for i, count in audit.get("instances", {}).get(topic, {}).items()
            if count > 0 and str(i).isdigit() and 0 <= int(i) <= 15
        )

    def unique_instance(key, topic):
        options = instances(topic)
        candidates[key] = options
        if len(options) == 1:
            put(key, options[0], topic + ": 로그에서 확인된 유일한 인스턴스")
        else:
            reasons[key] = (
                "여러 인스턴스 중 사용할 신호를 선택하세요."
                if options
                else "사용할 인스턴스가 기록에서 확인되지 않습니다."
            )

    put("autopilot", audit["autopilot"], "로그 형식")
    firmware = [
        m for m in audit.get("messages", []) if re.match(r"ArduCopter V\d", m)
    ]
    if audit["autopilot"] == "ardupilot":
        if len(set(firmware)) == 1:
            put("firmware", firmware[0], "MSG.Message")
        reviewed = bool(firmware) and all(
            re.match(r"ArduCopter V4\.4\.0(?:\s|$)", m) for m in firmware
        )
        if parameter("FRAME_CLASS") == 1:
            put("rotor_count", 4, "FRAME_CLASS=1 (Quad)", ARDU_PARAMS)
        functions = {i: parameter(f"SERVO{i}_FUNCTION") for i in range(1, 17)}
        if (
            reviewed
            and parameter("FRAME_CLASS") == 1
            and parameter("FRAME_TYPE") == 1
        ):
            # Standard Quad X: motor 1 FR, 2 RL, 3 FL, 4 RR. Never channel-count
            # inference.
            positions = dict(
                front_right=33, rear_left=34, front_left=35, rear_right=36
            )
            mapped = {
                pos: [i for i, value in functions.items() if value == fn]
                for pos, fn in positions.items()
            }
            changed_motor_assignment = any(
                len(v) > 1 and any(x in (33, 34, 35, 36) for x in v)
                for k, v in histories.items()
                if re.fullmatch(r"SERVO\d+_FUNCTION", k)
            )
            if (
                all(len(v) == 1 for v in mapped.values())
                and not changed_motor_assignment
            ):
                put(
                    "motor_channels",
                    {pos: v[0] for pos, v in mapped.items()},
                    (
                        "FRAME_CLASS=1 / FRAME_TYPE=1 + SERVOx_FUNCTION; "
                        "ArduCopter 4.4.0 표준 Quad X"
                    ),
                    ARDU_MATRIX,
                )
        unique_instance("imu_instance", "IMU")
        primary = audit.get("selectors", {}).get("XKF4.PI", [])
        mask = parameter("EK3_IMU_MASK")
        if (
            reviewed
            and parameter("AHRS_EKF_TYPE") == 3
            and mask is not None
            and mask == int(mask)
            and 0 < mask < 65536
        ):
            core_imus = [i for i in range(16) if int(mask) & (1 << i)]
            if (
                len(primary) == 1
                and type(primary[0]) is int
                and 0 <= primary[0] < len(core_imus)
                and all(i in instances("IMU") for i in core_imus)
            ):
                put(
                    "imu_instance",
                    core_imus[primary[0]],
                    (
                        f"전체 로그 XKF4.PI={primary[0]} 고정 / "
                        f"EK3_IMU_MASK={int(mask)} / AHRS_EKF_TYPE=3"
                    ),
                    (
                        "https://github.com/ArduPilot/ardupilot/blob/Copter-"
                        "4.4.0/libraries/AP_NavEKF3/AP_NavEKF3.cpp"
                    ),
                )
            elif len(primary) > 1:
                profile.pop("imu_instance", None)
                evidence.pop("imu_instance", None)
                reasons["imu_instance"] = (
                    f"사용한 추정기 코어가 {primary}로 바뀝니다. 보존 구간과 분석할 IMU를 확인하세요. "
                    f"자동 혼합하지 않습니다."
                )
        unique_instance("battery_instance", "BAT")
        if audit.get("counts", {}).get("RCOU", 0):
            # BIN RCOU is one channel array, not a selectable MAVLink output
            # bank.
            put(
                "actuator_instance",
                0,
                "BIN RCOU 단일 출력 배열 (실시간 port를 추정한 값이 아님)",
            )
        low, high = parameter("MOT_PWM_MIN"), parameter("MOT_PWM_MAX")
        if (
            parameter("MOT_PWM_TYPE") == 0
            and low is not None
            and high is not None
            and 0 < low < high
        ):
            put("output_min", low, "MOT_PWM_TYPE=0 / MOT_PWM_MIN", ARDU_PARAMS)
            put("output_max", high, "MOT_PWM_TYPE=0 / MOT_PWM_MAX", ARDU_PARAMS)
        battery = profile.get("battery_instance")
        if battery is not None:
            prefix = "BATT" if battery == 0 else f"BATT{battery + 1}"
            capacity = parameter(prefix + "_CAPACITY")
            if capacity is not None and capacity > 0:
                put(
                    "capacity_ah",
                    capacity / 1000,
                    prefix + "_CAPACITY (mAh → Ah, 기록된 설정값)",
                    ARDU_PARAMS,
                )
        reasons["series_cells"] = (
            "셀 수를 하나로 추정할 전압 근거가 부족하거나 설정이 충돌합니다."
        )
        # This is an application heuristic under a 4.2 V-class lithium
        # assumption,
        # not an ArduPilot cell-count formula or a measured hardware
        # specification.
        # BAT ranges aggregate instances: use them only for a single, matched
        # battery.
        voltage = audit.get("ranges", {}).get("BAT.Volt", [])
        bounds = {
            "MOT_BAT_VOLT_MAX": (4.0, 4.3),
            "MOT_BAT_VOLT_MIN": (2.6, 3.5),
            "BATT_LOW_VOLT": (2.9, 3.7),
            "BATT_CRT_VOLT": (2.7, 3.5),
        }
        settings = {key: parameter(key) for key in bounds}
        if (
            reviewed
            and instances("BAT") == [0]
            and parameter("MOT_BAT_IDX") == 0
            and audit.get("instances", {}).get("BAT", {}).get("0", 0) >= 100
            and audit.get("field_units", {}).get("BAT.Volt") == ["V"]
            and len(voltage) == 2
            and all(
                type(v) in (int, float) and math.isfinite(v) and v > 0
                for v in voltage
            )
            and voltage[0] <= voltage[1]
            and all(v is not None and v > 0 for v in settings.values())
            and settings["MOT_BAT_VOLT_MIN"] < settings["MOT_BAT_VOLT_MAX"]
            and settings["BATT_CRT_VOLT"]
            <= settings["BATT_LOW_VOLT"]
            < settings["MOT_BAT_VOLT_MAX"]
        ):
            options = [
                n
                for n in range(1, 33)
                if 3.8 <= voltage[1] / n <= 4.3
                and all(
                    lo <= settings[key] / n <= hi
                    for key, (lo, hi) in bounds.items()
                )
            ]
            if len(options) == 1:
                n = options[0]
                source = (
                    (
                        f"전압 기반 {n}S 추정 · BAT.Volt "
                        f"{voltage[0]:.2f}–{voltage[1]:.2f} V; "
                    )
                    + ", ".join(
                        f"{key}={value:g} V" for key, value in settings.items()
                    )
                    + ". 4.2 V급 리튬계 가정, 실물 미확인. 필요하면 수정하세요."
                )
                put("series_cells", n, source, ARDU_PARAMS)
                evidence["series_cells"].update(
                    status="estimated",
                    method="voltage-consistency-v1",
                    assumption=(
                        "4.2 V-class lithium battery; chemistry not verified"
                    ),
                    voltage_range_v=voltage,
                    parameters=settings,
                    per_cell_bounds_v=bounds,
                    candidates=options,
                )
    else:
        if audit.get("messages") and audit["messages"][0] != "unknown":
            put("firmware", audit["messages"][0], "ULog ver_sw")
        count = parameter("CA_ROTOR_COUNT")
        if count is not None and count == int(count) and 1 <= count <= 16:
            put("rotor_count", int(count), "CA_ROTOR_COUNT", PX4_PARAMS)
        unique_instance("battery_instance", "battery_status")
        unique_instance("actuator_instance", "actuator_outputs")
        angular, acceleration = (
            instances("vehicle_angular_velocity"),
            instances("vehicle_acceleration"),
        )
        candidates["imu_instance"] = sorted(set(angular) & set(acceleration))
        if len(angular) == len(acceleration) == 1 and angular == acceleration:
            put(
                "imu_instance",
                angular[0],
                "vehicle_angular_velocity / vehicle_acceleration의 유일한 공통 인스턴스",
            )
        battery = profile.get("battery_instance")
        if battery is not None:
            prefix = f"BAT{battery + 1}"
            cells, capacity = (
                parameter(prefix + "_N_CELLS"),
                parameter(prefix + "_CAPACITY"),
            )
            if cells is not None and cells == int(cells) and 1 <= cells <= 32:
                put(
                    "series_cells",
                    int(cells),
                    prefix + "_N_CELLS (명시 설정)",
                    PX4_PARAMS,
                )
            if capacity is not None and capacity > 0:
                put(
                    "capacity_ah",
                    capacity / 1000,
                    prefix + "_CAPACITY (mAh → Ah)",
                    PX4_PARAMS,
                )
        # PX4 output routing/mixer semantics vary by board/version. Expose
        # parameters
        # for review; do not equate output instance zero with a physical MAIN
        # connector.
        reasons["motor_channels"] = (
            "PX4 출력 포트·기능과 모터 위치의 연결을 확인하세요. 채널 번호만으로 위치를 추정하지 않습니다."
        )
        reasons["output_min"] = reasons["output_max"] = (
            "선택한 actuator_outputs의 출력 단위·범위를 확인하세요. 관측 최솟값/최댓값은 설정 범위가 아닙니다."
        )
    profile["provenance"] = dict(
        extraction_revision=REVISION,
        source_sha256=audit["sha256"],
        fields=evidence,
    )
    e = eligibility(profile)
    questions = [
        dict(
            field=k,
            label=LABELS[k],
            reason=reasons.get(
                k, "로그에서 확정할 근거가 부족하거나 설정이 변경되었습니다."
            ),
            candidates=candidates.get(k, []),
        )
        for k in e["log_missing_fields"]
    ]
    return dict(
        revision=REVISION,
        profile=profile,
        evidence=evidence,
        questions=questions,
        eligibility=e,
        source_sha256=audit["sha256"],
        health="unknown",
        input_fields_present=not audit.get("missing_candidate_fields"),
        parameter_conflicts={k: v for k, v in histories.items() if len(v) > 1},
        note=(
            "기록된 설정과 전압 기반 추정값을 구분해 표시합니다. 셀 수는 바로 수정할 수 있습니다. 건강 상태·기체 동일성은 "
            "추정하지 않습니다."
        ),
    )
