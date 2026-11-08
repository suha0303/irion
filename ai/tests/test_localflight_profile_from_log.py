"""Check profile suggestions against recorded log evidence and conflicts."""

import copy

from localflight.profile_from_log import suggest
from localflight.profiles import eligibility, save_profile


def voltage_audit(scale=1):
    row = audit()
    row["parameter_values"].update(
        {
            k: [v * scale]
            for k, v in dict(
                MOT_BAT_VOLT_MAX=49.2,
                MOT_BAT_VOLT_MIN=33.6,
                BATT_LOW_VOLT=37.2,
                BATT_CRT_VOLT=36,
            ).items()
        }
    )
    row["parameter_values"]["MOT_BAT_IDX"] = [0]
    row["ranges"] = {"BAT.Volt": [44 * scale, 50.06 * scale]}
    row["instances"]["BAT"] = {"0": 1000}
    row["field_units"] = {"BAT.Volt": ["V"]}
    return row


def test_voltage_estimate_is_editable_and_not_a_confirmed_measurement(tmp_path):
    for scale, count in ((1, 12), (0.5, 6)):
        result = suggest(voltage_audit(scale))
        assert result["profile"]["series_cells"] == count
        assert result["evidence"]["series_cells"]["status"] == "estimated"
        assert result["health"] == "unknown"
        assert result["questions"] == []
        saved = save_profile(tmp_path, result["profile"])
        assert saved["eligibility"]["profile_complete"]
        assert (
            saved["profile"]["provenance"]["fields"]["series_cells"]["status"]
            == "estimated"
        )


def test_voltage_estimate_requires_consistent_unambiguous_evidence():
    row = voltage_audit()
    variants = []
    for key in (
        "MOT_BAT_VOLT_MAX",
        "MOT_BAT_VOLT_MIN",
        "BATT_LOW_VOLT",
        "BATT_CRT_VOLT",
        "MOT_BAT_IDX",
    ):
        changed = copy.deepcopy(row)
        changed["parameter_values"][key].append(0)
        variants.append(changed)
    for update in (
        {"ranges": {"BAT.Volt": [44, 80]}},
        {"ranges": {"BAT.Volt": [44, float("nan")]}},
        {"field_units": {"BAT.Volt": ["mV"]}},
        {"instances": {"BAT": {"0": 100, "1": 100}}},
    ):
        variants.append({**row, **update})
    ambiguous = voltage_audit(2.4)
    variants.append(
        ambiguous
    )  # 28S and 29S satisfy this heuristic: do not pick one.
    for changed in variants:
        assert "series_cells" not in suggest(changed)["profile"]


def audit():
    params = dict(
        FRAME_CLASS=1,
        FRAME_TYPE=1,
        MOT_PWM_TYPE=0,
        MOT_PWM_MIN=1000,
        MOT_PWM_MAX=2000,
        BATT_CAPACITY=44000,
        AHRS_EKF_TYPE=3,
        EK3_IMU_MASK=7,
    )
    params.update({f"SERVO{i}_FUNCTION": 32 + i for i in range(1, 5)})
    return dict(
        filename="fixture.BIN",
        sha256="a" * 64,
        autopilot="ardupilot",
        parameter_values={k: [v] for k, v in params.items()},
        parameters=params,
        messages=["ArduCopter V4.4.0 (502702df)"],
        counts={"RCOU": 10},
        instances={"IMU": {"0": 10, "1": 10, "2": 10}, "BAT": {"0": 10}},
        selectors={"XKF4.PI": [1]},
        missing_candidate_fields={},
    )


def test_logged_values_fill_only_known_configuration():
    result = suggest(audit())
    p = result["profile"]
    assert p["rotor_count"] == 4 and p["capacity_ah"] == 44
    assert p["motor_channels"] == dict(
        front_right=1, front_left=3, rear_right=4, rear_left=2
    )
    assert (
        p["imu_instance"] == 1
        and p["battery_instance"] == 0
        and p["actuator_instance"] == 0
    )
    assert p["output_min"] == 1000 and p["output_max"] == 2000
    assert "series_cells" not in p and result["health"] == "unknown"
    assert [q["field"] for q in result["questions"]] == ["series_cells"]
    assert result["evidence"]["motor_channels"]["source_sha256"] == "a" * 64
    assert result["eligibility"]["timeline_ready"]


def test_parameter_and_sensor_conflicts_require_confirmation():
    changed = audit()
    changed["parameter_values"]["MOT_PWM_MAX"] = [2000, 1900]
    changed["parameter_values"]["BATT_CAPACITY"] = [44000, 22000]
    changed["parameter_values"]["SERVO5_FUNCTION"] = [0, 33]
    changed["selectors"]["XKF4.PI"] = [0, 1]
    result = suggest(changed)
    p = result["profile"]
    assert (
        not p["motor_channels"]
        and "capacity_ah" not in p
        and "output_max" not in p
    )
    assert "imu_instance" not in p
    assert next(q for q in result["questions"] if q["field"] == "imu_instance")[
        "candidates"
    ] == [
        0,
        1,
        2,
    ]
    changed = audit()
    changed["parameter_values"]["SERVO5_FUNCTION"] = [33]
    assert not suggest(changed)["profile"]["motor_channels"]
    changed = audit()
    changed["messages"] = ["Custom firmware"]
    assert not suggest(changed)["profile"]["motor_channels"]


def test_missing_and_zero_values_are_not_inferred_from_voltage_or_defaults():
    row = audit()
    row["parameter_values"]["MOT_PWM_MIN"] = [0]
    row["parameter_values"]["BATT_CAPACITY"] = [0]
    row["ranges"] = {"BAT.Volt": [38, 50.4]}
    row["parameter_values"]["MOT_BAT_VOLT_MAX"] = [50.4]
    result = suggest(row)
    for key in ("series_cells", "capacity_ah", "output_min"):
        assert key not in result["profile"]
    legacy = audit()
    legacy.pop("parameter_values")
    assert "capacity_ah" not in suggest(legacy)["profile"]


def test_record_only_information_does_not_block_analysis_or_identity(tmp_path):
    p = suggest(audit())["profile"]
    p["series_cells"] = 12
    for key in ("firmware", "battery_id", "mapping_evidence", "notes"):
        p.pop(key, None)
    assert eligibility(p)["profile_complete"]
    p["id"] = ""
    p["name"] = ""
    saved = save_profile(tmp_path, p)
    assert (
        saved["profile"]["id"].startswith("aircraft-")
        and saved["profile"]["name"]
    )
    assert saved["eligibility"]["profile_complete"]


def test_px4_explicit_cell_count_capacity_and_changed_setting():
    row = dict(
        filename="fixture.ulg",
        sha256="b" * 64,
        autopilot="px4",
        messages=["revision"],
        parameter_values={
            "BAT1_N_CELLS": [6],
            "BAT1_CAPACITY": [5000],
            "CA_ROTOR_COUNT": [4],
        },
        instances={
            k: {"0": 10}
            for k in (
                "battery_status",
                "actuator_outputs",
                "vehicle_angular_velocity",
                "vehicle_acceleration",
            )
        },
    )
    result = suggest(row)
    p = result["profile"]
    assert (
        p["series_cells"] == 6
        and p["capacity_ah"] == 5
        and p["imu_instance"] == 0
    )
    assert not p["motor_channels"] and "output_min" not in p
    row["parameter_values"]["BAT1_N_CELLS"] = [0, 6]
    assert "series_cells" not in suggest(row)["profile"]
    row["instances"]["battery_status"]["1"] = 20
    assert "capacity_ah" not in suggest(row)["profile"]


def test_inventory_retains_invalid_parameter_changes(monkeypatch):
    from pymavlink import DFReader

    from localflight.logs import scan_bin

    class Message:
        def __init__(self, value):
            self.value = value

        def get_type(self):
            return "PARM"

        def to_dict(self):
            return dict(Name="BATT_CAPACITY", Value=self.value)

    class Reader:
        def __init__(self, path):
            self.rows = iter([Message(44000), Message(float("nan")), None])

        def recv_msg(self):
            return next(self.rows)

        def close(self):
            pass

    monkeypatch.setattr(DFReader, "DFReader_binary", Reader)
    result = scan_bin("fixture.BIN")
    assert result["parameter_values"]["BATT_CAPACITY"] == [44000, None]
    row = audit()
    row["parameter_values"]["BATT_CAPACITY"] = result["parameter_values"][
        "BATT_CAPACITY"
    ]
    assert "capacity_ah" not in suggest(row)["profile"]
