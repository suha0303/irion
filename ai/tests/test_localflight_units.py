"""Compare physical units across DataFlash, uORB ULog and MAVLink.

Fixtures describe known signals, not real flight health. DataFlash decoding
uses the installed parser so integer storage scaling is exercised as well.
"""

import math
import struct

import pytest
from pymavlink.DFReader import DFFormat, DFMessage

from localflight.live import decode
from localflight.prepare import bin_values, px4_events

PROFILE = dict(
    imu_instance=0,
    battery_instance=0,
    actuator_instance=0,
    output_min=1000,
    output_max=2000,
    motor_channels=dict(front_right=1, rear_left=2, front_left=3, rear_right=4),
)
REQUEST = dict(imu_message="HIGHRES_IMU", imu_id=0)


def dataflash(kind, encoding, columns, units, multipliers, elements):
    """Build a stored record with actual DataFlash format/unit machinery."""
    fmt = DFFormat(1, kind, 0, encoding, columns)
    fmt.set_unit_ids(units, dict(d="deg", k="deg/s", n="m/s", o="m/s/s"))
    fmt.set_mult_ids(multipliers, {"0": 1, "B": 0.01})
    return DFMessage(fmt, elements, True, None)


def ulog(tmp_path, topic, fields):
    """Write one genuine ULog record; no parser or adapter mocking."""

    def message(kind, payload):
        return struct.pack("<HB", len(payload), ord(kind)) + payload

    types = {"uint64_t": "Q", "uint32_t": "I", "float": "f", "bool": "?"}
    definition = (
        topic
        + ":"
        + ";".join(f"{kind} {name}" for kind, name, _ in fields)
        + ";"
    )
    payload = struct.pack("<H", 0)
    for kind, _, value in fields:
        if "[" in kind:
            scalar, count = kind.rstrip("]").split("[")
            payload += struct.pack("<" + types[scalar] * int(count), *value)
        else:
            payload += struct.pack("<" + types[kind], value)
    content = b"ULog\x01\x12\x35\x01" + struct.pack("<Q", 1000000)
    content += message("F", definition.encode())
    content += message("A", struct.pack("<BH", 0, 0) + topic.encode())
    content += message("D", payload)
    path = tmp_path / "units.ulg"
    path.write_bytes(content)
    return path


@pytest.mark.parametrize(
    "altitude_valid,velocity_valid,expected",
    [
        (True, True, 2.5),
        (True, False, None),
        (False, True, 2.5),
        (False, False, None),
        (True, None, None),
    ],
)
def test_px4_climb_requires_velocity_validity(
    tmp_path, altitude_valid, velocity_valid, expected
):
    fields = [
        ("uint64_t", "timestamp", 1000000),
        ("float", "vz", -2.5),
        ("bool", "z_valid", altitude_valid),
    ]
    if velocity_valid is not None:
        fields.append(("bool", "v_z_valid", velocity_valid))
    events, _ = px4_events(
        ulog(tmp_path, "vehicle_local_position", fields), PROFILE
    )
    assert events[0][2]["climb_rate"] == expected


def test_centidegree_storage_is_not_scaled_twice():
    message = dataflash("ATT", "cc", "Roll,Pitch", "dd", "BB", [9000, -4500])
    assert message.to_dict()["Roll"] == 90
    assert message.fmt.units == ["deg", "deg"]
    expected = dict(roll=math.pi / 2, pitch=-math.pi / 4)
    assert bin_values(message, PROFILE) == pytest.approx(expected)
    live = decode(
        "ATTITUDE", dict(roll=math.pi / 2, pitch=-math.pi / 4), PROFILE, REQUEST
    )
    assert {k: live[k] for k in expected} == expected


@pytest.mark.parametrize(
    "encoding,stored", [("hh", [250, -125]), ("cc", [250, -125])]
)
def test_climb_prefix_and_storage_scale_each_apply_once(encoding, stored):
    message = dataflash("CTUN", encoding, "CRt,DCRt", "nn", "BB", stored)
    # h keeps centimeters; c has already become meters in DFReader.
    assert message.fmt.units == (
        ["cm/s"] * 2 if encoding == "hh" else ["m/s"] * 2
    )
    expected = dict(climb_rate=2.5, climb_target=-1.25)
    assert bin_values(message, PROFILE) == expected
    assert (
        decode("LOCAL_POSITION_NED", dict(vz=-2.5), PROFILE, REQUEST)[
            "climb_rate"
        ]
        == 2.5
    )
    assert (
        decode(
            "POSITION_TARGET_LOCAL_NED",
            dict(vz=1.25, coordinate_frame=1, type_mask=0),
            PROFILE,
            REQUEST,
        )["climb_target"]
        == -1.25
    )


def test_rate_and_targets_are_radians_per_second(tmp_path):
    message = dataflash(
        "RATE",
        "ffffff",
        "R,P,Y,RDes,PDes,YDes",
        "kkkkkk",
        "000000",
        [180, -90, 45, 90, -45, 0],
    )
    values = bin_values(message, PROFILE)
    assert [values["body_rate_" + axis] for axis in "pqr"] == pytest.approx(
        [math.pi, -math.pi / 2, math.pi / 4]
    )
    fields = [("uint64_t", "timestamp", 1000000)] + [
        ("float", name, value)
        for name, value in zip(
            ("roll", "pitch", "yaw"), (math.pi / 2, -math.pi / 4, 0)
        )
    ]
    events, _ = px4_events(
        ulog(tmp_path, "vehicle_rates_setpoint", fields), PROFILE
    )
    target = {
        "rate_target_" + axis: values["rate_target_" + axis] for axis in "pqr"
    }
    assert events[0][2] == pytest.approx(target)
    live = decode(
        "ATTITUDE_TARGET",
        dict(
            type_mask=0,
            body_roll_rate=math.pi / 2,
            body_pitch_rate=-math.pi / 4,
            body_yaw_rate=0,
        ),
        PROFILE,
        REQUEST,
    )
    assert live == pytest.approx(target)


@pytest.mark.parametrize(
    "message", ["SCALED_IMU", "SCALED_IMU2", "SCALED_IMU3", "HIGHRES_IMU"]
)
def test_acceleration_uses_specific_force_without_gravity_subtraction(message):
    raw = -9.80665 if message == "HIGHRES_IMU" else -1000
    values = decode(
        message,
        dict(id=0, fields_updated=7, xacc=0, yacc=0, zacc=raw),
        PROFILE,
        dict(imu_message=message, imu_id=0),
    )
    assert values == dict(
        specific_force_x=0, specific_force_y=0, specific_force_z=-9.80665
    )


@pytest.mark.parametrize(
    "voltages,extension",
    [([48000] + [65535] * 9, []), ([4000] * 10, [4000, 4000, 0, 0])],
)
def test_mavlink_millivolts_and_centiamps_match_px4_si(
    tmp_path, voltages, extension
):
    live = decode(
        "BATTERY_STATUS",
        dict(
            id=0,
            voltages=voltages,
            voltages_ext=extension,
            current_battery=2500,
        ),
        PROFILE,
        REQUEST,
    )
    fields = [
        ("uint64_t", "timestamp", 1000000),
        ("float", "voltage_v", 48),
        ("float", "current_a", 25),
    ]
    events, _ = px4_events(ulog(tmp_path, "battery_status", fields), PROFILE)
    assert live == events[0][2] == dict(pack_voltage=48, pack_current=25)


def test_unknown_onboard_units_are_rejected():
    message = dataflash("CTUN", "hh", "CRt,DCRt", "??", "00", [250, -125])
    with pytest.raises(ValueError, match="Unverified onboard unit"):
        bin_values(message, PROFILE)


@pytest.mark.parametrize("current", [-1, float("nan"), None])
def test_px4_unknown_battery_current_is_not_a_measurement(tmp_path, current):
    fields = [("uint64_t", "timestamp", 1000000), ("float", "voltage_v", 48)]
    if current is not None:
        fields.append(("float", "current_a", current))
    events, _ = px4_events(ulog(tmp_path, "battery_status", fields), PROFILE)
    assert events[0][2] == dict(pack_voltage=48, pack_current=None)


@pytest.mark.parametrize("current", [-2, 0, 25])
def test_px4_finite_current_is_not_clamped_to_a_training_range(
    tmp_path, current
):
    fields = [
        ("uint64_t", "timestamp", 1000000),
        ("float", "voltage_v", 48),
        ("float", "current_a", current),
    ]
    events, _ = px4_events(ulog(tmp_path, "battery_status", fields), PROFILE)
    assert events[0][2]["pack_current"] == current


@pytest.mark.parametrize("count", [0, 2, 4, 17, None])
def test_px4_only_declared_actuator_channels_are_used(tmp_path, count):
    fields = [
        ("uint64_t", "timestamp", 1000000),
        ("float[4]", "output", [1500] * 4),
    ]
    if count is not None:
        fields.append(("uint32_t", "noutputs", count))
    events, _ = px4_events(ulog(tmp_path, "actuator_outputs", fields), PROFILE)
    expected = {
        "motor_request_" + position: 0.5
        if count is not None and channel <= count <= 16
        else None
        for position, channel in PROFILE["motor_channels"].items()
    }
    assert events[0][2] == expected


@pytest.mark.parametrize("low,high,raw", [(1000, 2000, 1250), (0, 1000, 250)])
def test_px4_output_range_is_explicit_not_assumed_pwm(tmp_path, low, high, raw):
    fields = [
        ("uint64_t", "timestamp", 1000000),
        ("uint32_t", "noutputs", 4),
        ("float[4]", "output", [raw] * 4),
    ]
    profile = {**PROFILE, "output_min": low, "output_max": high}
    events, _ = px4_events(ulog(tmp_path, "actuator_outputs", fields), profile)
    assert set(events[0][2].values()) == {0.25}
