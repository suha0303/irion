"""Read-only onboard log inventory. Presence of a message is not AI
admission.
"""

import contextlib
import io
import math
from collections import Counter
from pathlib import Path

import numpy as numpy_module

from localflight import bin_reader, flight_events, profile_from_log

from .catalog import digest
from .timing import ClockAudit

PX4_REQUIRED = {
    "actuator_outputs": [
        "timestamp",
        "noutputs",
        "output[0]",
        "output[1]",
        "output[2]",
        "output[3]",
    ],
    "vehicle_angular_velocity": ["timestamp", "xyz[0]", "xyz[1]", "xyz[2]"],
    "vehicle_rates_setpoint": ["timestamp", "roll", "pitch", "yaw"],
    "vehicle_attitude": ["timestamp", "q[0]", "q[1]", "q[2]", "q[3]"],
    "vehicle_acceleration": ["timestamp", "xyz[0]", "xyz[1]", "xyz[2]"],
    "vehicle_local_position": ["timestamp", "vz", "v_z_valid"],
    "vehicle_local_position_setpoint": ["timestamp", "vz"],
    "battery_status": ["timestamp", "voltage_v", "current_a"],
    "vehicle_status": ["timestamp", "arming_state", "nav_state"],
    "vehicle_land_detected": ["timestamp", "landed"],
}
ARDUPILOT_REQUIRED = {
    "RATE": ["TimeUS", "R", "P", "Y", "RDes", "PDes", "YDes"],
    "RCOU": ["TimeUS", "C1", "C2", "C3", "C4"],
    "ATT": ["TimeUS", "Roll", "Pitch"],
    "IMU": ["TimeUS", "AccX", "AccY", "AccZ"],
    "CTUN": ["TimeUS", "CRt", "DCRt"],
    "BAT": ["TimeUS", "Volt", "Curr"],
    "MODE": ["TimeUS", "Mode"],
}


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def scan_bin(path):
    reader = bin_reader.open_bin(path)
    counts, fields, clocks, ranges, params = Counter(), {}, {}, {}, {}
    messages = []
    parameter_values, instances, unit_sets, selectors = {}, {}, {}, {}
    record_count = 0
    timing = ClockAudit()
    operational, truncated, last_primary = [], False, None
    try:
        while True:
            message = reader.recv_msg()
            if message is None:
                break
            kind = message.get_type()
            counts[kind] += 1
            record_count += 1
            if record_count > 10_000_000:
                raise ValueError(
                    "Log exceeds ten million records; split flight logs"
                )
            if kind not in ARDUPILOT_REQUIRED and kind not in {
                "MSG",
                "PARM",
                "ARM",
                "EV",
                "XKF4",
                "ERR",
            }:
                continue
            row = message.to_dict()
            stamp = row.get("TimeUS")
            if (
                type(stamp) is int
                and stamp >= 0
                and (
                    kind in {"ARM", "EV", "ERR"}
                    or (
                        kind == "XKF4"
                        and type(row.get("PI")) is int
                        and row["PI"] != last_primary
                    )
                )
            ):
                if len(operational) < 20000:
                    operational.append(
                        dict(
                            kind=kind,
                            **{
                                k: v
                                for k, v in row.items()
                                if k
                                in {
                                    "TimeUS",
                                    "PI",
                                    "Id",
                                    "Subsys",
                                    "ECode",
                                    "ArmState",
                                }
                                and type(v) is int
                            },
                        )
                    )
                else:
                    truncated = True
                if kind == "XKF4":
                    last_primary = row["PI"]
            fields.setdefault(kind, set()).update(row)
            if kind == "XKF4" and type(row.get("PI")) is int:
                selectors.setdefault("XKF4.PI", set()).add(row["PI"])
            if kind == "PARM" and isinstance(row.get("Name"), str):
                name = row["Name"]
                if name.startswith(
                    (
                        "BATT",
                        "MOT_",
                        "SERVO",
                        "FRAME",
                        "LOG_",
                        "AHRS",
                        "INS_",
                        "EK3_",
                        "RC3_",
                    )
                ):
                    value = (
                        row.get("Value") if _number(row.get("Value")) else None
                    )
                    if value is not None:
                        params[name] = value
                    values = parameter_values.setdefault(name, [])
                    if value not in values and len(values) < 32:
                        values.append(value)
            if kind == "MSG" and len(messages) < 32:
                messages.append(str(row.get("Message", ""))[:300])
            stamp = row.get("TimeUS")
            if _number(stamp):
                instance = row.get("I", row.get("Instance", 0))
                timing.add(f"{kind}:{instance}", stamp)
                c = clocks.setdefault(
                    kind, dict(first=stamp, last=stamp, samples=0, reversals=0)
                )
                c["reversals"] += int(stamp < c["last"])
                c["last"] = stamp
                c["samples"] += 1
            if kind in ARDUPILOT_REQUIRED:
                instance = row.get("I", row.get("Instance", 0))
                if type(instance) is int and 0 <= instance <= 255:
                    bucket = instances.setdefault(kind, {})
                    bucket[str(instance)] = bucket.get(str(instance), 0) + 1
                for field, unit in zip(
                    message.fmt.columns, message.fmt.units or []
                ):
                    unit_sets.setdefault(kind + "." + field, set()).add(unit)
                for key, value in row.items():
                    if key != "TimeUS" and _number(value):
                        r = ranges.setdefault(kind + "." + key, [value, value])
                        r[0], r[1] = min(r[0], value), max(r[1], value)
    finally:
        reader.close()
    return dict(
        autopilot="ardupilot",
        counts=dict(counts),
        fields={k: sorted(v) for k, v in fields.items()},
        clocks=clocks,
        ranges=ranges,
        parameters=params,
        parameter_values=parameter_values,
        instances=instances,
        selectors={k: sorted(v) for k, v in selectors.items()},
        field_units={k: sorted(v) for k, v in unit_sets.items()},
        messages=messages,
        timing=timing.result(),
        operational_events=operational,
        operational_events_truncated=truncated,
        parser=dict(
            name="pymavlink.DFReader_binary",
            indexer="python",
            native_indexer=False,
        ),
    )


def scan_ulog(path):
    np = numpy_module
    from pyulog import ULog

    # Parse only the relevant topic set; a file-size limit also bounds this
    # nonstreaming parser.
    ulog = ULog(str(path), message_name_filter_list=list(PX4_REQUIRED))
    counts, fields, clocks, ranges = {}, {}, {}, {}
    timing = ClockAudit()
    for data in ulog.data_list:
        key = f"{data.name}:{data.multi_id}"
        stamps = data.data.get("timestamp", np.array([], dtype="u8"))
        counts[key] = len(stamps)
        fields[key] = sorted(data.data)
        samples = data.data.get("timestamp_sample", stamps)
        for stamp, sample in zip(stamps, samples):
            timing.add(key, stamp, sample)
        if len(stamps):
            clocks[key] = dict(
                first=int(stamps[0]),
                last=int(stamps[-1]),
                samples=len(stamps),
                reversals=int(np.sum(stamps[1:] < stamps[:-1])),
            )
        for field, values in data.data.items():
            if field.startswith("timestamp") or not np.issubdtype(
                values.dtype, np.number
            ):
                continue
            finite = values[np.isfinite(values)]
            if len(finite):
                ranges[key + "." + field] = [
                    float(finite.min()),
                    float(finite.max()),
                ]
    parameters = {
        k: v
        for k, v in ulog.initial_parameters.items()
        if k.startswith(("BAT", "PWM", "DSHOT", "CA_ROTOR", "SYS_AUTOSTART"))
        and _number(v)
    }
    parameter_values = {k: [v] for k, v in parameters.items()}
    for _, name, value in ulog.changed_parameters:
        if name.startswith(
            ("BAT", "PWM", "DSHOT", "CA_ROTOR", "SYS_AUTOSTART")
        ):
            value = value if _number(value) else None
            values = parameter_values.setdefault(name, [])
            # An absent initial value cannot establish the configuration for the
            # whole log.
            if name not in parameters and not values:
                values.append(None)
            if value not in values and len(values) < 32:
                values.append(value)
    return dict(
        autopilot="px4",
        counts=counts,
        fields=fields,
        clocks=clocks,
        ranges=ranges,
        parameters=parameters,
        parameter_values=parameter_values,
        instances={
            topic: {
                str(d.multi_id): len(d.data.get("timestamp", []))
                for d in ulog.data_list
                if d.name == topic
            }
            for topic in PX4_REQUIRED
        },
        messages=[str(ulog.msg_info_dict.get("ver_sw", "unknown"))],
        timing={
            **timing.result(),
            "dropouts": [
                dict(timestamp_us=int(d.timestamp), duration_ms=int(d.duration))
                for d in ulog.dropouts
            ],
        },
    )


def inspect_log(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("A regular onboard log is required")
    if not 0 < path.stat().st_size <= 256 * 1024 * 1024:
        raise ValueError("Onboard log limit is 256 MiB")
    suffix = path.suffix.lower()
    if suffix not in {".bin", ".ulg"}:
        raise ValueError("Use an ArduPilot .BIN or PX4 .ulg onboard log")
    original = digest(path)
    warnings = io.StringIO()
    with (
        contextlib.redirect_stdout(warnings),
        contextlib.redirect_stderr(warnings),
    ):
        result = scan_bin(path) if suffix == ".bin" else scan_ulog(path)
    expected = ARDUPILOT_REQUIRED if suffix == ".bin" else PX4_REQUIRED
    missing = {}
    for topic, fields in expected.items():
        keys = (
            [topic]
            if suffix == ".bin"
            else [
                key for key in result["fields"] if key.startswith(topic + ":")
            ]
        )
        # A nonzero instance can be valid; do not combine fields across
        # instances.
        key = (
            min(
                keys,
                key=lambda k, fields=fields: len(
                    set(fields) - set(result["fields"].get(k, []))
                ),
            )
            if keys
            else topic
        )
        absent = sorted(set(fields) - set(result["fields"].get(key, [])))
        if absent:
            missing[key] = absent
    if original != digest(path):
        raise ValueError("Source log changed during inspection")
    audit = dict(
        schema="drone.local-log-audit/1",
        filename=path.name,
        sha256=original,
        bytes=path.stat().st_size,
        **result,
        missing_candidate_fields=missing,
        parser_warnings=warnings.getvalue()[:8000],
        health_status="unknown",
        model_forward_executed=False,
        training_eligible=False,
        ai_admission="pending_profile_and_semantic_mapping",
        scope=(
            "Onboard message/field inventory; not semantic equivalence or "
            "health validation"
        ),
    )

    audit["profile_suggestion"] = profile_from_log.suggest(audit)
    if suffix == ".bin":
        audit["flight_events"] = flight_events.summarize(audit)
    return audit
