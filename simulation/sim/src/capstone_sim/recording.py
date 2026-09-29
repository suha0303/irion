"""DRONE-RAW producer: no resampling, features or truth-based imputation."""

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import jsonschema

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"
GENERIC_METADATA = "drone.observation.metadata/3.0"
GENERIC_RAW = "drone.raw.event/3.0"


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generic_vehicle(profile):
    """A simulator profile never opts into the existing X500 AI input contract."""
    battery = profile["battery"]
    vehicle = dict(
        profile_id=profile["id"],
        profile_revision=profile["revision"],
        rotor_count=profile["rotor_count"],
        motor_order=list(profile["motor_order"]),
        nominal_series_cells=battery["ns"],
        nominal_parallel_cells=battery["np"],
        nominal_capacity_ah=battery["capacity_ah"],
        battery_profile_id=battery["profile_id"],
    )
    if profile.get("ai_compatible", False) is not False:
        raise ValueError("Generic simulation profiles are not AI compatible")
    if profile.get("simulation_profile_manifest", "reference/vehicle-profile.json") != (
        "reference/vehicle-profile.json"
    ):
        raise ValueError("Generic simulation profile manifest must remain run-local")
    if vehicle["profile_id"] == "x500_reference":
        raise ValueError("Reserved X500 AI profile cannot be used as a generic vehicle")
    if (
        type(vehicle["rotor_count"]) is not int
        or not 4 <= vehicle["rotor_count"] <= 8
        or len(vehicle["motor_order"]) != vehicle["rotor_count"]
        or len(set(vehicle["motor_order"])) != vehicle["rotor_count"]
    ):
        raise ValueError("Generic motor order must match rotor_count within 4..8")
    schema = json.loads((CONTRACTS / "generic-metadata.schema.json").read_text())
    jsonschema.Draft202012Validator(schema["properties"]["vehicle"]).validate(vehicle)
    if not math.isfinite(vehicle["nominal_capacity_ah"]):
        raise ValueError("Generic battery nameplate must be finite")
    return vehicle


def generic_binding(run, profile):
    path = Path(run) / "reference/vehicle-profile.json"
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Generic profile binding cannot contain symlinks")
    if not path.is_file():
        raise ValueError("Generic run requires a recorded simulation profile manifest")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != "drone.simulation.profile/1"
        or manifest.get("profile_id") != profile["id"]
        or manifest.get("observation_vehicle") != profile
    ):
        raise ValueError("Run profile manifest differs from the active vehicle profile")
    return dict(path="reference/vehicle-profile.json", sha256=sha(path))


def verify_generic_metadata(run, metadata):
    schema = json.loads((CONTRACTS / "generic-metadata.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(metadata)
    reference = metadata["simulation_profile_manifest"]
    path = Path(run) / reference["path"]
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Generic profile binding must be a regular run-local file")
    if sha(path) != reference["sha256"]:
        raise ValueError("Generic profile manifest digest mismatch")
    manifest = json.loads(path.read_text())
    profile = manifest["observation_vehicle"]
    if (
        manifest.get("schema") != "drone.simulation.profile/1"
        or manifest.get("profile_id") != metadata["vehicle"]["profile_id"]
        or generic_binding(run, profile) != reference
        or generic_vehicle(profile) != metadata["vehicle"]
    ):
        raise ValueError("Generic vehicle metadata differs from its profile manifest")


TOPICS = {
    "actuator_outputs": (
        "motor_request",
        {"output": "output_count"},
        "post_output_mapping; divide by 1000 for ESC duty",
    ),
    "vehicle_angular_velocity": (
        "body_rate",
        {"xyz": "rad/s"},
        "FRD bias corrected measured gyro",
    ),
    "vehicle_rates_setpoint": (
        "body_rate_target",
        {"roll": "rad/s", "pitch": "rad/s", "yaw": "rad/s"},
        "FRD controller target",
    ),
    "vehicle_attitude": ("attitude", {"q": "unitless"}, "wxyz FRD to NED"),
    "vehicle_acceleration": (
        "specific_force",
        {"xyz": "m/s^2"},
        "FRD specific force; rest z=-g",
    ),
    "vehicle_local_position": (
        "climb_rate",
        {"x": "m", "y": "m", "z": "m", "vz": "m/s", "z_valid": "boolean_as_number"},
        "NED; climb=-vz",
    ),
    "vehicle_local_position_setpoint": (
        "climb_rate_target",
        {"vx": "m/s", "vy": "m/s", "vz": "m/s"},
        "NED actual position controller velocity target; climb=-vz",
    ),
    "trajectory_setpoint": (
        "navigation_request",
        {"velocity": "m/s"},
        "NED; null means unavailable",
    ),
    "battery_status": (
        "pack_voltage_current",
        {"voltage_v": "V", "current_a": "A"},
        "discharge positive; PX4 Battery from sampled V/I; timestamp is original sensor measurement time including transport delay",
    ),
    "vehicle_status": (
        "operating_state",
        {"arming_state": "enum", "nav_state": "enum", "hil_state": "enum"},
        "PX4 v1.16 enum",
    ),
    "vehicle_land_detected": (
        "landed_state",
        {"landed": "boolean_as_number"},
        "PX4 land detector",
    ),
}


class Recorder:
    def __init__(self, run, origin="px4_sitl", timing=False, *, vehicle_profile=None):
        self.run = Path(run)
        self.id = self.run.name
        generic = vehicle_profile is not None
        vehicle = generic_vehicle(vehicle_profile) if generic else None
        profile_binding = generic_binding(self.run, vehicle_profile) if generic else None
        self.raw_schema = GENERIC_RAW if generic else "drone.raw.event/2.0"
        self.count = 0
        self.latest = {}
        self.last_sample = {}
        self.through = -1
        self.seq = defaultdict(int)
        self.pending = b""
        self.offset = 0
        self.stream = (self.run / "observations/streams.jsonl").open("w")
        self.watermarks = (self.run / "observations/watermarks.jsonl").open("w")
        self.diag = (self.run / "observations/diagnostics.jsonl").open("w")
        self.timing = (self.run / "native_logs/producer-timing.jsonl").open("w") if timing else None
        self.timing_seq = 0
        bindings = {}
        for topic, (role, units, meaning) in TOPICS.items():
            bindings[role] = {
                "source": "px4.uorb",
                "topic": topic,
                "instance": 0,
                "fields": list(units),
                "units": units,
                "meaning": meaning,
                "expected_period_us": {
                    "battery_status": 100000,
                    "vehicle_status": 500000,
                    "vehicle_land_detected": 1000000,
                }.get(topic, 10000),
                "availability": "must_confirm_from_run",
                "capture": "uORB latest sample polling at 100Hz; native ULog separately",
            }
        metadata = {
            "schema": "drone.observation.metadata/2.0",
            "run_id": self.id,
            "origin": origin,
            "clock": {
                "id": "px4-boot",
                "unit": "us",
                "mode": "capture_replay",
                "epoch": "simulation_boot",
            },
            "frames": {
                "body": "FRD",
                "navigation": "NED",
                "quaternion": "wxyz_body_to_navigation",
            },
            "vehicle": {
                "profile_id": "x500_reference",
                "rotor_count": 4,
                "motor_order": [
                    "front_right",
                    "rear_left",
                    "front_left",
                    "rear_right",
                ],
                "nominal_series_cells": 4,
                "nominal_capacity_ah": 6,
            },
            "actuation": {
                "semantic": "esc_request",
                "encoding": "output_count_0_1000",
                "stage": "post_output_mapping",
                "mapping_id": "px4-v1.16-SIM_GZ_EC-duty-v1",
            },
            "semantic_bindings": bindings,
            "conversion_history": [],
        }
        if generic:
            metadata.update(
                schema=GENERIC_METADATA,
                vehicle=vehicle,
                ai_compatible=False,
                simulation_profile_manifest=profile_binding,
            )
            verify_generic_metadata(self.run, metadata)
        dump(self.run / "observations/metadata.json", metadata)

    def stamp(self):
        return time.monotonic_ns() if self.timing is not None else 0

    def trace(self, fields):
        if self.timing is None:
            return
        self.timing.write(
            json.dumps(
                dict(
                    schema="x500.producer-timing/1",
                    run_id=self.id,
                    clock_id="px4-boot",
                    host_clock="CLOCK_MONOTONIC",
                    host_clock_unit="ns",
                    timing_seq=self.timing_seq,
                    **fields,
                ),
                allow_nan=False,
            )
            + "\n"
        )
        self.timing.flush()
        self.timing_seq += 1

    def consume(self, scheduled_host_monotonic_ns=None):
        batch = {
            "consume_start_host_monotonic_ns": self.stamp(),
            "scheduled_host_monotonic_ns": scheduled_host_monotonic_ns,
            "native_read_byte_start": self.offset,
            "pending_bytes_before": len(self.pending),
            "first_event_index": self.count + 1,
        }
        path = self.run / "native_logs/capture.jsonl"
        if not path.exists():
            self.trace(dict(kind="capture_missing", **batch))
            return
        batch["native_read_start_host_monotonic_ns"] = self.stamp()
        with path.open("rb") as f:
            f.seek(self.offset)
            new = f.read()
            self.offset = f.tell()
        batch["native_read_end_host_monotonic_ns"] = self.stamp()
        batch["native_read_byte_end"] = self.offset
        batch["native_read_bytes"] = len(new)
        lines = (self.pending + new).split(b"\n")
        self.pending = lines.pop()
        batch["pending_bytes_after"] = len(self.pending)
        if not lines:
            self.trace(dict(kind="capture_empty", **batch))
            return
        batch["transform_start_host_monotonic_ns"] = self.stamp()
        serialization_ns = 0
        for line in lines:
            e = json.loads(line)
            topic = e["topic"]
            self.latest[topic] = e["fields"]
            if topic not in TOPICS:
                raise ValueError(f"undeclared topic {topic}")
            sample = e["sample_time_us"]
            previous = self.last_sample.get(topic)
            expected = {
                "battery_status": 100000,
                "vehicle_status": 500000,
                "vehicle_land_detected": 1000000,
            }.get(topic, 10000)
            if previous is not None and sample - previous > 3 * expected:
                self.diag.write(
                    json.dumps(
                        dict(
                            kind="source_sample_gap",
                            topic=topic,
                            from_sample_us=previous,
                            to_sample_us=sample,
                            reason="source samples absent or latest-sample capture loss; physical cause not inferred",
                        )
                    )
                    + "\n"
                )
                self.diag.flush()
            self.last_sample[topic] = sample
            t = e["available_time_us"]
            if t < self.through:
                raise ValueError("available time regressed")
            e.update(
                schema=self.raw_schema,
                run_id=self.id,
                source="px4.uorb",
                instance=0,
                seq=self.seq[topic],
                clock_id="px4-boot",
            )
            e["field_quality"] = {
                k: (
                    "unavailable"
                    if v is None or (isinstance(v, list) and any(x is None for x in v))
                    else "valid"
                )
                for k, v in e["fields"].items()
            }
            self.seq[topic] += 1
            serialization_start = self.stamp()
            self.stream.write(json.dumps(e, allow_nan=False) + "\n")
            serialization_ns += self.stamp() - serialization_start
            self.count += 1
            self.through = t
        batch["transform_end_host_monotonic_ns"] = self.stamp()
        batch["stream_serialization_write_sum_ns"] = serialization_ns
        batch["last_event_index"] = self.count
        # Hold the newest timestamp back until a later timestamp is seen.
        self.flush(False, batch)

    def flush(self, closed, timing_batch=None):
        timing = dict(timing_batch or {})
        timing["stream_flush_start_host_monotonic_ns"] = self.stamp()
        self.stream.flush()
        timing["stream_flush_end_host_monotonic_ns"] = self.stamp()
        timing["stream_fsync_start_host_monotonic_ns"] = self.stamp()
        os.fsync(self.stream.fileno())
        timing["stream_fsync_end_host_monotonic_ns"] = self.stamp()
        w = {
            "schema": "drone.raw.watermark/2.0",
            "run_id": self.id,
            "clock_id": "px4-boot",
            "through_available_time_us": max(0, self.through if closed else self.through - 1),
            "recorded_events": self.count,
            "closed": closed,
        }
        timing["watermark_write_start_host_monotonic_ns"] = self.stamp()
        self.watermarks.write(json.dumps(w) + "\n")
        timing["watermark_write_end_host_monotonic_ns"] = self.stamp()
        timing["watermark_flush_start_host_monotonic_ns"] = self.stamp()
        self.watermarks.flush()
        timing["watermark_flush_end_host_monotonic_ns"] = self.stamp()
        timing["watermark_fsync_start_host_monotonic_ns"] = self.stamp()
        os.fsync(self.watermarks.fileno())
        timing["watermark_fsync_end_host_monotonic_ns"] = self.stamp()
        self.trace(
            dict(
                kind="watermark_commit",
                recorded_events=w["recorded_events"],
                through_available_time_us=w["through_available_time_us"],
                closed=closed,
                **timing,
            )
        )

    def release(self):
        # Called even if consume/flush fails; never emit a false closed watermark.
        for stream in (self.stream, self.watermarks, self.diag, self.timing):
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                pass

    def close(self):
        self.consume()
        if self.pending:
            self.diag.write(
                json.dumps({"kind": "incomplete_final_line", "bytes": len(self.pending)}) + "\n"
            )
        native_diag = self.run / "native_logs/capture.jsonl.diagnostics.jsonl"
        if native_diag.exists():
            self.diag.write(native_diag.read_text())
        path = self.run / "observations/metadata.json"
        metadata = json.loads(path.read_text())
        for binding in metadata["semantic_bindings"].values():
            count = self.seq[binding["topic"]]
            binding.update(availability="recorded" if count else "unavailable", event_count=count)
        dump(path, metadata)
        # Closed means all observation metadata is final for directory-tail readers.
        self.flush(True)
        self.stream.close()
        self.watermarks.close()
        self.diag.close()
        if self.timing is not None:
            self.timing.close()


def integrity(run):
    run = Path(run)
    dump(
        run / "integrity.json",
        {
            str(p.relative_to(run)): {"size": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(run.rglob("*"))
            if p.is_file() and p.name != "integrity.json"
        },
    )


def verify(run):
    run = Path(run)
    metadata = json.loads((run / "observations/metadata.json").read_text())
    events = []
    seq: dict[tuple[str, str, str, int], int] = {}
    errors = []
    generic = metadata.get("schema") == GENERIC_METADATA
    rotor_count = 4
    if generic:
        try:
            verify_generic_metadata(run, metadata)
            rotor_count = metadata["vehicle"]["rotor_count"]
        except (ValueError, KeyError, OSError, TypeError, jsonschema.ValidationError) as exc:
            errors.append("generic metadata/profile: " + str(exc))
    elif metadata.get("schema") != "drone.observation.metadata/2.0" or (
        metadata.get("ai_compatible") is False or "simulation_profile_manifest" in metadata
    ):
        errors.append("metadata contract/schema")
    raw_schema = GENERIC_RAW if generic else "drone.raw.event/2.0"
    for line in (run / "observations/streams.jsonl").read_text().splitlines():
        e = json.loads(line)
        topic = e["topic"]
        if e["schema"] != raw_schema or e["run_id"] != run.name:
            errors.append("identity/schema")
        if topic not in TOPICS:
            errors.append("undeclared topic")
            continue
        if set(e["fields"]) != set(TOPICS[topic][1]):
            errors.append("topic fields/leakage")
        if set(e["field_quality"]) != set(e["fields"]):
            errors.append("quality field mismatch")
        for name, value in e["fields"].items():
            length = {"output": rotor_count, "xyz": 3, "q": 4, "velocity": 3}.get(name)
            if length and (not isinstance(value, list) or len(value) != length):
                errors.append("array length")
            if e["field_quality"].get(name) not in [
                "valid",
                "unavailable",
                "invalid",
                "stale",
                "clipped",
            ]:
                errors.append("quality enum")
            if (value is None or isinstance(value, list) and None in value) and e[
                "field_quality"
            ].get(name) == "valid":
                errors.append("null marked valid")
        if e["sample_time_us"] > e["available_time_us"]:
            errors.append("sample after available")
        key = (e["clock_id"], e["source"], topic, e["instance"])
        if e["seq"] != seq.get(key, -1) + 1:
            errors.append("sequence gap")
        seq[key] = e["seq"]
        for v in e["fields"].values():
            values = v if isinstance(v, list) else [v]
            if any(x is not None and not math.isfinite(x) for x in values):
                errors.append("nonfinite")
        events.append(e)
    w = [json.loads(s) for s in (run / "observations/watermarks.jsonl").read_text().splitlines()]
    if not w or not w[-1]["closed"] or w[-1]["recorded_events"] != len(events):
        errors.append("final watermark")
    suffix_min = [float("inf")] * (len(events) + 1)
    for i in range(len(events) - 1, -1, -1):
        suffix_min[i] = min(suffix_min[i + 1], events[i]["available_time_us"])
    for point in w:
        n = point["recorded_events"]
        if n > len(events):
            errors.append("watermark exceeds events")
        if n <= len(events) and suffix_min[n] <= point["through_available_time_us"]:
            errors.append("early watermark")
    manifest = json.loads((run / "integrity.json").read_text())
    for p, v in manifest.items():
        if sha(run / p) != v["sha256"] or (run / p).stat().st_size != v["size"]:
            errors.append("hash " + p)
    expected = {b["topic"] for b in metadata["semantic_bindings"].values()}
    missing = sorted(expected - {e["topic"] for e in events})
    if missing:
        errors.append("missing topics " + ",".join(missing))
    result = {
        "status": "passed" if not errors else "failed",
        "errors": sorted(set(errors)),
        "event_count": len(events),
        "topics": {t: sum(e["topic"] == t for e in events) for t in TOPICS},
        "C9": "not_supported_generic_simulator_only" if generic else "deferred_ai_integration",
    }
    if generic:
        result["ai_compatible"] = False
    return result
