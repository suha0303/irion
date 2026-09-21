import argparse
import copy
import json
from pathlib import Path

from capstone_sim.recording import TOPICS

parser = argparse.ArgumentParser(description="Generate the X500 data-contract schemas")
parser.add_argument(
    "--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "contracts"
)
root = parser.parse_args().output_dir
root.mkdir(parents=True, exist_ok=True)
nonempty = {"type": "string", "minLength": 1}
integer = {"type": "integer", "minimum": 0}
scalar = {"type": ["number", "null"]}
value = {"anyOf": [scalar, {"type": "array", "items": scalar}]}
base = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
}
event = dict(
    base,
    properties={
        "schema": {"const": "drone.raw.event/2.0"},
        "run_id": nonempty,
        "source": nonempty,
        "topic": nonempty,
        "instance": integer,
        "seq": integer,
        "sample_time_us": integer,
        "available_time_us": integer,
        "clock_id": nonempty,
        "fields": {"type": "object", "minProperties": 1, "additionalProperties": value},
        "field_quality": {
            "type": "object",
            "additionalProperties": {
                "enum": ["valid", "unavailable", "invalid", "stale", "clipped"]
            },
        },
    },
)
event["required"] = list(event["properties"])
event["properties"]["topic"] = {"enum": list(TOPICS)}
event["allOf"] = []
lengths = {"output": 4, "xyz": 3, "q": 4, "velocity": 3}
for topic, (_, fields, _) in TOPICS.items():
    props = {
        k: (
            {"type": "array", "minItems": lengths[k], "maxItems": lengths[k], "items": scalar}
            if k in lengths
            else scalar
        )
        for k in fields
    }
    event["allOf"].append(
        {
            "if": {"properties": {"topic": {"const": topic}}},
            "then": {
                "properties": {
                    "fields": {
                        "type": "object",
                        "properties": props,
                        "required": list(fields),
                        "additionalProperties": False,
                    },
                    "field_quality": {
                        "required": list(fields),
                        "propertyNames": {"enum": list(fields)},
                    },
                }
            },
        }
    )
watermark = dict(
    base,
    properties={
        "schema": {"const": "drone.raw.watermark/2.0"},
        "run_id": nonempty,
        "clock_id": nonempty,
        "through_available_time_us": integer,
        "recorded_events": integer,
        "closed": {"type": "boolean"},
    },
)
watermark["required"] = list(watermark["properties"])
evaluation = dict(
    base,
    properties={
        "schema": {"const": "drone.evaluation.event/2.1"},
        "run_id": nonempty,
        "event_id": nonempty,
        "kind": nonempty,
        "target": nonempty,
        "onset_kind": {"enum": ["pre_existing", "in_flight_step", "in_flight_ramp", "unknown"]},
        "physical_onset_time_us": {"type": ["integer", "null"], "minimum": 0},
        "requested_time_us": integer,
        "applied_time_us": integer,
        "end_time_us": {"type": ["integer", "null"], "minimum": 0},
        "profile_revision": integer,
        "parameters": {"type": "object"},
    },
)
evaluation["required"] = list(evaluation["properties"])
evaluation["properties"].update(
    ramp_duration_us=integer,
    degradation_boundary_time_us={"type": ["integer", "null"], "minimum": 0},
    degradation_boundary_definition=nonempty,
)
for name, s in [
    ("raw-event", event),
    ("watermark", watermark),
    ("evaluation-event", evaluation),
]:
    (root / (name + ".schema.json")).write_text(json.dumps(s, indent=2) + "\n")
run_schema = dict(
    base,
    properties={
        "run_id": nonempty,
        "parent_attempt_id": {"type": ["string", "null"]},
        "origin": {"enum": ["px4_sitl", "synthetic_contract_fixture"]},
        "status": nonempty,
        "termination_reason": nonempty,
        "start_utc": nonempty,
        "end_utc": {"type": ["string", "null"]},
        "clock": {"type": "object"},
        "observation_metadata": nonempty,
        "source_lock": nonempty,
        "native_logs": {"type": "array", "items": nonempty},
        "process_exit_codes": {"type": "array", "items": {"type": "integer"}},
        "wall_duration_s": {"type": "number", "minimum": 0},
        "owned_process_groups": {"type": "object"},
    },
)
run_schema["required"] = [
    "run_id",
    "parent_attempt_id",
    "origin",
    "status",
    "termination_reason",
    "start_utc",
    "end_utc",
    "clock",
    "observation_metadata",
    "source_lock",
    "native_logs",
]
# Runtime diagnostics are optional; the common identity contract stays required.
failure = dict(
    base, properties={"stage": nonempty, "reason": {"type": "string"}}, required=["stage", "reason"]
)
run_schema["properties"].update(
    configuration=nonempty,
    instance=integer,
    gz_partition=nonempty,
    mavlink_source_checked={"type": "boolean"},
    wait_policy={"type": "object"},
    progress={"type": "object"},
    process_shutdown={"type": "array", "items": {"type": "object"}},
    px4_socket_cleanup=nonempty,
    capture_health_failure={"type": "object"},
    primary_failure=failure,
    secondary_failures={"type": "array", "items": failure},
    clock_stall_evidence=dict(
        base,
        properties={
            "phase": nonempty,
            "px4_boot_s": {"type": "number", "minimum": 0},
            "stalled_wall_s": {"type": "number", "minimum": 0},
            "limit_wall_s": {"type": "number", "exclusiveMinimum": 0},
            "last_clock_advance_wall": {"type": "number"},
        },
        required=[
            "phase",
            "px4_boot_s",
            "stalled_wall_s",
            "limit_wall_s",
            "last_clock_advance_wall",
        ],
    ),
)
metadata_schema = dict(
    base,
    properties={
        "schema": {"const": "drone.observation.metadata/2.0"},
        "run_id": nonempty,
        "origin": {"enum": ["px4_sitl", "synthetic_contract_fixture"]},
        "clock": {"type": "object"},
        "frames": {"type": "object"},
        "vehicle": {"type": "object"},
        "actuation": {"type": "object"},
        "semantic_bindings": {"type": "object", "minProperties": 9},
        "conversion_history": {"type": "array"},
    },
)
metadata_schema["required"] = list(metadata_schema["properties"])

# A separate simulator contract preserves the fixed four-motor AI contract above.
generic_event = copy.deepcopy(event)
generic_event = {
    "$schema": generic_event.pop("$schema"),
    "description": "Simulator-only vertical multirotor observations with 4..8 motor requests. The independent verifier checks the exact count against generic metadata. This is not the X500 AI input contract.",
    **generic_event,
}
generic_event["properties"]["schema"]["const"] = "drone.raw.event/3.0"
for rule in generic_event["allOf"]:
    if rule["if"]["properties"]["topic"]["const"] == "actuator_outputs":
        rule["then"]["properties"]["fields"]["properties"]["output"].update(minItems=4, maxItems=8)

generic_metadata = copy.deepcopy(metadata_schema)
generic_metadata = {
    "$schema": generic_metadata.pop("$schema"),
    "description": "Simulator-only vehicle and battery identity bound to a run-local simulation profile manifest. ai_compatible must remain false; no existing X500 model support or C9 connection is implied.",
    **generic_metadata,
}
generic_metadata["properties"]["schema"]["const"] = "drone.observation.metadata/3.0"
generic_metadata["properties"]["ai_compatible"] = {"const": False}
generic_metadata["properties"]["simulation_profile_manifest"] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "path": {"const": "reference/vehicle-profile.json"},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
    "required": ["path", "sha256"],
}
generic_vehicle_properties = {
    "profile_id": dict(nonempty, **{"not": {"const": "x500_reference"}}),
    "profile_revision": nonempty,
    "rotor_count": {"type": "integer", "minimum": 4, "maximum": 8},
    "motor_order": {
        "type": "array",
        "minItems": 4,
        "maxItems": 8,
        "uniqueItems": True,
        "items": nonempty,
    },
    "nominal_series_cells": {"type": "integer", "minimum": 1},
    "nominal_parallel_cells": {"type": "integer", "minimum": 1},
    "nominal_capacity_ah": {"type": "number", "exclusiveMinimum": 0},
    "battery_profile_id": nonempty,
}
generic_metadata["properties"]["vehicle"] = {
    "type": "object",
    "additionalProperties": False,
    "properties": generic_vehicle_properties,
    "required": list(generic_vehicle_properties),
}
generic_metadata["required"] += ["ai_compatible", "simulation_profile_manifest"]

for name, schema in [
    ("run", run_schema),
    ("metadata", metadata_schema),
    ("generic-raw-event", generic_event),
    ("generic-metadata", generic_metadata),
]:
    (root / (name + ".schema.json")).write_text(json.dumps(schema, indent=2) + "\n")
