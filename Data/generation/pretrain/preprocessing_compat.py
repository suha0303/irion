"""Validate renamed metadata against the unchanged, hash-locked preprocessing package."""

import copy
import re

CURRENT_BATTERY_ID = "vtc6_4s4p"
SEALED_BATTERY_ID = "vtc6_4s4p_r3_long15"
ALIAS = dict(
    schema="drone.metadata-validation-alias/1",
    field="vehicle.battery_profile_id",
    original=CURRENT_BATTERY_ID,
    sealed_validation=SEALED_BATTERY_ID,
    validation_only=True,
    nominal_series_cells=4,
    nominal_parallel_cells=4,
    nominal_capacity_ah=12,
)


def make_preprocessor(metadata, config, metadata_sha256):
    """Return an engine and optional validation receipt; never rewrite raw input bytes."""
    from capstone_ai.preprocessing import Preprocessor

    vehicle = metadata.get("vehicle", {})
    validation = metadata
    alias = None
    if vehicle.get("battery_profile_id") == CURRENT_BATTERY_ID:
        if not re.fullmatch(r"[0-9a-f]{64}", metadata_sha256):
            raise ValueError("Original metadata digest required for validation alias")
        expected = dict(
            profile_id="x500_reference",
            profile_revision="vtc6-4s4p",
            rotor_count=4,
            nominal_series_cells=4,
            nominal_parallel_cells=4,
            nominal_capacity_ah=12,
        )
        if metadata.get("schema") != "drone.observation.metadata/2.0" or any(
            vehicle.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Renamed collection battery metadata/nameplate mismatch")
        validation = copy.deepcopy(metadata)
        validation["vehicle"]["battery_profile_id"] = SEALED_BATTERY_ID
        alias = dict(ALIAS, metadata_sha256=metadata_sha256)
    # All original mapping, frames, bindings, units and nameplate checks still run.
    engine = Preprocessor(validation, config)
    # The name alias is needed only by the sealed constructor. Event processing,
    # checkpoint identity and output inventory retain the original metadata.
    engine.metadata = metadata
    return engine, alias


def verify_metadata_alias(summary):
    """Audit a declared alias without reading raw/truth data or changing its identity."""
    alias = summary.get("evidence", {}).get("metadata_validation_alias")
    if alias is not None:
        digest = summary["input_hashes"]["observations/metadata.json"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or alias != dict(ALIAS, metadata_sha256=digest):
            raise ValueError("Metadata validation alias/input identity mismatch")
