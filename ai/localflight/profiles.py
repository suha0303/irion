"""Versioned real-aircraft identity, without inventing missing nameplate values."""

import hashlib
import json
import math
import re
from pathlib import Path

from .catalog import read_json

MOTOR_POSITIONS = ("front_right", "front_left", "rear_right", "rear_left")
ANALYSIS_FIELDS = (
    "autopilot",
    "rotor_count",
    "series_cells",
    "capacity_ah",
    "motor_channels",
    "output_min",
    "output_max",
)
RECORD_FIELDS = ("name", "firmware", "battery_id", "mapping_evidence", "notes")


def validate(profile):
    allowed = {
        "schema",
        "id",
        "name",
        "autopilot",
        "firmware",
        "rotor_count",
        "series_cells",
        "capacity_ah",
        "battery_id",
        "motor_channels",
        "output_min",
        "output_max",
        "mapping_evidence",
        "notes",
        "provenance",
        "imu_instance",
        "battery_instance",
        "actuator_instance",
    }
    if (
        set(profile) - allowed
        or profile.get("schema") != "drone.real-aircraft/1"
    ):
        raise ValueError("Unknown aircraft profile schema or fields")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", profile.get("id", "")):
        raise ValueError(
            "Aircraft ID must contain 1..64 letters, digits, underscores or hyphens"
        )
    if (
        not isinstance(profile.get("name"), str)
        or not 1 <= len(profile["name"]) <= 120
    ):
        raise ValueError("Aircraft name required")
    if profile.get("autopilot") not in {"px4", "ardupilot", "unknown"}:
        raise ValueError("Unsupported autopilot")
    for key, low, high in (("rotor_count", 1, 16), ("series_cells", 1, 32)):
        v = profile.get(key)
        if v is not None and (type(v) is not int or not low <= v <= high):
            raise ValueError(f"Invalid {key}")
    for key in ("imu_instance", "battery_instance", "actuator_instance"):
        value = profile.get(key)
        if value is not None and (
            type(value) is not int or not 0 <= value <= 15
        ):
            raise ValueError(f"Invalid {key}")
    for key in ("capacity_ah", "output_min", "output_max"):
        v = profile.get(key)
        if v is not None and (
            type(v) not in (float, int) or not math.isfinite(v) or v < 0
        ):
            raise ValueError(f"Invalid {key}")
    if profile.get("capacity_ah") == 0:
        raise ValueError("Capacity must be positive")
    lo, hi = profile.get("output_min"), profile.get("output_max")
    if lo is not None and hi is not None and hi <= lo:
        raise ValueError("Motor output maximum must exceed minimum")
    channels = profile.get("motor_channels", {})
    if not isinstance(channels, dict) or set(channels) - set(MOTOR_POSITIONS):
        raise ValueError("Unknown motor position")
    if any(type(v) is not int or not 1 <= v <= 16 for v in channels.values()):
        raise ValueError("Motor channels are one-based integers 1..16")
    if len(set(channels.values())) != len(channels):
        raise ValueError("Motor channels must be distinct")
    for key in ("firmware", "battery_id", "mapping_evidence", "notes"):
        if key in profile and (
            not isinstance(profile[key], str) or len(profile[key]) > 2000
        ):
            raise ValueError(f"Invalid {key}")
    if not isinstance(profile.get("provenance", {}), dict):
        raise ValueError("Profile provenance must be an object")
    json.dumps(profile, allow_nan=False)
    return profile


def eligibility(profile):
    validate(profile)
    missing = [
        k
        for k in ("rotor_count", "series_cells", "capacity_ah")
        if not profile.get(k)
    ]
    if profile["autopilot"] == "unknown":
        missing.append("autopilot")
    if set(profile.get("motor_channels", {})) != set(MOTOR_POSITIONS):
        missing.append("motor_channels")
    for key in ("output_min", "output_max"):
        if profile.get(key) is None:
            missing.append(key)
    unsupported = profile.get("rotor_count") not in (None, 4)
    instance_missing = [
        k
        for k in ("imu_instance", "battery_instance", "actuator_instance")
        if profile.get(k) is None
    ]
    timeline_missing = [
        k for k in missing if k not in ("series_cells", "capacity_ah")
    ] + instance_missing
    return dict(
        profile_complete=not missing,
        missing_fields=missing,
        log_missing_fields=missing + instance_missing,
        timeline_missing_fields=timeline_missing,
        timeline_ready=not timeline_missing and not unsupported,
        analysis_fields=list(ANALYSIS_FIELDS),
        record_fields=list(RECORD_FIELDS),
        model_structure_compatible=not unsupported
        and profile.get("rotor_count") == 4,
        real_aircraft_validated=False,
        reasons=(
            ["Current model has four fixed motor inputs"] if unsupported else []
        ),
    )


def save_profile(root, profile):
    # Identity is application bookkeeping, not a required manufacturer specification.
    if not profile.get("id"):
        import uuid

        profile = dict(profile, id="aircraft-" + uuid.uuid4().hex[:12])
    if not profile.get("name"):
        profile = dict(profile, name=profile["id"])
    validate(profile)
    encoded = json.dumps(
        profile, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()
    version = hashlib.sha256(encoded).hexdigest()
    folder = Path(root) / "aircraft" / profile["id"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (version + ".json")
    try:
        with path.open("xb") as f:
            f.write(encoded + b"\n")
    except FileExistsError as error:
        if read_json(path) != profile:
            raise ValueError("Immutable profile version conflict") from error
    return dict(
        profile=profile, version=version, eligibility=eligibility(profile)
    )


def list_profiles(root):
    """Return each stored version with eligibility from the same JSON read."""
    records = []
    for path in sorted((Path(root) / "aircraft").glob("*/*.json")):
        profile = read_json(path)
        records.append(
            dict(
                profile=profile,
                version=path.stem,
                eligibility=eligibility(profile),
            )
        )
    return records
