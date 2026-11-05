"""Reproduce only the explicit Collection header; never rewrite the legacy profile."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim/src"))

from capstone_sim.profile import header_text, load_profile  # noqa: E402


def render():
    directory = ROOT / "profiles/collection"
    profile_path = directory / "vehicle.yaml"
    vehicle = load_profile(profile_path)
    battery = json.loads((directory / "battery.json").read_text())
    return header_text(vehicle, hashlib.sha256(profile_path.read_bytes()).hexdigest(), battery)


if __name__ == "__main__":
    (ROOT / "profiles/collection/vehicle_generated.hpp").write_text(render())
