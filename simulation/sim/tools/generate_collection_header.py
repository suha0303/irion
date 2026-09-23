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
    values = {
        "min_soc": vehicle["validity"]["min_soc"],
        "min_pack_voltage_v": vehicle["validity"]["min_pack_voltage_v"],
        "battery_heat_capacity": battery["thermal"]["heat_capacity_j_k"],
        "battery_cooling": battery["thermal"]["cooling_w_k"],
        "interconnect_resistance": battery["interconnect_resistance_ohm"],
    }
    text = header_text(vehicle, hashlib.sha256(profile_path.read_bytes()).hexdigest())
    text = text.replace("sim/tools/generate_profile.py; edit profiles/x500_reference.yaml",
                        "sim/tools/generate_collection_header.py; Collection profiles/collection")
    extra = "\n".join(f"inline constexpr double {key} = {value};" for key, value in values.items())
    return text.replace("\n}\n", "\n" + extra + "\n}\n")


if __name__ == "__main__":
    (ROOT / "profiles/collection/vehicle_generated.hpp").write_text(render())
