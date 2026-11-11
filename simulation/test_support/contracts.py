"""Test-only adapters for sealed fixtures; never imported by a flight runner."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(__file__).parent / "frozen_joint_v1"


def recorder_type(original):
    """Keep the old fixture's behavior and expose the current recorder option."""

    class RecorderFixture(original):
        def __init__(self, *args, sync_interval_s=0.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.sync_interval_s = sync_interval_s

    return RecorderFixture


def bind_recorder(test_module, monkeypatch):
    # Some preserved modules import the helper by its standalone filename.
    modules = [test_module, sys.modules.get("test_vehicle_runner")]
    for module in set(modules) - {None}:
        original = getattr(module, "RecorderFixture", None)
        if original is not None:
            monkeypatch.setattr(module, "RecorderFixture", recorder_type(original))


def bind_gap_policy(generation, monkeypatch, folder, limit=40_000):
    """Use a temporary, explicit acceptance policy for a synthetic test run."""
    policy = json.loads(generation.POLICY.read_text(encoding="utf-8"))
    policy["acceptance"]["native_consumption_max_gap_us"] = limit
    path = folder / "fixture-policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setattr(generation, "POLICY", path)


def historical_generator():
    """Load the exact generator/policy named by the frozen audit manifest."""
    manifest = json.loads(
        (ROOT / "audit/native_consumption_v2/manifest.json").read_text(encoding="utf-8")
    )
    for name, original in (
        ("joint_phase.py", "sim/src/capstone_sim/joint_phase.py"),
        ("policy.json", "profiles/collection/joint-phase-policy.json"),
    ):
        digest = hashlib.sha256((FROZEN / name).read_bytes()).hexdigest()
        if digest != manifest["generation_dependencies"][original]:
            raise ValueError("Historical audit dependency changed: " + name)
    spec = importlib.util.spec_from_file_location(
        "capstone_sim.historical_joint_phase_v1", FROZEN / "joint_phase.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.POLICY = FROZEN / "policy.json"
    return module


def bind_historical_audit(test_module, monkeypatch):
    generation = historical_generator()
    monkeypatch.setattr(test_module, "generation", generation)
    monkeypatch.setattr(test_module.audit, "POLICY", generation.POLICY)
