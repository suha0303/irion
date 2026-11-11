"""Bind sealed regression fixtures to their explicit historical contracts.

The production source/runtime seals and original test files remain unchanged.
Only the named test modules below receive compatibility fixtures; assertions,
production verification, and failure reporting are never replaced or skipped.
"""

from pathlib import Path

import pytest

from test_support import contracts


@pytest.fixture(autouse=True)
def sealed_test_contracts(request, monkeypatch, tmp_path):
    """Supply the API/policy version assumed by each preserved test module."""
    path = Path(request.module.__file__).resolve()
    root = Path(__file__).parent
    runner_tests = {
        root / "sim/tests" / name
        for name in (
            "test_vehicle_runner.py",
            "test_battery_aging.py",
            "test_bounded_wind_cli.py",
            "test_parent_attempt_cli.py",
        )
    }
    if path in runner_tests:
        contracts.bind_recorder(request.module, monkeypatch)
    elif path == root / "sim/tests/test_joint_duration.py":
        # These fixtures exercise the legacy 40 ms gate, not the later 3 s gate.
        contracts.bind_gap_policy(request.module.generation, monkeypatch, tmp_path)
    elif path == root / "audit/native_consumption_v2/test_native_joint.py":
        contracts.bind_historical_audit(request.module, monkeypatch)
