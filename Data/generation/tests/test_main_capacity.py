"""Capacity estimates are distinct from the unchanged actual reserve and scratch guard."""

from types import SimpleNamespace

import pytest

from pretrain import common, pilot45
from pretrain import main_collection as main


@pytest.mark.parametrize("fresh", [False, True])
def test_fresh_preflight_reports_estimate_shortfall_without_disabling_actual_guard(
    monkeypatch, tmp_path, fresh
):
    cfg = dict(
        source_sha256={},
        producer={},
        fresh_start={} if not fresh else {"fixture": True},
    )
    common.write(tmp_path / "config.json", cfg)
    (tmp_path / "config.sha256").write_text(
        common.sha(tmp_path / "config.json")
    )
    common.write(
        tmp_path / "budget.json",
        dict(
            batch_scratch_bytes=10_000_000_000,
            required_initial_free_bytes=100_000_000_000,
        ),
    )
    monkeypatch.setattr(main, "validate", lambda _: (cfg, dict(cases=[])))
    monkeypatch.setattr(
        main.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="")
    )
    monkeypatch.setattr(main, "dependencies", lambda _: {})
    cfg["dependency_handoff"] = "fixture"
    monkeypatch.setattr(main.ai19, "producer_binding", lambda: {})
    checks = []

    def resources(config, *, scratch=0):
        checks.append(scratch)
        return dict(disk_free_bytes=90_000_000_000)

    monkeypatch.setattr(main, "resources", resources)
    monkeypatch.setattr(main.native_flight, "environment_check", lambda: None)
    monkeypatch.setattr(main.pilot_parallel, "released", lambda: None)
    if fresh:
        result = main.preflight(tmp_path)
        assert result["status"] == "PASSED_NO_FLIGHT" and result["warnings"]
        assert (
            "100.0GB" in result["warnings"][0]
            and "90.0GB" in result["warnings"][0]
        )
    else:
        with pytest.raises(pilot45.Blocked, match="Planning upper"):
            main.preflight(tmp_path)
    assert 10_000_000_000 in checks


def test_actual_reserve_plus_bounded_scratch_still_blocks(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(main, "guard", lambda *a: tmp_path)
    monkeypatch.setattr(
        main,
        "collector",
        lambda: SimpleNamespace(
            memory_snapshot=lambda: dict(available_bytes=2_000_000_000)
        ),
    )
    monkeypatch.setattr(
        main.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            free=70_000_000_000
            if path == main.ROOT / "simulation/runs"
            else 55_000_000_000
        ),
    )
    with pytest.raises(pilot45.Blocked, match="Disk50GB reserve"):
        main.resources(
            dict(external_storage=str(tmp_path), storage_policy={}),
            scratch=10_000_000_000,
        )
