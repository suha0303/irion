import pytest

from capstone_ai.cli import live_evidence_verified


@pytest.mark.parametrize("status", ["mission", "armed", "airborne", "landing"])
def test_active_producer_requires_growth_and_close(status):
    assert live_evidence_verified("live", status, True, True)
    assert not live_evidence_verified("live", status, False, True)
    assert not live_evidence_verified("live", status, True, False)
    assert not live_evidence_verified("replay", status, True, True)


@pytest.mark.parametrize("status", ["completed", "failed", "finalizing", "disarmed", "unknown"])
def test_terminal_attachment_cannot_become_live(status):
    assert not live_evidence_verified("live", status, True, True)
