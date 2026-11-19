"""Check isolated worker recovery and prepared-data admission."""

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from pretrain import mavlink_prepare as batch


def test_native_worker_failure_retries_and_published_flight_is_reusable(
    tmp_path, monkeypatch
):
    item = dict(
        batch="batch", flight_id="flight-0001", run_id="example", seed=19
    )
    identity = dict(plan_sha256="plan", python="python")
    (tmp_path / "batch").mkdir()
    (tmp_path / ".attempts").mkdir()
    calls = []

    def worker(_command, *, input, **_options):
        job = json.loads(input)
        calls.append(job)
        if len(calls) == 1:
            return SimpleNamespace(returncode=-11, stdout="", stderr="SIGSEGV")
        target = Path(job["final"])
        target.mkdir()
        receipt = target / "receipt.json"
        receipt.write_text('{"raw_sha256":"original"}')
        npz = target / "run.npz"
        npz.write_bytes(b"prepared")
        batch.write(
            target / "admission.json",
            dict(
                **item,
                plan_sha256="plan",
                raw_sha256="original",
                npz_sha256=batch.digest(npz),
                receipt_sha256=batch.digest(receipt),
            ),
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(batch.subprocess, "run", worker)
    observed, exhausted = batch.run_flight(
        item, tmp_path, identity, {}, 0, Event()
    )
    assert not exhausted
    assert observed == batch.prepared(tmp_path, item, "plan")
    assert len(calls) == 2
    (failure,) = (tmp_path / ".attempts").glob("*/failure.json")
    assert json.loads(failure.read_text())["exit_code"] == -11

    mutated = tmp_path / "batch" / "flight-0001" / "run.npz"
    mutated.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="failed verification"):
        batch.prepared(tmp_path, item, "plan")
    assert mutated.read_bytes() == b"damaged"
