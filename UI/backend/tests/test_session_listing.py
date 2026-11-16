"""Keep completed-session listings independent of full journal replay."""

import json

import pytest
from test_pipeline import battery

from arion.contracts import SourceConfig
from arion.store_v2 import StoreV2


def finished_session(store):
    run = store.create_session(
        SourceConfig(kind="udp", profile_id="arion_cmu_v1")
    )
    run.append_raw(battery(0))
    run.append_raw(battery(100))
    store.finish_session(run.run_id)
    return run


def test_finished_raw_listing_uses_saved_summary(tmp_path, monkeypatch):
    store = StoreV2(tmp_path)
    run = finished_session(store)
    saved = json.loads(
        (tmp_path / "records" / run.run_id / "manifest.json").read_text()
    )

    def unexpected_replay(*_args):
        pytest.fail(
            "Listing a finished raw session must not replay its journal"
        )

    monkeypatch.setattr(store, "get", unexpected_replay)
    listed = store.list()
    assert listed == [
        dict(
            saved,
            created_at=store.catalog["sessions"][run.run_id]["created_at"],
        )
    ]


@pytest.mark.parametrize("summary", ["missing", "mismatched", "corrupt"])
def test_legacy_or_invalid_summary_falls_back(tmp_path, summary):
    store = StoreV2(tmp_path)
    run = finished_session(store)
    path = tmp_path / "records" / run.run_id / "manifest.json"
    if summary == "missing":
        path.unlink()
    elif summary == "corrupt":
        path.write_text("{")
    else:
        data = json.loads(path.read_text())
        data["input_sha256"] = "wrong"
        path.write_text(json.dumps(data))
    store._journal_cache.cache_clear()
    listed = store.list()
    assert listed[0]["accepted_records"] == 2
    assert listed[0]["input_sha256"] == run.sha256


def test_active_listing_updates_without_reusing_saved_summary(tmp_path):
    store = StoreV2(tmp_path)
    run = store.create_session(
        SourceConfig(kind="udp", profile_id="arion_cmu_v1")
    )
    try:
        run.append_raw(battery(0))
        assert store.list()[0]["accepted_records"] == 1
        run.append_raw(battery(100))
        listed = store.list()[0]
        assert listed["accepted_records"] == 2
        assert listed["source_kind"] == "live"
        assert listed["session"]["recording"]
    finally:
        store.finish_session(run.run_id)
