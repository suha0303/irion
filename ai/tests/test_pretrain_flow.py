"""Finite fixture flights only: duration ordering, bounded overlap and failure drain."""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from pretrain import cli, collection, common, display, flow, streaming


def test_real_300_manifest_groups_same_duration_without_changing_split_or_seed():
    rows = collection.cases(202680000, "balanced300")["cases"]
    original = copy.deepcopy(rows)
    groups = flow.homogeneous_batches(rows, 16)
    assert rows == original
    assert {g[0]["mission_seconds"] for g in groups} == {900, 480, 180}
    assert all(len({c["mission_seconds"] for c in g}) == 1 and len(g) <= 16 for g in groups)
    assert [g[0]["mission_seconds"] for g in groups] == sorted(
        [g[0]["mission_seconds"] for g in groups], reverse=True
    )
    assert sorted(c["case_id"] for g in groups for c in g) == sorted(c["case_id"] for c in rows)
    for seconds in (900, 480, 180):
        assert [c for g in groups for c in g if c["mission_seconds"] == seconds] == [
            c for c in rows if c["mission_seconds"] == seconds
        ]


class FixtureRunner:
    def __init__(self, task):
        self.task = task
        self.config = dict(flight_workers=2)
        self.state = dict(status="RUNNING", completed=[], stages=[], active=[])
        self.state_path = task / "operator/state.json"
        self.peaks = {}
        self.flights = []
        self.second_flight = threading.Event()

    def update(self, **fields):
        self.state.update(fields)
        common.write(self.state_path, self.state)

    def begin(self, stage, **fields):
        self.update(stage=stage, **fields)

    def finish(self, stage, result):
        self.state["completed"].append(stage)
        self.update()

    def flight(self, phase, workers, cases, label):
        self.flights.append(label)
        if len(self.flights) == 2:
            self.second_flight.set()


def test_next_flight_overlaps_cpu_but_third_group_waits_for_first_retirement(tmp_path, monkeypatch):
    r = FixtureRunner(tmp_path)
    cpu_started, release = threading.Event(), threading.Event()
    groups = [[dict(case_id=str(i))] for i in range(3)]
    monkeypatch.setattr(streaming, "fragment_freeze", lambda *a: (tmp_path, [dict(run_id="a")]))

    def process(rows, label, folder):
        if label.endswith("001"):
            cpu_started.set()
            assert release.wait(5)
        return rows

    pipeline = SimpleNamespace(r=r, process_chunk=process)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(flow.collect_overlapping, r, pipeline, [sum(groups, [])], [groups])
        try:
            assert cpu_started.wait(5) and r.second_flight.wait(5)
            assert len(r.flights) == 2  # CPU1 held; cannot admit a third raw group.
            assert common.read(tmp_path / "operator/processing-state.json")["stage"].endswith(
                "freeze"
            )
        finally:
            release.set()
        outcomes = future.result(timeout=5)
    assert len(outcomes) == len(r.flights) == 3
    assert pipeline.r is r and r.state["bounded_raw_runs"] == 4
    assert (
        common.read(tmp_path / "operator/processing-state.json")["status"] == "PROCESSING_COMPLETE"
    )
    # Main and CPU each write their own state file; no concurrent edits of common state.
    assert r.state.get("completed") == []


def test_cpu_failure_preserves_first_error_and_stops_new_groups(tmp_path, monkeypatch):
    r = FixtureRunner(tmp_path)
    groups = [[dict(case_id=str(i))] for i in range(4)]
    preserved = tmp_path / "pending-original.txt"
    preserved.write_text("keep")
    monkeypatch.setattr(streaming, "fragment_freeze", lambda *a: (tmp_path, [dict(run_id="a")]))

    def fail(rows, label, folder):
        assert r.second_flight.wait(5)
        raise ValueError("original capsule failure")

    pipeline = SimpleNamespace(r=r, process_chunk=fail)
    with pytest.raises(ValueError, match="original capsule failure"):
        flow.collect_overlapping(r, pipeline, [sum(groups, [])], [groups])
    signal = common.read(tmp_path / "operator/processing-stop.json")
    assert signal["reason"] == "ValueError: original capsule failure"
    assert len(r.flights) == 2 and preserved.read_text() == "keep" and pipeline.r is r


def test_cpu_view_is_read_only_and_stays_visible_above_16_flight_cards(tmp_path, monkeypatch):
    common.write(
        tmp_path / "config.json",
        dict(
            mode="collect",
            collection_profile="balanced300",
            collection_run_count=300,
            flight_workers=16,
            export_workers=4,
            io_workers=4,
            workers=4,
            audit_workers=4,
            gates=False,
            pipeline_overlap=True,
            storage_profile="stream_compact",
            use_available_ram=True,
            ignore_memory_estimate=True,
        ),
    )
    common.write(
        tmp_path / "operator/state.json",
        dict(
            status="RUNNING",
            pid=1,
            pid_start_ticks=123,
            pipeline_overlap=True,
            collection_ledger=str(tmp_path / "group/execution/ledger.json"),
            completed=["flight1"],
            active=[],
            stage="flight2",
            total=16,
            done=0,
        ),
    )
    cpu_path = tmp_path / "operator/processing-state.json"
    common.write(
        cpu_path,
        dict(
            status="RUNNING",
            stage="preprocess",
            done=3,
            total=16,
            completed=["extract"],
            stream_completed=2,
            active=[],
        ),
    )
    before = common.sha(tmp_path / "operator/state.json")
    monkeypatch.setattr(cli, "identity", lambda pid: 123)
    data = SimpleNamespace(
        snapshot=lambda plan: dict(
            total=16,
            archived=0,
            failed=0,
            workers=[dict(instance=i, scenario="hover_normal", status="FLYING") for i in range(16)],
            reason=[],
            eta_s=None,
        )
    )
    monkeypatch.setattr(cli, "load", lambda *a: data)
    text = cli.render(tmp_path)
    assert "병행 후처리" in text and "3/16" in text and "최대32run" in text
    assert "RSS8GB / OS9GB 유지" not in text
    screen = display.screen(text, 100, 23)
    assert "학습 전 통합 콘솔" in screen and "병행 후처리" in screen
    assert "15 호버" in screen and "화면이 작습니다" not in screen
    assert common.sha(tmp_path / "operator/state.json") == before
