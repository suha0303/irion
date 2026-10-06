"""The display must not turn incomplete or failed records into success."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "x500_data", Path(__file__).parents[1] / "x500_data.py"
)
display = importlib.util.module_from_spec(spec)
spec.loader.exec_module(display)


class DisplayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = self.root / "plan.json"
        self.plan.write_text(
            json.dumps({"campaign_id": "test", "execution_dir": str(self.root), "attempts_max": 3})
        )

    def ledger(self, state):
        (self.root / "ledger.json").write_text(json.dumps(state))

    def test_missing_is_not_running(self):
        state = display.snapshot(self.plan)
        self.assertEqual(state["status"], "NOT_STARTED_OR_NO_LEDGER")
        self.assertIsNone(state["percent"])

    def test_failure_and_source_drift_not_counted(self):
        self.ledger(
            {
                "status": "FAILED_DRAINED_STOPPED",
                "cases": {
                    "a": {"status": "ARCHIVED_PENDING_QUALITY"},
                    "b": {"status": "PRESERVED_SOURCE_DRIFT"},
                    "c": {"status": "FAILED_PRESERVED"},
                },
            }
        )
        state = display.snapshot(self.plan)
        self.assertEqual(state["archived"], 1)
        self.assertLess(state["percent"], 34)
        self.assertIn("품질 승인·전처리·학습 완료가 아닙니다", display.render(state))

    def test_no_path_traversal(self):
        self.ledger({"status": "RUNNING", "cases": {"../outside": {"status": "ARCHIVING"}}})
        self.assertEqual(display.snapshot(self.plan)["workers"], [])

    def test_completed_collection_still_pending_quality(self):
        self.ledger(
            {
                "status": "COLLECTED_PENDING_QUALITY_STOPPED",
                "cases": {str(i): {"status": "ARCHIVED_PENDING_QUALITY"} for i in range(3)},
            }
        )
        state = display.snapshot(self.plan)
        self.assertEqual(state["percent"], 100)
        self.assertIn("PENDING_QUALITY", state["status"])


if __name__ == "__main__":
    unittest.main()


def test_resource_and_effective_rtf_display(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(campaign_id="R3", execution_dir=str(tmp_path), attempts_max=1, phase="benchmark")
        )
    )
    (tmp_path / "ledger.json").write_text(
        json.dumps(
            dict(
                status="RUNNING",
                cases={
                    "case1": dict(
                        status="VERIFYING",
                        assessment=dict(
                            observed_valid_seconds=100, flight_command_wall_s=80, effective_rtf=1.25
                        ),
                    )
                },
            )
        )
    )
    (tmp_path / "resource-progress.json").write_text(
        json.dumps(
            dict(
                available_bytes=5e9,
                owned_tree_rss_bytes=3e9,
                cgroup_total_rss_bytes=None,
                local_free_bytes=50e9,
                external_free_bytes=100e9,
            )
        )
    )
    text = display.render(display.snapshot(plan))
    assert "MemAvailable: 5.00 GB" in text
    assert "cgroup total RSS: 미측정" in text
    assert "유효 RTF 1.250" in text
    assert "학습·최종 시험 데이터에 포함하지 않음" in text


def test_progress_does_not_invent_simulation_time_or_eta(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                campaign_id="R3",
                execution_dir=str(tmp_path),
                attempts_max=50,
                concurrency=4,
                mission_seconds=900,
            )
        )
    )
    (tmp_path / "ledger.json").write_text(
        json.dumps(
            dict(
                status="RUNNING",
                cases={
                    "c1": dict(
                        status="FLYING",
                        case=dict(scenario="hover_normal", mission_seconds=900),
                        lane=dict(instance=0),
                    )
                },
            )
        )
    )
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1/flight-progress.json").write_text(
        json.dumps(dict(wall_elapsed_s=123, observed_simulation_elapsed_s=None, live_rtf=None))
    )
    state = display.snapshot(plan)
    text = display.render(state)
    assert "simulation 경과 미확인 / 목표 900s" in text
    assert "wall 123s" in text
    assert "남은 예상시간: 미확인" in text
    assert state["eta"]["seconds"] is None


def test_eta_uses_measured_campaign_throughput_and_failure_disables_it():
    plan = dict(concurrency=2, mission_seconds=900)
    rows = {
        str(i): dict(status="ARCHIVED_PENDING_QUALITY", case=dict(mission_seconds=900))
        for i in range(2)
    }
    ledger = dict(status="RUNNING", elapsed_s=1000, planned_mission_seconds_total=3600)
    result = display.estimate_eta(plan, ledger, {}, rows)
    assert result["seconds"] == 1000
    assert result["confidence"].startswith("low")
    ledger["status"] = "FAILED_DRAINED_STOPPED"
    assert display.estimate_eta(plan, ledger, {}, rows)["seconds"] is None


def test_live_simulation_clock_display_and_stale_suppression(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(campaign_id="R3", execution_dir=str(tmp_path), attempts_max=1, mission_seconds=900)
        )
    )
    (tmp_path / "ledger.json").write_text(
        json.dumps(
            dict(
                status="RUNNING",
                cases={
                    "c1": dict(
                        status="FLYING", case=dict(scenario="hover_normal"), lane=dict(instance=0)
                    )
                },
            )
        )
    )
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1/flight-progress.json").write_text(
        json.dumps(
            dict(
                wall_elapsed_s=30,
                observed_simulation_elapsed_s=45,
                live_rtf=2,
                producer_run_id="a" * 32,
                producer_progress_updated_utc=now.isoformat(),
            )
        )
    )
    monkeypatch.setattr(display.time, "time", lambda: now.timestamp() + 1)
    text = display.render(display.snapshot(plan))
    assert "simulation 경과 45 / 목표 900s" in text
    assert "실시간 RTF 2" in text
    assert "producer 갱신" in text
    monkeypatch.setattr(display.time, "time", lambda: now.timestamp() + 11)
    state = display.snapshot(plan)
    assert state["workers"][0]["flight"]["observed_simulation_elapsed_s"] is None
    assert state["workers"][0]["flight"]["live_rtf"] is None


def test_retry_uses_current_attempt_progress_only(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(dict(campaign_id="retry", execution_dir=str(tmp_path), attempts_max=1))
    )
    (tmp_path / "ledger.json").write_text(
        json.dumps(
            dict(status="RUNNING", cases={"c1": dict(status="FLYING", attempt=2, attempts_max=3)})
        )
    )
    (tmp_path / "c1/attempt-02").mkdir(parents=True)
    (tmp_path / "c1/flight-progress.json").write_text(json.dumps(dict(marker="old")))
    (tmp_path / "c1/attempt-02/flight-progress.json").write_text(json.dumps(dict(marker="current")))
    state = display.snapshot(plan)
    assert state["workers"][0]["flight"]["marker"] == "current"
    assert state["workers"][0]["attempt"] == 2
    assert state["archived"] == 0
