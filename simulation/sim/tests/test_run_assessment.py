"""Synthetic fixtures only: no flight, process, network or persistent run creation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from capstone_sim import collection_config, run_assessment
from capstone_sim.recording import TOPICS, Recorder, dump, integrity


class RunAssessmentTest(unittest.TestCase):
    def fixture(self, root, anomaly=False, gap=False):
        run = root / "synthetic"
        for part in ("reference", "evaluation", "observations", "native_logs"):
            (run / part).mkdir(parents=True)
        rec = Recorder(run, "synthetic_contract_fixture")
        with (run / "native_logs/capture.jsonl").open("w") as stream:
            for t in range(0, 2100000, 100000):
                for topic, (_, units, _) in TOPICS.items():
                    if gap and topic == "battery_status" and 500000 < t < 1500000:
                        continue
                    fields = {key: 0.0 for key in units}
                    for key in fields:
                        size = {"output": 4, "xyz": 3, "q": 4, "velocity": 3}.get(key)
                        if size:
                            fields[key] = [0.] * size
                    stream.write(json.dumps(dict(topic=topic, sample_time_us=t,
                                                 available_time_us=t+1, fields=fields))+"\n")
        rec.close()
        meta = run_assessment.read(run / "observations/metadata.json")
        meta["vehicle"].update(nominal_capacity_ah=12, nominal_parallel_cells=4,
                               battery_profile_id=collection_config.battery()["profile_id"])
        dump(run / "observations/metadata.json", meta)
        # Mock execution labels test adapter predicates, not actual SITL provenance.
        dump(run / "run.json", dict(origin="px4_sitl", status="completed", instance=2,
                                    gz_partition="x500_synthetic", configuration="collection",
                                    mavlink_source_checked=True))
        dump(run / "evaluation/execution.json", [dict(state="airborne", sim_time=.2),
                                                 dict(state="landing", sim_time=1.7)])
        dump(run / "evaluation/scenario.json", dict(
            name="battery_resistance_existing" if anomaly else "hover_normal",
            requested_duration_s=1.5, parameters={}, normal_range_revision=collection_config.vehicle()["revision"],
            health_eligibility="abnormal" if anomaly else "confirmed_healthy"))
        path = "profiles/collection/vehicle.yaml"
        dump(run / "reference/source_lock.json", dict(local_sources={
            path:run_assessment.file_hash(collection_config.ROOT/path)}, runtime_binary_sha256={}))
        dump(run / "reference/fc-configuration.json", dict(
            verified=True, expected=collection_config.fc_parameters(),
            observed=collection_config.fc_parameters()))
        with (run / "evaluation/truth.jsonl").open("w") as stream:
            for t in range(0, 2100000, 100000):
                stream.write(json.dumps(dict(time_us=t, voltage_v=10 if anomaly else 15,
                    soc=.1 if anomaly else .8, command=[.95 if anomaly else .6]*4,
                    thrust_n=[6]*4, solver_residual_v=0, model_valid=not anomaly))+"\n")
        integrity(run)
        return run

    def assess(self, run, instance=2, *, legacy_fc=False):
        original = run_assessment.read
        def fixture_read(path):
            if str(path).endswith(("artifacts/runtime-manifest.json", "round001-r3/producer-lock.json")):
                return ({"files": {}} if legacy_fc else
                        {"files": {}, "fc_parameters": collection_config.fc_parameters()})
            return original(path)
        with patch.object(run_assessment, "read", side_effect=fixture_read):
            return run_assessment.assess(run, instance, 1.5, collection_config.battery()["profile_id"])

    def test_touchdown_policy_is_selected_by_producer_not_self_reported_keys(self):
        with tempfile.TemporaryDirectory() as td:
            run = self.fixture(Path(td))
            path = run / "reference/fc-configuration.json"
            fc = run_assessment.read(path)
            del fc["expected"]["MPC_LAND_SPEED"]
            del fc["observed"]["MPC_LAND_SPEED"]
            dump(path, fc)
            integrity(run)
            self.assertIn("actual FC parameter gate", self.assess(run)["errors"])
            historical = self.assess(run, legacy_fc=True)
            self.assertTrue(historical["contract_passed"], historical["errors"])
            fc["observed"]["BAT1_CAPACITY"] = 6000
            dump(path, fc)
            integrity(run)
            self.assertIn("actual FC parameter gate", self.assess(run, legacy_fc=True)["errors"])

    def test_normal_contract_and_anomaly_gate_are_distinct(self):
        for anomaly in (False, True):
            with tempfile.TemporaryDirectory() as td:
                result = self.assess(self.fixture(Path(td), anomaly=anomaly))
                for key in ("contract_passed", "instance_isolation_passed", "missing_data_passed", "physical_gate_passed"):
                    self.assertTrue(result[key], (key, result))
                self.assertEqual(result["normal_endurance_gate_passed"], not anomaly)
                self.assertEqual(result["observed_valid_seconds"], 1.5)
                self.assertEqual(result["schema"], "drone.run-assessment/1")

    def test_legacy_identifiers_are_read_only_and_retain_source_hash_checks(self):
        with tempfile.TemporaryDirectory() as td:
            run = self.fixture(Path(td))
            state = run_assessment.read(run / "run.json")
            state["configuration"] = collection_config.LEGACY_NAME
            dump(run / "run.json", state)
            metadata = run_assessment.read(run / "observations/metadata.json")
            metadata["vehicle"]["battery_profile_id"] = collection_config.LEGACY_BATTERY_ID
            dump(run / "observations/metadata.json", metadata)
            scenario = run_assessment.read(run / "evaluation/scenario.json")
            scenario["normal_range_revision"] = collection_config.LEGACY_REVISION
            dump(run / "evaluation/scenario.json", scenario)
            source = run_assessment.read(run / "reference/source_lock.json")
            source["local_sources"] = {collection_config.LEGACY_PROFILE_PATH: collection_config.LEGACY_PROFILE_SHA256}
            dump(run / "reference/source_lock.json", source)
            integrity(run)
            original = run_assessment.read

            def descriptor(path):
                return {"files": {}} if str(path).endswith("round001-r3/producer-lock.json") else original(path)

            with patch.object(run_assessment, "read", side_effect=descriptor):
                result = run_assessment.assess(run, 2, 1.5, collection_config.LEGACY_BATTERY_ID)
            self.assertTrue(result["contract_passed"], result["errors"])
            self.assertTrue(result["instance_isolation_passed"])
            self.assertEqual(result["schema"], "drone.run-assessment/1")
            source["local_sources"][collection_config.LEGACY_PROFILE_PATH] = "0" * 64
            dump(run / "reference/source_lock.json", source)
            integrity(run)
            with patch.object(run_assessment, "read", side_effect=descriptor):
                changed = run_assessment.assess(run, 2, 1.5, collection_config.LEGACY_BATTERY_ID)
            self.assertIn("profile source hash", changed["errors"])

    def test_gap_tamper_and_wrong_instance_fail_separate_gates(self):
        with tempfile.TemporaryDirectory() as td:
            run = self.fixture(Path(td), gap=True)
            result = self.assess(run, instance=3)
            self.assertFalse(result["missing_data_passed"])
            self.assertFalse(result["instance_isolation_passed"])
            self.assertTrue(result["contract_passed"])
            with (run / "observations/diagnostics.jsonl").open("a") as f:
                f.write("{}\n")
            self.assertFalse(self.assess(run)["contract_passed"])

    def test_early_watermark_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            run = self.fixture(Path(td))
            path = run / "observations/watermarks.jsonl"
            original = path.read_text()
            early = dict(closed=False, recorded_events=1, through_available_time_us=2000000)
            path.write_text(json.dumps(early)+"\n"+original)
            integrity(run)
            result = self.assess(run)
            self.assertFalse(result["contract_passed"])
            self.assertIn("causal order", result["errors"])
