"""Joint health policy and independent checks of every interpolated native parameter."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from capstone_sim import collection_config, runner
from capstone_sim import propulsion_health as health


class PropulsionHealthTest(unittest.TestCase):
    def test_only_three_joint_scenarios_and_two_endpoints_can_be_selected(self):
        self.assertEqual({s for s in runner.SCENARIOS if s.startswith("propulsion_")}, health.SCENARIOS)
        self.assertEqual(collection_config.ANOMALY, health.SCENARIOS |
                         {"battery_aging_existing", "battery_aging_step", "battery_aging_ramp"})
        self.assertEqual(set(health.load_catalog()["states"]), {"normal", "abnormal"})
        normal = health.resolve("hover_normal")
        for scenario in health.SCENARIOS:
            selected = health.resolve(scenario)
            self.assertEqual(selected["target_state"], "abnormal")
            self.assertEqual(selected["normal"], normal["initial"])
            self.assertEqual(selected["fixed_for_flight"], scenario.endswith("existing"))
            self.assertEqual(selected["transition_s"],
                             0 if scenario.endswith("existing") else 1 if scenario.endswith("step") else 20)
        for seconds in (0, -1, float("nan"), float("inf"), True, 1801):
            with self.assertRaises(ValueError):
                health.resolve("propulsion_loss_ramp", transition_s=seconds)
        with self.assertRaises(ValueError):
            health.resolve("propulsion_loss_existing", transition_s=20)

    def test_independent_truth_verifier_rejects_partial_or_unsynchronized_impairment(self):
        selected = health.resolve("propulsion_loss_ramp", transition_s=20)
        rows = []
        for t in (0, 10, 15, 20, 30, 35):
            fraction = max(0., min(1., (t-10)/20))
            row = dict(time_us=t*1_000_000, propulsion_profile_fraction=fraction,
                       propulsion_profile_onset_us=10_000_000)
            for key, observed in zip(health.KEYS, health.TRUTH, strict=True):
                a, b = selected["initial"][key], selected["target"][key]
                row[observed] = a + fraction*(b-a)
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"truth.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            self.assertEqual(health.verify_truth(path, selected)["status"], "passed")
            rows[3]["kt_prop"] = selected["initial"]["kt_prop"]
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "kt_prop"):
                health.verify_truth(path, selected)

    def test_sealed_endpoints_reject_per_run_tuning(self):
        selected = health.resolve("propulsion_loss_existing")
        digest = hashlib.sha256(health.CATALOG.read_bytes()).hexdigest()
        health.verify_selection(selected, health.CATALOG, "propulsion_loss_existing", digest)
        selected["target"]["friction"] *= .5
        with self.assertRaises(ValueError):
            health.verify_selection(selected, health.CATALOG, "propulsion_loss_existing", digest)

    def test_mission_relative_transition_must_finish_and_accepts_custom_duration(self):
        for scenario in health.DYNAMIC:
            with self.assertRaises(ValueError):
                collection_config.validate_request(scenario, 200, 180, propulsion_transition_s=30)
            collection_config.validate_request(scenario, 480, 180, propulsion_transition_s=30)
