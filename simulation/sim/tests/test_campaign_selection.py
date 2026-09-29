"""Legacy campaigns retain X500 defaults and never silently discard vehicle settings."""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from capstone_sim import campaign as module


class CampaignSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original_root = module.ROOT
        guard = patch.object(module, "ROOT", self.root)
        guard.start()
        self.addCleanup(guard.stop)
        self.path = self.root / "campaign.yaml"
        self.config = dict(checkpoint="campaign.json", cases=[dict(scenario="hover_normal")])

    def write(self, config=None):
        self.path.write_text(yaml.safe_dump(self.config if config is None else config))
        return self.path

    def test_all_retained_examples_plan_without_starting_a_flight(self):
        with patch.object(module, "run_flight", side_effect=AssertionError("no flight")):
            for path in (self.original_root / "profiles/campaigns").glob("*.yaml"):
                with self.subTest(path=path.name):
                    result = module.plan(path)
                    self.assertEqual(result["vehicle"], "x500_reference")
                    self.assertGreater(result["cases"], 0)
        self.assertFalse((self.root / "artifacts").exists())

    def test_default_x500_dispatch_is_unchanged(self):
        self.config["checkpoint"] = "nested/campaign.json"
        with (
            patch.object(module, "run_flight", return_value=dict(
                status="completed", run=str(self.root / ("a" * 32)),
            )) as flight,
            patch.object(module, "verify", return_value=dict(status="passed")),
        ):
            result = module.campaign(self.write())
        flight.assert_called_once_with("hover_normal", 17, 90, 0, None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total"], 1)
        self.assertTrue((self.root / "artifacts/nested/campaign.config.json").is_file())

    def test_unsupported_explicit_choices_fail_before_any_job_or_artifact(self):
        for key in ("vehicle", "vehicle_profile", "configuration", "real_time_factor",
                    "normal_variation", "typo"):
            for where in ("top", "case"):
                with self.subTest(key=key, where=where):
                    config = copy.deepcopy(self.config)
                    (config if where == "top" else config["cases"][0])[key] = "ignored before"
                    with patch.object(module, "run_flight") as flight:
                        with self.assertRaisesRegex(ValueError, "run --vehicle-profile"):
                            module.campaign(self.write(config))
                    flight.assert_not_called()
        self.assertFalse((self.root / "artifacts").exists())

    def test_checkpoint_cannot_escape_replace_a_seal_or_follow_a_symlink(self):
        for value in (None, "", "/tmp/campaign.json", "../escape.json", "a/../../escape.json",
                      "a\\escape.json", "not-json.yaml", "source_lock.json",
                      "runtime-manifest.json", "battery-fit.json"):
            with self.subTest(value=value):
                config = dict(self.config, checkpoint=value)
                with self.assertRaises(ValueError):
                    module.plan(self.write(config))
        (self.root / "artifacts").mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "artifacts/linked").symlink_to(outside, target_is_directory=True)
        self.config["checkpoint"] = "linked/campaign.json"
        with self.assertRaisesRegex(ValueError, "symlinks"):
            module.plan(self.write())

    def test_lane_values_are_validated_in_plan(self):
        for key, values in (("concurrency", (0, 3, True, "2", 2.0)),
                            ("instance_offset", (-1, 10, True, "0", 0.0))):
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        module.plan(self.write(dict(self.config, **{key: value})))
        config = dict(self.config, concurrency=2, instance_offset=8)
        self.assertEqual(module.plan(self.write(config))["concurrency"], 2)
        config["instance_offset"] = 9
        with self.assertRaises(ValueError):
            module.plan(self.write(config))

    def test_invalid_case_values_are_input_errors(self):
        for key, values in (
            ("scenario", (None, "unknown", ["hover_normal"])),
            ("seed", (True, -1, 2**32, 0.5, "17")),
            ("duration", (True, 0, -1, 1801, math.nan, math.inf, "90")),
            ("parent_attempt_id", (True, 2, "not-a-run")),
        ):
            for value in values:
                with self.subTest(key=key, value=value):
                    config = copy.deepcopy(self.config)
                    config["cases"][0][key] = value
                    with self.assertRaises(ValueError):
                        module.plan(self.write(config))
        self.config["cases"][0]["seed"] = 0
        self.assertEqual(module.plan(self.write())["cases"], 1)

    def test_bad_structure_and_yaml_are_input_errors(self):
        for config in (None, [], {}, dict(self.config, cases=[]),
                       dict(self.config, cases={}), dict(self.config, cases=[None])):
            with self.subTest(config=config):
                self.path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    module.plan(self.path)
        self.path.write_text("cases: [\n")
        with self.assertRaisesRegex(ValueError, "Invalid campaign YAML"):
            module.plan(self.path)


if __name__ == "__main__":
    unittest.main()
