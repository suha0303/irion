"""Bounded environment semantics and continuous-time path budgets, not fitted wind data."""

import hashlib
import json
import math
import unittest

from capstone_sim import (
    bounded_wind,
    flight_program,
    normal_variation,
    operating_envelope,
    random_flight,
    wind_model,
)


class BoundedWindTest(unittest.TestCase):
    def spec(self, environment, seed=94108, direction=0., variation=1.):
        result = normal_variation.sample("hover_normal", seed)
        result.update(soc=.9, temperature=298.15, payload=0., wind_e=0., wind_n=0., gust_sigma=0.,
                      wind_profile=bounded_wind.profile(environment, direction, variation))
        return result

    def test_all27_conditions_keep_motion_and_bounded_set_planning(self):
        for environment in ("calm", "low", "strong"):
            for kind in (*flight_program.BASIC, "mixed", "random"):
                spec = self.spec(environment)
                mode = kind if kind in ("mixed", "random") else "basic"
                result = flight_program.build(spec, mode, 94108, 480,
                    flight_program.BASIC.index(kind) if mode == "basic" else 0)
                self.assertEqual(result["path"]["kind"], kind)
                self.assertLessEqual(operating_envelope.commanded_airspeed_bound(result), 7+1e-8)
                if kind == "random":
                    self.assertEqual(result["path"]["revision"], random_flight.BOUNDED_REVISION)

    def test_strong_random_bound_covers_every3d_wind_direction(self):
        for seed in range(94108, 94112):
            spec = self.spec("strong", seed)
            spec["path"] = random_flight.build(spec, seed, 480)
            self.assertAlmostEqual(spec["path"]["limits"]["horizontal_speed_mps"], 1.6)
            metrics = random_flight.metrics(spec["path"])
            self.assertGreaterEqual(metrics["horizontal_speed_mps"], 1.)
            self.assertGreaterEqual(metrics["cruise_speed_min_mps"], .3*1.6-1e-8)
            for tick in range(481):
                velocity = random_flight.kinematics(tick, spec["path"])[1][:3]
                self.assertLessEqual(5+math.sqrt(sum(v*v for v in velocity)), 7+1e-8)

    def test_preserved_versions_reject_new_environment_semantics(self):
        for revision in (random_flight.LEGACY_REVISION, random_flight.FIXED_WIND_REVISION, random_flight.REVISION):
            with self.assertRaisesRegex(ValueError, "Preserved"):
                random_flight.build(self.spec("strong"), 94108, 480, revision=revision)
        plain = normal_variation.sample("hover_normal", 84108)
        plain.update(soc=.9, temperature=298.15, payload=0., wind_e=0., wind_n=5., gust_sigma=.4)
        path = random_flight.build(plain, 84108, 480, revision=random_flight.REVISION)
        digest = hashlib.sha256(json.dumps(path, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, "33bc0ba47916f4236ad5b1c0b91bbc51689c06a29228302575b088287f89b55c")

    def test_legacy_sigma_is_not_reinterpreted_as_bounded_intensity(self):
        spec = self.spec("strong")
        spec["gust_sigma"] = .4
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            normal_variation.validate(spec)
        definition = bounded_wind.definition(94108, self.spec("strong")["wind_profile"])
        self.assertEqual(definition["latent_std_dimensionless"], 1.)
        self.assertEqual(definition["generator_speed_bounds_mps"], [2.75, 4.75])
        self.assertNotIn("sigma_per_axis_mps", definition)
        scenario = dict(seed=94108, wind_profile=definition["profile"], parameters={})
        self.assertEqual(wind_model.expected_metadata(scenario), definition)

    def test_invalid_profile_parameters_fail_before_generation(self):
        for environment, direction, intensity in (("unknown", 0., 1.), ("strong", float("nan"), 1.),
                                                  ("strong", 0., .49), ("strong", 0., 1.51)):
            with self.assertRaises(ValueError):
                bounded_wind.profile(environment, direction, intensity)

    def test_physical_step_bounds_and_frame_mismatches_cannot_be_hidden(self):
        value = bounded_wind.definition(94108, self.spec("strong")["wind_profile"])
        record = dict(wind_sample_index=1, wind_force_evaluations=10, wind_enu_mps=[0., 3.75, 0.],
                      ground_velocity_enu_mps=[0., 0., 0.], relative_airspeed_mps=3.75,
                      wind_observed_min_norm_mps=3.75, wind_observed_max_norm_mps=3.75)
        bounded_wind.verify_record(record, value)
        record["wind_observed_max_norm_mps"] = 5.01
        with self.assertRaisesRegex(ValueError, "environment"):
            bounded_wind.verify_record(record, value)
        record["wind_observed_max_norm_mps"] = 3.75
        record["relative_airspeed_mps"] = 0
        with self.assertRaisesRegex(ValueError, "frame"):
            bounded_wind.verify_record(record, value)
