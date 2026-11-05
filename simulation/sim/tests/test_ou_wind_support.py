"""High-mean OU planning remains moving, three dimensional, seeded and bounded."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from capstone_sim import (
    flight_program,
    normal_variation,
    operating_envelope,
    random_flight,
    wind_model,
)


class OUWindSupportTest(unittest.TestCase):
    def spec(self, wind, sigma, heading=0, seed=84108):
        spec = normal_variation.sample("hover_normal", seed)
        spec.update(soc=.9, temperature=298.15, payload=0., wind_e=wind*math.cos(heading),
                    wind_n=wind*math.sin(heading), gust_sigma=sigma)
        return spec

    def test_all27_upper_sigma_conditions_keep_their_motion(self):
        for wind in (0., 2.5, 5.):
            for kind in (*flight_program.BASIC, "mixed", "random"):
                spec = self.spec(wind, .4)
                mode = kind if kind in ("mixed", "random") else "basic"
                result = flight_program.build(spec, mode, 84108, 480,
                    flight_program.BASIC.index(kind) if mode == "basic" else 0,
                    random_revision=random_flight.REVISION)
                self.assertEqual(result["path"]["kind"], kind)
                self.assertLessEqual(operating_envelope.commanded_airspeed_bound(result), 7+1e-8)

    def test_directional_budget_solves_collision_without_lowering_movement_checks(self):
        # The preserved isotropic cap is <1 at this corner; it is not enlarged silently.
        self.assertLess(operating_envelope.random_speed_limit(self.spec(5, .4)), 1.)
        for seed in range(84108, 84112):
            for angle in (0., math.pi/2, math.pi, -math.pi/2):
                spec = self.spec(5, .4, angle, seed)
                path = random_flight.build(spec, seed, 480)
                spec["path"] = path
                normal_variation.validate(spec)
                result = random_flight.metrics(path)
                self.assertGreaterEqual(result["horizontal_speed_mps"], 1-1e-8)
                self.assertGreaterEqual(result["cruise_speed_min_mps"], .3-1e-8)
                self.assertGreaterEqual(min(result["climb_speed_mps"], result["descent_speed_mps"]), .15)
                self.assertGreaterEqual(min(result["north_span_m"], result["east_span_m"]), 15.)
                self.assertLessEqual(operating_envelope.commanded_airspeed_bound(spec), 7+1e-8)
                for tick in range(481):
                    v = random_flight.kinematics(tick, path)[1]
                    relative = math.sqrt((v[0]-spec["wind_n"])**2+(v[1]-spec["wind_e"])**2+v[2]**2)
                    self.assertLessEqual(relative+1.2, 7+1e-8)

    def test_archived_revisions_keep_original_gust_scope(self):
        for revision, wind in ((random_flight.LEGACY_REVISION, 2.5),
                               (random_flight.FIXED_WIND_REVISION, 5.)):
            with self.assertRaises(ValueError):
                random_flight.build(self.spec(wind, .4), 84108, 480, revision=revision)
        for revision in (random_flight.LEGACY_REVISION, random_flight.FIXED_WIND_REVISION):
            spec = self.spec(0., .1)
            spec["path"] = random_flight.build(spec, 84108, 480, revision=revision)
            frozen = copy.deepcopy(spec)
            normal_variation.validate(spec)
            self.assertEqual(spec, frozen)

    def test_gaussian_mean_limit_is_not_a_clipped_instantaneous_vector(self):
        definition = wind_model.metadata(84108, self.spec(5., .4))
        self.assertAlmostEqual(definition["radial3sigma_point_coverage"], .970709113465, places=10)
        record = dict(time_us=10000, wind_update_time_us=10000, wind_update_index=1,
                      gust_enu_mps=[1.5, -.1, .7], wind_enu_mps=[6.5, -.1, .7],
                      ground_velocity_enu_mps=[0., 0., 0.], relative_airspeed_mps=math.sqrt(6.5**2+.1**2+.7**2))
        wind_model.verify_record(record, definition)
        record["wind_enu_mps"][0] = 5.
        with self.assertRaisesRegex(ValueError, "mean plus gust"):
            wind_model.verify_record(record, definition)

    def test_invalid_sigma_and_mean_reject_before_generation(self):
        for wind, sigma in ((5.0001, .4), (5., .4001), (5., -.01), (5., float("nan"))):
            with self.assertRaises(ValueError):
                random_flight.build(self.spec(wind, sigma), 84108, 480)

    def test_native_trace_binds_time_axis_and_applied_vector(self):
        definition=wind_model.metadata(84108,self.spec(5.,.4))
        header=dict(schema="drone.ou-wind-trace/1",kind="configuration",seed_u32=84108,
                    sigma_per_axis_mps=.4,tau_s=2.,mean_enu_mps=[5.,0.,0.])
        rows=[dict(kind="update",index=i+1,time_us=10000*(i+1),dt_s=.01,
                   gust_enu_mps=[.1+i*.01,.2,.3]) for i in range(2)]
        truth=[dict(time_us=row["time_us"],wind_update_time_us=row["time_us"],
                    wind_update_index=row["index"],gust_enu_mps=row["gust_enu_mps"],
                    wind_enu_mps=[5+row["gust_enu_mps"][0],.2,.3],
                    ground_velocity_enu_mps=[0.,0.,0.],
                    relative_airspeed_mps=math.sqrt((5+row["gust_enu_mps"][0])**2+.2**2+.3**2))
               for row in rows]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"truth.jsonl"
            trace=Path(str(path)+wind_model.TRACE_SUFFIX)
            def write():
                path.write_text("".join(json.dumps(r)+"\n" for r in truth))
                trace.write_text("".join(json.dumps(r)+"\n" for r in [header,*rows]))
            write()
            self.assertEqual(wind_model.verify_trace(path,definition)["native_updates"],2)
            rows[1]["dt_s"] = .02
            write()
            with self.assertRaisesRegex(ValueError,"elapsed time"):
                wind_model.verify_trace(path,definition)
            rows[1]["dt_s"] = .01
            truth[1]["wind_update_time_us"] = 19999
            write()
            with self.assertRaisesRegex(ValueError,"does not match"):
                wind_model.verify_trace(path,definition)

    def test_native_header_cannot_disguise_sigma_or_seed(self):
        definition=wind_model.metadata(84108,self.spec(5.,.4))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"trace.jsonl"
            path.write_text(json.dumps(dict(schema="drone.ou-wind-trace/1",kind="configuration",
                seed_u32=84109,sigma_per_axis_mps=.2,tau_s=2.,mean_enu_mps=[5.,0.,0.]))+"\n")
            with self.assertRaisesRegex(ValueError,"configuration"):
                list(wind_model.trace_records(path,definition))
