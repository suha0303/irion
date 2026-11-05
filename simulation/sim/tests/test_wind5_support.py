"""Wind-relative geometric limits; no flight or relaxed physical/data gates."""

import copy
import math
import unittest

from capstone_sim import flight_program, normal_variation, operating_envelope, random_flight


class Wind5SupportTest(unittest.TestCase):
    def spec(self, wind):
        spec = normal_variation.sample("hover_normal", 73001)
        spec.update(soc=.9, temperature=298.15, payload=0., gust_sigma=0., wind_e=wind, wind_n=0.)
        return spec

    def test_all27_boundaries_generate_without_substituting_motion(self):
        for wind in (0.,2.5,5.):
            for kind in (*flight_program.BASIC,"mixed","random"):
                spec = self.spec(wind)
                mode = kind if kind in ("mixed","random") else "basic"
                index = flight_program.BASIC.index(kind) if mode == "basic" else 0
                result = flight_program.build(spec,mode,73001,480,index,random_revision=random_flight.REVISION)
                self.assertEqual(result["path"]["kind"],kind)
                normal_variation.validate(result)
                self.assertLessEqual(operating_envelope.commanded_airspeed_bound(result),7+1e-8)
                if kind == "random":
                    self.assertEqual(result["path"]["revision"],random_flight.REVISION)
                    self.assertGreaterEqual(result["path"]["limits"]["horizontal_speed_mps"],1.)

    def test_wind5_random_exact_velocity_knots_keep_full3d_margin(self):
        spec = self.spec(5.)
        path = random_flight.build(spec,73001,480)
        expected = math.sqrt(7**2-1.2**2)-5
        self.assertAlmostEqual(path["limits"]["horizontal_speed_mps"],expected)
        for tick in range(4801):
            velocity = random_flight.kinematics(tick/10,path)[1]
            self.assertLessEqual(math.sqrt(velocity[0]**2+(velocity[1]-5)**2+velocity[2]**2),7+1e-8)
        self.assertGreater(random_flight.metrics(path)["distance_m"],100.)

    def test_above_maximum_and_preserved_v4_extended_gust_are_rejected(self):
        for wind,gust in ((5.0001,0.),(6.,0.)):
            spec = self.spec(wind)
            spec["gust_sigma"] = gust
            with self.assertRaises(ValueError):
                normal_variation.validate(spec)
            with self.assertRaises(ValueError):
                random_flight.build(spec,73001,480)

        for wind in (2.5,5.):
            with self.assertRaisesRegex(ValueError,"Preserved v4"):
                spec=self.spec(wind)
                spec["gust_sigma"] = .01
                random_flight.build(spec,73001,480,revision=random_flight.FIXED_WIND_REVISION)

    def test_preserved_v3_keeps_its_original_wind_scope(self):
        spec = self.spec(2.)
        path = random_flight.build(spec,73001,480)
        # A legacy3m/s cap at wind2 is outside the original2.5m/s budget.
        spec["path"] = dict(path,revision=random_flight.LEGACY_REVISION)
        with self.assertRaises(ValueError):
            normal_variation.validate(spec)
        calm = self.spec(0.)
        calm["path"] = random_flight.build(calm,73001,480)
        calm["path"]["revision"] = random_flight.LEGACY_REVISION
        snapshot = copy.deepcopy(calm)
        normal_variation.validate(calm)
        self.assertEqual(calm,snapshot)
        calm["wind_e"] = 5.
        with self.assertRaisesRegex(ValueError,"Preserved v3"):
            normal_variation.validate(calm)
