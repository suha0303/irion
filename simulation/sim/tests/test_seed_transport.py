"""Logical uint32 identity and the exact Gazebo8 CLI option; no native launch."""
import unittest

from capstone_sim.runner import gazebo_seed_transport


class SeedTransportTest(unittest.TestCase):
    def test_boundaries_are_reversible_without_losing_upper_bit(self):
        for seed, signed in ((0,0),(17,17),(2**31-1,2**31-1),(2**31,-2**31),(2**32-1,-1),(2993778865,-1301188431)):
            with self.subTest(seed=seed):
                value=gazebo_seed_transport(seed)
                self.assertEqual(value["logical_seed_u32"],seed)
                self.assertEqual(value["gazebo_native_seed_u32"],seed)
                self.assertEqual(value["gazebo_cli_seed_i32"],signed)
                self.assertEqual(signed % 2**32,seed)
                self.assertEqual(value["gazebo_cli_option"],f"--seed={signed}")

    def test_zero_default_semantics_are_explicit(self):
        self.assertEqual(gazebo_seed_transport(0)["gazebo_seed_mode"],"native_default_zero_sentinel")
        self.assertEqual(gazebo_seed_transport(17)["gazebo_seed_mode"],"explicit_uint32")

    def test_invalid_domain_fails_before_launch(self):
        for invalid in (True,False,-1,2**32,17.0,"17",None):
            with self.subTest(seed=invalid),self.assertRaisesRegex(ValueError,"uint32"):
                gazebo_seed_transport(invalid)
