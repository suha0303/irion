"""Roaming route geometry, independent derivative checks, and MAVLink integration."""

import ast
import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from capstone_sim.flight_program import build
from capstone_sim.normal_variation import sample, validate
from capstone_sim.random_flight import REVISION, kinematics, metrics


def route(seed=17, duration=900, *, wind=0.0, gust=0.0):
    spec = sample("hover_normal", seed)
    spec.update(wind_e=wind, wind_n=0.0, gust_sigma=gust)
    return build(spec, "random", seed, duration, 0)


class RandomFlightTest(unittest.TestCase):
    def test_crash_seed_preserves_exact_frozen_route_and_cruise_speed_metrics(self):
        # The preserved 2026-10-31 core stopped in metrics/CALL_FUNCTION_EX
        # for this payload case. Explicit arguments avoid that star-slice path.
        spec = sample("payload_normal", 374097348)
        spec.update(
            gust_sigma=0.058473704825859105,
            payload=0.174888937228785,
            soc=0.8738984886942539,
            temperature=296.0048720872183,
            wind_e=0.3051568465143509,
            wind_n=0.45682947379638295,
        )
        path = build(spec, "random", 374097348, 900, 0)["path"]
        digest = hashlib.sha256(
            json.dumps(path, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(
            digest, "2ca77b4995b2b1c80c268f8175ac98fecde05da0fd3ca5e8e737958a8a1d304e"
        )
        old = [math.hypot(*p[5:7]) for p in path["program"][1:-1]]
        explicit = [math.hypot(p[5], p[6]) for p in path["program"][1:-1]]
        self.assertEqual(old, explicit)
        result = metrics(path)
        self.assertEqual(result["horizontal_speed_mps"], max(old))
        self.assertEqual(result["cruise_knot_speed_min_mps"], min(old))

    def test_actual_four_balanced_sets_preserve_1200_frozen_programs(self):
        # In-memory manifests only: no simulator, collection, or output data.
        root = Path(__file__).resolve().parents[3]
        with patch.object(sys, "path", [str(root / "Data/generation"), str(root / "ai"), *sys.path]):
            from pretrain.balanced import cases
            from pretrain.balanced import validate as validate_balanced

        expected = (
            "e410ae260b5ccc86c59adb6414f55ab00f1ec7c95a675279c38730244d421358",
            "1d99c145d35a1f63a1a254f41630a2822f05af4d0e1806f9c61b3dd4d654c036",
            "d4c57a4d61c3d92932c53917c1da20220b1d3612a608bb97aca5701014d7eca4",
            "fbffaa79dc8b528e2e0792e7c03e68f6b6f9c3a1105f9ce48a11c4bf4a1a8992",
        )
        random_count, total = 0, 0
        for index, expected_digest in enumerate(expected):
            with self.subTest(set_index=index + 1):
                rows = cases(374096325 + index * 1000)
                self.assertEqual(len(rows), 300)
                validate_balanced(rows)
                digest = hashlib.sha256(
                    json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                self.assertEqual(digest, expected_digest)
                total += len(rows)
                for row in rows:
                    if row.get("path_mode") != "random":
                        continue
                    random_count += 1
                    path = row["normal_variation"]["path"]
                    old = [math.hypot(*p[5:7]) for p in path["program"][1:-1]]
                    explicit = [math.hypot(p[5], p[6]) for p in path["program"][1:-1]]
                    self.assertEqual(old, explicit)
                    result = metrics(path)
                    self.assertEqual(result["horizontal_speed_mps"], max(old))
                    self.assertEqual(result["cruise_knot_speed_min_mps"], min(old))
        self.assertEqual(total, 1200)
        self.assertEqual(random_count, 312)

    def test_distant_landing_has_explicit_physical_ground_at_zero_height(self):
        from capstone_sim.runner import prepare_random_ground

        path = route(934215133)["path"]
        self.assertGreater(max(abs(v) for v in path["program"][-1][1:3]), 1050)
        with tempfile.TemporaryDirectory() as folder:
            world = Path(folder) / "world.sdf"
            world.write_text(
                '<sdf><world><model name="ground_plane"><static>true</static><link>'
                '<collision><geometry><plane><normal>0 0 1</normal><size>3000 3000</size>'
                '</plane></geometry><surface><friction><ode/></friction></surface></collision>'
                '<visual><geometry><plane><size>3000 3000</size></plane></geometry></visual>'
                '</link></model></world></sdf>'
            )
            original_path = copy.deepcopy(path)
            evidence = prepare_random_ground(world, path)
            tree = ET.parse(world)
            collision = tree.find("world/model/link/collision")
            self.assertIsNone(collision.find("geometry/plane"))
            size = [float(v) for v in collision.findtext("geometry/box/size").split()]
            pose = [float(v) for v in collision.findtext("pose").split()]
            self.assertEqual(pose[2] + size[2] / 2, 0)
            self.assertIsNotNone(collision.find("surface/friction/ode"))
            self.assertEqual(tree.findtext("world/model/static"), "true")
            self.assertEqual(evidence["top_z_m"], 0)
            self.assertEqual(path, original_path)
            for tick in range(9001):
                x, y, _, _ = kinematics(tick / 10, path)[0]
                # The collision must cover both NED-to-ENU axes with real margin.
                self.assertLessEqual(max(abs(x), abs(y)) + 100 - 1e-7, min(size[:2]) / 2)

    def test_heartbeat_continues_during_landing_without_duplicate_clock_sends(self):
        source = Path(__file__).parents[1] / "src/capstone_sim/runner.py"
        tree = ast.parse(source.read_text())
        block = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and any(
                isinstance(s, ast.Expr)
                and isinstance(s.value, ast.Call)
                and isinstance(s.value.func, ast.Attribute)
                and s.value.func.attr == "heartbeat_send"
                for s in n.body
            )
        )
        compiled = compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec")
        sent = []
        env = dict(
            phase="flying",
            now=900.0,
            next_heartbeat_us=0,
            conn=SimpleNamespace(mav=SimpleNamespace(heartbeat_send=lambda *args: sent.append(args))),
        )
        exec(compiled, env)
        env["phase"] = "landing"
        exec(compiled, env)
        self.assertEqual(len(sent), 1)
        env["now"] = 900.1
        exec(compiled, env)
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0], sent[1])
        env["now"], env["phase"] = 901.0, "waiting"
        exec(compiled, env)
        self.assertEqual(len(sent), 2)

    def test_reproducible_continuous_steering_without_radius_or_return(self):
        furthest = 0
        for seed in range(40):
            for duration in (180, 900):
                spec = route(seed, duration)
                self.assertEqual(spec, route(seed, duration))
                p = spec["path"]
                self.assertEqual(p["revision"], REVISION)
                self.assertNotIn("radius_m", p["limits"])
                self.assertNotEqual(p["program"][-1][1:3], [0, 0])
                self.assertEqual(p["program"][-1][0], duration)
                m = metrics(p)
                self.assertGreaterEqual(min(m["climb_speed_mps"], m["descent_speed_mps"]), 0.15)
                self.assertGreaterEqual(
                    m["horizontal_speed_mps"] / m["cruise_knot_speed_min_mps"], 1.6
                )
                self.assertGreaterEqual(m["cruise_speed_min_mps"], 0.9 - 1e-8)
                furthest = max(furthest, m["max_distance_from_start_m"])
        self.assertGreater(furthest, 300)

    def test_derivatives_and_nonzero_velocity_are_continuous_at_input_changes(self):
        p = route()["path"]
        for a, b in zip(p["program"], p["program"][1:]):
            for fraction in (0.1, 0.25, 0.5, 0.75, 0.9):
                t = a[0] + fraction * (b[0] - a[0])
                dt = 0.001
                pos, vel, acc = kinematics(t, p)
                before, vb, ab = kinematics(t - dt, p)
                after, va, aa = kinematics(t + dt, p)
                for i in range(4):
                    self.assertAlmostEqual((after[i] - before[i]) / (2 * dt), vel[i], places=6)
                    self.assertAlmostEqual((va[i] - vb[i]) / (2 * dt), acc[i], places=6)
                self.assertLessEqual(math.hypot(*vel[:2]), 3 + 1e-8)
                self.assertLessEqual(abs(vel[2]), 1.2 + 1e-8)
                self.assertLessEqual(math.sqrt(sum(x * x for x in acc[:3])), 1.4 + 1e-8)
                self.assertLessEqual(
                    math.sqrt(sum(((y - x) / (2 * dt)) ** 2 for x, y in zip(ab[:3], aa[:3]))),
                    1.5 + 1e-8,
                )
                self.assertTrue(all(math.isfinite(x) for x in pos))
        for point in p["program"][1:-1]:
            pos, vel, acc = kinematics(point[0], p)
            for actual, expected in zip(pos, point[1:5]):
                self.assertAlmostEqual(actual, expected)
            self.assertGreater(math.hypot(*vel[:2]), 0.9)
            self.assertLess(max(map(abs, acc)), 1e-8)
            left = kinematics(point[0] - 1e-5, p)
            right = kinematics(point[0] + 1e-5, p)
            for before, after in zip(left, right):
                self.assertLess(max(abs(x - y) for x, y in zip(before, after)), 0.0001)
        for t in (0, 900):
            self.assertLess(max(map(abs, kinematics(t, p)[1])), 1e-8)
        self.assertEqual(kinematics(-1, p)[0], tuple(p["program"][0][1:5]))
        for x, y in zip(kinematics(1000, p)[0], p["program"][-1][1:5]):
            self.assertAlmostEqual(x, y)

    def test_entire_cruise_has_no_stops_or_slow_checkpoint_neighborhoods(self):
        # Numeric sampling is independent of the analytic acceptance bounds.
        for seed, wind, gust in ((20261030, 0, 0), (9, 2, 0.4), (18, 0, 0)):
            p = route(seed, wind=wind, gust=gust)["path"]
            cap = p["limits"]["horizontal_speed_mps"]
            for tick in range(120, 8901):
                pos, vel, _ = kinematics(tick / 10, p)
                self.assertGreaterEqual(math.hypot(*vel[:2]), 0.3 * cap - 1e-8)
                self.assertGreaterEqual(-pos[2], 3 - 1e-8)
                self.assertLessEqual(-pos[2], 30 + 1e-8)

    def test_wind_budget_and_malformed_or_retired_programs_rejected(self):
        windy = route(17, wind=2.0, gust=0.4)
        self.assertAlmostEqual(windy["path"]["limits"]["horizontal_speed_mps"], 1.3)
        validate(windy)
        windy = route(17)
        windy.update(wind_e=2.0, gust_sigma=0.4)
        with self.assertRaisesRegex(ValueError, "wind-adjusted"):
            validate(windy)
        retired = sample("hover_normal", 17)
        retired["path"].update(
            kind="random", duration_s=900, program=[[0, 0, 0, -3, 0], [900, 1, 0, -3, 0]]
        )
        with self.assertRaisesRegex(ValueError, "Retired"):
            validate(retired)
        retired = route()
        retired["path"]["revision"] = "random-roaming-v2"
        with self.assertRaisesRegex(ValueError, "Retired"):
            validate(retired)
        for mutation in (
            "timing",
            "position",
            "altitude",
            "yaw",
            "limit",
            "nan",
            "stop",
            "reverse",
            "overspeed",
        ):
            damaged = copy.deepcopy(route())
            p = damaged["path"]
            if mutation == "timing":
                p["program"][1][0] = 0
            elif mutation == "position":
                p["program"][1][1] += 1
            elif mutation == "altitude":
                p["program"][1][3] = -31
            elif mutation == "yaw":
                p["program"][1][4] = 7
            elif mutation == "limit":
                p["limits"]["min_cruise_speed_fraction"] = 0
            elif mutation == "nan":
                p["program"][1][1] = float("nan")
            else:
                row = p["program"][2]
                row[5:7] = (
                    [0, 0]
                    if mutation == "stop"
                    else [-v for v in p["program"][1][5:7]]
                    if mutation == "reverse"
                    else [10, 10]
                )
                # Keep position integration consistent to expose the speed guard.
                for a, b in zip(p["program"], p["program"][1:]):
                    b[1:4] = [
                        x + (b[0] - a[0]) * (v + w) / 2 for x, v, w in zip(a[1:4], a[5:8], b[5:8])
                    ]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(damaged)

    def test_real_command_block_enables_feedforward_only_for_random_flight(self):
        source = Path(__file__).parents[1] / "src/capstone_sim/runner.py"
        tree = ast.parse(source.read_text())
        block = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and any(
                isinstance(s, ast.Expr)
                and isinstance(s.value, ast.Call)
                and isinstance(s.value.func, ast.Attribute)
                and s.value.func.attr == "set_position_target_local_ned_send"
                for s in n.body
            )
        )
        compiled = compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec")
        for kind, phase in (("random", "flying"), ("hover", "flying"), ("random", "streaming")):
            sent = []
            spec = route() if kind == "random" else sample("hover_normal", 17)
            env = dict(
                __package__="capstone_sim",
                math=math,
                phase=phase,
                now=20.0,
                reached_t=1.0,
                next_send_us=0,
                cfg={"mission": "hover"},
                normal_variation=spec,
                conn=SimpleNamespace(
                    target_system=1,
                    mav=SimpleNamespace(
                        set_position_target_local_ned_send=lambda *args: sent.append(args),
                        heartbeat_send=lambda *args: None,
                    ),
                ),
            )
            exec(compiled, env)
            message = sent[0]
            if kind == "random" and phase == "flying":
                _, v, a = kinematics(19, spec["path"])
                self.assertEqual(message[4], 2048)
                self.assertEqual(message[8:11], v[:3])
                self.assertEqual(message[11:14], a[:3])
                self.assertLessEqual(abs(message[14]), math.pi)
            else:
                self.assertEqual(message[4], 2552)
                self.assertEqual(message[8:14], (0.0,) * 6)


if __name__ == "__main__":
    unittest.main()
