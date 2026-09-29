"""Execute the real flight-loop command guard/deadline without starting SITL."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def schedule():
    tree = ast.parse((ROOT / "simulation/sim/src/capstone_sim/runner.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_flight"
    )
    block = next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.If)
        and any(
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Attribute)
            and s.value.func.attr == "set_position_target_local_ned_send"
            for s in n.body
        )
    )
    update = next(
        n
        for n in block.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id.startswith("next_send") for t in n.targets)
    )
    return compile(ast.Expression(block.test), "real_command_guard", "eval"), compile(
        ast.Module(body=[update], type_ignores=[]), "real_command_deadline", "exec"
    )


class CommandCadenceTest(unittest.TestCase):
    def environment(self):
        self.guard, self.update = schedule()
        self.wall = 0.0
        return dict(
            phase="flying",
            now=0.0,
            next_send=0.0,
            next_send_us=0,
            time=SimpleNamespace(monotonic=lambda: self.wall),
        )

    def emit(self, env, now, wall):
        env["now"], self.wall = now, wall
        if eval(self.guard, env):
            exec(self.update, env)
            return True
        return False

    def test_same_simulation_command_grid_at_two_four_eight(self):
        traces = []
        for factor in (2, 4, 8):
            env = self.environment()
            sent = []
            for sim_us in range(0, 2_000_000, 1000):
                if self.emit(env, sim_us / 1e6, sim_us / 1e6 / factor):
                    sent.append(sim_us)
            self.assertEqual(sent, list(range(0, 2_000_000, 100_000)))
            traces.append(sent)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[1], traces[2])

    def test_paused_simulation_does_not_send_again_as_wall_time_passes(self):
        env = self.environment()
        self.assertTrue(self.emit(env, 0.1, 1.0))
        self.assertFalse(self.emit(env, 0.1, 10.0))
        self.assertFalse(self.emit(env, 0.1, 100.0))
        self.assertFalse(self.emit(env, 0.199, 101.0))
        self.assertTrue(self.emit(env, 0.2, 102.0))

    def test_delay_skips_old_slots_without_burst_or_deadline_drift(self):
        env = self.environment()
        self.assertTrue(self.emit(env, 0.12, 1.0))
        self.assertTrue(self.emit(env, 0.47, 1.01))
        self.assertFalse(self.emit(env, 0.47, 10.0))
        self.assertFalse(self.emit(env, 0.499, 11.0))
        self.assertTrue(self.emit(env, 0.5, 12.0))
        env["phase"] = "landing"
        self.assertFalse(self.emit(env, 0.6, 13.0))


if __name__ == "__main__":
    unittest.main()
