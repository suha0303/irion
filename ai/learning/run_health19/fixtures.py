"""Deterministic synthetic *software tests*, never real SITL or detection evidence."""

import numpy as np

from .data import Run


def normal_runs(role, count, *, seed=0):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        n = 205 + 17 * i
        t = np.arange(n) / 10
        x = rng.normal(0, 0.05, (n, 19))
        current = 20 + 2 * np.sin(t / 2) + rng.normal(0, 0.2, n)
        x[:, :4] = 0.5 + 0.002 * current[:, None] + rng.normal(0, 0.005, (n, 4))
        x[:, 17] = 4 * (4.1 - 0.002 * current - 0.0001 * t + rng.normal(0, 0.001, n))
        x[:, 18] = current
        name = f"synthetic-{role}-{seed}-{i}"
        result.append(
            Run(
                name,
                role,
                (name,),
                "confirmed_healthy",
                "synthetic_unit_fixture",
                x,
                np.arange(n, dtype=np.int64) * 100000,
            )
        )
    return result
