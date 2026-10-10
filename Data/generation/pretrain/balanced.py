"""Balanced300 design: finite roles, stratified healthy conditions and frozen paths."""

import copy
import math
import random
import sys
from collections import Counter

from .common import RESOURCES, ROOT, read

ROLES = {
    "normal_train": 180,
    "normal_validation": 26,
    "normal_calibration": 26,
    "normal_dev_eval": 32,
    "anomaly_dev": 36,
}
MODES = {
    "normal_train": (72, 54, 54),
    "normal_validation": (11, 7, 8),
    "normal_calibration": (10, 8, 8),
    "normal_dev_eval": (10, 8, 8),
}


def environment(spec, scenario, index, count, design_seed, seed):
    result = copy.deepcopy(spec)
    ranges = dict(
        soc=(0.87, 0.95),
        temperature=(293.15, 303.15),
        payload=(0, 0.08),
        speed=(0, 0.7),
        direction=(-math.pi, math.pi),
        gust_sigma=(0, 0.12),
    )
    if scenario == "cold_normal":
        ranges.update(soc=(0.9, 0.95), temperature=(283.15, 291.15), payload=(0, 0.05))
    elif scenario == "low_soc_normal":
        ranges.update(soc=(0.4, 0.5), payload=(0, 0.05), speed=(0, 0.4))
    elif scenario == "payload_normal":
        ranges["payload"] = (0.1, 0.2)
    elif scenario == "wind_normal":
        ranges.update(speed=(0.5, 2), gust_sigma=(0.1, 0.4))
    values = {}
    for k, (key, (low, high)) in enumerate(ranges.items()):
        rng = random.Random(design_seed + 7919 * k)
        bins = list(range(count))
        rng.shuffle(bins)
        u = random.Random(seed + 1543 * k).random()
        values[key] = low + (high - low) * (bins[index] + u) / count
    speed, direction = values.pop("speed"), values.pop("direction")
    result.update(values, wind_e=speed * math.cos(direction), wind_n=speed * math.sin(direction))
    return result


def cases(seed_base):
    sys.path.insert(0, str(ROOT / "simulation/sim/src"))
    from capstone_sim.flight_program import build
    from capstone_sim.normal_variation import sample

    templates = read(RESOURCES / "case_manifest.json")["cases"]
    normal = {c["scenario"]: c for c in templates if c["role"] == "normal_train"}
    conditions = sorted(normal)
    rng = random.Random(seed_base ^ 300)
    rows = []
    serial = 0
    basic_index = 0
    for role_index, (role, quota) in enumerate(MODES.items()):
        count = sum(quota)
        order = [conditions[i % 9] for i in range(count)]
        if role != "normal_train":
            order = conditions * 2 + [conditions[(i + role_index * 3) % 9] for i in range(8)]
        modes = ["basic"] * quota[0] + ["mixed"] * quota[1] + ["random"] * quota[2]
        if role == "normal_train":
            reserved = ["basic"] * 9 + ["mixed"] * 9 + ["random"] * 9
            rest = (
                ["basic"] * (quota[0] - 9)
                + ["mixed"] * (quota[1] - 9)
                + ["random"] * (quota[2] - 9)
            )
            rng.shuffle(rest)
            modes = reserved + rest
        else:
            rng.shuffle(modes)
        totals = Counter(order)
        indices = Counter()
        for scenario, mode in zip(order, modes):
            seed = seed_base + serial
            serial += 1
            row = copy.deepcopy(normal[scenario])
            spec = environment(
                sample(scenario, seed),
                scenario,
                indices[scenario],
                totals[scenario],
                seed_base + role_index * 10000 + conditions.index(scenario) * 101,
                seed,
            )
            indices[scenario] += 1
            row.update(
                case_id=f"B300_{role}_{scenario}_{seed}",
                role=role,
                seed=seed,
                group_id=f"B300-normal-{seed}",
                composition_seed=seed_base,
                path_mode=mode,
                normal_variation=build(spec, mode, seed, row["mission_seconds"], basic_index),
            )
            if mode == "basic":
                basic_index += 1
            rows.append(row)
    anomaly = {c["scenario"]: c for c in templates if c["role"] == "anomaly_dev"}
    control = next(
        c for c in templates if c["role"] == "normal_dev_eval" and c["mission_seconds"] == 480
    )
    for pair in range(6):
        seed = seed_base + 258 + pair
        for template in [control] + [anomaly[s] for s in sorted(anomaly)]:
            row = copy.deepcopy(template)
            row.update(
                case_id=f"B300_{row['scenario']}_{seed}",
                seed=seed,
                group_id=f"B300-development-{seed}",
                composition_seed=seed_base,
            )
            row.pop("normal_variation", None)
            rows.append(row)
    rng.shuffle(rows)
    for i, row in enumerate(rows, 1):
        row["generation_index"] = i
    return rows


def validate(cases_record):
    if len(cases_record) != 300 or Counter(c["role"] for c in cases_record) != Counter(ROLES):
        raise ValueError("Balanced300 requires roles180/26/26/32/36")
    base = cases_record[0].get("composition_seed")
    if type(base) is not int or cases_record != cases(base):
        raise ValueError("Balanced300 membership/paths differ from frozen deterministic design")
