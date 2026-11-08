"""Build only the user-approved1674 plan. This module never starts
collection/training.
"""

import copy
import secrets
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import training19
from .common import ROOT, now, read, sha, verify, write

AUTH = (
    ROOT / "Data/operations/training19/authorizations/main1674-20261107-01.json"
)
DEFAULT_PLAN = training19.PLANS / "main1674-20261107-01"


def approval():
    if not AUTH.is_file():
        raise ValueError(
            "User final1674 approval required before main manifest creation"
        )
    value = read(AUTH)
    if (
        value.get("direct_user_answer") != "전체 1,674런 구성 승인"
        or value.get("approved_total") != 1674
        or value.get("preliminary_data_connected") is not False
    ):
        raise ValueError(
            "User final1674 approval required before main manifest creation"
        )
    if value.get("lr108_reuse_same_roles") != dict(
        normal_train=81, normal_validation=27
    ):
        raise ValueError("Exact LR train/validation subset approval required")
    return value


def build(lr, seed, excluded):
    """Independent groups get unique seeds; each evaluation pair shares one
    seed.
    """

    sys.path.insert(0, str(ROOT / "simulation/sim/src"))
    from capstone_sim import (
        bounded_wind,
        flight_program,
        normal_variation,
        random_flight,
    )

    rows = [copy.deepcopy(c) for c in lr["cases"]]
    used = set(excluded) | {c["seed"] for c in rows}
    counter = 0

    def fresh():
        nonlocal counter
        for _ in range(10000):
            counter += 1
            value = (seed + counter * 1009) % 2**32
            if value not in used:
                used.add(value)
                return value
        raise ValueError("Main seed allocation exhausted")

    def operation(kind, env, value):
        spec = normal_variation.sample("hover_normal", value)
        spec.update(
            soc=0.9,
            temperature=298.15,
            payload=0.0,
            wind_e=0.0,
            wind_n=0.0,
            gust_sigma=0.0,
            wind_profile=bounded_wind.profile(env, value % 360, 1.0),
        )
        mode = kind if kind in {"mixed", "random"} else "basic"
        return flight_program.build(
            spec,
            mode,
            value,
            480,
            flight_program.BASIC.index(kind) if mode == "basic" else 0,
            random_revision=random_flight.BOUNDED_REVISION,
        )

    def normal(role, cohort, repetitions, offset=0):
        for repeat in range(offset, offset + repetitions):
            for mi, kind in enumerate(training19.KINDS):
                for ei, env in enumerate(training19.ENVIRONMENTS):
                    value = fresh()
                    cid = f"main-{cohort}-{mi:02}-{ei}-{repeat}"
                    rows.append(
                        dict(
                            case_id=cid,
                            role=role,
                            cohort=cohort,
                            purpose="main_dataset_preparation",
                            category="standalone_normal",
                            group_id=cid,
                            retry_family_id=cid,
                            paired_case_id=f"x500-{kind}-seed{value}",
                            seed=value,
                            kind=kind,
                            environment=env,
                            operation_variation=operation(kind, env, value),
                            scenario="hover_normal",
                            duration=480,
                            battery_stage=None,
                            fault_after_mission_s=None,
                            anomaly_family=None,
                            anomaly_timing=None,
                            attempts_max=3,
                        )
                    )

    normal("normal_train", "normal_train", 17, 3)
    normal("normal_validation", "normal_validation", 2, 1)
    normal("normal_calibration", "normal_calibration", 3)
    normal("normal_test", "development", 3)
    normal("normal_test", "final", 9)
    for cohort in ("development", "final"):
        for fi, family in enumerate(("battery", "propulsion", "combined")):
            for ti, timing in enumerate(("existing", "step", "ramp")):
                for mi, kind in enumerate(training19.KINDS):
                    winds = (
                        [(mi + fi + ti) % 3]
                        if cohort == "development"
                        else range(3)
                    )
                    for ei in winds:
                        env, value = training19.ENVIRONMENTS[ei], fresh()
                        group = f"main-{cohort}-{family}-{timing}-{mi:02}-{ei}"
                        op = operation(kind, env, value)
                        onset = (180, 240, 300)[
                            (mi // 3 + fi + 2 * ti) % 3
                            if cohort == "development"
                            else (mi + ei + fi + ti) % 3
                        ]
                        for abnormal in (False, True):
                            cid = group + (
                                "-anomaly" if abnormal else "-normal"
                            )
                            scenario = (
                                (
                                    (
                                        "battery_aging_"
                                        if family == "battery"
                                        else "propulsion_loss_"
                                    )
                                    + timing
                                )
                                if abnormal
                                else "hover_normal"
                            )
                            rows.append(
                                dict(
                                    case_id=cid,
                                    role="anomaly_test"
                                    if abnormal
                                    else "normal_test",
                                    cohort=cohort,
                                    purpose="model_development"
                                    if cohort == "development"
                                    else "independent_final_test",
                                    category="anomaly"
                                    if abnormal
                                    else "paired_normal",
                                    group_id=group,
                                    retry_family_id=cid,
                                    paired_case_id=f"x500-{kind}-seed{value}",
                                    seed=value,
                                    kind=kind,
                                    environment=env,
                                    operation_variation=copy.deepcopy(op),
                                    scenario=scenario,
                                    duration=480,
                                    battery_stage="rpt_16"
                                    if abnormal and family == "combined"
                                    else None,
                                    fault_after_mission_s=onset
                                    if abnormal and timing != "existing"
                                    else None,
                                    anomaly_family=family if abnormal else None,
                                    anomaly_timing=timing if abnormal else None,
                                    attempts_max=3,
                                )
                            )
    return rows


def validate(rows, lr):
    if len(rows) != 1674 or len({c["case_id"] for c in rows}) != 1674:
        raise ValueError("Full1674 unique case denominator required")
    if Counter(c["cohort"] for c in rows) != dict(
        normal_train=540,
        normal_validation=81,
        normal_calibration=81,
        development=243,
        final=729,
    ):
        raise ValueError("Approved main roles/cohorts changed")
    by_id = {c["case_id"]: c for c in rows}
    if any(by_id.get(c["case_id"]) != c for c in lr["cases"]):
        raise ValueError(
            "LR108 must be retained byte-equivalent in the same role/cohort"
        )
    ownership = {}
    groups = defaultdict(list)
    for c in rows:
        op = c["operation_variation"]
        if (
            c["duration"] != 480
            or c["attempts_max"] != 3
            or (op["soc"], op["temperature"], op["payload"])
            != (0.9, 298.15, 0.0)
        ):
            raise ValueError("Approved operation/duration/retry changed")
        for k in ("group_id", "seed", "paired_case_id"):
            key = (k, c[k])
            if ownership.setdefault(key, c["cohort"]) != c["cohort"]:
                raise ValueError("Cross-cohort planned identity leakage")
        groups[c["group_id"]].append(c)
    for members in groups.values():
        if len(members) == 1:
            if members[0]["category"] != "standalone_normal":
                raise ValueError("Unpaired anomaly/control")
            continue
        if len(members) != 2 or {c["category"] for c in members} != {
            "paired_normal",
            "anomaly",
        }:
            raise ValueError(
                "Exactly one independent normal control per anomaly"
            )
        if (
            any(
                members[0][k] != members[1][k]
                for k in (
                    "cohort",
                    "seed",
                    "paired_case_id",
                    "kind",
                    "environment",
                    "operation_variation",
                    "duration",
                )
            )
            or members[0]["retry_family_id"] == members[1]["retry_family_id"]
        ):
            raise ValueError("Pair conditions/retry identities changed")
    for cohort, quota in (
        ("normal_train", 20),
        ("normal_validation", 3),
        ("normal_calibration", 3),
        ("development", 3),
        ("final", 9),
    ):
        normal_cells = Counter(
            (c["kind"], c["environment"])
            for c in rows
            if c["cohort"] == cohort and c["category"] == "standalone_normal"
        )
        if len(normal_cells) != 27 or set(normal_cells.values()) != {quota}:
            raise ValueError(
                (
                    "Every standalone normal condition must retain its "
                    "approved quota"
                )
            )
    for cohort, expected in (("development", 81), ("final", 243)):
        selected = [c for c in rows if c["cohort"] == cohort]
        if Counter(c["category"] for c in selected) != dict(
            standalone_normal=expected, paired_normal=expected, anomaly=expected
        ):
            raise ValueError("Equal normal/control/anomaly quantities required")
        for fi in ("battery", "propulsion", "combined"):
            for timing in ("existing", "step", "ramp"):
                subset = [
                    c
                    for c in selected
                    if c["anomaly_family"] == fi
                    and c["anomaly_timing"] == timing
                ]
                if len(subset) != (9 if cohort == "development" else 27):
                    raise ValueError("Anomaly family/timing coverage changed")
                if (
                    cohort == "final"
                    and len({(c["kind"], c["environment"]) for c in subset})
                    != 27
                ):
                    raise ValueError(
                        "Final anomaly must cover all27 conditions"
                    )
    return rows


def prepare(path=DEFAULT_PLAN, lr_path=training19.DEFAULT_PLAN):
    approval()
    path = Path(path).absolute()
    if path.parent != training19.PLANS or path.is_symlink():
        raise ValueError("Registered main plan root required")
    lr_path = training19.prepare_plan(lr_path)
    lr = read(lr_path / "manifest.json")
    if path.exists():
        verify(
            {
                str(path / "manifest.json"): (path / "manifest.sha256")
                .read_text(encoding="utf-8")
                .strip()
            }
        )
        m = read(path / "manifest.json")
        validate(m["cases"], lr)
        return path
    seed = secrets.randbits(32)
    excluded = {s for scenario, s in training19.prior_seeds() if s is not None}
    rows = validate(build(lr, seed, excluded), lr)
    path.mkdir()
    manifest = dict(
        schema="drone-data.main1674.plan/1",
        status="APPROVED_PLAN_NOT_COLLECTED",
        purpose="main_dataset_preparation",
        cases=rows,
        seed_base=seed,
        approved_total=1674,
        reused_lr_cases=108,
        new_collection_cases=1566,
        lr_plan=str(lr_path),
        lr_manifest_sha256=sha(lr_path / "manifest.json"),
        lr_subset_ids=[c["case_id"] for c in lr["cases"]],
        approval=dict(path=str(AUTH), sha256=sha(AUTH)),
        role_counts=dict(Counter(c["cohort"] for c in rows)),
        collection_start_authorized=False,
        model_training_authorized=False,
        execution_ready=False,
        preliminary_data_connected=False,
        legacy_19plus6_connected=False,
        flight_workers=16,
        requested_real_time_factor=2,
        aggregate_nonflight_workers=4,
        created_utc=now(),
        final_evaluation_locked=True,
    )
    write(path / "manifest.json", manifest)
    (path / "manifest.sha256").write_text(
        sha(path / "manifest.json") + "\n", encoding="utf-8"
    )
    return path
