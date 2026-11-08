"""Approved LR/main dataset plans only. Collection and training are not
dispatched here.
"""

import argparse
import fcntl
import secrets
import sys
from collections import Counter
from pathlib import Path

from . import pilot45
from .common import HOME, ROOT, now, read, sha, verify, write

LR_PURPOSE = "learning_rate_selection"

PLANS = HOME.parent / "training19/plans"

DEFAULT_PLAN = PLANS / "lr108-20261107-01"

REPORT = ROOT / "docs/data/reports/training19-preparation-20261107/attempt-01"

KINDS = (
    "hover",
    "altitude",
    "line",
    "ellipse",
    "figure8",
    "lissajous",
    "yaw",
    "mixed",
    "random",
)

ENVIRONMENTS = ("calm", "low", "strong")


def cases(seed):
    sys.path.insert(0, str(ROOT / "simulation/sim/src"))
    from capstone_sim import (
        bounded_wind,
        flight_program,
        normal_variation,
        random_flight,
    )

    rows = []
    for role, repetitions, shift in (
        ("normal_train", 3, 0),
        ("normal_validation", 1, 3),
    ):
        for repeat in range(repetitions):
            for mi, kind in enumerate(KINDS):
                for ei, environment in enumerate(ENVIRONMENTS):
                    value = (
                        seed
                        + 1000003 * (repeat + shift)
                        + 1009 * (3 * mi + ei + 1)
                    ) % 2**32
                    spec = normal_variation.sample("hover_normal", value)
                    spec.update(
                        soc=0.9,
                        temperature=298.15,
                        payload=0.0,
                        wind_e=0.0,
                        wind_n=0.0,
                        gust_sigma=0.0,
                        wind_profile=bounded_wind.profile(
                            environment, value % 360, 1.0
                        ),
                    )
                    mode = kind if kind in {"mixed", "random"} else "basic"
                    operation = flight_program.build(
                        spec,
                        mode,
                        value,
                        480,
                        flight_program.BASIC.index(kind)
                        if mode == "basic"
                        else 0,
                        random_revision=random_flight.BOUNDED_REVISION,
                    )
                    cid = f"lr-{role}-{mi:02}-{ei}-{repeat}"
                    rows.append(
                        dict(
                            case_id=cid,
                            role=role,
                            cohort=role,
                            purpose=LR_PURPOSE,
                            category="standalone_normal",
                            group_id=cid,
                            retry_family_id=cid,
                            paired_case_id=f"x500-{kind}-seed{value}",
                            seed=value,
                            kind=kind,
                            environment=environment,
                            operation_variation=operation,
                            scenario="hover_normal",
                            duration=480,
                            battery_stage=None,
                            fault_after_mission_s=None,
                            anomaly_family=None,
                            anomaly_timing=None,
                            attempts_max=3,
                        )
                    )
    return rows


def validate(manifest):
    rows = manifest["cases"]
    if manifest.get("purpose") != LR_PURPOSE or rows != cases(
        manifest["seed_base"]
    ):
        raise ValueError(
            "Only the exact new LR108 normal81/27 plan is supported"
        )
    if Counter(c["role"] for c in rows) != dict(
        normal_train=81, normal_validation=27
    ):
        raise ValueError("LR split must be81/27")
    if any(
        len({c[k] for c in rows}) != 108
        for k in (
            "case_id",
            "seed",
            "group_id",
            "retry_family_id",
            "paired_case_id",
        )
    ):
        raise ValueError("Every LR flight has independent planned identities")
    return rows


def prior_seeds():
    values = {
        (r.get("scenario"), r.get("seed"))
        for r in pilot45.protection()["records"]
    }
    for p in [
        *(HOME / "tasks").glob("*/manifest.json"),
        *PLANS.glob("*/manifest.json"),
    ]:
        value = read(p)
        for row in value.get("cases", []):
            values.add((row.get("scenario"), row.get("seed")))
    return values


def prepare_plan(path=DEFAULT_PLAN, seed=None):
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("uint32 LR seed required")
    path = Path(path).absolute()
    if path.parent != PLANS or path.is_symlink():
        raise ValueError("Registered training19 plan root required")
    if path.exists():
        value = read(path / "manifest.json")
        validate(value)
        verify(
            {
                str(path / "manifest.json"): (path / "manifest.sha256")
                .read_text(encoding="utf-8")
                .strip()
            }
        )
        return path
    PLANS.mkdir(parents=True, exist_ok=True)
    with (PLANS / "planning.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prior = prior_seeds()
        for _ in range(128):
            value = seed if seed is not None else secrets.randbits(32)
            rows = cases(value)
            if not any((c["scenario"], c["seed"]) in prior for c in rows):
                break
            if seed is not None:
                raise ValueError(
                    "Explicit seed collides with previous data/plans"
                )
        else:
            raise ValueError("Unique LR seed allocation exhausted")
        path.mkdir()
        manifest = dict(
            schema="drone-data.lr108.plan/1",
            purpose=LR_PURPOSE,
            seed_base=value,
            status="PLANNED_NOT_COLLECTED",
            cases=rows,
            requested_mission_seconds=480,
            flight_workers=16,
            requested_real_time_factor=2,
            aggregate_nonflight_workers=4,
            mission_policy="mission_only_actual_execution/1",
            mission_boundary_source=(
                "fresh raw actual native start/AUTO_LAND; independent "
                "LR-purpose admission pending"
            ),
            native_phase_policy="joint-native-consumption-c2-v1",
            clock_definition_source=(
                "simulation/profiles/collection/joint-phase-policy.json"
            ),
            neural_features=19,
            legacy_context_features=0,
            existing_data_reused=False,
            preliminary_data_connected=False,
            execution_ready=False,
            lr_native_boundary_admission="PENDING_AI_INDEPENDENT_LR_API",
            aggregate_fit_input="PENDING_BOUNDED_WRITER_AND_ACTUAL_INTAKE",
            main_training_approved=False,
            collect_on_user_start_only=True,
            model_training_authorized=False,
        )
        validate(manifest)
        write(path / "manifest.json", manifest)
        (path / "manifest.sha256").write_text(
            sha(path / "manifest.json") + "\n", encoding="utf-8"
        )
        write(
            path / "authorization.json",
            dict(
                scope="LR108 preparation and user-start collection only",
                direct_user_instruction=(
                    "LR용108런이랑 본학습 준비해. 예비는 아예 연결하지 마. 본학습용은 "
                    "생성 전 최종승인."
                ),
                main_approved=False,
                training_authorized=False,
                prepared_utc=now(),
            ),
        )
    return path


def menu_lr():
    plan = prepare_plan()
    print("LR용108런: 정상train81 + validation27 /9기동×3바람 /각480초 /19입력")
    print("비행16·요청RTF2·비비행합계4 /예비자료 연결 없음 /학습 실행 없음")
    print("계획:", plan / "manifest.json")
    print(
        "구성 준비됨 · LR 독립 native 경계·bounded 입력 합산 인수 대기 · 수집/학습 시작 안 함"
    )


def create_task(plan=DEFAULT_PLAN):
    raise ValueError(
        "LR108 계획만 준비됐습니다. LR 독립 native 경계·bounded 합산 인수 전에는 수집을 제출하지 않습니다."
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "main-plan", "menu"))
    a = p.parse_args(argv)
    if a.action == "plan":
        print(prepare_plan())
    elif a.action == "menu":
        menu_lr()
    else:
        from . import training19_main

        if not training19_main.AUTH.exists():
            print(
                "본학습용: 사용자 최종 승인 대기. manifest/수집 작업은 생성하지 않았습니다."
            )
            print(REPORT / "MAIN_PROPOSAL.md")
        else:
            path = training19_main.prepare()
            print(
                (
                    "승인된 본학습세트1674런: "
                    "train540/validation81/calibration81/개발243/최종729"
                )
            )
            print(
                "LR108 같은역할 포함 · 추가1566런 · 예비연결없음 · 수집/학습 시작 안 함"
            )
            print(path / "manifest.json")


if __name__ == "__main__":
    main()
