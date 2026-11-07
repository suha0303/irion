"""Independent MAIN480 authority and frozen plan checks; no collection or fitting."""

from collections import Counter
from pathlib import Path

from learning.admission19 import PRODUCER_FIELDS, check_producer
from learning.history19.joint_native import ORIGIN, WIND_CLOCK, check_policy
from learning.history19.pilot import ROOT, json_file, regular, sha

TASK = "MAIN_COLLECTION_CONNECTION_20261107"
AUTH = (
    ROOT
    / "ai/reports/collection-hardening-20261108/attempt-01/authorization.json"
)
AUTH_SHA256 = "3a1e5d7952af7eb38ec0bf1f5afb5cc97200b3413d79a31f4960d18db470933f"
COUNTS = dict(
    normal_train=540,
    normal_validation=81,
    normal_calibration=81,
    development=243,
    final=729,
)


def bound(ref):
    path = regular(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("Collection evidence changed: " + str(path))
    return path


def authorization():
    if sha(regular(AUTH)) != AUTH_SHA256:
        raise ValueError("MAIN collection implementation authority changed")
    value = json_file(AUTH)
    scope = value.get("user_run_scope", {})
    if (
        value.get("task_id") != TASK
        or scope.get("menu2_cases") != 1674
        or scope.get("main_cohorts") != COUNTS
        or scope.get("lr_subset_roles")
        != dict(normal_train=81, normal_validation=27)
        or scope.get("mission_seconds") != 480
        or scope.get("training_or_model_forward_or_GPU") is not False
        or scope.get("final729_locked_for_performance") is not True
        or scope.get("prior_pilot_data_reuse") is not False
    ):
        raise ValueError(
            "Explicit unchanged MAIN1674/LR108/no-model authority required"
        )
    return value


def check_scope():
    """Read bounded plan metadata only; historical execution_ready flags stay unchanged."""
    auth = authorization()
    plans = auth["frozen_plans"]
    main, lr = json_file(bound(plans["main"])), json_file(bound(plans["LR"]))
    bound(plans["prior_plan_approval"])
    cases = main["cases"]
    by_id = {c["case_id"]: c for c in cases}
    lr_ids = {c["case_id"] for c in lr["cases"]}
    if (
        len(cases) != len(by_id)
        or len(cases) != 1674
        or Counter(c["cohort"] for c in cases) != COUNTS
        or len(lr["cases"]) != len(lr_ids)
        or len(lr_ids) != 108
        or Counter(c["role"] for c in lr["cases"])
        != dict(normal_train=81, normal_validation=27)
        or any(c != by_id.get(c["case_id"]) for c in lr["cases"])
        or any(
            c.get("duration") != 480 or c.get("attempts_max") != 3
            for c in cases
        )
        or len({c["group_id"] for c in cases}) != 1350
        or len({c["group_id"] for c in cases if c["category"] == "anomaly"})
        != 324
    ):
        raise ValueError(
            "Frozen1674 split/groups or same-role108 subset differs"
        )
    return dict(
        authorization=auth, main=main, lr=lr, cases=by_id, lr_ids=lr_ids
    )


def write_new(path, value):
    import json

    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("New regular collection output required")
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    return dict(path=str(path), sha256=sha(path))


def prepare_policy(
    native_policy_path, producer_binding_path, output, *, fixture_only=False
):
    scope = check_scope()
    auth = scope["authorization"]
    if (auth.get("fixture_only") is True) is not fixture_only:
        raise ValueError(
            "Synthetic policy requires an explicit isolated fixture authority"
        )
    native = json_file(native_policy_path)
    check_policy(native, duration_profile="standard480")
    producer = json_file(producer_binding_path)
    check_producer(producer, producer)
    producer = {key: producer[key] for key in PRODUCER_FIELDS}
    if (producer["source_manifest_sha256"], producer["runtime_sha256"]) != (
        auth["producer"]["source_sha256"],
        auth["producer"]["runtime_sha256"],
    ) or native["revision"] != "joint-native-consumption-c2-v2":
        raise ValueError(
            "Independent current-v2 collection producer binding required"
        )
    value = dict(
        schema="drone-ai.paired-source-clock-policy/1",
        status="FROZEN",
        scope_task=TASK,
        purpose="main_dataset_preparation",
        approved_for_collection=True,
        approved_for_pilot=False,
        approved_for_short_cycle=False,
        approved_for_final16=False,
        authorization_sha256=AUTH_SHA256,
        frozen_plans=auth["frozen_plans"],
        duration_profile="standard480",
        policy_id=native["revision"],
        native_adapter=native["revision"],
        alignment_mode="versioned_joint_phase",
        mission_policy="mission_only_actual_execution/1",
        posthoc_adjustment=False,
        producer=producer,
        native_policy_sha256=sha(regular(native_policy_path)),
        native_consumption_max_gap_us=native["acceptance"][
            "native_consumption_max_gap_us"
        ],
        native_receipt_schema="drone.joint-phase-native-receipt/1",
        mission_source=ORIGIN,
        wind_model_clock=WIND_CLOCK,
        fixture_only=fixture_only,
        limits=dict(
            origin_delta_us=0,
            wind_rms_mps=2e-9,
            wind_peak_mps=2e-9,
            receipt_min_offset_us=0,
            receipt_max_offset_us=500000,
        ),
        prior_plan_native_policy_note="Frozen LR v1/execution_ready=false is historical planning metadata; this separate contract binds actual current v2. Original plans are unchanged.",
        authorizes_automatic_start=False,
        authorizes_training=False,
    )
    write_new(output, value)
    return value


def validate_policy(policy, native_policy_path):
    auth = authorization()
    native = json_file(native_policy_path)
    check_policy(native, duration_profile="standard480")
    check_producer(policy["producer"], policy["producer"])
    if (
        policy.get("scope_task") != TASK
        or policy.get("approved_for_collection") is not True
        or policy.get("purpose") != "main_dataset_preparation"
        or policy.get("authorization_sha256") != AUTH_SHA256
        or policy.get("frozen_plans") != auth["frozen_plans"]
        or policy.get("duration_profile") != "standard480"
        or policy.get("native_adapter") != "joint-native-consumption-c2-v2"
        or policy.get("native_policy_sha256")
        != sha(regular(native_policy_path))
        or policy.get("native_consumption_max_gap_us", 40000)
        != native["acceptance"]["native_consumption_max_gap_us"]
        or policy["producer"]["source_manifest_sha256"]
        != auth["producer"]["source_sha256"]
        or policy["producer"]["runtime_sha256"]
        != auth["producer"]["runtime_sha256"]
        or policy.get("fixture_only") is not (auth.get("fixture_only") is True)
    ):
        raise ValueError("Independent MAIN480 execution policy differs")
    return policy


def case_key(case_id):
    if not isinstance(case_id, str):
        raise ValueError("Explicit frozen case ID required")
    return case_id.rsplit(":", 1)[-1]


def check_case_record(row, case):
    for key in ("role", "cohort", "purpose"):
        if row.get(key) != case[key]:
            raise ValueError(
                "Frozen collection role/cohort/purpose changed: " + key
            )
    if case_key(row.get("case_id")) != case["case_id"]:
        raise ValueError("Run does not belong to the frozen logical case")
    namespace = row["case_id"][: -len(case["case_id"])]
    if row.get("paired_case_id") != case["paired_case_id"]:
        raise ValueError("Frozen paired identity changed")
    for key in ("group_id", "retry_family_id"):
        if row.get(key) != namespace + case[key]:
            raise ValueError("Frozen group/retry identity changed")


def read_protection(value, identities=None):
    """Old JSON registries or bounded JSONL entries; never an unbounded global ledger."""
    import json

    if isinstance(value, dict):
        return value
    path = regular(value)
    if path.suffix != ".jsonl":
        return json_file(path)
    entries = []
    with path.open() as stream:
        for line in stream:
            if len(line) > 65536:
                raise ValueError("Protection JSONL entry exceeds64KiB")
            row = json.loads(line)
            if (
                identities is None
                or set(row.get("identities", [])) & identities
            ):
                entries.append(row)
    return dict(schema="drone-ai.global-protection/1", entries=entries)
