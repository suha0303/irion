"""FINAL16 fixtures: independent scope, actual shared19 case engine, never real flights."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_short_cycle_inputs import refresh, short_request, write
from test_wind_metadata import fixture as wind_fixture

from learning.admission19 import FINAL16, check_protection, validate_lineages
from learning.history19 import final16
from learning.history19.pilot import CRITICAL, ROOT, json_file, sha
from learning.history19.pilot_connector import (
    native_mission_interval,
    project_pair_receipt,
)


@pytest.fixture
def final_request(tmp_path, monkeypatch):
    path = short_request(tmp_path / "unit")
    old = json_file(path)
    refs = old["refs"]

    def read_ref(key):
        p = Path(refs[key]["path"])
        return json_file(p if p.is_absolute() else path.parent / p)

    def put(key, value):
        refs[key] = write(path.parent / ("final16-" + key + ".json"), value)

    producer = read_ref("expected_producer")
    auth = json_file(final16.AUTH)
    auth.update(
        fixture_only=True,
        producer=dict(
            source_sha256=producer["source_manifest_sha256"],
            runtime_sha256=producer["runtime_sha256"],
        ),
    )
    auth_path = path.parent / "authorization.json"
    write(auth_path, auth)
    monkeypatch.setattr(final16, "AUTH", auth_path)
    monkeypatch.setattr(final16, "AUTH_SHA256", sha(auth_path))
    definitions = json_file(final16.EXPECTED)
    definitions["authorization_sha256"] = sha(auth_path)
    definitions_path = path.parent / "expected.json"
    write(definitions_path, definitions)
    monkeypatch.setattr(final16, "EXPECTED", definitions_path)
    monkeypatch.setattr(final16, "EXPECTED_SHA256", sha(definitions_path))
    policy_path = path.parent / "final16-policy.json"
    final16.prepare_policy(
        refs["native_policy"]["path"],
        refs["expected_producer"]["path"],
        policy_path,
        fixture_only=True,
    )
    policy = json_file(policy_path)
    refs["paired_policy"] = dict(path=str(policy_path), sha256=sha(policy_path))
    manifest = final16.expected_plan("final16-unit", fixture_only=True)
    put("pilot_manifest", manifest)
    case, partner = manifest["cases"][:2]
    normal = Path(old["case_origin"]["run"]["path"]).parent
    anomaly = normal.parent / "synthetic-0-anomaly"
    source = next(
        c for c in definitions["cases"] if c["case_id"].endswith("01-normal")
    )
    scenario = wind_fixture(case["environment"], case["motion"])
    scenario.update(
        operation_variation=source["operation_variation"],
        wind_profile=source["operation_variation"]["wind_profile"],
        seed=case["seed"],
        paired_case_id=case["pair_id"],
        health_eligibility="confirmed_healthy",
        requested_duration_s=180,
        producer_lock_sha256=producer["runtime_sha256"],
        joint_phase_policy=json_file(normal / "evaluation/scenario.json")[
            "joint_phase_policy"
        ],
    )
    scenario["wind_model"].update(
        profile=scenario["wind_profile"], clock=policy["wind_model_clock"]
    )
    write(normal / "evaluation/scenario.json", scenario)
    other = json_file(anomaly / "evaluation/scenario.json")
    other.update(
        seed=case["seed"],
        operation_variation=scenario["operation_variation"],
        wind_profile=scenario["wind_profile"],
        paired_case_id=case["pair_id"],
    )
    write(anomaly / "evaluation/scenario.json", other)
    refresh(normal)
    refresh(anomaly)
    refs["scenario"] = dict(
        path=str(normal / "evaluation/scenario.json"),
        sha256=sha(normal / "evaluation/scenario.json"),
    )
    refs["raw_integrity"] = dict(
        path=str(normal / "integrity.json"),
        sha256=sha(normal / "integrity.json"),
    )
    normal_id, anomaly_id = (
        json_file(normal / "run.json")["run_id"],
        json_file(anomaly / "run.json")["run_id"],
    )
    ledger = dict(
        schema="drone-ai.final16.ledger/1", task_id=final16.TASK, cases=[]
    )
    registry = dict(schema="drone-ai.global-protection/1", entries=[])
    for i, c in enumerate(manifest["cases"]):
        attempts = []
        if i < 2:
            rid = (normal_id, anomaly_id)[i]
            attempts = [
                dict(
                    index=1,
                    run_id=rid,
                    parent_attempt_id=None,
                    status="completed",
                    seed=c["seed"],
                    lineage=c["lineage"],
                )
            ]
            registry["entries"].append(
                dict(
                    identities=["run:" + rid],
                    forbidden_uses=["fit", "calibration", "final_test"],
                    purpose=FINAL16,
                )
            )
        ledger["cases"].append(
            dict(
                case_id=c["case_id"],
                status="completed" if i < 2 else "pending",
                attempts=attempts,
            )
        )
        registry["entries"].append(
            dict(
                identities=["case:" + c["case_id"]],
                forbidden_uses=["fit", "calibration", "final_test"],
                purpose=FINAL16,
            )
        )
    put("ledger", ledger)
    put("registry", registry)
    interval = native_mission_interval(
        normal / "evaluation/joint-origin.json",
        policy_path=policy_path,
        native_policy_path=refs["native_policy"]["path"],
        run_path=normal / "run.json",
    )
    for key in ("online_manifest", "run_manifest"):
        v = read_ref(key)
        row = v["run"] if key == "online_manifest" else v["runs"][0]
        extras = [
            x for x in row["lineage"] if x.startswith(("run:", "raw-sha256:"))
        ]
        row.update(
            case_id=case["case_id"],
            purpose=FINAL16,
            cohort=FINAL16,
            lineage=case["lineage"] + extras,
            mission_interval=interval,
        )
        row["source_evidence"]["scenario_name"] = scenario["name"]
        put(key, v)
    capsule = read_ref("capsule")
    capsule.update(
        paired_policy_sha256=sha(policy_path),
        files={
            n: sha(ROOT / n)
            for n in (*CRITICAL, "Data/generation/pretrain/common.py")
        },
    )
    put("capsule", capsule)
    report = normal.parent / "pair-0.json"
    # The native fixture names its pair report here; discover only direct JSON files.
    report = next(
        p
        for p in normal.parent.glob("*.json")
        if json_file(p).get("schema") == "drone.joint-phase-pair-check/1"
    )
    projected = project_pair_receipt(report, policy_path=policy_path)
    put(
        "pair_receipt",
        dict(
            projected,
            schema="drone-ai.pilot-pair-check/1",
            producer=producer,
            pair_id=case["pair_id"],
            paired_policy_sha256=sha(policy_path),
            native_evidence=dict(path=str(report), sha256=sha(report)),
        ),
    )
    old.update(
        schema="drone-ai.final16.check/1",
        task_id=final16.TASK,
        purpose=FINAL16,
        case_id=case["case_id"],
    )
    write(path, old)
    return path


def edit(path, key, change):
    final_request = json_file(path)
    p = Path(final_request["refs"][key]["path"])
    value = json_file(p)
    change(value)
    final_request["refs"][key] = write(p, value)
    write(path, final_request)


def run(path, capsys):
    code = final16.main(path)
    return code, json.loads(capsys.readouterr().out)


def test_shared19_full_case_and_native_pair_remain_causal_and_nonempty(
    final_request, capsys
):
    code, result = run(final_request, capsys)
    assert code == 0, result
    assert result["status"] == "SYNTHETIC_FINAL16_CASE_INPUTS_CHECKED_NO_FIT"
    assert result["frames"] == 1801 and result["online"]["windows"] == 170
    assert (
        result["pair_verified"] is True
        and result["actual_data_verified"] is False
    )
    assert result["ledger_counts"]["planned_cases"] == 16
    assert result["ledger_counts"]["physical_attempts"] == 2


@pytest.mark.parametrize(
    "damage",
    [
        "drop_case",
        "duplicate",
        "seed",
        "operation",
        "group",
        "old_namespace",
        "old45_schema",
        "retry",
        "parent",
        "bool_index",
        "quality_retry",
        "missing_exclusion",
        "old_data_usage",
        "scope",
        "auth",
        "producer",
    ],
)
def test_final16_scope_cannot_weaken_old_or_new_limits(
    final_request, capsys, damage
):
    if damage in {
        "drop_case",
        "duplicate",
        "seed",
        "operation",
        "group",
        "old_namespace",
        "old45_schema",
    }:

        def change(m):
            if damage == "drop_case":
                m["cases"].pop()
            if damage == "duplicate":
                m["cases"][-1] = m["cases"][0]
            if damage == "seed":
                m["cases"][0]["seed"] += 1
            if damage == "operation":
                m["cases"][0]["operation_sha256"] = "0" * 64
            if damage == "group":
                m["cases"][0]["lineage"] = []
            if damage == "old_namespace":
                m["cases"][0]["case_id"] = "short180-old:final16-pair-01-normal"
            if damage == "old45_schema":
                m["schema"] = "drone-ai.pilot45.manifest/1"

        edit(final_request, "pilot_manifest", change)
    if damage in {"retry", "parent", "bool_index", "quality_retry"}:

        def change(ledger):
            a = ledger["cases"][0]["attempts"][0]
            if damage == "retry":
                ledger["cases"][0]["attempts"].append(copy.deepcopy(a))
            if damage == "parent":
                a["parent_attempt_id"] = "prior45-raw"
            if damage == "bool_index":
                a["index"] = True
            if damage == "quality_retry":
                a["status"] = "quality_rejected"

        edit(final_request, "ledger", change)
    if damage == "missing_exclusion":
        edit(final_request, "registry", lambda v: v["entries"].pop())
    if damage == "old_data_usage":
        edit(
            final_request,
            "registry",
            lambda v: v["entries"].append(
                dict(
                    purpose="prior45",
                    identities=["run:prior45"],
                    forbidden_uses=["fit"],
                )
            ),
        )
    if damage == "scope":
        edit(
            final_request,
            "paired_policy",
            lambda p: p.update(scope_task="SHORT_CYCLE_20261107"),
        )
    if damage == "auth":
        edit(
            final_request,
            "paired_policy",
            lambda p: p.update(authorization_sha256="0" * 64),
        )
    if damage == "producer":
        edit(
            final_request,
            "expected_producer",
            lambda p: p.update(runtime_sha256="0" * 64),
        )
    code, result = run(final_request, capsys)
    assert code == 1 and result["actual_data_verified"] is False


def test_physical_failure_stays_in_sixteen_denominator_and_cannot_pass_final(
    final_request, capsys
):
    def failure(ledger):
        row = ledger["cases"][0]
        row["status"] = row["attempts"][0]["status"] = "failed"
        row["attempts"][0]["failure_reason"] = (
            "synthetic launch failure retained"
        )

    edit(final_request, "ledger", failure)
    r = json_file(final_request)
    r["phase"] = "prerequisites"
    write(final_request, r)
    code, result = run(final_request, capsys)
    assert code == 0 and result["ledger_counts"]["planned_cases"] == 16
    assert result["ledger_counts"]["physical_failed_attempts"] == 1
    r["phase"] = "final"
    write(final_request, r)
    code, result = run(final_request, capsys)
    assert (
        code == 2 and result["ledger_counts"]["physical_failed_attempts"] == 1
    )


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "drop",
        "reused_pair",
        "windows0",
        "fixture",
        "failed",
        "old_scope",
        "producer",
        "duplicate",
    ],
)
def test_final_requires_all_sixteen_and_eight_distinct_actual_pair_receipts(
    final_request, capsys, damage
):
    from learning.history19.pilot import Evidence

    ctx = Evidence(final_request)
    _, _, _, cases, _, _ = final16.prerequisites(ctx)
    r = json_file(final_request)
    ledger = json_file(r["refs"]["ledger"]["path"])
    registry = json_file(r["refs"]["registry"]["path"])
    refs = []
    for i, (cid, case) in enumerate(cases.items()):
        rid = "synthetic-final16-" + str(i)
        ledger["cases"][i].update(
            status="completed",
            attempts=[
                dict(
                    index=1,
                    run_id=rid,
                    parent_attempt_id=None,
                    status="completed",
                    seed=case["seed"],
                    lineage=case["lineage"],
                )
            ],
        )
        registry["entries"].append(
            dict(
                identities=["run:" + rid],
                forbidden_uses=["fit", "calibration", "final_test"],
                purpose=FINAL16,
            )
        )
        evidence = {
            k: ctx.hashes[k]
            for k in (
                "source_manifest",
                "runtime_manifest",
                "paired_policy",
                "capsule",
                "final16_authorization",
                "final16_expected_cases",
            )
        }
        evidence["pilot_pair_native_evidence"] = hashlib.sha256(
            case["pair_id"].encode()
        ).hexdigest()
        receipt = dict(
            schema="drone-ai.final16.check-receipt/1",
            task_id=final16.TASK,
            purpose=FINAL16,
            phase="case",
            status="FINAL16_CASE_INPUTS_CHECKED_NO_FIT",
            actual_data_verified=True,
            fixture_only=False,
            exit_code=0,
            duration_profile="short180",
            mission_duration_us=180000000,
            frames=1801,
            online=dict(windows=170),
            pair_verified=True,
            case_id=cid,
            run_id=rid,
            evidence_sha256=evidence,
            model_forward_steps=0,
            optimizer_steps=0,
            scaler_threshold_fits=0,
            GPU_jobs=0,
        )
        # Inner metadata exercises aggregation; the outer fixture remains synthetic.
        if damage == "reused_pair":
            evidence["pilot_pair_native_evidence"] = "0" * 64
        if i == 0:
            if damage == "windows0":
                receipt["online"]["windows"] = 0
            if damage == "fixture":
                receipt["fixture_only"] = True
            if damage == "old_scope":
                receipt["task_id"] = "SHORT_CYCLE_20261107"
            if damage == "producer":
                evidence["runtime_manifest"] = "0" * 64
            if damage == "failed":
                ledger["cases"][i]["status"] = ledger["cases"][i]["attempts"][
                    0
                ]["status"] = "failed"
                ledger["cases"][i]["attempts"][0]["failure_reason"] = (
                    "preserved failure"
                )
        refs.append(
            write(
                final_request.parent / ("receipt-" + str(i) + ".json"), receipt
            )
        )
    r["refs"]["ledger"] = write(Path(r["refs"]["ledger"]["path"]), ledger)
    r["refs"]["registry"] = write(Path(r["refs"]["registry"]["path"]), registry)
    r.update(
        phase="final", case_receipts=refs[:-1] if damage == "drop" else refs
    )
    if damage == "duplicate":
        r["case_receipts"][-1] = refs[0]
    write(final_request, r)
    code, result = run(final_request, capsys)
    assert code == (0 if damage is None else 1), result
    assert result["actual_data_verified"] is False
    if damage is None:
        assert result["cases_checked"] == 16 and result["pairs_checked"] == 8
        assert result["ledger_counts"]["physical_attempts"] == 16


def test_final16_cannot_enter_fit_or_be_relabelled():
    row = dict(
        run_id="fixture",
        lineage=["run:fixture"],
        role="normal_test",
        cohort=FINAL16,
        purpose=FINAL16,
    )
    with pytest.raises(ValueError):
        check_protection(
            [row],
            dict(schema="drone-ai.global-protection/1", entries=[]),
            usage="fit",
        )
    with pytest.raises(ValueError):
        validate_lineages([dict(row, cohort="final")])
