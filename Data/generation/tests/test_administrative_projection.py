"""A UUID-less administrative record is retained separately, never invented
as physical.
"""

import copy

import pytest

from pretrain import common, pilot45, pilot_parallel, short_resume

REPORT = (
    common.ROOT / "docs/data/reports/short-cycle-20261107/attempt-01/resume-c05"
)
PREVIOUS = common.HOME / "tasks/short180-20261107-c04-45"


def proof_ref():
    p = REPORT / "not-submitted-proof.json"
    return dict(path=str(p), sha256=common.sha(p))


@pytest.mark.local_evidence
def test_c04_projection_keeps_original52_and_physical51():
    before = common.sha(PREVIOUS / "ledger.json")
    old = common.read(PREVIOUS / "ledger.json")
    projected = short_resume.physical_projection(
        PREVIOUS, dict(administrative_not_submitted=proof_ref()), old
    )
    assert len(old["attempts"]) == 52 and len(projected["attempts"]) == 51
    assert sum(not a.get("run_id") for a in old["attempts"]) == 1
    assert all(a.get("run_id") for a in projected["attempts"])
    assert len(projected["administrative_attempts"]) == 1
    record = projected["administrative_attempts"][0]
    assert record["original_flat_attempt"]["state"] == "RUNNING"
    assert record["original_flat_attempt"]["number"] == 2
    assert record["physical_budget_consumed"] is False
    assert len(projected["cases"]["s180-pair-00-normal"]["attempts"]) == 1
    assert (
        projected["cases"]["s180-pair-00-normal"]["attempts"][0]["run_id"]
        == "a92abf2e82ff4911a50d4cc966cf7d1e"
    )
    assert common.sha(PREVIOUS / "ledger.json") == before


@pytest.mark.local_evidence
@pytest.mark.parametrize(
    "mutation",
    ["absent_proof", "bad_hash", "budget_true", "wrong_record", "fake_uuid"],
)
def test_uncertified_or_mismatched_unknown_is_never_discarded(
    tmp_path, mutation
):
    old = common.read(PREVIOUS / "ledger.json")
    ref = proof_ref()
    value = dict(administrative_not_submitted=ref)
    if mutation == "absent_proof":
        value = {}
    elif mutation == "bad_hash":
        value["administrative_not_submitted"] = dict(ref, sha256="0" * 64)
    elif mutation == "fake_uuid":
        next(a for a in old["attempts"] if not a.get("run_id"))["run_id"] = (
            "invented-UUID"
        )
    else:
        proof = common.read(ref["path"])
        if mutation == "budget_true":
            proof["physical_budget_consumed"] = True
        else:
            proof["original_flat_attempt"]["number"] = 3
        p = tmp_path / "bad-proof.json"
        common.write(p, proof)
        value["administrative_not_submitted"] = dict(
            path=str(p), sha256=common.sha(p)
        )
    with pytest.raises(pilot45.Blocked):
        short_resume.physical_projection(PREVIOUS, value, copy.deepcopy(old))


@pytest.mark.local_evidence
@pytest.mark.parametrize("failure", [None, "failed", "unknown", "retry"])
def test_c05_counts_new_reused_and_administrative_records(
    failure,
):
    task = common.HOME / "tasks/short180-20261107-c05-45"
    cfg = common.read(task / "config.json")
    ledger = copy.deepcopy(common.read(task / "ledger.json"))
    pending = cfg["short_resume"]["pending_native_cases"]
    # Reconstruct the initial C05 counter fixture after the live C05 task has
    # completed.
    ledger["attempts"] = [
        a for a in ledger["attempts"] if a.get("execution_task") != str(task)
    ]
    for e in ledger["cases"].values():
        e["attempts"] = [
            a for a in e["attempts"] if a.get("execution_task") != str(task)
        ]
    for cid in pending:
        prior = ledger["cases"][cid]["attempts"]
        ledger["attempts"].append(
            dict(
                case_id=cid,
                execution_task=str(task),
                state="COMPLETED",
                number=len(prior) + 1,
            )
        )
    if failure == "failed":
        ledger["attempts"][-1]["state"] = "FAILED_PRESERVED"
    elif failure == "unknown":
        ledger["attempts"][-1]["state"] = "RUNNING"
    elif failure == "retry":
        ledger["attempts"].append(
            dict(ledger["attempts"][-1], number=2, state="FAILED_PRESERVED")
        )
    q = dict(
        status="PILOT45_QUALITY_PASSED_STOPPED",
        short_ai_final_passed=True,
        inherited_successes=31,
        planned_cases=45,
        completed_raw=45,
        prepared_and_audited=45,
        online_available=45,
        run_available=45,
        ai_case_checks_passed=45,
    )
    q.update(
        pilot_parallel.current_execution_counts(
            task, ledger, planned_case_ids=pending
        )
    )
    assert (
        q["historical_physical_attempts"] == 51
        and q["historical_failed_attempts"] == 17
    )
    assert (
        q["historical_quality_rejected_attempts"] == 3
        and q["administrative_not_submitted_attempts"] == 1
    )
    assert pilot_parallel.clean_short_inputs(q, ledger) is False
    assert pilot_parallel.current_execution_inputs_passed(q) is (
        failure is None
    )
    if failure is None:
        assert (
            len(ledger["attempts"]) == 65
            and q["physical_completed_attempts"] == 48
        )
