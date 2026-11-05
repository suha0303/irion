"""Continuation boundaries keep physical history separate from data rejection; no flights."""

import copy
from pathlib import Path

import pytest
from pretrain import common, pilot19, pilot45, pilot_bridge, pilot_parallel, short_resume


@pytest.mark.parametrize("kind", ["quality", "native"])
def test_exact_reviewed_parent_can_enter_once_without_reclassifying_physical_state(tmp_path, kind):
    cid = "case"
    old = dict(
        number=1,
        run_id="original-raw",
        state="COMPLETED" if kind == "quality" else "FAILED_PRESERVED",
        parent_attempt_id=None,
        retry_allowed=False,
    )
    previous = tmp_path / "previous"
    common.write(previous / "ledger.json", dict(cases={cid: dict(attempts=[old])}))
    proof = tmp_path / "accepted.json"
    common.write(proof, dict(fixture_only=True))
    ref = dict(path=str(tmp_path / "quality.json"), sha256="a" * 64)
    resume = dict(
        previous_task=str(previous),
        files={str(proof): common.sha(proof)},
        quality_rejections={cid: ref} if kind == "quality" else {},
        native_retries={cid: dict(run_id=old["run_id"])} if kind == "native" else {},
    )
    prior = dict(old, source_task=str(previous))
    if kind == "quality":
        prior["quality_rejection"] = ref
    before = copy.deepcopy(prior)
    assert short_resume.allowed(dict(short_resume=resume), dict(case_id=cid), prior)
    assert prior == before
    assert common.read(previous / "ledger.json")["cases"][cid]["attempts"] == [old]
    prior["number"] = 2
    with pytest.raises(pilot45.Blocked):
        short_resume.allowed(dict(short_resume=resume), dict(case_id=cid), prior)
    prior = dict(before, source_task="another-current-failure")
    assert short_resume.allowed(dict(short_resume=resume), dict(case_id=cid), prior) is False


def test_ordinary_accepted_completed_raw_has_no_retry_exception(tmp_path):
    cfg = dict(
        short_resume=dict(previous_task="previous", quality_rejections={}, native_retries={})
    )
    prior = dict(source_task="previous", state="COMPLETED", number=2, run_id="accepted-raw")
    assert short_resume.allowed(cfg, dict(case_id="accepted"), prior) is False


@pytest.mark.parametrize("mutation", [None, "missing", "new_failure", "extra_retry", "fake_reuse"])
def test_resume_current17_and_cumulative65_counts_keep_three_rejections_physical_completed(
    mutation,
):
    task = Path("current")
    accepted = [str(i) for i in range(28)]
    pending = [str(i) for i in range(28, 45)]
    history = [
        dict(case_id=accepted[i % 28], source_task="prior", number=1, state="FAILED_PRESERVED")
        for i in range(17)
    ]
    history += [
        dict(
            case_id=accepted[i % 28],
            source_task="prior",
            number=2,
            state="COMPLETED",
            **(dict(quality_rejection={"path": "proof", "sha256": "a" * 64}) if i < 3 else {}),
        )
        for i in range(31)
    ]
    current = [
        dict(case_id=cid, execution_task=str(task), number=2 if i < 4 else 1, state="COMPLETED")
        for i, cid in enumerate(pending)
    ]
    if mutation == "missing":
        current.pop()
    elif mutation == "new_failure":
        current[0]["state"] = "FAILED_PRESERVED"
    elif mutation == "extra_retry":
        current.append(dict(current[0], number=3))
    ledger = dict(
        cases={str(i): dict(inherited_from="prior") if i < 28 else {} for i in range(45)},
        attempts=history + current,
    )
    q = dict(
        status="PILOT45_QUALITY_PASSED_STOPPED",
        short_ai_final_passed=True,
        inherited_successes=28,
        planned_cases=45,
        completed_raw=45,
        prepared_and_audited=45,
        online_available=45,
        run_available=45,
        ai_case_checks_passed=45,
    )
    q.update(pilot_parallel.current_execution_counts(task, ledger, planned_case_ids=pending))
    if mutation == "fake_reuse":
        q["inherited_successes"] = 27
    assert q["historical_physical_attempts"] == 48
    assert q["historical_failed_attempts"] == 17
    assert q["historical_quality_rejected_attempts"] == 3
    assert pilot_parallel.clean_short_inputs(q, ledger) is False
    assert pilot_parallel.current_execution_inputs_passed(q) is (mutation is None)
    if mutation is None:
        assert len(ledger["attempts"]) == 65 and q["physical_completed_attempts"] == 48


def test_actual_C03_snapshot_projects_only_approved_three_and_preserves_original_ledger(tmp_path):
    previous = common.HOME / "tasks/short180-20261107-c03-45"
    task = tmp_path / "projection"
    cfg = common.read(previous / "config.json")
    ledger = common.read(previous / "ledger.json")
    handoff = common.read(
        common.ROOT / "ai/reports/quality-rejection-20261107/attempt-01/data-handoff.json"
    )
    refs = handoff["quality_rejection"]["receipts"]
    for e in ledger["cases"].values():
        for a in e["attempts"]:
            if a["run_id"] in refs:
                a["quality_rejection"] = refs[a["run_id"]]
    for name, data in (
        ("config.json", cfg),
        ("manifest.json", common.read(previous / "manifest.json")),
        ("ai-manifest.json", common.read(previous / "ai-manifest.json")),
        ("ledger.json", ledger),
        ("global-protection.json", common.read(previous / "global-protection.json")),
    ):
        common.write(task / name, data)
    before = common.sha(task / "ledger.json")
    directory = tmp_path / "snapshot"
    directory.mkdir()
    pilot_bridge.snapshot(task, directory)
    projected = common.read(directory / "ledger.json")
    rejections = [
        a for c in projected["cases"] for a in c["attempts"] if a["status"] == "quality_rejected"
    ]
    assert len(rejections) == 3 and {a["run_id"] for a in rejections} == set(refs)
    assert all(
        a["physical_status"] == "completed" and a["index"] == 1 and a["parent_attempt_id"] is None
        for a in rejections
    )
    assert common.sha(task / "ledger.json") == before
    assert sum(a["state"] == "COMPLETED" for a in ledger["attempts"]) == 31


def test_actual_28_reader_evidence_binds_exact_preserved_exports():
    previous = common.HOME / "tasks/short180-20261107-c03-45"
    ledger = common.read(previous / "ledger.json")
    handoff = common.read(
        common.ROOT / "ai/reports/quality-rejection-20261107/attempt-01/data-handoff.json"
    )
    resume = dict(
        accepted_cases=[cid for cid, e in ledger["cases"].items() if e.get("status") == "COMPLETE"]
    )
    item = short_resume.current_reader_evidence(handoff, previous, ledger, resume)
    assert common.sha(item["path"]) == item["sha256"]
    assert len(pilot19.helper_inventory()) == 19
