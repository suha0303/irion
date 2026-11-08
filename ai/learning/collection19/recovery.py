"""Explicit finite recovery authorization; original physical outcomes stay
immutable.
"""

from learning.history19.pilot import ROOT, json_file

from .contract import bound

AUTH = (
    ROOT / "ai/reports/targeted-recovery-20261108/attempt-01/authorization.json"
)
AUTH_SHA = "5343e6817bac49980a4979857d4aa8a1bd3b6a83f91aec4e63fbe019ab0d2273"
NEXT_AUTH = (
    ROOT / "ai/reports/targeted-recovery-20261108/attempt-02/authorization.json"
)
NEXT_SHA = "5c4f6587205f3f177e04d4f5855d1a13fd61e2ee66c5861fdd654d086e93a9e8"
THIRD_AUTH = (
    ROOT / "ai/reports/targeted-recovery-20261108/attempt-03/authorization.json"
)
THIRD_SHA = "5ec2cae7b1a781ca2742143ee5c8689d28a2ad018e3963accb0884199cfdebe4"
FOURTH_AUTH = (
    ROOT / "ai/reports/targeted-recovery-20261108/attempt-04/authorization.json"
)
FOURTH_SHA = "a3716652c21834aeeccd7b4754979d5995fe6a8ecf1a2bf405bffea2f25b3209"


def entry(ref, case):
    if ref not in (
        dict(path=str(AUTH), sha256=AUTH_SHA),
        dict(path=str(NEXT_AUTH), sha256=NEXT_SHA),
        dict(path=str(THIRD_AUTH), sha256=THIRD_SHA),
        dict(path=str(FOURTH_AUTH), sha256=FOURTH_SHA),
    ):
        raise ValueError("Unknown targeted recovery authority")
    auth = json_file(bound(ref))
    item = auth["cases"].get(case["case_id"])
    if item is None:
        return None
    before = json_file(bound(item["before"]))
    attempts = before["attempts"]
    extra = auth.get("additional_attempts_per_case")
    maximum = (
        len(attempts) + (extra if item["mode"] == "reflight" else 0)
        if extra in (1, 3)
        else max(3, len(attempts) + 1)
    )
    if (
        auth.get("schema") != "drone-data.targeted-recovery-authorization/1"
        or auth.get("training_authorized") is not False
        or before["case"] != case
        or before.get("status") not in {"UNAVAILABLE", "FAILED_PRESERVED"}
        or item["initial_attempts"] != len(attempts)
        or not 1 <= len(attempts) <= (5 if extra in (1, 3) else 3)
        or item["original_last_run_id"] != attempts[-1]["run_id"]
        or item["max_physical_attempts"] != maximum
    ):
        raise ValueError("Recovery differs from original unavailable case")
    if item["mode"] == "reflight":
        proof = json_file(bound(item["evidence"]))
        run_id = proof.get("run_id")
        if run_id is None and isinstance(proof.get("run"), dict):
            run_id = proof["run"].get("run_id")
        if item["reason"] == "physical_attempts_exhausted":
            from pathlib import Path

            run_id = Path(proof["run"]).name
            valid = proof.get("status") == "failed" and not proof.get(
                "main_shared_failure"
            )
        elif item["reason"] == "run_view_missing_features":
            valid = (
                proof.get("status") == "EXPORTED_WITHOUT_FIT"
                and proof["run_health"].get("status") == "unavailable"
                and proof["run_health"].get("reasons")
                == ["missing_or_nonvalid_feature"]
            )
        else:
            valid = (
                proof.get("schema") == "drone.run-assessment/1"
                and proof.get("instance_isolation_passed") is True
                and any(
                    proof.get(k) is False
                    for k in (
                        "contract_passed",
                        "missing_data_passed",
                        "physical_gate_passed",
                    )
                )
            )
        if not valid or run_id != item["original_last_run_id"]:
            raise ValueError(
                "Recovery lacks matching original rejection evidence"
            )
    elif item["mode"] != "reuse_pair":
        raise ValueError("Unknown recovery mode")
    return dict(item, before_value=before)


def ledger_permission(ref, case, attempts):
    item = entry(ref, case)
    if item is None:
        raise ValueError("Unselected case cannot carry recovery permission")
    originals = item["before_value"]["attempts"]
    if len(attempts) < len(originals):
        raise ValueError("Recovery erased previous attempts")
    for old, current in zip(originals, attempts):
        status = (
            "completed" if old["state"].startswith("COMPLETED") else "failed"
        )
        if (
            current.get("run_id") != old["run_id"]
            or current.get("parent_attempt_id") != old["parent_attempt_id"]
            or current.get("index") != old["number"]
            or current.get("status") != status
        ):
            raise ValueError("Recovery changed the original physical history")
    limit = (
        item["max_physical_attempts"]
        if item["mode"] == "reflight"
        else len(originals)
    )
    rejected = (
        {item["original_last_run_id"]} if item["mode"] == "reflight" else set()
    )
    if ref in (
        dict(path=str(NEXT_AUTH), sha256=NEXT_SHA),
        dict(path=str(THIRD_AUTH), sha256=THIRD_SHA),
        dict(path=str(FOURTH_AUTH), sha256=FOURTH_SHA),
    ):
        previous = json_file(bound(ref))["previous_authorization"]
        expected = (
            dict(path=str(THIRD_AUTH), sha256=THIRD_SHA)
            if ref["sha256"] == FOURTH_SHA
            else dict(path=str(NEXT_AUTH), sha256=NEXT_SHA)
            if ref["sha256"] == THIRD_SHA
            else dict(path=str(AUTH), sha256=AUTH_SHA)
        )
        if previous != expected:
            raise ValueError(
                "Recovery history must preserve the earlier authority"
            )
        prior = entry(previous, case)
        if prior is not None:
            _, earlier_rejected = ledger_permission(previous, case, attempts)
            rejected.update(earlier_rejected)
    return limit, rejected
