"""Synthetic external audit integrity/denominator guards; no actual trace replay."""

import json
from pathlib import Path

import pytest
from test_pilot_publisher import fixture

from learning.history19 import joint_native
from learning.history19.native_audit import RAW_MEMBERS, REVISION, verified_check
from learning.history19.pilot import json_file, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return dict(path=str(path.absolute()), sha256=sha(path))


def external_fixture(folder):
    args = fixture(folder)
    raw = folder / "synthetic-0-normal"
    producer = json_file(args["producer_binding"])
    checked = json_file(raw / joint_native.CHECK)
    checked.update(
        same_source_time_consumptions=2,
        unique_source_time_samples=checked["consumed_samples"] - 2,
        audit_revision=REVISION,
        backward_source_time_count=0,
        conflicting_same_source_time_count=0,
        first_consumption_source_us=checked["actual_start_us"],
        last_consumption_source_us=checked["actual_landing_us"] - 4000,
    )
    (raw / joint_native.CHECK).unlink()
    for member in RAW_MEMBERS:
        p = raw / member
        if not p.exists():
            if member == "evaluation/joint-phase-policy.json":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(args["native_policy"].read_bytes())
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("{}\n")
    integrity = {n: dict(sha256=sha(raw / n), size=(raw / n).stat().st_size) for n in RAW_MEMBERS}
    write(raw / "integrity.json", integrity)
    package = folder / "auditor"
    package.mkdir()
    (package / "api.py").write_text("# synthetic auditor identity\n")
    gen = dict(
        source_sha256=producer["source_manifest_sha256"],
        runtime_sha256=producer["runtime_sha256"],
        policy_sha256=sha(args["native_policy"]),
    )
    manifest = dict(
        schema="drone.native-consumption-audit-manifest/1",
        revision=REVISION,
        audit_files={"api.py": sha(package / "api.py")},
        generation_dependencies={
            "config/producer-sources.json": gen["source_sha256"],
            "artifacts/runtime-manifest.json": gen["runtime_sha256"],
        },
        **{"generation_" + k: v for k, v in gen.items()},
    )
    mref = write(package / "manifest.json", manifest)
    bref = write(
        folder / "binding.json",
        dict(
            schema="drone-ai.verified-auditor-binding/1",
            revision=REVISION,
            manifest=mref,
            audit_files=manifest["audit_files"],
            generation_dependencies=manifest["generation_dependencies"],
            generation=gen,
            generation_dependencies_verified_at_freeze=True,
        ),
    )
    state = json_file(raw / "run.json")
    proof = dict(
        revision=REVISION,
        manifest_sha256=mref["sha256"],
        manifest_path=mref["path"],
        run_id=state["run_id"],
        raw_path=str(raw.absolute()),
        raw_integrity_sha256=sha(raw / "integrity.json"),
        raw_files_sha256={n: integrity[n]["sha256"] for n in RAW_MEMBERS},
        original_raw_unchanged=True,
        raw_joint_check_present=False,
        original_termination_reason=None,
        original_primary_failure=None,
        **{"generation_" + k: v for k, v in gen.items()},
    )
    full = dict(
        schema="drone.run-assessment/1",
        run_id=state["run_id"],
        cpu_audit=proof,
        native_consumption_audit=checked,
        audit_full_gate_passed=True,
        contract_passed=True,
        instance_isolation_passed=True,
        missing_data_passed=True,
        physical_gate_passed=True,
    )
    fref = write(folder / "full.json", full)
    rref = write(
        folder / "external.json",
        dict(
            checked,
            schema="drone.native-consumption-audit/2",
            audit=proof,
            full_gate_passed=True,
            full_gate_receipt=fref,
        ),
    )
    return dict(
        run_path=raw / "run.json",
        external_audit=dict(receipt=rref, auditor_binding=bref),
        generation=producer,
        policy_sha256=gen["policy_sha256"],
        duration_us=480000000,
        fixture_only=True,
    )


def test_explicit_external_check_keeps_original_raw_check_absent(tmp_path):
    args = external_fixture(tmp_path / "unit")
    raw = args["run_path"].parent
    before = sha(raw / "integrity.json")
    checked, evidence = verified_check(**args)
    assert (
        checked["same_source_time_consumptions"] == 2
        and evidence["generation"]["runtime_sha256"] == args["generation"]["runtime_sha256"]
    )
    assert not (raw / joint_native.CHECK).exists() and sha(raw / "integrity.json") == before


@pytest.mark.parametrize(
    "damage",
    [
        "native_only",
        "full_gate",
        "false_contract",
        "generation",
        "audit_code",
        "raw_csv",
        "wrong_duration",
    ],
)
def test_full_gate_and_original_hashes_are_mandatory(tmp_path, damage):
    args = external_fixture(tmp_path / "unit")
    ref = args["external_audit"]["receipt"]
    path = Path(ref["path"])
    receipt = json_file(path)
    if damage == "native_only":
        receipt.pop("full_gate_receipt")
        write(path, receipt)
        ref["sha256"] = sha(path)
    elif damage == "full_gate":
        receipt["full_gate_passed"] = False
        write(path, receipt)
        ref["sha256"] = sha(path)
    elif damage == "false_contract":
        fp = Path(receipt["full_gate_receipt"]["path"])
        v = json_file(fp)
        v["contract_passed"] = False
        write(fp, v)
        receipt["full_gate_receipt"]["sha256"] = sha(fp)
        write(path, receipt)
        ref["sha256"] = sha(path)
    elif damage == "generation":
        args["generation"]["runtime_sha256"] = "0" * 64
    elif damage == "audit_code":
        (tmp_path / "unit/auditor/api.py").write_text("# changed auditor\n")
    elif damage == "raw_csv":
        (args["run_path"].parent / "evaluation/joint-consumption.csv").write_text("changed\n")
    else:
        args["duration_us"] = 180000000
    with pytest.raises((ValueError, KeyError)):
        verified_check(**args)


@pytest.mark.parametrize("changed", [False, True])
def test_hash_identical_historical_auditor_location_is_explicitly_verified(tmp_path, changed):
    args = external_fixture(tmp_path / "unit")
    old = tmp_path / "unit/auditor"
    archive = tmp_path / "archive"
    archive.mkdir()
    for name in ("api.py", "manifest.json"):
        (archive / name).write_bytes((old / name).read_bytes())
    ref = args["external_audit"]["receipt"]
    rp = Path(ref["path"])
    receipt = json_file(rp)
    receipt["audit"]["manifest_path"] = str((archive / "manifest.json").absolute())
    fp = Path(receipt["full_gate_receipt"]["path"])
    full = json_file(fp)
    full["cpu_audit"] = receipt["audit"]
    write(fp, full)
    receipt["full_gate_receipt"]["sha256"] = sha(fp)
    write(rp, receipt)
    ref["sha256"] = sha(rp)
    if changed:
        (archive / "api.py").write_text("# changed copied auditor\n")
        with pytest.raises(ValueError, match="package differs"):
            verified_check(**args)
    else:
        assert verified_check(**args)[0]["status"] == "passed"
