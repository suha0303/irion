"""Explicit, hash-pinned CPU auditor; all generation files and raw remain unchanged."""
import hashlib
import importlib.util
import json
from pathlib import Path

from . import native_joint

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
REVISION = "native-consumption-identical-clock-v2"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest():
    from capstone_sim.runtime_manifest import verify_runtime

    path = PACKAGE / "manifest.json"
    value = json.loads(path.read_text())
    if value.get("revision") != REVISION:
        raise ValueError("Unsupported CPU audit revision")
    for name, expected in value["audit_files"].items():
        if sha(PACKAGE / name) != expected:
            raise ValueError("CPU audit bundle changed: " + name)
    for name, expected in value["generation_dependencies"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Generation dependency changed: " + name)
    verify_runtime(ROOT)
    return value, sha(path)


def evidence(run):
    run = Path(run).resolve()
    value, digest = manifest()
    state = json.loads((run / "run.json").read_text())
    scenario = json.loads((run / "evaluation/scenario.json").read_text())
    integrity = json.loads((run / "integrity.json").read_text())
    if scenario["producer_lock_sha256"] != value["generation_runtime_sha256"]:
        raise ValueError("Raw generation seal differs from reviewed audit scope")
    if scenario["joint_phase_policy"]["policy_sha256"] != value["generation_policy_sha256"]:
        raise ValueError("Raw generation policy differs from reviewed audit scope")
    names = ["run.json", "evaluation/scenario.json", "reference/source_lock.json",
             "evaluation/joint-origin.json", "evaluation/joint-landing.json",
             "evaluation/joint-consumption.csv", "evaluation/joint-reference.csv",
             "evaluation/joint-physics-origin.json", "evaluation/joint-phase-policy.json"]
    hashes = {}
    for name in names:
        path = run / name
        actual = sha(path)
        if integrity[name]["sha256"] != actual or integrity[name]["size"] != path.stat().st_size:
            raise ValueError("Original raw integrity mismatch: " + name)
        hashes[name] = actual
    return dict(revision=REVISION, manifest_sha256=digest,
        manifest_path=str(PACKAGE / "manifest.json"), run_id=run.name, raw_path=str(run),
        generation_source_sha256=value["generation_source_sha256"],
        generation_runtime_sha256=value["generation_runtime_sha256"],
        generation_policy_sha256=value["generation_policy_sha256"],
        raw_integrity_sha256=sha(run / "integrity.json"), raw_files_sha256=hashes,
        original_termination_reason=state.get("termination_reason"),
        original_primary_failure=state.get("primary_failure"),
        raw_joint_check_present=(run / "evaluation/joint-check.json").exists(),
        raw_binding_scope="Listed files plus original closing-integrity digest; full assessment verifies every closing-integrity entry",
        original_raw_unchanged=True)


def verify(run):
    proof = evidence(run)
    result = native_joint.verify(run)
    return dict(result, schema="drone.native-consumption-audit/2", audit=proof)


def assess(run, instance, duration, profile_id):
    proof = evidence(run)
    # Relative capstone_sim imports retain the original sealed dependency paths.
    # This is a separate assessor module; no installed module/function is replaced.
    import capstone_sim  # noqa: F401

    spec = importlib.util.spec_from_file_location("capstone_sim.native_consumption_v2_assessment", PACKAGE / "run_assessment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.assess(run, instance, duration, profile_id)
    result["cpu_audit"] = proof
    try:
        result["native_consumption_audit"] = native_joint.verify(run)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["native_consumption_audit"] = dict(status="failed", error=str(error))
    result["audit_full_gate_passed"] = (
        all(result[key] for key in ("contract_passed", "instance_isolation_passed", "missing_data_passed", "physical_gate_passed"))
        and result["native_consumption_audit"]["status"] == "passed")
    if evidence(run) != proof:
        raise ValueError("Raw or audit evidence changed during full assessment")
    return result


def verify_pair(normal_run, anomaly_run, output):
    proof = dict(normal=evidence(normal_run), anomaly=evidence(anomaly_run))
    return native_joint.verify_pair(normal_run, anomaly_run, output, cpu_audit=proof)
