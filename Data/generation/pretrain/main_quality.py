"""Data control for AI-verified production rejection; original physical state is immutable."""

from . import ai19
from .common import read, verify

ERROR = "Joint native consumption: Incomplete actual reference consumption coverage"


class QualityRejected(ValueError):
    def __init__(self, ref):
        self.ref = ref
        super().__init__(ERROR)


def publish(raw, case, binding, assessment_path, output, *, policy_bundle):
    ai19.ai_modules()
    from learning.collection19 import publish_quality_rejection

    from .common import sha

    result = publish_quality_rejection(
        raw,
        case_id=case["case_id"],
        producer=binding,
        policy_bundle=policy_bundle,
        assessment_ref=dict(path=str(assessment_path), sha256=sha(assessment_path)),
        output=output,
    )
    import os
    from pathlib import Path

    for path in (assessment_path, result["path"]):
        with Path(path).open("rb") as stream:
            os.fsync(stream.fileno())
    fd = os.open(Path(result["path"]).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return result


def allowed(cfg, case, prior):
    recovery = recovery_entry(cfg, case)
    if (
        recovery is not None
        and recovery["mode"] == "reflight"
        and prior
        and prior.get("run_id") == recovery["original_last_run_id"]
        and prior.get("number") == recovery["initial_attempts"]
    ):
        return True
    if not prior or not prior.get("quality_rejection"):
        return False
    ref = prior["quality_rejection"]
    verify({ref["path"]: ref["sha256"]})
    value = read(ref["path"])
    if (
        prior.get("state") != "COMPLETED"
        or value.get("case_id") != case["case_id"]
        or value.get("run_id") != prior.get("run_id")
    ):
        raise ValueError("Completed retry requires this exact independently rejected case/UUID")
    ai19.ai_modules()
    from learning.collection19 import check_quality_rejection

    check_quality_rejection(
        ref,
        dict(
            run_id=prior["run_id"],
            status="completed",
            seed=case["seed"],
            parent_attempt_id=prior["parent_attempt_id"],
        ),
        cfg["producer"],
        policy_bundle=cfg["policy_bundle"],
        case_id=case["case_id"],
    )
    return True


def recovery_entry(cfg, case):
    ref = cfg.get("recovery_authorization")
    if ref is None:
        return None
    ai19.ai_modules()
    from learning.collection19.recovery import entry

    return entry(ref, case)


def attempts_max(cfg, case):
    recovery = recovery_entry(cfg, case)
    if recovery is not None and recovery["mode"] == "reflight":
        return recovery["max_physical_attempts"]
    return case["attempts_max"]
