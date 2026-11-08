"""Production-role exports using the existing19 transformations; no pilot relabelling."""

import json
import stat
import sys
from pathlib import Path

import numpy as np

from . import ai19, pilot19
from .common import ROOT, collector, read, sha, write


def closed_source(raw):
    """Reuse the producer's closing inventory without making an intermediate archive."""
    integrity = read(raw / "integrity.json")
    members = {}
    for path, mode in collector().files(raw):
        if not stat.S_ISREG(mode):
            continue
        name = str(path.relative_to(raw))
        info = path.stat()
        original = integrity.get(name)
        if original is not None and info.st_size != original["size"]:
            raise ValueError("Closed raw size changed: " + name)
        members[name] = dict(
            kind="file",
            bytes=info.st_size,
            sha256=original["sha256"] if original is not None else sha(path),
        )
    if not ai19.CONSUMER <= set(members):
        raise ValueError("Closed raw consumer inventory incomplete")
    return dict(
        kind="closed_raw_directory",
        path=str(raw),
        integrity_sha256=sha(raw / "integrity.json"),
        members=members,
        intermediate_archive=False,
        replay_source="original_read_only",
    )


def prepare(
    selection,
    output,
    expected_path,
    archive_cache,
    protection_path,
    *,
    policy_bundle,
    case,
    producer_binding=None,
):
    mapping, _, _, _ = ai19.ai_modules()
    from learning.admission19 import (
        check_producer,
        check_protection,
        source_evidence,
    )

    output, archive_cache = Path(output), Path(archive_cache)
    if output.exists():
        raise ValueError(
            "Never overwrite a previous production processing attempt"
        )
    if selection["cohort"] not in {
        "normal_train",
        "normal_validation",
        "normal_calibration",
        "development",
        "final",
    }:
        raise ValueError("Actual approved production role/cohort required")
    binding = (
        producer_binding
        if producer_binding is not None
        else ai19.producer_binding()
    )
    check_producer(binding, read(expected_path))
    assessment_path = archive_cache.parent / (
        output.parent.name + "-" + output.name + "-physical-assessment.json"
    )
    try:
        row, evidence = ai19.admit(
            selection,
            binding,
            policy_bundle=policy_bundle,
            assessment_path=assessment_path,
        )
    except ValueError:
        if assessment_path.exists():
            from .main_quality import ERROR, QualityRejected, publish

            if read(assessment_path).get("errors") == [ERROR]:
                ref = publish(
                    selection["path"],
                    case,
                    binding,
                    assessment_path,
                    archive_cache.parent
                    / (
                        output.parent.name
                        + "-"
                        + output.name
                        + "-quality-rejection.json"
                    ),
                    policy_bundle=policy_bundle,
                )
                raise QualityRejected(ref) from None
        raise
    row.update(
        purpose=selection["purpose"],
        lineage=ai19.lineage(row),
        mission_interval=evidence["mission_interval"],
        source_evidence=pilot19.source_evidence(
            Path(row["path"]),
            binding,
            read(Path(row["path"]) / "evaluation/scenario.json"),
        ),
    )
    source_evidence(row, binding)
    protection = read(protection_path)
    check_protection([row], protection, usage="input_check")
    output.mkdir(parents=True)
    write(output / "assessment.json", evidence["assessment"])
    write(output / "source-admission.json", evidence)
    write(output / "protection.json", protection)
    raw = Path(row["path"])
    source = closed_source(raw)
    engine, alias = ai19.replay(
        raw,
        read(ROOT / "ai/learning/history19/preprocessing.json"),
        expected_profile_revision=binding["vehicle_profile_revision"],
    )
    for name in ("frames", "windows", "exclusions"):
        with (output / (name + ".jsonl")).open("x", encoding="utf-8") as stream:
            for point in getattr(engine, name):
                stream.write(json.dumps(point, allow_nan=False) + "\n")
    arrays, exclusions = ai19.online_arrays(engine, row["mission_interval"])
    physical, coverage = ai19.run_array(engine, row["mission_interval"])
    online = dict(
        available=arrays is not None,
        valid_windows=0,
        target_valid_seconds=0,
        exclusions=exclusions,
    )
    if arrays is not None:
        np.savez(output / "online.npz", **arrays)
        write(
            output / "online.json",
            dict(
                schema="drone-ai.history19.run/1",
                units="nameplate_transformed_common19_cellV_Crate",
                interval_policy=ai19.MISSION_POLICY,
                feature_order=mapping.FEATURES,
                target_order=mapping.TARGETS,
                step_us=100000,
                history=100,
                future=10,
                emit_every_us=1000000,
                producer=binding,
                run=row,
                file=ai19.reference(output / "online.npz", output),
            ),
        )
        online.update(
            valid_windows=len(arrays["X"]),
            target_valid_seconds=len(arrays["X"]),
            manifest=str(output / "online.json"),
        )
    whole = dict(coverage, manifest=None, post_initial100_seconds=0)
    if physical is not None:
        np.savez(output / "run.npz", **physical)
        item = dict(
            row,
            health=row["health_eligibility"],
            **ai19.reference(output / "run.npz", output),
        )
        write(
            output / "run.json",
            dict(
                schema="drone-ai.run-health19.dataset/1",
                units="physical_common19_pack_V_A",
                step_us=100000,
                feature_order=mapping.FEATURES,
                vehicle_profile="x500_reference",
                battery_policy=ai19.POLICY,
                interval_policy=ai19.MISSION_POLICY,
                producer=binding,
                runs=[item],
            ),
        )
        whole.update(
            manifest=str(output / "run.json"),
            post_initial100_seconds=max(0, len(physical["time_us"]) - 100) / 10,
        )
    if "torch" in sys.modules:
        raise ValueError("Data preprocessing may not import models")
    result = dict(
        schema="drone-data.main19.run/1",
        status="EXPORTED_WITHOUT_FIT",
        run=row,
        producer=binding,
        online=online,
        run_health=whole,
        source=source,
        consumer_path=str(raw),
        validation_alias=alias,
        assessment_path=str(output / "assessment.json"),
        no_fit=True,
        training_started=False,
        inference_started=False,
    )
    result["files"] = {
        str(p.relative_to(output)): sha(p)
        for p in output.rglob("*")
        if p.is_file()
    }
    write(output / "receipt.json", result)
    write(
        output / "seal.json", dict(receipt_sha256=sha(output / "receipt.json"))
    )
    return result
