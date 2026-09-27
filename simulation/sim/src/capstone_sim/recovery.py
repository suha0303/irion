"""Recover complete producer rows into a separate artifact, preserving the source."""

import json
from pathlib import Path

from .recording import GENERIC_METADATA, GENERIC_RAW, dump, sha, verify_generic_metadata


def recover(source, out):
    source = Path(source).resolve()
    out = Path(out).resolve()
    if out == source or source in out.parents:
        raise ValueError("recovery output must be separate from source")
    metadata_path = source / "observations/metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    generic = metadata is not None and metadata.get("schema") == GENERIC_METADATA
    if generic:
        assert metadata is not None
        verify_generic_metadata(source, metadata)
    out.mkdir(parents=True, exist_ok=False)
    raw = (source / "observations/streams.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    count = 0
    last = 0
    discarded = 0
    with (out / "streams.jsonl").open("wb") as target:
        for i, line in enumerate(lines):
            if not line.endswith(b"\n"):
                discarded += len(line)
                break
            e = json.loads(line)
            if e["schema"] != (GENERIC_RAW if generic else "drone.raw.event/2.0"):
                raise ValueError("unsupported event schema")
            if generic:
                assert metadata is not None
                if (
                    e.get("run_id") != metadata["run_id"]
                    or e.get("clock_id") != metadata["clock"]["id"]
                ):
                    raise ValueError("Recovery observation identity differs from bound metadata")
                if (
                    e["topic"] == "actuator_outputs"
                    and len(e["fields"]["output"]) != metadata["vehicle"]["rotor_count"]
                ):
                    raise ValueError("Recovery rotor count differs from the bound profile")
            target.write(line)
            count += 1
            last = e["available_time_us"]
    report = dict(
        source=str(source),
        source_sha256=sha(source / "observations/streams.jsonl"),
        artifact_kind="observation_fragment",
        full_run=False,
        closed_watermark_created=False,
        training_eligible=False,
        C9="not_verified_recovery_fragment",
        recovered_events=count,
        discarded_incomplete_bytes=discarded,
        derived_from_same_run=True,
        new_flight=False,
        through_available_time_us=last,
    )
    dump(out / "recovery.json", report)
    if generic:
        assert metadata is not None
        (out / "metadata.json").write_bytes(metadata_path.read_bytes())
        reference = out / "reference/vehicle-profile.json"
        reference.parent.mkdir()
        reference.write_bytes((source / "reference/vehicle-profile.json").read_bytes())
        verify_generic_metadata(out, json.loads((out / "metadata.json").read_text()))
        report["ai_compatible"] = False
        dump(out / "recovery.json", report)
    return report
