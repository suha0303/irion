"""Verify compressed source and materialize only six consumer files, never truth."""

import hashlib
import tarfile
from pathlib import PurePosixPath

from .control import RAW_FILES, STORAGE, external, regular, sha


def extract_consumer(row, budget):
    reference = row["archive"]
    archive = regular(reference["path"])
    if not archive.is_relative_to(STORAGE / "archives"):
        raise ValueError("new R3 external archive required")
    if archive.stat().st_size != reference["bytes"] or sha(archive) != reference["sha256"]:
        raise ValueError("compressed source hash/size mismatch")
    if set(row["files"]) != RAW_FILES:
        raise ValueError("exact consumer inventory required")
    metadata = reference["members"]
    expected_size = 0
    for name in RAW_FILES:
        member = metadata[name]
        if (
            member["kind"] != "file"
            or member["sha256"] != row["files"][name]
            or type(member["bytes"]) is not int
            or not 0 <= member["bytes"] <= 8_000_000_000
        ):
            raise ValueError("consumer member hash/type/size mismatch")
        expected_size += member["bytes"]
    if expected_size > 10_000_000_000:
        raise ValueError("per-run consumer expansion exceeds fixed10GB cap")
    target = external(row["path"], "consumer")
    target.mkdir(exist_ok=False)
    seen, selected = set(), set()
    # Iterate members rather than extract/extractall: links and paths are never followed.
    with tarfile.open(archive, mode="r|gz") as stream:
        for member in stream:
            budget.check()
            name = member.name
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or "\\" in name or name in seen:
                raise ValueError("unsafe/duplicate archive member")
            seen.add(name)
            if name not in RAW_FILES:
                # Even a safe link to a selected member is never dereferenced or materialized.
                continue
            if not member.isfile() or member.issym() or member.islnk() or member.sparse:
                raise ValueError("consumer member must be regular non-sparse file")
            if member.size != metadata[name]["bytes"]:
                raise ValueError("consumer member size changed")
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            digest, total = hashlib.sha256(), 0
            with stream.extractfile(member) as source, output.open("xb") as sink:
                while block := source.read(1024 * 1024):
                    budget.check()
                    total += len(block)
                    if total > member.size:
                        raise ValueError("member expansion limit exceeded")
                    digest.update(block)
                    sink.write(block)
            if total != member.size or digest.hexdigest() != row["files"][name]:
                raise ValueError("extracted consumer hash/size mismatch")
            selected.add(name)
    if selected != RAW_FILES or sha(archive) != reference["sha256"]:
        raise ValueError("missing consumer files or archive changed")
    # Partial output is deliberately retained on all failures; automatic retry forbidden.
    return {
        "archive_sha256": reference["sha256"],
        "files": row["files"],
        "consumer_bytes": expected_size,
        "truth_extracted": False,
    }
