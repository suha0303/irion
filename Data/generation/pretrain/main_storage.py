"""Durable case capsules and exact owned-only retirement for the user-run campaign."""

import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path

from x500_storage import guard

from . import ai19
from .common import Budget, read, sha, verify
from .storage import pack_files, verify_bundle
from .streaming import retire_owned, tree_stamp

WRITE_POLICY = "single_xz0_write_no_readback_user_20261108"


def durable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def committed(path, *, deep=True):
    item = read(path)
    written = (
        item.get("status") == "RUN_CAPSULE_WRITTEN"
        and item.get("storage_policy") == WRITE_POLICY
        and item.get("artifact", {}).get("readback_verified") is False
    )
    if item.get("status") != "RUN_CAPSULE_VERIFIED" and not written:
        raise ValueError("Case not durably committed")
    if written and not deep:
        # Ordinary history reads do not rehash already-written payloads.
        expected_sizes = {
            item["artifact"]["path"]: item["artifact"]["bytes"],
            **item["export_sizes"],
        }
        for name, size in expected_sizes.items():
            source = ai19.no_links(Path(name))
            if not source.is_file() or source.stat().st_size != size:
                raise ValueError(
                    "Committed artifact missing or resized: " + name
                )
    else:
        verify(
            {
                item["artifact"]["path"]: item["artifact"]["sha256"],
                **item["export_files"],
            }
        )
    if deep and not written:
        verify_bundle(
            item["artifact"]["path"], item["artifact"]["members"], Budget()
        )
    return item


def commit(task, entry, target, baseline):
    """Keep every regular archived raw byte plus exports; retire only current intermediates."""
    task, target = Path(task), Path(target)
    raw, output = map(Path, (entry["run"], entry["output"]))
    prepared = read(entry["receipt"])
    audit = read(entry["audit_receipt"])
    if (
        raw.name in baseline
        or raw.parent != ai19.ROOT / "simulation/runs"
        or prepared["run"]["run_id"] != raw.name
        or audit["export_receipt_sha256"] != sha(output / "receipt.json")
        or not entry.get("online_available")
        or not entry.get("run_available")
        or entry["attempts"][-1].get("run_id") != raw.name
        or entry["attempts"][-1].get("quality_rejection")
    ):
        raise ValueError(
            "New owned successful raw and independent two-view audit required"
        )
    direct = prepared.get("source", {}).get("kind") == "closed_raw_directory"
    archive = None if direct else Path(prepared["archive"]["path"])
    consumer = Path(prepared["consumer_path"])
    cfg = read(task / "config.json")
    storage = guard(cfg["external_storage"], cfg.get("storage_policy"))
    if (
        target
        != storage / "capsules/main19" / task.name / entry["case"]["case_id"]
    ):
        raise ValueError(
            "Capsule destination must belong to this exact task/case"
        )
    ai19.no_links(target)
    ai19.no_links(output)
    if not output.is_relative_to(storage / "processed/main19" / task.name):
        raise ValueError("Retirement paths must belong to this task")
    if direct:
        if prepared["source"]["path"] != str(raw) or consumer != raw:
            raise ValueError("Closed direct replay source identity changed")
        targets = [raw]
    else:
        if not archive.is_relative_to(
            storage / "archives/main19" / task.name
        ) or not consumer.is_relative_to(
            storage / "consumer/main19" / task.name
        ):
            raise ValueError("Retirement paths must belong to this task")
        targets = [raw, archive, consumer]
    target.mkdir(parents=True, exist_ok=True)
    receipt_path = target / "commit.json"
    if receipt_path.exists():
        result = committed(receipt_path, deep=False)
        if result["run_id"] != raw.name or result["task"] != str(task):
            raise ValueError("Existing committed identity changed")
    else:
        if direct:
            if (
                sha(raw / "integrity.json")
                != prepared["source"]["integrity_sha256"]
            ):
                raise ValueError("Closed direct replay integrity changed")
            members = prepared["source"]["members"]
        else:
            verify({str(archive): prepared["archive"]["sha256"]})
            members = read(
                entry.get(
                    "archive_cache",
                    str(
                        task
                        / "pilot/execution"
                        / entry["case"]["case_id"]
                        / "archive-cache.json"
                    ),
                )
            )["archive"]["members"]
        # Freeze identities before reading any payload. A file added/changed after
        # packing must never become eligible for deletion just because the final
        # tree was sampled after compression.
        stamps = {str(p): tree_stamp(p) for p in targets}
        regular_members = {
            str(Path(name).relative_to(raw))
            for name, info in stamps[str(raw)].items()
            if stat.S_ISREG(info[2])
        }
        if regular_members != {
            name for name, info in members.items() if info["kind"] == "file"
        }:
            raise ValueError("Closed raw membership changed before packing")
        sources, expected = {}, {}
        for name, info in members.items():
            if info["kind"] != "file":
                continue
            source = ai19.no_links(raw / name)
            if source.stat().st_size != info["bytes"]:
                raise ValueError(
                    "Source archive member changed before durable capsule"
                )
            sources["raw/" + name] = source
            expected["raw/" + name] = {k: info[k] for k in ("bytes", "sha256")}
        sources.update(
            {
                "export/" + str(p.relative_to(output)): p
                for p in output.rglob("*")
                if p.is_file()
            }
        )
        sources["audit/input-quality.json"] = Path(entry["audit_receipt"])
        if entry.get("pair_receipt"):
            pair = Path(entry["pair_receipt"])
            sources["pair/projection.json"] = pair
            sources["pair/native.json"] = Path(
                read(pair)["native_evidence"]["path"]
            )
        for name, path in sources.items():
            if name not in expected:
                digest = entry["files"].get(str(path))
                expected[name] = dict(
                    bytes=path.stat().st_size, sha256=digest or sha(path)
                )
        source_identity = hashlib.sha256(
            json.dumps(stamps, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        artifact = publish(
            sources, target, expected=expected, source_identity=source_identity
        )
        result = dict(
            status="RUN_CAPSULE_WRITTEN",
            storage_policy=WRITE_POLICY,
            task=str(task),
            case=entry["case"],
            run_id=raw.name,
            artifact=artifact,
            archive_members=members,
            original_archive_sha256=None
            if direct
            else prepared["archive"]["sha256"],
            original_integrity_sha256=(
                prepared["source"]["integrity_sha256"] if direct else None
            ),
            export_files=entry["files"],
            export_sizes={
                name: Path(name).stat().st_size for name in entry["files"]
            },
            input_quality_sha256=sha(entry["audit_receipt"]),
            physical_attempts=entry["attempts"],
            retirement_stamps=stamps,
            canonical_frames_preserved=True,
            full_regular_raw_bytes_preserved=True,
        )
        durable(receipt_path, result)
    retire(task, result, receipt_path, targets, target)
    return dict(path=str(receipt_path), sha256=sha(receipt_path))


def publish(sources, target, *, expected=None, source_identity=None):
    final = target / "capsule.tar.xz"
    expected = (
        expected
        if expected is not None
        else {
            name: dict(bytes=Path(p).stat().st_size, sha256=sha(p))
            for name, p in sources.items()
        }
    )
    publication = target / "capsule-write.json"
    pending = target / "capsule-pending.json"

    def written_artifact(path):
        path = ai19.no_links(path)
        if not path.is_file():
            raise ValueError("Completed capsule is not a regular file")
        artifact = read(publication)
        if (
            artifact["path"] != str(final)
            or artifact["members"] != expected
            or artifact.get("readback_verified") is not False
            or artifact.get("source_identity") != source_identity
            or artifact["bytes"] != path.stat().st_size
            or artifact["sha256"] != sha(path)
        ):
            raise ValueError(
                "Existing capsule differs from its completed write receipt"
            )
        return artifact

    if final.exists():
        # A completed atomic write can precede the case commit after power loss.
        # Check that exact write receipt, without decompressing it again.
        return written_artifact(final)
    if pending.exists():
        previous = read(pending)
        candidate = ai19.no_links(Path(previous["path"]))
        if (
            candidate.parent != target
            or re.fullmatch(r"capsule-[0-9]+\.partial\.tar\.xz", candidate.name)
            is None
        ):
            raise ValueError("Pending capsule path is not owned by this case")
        if candidate.exists():
            if publication.exists() and read(publication).get(
                "candidate_path"
            ) == str(candidate):
                # fsync/write-receipt succeeded but atomic publication was interrupted.
                artifact = written_artifact(candidate)
                candidate.replace(final)
                fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                return artifact
            if not candidate.is_file():
                raise ValueError("Pending capsule is not a regular file")
            discarded = dict(
                previous,
                discarded_bytes=candidate.stat().st_size,
                reason="interrupted_before_completed_write_receipt",
            )
            durable(target / (candidate.name + ".discarded.json"), discarded)
            candidate.unlink()  # Only the exact incomplete path recorded by this case.
    candidate = target / ("capsule-" + str(time.time_ns()) + ".partial.tar.xz")
    durable(pending, dict(path=str(candidate), source_identity=source_identity))
    artifact = pack_files(
        sources,
        candidate,
        Budget(),
        preset=0,
        readback=False,
        expected=expected,
    )
    with candidate.open("rb") as stream:
        os.fsync(stream.fileno())
    artifact["path"] = str(final)
    artifact["candidate_path"] = str(candidate)
    artifact["source_identity"] = source_identity
    durable(publication, artifact)
    candidate.replace(final)
    fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return artifact


def remaining_owned(path, original):
    current = tree_stamp(path)
    if not set(current) <= set(original):
        raise ValueError("New path appeared inside interrupted retirement")
    for name, item in current.items():
        old = original[name]
        if stat.S_ISDIR(item[2]):
            if item[:3] != old[:3]:
                raise ValueError("Retirement directory was replaced")
        elif item != old:
            raise ValueError("Remaining retirement file changed")
    return current


def retire(task, result, receipt_path, paths, target):
    journal = target / "cleanup-current.json"
    stamp = result["retirement_stamps"]
    if set(stamp) != {str(p) for p in paths}:
        raise ValueError("Certified retirement membership changed")
    state = (
        read(journal)
        if journal.exists()
        else dict(
            task=str(task),
            run_id=result["run_id"],
            commit_sha256=sha(receipt_path),
            paths=[str(p) for p in paths],
            deleted=[],
            in_progress=None,
        )
    )
    if (
        state.get("task"),
        state.get("run_id"),
        state.get("commit_sha256"),
        state.get("paths"),
    ) != (
        str(task),
        result["run_id"],
        sha(receipt_path),
        [str(p) for p in paths],
    ):
        raise ValueError(
            "Interrupted retirement does not bind this exact commit"
        )
    for path in paths:
        key = str(path)
        if key in state["deleted"]:
            if path.exists():
                raise ValueError("Previously retired target reappeared")
            continue
        if not path.exists():
            if state.get("in_progress") != key:
                raise ValueError("Uncertified missing retirement target")
        else:
            current = (
                remaining_owned(path, stamp[key])
                if state.get("in_progress") == key
                else tree_stamp(path)
            )
            if state.get("in_progress") != key and current != stamp[key]:
                raise ValueError("Retirement target changed")
            state["in_progress"] = key
            durable(
                journal, state
            )  # Exact deletion intent is durable before its first unlink.
            retire_owned(
                [path],
                {key: current},
                target / "unlink-progress.json",
                proof=result,
                budget=Budget(),
            )
        state["deleted"].append(key)
        state["in_progress"] = None
        durable(journal, state)
    durable(
        target / "cleanup.json",
        dict(
            state,
            status="OWN_OUTPUTS_RETIRED_AFTER_WRITE"
            if result["status"] == "RUN_CAPSULE_WRITTEN"
            else "VERIFIED_OWN_OUTPUTS_RETIRED",
        ),
    )
