"""Lossless portable prepared datasets. Never deletes raw or existing outputs."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

from .common import Budget, external, read, sha, write


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or not p.parts:
        raise ValueError("Unsafe portable member: " + name)


def verify_bundle(path, expected, budget, destination=None, progress=None):
    seen = set()
    with tarfile.open(path, "r|xz") as tar:
        for member in tar:
            budget.check()
            safe_name(member.name)
            if (
                member.name in seen
                or member.name not in expected
                or not member.isfile()
                or member.issparse()
            ):
                raise ValueError("Unexpected/duplicate portable member")
            seen.add(member.name)
            h = hashlib.sha256()
            target = None
            if destination is not None:
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream:
                output = target.open("xb") if target else None
                try:
                    size = 0
                    while block := stream.read(1048576):
                        budget.check()
                        h.update(block)
                        size += len(block)
                        if output:
                            output.write(block)
                finally:
                    if output:
                        output.close()
            if (
                h.hexdigest() != expected[member.name]["sha256"]
                or size != expected[member.name]["bytes"]
            ):
                raise ValueError("Portable member hash/size mismatch: " + member.name)
            if progress:
                progress(done=len(seen), total=len(expected))
    if seen != set(expected):
        raise ValueError("Missing portable members")


def pack_files(sources, path, budget):
    """Losslessly seal explicitly selected files and verify reconstructed bytes."""
    inventory = {}
    with tarfile.open(path, "x:xz", preset=3) as tar:
        for name, source in sorted(sources.items()):
            budget.check()
            safe_name(name)
            source = Path(source)
            if source.is_symlink() or not source.is_file():
                raise ValueError("Regular capsule source required")
            inventory[name] = dict(bytes=source.stat().st_size, sha256=sha(source))
            tar.add(source, arcname=name, recursive=False)
    verify_bundle(path, inventory, budget)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, members=inventory)


def event_body(row, budget):
    """Read only small development labels; permit verified retired-raw capsules."""
    if row["role"] not in {"normal_dev_eval", "anomaly_dev"}:
        raise ValueError("Development event source required")
    expected = row["archive"]["members"]["evaluation/events.jsonl"]
    if expected["kind"] != "file" or not 0 <= expected["bytes"] <= 10_000_000:
        raise ValueError("Bounded regular development truth required")
    compact = row.get("compact_source")
    if compact:
        p = Path(compact["receipt"])
        if sha(p) != compact["sha256"]:
            raise ValueError("Compact source receipt changed")
        receipt = read(p)
        if receipt["run_id"] != row["run_id"] or receipt["status"] != "RUN_CAPSULE_VERIFIED":
            raise ValueError("Wrong compact source identity")
        event = Path(receipt["event_path"])
        if event.is_symlink() or not event.is_file() or event.stat().st_size != expected["bytes"]:
            raise ValueError("Bounded regular compact truth required")
        with event.open("rb") as stream:
            body = stream.read(10_000_001)
    else:
        ref = row["archive"]
        raw = Path(ref["path"])
        if raw.stat().st_size != ref["bytes"] or sha(raw) != ref["sha256"]:
            raise ValueError("Development source archive changed")
        body = None
        with tarfile.open(raw, "r|gz") as source:
            for member in source:
                budget.check()
                if member.name != "evaluation/events.jsonl":
                    continue
                if (
                    body is not None
                    or not member.isfile()
                    or member.issparse()
                    or member.size != expected["bytes"]
                ):
                    raise ValueError("Invalid development event member")
                with source.extractfile(member) as stream:
                    body = stream.read(10_000_001)
    if (
        body is None
        or len(body) != expected["bytes"]
        or hashlib.sha256(body).hexdigest() != expected["sha256"]
    ):
        raise ValueError("Development truth digest mismatch")
    return body


def pack(prepared, output, progress=lambda **kw: None, *, budget=None):
    prepared, output = Path(prepared).resolve(), external(output)
    budget = budget or Budget()
    handoff = read(prepared / "handoff.json")
    quality = read(prepared / "quality.json")
    if (
        not quality.get("passed")
        or not quality.get("train_only_scaler_verified")
        or not handoff.get("training_eligible")
    ):
        raise ValueError("Complete independently verified dataset required before packing")
    dataset = Path(handoff["dataset"]).resolve()
    seal = read(prepared / "dataset-seal.json")
    sources = {"dataset/" + name: (dataset / name, digest) for name, digest in seal.items()}
    for name, key in [
        ("dataset-seal.json", "dataset_seal_sha256"),
        ("quality.json", "quality_sha256"),
        ("frozen-manifest.json", "frozen_manifest_sha256"),
    ]:
        sources["evidence/" + name] = (prepared / name, handoff[key])
    sources["evidence/handoff.json"] = (prepared / "handoff.json", sha(prepared / "handoff.json"))
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    archive = output / "dataset.tar.xz"
    with tarfile.open(archive, "w:xz", preset=3) as tar:
        for i, (name, (source, digest)) in enumerate(sorted(sources.items())):
            budget.check()
            safe_name(name)
            if source.is_symlink() or not source.is_file() or sha(source) != digest:
                raise ValueError("Prepared source changed: " + str(source))
            inventory[name] = dict(bytes=source.stat().st_size, sha256=digest)
            tar.add(source, arcname=name, recursive=False)
            progress(phase="pack_dataset", done=i + 1, total=len(sources))
    verify_bundle(archive, inventory, budget)
    # Only compact event labels enter this separate evaluator-only archive.
    truth_inventory = {}
    rows = read(dataset / "manifest.json")["runs"]
    dev = [r for r in rows if r["role"] in {"normal_dev_eval", "anomaly_dev"}]
    truth_archive = output / "evaluation-truth.tar.xz"
    with tarfile.open(truth_archive, "w:xz", preset=3) as tar:
        for i, row in enumerate(dev):
            budget.check()
            body = event_body(row, budget)
            expected = row["archive"]["members"]["evaluation/events.jsonl"]
            name = row["run_id"] + ".jsonl"
            safe_name(name)
            if name in truth_inventory:
                raise ValueError("Duplicate development run")
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
            truth_inventory[name] = dict(bytes=len(body), sha256=expected["sha256"])
            progress(phase="pack_evaluator_truth", done=i + 1, total=len(dev))
    verify_bundle(truth_archive, truth_inventory, budget)
    receipt = dict(
        status="PORTABLE_DATASET_VERIFIED",
        format="lossless-tar-xz-v1",
        source_handoff=str(prepared / "handoff.json"),
        source_handoff_sha256=sha(prepared / "handoff.json"),
        dataset=dict(
            path=archive.name, sha256=sha(archive), bytes=archive.stat().st_size, members=inventory
        ),
        evaluator_truth=dict(
            path=truth_archive.name,
            sha256=sha(truth_archive),
            bytes=truth_archive.stat().st_size,
            members=truth_inventory,
        ),
        uncompressed_dataset_bytes=sum(v["bytes"] for v in inventory.values()),
        raw_included=False,
        raw_deleted=any(row.get("raw_retained") is False for row in rows),
        final_test_included=False,
        limitation="Restores frozen preprocessing only; raw parser/filter changes require original logs. Truth is evaluator-only.",
    )
    write(output / "receipt.json", receipt)
    write(
        output / "retention-plan.json",
        dict(
            action="REVIEW_ONLY_NO_DELETION",
            existing_data_protected=True,
            keep=[
                "dataset.tar.xz",
                "evaluation-truth.tar.xz",
                "receipt.json",
                "task config/manifests/logs and failure evidence",
            ],
            candidates_after_restore_test=[
                "new task raw UUID directories",
                "new task raw archives",
                "new task extracted consumers",
                "new task audit caches",
            ],
            warning="No path is automatically deleted. Raw loss prevents changing preprocessing or repairing source parsing.",
        ),
    )
    return receipt


def restore(bundle, destination, *, truth=False, budget=None, progress=None):
    bundle, destination = Path(bundle).resolve(), external(destination)
    receipt = read(bundle / "receipt.json")
    if receipt["status"] != "PORTABLE_DATASET_VERIFIED":
        raise ValueError("Verified portable receipt required")
    ref = receipt["evaluator_truth" if truth else "dataset"]
    path = bundle / ref["path"]
    if path.stat().st_size != ref["bytes"] or sha(path) != ref["sha256"]:
        raise ValueError("Portable archive changed")
    destination.mkdir(parents=True, exist_ok=False)
    verify_bundle(path, ref["members"], budget or Budget(), destination, progress=progress)
    if not truth:
        h = read(destination / "evidence/handoff.json")
        h.update(
            dataset=str(destination / "dataset"),
            portable_source_handoff_sha256=receipt["source_handoff_sha256"],
        )
        write(destination / "evidence/handoff.json", h)
    return destination


def truth_provider(bundle):
    """Explicit evaluator callback; does not expose truth to dataset/fit consumers."""
    bundle = Path(bundle).resolve()
    ref = read(bundle / "receipt.json")["evaluator_truth"]
    archive = bundle / ref["path"]
    if archive.stat().st_size != ref["bytes"] or sha(archive) != ref["sha256"]:
        raise ValueError("Portable evaluator archive changed")

    def provide(row, target, progress):
        if row["role"] not in {"normal_dev_eval", "anomaly_dev"}:
            raise ValueError("Only development evaluator may join compact truth")
        name = row["run_id"] + ".jsonl"
        safe_name(name)
        expected = row["archive"]["members"]["evaluation/events.jsonl"]
        if ref["members"].get(name) != {k: expected[k] for k in ("sha256", "bytes")}:
            raise ValueError("Portable truth differs from frozen source")
        with tarfile.open(archive, "r|xz") as tar:
            for member in tar:
                progress.check()
                if member.name == name:
                    if not member.isfile() or member.issparse() or member.size != expected["bytes"]:
                        raise ValueError("Invalid portable event member")
                    with tar.extractfile(member) as stream:
                        body = stream.read(10_000_001)
                    if (
                        len(body) != expected["bytes"]
                        or hashlib.sha256(body).hexdigest() != expected["sha256"]
                    ):
                        raise ValueError("Portable event changed")
                    with Path(target).open("xb") as output:
                        output.write(body)
                    return
        raise ValueError("Missing portable development event")

    return provide


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("pack")
    a.add_argument("prepared", type=Path)
    a.add_argument("output", type=Path)
    a = sub.add_parser("restore")
    a.add_argument("bundle", type=Path)
    a.add_argument("destination", type=Path)
    a.add_argument("--evaluator-truth", action="store_true")
    args = p.parse_args()
    if args.action == "pack":
        print(json.dumps(pack(args.prepared, args.output), indent=2))
    else:
        print(restore(args.bundle, args.destination, truth=args.evaluator_truth))


if __name__ == "__main__":
    main()
