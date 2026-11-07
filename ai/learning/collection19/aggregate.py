"""Bounded dataset/scaler construction and an independent canonical re-audit."""

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from capstone_ai.dataset import feature_transform
from capstone_ai.mapping import FEATURES, TARGETS
from execution.common import child
from execution.prepare import npz_headers
from learning.admission19 import (
    ONLINE_UNITS,
    check_producer,
    check_protection,
    validate_lineages,
)
from learning.history19.data import ARRAYS
from learning.history19.pilot import ROOT, json_file, regular, sha
from learning.run_health19.data import Scaler, iter_manifests

from .cases import reference
from .contract import (
    AUTH_SHA256,
    TASK,
    bound,
    case_key,
    check_case_record,
    check_scope,
    read_protection,
    validate_policy,
    write_new,
)
from .streams import Moments, array_blocks, canonical_blocks, grid, write_npz

ROW_FIELDS = {
    "run_id",
    "case_id",
    "raw_flight_id",
    "paired_case_id",
    "group_id",
    "retry_family_id",
    "previous_attempts",
    "parent_attempt_id",
    "raw_sha256",
    "role",
    "cohort",
    "purpose",
    "health",
    "health_eligibility",
    "origin",
    "lineage",
    "mission_interval",
    "source_evidence",
    "seed",
}


def _reader_row(row):
    value = {k: v for k, v in row.items() if k in ROW_FIELDS}
    if len(json.dumps(value)) > 16384:
        raise ValueError(
            "Model row exceeds16KiB; keep archive inventories in the case capsule"
        )
    return value


def _file_identity(path):
    info = path.stat()
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _input_path(record, key):
    """Reuse one hash binding only while this invocation's source identity stays fixed."""
    ref = record["audit"]["input_refs"][key]
    cached = record.setdefault("input_paths", {})
    if key not in cached:
        path = bound(ref)
        cached[key] = path, ref["sha256"], _file_identity(path)
    path, digest, identity = cached[key]
    if (
        regular(ref["path"]) != path
        or ref["sha256"] != digest
        or _file_identity(path) != identity
    ):
        raise ValueError(
            "Admitted case input changed during aggregation: " + str(path)
        )
    return path


def _progress(
    root, stage, done, total, callback=None, *, status="running", **details
):
    import os
    import time

    value = dict(
        stage=stage,
        done=done,
        total=total,
        status=status,
        updated_monotonic=time.monotonic(),
        scope="aggregate_input_quality_only",
        whole_task_complete=False,
        **details,
    )
    if callback is not None:
        callback(value)
    root = Path(root)
    if root.is_dir():
        temp = root / "progress.json.tmp"
        temp.write_text(json.dumps(value) + "\n")
        os.replace(temp, root / "progress.json")


def _ledger(ref, case, producer, all_raw_ids, policy_bundle):
    value = json_file(bound(ref))
    attempts = value.get("attempts", [])
    maximum, recovery_rejected = 3, set()
    recovery = value.get("recovery_authorization")
    if recovery is not None:
        from .recovery import ledger_permission

        maximum, recovery_rejected = ledger_permission(recovery, case, attempts)
    if (
        value.get("schema") != "drone-ai.collection19.case-ledger/1"
        or case_key(value.get("case_id")) != case["case_id"]
        or len(attempts) > maximum
    ):
        raise ValueError("Per-case bounded original retry ledger required")
    failures, rejected, original_groups = 0, 0, None
    for i, a in enumerate(attempts, 1):
        if (
            not {
                "index",
                "run_id",
                "seed",
                "parent_attempt_id",
                "status",
                "lineage",
            }
            <= set(a)
            or type(a.get("index")) is not int
            or a["index"] != i
            or not isinstance(a.get("run_id"), str)
            or not a["run_id"]
            or a["run_id"] in all_raw_ids
            or a.get("seed") != case["seed"]
            or a.get("parent_attempt_id")
            != (attempts[i - 2]["run_id"] if i > 1 else None)
            or a.get("status")
            not in {"completed", "failed", "stopped", "quality_rejected"}
            or not a.get("lineage")
        ):
            raise ValueError(
                "Original attempt/index/seed/parent/lineage changed"
            )
        all_raw_ids.add(a["run_id"])
        groups = {
            x
            for x in a["lineage"]
            if x.startswith(("pair:", "group:", "retry:"))
        }
        if original_groups is not None and groups != original_groups:
            raise ValueError(
                "Retry changed its original paired/group/family identities"
            )
        original_groups = groups
        for field, prefix in (
            ("paired_case_id", "pair:"),
            ("group_id", "group:"),
            ("retry_family_id", "retry:"),
        ):
            if field in case and not any(
                x.startswith(prefix) and x.rsplit(":", 1)[-1] == case[field]
                for x in groups
            ):
                raise ValueError(
                    "Attempt lineage differs from immutable source case groups"
                )
        if a["status"] in {"failed", "stopped"} and not a.get("failure_reason"):
            raise ValueError("Physical failure must retain its reason")
        quality = a.get("quality_rejection")
        if a["status"] == "quality_rejected" or quality is not None:
            from .rejection import verify_quality_rejection

            if not isinstance(quality, dict):
                raise ValueError(
                    "Explicit independent quality rejection receipt required"
                )
            verify_quality_rejection(
                quality, a, producer, policy_bundle, case_id=case["case_id"]
            )
            if (
                a["status"] == "quality_rejected"
                and a.get("physical_status") != "completed"
            ):
                raise ValueError(
                    "Physical COMPLETED cannot be disguised as quality failure"
                )
            rejected += 1
        elif a["run_id"] in recovery_rejected and a["status"] == "completed":
            # Physical completion is preserved; the bound original input rejection
            # and explicit user authority permit only this selected replacement.
            rejected += 1
        elif a["status"] == "completed" and i < len(attempts):
            raise ValueError(
                "A valid completed raw cannot be blindly regenerated"
            )
        failures += a["status"] == "failed"
    return value, dict(
        physical_attempts=len(attempts),
        physical_failed_attempts=failures,
        quality_rejected_attempts=rejected,
    )


def _records(index_path, scope, producer, protection, policy_bundle):
    if scope not in {"main", "lr108"}:
        raise ValueError("Separate main540 or lr81 derived scope required")
    plans = check_scope()
    policy = json_file(policy_bundle["paired_policy"])
    validate_policy(policy, policy_bundle["native_policy"])
    check_producer(policy["producer"], producer)
    expected_ids = set(plans["cases"]) if scope == "main" else plans["lr_ids"]
    selected, seen, raw_ids, rows = [], set(), set(), []
    totals = Counter()
    with regular(index_path).open() as stream:
        for line in stream:
            if len(line) > 65536:
                raise ValueError(
                    "Collection index entry exceeds64KiB; use per-case references"
                )
            record = json.loads(line)
            cid = case_key(record["case_id"])
            if (
                record.get("schema") != "drone-ai.collection19.index-entry/1"
                or cid in seen
                or cid not in plans["cases"]
            ):
                raise ValueError("Duplicate/foreign case in collection index")
            seen.add(cid)
            if len(seen) > 1674:
                raise ValueError("Index exceeds approved logical case count")
            case = plans["cases"][cid]
            ledger, counts = _ledger(
                record["case_ledger"], case, producer, raw_ids, policy_bundle
            )
            if cid not in expected_ids:
                continue
            totals.update(counts)
            if record.get("status") not in {
                "accepted",
                "failed",
                "unavailable",
            }:
                raise ValueError(
                    "Every logical case needs accepted/failed/unavailable status"
                )
            totals[record["status"]] += 1
            if record["status"] != "accepted":
                if not record.get("failure_reason"):
                    raise ValueError(
                        "Failed/unavailable case reason cannot be dropped"
                    )
                continue
            audit_path = bound(record["audit"])
            audit = json_file(audit_path)
            fixture = policy.get("fixture_only") is True
            if (
                audit.get("schema") != "drone-ai.collection19.case-audit/1"
                or audit.get("passed") is not True
                or audit.get("task_id") != TASK
                or audit.get("actual_data_verified") is not (not fixture)
                or audit.get("fixture_only") is not fixture
                or audit.get("policy_sha256")
                != sha(regular(policy_bundle["paired_policy"]))
            ):
                raise ValueError("Actual scoped case audit receipt required")
            check_producer(audit["producer"], producer)
            row = _reader_row(audit["row"])
            check_case_record(row, case)
            if (
                not ledger["attempts"]
                or ledger.get("accepted_run_id") != row["run_id"]
                or ledger["attempts"][-1]["run_id"] != row["run_id"]
                or ledger["attempts"][-1]["status"] != "completed"
                or ledger["attempts"][-1].get("quality_rejection") is not None
            ):
                raise ValueError(
                    "Accepted export must bind the last original successful physical raw"
                )
            export = Path(record["export_root"]).absolute()
            if set(audit.get("input_refs", {})) != {
                "canonical",
                "online_manifest",
                "whole_manifest",
                "online_array",
                "whole_array",
            }:
                raise ValueError(
                    "Complete previously audited input references required"
                )
            input_paths = {}
            for key, ref in audit["input_refs"].items():
                path = bound(ref)
                input_paths[key] = path, ref["sha256"], _file_identity(path)
                if (
                    key in {"canonical", "online_manifest", "whole_manifest"}
                    and path
                    != export
                    / dict(
                        canonical="frames.jsonl",
                        online_manifest="online.json",
                        whole_manifest="run.json",
                    )[key]
                ):
                    raise ValueError("Case audit belongs to another export")
            if row["origin"] != (
                "synthetic_unit_fixture" if fixture else "px4_sitl"
            ):
                raise ValueError(
                    "Fixture cannot substitute actual collection data"
                )
            if (
                case["category"] in {"paired_normal", "anomaly"}
                and record.get("pair_evidence") is None
            ):
                raise ValueError(
                    "Paired case needs preserved official native pair evidence"
                )
            rows.append(row)
            # Keep bounded reader metadata, never archive inventories or whole raw receipts.
            audit = {
                k: v
                for k, v in audit.items()
                if k
                in {
                    "frames",
                    "first_time_us",
                    "last_time_us",
                    "segment_id",
                    "clock_id",
                    "windows",
                    "input_refs",
                    "native_check",
                }
            }
            selected.append(
                dict(
                    record=record,
                    row=row,
                    audit=audit,
                    audit_path=audit_path,
                    export=export,
                    input_paths=input_paths,
                )
            )
    if not expected_ids <= seen:
        raise ValueError(
            "All planned logical cases, including failures, must remain in index"
        )
    validate_lineages(rows)
    identities = {"run:" + rid for rid in raw_ids} | {
        key
        for row in rows
        for key in [
            "case:" + row["case_id"],
            "run:" + row["run_id"],
            *row["lineage"],
        ]
    }
    registry = read_protection(protection, identities)
    check_protection(rows, registry, usage="input_check")
    totals["planned_cases"] = len(expected_ids)
    return selected, registry, dict(totals), policy


def _pairs(records, producer, policy_bundle):
    """Audit immutable before-retirement pair report plus rechecked case consumption proofs.

    Applied-wind truth evaluation is preserved official evidence, not rerun from retired raw.
    """
    groups = {}
    policy = json_file(policy_bundle["paired_policy"])
    for record in records:
        ref = record["record"].get("pair_evidence")
        if ref is not None:
            groups.setdefault(record["row"]["paired_case_id"], []).append(
                (record, ref)
            )
    seen_reports = set()
    for pair in groups.values():
        if len(pair) != 2 or pair[0][1]["sha256"] != pair[1][1]["sha256"]:
            raise ValueError(
                "Both members must bind one original official pair report"
            )
        bound(pair[1][1])
        path = bound(pair[0][1])
        digest = sha(path)
        if digest in seen_reports:
            raise ValueError(
                "Native report cannot stand in for independent pairs"
            )
        seen_reports.add(digest)
        report = json_file(path)
        if (
            report.get("schema") != "drone.joint-phase-pair-check/1"
            or report.get("status") != "passed"
            or report.get("errors")
            or report.get("original_interval_us") != [0, 480000000]
            or report.get("source_sha256") != producer["source_manifest_sha256"]
            or report.get("runtime_sha256") != producer["runtime_sha256"]
            or report.get("policy", {}).get("policy_sha256")
            != policy["native_policy_sha256"]
            or report.get("policy", {}).get("revision")
            != policy["native_adapter"]
            or report.get("interpolation_used") is not False
            or report.get("wind", {}).get("samples") != 480001
            or report["wind"].get("expected_samples") != 480001
            or report["wind"].get("missing_samples") != 0
            or report.get("reference", {}).get("rows") != 4801
            or report["reference"].get("missing_rows") != 0
        ):
            raise ValueError(
                "Original full480 official pair scope/coverage differs"
            )
        for container, key, limit in (
            ("wind", "max_vector_difference_mps", 2e-9),
            ("wind", "rms_vector_difference_mps", 2e-9),
            ("reference", "max_position_difference_m", 1e-8),
            ("reference", "max_velocity_difference_mps", 1e-8),
            ("reference", "max_circular_yaw_difference_rad", 1e-8),
        ):
            value = report[container].get(key)
            if (
                type(value) not in (int, float)
                or not np.isfinite(value)
                or not 0 <= value <= limit
            ):
                raise ValueError("Preserved native pair criterion failed")
        checks = report.get("wind_function_checks", [])
        if len(checks) != 2 or any(
            c.get("status") != "passed"
            or c.get("mission_samples") != 480001
            or not 0 <= c.get("max_abs_component_error_mps", -1) <= 2e-9
            for c in checks
        ):
            raise ValueError("Official full applied-wind oracle proof missing")
        for member, _ in pair:
            role = (
                "anomaly"
                if member["row"]["role"] == "anomaly_test"
                else "normal"
            )
            if (
                Path(report[role + "_run"]).name != member["row"]["run_id"]
                or report["native_consumption"][role]
                != member["audit"]["native_check"]
                or report["origins"][role]
                != member["row"]["mission_interval"]["start_us"]
            ):
                raise ValueError(
                    "Pair report differs from preserved physical case receipts"
                )
    return len(groups)


def _new_root(output):
    path = Path(output).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("New regular dataset root required")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _compile(
    records, output, *, scope, producer, registry, fixture_only, progress=None
):
    """Low-level bounded writer; production entry below enforces the complete frozen scope."""
    root = _new_root(output)
    records = sorted(records, key=lambda r: r["row"]["run_id"])
    import re

    if any(
        not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", r["row"]["run_id"])
        for r in records
    ):
        raise ValueError(
            "Safe original run identifiers required for dataset filenames"
        )
    train = [r for r in records if r["row"]["role"] == "normal_train"]
    validation = [r for r in records if r["row"]["role"] == "normal_validation"]
    if not train or not validation:
        raise ValueError(
            "Independent train and validation required; no scaler from other roles"
        )
    train_ids = sorted(r["row"]["run_id"] for r in train)
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Duplicate physical train run")
    check_protection([r["row"] for r in train], registry, usage="fit")
    config = json_file(ROOT / "ai/learning/history19/preprocessing.json")
    moments = Moments()
    nframes = sum(grid(r["row"])[2] for r in train)
    max_run = max(len(r["row"]["run_id"]) for r in train)
    max_clock = max(
        len(r["row"]["mission_interval"]["clock_id"]) for r in train
    )

    def frames(field):
        for run_index, record in enumerate(train, 1):
            row = record["row"]
            if field == "values":
                for block, _, _ in canonical_blocks(
                    record["export"] / "frames.jsonl", row
                ):
                    values = feature_transform(block, config)
                    moments.add(values)
                    yield values
                _progress(
                    root,
                    "train_statistics",
                    run_index,
                    len(train),
                    progress,
                    scaler_scope=scope,
                )
            else:
                first, _, size = grid(row)
                for offset in range(0, size, 8192):
                    count = min(8192, size - offset)
                    if field == "time_us":
                        yield (
                            first
                            + np.arange(offset, offset + count, dtype=np.int64)
                            * 100000
                        )
                    elif field == "segment_id":
                        yield np.full(
                            count, record["audit"]["segment_id"], dtype=np.int64
                        )
                    elif field == "run_id":
                        yield np.full(
                            count, row["run_id"], dtype=f"<U{max_run}"
                        )
                    else:
                        yield np.full(
                            count,
                            row["mission_interval"]["clock_id"],
                            dtype=f"<U{max_clock}",
                        )

    frame_specs = {"values": ((nframes, 19), "<f8", frames("values"))}
    for key, dtype in (
        ("run_id", f"<U{max_run}"),
        ("clock_id", f"<U{max_clock}"),
        ("segment_id", "<i8"),
        ("time_us", "<i8"),
    ):
        frame_specs[key] = ((nframes,), dtype, frames(key))
    write_npz(root / "normal_train_frames.npz", frame_specs)
    mean, scale = moments.finish()
    target_indices = [FEATURES.index(k) for k in TARGETS]
    scaler = dict(
        x_mean=mean,
        x_scale=scale,
        y_mean=mean[target_indices],
        y_scale=scale[target_indices],
    )
    np.savez(root / "scaler.npz", **scaler)
    provenance = dict(
        scope=scope,
        scaler_scope="normal_train_unique_valid_frames",
        fit_run_ids=train_ids,
        unique_valid_mission_frames=moments.count,
        ddof=0,
        scale_floor=1e-5,
    )
    write_new(
        root / "whole-scaler.json",
        dict(
            mean=mean.tolist(),
            scale=scale.tolist(),
            fit_run_ids=train_ids,
            provenance=provenance,
        ),
    )
    write_new(root / "scaler-provenance.json", provenance)
    files = {
        key: dict(path=name, sha256=sha(root / name))
        for key, name in (
            ("normal_train_frames", "normal_train_frames.npz"),
            ("scaler", "scaler.npz"),
        )
    }
    for role, selected in (
        ("normal_train", train),
        ("normal_validation", validation),
    ):
        # _records already bound these immutable case inputs in this invocation.
        sources = [_input_path(r, "online_array") for r in selected]
        headers = [npz_headers(p) for p in sources]
        count = sum(h["X"]["shape"][0] for h in headers)
        specs = {}
        for key in sorted(ARRAYS):
            shape = (count, *(headers[0][key]["shape"][1:]))
            if key in {"X", "Y"}:
                dtype = np.dtype("<f8")
            elif key in {"run_id", "clock_id"}:
                width = max(
                    np.dtype(h[key]["descr"]).itemsize
                    // (4 if np.dtype(h[key]["descr"]).kind == "U" else 1)
                    for h in headers
                )
                dtype = np.dtype(f"<U{width}")
            else:
                dtype = np.dtype("<i8")

            def blocks(sources=sources, key=key, role=role):
                for run_index, source in enumerate(sources, 1):
                    for _, a in array_blocks(source, key):
                        yield a
                    _progress(
                        root,
                        "copy_online",
                        run_index,
                        len(sources),
                        progress,
                        role=role,
                        array=key,
                    )

            specs[key] = (shape, dtype, blocks())
        write_npz(root / (role + ".npz"), specs)
        files[role] = dict(
            path=role + ".npz", sha256=sha(root / (role + ".npz"))
        )
    manifest = dict(
        schema="drone-ai.history19.dataset/1",
        fixture_only=fixture_only,
        feature_profile="common19_ctx0",
        feature_order=FEATURES,
        target_order=TARGETS,
        history=100,
        future=10,
        step_us=100000,
        emit_every_us=1000000,
        units=ONLINE_UNITS,
        interval_policy="mission_only_actual_execution/1",
        producer=producer,
        scaler_scope="normal_train_unique_valid_frames",
        scaler_fit_run_ids=train_ids,
        runs=[
            r["row"]
            for r in records
            if r["row"]["role"] in {"normal_train", "normal_validation"}
        ],
        files=files,
        collection_scope=scope,
    )
    write_new(root / "history19-manifest.json", manifest)
    (root / "runs").mkdir()
    whole_rows, shards = [], []

    def flush_shard():
        if not whole_rows:
            return
        name = "run-health19-" + str(len(shards)).zfill(3) + ".json"
        write_new(
            root / name,
            dict(
                schema="drone-ai.run-health19.dataset/1",
                feature_order=FEATURES,
                units="physical_common19_pack_V_A",
                step_us=100000,
                interval_policy="mission_only_actual_execution/1",
                vehicle_profile="x500_reference",
                battery_policy="cmu-vah01-rpt16-only/1",
                producer=producer,
                runs=list(whole_rows),
                collection_scope=scope,
            ),
        )
        shards.append(dict(path=name, sha256=sha(root / name)))
        whole_rows.clear()

    for run_index, record in enumerate(records, 1):
        source = _input_path(record, "whole_array")
        relative = "runs/" + record["row"]["run_id"] + ".npz"
        target = root / relative
        shutil.copyfile(source, target)
        copied_sha = sha(target)
        if copied_sha != record["audit"]["input_refs"]["whole_array"]["sha256"]:
            raise ValueError("Whole-run copy differs")
        original = _reader_row(
            json_file(record["export"] / "run.json")["runs"][0]
        )
        whole_rows.append(dict(original, path=relative, sha256=copied_sha))
        if len(whole_rows) == 64:
            flush_shard()
        _progress(root, "copy_whole", run_index, len(records), progress)
    flush_shard()
    write_new(
        root / "run-health19-manifest.json",
        dict(
            schema="drone-ai.run-health19.index/1",
            manifests=shards,
            scaler=dict(
                path="whole-scaler.json", sha256=sha(root / "whole-scaler.json")
            ),
            collection_scope=scope,
        ),
    )
    write_new(root / "protection.json", registry)
    with (root / "online-index.jsonl").open("x") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    dict(
                        case_id=case_key(record["row"]["case_id"]),
                        role=record["row"]["role"],
                        cohort=record["row"].get("cohort"),
                        manifest=record["audit"]["input_refs"][
                            "online_manifest"
                        ],
                    )
                )
                + "\n"
            )
    return root, provenance


def aggregate(
    index_path,
    output,
    *,
    scope,
    producer,
    protection,
    policy_bundle,
    progress=None,
):
    records, registry, counts, policy = _records(
        index_path, scope, producer, protection, policy_bundle
    )
    if counts.get("accepted", 0) != counts["planned_cases"]:
        root = _new_root(output)
        result = dict(
            schema="drone-ai.collection19.aggregate/1",
            passed=False,
            status="INCOMPLETE_NO_DATASET_COMMIT",
            scope=scope,
            counts=counts,
            eligible_for_fit=False,
        )
        write_new(root / "incomplete.json", result)
        _progress(
            root,
            "incomplete",
            counts.get("accepted", 0),
            counts["planned_cases"],
            progress,
            status="incomplete",
        )
        return result
    pairs = _pairs(records, producer, policy_bundle)
    if pairs != (324 if scope == "main" else 0):
        raise ValueError("Complete frozen native pair denominator required")
    try:
        root, provenance = _compile(
            records,
            output,
            scope=scope,
            producer=producer,
            registry=registry,
            fixture_only=policy.get("fixture_only") is True,
            progress=progress,
        )
        # Standalone audit_aggregate still binds inputs afresh. This in-process
        # audit reuses the same admitted records and performs the output checks.
        result = _audit_compiled(
            records,
            root,
            scope=scope,
            producer=producer,
            registry=registry,
            progress=progress,
        )
        result.update(
            index=reference(index_path),
            authorization_sha256=AUTH_SHA256,
            counts=counts,
        )
        independent = write_new(root / "independent-audit.json", result)
    except Exception as error:
        _progress(
            output,
            "error",
            0,
            len(records),
            progress,
            status="failed",
            reason=str(error),
        )
        raise
    result.update(
        counts=counts,
        native_pairs=pairs,
        scaler_provenance=provenance,
        independent_audit=independent,
    )
    receipt = write_new(root / "aggregate-receipt.json", result)
    _progress(
        root,
        "aggregate_inputs_audited",
        len(records),
        len(records),
        progress,
        status="complete",
    )
    return dict(result, receipt=receipt)


def whole_manifest_paths(dataset_root):
    root = Path(dataset_root)
    value = json_file(root / "run-health19-manifest.json")
    if value.get("schema") != "drone-ai.run-health19.index/1":
        raise ValueError("Sharded whole-run index required")
    paths = []
    for ref in value["manifests"]:
        p = child(root, ref["path"])
        if sha(p) != ref["sha256"]:
            raise ValueError("Whole-run manifest shard changed")
        paths.append(p)
    return paths


def load_shared_scaler(dataset_root):
    root = Path(dataset_root)
    manifest = json_file(root / "run-health19-manifest.json")
    path = child(root, manifest["scaler"]["path"])
    if sha(path) != manifest["scaler"]["sha256"]:
        raise ValueError("Shared scaler hash changed")
    return Scaler.from_dict(json_file(path))


def _compare_packed(path, key, expected):
    packed = iter(array_blocks(path, key))
    remainder = None
    for values in expected:
        offset = 0
        while offset < len(values):
            if remainder is None or not len(remainder):
                try:
                    _, remainder = next(packed)
                except StopIteration:
                    raise ValueError(
                        "Packed member lost canonical/source rows"
                    ) from None
            n = min(len(remainder), len(values) - offset)
            if not np.array_equal(remainder[:n], values[offset : offset + n]):
                raise ValueError(
                    "Packed member differs from original canonical/source values or identity"
                )
            remainder, offset = remainder[n:], offset + n
    if (remainder is not None and len(remainder)) or next(
        packed, None
    ) is not None:
        raise ValueError("Packed member has excess source rows")


def _audit_compiled(
    records, dataset_root, *, scope, producer, registry, progress=None
):
    """Separate two-pass long-double canonical statistics and source-to-packed row comparison."""
    root = Path(dataset_root)
    manifest = json_file(root / "history19-manifest.json")
    if manifest.get("collection_scope") != scope:
        raise ValueError("MAIN540 and LR81 scaler scopes cannot be mixed")
    train = sorted(
        (r for r in records if r["row"]["role"] == "normal_train"),
        key=lambda r: r["row"]["run_id"],
    )
    train_ids = sorted(r["row"]["run_id"] for r in train)
    if manifest["scaler_fit_run_ids"] != train_ids:
        raise ValueError("Train-only membership changed")
    # _records verified the successful per-case audit and every bound input hash.
    # Reuse that unchanged-input proof; below we independently check the newly
    # compiled arrays and train-only scalers, rather than rebuilding case views.
    _progress(
        root, "reuse_verified_case_inputs", len(records), len(records), progress
    )
    with np.load(root / "scaler.npz", allow_pickle=False) as a:
        scaler = dict(a)
    config = json_file(ROOT / "ai/learning/history19/preprocessing.json")
    count, total = 0, np.zeros(19, dtype=np.longdouble)
    for run_index, record in enumerate(train, 1):
        for values, _, segment in canonical_blocks(
            record["export"] / "frames.jsonl", record["row"]
        ):
            if segment != record["audit"]["segment_id"]:
                raise ValueError(
                    "Canonical mission segment differs from admitted case audit"
                )
            block = feature_transform(values, config).astype(np.longdouble)
            total += block.sum(0)
            count += len(block)
        _progress(
            root,
            "verify_statistics_mean",
            run_index,
            len(train),
            progress,
            scaler_scope=scope,
        )
    center = total / count
    squares = np.zeros(19, dtype=np.longdouble)
    for run_index, record in enumerate(train, 1):
        for values, _, _ in canonical_blocks(
            record["export"] / "frames.jsonl", record["row"]
        ):
            squares += (
                (
                    feature_transform(values, config).astype(np.longdouble)
                    - center
                )
                ** 2
            ).sum(0)
        _progress(
            root,
            "verify_statistics_scale",
            run_index,
            len(train),
            progress,
            scaler_scope=scope,
        )
    scale = np.maximum(np.sqrt(squares / count), 1e-5)
    if not np.allclose(
        scaler["x_mean"], center, atol=1e-10, rtol=1e-10
    ) or not np.allclose(scaler["x_scale"], scale, atol=1e-10, rtol=1e-10):
        raise ValueError(
            "Scaler differs from independent unique mission canonical statistics"
        )
    indices = [FEATURES.index(k) for k in TARGETS]
    if any(
        not np.array_equal(scaler["y_" + k], scaler["x_" + k][indices])
        for k in ("mean", "scale")
    ):
        raise ValueError(
            "Y scaler cannot be independently refitted or use future targets"
        )
    frame_fields = ("values", "run_id", "clock_id", "segment_id", "time_us")
    frame_path = root / "normal_train_frames.npz"
    frame_headers = npz_headers(frame_path)
    if set(frame_headers) != set(frame_fields):
        raise ValueError(
            "Train scaler requires19-column frames and identity only"
        )
    for field in frame_fields:
        shape = frame_headers[field]["shape"]
        dtype = np.dtype(frame_headers[field]["descr"])
        expected_kind = (
            "f"
            if field == "values"
            else "US"
            if field in {"run_id", "clock_id"}
            else "iu"
        )
        if (
            tuple(shape) != ((count, 19) if field == "values" else (count,))
            or dtype.kind not in expected_kind
        ):
            raise ValueError("Train frame dtype/identity mismatch")

    for field in frame_fields:

        def expected(field=field):
            for record in train:
                row = record["row"]
                if field == "values":
                    for values, _, _ in canonical_blocks(
                        record["export"] / "frames.jsonl", row
                    ):
                        yield feature_transform(values, config)
                    continue
                # The canonical passes above already established the complete
                # grid and one segment. Generate its identities in bounded blocks
                # instead of reparsing the same JSON four more times.
                first, _, size = grid(row)
                for offset in range(0, size, 8192):
                    n = min(8192, size - offset)
                    if field == "time_us":
                        yield (
                            first
                            + np.arange(offset, offset + n, dtype=np.int64)
                            * 100000
                        )
                    elif field == "segment_id":
                        yield np.full(
                            n, record["audit"]["segment_id"], dtype=np.int64
                        )
                    else:
                        yield np.full(
                            n,
                            row["run_id"]
                            if field == "run_id"
                            else row["mission_interval"]["clock_id"],
                        )

        _compare_packed(frame_path, field, expected())
    shared = load_shared_scaler(root)
    if (
        list(shared.fit_run_ids) != train_ids
        or not np.array_equal(shared.mean, scaler["x_mean"])
        or not np.array_equal(shared.scale, scaler["x_scale"])
    ):
        raise ValueError(
            "Both model representations must share exactly the same scaler"
        )
    # Compare every packed source row; hashes/headers alone are not a coverage test.
    for role in ("normal_train", "normal_validation"):
        selected = sorted(
            (r for r in records if r["row"]["role"] == role),
            key=lambda r: r["row"]["run_id"],
        )
        for key in sorted(ARRAYS):

            def expected(key=key):
                for record in selected:
                    for _, values in array_blocks(
                        _input_path(record, "online_array"), key
                    ):
                        yield values

            _compare_packed(root / (role + ".npz"), key, expected())
            _progress(
                root,
                "verify_online",
                list(sorted(ARRAYS)).index(key) + 1,
                len(ARRAYS),
                progress,
                role=role,
                array=key,
            )
    seen = 0
    for run in iter_manifests(
        whole_manifest_paths(root),
        expected_producer=producer,
        protection=registry,
    ):
        seen += 1
        del run
        _progress(root, "verify_whole", seen, len(records), progress)
    if seen != len(records):
        raise ValueError("Whole-run manifest lost a logical case")
    for record in records:
        for key in record["audit"]["input_refs"]:
            _input_path(record, key)
    return dict(
        schema="drone-ai.collection19.aggregate-audit/1",
        task_id=TASK,
        passed=True,
        status="DATASET_INPUTS_AUDITED_NOT_TRAINED",
        scope=scope,
        unique_valid_train_frames=count,
        fit_run_ids=train_ids,
        scaler_fit_roles=["normal_train"],
        canonical_scaler_tolerance=dict(atol=1e-10, rtol=1e-10),
        fixture_only=all(
            r["row"]["origin"] == "synthetic_unit_fixture" for r in records
        ),
        actual_data_verified=all(
            r["row"]["origin"] == "px4_sitl" for r in records
        ),
        eligible_for_fit=False,
        authorizes_training=False,
        model_forward_steps=0,
        optimizer_steps=0,
        GPU_jobs=0,
    )


def audit_aggregate(
    index_path,
    dataset_root,
    *,
    scope,
    producer,
    protection,
    policy_bundle,
    output=None,
    progress=None,
):
    records, registry, counts, _ = _records(
        index_path, scope, producer, protection, policy_bundle
    )
    if counts.get("accepted", 0) != counts["planned_cases"]:
        raise ValueError("Incomplete collection cannot pass aggregate quality")
    pairs = _pairs(records, producer, policy_bundle)
    if pairs != (324 if scope == "main" else 0):
        raise ValueError("Complete frozen native pair denominator required")
    result = _audit_compiled(
        records,
        dataset_root,
        scope=scope,
        producer=producer,
        registry=registry,
        progress=progress,
    )
    result.update(
        index=reference(index_path),
        authorization_sha256=AUTH_SHA256,
        counts=counts,
    )
    receipt = write_new(output, result) if output is not None else None
    return dict(result, receipt=receipt)
