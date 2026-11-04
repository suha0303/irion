"""Frozen normal-only samples for one finite batch-sensitivity check.

Selection uses metadata, never predictions, losses or event labels. The one
explicit prior regression case is identified separately from ordinary coverage.
No fallback run, shorter block or discontinuity skipping is allowed at runtime.
"""

import hashlib
from pathlib import Path

import numpy as np

from execution.common import child, read, sha

SCHEMA = "drone-ai.sensitivity-selection/1"
SCENARIOS = ("hover_normal", "payload_normal", "low_soc_normal")
NORMAL_ROLES = {"normal_validation", "normal_calibration"}
REGRESSION_RUN = "7b3f22f62fd44691bb51d04de9054631"
REGRESSION_ORIGIN_US = 627_000_000
RANK_PREFIX = "x500-sensitivity-v1|"
STEP_US = 1_000_000
_ARRAY_SHAPES = {"X": (100, 19), "C": (6,), "Y": (10, 4)}


def _text(value, label):
    if not isinstance(value, str) or not value or len(value) > 255 or any(
        ord(c) < 32 for c in value
    ):
        raise ValueError("Missing or invalid selection identity: " + label)
    return value


def _rank(row):
    return hashlib.sha256((RANK_PREFIX + row["run_id"]).encode()).hexdigest(), row["run_id"]


def build_plan(manifest_path):
    """Select ten fixed run blocks and bind the complete upstream manifest hash."""
    manifest_path = Path(manifest_path)
    if (any(p.is_symlink() for p in (manifest_path, *manifest_path.parents))
            or not manifest_path.is_file()):
        raise ValueError("A regular frozen dataset manifest is required")
    rows = read(manifest_path).get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Frozen manifest needs nonempty runs")
    seen = set()
    normal = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid manifest run")
        run_id = _text(row.get("run_id"), "run_id")
        if run_id in seen:
            raise ValueError("Duplicate manifest run identity: " + run_id)
        seen.add(run_id)
        if row.get("role") in NORMAL_ROLES:
            _text(row.get("paired_case_id"), "paired_case_id")
            if row.get("health_eligibility") != "confirmed_healthy":
                raise ValueError("Diagnostic selection requires confirmed healthy normal runs")
            normal.append(row)
    regression = next((r for r in normal if r["run_id"] == REGRESSION_RUN), None)
    if (regression is None or regression["role"] != "normal_validation"
            or regression.get("scenario") != "payload_normal"):
        raise ValueError("Expected prior normal-validation payload regression run is absent")
    used_ids = {REGRESSION_RUN}
    used_groups = {regression["paired_case_id"]}
    selections = {"scale": [], "calibration": [], "check": []}
    for purpose, role in (("scale", "normal_validation"),
                          ("calibration", "normal_calibration"),
                          ("check", "normal_validation")):
        for scenario in SCENARIOS:
            candidates = sorted(
                (r for r in normal if r["role"] == role and r.get("scenario") == scenario
                 and r["run_id"] not in used_ids and r["paired_case_id"] not in used_groups),
                key=_rank,
            )
            if not candidates:
                raise ValueError(f"No unused normal run/group for {purpose}/{scenario}")
            row = candidates[0]
            used_ids.add(row["run_id"])
            used_groups.add(row["paired_case_id"])
            selections[purpose].append(dict(
                role=role, scenario=scenario, run_id=row["run_id"],
                paired_case_id=row["paired_case_id"], position="first",
                windows=257 if purpose == "check" and scenario != "low_soc_normal" else 96,
                provenance="deterministic_metadata_rank",
            ))
    selections["check"].append(dict(
        role="normal_validation", scenario="payload_normal", run_id=REGRESSION_RUN,
        paired_case_id=regression["paired_case_id"], windows=96, position="around",
        origin_us=REGRESSION_ORIGIN_US, provenance="prior_regression",
    ))
    return dict(
        schema=SCHEMA, manifest_sha256=sha(manifest_path), selections=selections,
        ranking="sha256(" + RANK_PREFIX + "run_id), ascending; then run_id",
        origin_step_us=STEP_US, ordinary_selection_uses_predictions=False,
        development_or_final_test_used=False, diagnostic_calibration_reused_for_full=False,
        run_and_paired_group_disjoint=True,
        failure_policy="Stop on missing, short, duplicate or discontinuous input; no fallback",
        windows={purpose: sum(s["windows"] for s in parts)
                 for purpose, parts in selections.items()},
    )


def validate_plan(plan, manifest_path):
    """Require the exact metadata-derived plan, including recorded regression scope."""
    expected = build_plan(manifest_path)
    if plan != expected:
        raise ValueError("Frozen sensitivity selection differs from the dataset manifest")
    return plan


def verify_selection_layout(plan, dataset_path):
    """Check availability before GPU submission, reading identities and headers only.

    Broadcast zeros are shape-only placeholders, never observations. The report
    expressly does not establish feature quality; the allocated worker must hash
    the frozen inputs and validate the real normalized values before prediction.
    """
    from execution.prepare import npz_headers

    dataset_path = Path(dataset_path)
    validate_plan(plan, child(dataset_path, "manifest.json"))
    report = dict(
        schema="drone-ai.sensitivity-layout/1", status="LAYOUT_ONLY_VERIFIED",
        manifest_sha256=plan["manifest_sha256"], feature_values_loaded=False,
        feature_quality_checked=False, model_executed=False, roles={}, selection_evidence={},
        limitation="Headers and identity continuity only; worker must verify hashes and real values",
    )
    required = {*_ARRAY_SHAPES, "run_id", "segment", "origin_us"}
    for role in sorted(NORMAL_ROLES):
        path = child(dataset_path, role + ".npz")
        headers = npz_headers(path)
        if not required <= set(headers):
            raise ValueError("Missing normal-role NPZ headers: " + role)
        identity_header = headers["run_id"]
        shape = tuple(identity_header["shape"])
        if (len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0
                or np.dtype(identity_header["descr"]).kind not in "US"):
            raise ValueError("Invalid normal-role run identity header")
        count = shape[0]
        for key, tail in _ARRAY_SHAPES.items():
            header = headers[key]
            if (tuple(header["shape"]) != (count, *tail)
                    or np.dtype(header["descr"]).kind != "f"):
                raise ValueError("Invalid real normal-role feature header: " + key)
        for key in ("segment", "origin_us"):
            header = headers[key]
            if tuple(header["shape"]) != (count,) or np.dtype(header["descr"]).kind not in "iu":
                raise ValueError("Invalid normal-role time identity header: " + key)
        with np.load(path, allow_pickle=False) as archive:
            # Do not index X/C/Y members: even np.load on compressed members would
            # materialize the entire role, defeating this host-only admission.
            identities = {key: archive[key] for key in ("run_id", "segment", "origin_us")}
        layout = {
            **identities,
            **{key: np.broadcast_to(np.array(0, dtype=np.dtype(headers[key]["descr"])),
                                   tuple(headers[key]["shape"]))
               for key in _ARRAY_SHAPES},
        }
        report["roles"][role] = dict(
            file=path.name, rows=count,
            headers={key: dict(shape=list(headers[key]["shape"]), dtype=headers[key]["descr"])
                     for key in sorted(required)},
            loaded_members=["run_id", "segment", "origin_us"],
        )
        for purpose, selections in plan["selections"].items():
            if selections[0]["role"] == role:
                selected = select_arrays(layout, selections)
                report["selection_evidence"][purpose] = selected["selection_evidence"]
        del layout, identities
    if set(report["selection_evidence"]) != {"scale", "calibration", "check"}:
        raise ValueError("Missing frozen selection layout evidence")
    return report


def select_arrays(normal_role_arrays, selections):
    """Copy only the frozen contiguous blocks from one normalized normal-role array.

    Returned X/C/Y and identities are in selection order. ``source_indices``
    identifies original rows; ``selection_index`` marks the block of each row.
    ``selection_evidence`` contains JSON-safe block boundaries. The caller must
    validate the frozen plan and load the role declared by every selection.
    """
    if not isinstance(selections, list) or not 1 <= len(selections) <= 10:
        raise ValueError("A bounded nonempty list of frozen normal selections is required")
    roles = {s.get("role") for s in selections if isinstance(s, dict)}
    if len(roles) != 1 or not roles <= NORMAL_ROLES:
        raise ValueError("Selections must come from one normal role only")
    required = {*_ARRAY_SHAPES, "run_id", "segment", "origin_us"}
    if not isinstance(normal_role_arrays, dict) or not required <= set(normal_role_arrays):
        raise ValueError("Normalized normal arrays and identities are required")
    arrays = {key: np.asarray(normal_role_arrays[key]) for key in required}
    ids = arrays["run_id"]
    if ids.ndim != 1 or ids.dtype.kind not in "US" or len(ids) == 0:
        raise ValueError("One-dimensional nonempty string run identities required")
    ids = ids.astype(str, copy=False)
    count = len(ids)
    for key, shape in _ARRAY_SHAPES.items():
        value = arrays[key]
        if value.shape != (count, *shape) or value.dtype.kind != "f":
            raise ValueError("Unexpected normal diagnostic array shape/type: " + key)
    for key in ("segment", "origin_us"):
        value = arrays[key]
        if value.shape != (count,) or value.dtype.kind not in "iu":
            raise ValueError("Integer normal diagnostic identity required: " + key)
    used_runs, used_groups = set(), set()
    all_indices, blocks, evidence = [], [], []
    for block, selection in enumerate(selections):
        if not isinstance(selection, dict):
            raise ValueError("Invalid frozen normal selection")
        run_id = _text(selection.get("run_id"), "run_id")
        group = _text(selection.get("paired_case_id"), "paired_case_id")
        if run_id in used_runs or group in used_groups:
            raise ValueError("Duplicate run/group in diagnostic selections")
        used_runs.add(run_id)
        used_groups.add(group)
        windows = selection.get("windows")
        if type(windows) is not int or not 1 <= windows <= 257:
            raise ValueError("Diagnostic block size must be in1..257")
        indices = np.flatnonzero(ids == run_id)
        if len(indices) < windows:
            raise ValueError("Frozen diagnostic run is missing or too short: " + run_id)
        if np.any(np.diff(indices) != 1):
            raise ValueError("Run rows are interleaved; frozen order must not be repaired")
        origins, segments = arrays["origin_us"][indices], arrays["segment"][indices]
        limit = np.iinfo(np.int64).max
        if (np.any(origins < 0) or np.any(segments < 0)
                or np.any(origins > limit) or np.any(segments > limit)):
            raise ValueError("Normal time/segment identity must be nonnegative signed64")
        origins, segments = origins.astype(np.int64), segments.astype(np.int64)
        identity = np.stack((segments, origins), axis=1)
        if len(np.unique(identity, axis=0)) != len(identity):
            raise ValueError("Duplicate normal run/segment/origin identity")
        position = selection.get("position")
        if position == "first":
            if "origin_us" in selection:
                raise ValueError("First-block selection cannot silently specify an anchor")
            start = 0
        elif position == "around":
            anchor = selection.get("origin_us")
            if type(anchor) is not int or anchor < 0:
                raise ValueError("Exact integer regression origin required")
            centers = np.flatnonzero(origins == anchor)
            if len(centers) != 1:
                raise ValueError("Exact regression origin is missing or ambiguous")
            start = int(centers[0]) - windows // 2
            if start < 0 or start + windows > len(indices):
                raise ValueError("Centered regression block cannot be shortened or shifted")
        else:
            raise ValueError("Unknown frozen diagnostic block position")
        selected = indices[start:start + windows]
        selected_segments = arrays["segment"][selected]
        selected_origins = arrays["origin_us"][selected]
        if (len(np.unique(selected_segments)) != 1
                or np.any(np.diff(selected_origins) != STEP_US)):
            raise ValueError("Frozen diagnostic block must stay within one contiguous1s segment")
        if any(not np.isfinite(arrays[key][selected]).all() for key in _ARRAY_SHAPES):
            raise ValueError("Nonfinite selected normal diagnostic values")
        offset = sum(len(x) for x in all_indices)
        evidence.append(dict(
            **selection, selected_start=offset, selected_stop=offset + windows,
            source_start_index=int(selected[0]), source_stop_index=int(selected[-1]) + 1,
            segment=int(selected_segments[0]), first_origin_us=int(selected_origins[0]),
            last_origin_us=int(selected_origins[-1]),
        ))
        all_indices.append(selected)
        blocks.append(np.full(windows, block, dtype=np.int64))
    chosen = np.concatenate(all_indices)
    if len(chosen) > 2570 or len(np.unique(chosen)) != len(chosen):
        raise ValueError("Diagnostic selections overlap or exceed finite bounds")
    result = {key: value[chosen].copy() for key, value in arrays.items()}
    result["run_id"] = result["run_id"].astype(str)
    result.update(source_indices=chosen, selection_index=np.concatenate(blocks),
                  selection_evidence=evidence)
    return result
