"""Metadata-only fixed samples, disjoint groups and contiguous time integrity."""

import copy

import numpy as np
import pytest

from execution.common import write
from learning.sensitivity_data import (
    REGRESSION_ORIGIN_US,
    REGRESSION_RUN,
    SCENARIOS,
    build_plan,
    select_arrays,
    validate_plan,
    verify_selection_layout,
)


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for role in ("normal_validation", "normal_calibration"):
        for scenario in SCENARIOS:
            for i in range(4):
                identity = f"{role}-{scenario}-{i}"
                rows.append(dict(
                    run_id=identity, role=role, scenario=scenario,
                    paired_case_id="group-" + identity, health_eligibility="confirmed_healthy",
                ))
    rows.append(dict(run_id=REGRESSION_RUN, role="normal_validation", scenario="payload_normal",
                     paired_case_id="known-regression-group", health_eligibility="confirmed_healthy"))
    rows.append(dict(run_id="never-evaluate", role="anomaly_dev", scenario="battery_failure",
                     paired_case_id="unknown", health_eligibility="abnormal"))
    path = tmp_path / "manifest.json"
    write(path, {"runs": rows})
    return path, rows


def selection(run="example", *, count=96, **extra):
    return dict(role="normal_validation", run_id=run, scenario="hover_normal",
                paired_case_id="group-" + run, windows=count, position="first",
                provenance="deterministic_metadata_rank", **extra)


def arrays(runs=("example",), *, count=300):
    total = count * len(runs)
    data = dict(
        X=np.zeros((total, 100, 19), dtype=np.float32),
        C=np.zeros((total, 6), dtype=np.float32),
        Y=np.zeros((total, 10, 4), dtype=np.float32),
        run_id=np.repeat(np.array(runs), count),
        segment=np.ones(total, dtype=np.int64),
        origin_us=np.tile(400_000_000 + np.arange(count, dtype=np.int64) * 1_000_000, len(runs)),
    )
    data["Y"][:, :, :] = np.arange(total)[:, None, None]
    return data


def test_plan_is_bounded_metadata_only_and_all_groups_are_disjoint(manifest):
    path, _ = manifest
    plan = build_plan(path)
    assert validate_plan(plan, path) == plan
    assert plan["windows"] == {"scale": 288, "calibration": 288, "check": 706}
    assert plan["ordinary_selection_uses_predictions"] is False
    assert plan["development_or_final_test_used"] is False
    assert plan["diagnostic_calibration_reused_for_full"] is False
    chosen = [s for parts in plan["selections"].values() for s in parts]
    assert len(chosen) == len({s["run_id"] for s in chosen}) == 10
    assert len({s["paired_case_id"] for s in chosen}) == 10
    assert "never-evaluate" not in {s["run_id"] for s in chosen}
    regression = plan["selections"]["check"][-1]
    assert regression["run_id"] == REGRESSION_RUN
    assert regression["origin_us"] == REGRESSION_ORIGIN_US
    assert regression["provenance"] == "prior_regression"


def test_ranking_does_not_depend_on_manifest_row_order_but_hash_binds_it(manifest):
    path, rows = manifest
    first = build_plan(path)
    write(path, {"runs": list(reversed(rows))})
    second = build_plan(path)
    assert first["selections"] == second["selections"]
    assert first["manifest_sha256"] != second["manifest_sha256"]
    with pytest.raises(ValueError, match="differs"):
        validate_plan(first, path)


def test_modified_selection_is_rejected_before_model_use(manifest):
    path, _ = manifest
    plan = build_plan(path)
    plan["selections"]["check"][0]["windows"] = 255
    with pytest.raises(ValueError, match="differs"):
        validate_plan(plan, path)


def test_manifest_symlink_is_rejected(manifest, tmp_path):
    path, _ = manifest
    link = tmp_path / "alias.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular frozen"):
        build_plan(link)


@pytest.mark.parametrize("mutation", ["duplicate", "regression_missing", "unhealthy", "coverage_missing"])
def test_invalid_metadata_never_substitutes_another_condition(manifest, mutation):
    path, rows = manifest
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "regression_missing":
        rows = [r for r in rows if r["run_id"] != REGRESSION_RUN]
    elif mutation == "unhealthy":
        rows[0]["health_eligibility"] = "unknown"
    else:
        rows = [r for r in rows if r.get("scenario") != "low_soc_normal"]
    write(path, {"runs": rows})
    with pytest.raises(ValueError):
        build_plan(path)


def test_group_overlap_is_excluded_from_every_fit_and_check_set(manifest):
    path, rows = manifest
    # A role label alone is not sufficient: no chosen group may leak through
    # another role or through the known regression's ordinary candidates.
    for row in rows:
        if row["run_id"].endswith("-0"):
            row["paired_case_id"] = "known-regression-group"
    write(path, {"runs": rows})
    plan = build_plan(path)
    chosen = [s for group in plan["selections"].values() for s in group]
    assert len({s["paired_case_id"] for s in chosen}) == 10
    assert not any(s["run_id"].endswith("-0") for s in chosen)


def test_selected_arrays_preserve_input_order_and_boundaries_without_mutation():
    source = arrays(("first", "second"))
    result = select_arrays(source, [selection("second"), selection("first", count=257)])
    expected = np.r_[np.arange(300, 396), np.arange(257)]
    np.testing.assert_array_equal(result["source_indices"], expected)
    np.testing.assert_array_equal(result["Y"], source["Y"][expected])
    np.testing.assert_array_equal(result["selection_index"], np.r_[np.zeros(96), np.ones(257)])
    assert result["selection_evidence"][0]["selected_stop"] == 96
    assert result["selection_evidence"][1]["selected_start"] == 96
    assert result["selection_evidence"][1]["last_origin_us"] == 656_000_000
    result["X"][0, 0, 0] = 100
    assert source["X"][300, 0, 0] == 0


def test_regression_anchor_is_exact_and_centered():
    source = arrays((REGRESSION_RUN,))
    spec = selection(REGRESSION_RUN)
    spec.update(position="around", origin_us=REGRESSION_ORIGIN_US,
                provenance="prior_regression")
    result = select_arrays(source, [spec])
    assert result["origin_us"][48] == REGRESSION_ORIGIN_US
    assert result["source_indices"][0] == 179
    assert result["source_indices"][-1] == 274


@pytest.mark.parametrize("mutation", ["missing_anchor", "edge_anchor", "gap", "segment",
                                      "duplicate", "interleaved", "nonfinite", "short"])
def test_bad_sequence_stops_without_skipping_or_repair(mutation):
    source = arrays()
    spec = selection()
    if mutation == "missing_anchor":
        spec.update(position="around", origin_us=REGRESSION_ORIGIN_US + 1)
    elif mutation == "edge_anchor":
        spec.update(position="around", origin_us=int(source["origin_us"][2]))
    elif mutation == "gap":
        source["origin_us"][30:] += 1_000_000
    elif mutation == "segment":
        source["segment"][30:] = 2
    elif mutation == "duplicate":
        source["origin_us"][250] = source["origin_us"][249]
    elif mutation == "interleaved":
        source["run_id"][100] = "other"
    elif mutation == "nonfinite":
        source["X"][5, 0, 0] = np.nan
    else:
        source = arrays(count=40)
    with pytest.raises(ValueError):
        select_arrays(source, [spec])


@pytest.mark.parametrize("kind", ["mixed_roles", "non_normal", "duplicate_run", "duplicate_group",
                                  "wrong_shape", "bad_identity_dtype", "unbounded", "empty"])
def test_role_identity_and_shape_guards(kind):
    source = arrays(("one", "two"))
    specs = [selection("one"), selection("two")]
    if kind == "mixed_roles":
        specs[1]["role"] = "normal_calibration"
    elif kind == "non_normal":
        for s in specs:
            s["role"] = "normal_dev_eval"
    elif kind == "duplicate_run":
        specs[1] = copy.deepcopy(specs[0])
    elif kind == "duplicate_group":
        specs[1]["paired_case_id"] = specs[0]["paired_case_id"]
    elif kind == "wrong_shape":
        source["X"] = source["X"][:, :, :18]
    elif kind == "bad_identity_dtype":
        source["origin_us"] = source["origin_us"].astype(float)
    elif kind == "unbounded":
        specs[0]["windows"] = 258
    else:
        specs = []
    with pytest.raises(ValueError):
        select_arrays(source, specs)


@pytest.fixture
def layout_dataset(manifest):
    path, _ = manifest
    plan = build_plan(path)
    for role in ("normal_validation", "normal_calibration"):
        runs = [s["run_id"] for selections in plan["selections"].values()
                for s in selections if s["role"] == role]
        source = arrays(tuple(runs))
        # Layout admission must not interpret these feature values as real,
        # validated observations. The worker's value check remains mandatory.
        source["X"][0, 0, 0] = np.nan
        np.savez_compressed(path.parent / (role + ".npz"), **source)
    return plan, path.parent


def test_layout_verification_loads_only_identity_members_and_labels_its_limit(layout_dataset,
                                                                            monkeypatch):
    plan, dataset = layout_dataset
    original = np.load
    accesses = []

    class IdentityOnly:
        def __init__(self, path, **kwargs):
            self.archive = original(path, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            accesses.append(key)
            assert key in {"run_id", "segment", "origin_us"}, "Full feature array was read"
            return self.archive[key]

    monkeypatch.setattr(np, "load", IdentityOnly)
    report = verify_selection_layout(plan, dataset)
    assert report["status"] == "LAYOUT_ONLY_VERIFIED"
    assert report["feature_values_loaded"] is report["feature_quality_checked"] is False
    assert report["model_executed"] is False
    assert set(report["selection_evidence"]) == {"scale", "calibration", "check"}
    assert report["selection_evidence"]["check"][-1]["first_origin_us"] == 579_000_000
    assert accesses == ["run_id", "segment", "origin_us"] * 2
    assert report["roles"]["normal_validation"]["headers"]["X"]["shape"] == [2100, 100, 19]


@pytest.mark.parametrize("kind", ["shape", "dtype", "identity_dtype", "gap", "short"])
def test_layout_verification_rejects_true_headers_and_bad_availability(layout_dataset, kind):
    plan, dataset = layout_dataset
    path = dataset / "normal_calibration.npz"
    with np.load(path, allow_pickle=False) as source:
        changed = {key: source[key] for key in source.files}
    if kind == "shape":
        changed["X"] = changed["X"][:, :, :18]
    elif kind == "dtype":
        changed["C"] = changed["C"].astype(np.int64)
    elif kind == "identity_dtype":
        changed["origin_us"] = changed["origin_us"].astype(float)
    elif kind == "gap":
        changed["origin_us"][5] += 100_000
    else:
        changed = {key: value[250:] for key, value in changed.items()}
    np.savez_compressed(path, **changed)
    with pytest.raises(ValueError):
        verify_selection_layout(plan, dataset)


def test_layout_verification_revalidates_frozen_plan_before_npz_access(manifest):
    path, _ = manifest
    plan = build_plan(path)
    plan["selections"]["scale"][0]["windows"] = 1
    with pytest.raises(ValueError, match="differs"):
        verify_selection_layout(plan, path.parent)
