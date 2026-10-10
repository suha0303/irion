"""Current collection names retain the unchanged sealed preprocessing guards."""

import copy

import pytest
from pretrain import collection, common


def admission():
    common.package_imports()
    return common.load("collection_identity_admission", common.ADMISSION / "freeze.py")


def test_new_collection_cases_have_no_historical_round_identity():
    record = collection.cases(124500, "balanced300")
    assert record["revision"] == "COLLECTION"
    assert record["data_pool_id"] == "collection/x500"
    assert len(record["cases"]) == 300
    assert all(not row["case_id"].startswith("R3_") for row in record["cases"])


@pytest.mark.parametrize("historical", [False, True])
def test_collection_admission_translates_only_identity_and_storage(historical):
    module = admission()
    plan = common.read(common.RESOURCES / "collection-plan.json")
    plan["storage_root"] = str(module.STORAGE)
    if historical:
        plan.update(round_id="ROUND_001", revision="R3_LONG15")
    before = copy.deepcopy(plan)
    module.validate_plan(plan)
    assert plan == before
    # Semantic policy must still pass through the sealed guard.
    plan["features"]["history_steps"] = 50
    with pytest.raises(ValueError, match="forecast contract changed"):
        module.validate_plan(plan)


@pytest.mark.parametrize(
    "identity", [("ROUND_001", "COLLECTION"), ("COLLECTION", "R3_LONG15"), ("OTHER", "OTHER")]
)
def test_collection_admission_rejects_mixed_or_unknown_identity(identity):
    module = admission()
    plan = common.read(common.RESOURCES / "collection-plan.json")
    plan.update(round_id=identity[0], revision=identity[1], storage_root=str(module.STORAGE))
    with pytest.raises(ValueError, match="identity mismatch"):
        module.validate_plan(plan)


def test_generic_collector_protocol_keeps_legacy_read_identity_separate():
    module = common.collector()
    current = common.read(common.RESOURCES / "collection/plan.json")
    assert current["protocol"] == module.COLLECTION_PROTOCOL == "DRONE_COLLECTION"
    assert current["producer_args"] == ["--configuration", "collection"]
    assert current["data_pool_id"] == "collection/x500"
    assert current["profile_id"] == "vtc6_4s4p"
    with pytest.raises(module.Stop, match="separate data pool"):
        module.collection_prerequisites(dict(current, data_pool_id="ROUND_001/R3_LONG15"), True)
    with pytest.raises(module.Stop, match="explicit new producer configuration"):
        module.collection_prerequisites(
            dict(current, producer_args=["--configuration", "r3_long15"], assessment_command=["not-run"]),
            True
        )


@pytest.mark.parametrize("configuration", ["collection", "r3_long15", None, "unknown"])
def test_progress_reader_accepts_only_current_and_historical_configuration(tmp_path, configuration):
    from datetime import datetime, timezone

    module = common.collector()
    run_id = "a" * 32
    run = tmp_path / "runs" / run_id
    now = datetime.now(timezone.utc)
    common.write(
        run / "run.json",
        dict(
            run_id=run_id,
            instance=3,
            configuration=configuration,
            gz_partition="x500_" + run_id,
            clock=dict(id="px4-boot", epoch="simulation_boot", unit="us"),
            progress=dict(sim_elapsed_s=10, target_s=120, px4_boot_s=20, last_update_utc=now.isoformat()),
        ),
    )
    common.write(run / "evaluation/scenario.json", dict(name="hover_normal", seed=123))
    progress = module.MissionProgress(
        tmp_path, set(), dict(scenario="hover_normal", seed=123), dict(instance=3), 120
    ).sample(now_utc=now)
    expected = run_id if configuration in {"collection", "r3_long15"} else None
    assert progress["producer_run_id"] == expected


def metadata():
    record = common.read(common.ROOT / "ai/tests/fixtures/synthetic_metadata.json")
    record["vehicle"].update(
        nominal_capacity_ah=12,
        nominal_parallel_cells=4,
        battery_profile_id="vtc6_4s4p",
        profile_revision="vtc6-4s4p",
    )
    return record


def test_preprocessor_alias_validates_names_without_rewriting_original(tmp_path):
    common.package_imports()
    from capstone_ai.preprocessing import Preprocessor
    from pretrain.preprocessing_compat import make_preprocessor, verify_metadata_alias

    path = tmp_path / "metadata.json"
    original = metadata()
    common.write(path, original)
    before = path.read_bytes()
    digest = common.sha(path)
    config = common.read(common.PACKAGE / "preprocessing.json")
    with pytest.raises(ValueError, match="nameplate mismatch"):
        Preprocessor(original, config)
    engine, alias = make_preprocessor(original, config, digest)
    assert engine.metadata is original
    assert engine.metadata["vehicle"]["battery_profile_id"] == "vtc6_4s4p"
    assert engine.capacity_ah == 12
    assert path.read_bytes() == before and common.sha(path) == digest
    assert original == metadata()
    summary = dict(
        evidence=dict(metadata_validation_alias=alias),
        input_hashes={"observations/metadata.json": digest},
    )
    verify_metadata_alias(summary)
    alias["nominal_capacity_ah"] = 6
    with pytest.raises(ValueError, match="alias/input identity mismatch"):
        verify_metadata_alias(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [("nominal_capacity_ah", 6), ("nominal_parallel_cells", 2),
     ("nominal_series_cells", 6), ("rotor_count", 6),
     ("profile_id", "other_drone"), ("profile_revision", "unknown")],
)
def test_metadata_alias_never_normalizes_different_physical_nameplate(field, value):
    common.package_imports()
    from pretrain.preprocessing_compat import make_preprocessor

    record = metadata()
    record["vehicle"][field] = value
    with pytest.raises(ValueError, match="metadata/nameplate mismatch"):
        make_preprocessor(record, common.read(common.PACKAGE / "preprocessing.json"), "a" * 64)


def test_alias_preserves_original_binding_checks_and_historical_metadata():
    common.package_imports()
    from pretrain.preprocessing_compat import make_preprocessor

    record = metadata()
    config = common.read(common.PACKAGE / "preprocessing.json")
    record["semantic_bindings"]["pack_voltage_current"]["units"]["voltage_v"] = "mV"
    with pytest.raises(ValueError, match="voltage_v unit/field mismatch"):
        make_preprocessor(record, config, "a" * 64)
    record = metadata()
    record["vehicle"]["battery_profile_id"] = "vtc6_4s4p_r3_long15"
    record["vehicle"]["profile_revision"] = "vtc6-4s4p-r3-long15"
    engine, alias = make_preprocessor(record, config, "a" * 64)
    assert alias is None and engine.metadata is record
