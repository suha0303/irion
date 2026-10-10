"""Synthetic subprocesses only:16-lane admission, independent archives, optional gates."""

from pathlib import Path

import pytest
from test_collect_long_runs import (  # noqa: F401 - pytest fixture dependencies
    binding_files,
    c,
    collection_setup,
    dump,
    setup,
)

from pretrain.collection import lanes


def test_expanded_memory_only_for_bound_benchmark(collection_setup):  # noqa: F811
    plan, _, _, tmp = collection_setup
    config_path, memory_path = tmp / "config.json", tmp / "memory.json"
    dump(config_path, dict(mode="test", use_available_ram=True, min_available_ram_gb=1))
    dump(
        memory_path,
        dict(available_bytes=14_000_000_000, max_bytes=13_000_000_000, reserve_bytes=1_000_000_000),
    )
    plan.update(
        phase="benchmark",
        gate_policy="optional_pretrain_cli_v1",
        pretrain_cli_approval=str(config_path),
        test_memory_override={
            "config": dict(path=str(config_path), sha256=c.sha(config_path)),
            "memory": dict(path=str(memory_path), sha256=c.sha(memory_path)),
        },
    )
    plan["budgets"].update(
        min_available_ram_bytes=1_000_000_000, memory_soft_cap_bytes=13_000_000_000
    )
    c.collection_prerequisites(plan, dry_run=True)
    plan["phase"] = "collection"
    with pytest.raises(c.Stop, match="Invalid finite-task memory override"):
        c.collection_prerequisites(plan, dry_run=False)
    plan["phase"] = "benchmark"
    plan["budgets"]["min_available_ram_bytes"] = 999_999_999
    with pytest.raises(c.Stop, match="Invalid finite-task memory override"):
        c.collection_prerequisites(plan, dry_run=True)


@pytest.mark.parametrize("phase", ["collection", "benchmark", "endurance"])
def test_expanded_memory_collect_and_optional_gates_bound(collection_setup, phase):  # noqa: F811
    plan, _, _, tmp = collection_setup
    config_path, memory_path = tmp / "config.json", tmp / "memory.json"
    dump(
        config_path,
        dict(mode="collect", gates=True, use_available_ram=True, min_available_ram_gb=1),
    )
    dump(
        memory_path,
        dict(available_bytes=14_000_000_000, max_bytes=13_000_000_000, reserve_bytes=1_000_000_000),
    )
    plan.update(
        phase=phase,
        gate_policy="optional_pretrain_cli_v1",
        pretrain_cli_approval=str(config_path),
        pretrain_memory_override={
            "config": dict(path=str(config_path), sha256=c.sha(config_path)),
            "memory": dict(path=str(memory_path), sha256=c.sha(memory_path)),
        },
    )
    plan["budgets"].update(
        min_available_ram_bytes=1_000_000_000, memory_soft_cap_bytes=13_000_000_000
    )
    c.collection_prerequisites(plan, dry_run=phase != "collection")
    sample = dict(owned_tree_rss_bytes=10_000_000_000, cgroup_memory_current_bytes=12_000_000_000)
    assert c.budget_memory_used(plan, sample) == 10_000_000_000
    # An unrelated modified approval cannot silently inherit the expanded policy.
    dump(config_path, dict(mode="prepare", use_available_ram=True, min_available_ram_gb=1))
    with pytest.raises(c.Stop, match="Evidence hash mismatch"):
        c.collection_prerequisites(plan, dry_run=phase != "collection")


@pytest.mark.parametrize("export_workers", [1, 16])
@pytest.mark.parametrize("expanded", [False, True])
def test_optional_gate_sixteen_collection_and_exports(collection_setup, export_workers, expanded):  # noqa: F811
    plan, _, helper, tmp = collection_setup
    plan.update(
        concurrency=16,
        attempts_max=16,
        lanes=lanes(16),
        export_workers=export_workers,
        gate_policy="optional_pretrain_cli_v1",
        pretrain_cli_approval="explicit-fixture-only",
    )
    plan.pop("gate_receipt")
    if expanded:
        config_path, memory_path = tmp / "config.json", tmp / "memory.json"
        dump(config_path, dict(mode="collect", use_available_ram=True, min_available_ram_gb=1))
        dump(
            memory_path,
            dict(
                available_bytes=11_000_000_000,
                max_bytes=10_000_000_000,
                reserve_bytes=1_000_000_000,
            ),
        )
        plan.update(
            pretrain_cli_approval=str(config_path),
            pretrain_memory_override={
                "config": dict(path=str(config_path), sha256=c.sha(config_path)),
                "memory": dict(path=str(memory_path), sha256=c.sha(memory_path)),
            },
        )
        plan["budgets"].update(
            min_available_ram_bytes=1_000_000_000, memory_soft_cap_bytes=10_000_000_000
        )
    rows = [
        dict(
            case_id=f"synthetic16-{i}",
            scenario="hover_normal",
            seed=8000 + i,
            role="normal_train",
            group_id=f"synthetic16-{i}",
            mission_seconds=900,
        )
        for i in range(16)
    ]
    path = Path(plan["isolation_evidence"]["path"])
    proof = c.read(path)
    proof["lanes"] = plan["lanes"]
    dump(path, proof)
    plan["isolation_evidence"]["sha256"] = c.sha(path)
    accepted, cases = c.bound(*binding_files(plan, rows, tmp))
    result = c.execute(accepted, cases, helper)
    assert result["status"] == "COLLECTED_PENDING_QUALITY_STOPPED"
    records = list(result["cases"].values())
    assert len(records) == len({r["run_id"] for r in records}) == 16
    assert len({r["lane"]["instance"] for r in records}) == 16
    for row in records:
        assert row["verification_started_monotonic"] <= row["verification_finished_monotonic"]
        assert row["archive_started_monotonic"] <= row["archive_finished_monotonic"]
    plan.pop("pretrain_cli_approval")
    with pytest.raises(c.Stop, match="approval required|Invalid finite-task memory override"):
        c.bound(*binding_files(plan, rows, tmp))


def test_fresh_benchmark_freeze_reads_only_new_archives(tmp_path, monkeypatch):
    import hashlib
    import io
    import json
    import tarfile

    from pretrain import collection, common

    common.package_imports()
    from preprocessing_tools.control import RAW_FILES

    store = tmp_path / "store"
    monkeypatch.setattr(common, "STORE", store)
    monkeypatch.setattr(collection, "STORE", store)
    folder = tmp_path / "new-test"
    cases, records = [], {}
    for index in range(3):
        run_id = f"{index + 1:032x}"
        case = dict(
            case_id=f"probe-{index}",
            scenario="hover_normal",
            seed=index,
            role="benchmark",
            mission_seconds=120,
            group_id=f"g{index}",
        )
        cases.append(case)
        archive = store / "archives" / f"{run_id}.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        values = {name: {} for name in RAW_FILES}
        values.update(
            {
                "run.json": dict(run_id=run_id, origin="px4_sitl", status="completed"),
                "evaluation/scenario.json": dict(
                    name="hover_normal",
                    seed=index,
                    requested_duration_s=120,
                    continuity="independent_reset",
                    health_eligibility="confirmed_healthy",
                ),
                "reference/source_lock.json": dict(local_sources={}),
            }
        )
        members = {}
        with tarfile.open(archive, "w:gz") as tar:
            for name, value in values.items():
                body = json.dumps(value).encode()
                info = tarfile.TarInfo(name)
                info.size = len(body)
                tar.addfile(info, io.BytesIO(body))
                members[name] = dict(
                    kind="file", bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
                )
        records[case["case_id"]] = dict(
            case=case,
            run_id=run_id,
            status="ARCHIVED_PENDING_QUALITY",
            assessment=dict(
                run_id=run_id,
                contract_passed=True,
                instance_isolation_passed=True,
                missing_data_passed=True,
                physical_gate_passed=True,
            ),
            export=dict(
                archive=str(archive),
                sha256=common.sha(archive),
                bytes=archive.stat().st_size,
                members=members,
            ),
        )
    common.write(folder / "manifest.json", dict(cases=cases))
    ledger = dict(
        phase="benchmark",
        status="COLLECTED_PENDING_QUALITY_STOPPED",
        not_started=[],
        stop_reason=None,
        cases=records,
    )
    common.write(folder / "execution/ledger.json", ledger)
    result = collection.freeze_benchmark(folder)
    assert result["training_eligible"] is False
    assert {r["role"] for r in result["runs"]} == {
        "normal_train",
        "normal_validation",
        "normal_calibration",
    }
    assert all(
        r["collection_role"] == "benchmark" and not r["training_eligible"] for r in result["runs"]
    )
    # Identity corruption cannot be waved through as a fresh valid benchmark.
    records["probe-0"]["run_id"] = "bad"
    common.write(folder / "execution/ledger.json", ledger)
    with pytest.raises(ValueError, match="assessment"):
        collection.freeze_benchmark(folder)
