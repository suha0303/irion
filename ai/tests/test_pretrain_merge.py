import copy
import shutil

import numpy as np
import pytest

from pretrain import assembly, common, composition, merge


@pytest.fixture
def parts(monkeypatch):
    def validate(rows, cases, protected, profile, *, case_sets):
        assert profile == "balanced300"
        assert len({r["run_id"] for r in rows}) == len(rows)
        assert [r["case_id"] for r in rows] == [r["case_id"] for r in cases]
        assert sum(case_sets, []) == [r["case_id"] for r in rows]

    monkeypatch.setattr(composition, "validate_roles", validate)
    result = []
    for i in range(2):
        row = dict(run_id=str(i), case_id=str(i), role="normal_train")
        lock = dict.fromkeys(merge.COMMON, "shared")
        lock["producer_lock_sha256"] = str(i)
        frozen = dict(status="DATA_FROZEN", revision="COLLECTION", revision_lock=lock,
                      collection_profile="balanced300", protected_inventory_sha256="fixed",
                      runs=[row], collection_set_case_ids=[[str(i)]])
        result.append(dict(frozen=frozen, cases=[dict(row)]))
    return result


def test_revision_provenance_and_source_splits_are_preserved(parts):
    before = copy.deepcopy(parts)
    frozen, cases = merge.merge_records(parts)
    assert parts == before
    assert len(frozen["runs"]) == len(cases) == 2
    assert frozen["collection_sets"] == 2
    assert [r["producer_lock_sha256"] for r in frozen["source_revision_locks"]] == ["0", "1"]
    assert "producer_lock_sha256" not in frozen["revision_lock"]
    assert not frozen["training_authorized"]


@pytest.mark.parametrize("key", merge.COMMON)
def test_incompatible_physics_or_preprocessing_stops(parts, key):
    parts[1]["frozen"]["revision_lock"][key] = "changed"
    with pytest.raises(ValueError, match="Incompatible"):
        merge.merge_records(parts)


def test_locked_final_test_never_enters_merge(parts):
    parts[1]["frozen"]["runs"][0]["role"] = "final_test"
    with pytest.raises(ValueError, match="Only training"):
        merge.merge_records(parts)


def test_restore_reports_verified_members_without_mutating_bundle(tmp_path, monkeypatch):
    from pretrain import storage

    monkeypatch.setattr(storage, "external", lambda p: p)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    original = bundle / "source.txt"
    original.write_text("preserve")
    ref = storage.pack_files({"first": original, "second": original},
                             bundle / "truth.tar.xz", common.Budget())
    ref["path"] = "truth.tar.xz"
    common.write(bundle / "receipt.json", dict(status="PORTABLE_DATASET_VERIFIED",
                                              evaluator_truth=ref))
    counts = []
    storage.restore(bundle, tmp_path / "restored", truth=True,
                    progress=lambda **p: counts.append(p))
    assert counts == [dict(done=1, total=2), dict(done=2, total=2)]
    assert original.read_text() == "preserve"
    assert common.sha(bundle / "truth.tar.xz") == ref["sha256"]


def test_merge_cli_does_not_render_unstarted_collection_counters(tmp_path):
    from pretrain import cli

    common.write(tmp_path / "config.json", dict(mode="merge_prepared", collection_run_count=1200,
                 audit_workers=4, storage_profile="compact", dataset_scope="all_selected_runs"))
    text = cli.render(tmp_path)
    assert "1200run 통합" in text and "0/7" in text
    assert "수집 완료" not in text and "수집 계획" not in text
    assert "임시 정리" in text


def test_merge_memory_policy_reserves_one_gigabyte(monkeypatch):
    from types import SimpleNamespace

    from pretrain import cli

    monkeypatch.setattr(cli, "collector", lambda: SimpleNamespace(
        memory_snapshot=lambda: dict(available_bytes=12_000_000_000)))
    monkeypatch.setattr(cli, "free_swap_bytes", lambda: 4_000_000_000)
    result = cli.service_memory(dict(mode="merge_prepared", use_available_ram=True,
                                    min_available_ram_gb=1, use_swap=True))
    assert result["reserve_bytes"] == 1_000_000_000
    assert result["max_bytes"] == 11_000_000_000
    assert result["swap_max_bytes"] == 4_000_000_000


def test_merge_runner_rejects_flight_calls(tmp_path):
    from pretrain.runner import Runner

    common.write(tmp_path / "config.json", dict(mode="merge_prepared", unit_attempts_max=3))
    with pytest.raises(ValueError):
        Runner(tmp_path).flight("collection", 1, [], "never")


def test_two_sources_refit_one_train_scaler_and_independently_verify(tmp_path):
    common.package_imports()
    fixture = common.load("merge_numeric_fixture", common.ROOT / "ai/tests/fixtures/pretrain_numeric.py")
    rows = []
    for i in range(2):
        source_rows = [dict(run_id=f"source{i}-{role}", role=role) for role in assembly.FIT]
        source = tmp_path / f"source{i}"
        fixture.make_dataset(source, source_rows)
        for row in source_rows:
            row["path"] = str(source / "canonical" / row["run_id"])
        rows.extend(source_rows)
    frozen = dict(runs=rows)

    def adopt(raw, target, config):
        shutil.copytree(raw, target)
        return common.read(target / "summary.json")

    target = tmp_path / "merged"
    assembly.prepare(frozen, target, common.read(common.PACKAGE / "preprocessing.json"),
                     common.Budget(), process_run=adopt)
    result = assembly.inspect_dataset(target, frozen, common.Budget(),
                                      inspect_run_fn=fixture.audit.inspect_run)
    assert result["by_role"]["normal_train"]["runs"] == 2
    assert result["train_only_scaler_verified"]
    expected = common.read(tmp_path / "source0/manifest.json")["unique_train_frames"]
    assert common.read(target / "manifest.json")["unique_train_frames"] == 2 * expected
    with np.load(target / "normal_train.npz", allow_pickle=False) as arrays:
        assert set(arrays["run_id"]) == {r["run_id"] for r in rows if r["role"] == "normal_train"}
    assert (tmp_path / "source0/scaler.npz").is_file()
    assert (tmp_path / "source1/scaler.npz").is_file()
