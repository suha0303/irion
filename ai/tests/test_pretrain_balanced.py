"""Finite300 design, disk numeric equivalence, bounded display and swap policy."""

import copy
import shutil
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from pretrain import assembly, balanced, cli, collection, common, composition, display


def test_balanced_300_counts_coverage_and_independent_multipliers():
    rows = collection.cases(202680000, "balanced300")["cases"]
    assert rows == balanced.cases(202680000)
    assert Counter(c["role"] for c in rows) == balanced.ROLES
    assert Counter(c["path_mode"] for c in rows if "path_mode" in c) == {
        "basic": 103,
        "mixed": 77,
        "random": 78,
    }
    train = [c for c in rows if c["role"] == "normal_train"]
    assert set(Counter(c["scenario"] for c in train).values()) == {20}
    assert all(
        {c["path_mode"] for c in train if c["scenario"] == s} == {"basic", "mixed", "random"}
        for s in {c["scenario"] for c in train}
    )
    subsets = composition.validation_subsets(rows, "balanced300")
    assert len(subsets) == 20 and all(len(s) == 50 for s in subsets)
    other = balanced.cases(202680000 + composition.seed_stride("balanced300"))
    assert not {c["seed"] for c in rows} & {c["seed"] for c in other}
    damaged = copy.deepcopy(rows)
    next(c for c in damaged if "normal_variation" in c)["normal_variation"]["payload"] += 0.00001
    with pytest.raises(ValueError):
        composition.validation_subsets(damaged, "balanced300")


def test_mixed_programs_retain_their_existing_motion_bounds():
    import sys

    sys.path.insert(0, str(common.ROOT / "simulation/sim/src"))
    from capstone_sim.normal_variation import target

    # Mixed paths keep their original contract; roaming random paths have separate tests.
    maxima = np.zeros(3)
    for case in balanced.cases(202680000):
        if case.get("path_mode") != "mixed":
            continue
        spec = case["normal_variation"]
        times = np.arange(0, case["mission_seconds"] + 0.01, 0.1)
        xyz = np.asarray([target(float(t), spec) for t in times])
        assert np.array_equal(xyz[0], [0, 0, -3, 0])
        assert -4.600001 <= xyz[:, 2].min() <= xyz[:, 2].max() <= -2.399999
        for n in (1, 2, 3):
            maxima[n - 1] = max(
                maxima[n - 1],
                np.linalg.norm(np.diff(xyz[:, :3], n=n, axis=0) / 0.1**n, axis=1).max(),
            )
    assert maxima[0] < 0.65 and maxima[1] < 0.2 and maxima[2] < 0.1, maxima


@pytest.mark.parametrize("tamper", [None, "array", "scaler", "membership"])
def test_disk_assembly_matches_legacy_and_independent_audit(tmp_path, tamper):
    common.package_imports()
    fixture = common.load(
        "balanced_numeric_fixture", common.ROOT / "ai/tests/fixtures/pretrain_numeric.py"
    )
    fixture.make_dataset(tmp_path / "reference")
    frozen = dict(runs=common.read(tmp_path / "reference/manifest.json")["runs"])
    for row in frozen["runs"]:
        row["path"] = str(tmp_path / "reference/canonical" / row["run_id"])
    legacy = common.load(
        "pipeline.balanced_legacy_prepare", common.ROOT / "ai/pipeline/pipeline.py"
    )
    config = common.read(common.PACKAGE / "preprocessing.json")

    def adopt(raw, target, config):
        shutil.copytree(raw, target)
        return common.read(target / "summary.json")

    legacy.prepare(frozen, tmp_path / "legacy", config, common.Budget(), process_run=adopt)
    assembly.prepare(frozen, tmp_path / "disk", config, common.Budget(), process_run=adopt)
    for filename in [r + ".npz" for r in assembly.FIT] + ["scaler.npz"]:
        with (
            np.load(tmp_path / "legacy" / filename, allow_pickle=False) as a,
            np.load(tmp_path / "disk" / filename, allow_pickle=False) as b,
        ):
            assert a.files == b.files
            for key in a.files:
                if a[key].dtype.kind in "US":
                    assert np.array_equal(a[key], b[key])
                else:
                    np.testing.assert_allclose(a[key], b[key], atol=1e-10, rtol=1e-10)
    dataset = tmp_path / "disk"
    if tamper in {"array", "scaler"}:
        path = dataset / ("normal_train.npz" if tamper == "array" else "scaler.npz")
        with np.load(path, allow_pickle=False) as data:
            values = dict(data)
        values["X" if tamper == "array" else "x_mean"].flat[0] += 0.1
        np.savez(path, **values)
        manifest = common.read(dataset / "manifest.json")
        manifest["hashes"][path.name] = common.sha(path)
        common.write(dataset / "manifest.json", manifest)
    elif tamper == "membership":
        frozen["runs"] = frozen["runs"][:-1]
    if tamper:
        with pytest.raises(ValueError):
            assembly.inspect_dataset(
                dataset, frozen, common.Budget(), inspect_run_fn=fixture.audit.inspect_run
            )
    else:
        result = assembly.inspect_dataset(
            dataset, frozen, common.Budget(), inspect_run_fn=fixture.audit.inspect_run
        )
        expected = fixture.audit.inspect_dataset(tmp_path / "legacy", frozen)
        assert result["dimensions"] == expected["dimensions"]
        assert result["by_role"] == expected["by_role"]
        assert result["train_only_scaler_verified"]
        assert not (tmp_path / "assembly-work").exists()
        assert not (tmp_path / "quality-work").exists()


def test_grid_shows_all16_in_eight_rows_and_keeps_header():
    lines = display.grid([f"{i:02} 비행 02:00/15:00 1/3" for i in range(16)])
    assert len(lines) == 8
    assert "00" in lines[0] and "08" in lines[0]
    assert "07" in lines[-1] and "15" in lines[-1]
    text = "\n".join(
        ["X500 / RUNNING", "설정 비행16 / 나머지4"]
        + ["자원 계획 설명"] * 20
        + lines
        + ["로그 /tmp/log", "화면만 갱신"]
    )
    fitted = display.screen(text, 80, 24)
    assert len(fitted.splitlines()) <= 24 and fitted.startswith("X500")
    assert all(f"{i:02}" in fitted for i in range(16))
    assert all(
        sum(2 if __import__("unicodedata").east_asian_width(c) in "WF" else 1 for c in line) < 80
        for line in fitted.splitlines()
    )


def test_new_defaults_and_swap_keeps_ram_reserve(monkeypatch):
    args = cli.parser().parse_args(["init", "--name", "test", "--mode", "collect"])
    assert args.collection_profile == "balanced300" and args.flight_workers == 16
    assert args.workers == args.io_workers == args.audit_workers == args.export_workers == 4
    assert args.use_swap
    assert args.seed_base is None
    monkeypatch.setattr(
        cli,
        "collector",
        lambda: SimpleNamespace(memory_snapshot=lambda: {"available_bytes": 11_000_000_000}),
    )
    monkeypatch.setattr(cli, "free_swap_bytes", lambda: 8_600_000_000)
    config = dict(mode="collect", use_available_ram=True, min_available_ram_gb=1, use_swap=True)
    policy = cli.service_memory(config)
    assert policy["max_bytes"] == 10_000_000_000 and policy["swap_max_bytes"] == 8_600_000_000
    config.pop("use_swap")
    assert cli.service_memory(config)["swap_max_bytes"] == 0


def test_seed_prompt_blank_or_explicit_and_zero(monkeypatch):
    for value, expected in (("", None), ("17", 17), ("0", 0)):
        monkeypatch.setattr("builtins.input", lambda prompt: value)
        assert cli.input_seed() == expected
    args = cli.parser().parse_args(["init", "--name", "test", "--seed-base", "17"])
    assert args.seed_base == 17
