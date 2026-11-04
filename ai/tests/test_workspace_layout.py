"""Category layout and compatibility preserve runtime guards; synthetic collection only."""

from pathlib import Path

from test_collect_long_runs import c, setup  # noqa: F401

from pretrain.common import PACKAGE, ROOT, read, sha


def test_four_real_categories_and_no_category_aliases():
    actual = {
        p.name
        for p in ROOT.iterdir()
        if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")
        # Published UI/student records and export copies are separate from runtime categories.
        and p.name not in {"commit-bundles", "UI", "학적"}
    }
    assert actual == {"simulation", "Data", "ai", "docs"}
    assert all(not (ROOT / name).is_symlink() for name in ("sim", "profiles", "contracts", "vendor"))
    assert not any(p.is_symlink() for p in ROOT.iterdir())


def test_installed_runtime_and_sealed_package():
    from capstone_sim import collection_config

    collection_config.assert_frozen()
    current = read(collection_config.producer_lock_path())
    assert current["revision"] == "X500_CURRENT"
    assert collection_config.producer_lock_path().name == "runtime-manifest.json"
    for name, digest in read(PACKAGE / "PACKAGE.json")["files"].items():
        assert sha(PACKAGE / name) == digest
    assert not any(p.is_symlink() for p in PACKAGE.parents)


def test_collection_with_physical_simulator_root_without_aliases(setup):  # noqa: F811
    plan, cases, helper, _ = setup
    root = Path(plan["root"])
    runs = root / "runs"
    plan.update(project_root=str(root.parent), runs_root=str(runs))
    result = c.execute(plan, cases[:1], helper)
    assert result["status"] == "COLLECTED_PENDING_QUALITY_STOPPED"
    assert len(list(runs.iterdir())) == 1
    assert result["cases"]["c1"]["status"] == "ARCHIVED_PENDING_QUALITY"
    assert Path(result["preserved_new_local_run_paths"][0]).parent == runs


def test_headless_cli_plans_collection_without_legacy_import_paths(tmp_path):
    import os
    import subprocess
    import sys

    code = r"""
from pathlib import Path
import shutil,sys
from pretrain import cli,collection
from pretrain.common import HOME,ROOT,read,write,verify,sha
home=Path(sys.argv[1])/'control'
home.mkdir()
cli.HOME=home
args=cli.parser().parse_args(['init','--name','isolated-paths','--mode','collect'])
task=cli.create(args)
verify(read(task/'config.json')['source_sha256'])
write(task/'operator/memory-policy.json',dict(available_bytes=16_000_000_000,max_bytes=15_000_000_000,reserve_bytes=1_000_000_000))
rows=collection.cases(1_900_000,'balanced300')['cases']
folder=collection.plan(task,'collection',16,rows,'collection',export_workers=4)
plan=read(folder/'plan.json')
assert Path(plan['root'])==ROOT/'simulation'
assert Path(plan['project_root'])==ROOT
assert Path(plan['runs_root'])==ROOT/'simulation/runs'
assert Path(plan['assessment_command'][0]).is_file()
collector=collection.collector()
collector.bound(folder/'plan.json',folder/'manifest.json',folder/'approval.json')
for unsafe in ('../escape','/absolute'):
    plan['project_source_hashes']={unsafe:'a'*64}
    isolation=read(folder/'isolation.json')
    isolation['project_source_hashes']=plan['project_source_hashes']
    write(folder/'isolation.json',isolation)
    plan['isolation_evidence']['sha256']=sha(folder/'isolation.json')
    write(folder/'plan.json',plan)
    approval=read(folder/'approval.json')
    approval['plan_sha256']=sha(folder/'plan.json')
    write(folder/'approval.json',approval)
    try:
        collector.bound(folder/'plan.json',folder/'manifest.json',folder/'approval.json')
    except collector.Stop as exc:
        assert 'Unsafe source lock' in str(exc),str(exc)
    else:
        raise AssertionError('Unsafe project source accepted')
assert not (task/'operator/claimed').exists()
print('PHYSICAL_PATHS_PLAN_BOUND_NO_EXECUTION')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "Data/generation") + ":" + str(ROOT / "ai")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PHYSICAL_PATHS_PLAN_BOUND_NO_EXECUTION" in result.stdout
