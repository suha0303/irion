"""Preparation storage safety and packaged imports; no real data restore/GPU job."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from execution.common import ROOT
from learning import prepare_compare


def test_unmounted_external_root_is_rejected_without_creating_output(monkeypatch):
    root = Path("/media/sa/X31/Data/X500/ROUND_001/R3_LONG15/gpu1200/uncreated-test")
    monkeypatch.setattr(Path, "is_mount", lambda p: False)
    with pytest.raises(ValueError, match="Mounted external"):
        prepare_compare.external_root(root)


def test_mutating_actions_reject_missing_mount_before_read_or_ssh(tmp_path, monkeypatch):
    def unavailable(path):
        raise ValueError("Mounted external unavailable")

    monkeypatch.setattr(prepare_compare, "external_root", unavailable)
    monkeypatch.setattr(prepare_compare.subprocess, "run", lambda *a, **k: pytest.fail("No SSH"))
    from console.profile import write

    profile = tmp_path / "server-profile.json"
    write(
        profile,
        dict(
            schema="drone-ai.profile/1",
            name="fixture",
            project_root=str(tmp_path),
            python=sys.executable,
            runtime_profile="torch21",
            dataset_root=str(tmp_path / "dataset"),
            output_root=str(tmp_path / "output"),
            ssh=None,
            slurm=dict(
                partition="p02", node="DIS01", nodes=1, ntasks=1, gpus=1, cpus=4, memory_gib=16
            ),
        ),
    )
    monkeypatch.setenv("DRONE_AI_PROFILE", str(profile))
    for action in (
        lambda: prepare_compare.authorize(tmp_path, True),
        lambda: prepare_compare.deploy(tmp_path),
        lambda: prepare_compare.launchers(tmp_path, Path("/srv/drone-ai/fixture")),
    ):
        with pytest.raises(ValueError, match="Mounted external"):
            action()
    assert list(tmp_path.iterdir()) == [profile]


def test_mount_loss_stops_checks_and_prevents_status_write_fallback(tmp_path, monkeypatch):
    mounted = True

    def external(path):
        if not mounted:
            raise ValueError("Mounted external lost")
        return path

    monkeypatch.setattr(prepare_compare, "external_root", external)
    state = tmp_path / "progress.json"
    progress = prepare_compare.PreparationProgress(state, storage_root=tmp_path, hours=1)
    progress.update(stage="fixture")
    before = state.read_bytes()
    mounted = False
    with pytest.raises(ValueError, match="Mounted external lost"):
        progress.check()
    with pytest.raises(ValueError, match="Mounted external lost"):
        progress.update(status="FAILED_STOPPED")
    assert state.read_bytes() == before


def test_external_root_rejects_traversal():
    with pytest.raises(ValueError, match="traversal"):
        prepare_compare.external_root(
            Path("/media/sa/X31/Data/X500/ROUND_001/R3_LONG15/../../escape")
        )


def test_deployed_snapshot_imports_without_pretrain_or_retired_r1_launchers(tmp_path):
    stage = tmp_path / "input/ai"
    for folder in ("execution", "learning", "pipeline", "console", "src/capstone_ai"):
        for source in (ROOT / "ai" / folder).rglob("*.py"):
            target = stage / source.relative_to(ROOT / "ai")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    assert not (stage / "pretrain").exists()
    assert not (stage / "pipeline/build_package.py").exists()
    code = """
import importlib,sys,pathlib
stage=pathlib.Path(sys.argv[1])
original=pathlib.Path(sys.argv[2])
excluded={str(original),str(original/'ai'),str(original/'ai/src'),str(original/'sim/src')}
sys.path=[str(stage),str(stage/'src')]+[p for p in sys.path if p not in excluded]
names=['learning.prepare_compare','learning.lifecycle','learning.operator',
       'execution.engine','execution.evaluate','pipeline.pipeline','pipeline.runtime','console.profile','capstone_ai.training']
for name in names:
    module=importlib.import_module(name)
    assert pathlib.Path(module.__file__).is_relative_to(stage),module.__file__
import torch
assert not torch.cuda.is_initialized()
print('ISOLATED_SNAPSHOT_IMPORTS_PASSED_NO_CUDA')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(stage), str(ROOT)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ISOLATED_SNAPSHOT_IMPORTS_PASSED_NO_CUDA" in result.stdout
