"""Source distribution excludes machine state and retains executable data contracts."""

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from tools import workspace


def test_source_distribution_has_complete_sealed_package_and_no_machine_state():
    files = workspace.source_files()
    prefix = "Data/generation/pretrain/resources/preprocessing/"
    package = json.loads((workspace.ROOT / prefix / "PACKAGE.json").read_text())
    assert all(prefix + name in files for name in package["files"])
    assert all(workspace.sha(files[prefix + name]) == digest for name, digest in package["files"].items())
    assert "Data/generation/drone-data" in files
    assert "Data/generation/drone-pretrain" in files
    assert "Data/generation/x500-data" not in files
    assert "Data/generation/x500-pretrain" not in files
    assert "simulation/config/producer-sources.json" in files
    assert "simulation/sim/tools/bootstrap.sh" in files
    assert "simulation/sim/tests/fixtures/metadata/manifest.json" in files
    for name in files:
        assert not set(Path(name).parts) & {".venv", "__pycache__", "experiments", "archive"}
        assert not name.startswith(("Data/processed/", "Data/operations/", "simulation/runtime/"))
        assert not name.endswith((".pt", ".pth", ".pem", ".key"))


def test_source_export_is_verified_and_launcher_is_relocatable(tmp_path, monkeypatch):
    archive = tmp_path / "source.tar.gz"
    monkeypatch.setattr(workspace, "ROOT", Path(workspace.ROOT))
    assert workspace.export(argparse.Namespace(output=str(archive))) == 0
    with tarfile.open(archive) as stream:
        stream.extractall(tmp_path / "clone with spaces", filter="data")
    clone = tmp_path / "clone with spaces/Capstone"
    record = json.loads((clone / "SOURCES.json").read_text())
    assert all(workspace.sha(clone / name) == digest for name, digest in record["files"].items())
    assert not record["includes_user_data"] and not record["includes_host_runtime"]
    command = clone / "Data/generation/drone-data"
    result = subprocess.run([str(command), "--help"], cwd="/tmp", text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert all(name in result.stdout for name in ("setup", "doctor", "export"))
    # A clean clone fails with an actionable setup message, never an old machine path.
    result = subprocess.run([str(clone / "Data/generation/drone-pretrain"), "--help"],
                            cwd="/tmp", text=True, capture_output=True)
    assert result.returncode == 3
    assert "setup" in result.stderr
    assert "/home/sa" not in result.stderr
    with pytest.raises(ValueError, match="NEW path"):
        workspace.export(argparse.Namespace(output=str(archive)))


def test_source_export_rejects_source_symlinks_and_excludes_credentials(tmp_path):
    root = tmp_path / "checkout"
    (root / "Data/generation").mkdir(parents=True)
    (root / "Data/generation/.env").write_text("private")
    (root / "Data/generation/private.key").write_text("private")
    (root / "Data/generation/id_ed25519").write_text("private")
    (root / "Data/generation/source.py").write_text("pass\n")
    files = workspace.source_files(root)
    assert set(files) == {"Data/generation/source.py"}
    (root / "Data/generation/link.py").symlink_to(root / "Data/generation/source.py")
    with pytest.raises(ValueError, match="Symlink"):
        workspace.source_files(root)


def test_source_export_excludes_generated_python_packaging_metadata(tmp_path):
    root = tmp_path / "checkout"
    source = root / "ai/src/capstone_ai"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("\"\"\"Actual package source.\"\"\"\n")
    for suffix in (".egg-info", ".dist-info"):
        metadata = source.parent / ("capstone_ai" + suffix)
        metadata.mkdir()
        (metadata / "PKG-INFO").write_text("Generated on the original host\n")
        (metadata / "SOURCES.txt").write_text("generated metadata\n")
    assert set(workspace.source_files(root)) == {"ai/src/capstone_ai/__init__.py"}


def test_export_excludes_ai_profiles_jobs_and_weights_but_keeps_source_fixtures(tmp_path):
    root = tmp_path / "checkout"
    for directory in ("ai/.local", "ai/operations", "ai/outputs", "ai/runs"):
        path = root / directory
        path.mkdir(parents=True)
        (path / "private.json").write_text("machine state")
    package = root / "ai/console"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "model.safetensors").write_bytes(b"weights")
    fixture = root / "simulation/sim/tests/runs"
    fixture.mkdir(parents=True)
    (fixture / "case.json").write_text("{}")
    assert set(workspace.source_files(root)) == {
        "ai/console/__init__.py", "simulation/sim/tests/runs/case.json"
    }


def test_data_launcher_shell_syntax():
    shell = shutil.which("bash")
    assert shell
    for name in ("drone-data", "drone-pretrain"):
        result = subprocess.run([shell, "-n", str(workspace.ROOT / "Data/generation" / name)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_current_collector_loads_exact_preserved_process_helper_without_execution():
    from pretrain.common import ROOT, collector, sha

    runner = collector()
    support = ROOT / "Data/generation/tools/collection_support.py"
    assert sha(support) == runner.HELPER_SHA
    loaded = runner.helper(ROOT)
    assert Path(loaded.__file__) == support
    assert callable(loaded.command) and callable(loaded.export_run)


def test_setup_refuses_an_active_pipeline_before_installing(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setattr(workspace, "ROOT", tmp_path)
    path = tmp_path / "Data/operations/pretrain/pretrain.lock"
    path.parent.mkdir(parents=True)
    with path.open("a") as active:
        fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="pipeline is active"):
            with workspace.setup_lock():
                pytest.fail("Setup must not install during a pipeline")


def test_completed_cleanup_count_uses_proof_without_rewriting_history(tmp_path):
    from pretrain import cli
    from pretrain.common import read, write

    path = tmp_path / "operator/state.json"
    state = dict(status="PRETRAIN_COMPACT_READY_STOPPED_RESTORE_REQUIRED", total=1, done=0,
                 stage="merge-retire", completed=["merge-retire"],
                 stages=[dict(name="merge-retire", passed=True)])
    write(path, state)
    assert cli.snapshot(tmp_path)["done"] == 1
    assert read(path)["done"] == 0
    state["status"] = "FAILED_STOPPED"
    write(path, state)
    assert cli.snapshot(tmp_path)["done"] == 0


def test_export_failure_preserves_previous_failure_evidence(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("original\n")
    output = tmp_path / "export.tar.gz"
    previous = tmp_path / "export.tar.gz.incomplete"
    previous.write_bytes(b"previous failure evidence")
    monkeypatch.setattr(workspace, "source_files", lambda: {"source.py": source})
    original = workspace.sha
    calls = 0

    def changed(path):
        nonlocal calls
        calls += 1
        return original(path) if calls == 1 else "changed"

    monkeypatch.setattr(workspace, "sha", changed)
    with pytest.raises(ValueError, match="Source changed"):
        workspace.export(argparse.Namespace(output=str(output)))
    assert previous.read_bytes() == b"previous failure evidence"
    assert not output.exists()
    assert len(list(tmp_path.glob("export.tar.gz.failed-*/export.tar.gz.incomplete"))) == 1


def test_service_start_does_not_expand_dollar_in_project_path(tmp_path, monkeypatch):
    from pretrain import cli
    from pretrain.common import read, sha, write

    home = tmp_path / "checkout ${HOME}" / "control"
    task = home / "tasks/dollar-path"
    write(task / "config.json", dict(source_sha256={}, external_storage=str(tmp_path / "store")))
    (task / "config.sha256").write_text(sha(task / "config.json"))
    calls = []
    monkeypatch.setattr(cli, "HOME", home)
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: calls.append(command))
    cli.start(task)
    assert len(calls) == 1 and "--expand-environment=no" in calls[0]
    assert calls[0][-1] == str(task)
    assert read(task / "operator/service.json")["restart"] == "no"


@pytest.mark.parametrize("legacy", [False, True])
def test_service_identity_retains_exact_current_or_historical_unit(tmp_path, legacy):
    from pretrain.common import service_unit, write

    task = tmp_path / "tasks/fixture"
    expected = ("x500" if legacy else "drone") + "-pretrain-fixture.service"
    config = {} if legacy else dict(service_unit=expected)
    write(task / "config.json", config)
    assert service_unit(task) == expected
    write(task / "operator/service.json", dict(unit=expected, restart="no"))
    assert service_unit(task) == expected
    write(task / "operator/service.json", dict(unit="drone-pretrain-other.service"))
    with pytest.raises(ValueError, match="service identity mismatch"):
        service_unit(task)


def test_service_identity_uses_frozen_config_when_submission_exits_before_receipt(tmp_path):
    from pretrain.common import service_unit, write

    task = tmp_path / "tasks/new-task"
    config = dict(service_unit="drone-pretrain-new-task.service")
    write(task / "config.json", config)
    assert service_unit(task) == config["service_unit"]
    write(task / "config.json", dict(service_unit="unrelated.service"))
    with pytest.raises(ValueError, match="service identity mismatch"):
        service_unit(task)


def test_export_cannot_bypass_checkout_boundary_with_traversal_or_symlink(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(workspace, "ROOT", checkout)
    (tmp_path / "holder").mkdir()
    with pytest.raises(ValueError, match="NEW path"):
        workspace.export(argparse.Namespace(output=str(tmp_path / "holder/../checkout/inside.tar.gz")))
    alias = tmp_path / "alias"
    alias.symlink_to(checkout, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        workspace.export(argparse.Namespace(output=str(alias / "inside.tar.gz")))
