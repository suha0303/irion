"""Selected local/custom storage reaches frozen plans and real CPU extraction workers."""

import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import x500_storage as paths
from pretrain import cli, collection, common


def test_default_is_project_data_and_custom_path_is_selectable():
    args = cli.parser().parse_args(["init", "--name", "fixture"])
    assert args.storage_root == common.ROOT / "Data"
    custom = cli.parser().parse_args(
        ["init", "--name", "fixture", "--storage-root", "/tmp/custom data"]
    )
    assert custom.storage_root == Path("/tmp/custom data")


def test_selected_storage_stops_on_replaced_device_and_rejects_links(tmp_path):
    root = tmp_path / "not-created/Data"
    policy = paths.capture(root)
    assert paths.guard(root / "processed", policy) == root / "processed"
    assert not root.exists()
    with pytest.raises(ValueError, match="filesystem unavailable/changed"):
        paths.guard(root, dict(policy, device=policy["device"] + 1))
    with pytest.raises(ValueError, match="outside selected"):
        paths.guard(tmp_path / "elsewhere", policy)
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        paths.capture(tmp_path / "link/Data")
    with pytest.raises(ValueError, match="traversal"):
        paths.capture(root / ".." / "escape")


def test_task_storage_is_frozen_into_plan_and_independent_launch(tmp_path, monkeypatch):
    root = tmp_path / "custom storage"
    home = tmp_path / "home"
    task = home / "tasks/fixture"
    config = dict(
        external_storage=str(root),
        storage_policy=paths.capture(root),
        pipeline_overlap=False,
        source_sha256={},
    )
    common.write(task / "config.json", config)
    (task / "config.sha256").write_text(common.sha(task / "config.json"))
    cases = collection.cases(8_000_000, "balanced300")["cases"]
    folder = collection.plan(task, "collection", 16, cases, "local", export_workers=4)
    plan = common.read(folder / "plan.json")
    assert Path(plan["external_output"]).is_relative_to(root / "archives")
    assert plan["storage_policy"] == config["storage_policy"]
    assert "ai/x500_storage.py" in plan["project_source_hashes"]
    from pretrain.common import collector

    c = collector()
    c.validate_storage_selection(plan)
    with pytest.raises(c.Stop, match="Selected storage"):
        c.validate_storage_selection(
            dict(
                plan, storage_policy=dict(plan["storage_policy"], root=str(tmp_path / "elsewhere"))
            )
        )
    # The actual collector space guard must not call its legacy X31-only checker.
    c.check_output_storage(
        plan,
        Path(plan["external_output"]),
        SimpleNamespace(check_mount=lambda p: pytest.fail("Legacy mount checker")),
    )
    monkeypatch.setattr(cli, "HOME", home)
    commands = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kw: commands.append(command))
    cli.start(task)
    assert "--setenv=X500_STORAGE_ROOT=" + str(root) in commands[0]
    frozen_policy = next(
        s.split("=", 2)[2] for s in commands[0] if s.startswith("--setenv=X500_STORAGE_POLICY=")
    )
    assert json.loads(frozen_policy) == config["storage_policy"]
    assert not root.exists()  # Planning/submission stub never creates or collects data.


def test_real_extraction_worker_uses_custom_disk_without_x31(tmp_path):
    root = tmp_path / "custom data"
    archive = root / "archives/fixture.tar.gz"
    archive.parent.mkdir(parents=True)
    names = [
        "run.json",
        "reference/source_lock.json",
        "observations/metadata.json",
        "observations/streams.jsonl",
        "observations/watermarks.jsonl",
        "observations/diagnostics.jsonl",
    ]
    members, files = {}, {}
    with tarfile.open(archive, "w:gz") as tar:
        for name in names + ["evaluation/events.jsonl"]:
            body = (name + "\n").encode()
            member = tarfile.TarInfo(name)
            member.size = len(body)
            tar.addfile(member, io.BytesIO(body))
            members[name] = dict(
                kind="file", bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
            )
            if name in names:
                files[name] = members[name]["sha256"]
    target = root / "consumer/pretrain/task/run1"
    target.parent.mkdir(parents=True)
    request = dict(
        action="extract",
        row=dict(
            run_id="run1",
            path=str(target),
            files=files,
            archive=dict(
                path=str(archive),
                bytes=archive.stat().st_size,
                sha256=common.sha(archive),
                members=members,
            ),
        ),
        out=str(root / "evidence"),
        status=str(tmp_path / "status.json"),
    )
    common.write(tmp_path / "request.json", request)
    config = dict(external_storage=str(root), storage_policy=paths.capture(root))
    result = subprocess.run(
        [str(common.PYTHON), "-B", "-m", "pretrain.worker", str(tmp_path / "request.json")],
        env={**os.environ, **paths.environment(config), "PYTHONPATH": common.pythonpath()},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    status = common.read(request["status"])
    assert status["status"] == "COMPLETE" and status["result"]["truth_extracted"] is False
    assert {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()} == set(names)
    assert {name: common.sha(target / name) for name in names} == files
