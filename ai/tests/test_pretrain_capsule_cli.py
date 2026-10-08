"""Capsule recovery is exposed as a non-flight, finite task."""

from pathlib import Path

import pytest

from pretrain import cli, runner


def test_capsule_recovery_cli_parses_existing_source():
    args = cli.parser().parse_args(
        ["recover-capsules", "/tmp/source", "--name", "recover-fixture"]
    )
    assert args.source == Path("/tmp/source")
    assert args.name == "recover-fixture"


def test_capsule_recovery_never_enters_flight():
    instance = object.__new__(runner.Runner)
    instance.config = {"mode": "recover_capsules"}
    with pytest.raises(ValueError, match="Recollection forbidden"):
        instance.flight("collection", 16, [], "forbidden")


def test_capsule_recovery_uses_existing_memory_reserve(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        cli,
        "collector",
        lambda: SimpleNamespace(
            memory_snapshot=lambda: {"available_bytes": 6_000_000_000}
        ),
    )
    result = cli.service_memory(
        dict(mode="recover_capsules", use_available_ram=True, min_available_ram_gb=1)
    )
    assert result["max_bytes"] == 5_000_000_000
    assert result["reserve_bytes"] == 1_000_000_000


def test_capsule_display_never_claims_new_collection(tmp_path, monkeypatch):
    from pretrain.common import write

    write(
        tmp_path / "config.json",
        dict(
            mode="recover_capsules",
            storage_profile="stream_compact",
            collection_run_count=1200,
            collection_sets=4,
            flight_workers=16,
            audit_workers=4,
            gates=False,
            pipeline_overlap=False,
        ),
    )
    monkeypatch.setattr(
        cli, "snapshot", lambda task: dict(status="RUNNING", stage="prepared-restore")
    )
    result = cli.render(tmp_path)
    assert "보존 캡슐 1200run" in result
    assert "재수집·학습 없음" in result
    assert "수집 계획" not in result
    assert "압축 완료   0/1200" not in result


def test_recovery_service_sets_allocator_before_python_starts(tmp_path, monkeypatch):
    from pretrain.common import read, sha, write

    home = tmp_path / "home"
    task = home / "tasks/recovery"
    write(task / "config.json", dict(
        mode="recover_capsules", source_sha256={}, python_allocator="debug",
        external_storage=str(tmp_path / "Data"),
    ))
    (task / "config.sha256").write_text(sha(task / "config.json"))
    monkeypatch.setattr(cli, "HOME", home)
    commands = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kw: commands.append(command))
    cli.start(task)
    assert "--setenv=PYTHONMALLOC=debug" in commands[0]
    assert "--setenv=PYTHONFAULTHANDLER=1" in commands[0]
    assert read(task / "operator/service.json")["python_allocator"] == "debug"


def test_worker_rejects_missing_allocator_before_claim(tmp_path, monkeypatch):
    from pretrain.common import sha, write

    write(tmp_path / "config.json", dict(
        mode="recover_capsules", source_sha256={}, python_allocator="debug",
    ))
    (tmp_path / "config.sha256").write_text(sha(tmp_path / "config.json"))
    monkeypatch.delenv("PYTHONMALLOC", raising=False)
    with pytest.raises(ValueError, match="allocator mismatch"):
        runner.Runner(tmp_path).run()
    assert not (tmp_path / "operator/claimed").exists()
