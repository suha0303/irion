"""Exercise offline migration without rewriting raw or hash-bound content."""

import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_pipeline import battery

from arion.data_lock import INCOMPLETE, data_lock
from arion.data_transfer import inventory, migrate
from arion.file_io import atomic_json, read_json
from arion.main import create_app


def saved_data(root):
    root.mkdir()
    raw = root / "logs/flight.jsonl"
    raw.parent.mkdir()
    raw.write_text(
        "\n".join(json.dumps(battery(t)) for t in (0, 100, 200)),
        encoding="utf-8",
    )
    atomic_json(
        root / "catalog.json",
        dict(schema_version="arion-files-v1", runs={}, sessions={}),
    )
    profile = dict(id="fixture", notes=str(root / "keep-this-note"))
    atomic_json(root / "local-ai/aircraft/fixture/hash.json", profile)
    atomic_json(
        root / "local-ai/jobs/job.json",
        dict(
            id="job",
            action="inspect",
            status="complete",
            created_at=0,
            request=dict(path=str(raw), profile=profile),
            result=dict(sha256="fixed"),
        ),
    )
    atomic_json(
        root / "local-ai/prepared/fixture/manifest.json",
        dict(file={"sha256": "unchanged"}),
    )
    return raw


def test_copy_preserves_source_and_hash_bound_files_and_can_move_again(
    tmp_path,
):
    source, destination = tmp_path / "source", tmp_path / "destination"
    raw = saved_data(source)
    before = inventory(source)
    result = migrate(source, destination)
    assert inventory(source) == before
    assert not (destination / INCOMPLETE).exists()
    report = read_json(result["report"])
    assert report["immutable_files_preserved"]
    job = read_json(destination / "local-ai/jobs/job.json")
    assert job["request"]["path"] == str(destination / raw.relative_to(source))
    assert job["request"]["profile"]["notes"] == str(source / "keep-this-note")
    for relative in (
        "logs/flight.jsonl",
        "local-ai/aircraft/fixture/hash.json",
        "local-ai/prepared/fixture/manifest.json",
    ):
        assert (source / relative).read_bytes() == (
            destination / relative
        ).read_bytes()
    original = next(
        (destination / ".migrations").glob(
            "*/original-metadata/local-ai/jobs/job.json"
        )
    )
    assert (
        original.read_bytes()
        == (source / "local-ai/jobs/job.json").read_bytes()
    )
    second = tmp_path / "second"
    migrate(destination, second)
    assert read_json(second / "local-ai/jobs/job.json")["request"][
        "path"
    ] == str(second / "logs/flight.jsonl")


def test_live_app_blocks_migration_and_second_app_before_initialization(
    tmp_path,
):
    source = tmp_path / "source"
    with TestClient(create_app(source)):
        before = inventory(source)
        with pytest.raises(ValueError, match="사용 중"):
            migrate(source, tmp_path / "destination")
        with pytest.raises(ValueError, match="사용 중"):
            create_app(source)
        assert inventory(source) == before
    with data_lock(source):
        pass


def test_replay_after_original_directory_becomes_unavailable(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    content = "\n".join(json.dumps(battery(t)) for t in (0, 100, 200)).encode()
    with TestClient(create_app(source)) as client:
        response = client.post(
            "/api/runs/import", files={"file": ("flight.jsonl", content)}
        )
        assert response.status_code == 200, response.text
        run_id = response.json()["run_id"]
        frame = client.get(f"/api/runs/{run_id}/frame?t_ms=100").json()
    migrate(source, destination)
    hidden = tmp_path / "original-preserved"
    source.rename(hidden)
    try:
        with TestClient(create_app(destination)) as client:
            assert (
                client.get(f"/api/runs/{run_id}/frame?t_ms=100").json() == frame
            )
    finally:
        hidden.rename(source)


@pytest.mark.parametrize(
    "problem",
    ["existing", "nested", "symlink", "external", "legacy", "running"],
)
def test_rejects_unsafe_or_unsupported_transfer_before_copy(tmp_path, problem):
    source, destination = tmp_path / "source", tmp_path / "destination"
    saved_data(source)
    if problem == "existing":
        destination.mkdir()
        (destination / "keep").write_text("user data", encoding="utf-8")
    elif problem == "nested":
        destination = source / "nested"
    elif problem == "symlink":
        (source / "linked").symlink_to(tmp_path)
    elif problem == "external":
        atomic_json(
            source / "local-ai/jobs/job.json",
            dict(request=dict(path=str(tmp_path / "outside.BIN"))),
        )
    elif problem == "legacy":
        (source / "arion.sqlite3").write_bytes(b"not migrated")
    elif problem == "running":
        atomic_json(source / "server-8765.json", dict(pid=os.getpid()))
    with pytest.raises(ValueError):
        migrate(source, destination)
    if problem == "existing":
        assert (destination / "keep").read_text(encoding="utf-8") == "user data"
    else:
        assert not destination.exists()


def test_disk_shortage_leaves_source_and_destination_untouched(
    tmp_path, monkeypatch
):
    source, destination = tmp_path / "source", tmp_path / "destination"
    saved_data(source)
    before = inventory(source)
    monkeypatch.setattr(
        "arion.data_transfer.shutil.disk_usage",
        lambda _: SimpleNamespace(free=0),
    )
    with pytest.raises(ValueError, match="디스크"):
        migrate(source, destination)
    assert inventory(source) == before and not destination.exists()


def test_detects_uncooperative_writer_and_blocks_partial_copy(
    tmp_path, monkeypatch
):
    import shutil

    source, destination = tmp_path / "source", tmp_path / "destination"
    raw = saved_data(source)
    original_copy = shutil.copy2
    changed = False

    def concurrent_write(src, dst):
        nonlocal changed
        result = original_copy(src, dst)
        if not changed:
            raw.write_bytes(raw.read_bytes() + b"\n")
            changed = True
        return result

    monkeypatch.setattr("arion.data_transfer.shutil.copy2", concurrent_write)
    with pytest.raises(ValueError):
        migrate(source, destination)
    assert (destination / INCOMPLETE).exists()
    with pytest.raises(ValueError, match="완료되지 않은"):
        create_app(destination)
