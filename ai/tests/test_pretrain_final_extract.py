"""Exercise final workflow directory admission with the real six-file extractor."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import x500_storage as storage

from pretrain import common
from pretrain import final_dataset as final


@pytest.fixture
def final_extraction(tmp_path, monkeypatch):
    common.package_imports()
    from preprocessing_tools.control import RAW_FILES

    from pretrain import archive, worker

    root = tmp_path / "selected data"
    task = tmp_path / "tasks/final-fixture"
    monkeypatch.setenv(storage.ENV_ROOT, str(root))
    monkeypatch.setenv(storage.ENV_POLICY, json.dumps(storage.capture(root)))
    monkeypatch.setattr(final, "STORE", root)
    monkeypatch.setattr(archive, "STORAGE", root)
    rows = final.cases(170001000)
    planned, protection = task / "planned-cases.json", task / "protection.json"
    common.write(planned, dict(sets=[rows]))
    empty = dict(sources={}, seeds=[], group_ids=[], raw_ids=[], raw_sha256=[])
    common.write(protection, empty)
    monkeypatch.setattr(final, "protection_inputs", lambda **kwargs: empty)
    folder = task / "final-collection"
    common.write(folder / "plan.json", dict(source_hashes={"fixture": "bound"}))
    common.write(folder / "execution/ledger.json", dict(status="fixture"))
    admitted = [
        dict(
            row,
            run_id=f"{i:032x}",
            path=str(root / "consumer/final-test" / task.name / f"{i:032x}"),
        )
        for i, row in enumerate(rows, 1)
    ]
    compressed = root / "archives/fixture.tar.gz"
    compressed.parent.mkdir(parents=True)
    members = {}
    with tarfile.open(compressed, "w:gz") as stream:
        for name in sorted(RAW_FILES | {"evaluation/events.jsonl"}):
            content = (name + "\n").encode()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
            members[name] = dict(
                kind="file", bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
            )
    first = admitted[0]
    first.update(
        files={name: members[name]["sha256"] for name in RAW_FILES},
        archive=dict(
            path=str(compressed),
            bytes=compressed.stat().st_size,
            sha256=common.sha(compressed),
            members=members,
        ),
    )
    monkeypatch.setattr(final, "admit_rows", lambda *args: admitted)
    actions = []
    status = task / "extract-status.json"

    def batch(name, action, items, workers, out, **kwargs):
        actions.append(action)
        if action == "extract":
            # Do not create the missing parent here: the workflow owns that step.
            worker.execute(dict(action="extract", row=first, out=str(out), status=str(status)))
        else:
            for row in items:
                path = Path(out) / row["run_id"]
                common.write(
                    path / ("summary.json" if action == "preprocess" else "report.json"),
                    dict(output_hashes={}) if action == "preprocess" else dict(accepted_targets=1),
                )

    runner = SimpleNamespace(
        task=task,
        config=dict(
            planned_cases=str(planned),
            unit_attempts_max=3,
            pipeline_overlap=False,
            final_protection=str(protection),
            flight_workers=4,
            io_workers=2,
            workers=2,
            audit_workers=2,
        ),
        update=lambda **kwargs: None,
        flight=lambda *args: folder,
        begin=lambda *args, **kwargs: None,
        finish=lambda *args: None,
        batch=batch,
    )
    return SimpleNamespace(
        root=root,
        task=task,
        runner=runner,
        row=first,
        status=status,
        actions=actions,
        archive=archive,
        worker=worker,
        consumer=Path(first["path"]).parent,
    )


def test_final_workflow_creates_nested_parent_before_real_extraction(final_extraction):
    fixture = final_extraction
    assert not fixture.consumer.exists()
    # Reproduce the old call ordering: the per-run extractor cannot create its parent.
    with pytest.raises(FileNotFoundError):
        fixture.archive.extract_consumer(fixture.row, common.Budget())
    final.run(fixture.runner)
    assert fixture.actions == ["extract", "preprocess", "audit"]
    status = common.read(fixture.status)
    assert status["status"] == "COMPLETE"
    assert status["result"]["truth_extracted"] is False
    target = Path(fixture.row["path"])
    assert {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()} == set(
        fixture.row["files"]
    )
    assert {name: common.sha(target / name) for name in fixture.row["files"]} == fixture.row[
        "files"
    ]
    assert not (target / "evaluation").exists()
    with pytest.raises(FileExistsError):
        fixture.archive.extract_consumer(fixture.row, common.Budget())
    assert {name: common.sha(target / name) for name in fixture.row["files"]} == fixture.row[
        "files"
    ]


def test_final_workflow_preserves_existing_consumer_directory(final_extraction):
    fixture = final_extraction
    fixture.consumer.mkdir(parents=True)
    sentinel = fixture.consumer / "previous-evidence"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        final.run(fixture.runner)
    assert fixture.actions == []
    assert sentinel.read_text() == "preserve"
    assert not Path(fixture.row["path"]).exists()


def test_final_workflow_rejects_symlink_in_consumer_parent(final_extraction, tmp_path):
    fixture = final_extraction
    outside = tmp_path / "outside"
    outside.mkdir()
    (fixture.root / "consumer").mkdir()
    (fixture.root / "consumer/final-test").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        final.run(fixture.runner)
    assert fixture.actions == []
    assert list(outside.iterdir()) == []
