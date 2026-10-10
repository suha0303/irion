"""Only the missing audit may run; completed dataset/other audits are read-only."""

from types import SimpleNamespace

import numpy as np
import pytest

from pretrain import audit_recovery, common, runner


@pytest.mark.parametrize("fails", [False, True])
def test_one_audit_and_no_preprocessing(tmp_path, monkeypatch, fails):
    task = tmp_path / "new"
    source = tmp_path / "old"
    common.write(
        task / "config.json",
        dict(
            mode="finish_audit",
            source_sha256={},
            audit_source_task=str(source),
            audit_failed_run="bad",
            reused_audit_count=1,
        ),
    )
    r = runner.Runner(task)
    frozen = dict(runs=[dict(run_id="good"), dict(run_id="bad")])
    monkeypatch.setattr(
        audit_recovery, "inputs", lambda p: (source / "prepared", frozen, {"good": {}}, "bad", [])
    )
    monkeypatch.setattr(audit_recovery, "external", lambda p: tmp_path / "output")
    calls = []

    def batch(name, action, rows, workers, out, *, canonical):
        assert action == "audit" and workers == 1 and rows == [dict(run_id="bad")]
        assert canonical == source / "prepared/dataset/canonical"
        calls.append("one-audit")
        if fails:
            raise RuntimeError("audit failed")

    monkeypatch.setattr(r, "batch", batch)

    def quality(frozen, root, label, dataset, *, audit_sources):
        assert dataset == source / "prepared/dataset"
        assert set(audit_sources) == {"good", "bad"}
        assert audit_sources["good"][0] == source / "prepared/audit-work/good"
        calls.append("quality")
        return root

    monkeypatch.setattr(r, "finish_quality", quality)
    if fails:
        with pytest.raises(RuntimeError, match="audit failed"):
            audit_recovery.finish_audit(r)
        assert calls == ["one-audit"]
    else:
        audit_recovery.finish_audit(r)
        assert calls == ["one-audit", "quality"]
    with pytest.raises(ValueError, match="Recollection forbidden"):
        r.flight("collection", 16, [], "forbidden")


def test_quality_reuses_audit_and_dataset_without_recomputing(tmp_path, monkeypatch):
    task = tmp_path / "task"
    common.write(task / "config.json", dict(mode="finish_audit", source_sha256={}))
    r = runner.Runner(task)
    old = tmp_path / "old"
    dataset = old / "dataset"
    canonical = dataset / "canonical/good"
    canonical.mkdir(parents=True)
    (canonical / "frames.jsonl").write_text("{}\n")
    common.write(
        canonical / "summary.json",
        dict(output_hashes={"frames": common.sha(canonical / "frames.jsonl")}),
    )
    cache = old / "audit/good"
    cache.mkdir(parents=True)
    np.savez(cache / "numeric.npz", train_frames=np.array([[1.0]]))
    common.write(cache / "report.json", dict(run_id="good"))
    status = old / "status.json"
    common.write(
        status,
        dict(
            status="COMPLETE",
            run_id="good",
            result=dict(
                numeric_sha256=common.sha(cache / "numeric.npz"),
                report_sha256=common.sha(cache / "report.json"),
            ),
        ),
    )
    frozen = dict(runs=[dict(run_id="good")])
    root = tmp_path / "new-output"
    common.write(root / "frozen-manifest.json", frozen)
    before = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}

    def inspect(ds, frozen, *, inspect_run_fn):
        assert ds == dataset
        report, samples, frames = inspect_run_fn(canonical, frozen["runs"][0])
        assert report == dict(run_id="good") and samples == []
        assert frames[0].tolist() == [1.0]
        return dict(fixture=True)

    monkeypatch.setattr(runner, "load", lambda *a: SimpleNamespace(inspect_dataset=inspect))
    r.finish_quality(frozen, root, "prepared", dataset, audit_sources={"good": (cache, status)})
    assert before == {p: p.read_bytes() for p in before}
    assert common.read(root / "handoff.json")["dataset"] == str(dataset)
    assert common.read(root / "handoff.json")["training_authorized"] is False
