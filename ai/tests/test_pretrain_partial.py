"""One-run recovery must not repeat extraction or successful preprocessing."""

import pytest

from pretrain import common, composition, partial, runner


@pytest.mark.parametrize("fails", [False, True])
def test_one_run_only_and_failure_blocks_aggregation(tmp_path, monkeypatch, fails):
    source = tmp_path / "source"
    task = tmp_path / "new"
    config = dict(
        mode="finish_preprocess",
        source_sha256={},
        partial_source_task=str(source),
        partial_failed_run="bad",
        reused_run_count=1,
        workers=4,
    )
    common.write(task / "config.json", config)
    common.write(source / "cases.json", dict(cases=[]))
    r = runner.Runner(task)
    rows = [dict(run_id="good"), dict(run_id="bad")]
    frozen = dict(runs=rows, collection_profile="train104")
    monkeypatch.setattr(
        partial, "inputs", lambda p: (source / "prepared", frozen, {"good": {}}, "bad", [])
    )
    monkeypatch.setattr(partial, "external", lambda p: tmp_path / "output")
    monkeypatch.setattr(composition, "validate_roles", lambda *a, **k: None)
    calls = []

    def batch(name, action, rows, workers, out):
        assert action == "preprocess" and workers == 1 and [v["run_id"] for v in rows] == ["bad"]
        calls.append("single")
        if fails:
            raise RuntimeError("crash")

    monkeypatch.setattr(r, "batch", batch)

    def finish(frozen, root, label, workers, *, canonical_sources):
        assert len(frozen["runs"]) == 2 and workers == 4
        assert canonical_sources == {"good": source / "prepared/canonical-work/good"}
        calls.append("aggregate")
        return root

    monkeypatch.setattr(r, "finalize_prepared", finish)
    if fails:
        with pytest.raises(RuntimeError, match="crash"):
            partial.finish_partial(r)
        assert calls == ["single"]
    else:
        partial.finish_partial(r)
        assert calls == ["single", "aggregate"]
    with pytest.raises(ValueError, match="Recollection forbidden"):
        r.flight("collection", 16, [], "forbidden")


def test_successful_canonical_copy_preserves_original_and_checks_hash(tmp_path, monkeypatch):
    task = tmp_path / "task"
    common.write(task / "config.json", dict(mode="finish_preprocess", source_sha256={}))
    r = runner.Runner(task)
    old = tmp_path / "old/good"
    old.mkdir(parents=True)
    hashes = {}
    for n in ("frames", "windows", "exclusions"):
        (old / (n + ".jsonl")).write_text("{}\n")
        hashes[n] = common.sha(old / (n + ".jsonl"))
    summary = dict(
        config={"fixture": True}, input_hashes={"raw": "digest"}, closed=True, output_hashes=hashes
    )
    common.write(old / "summary.json", summary)
    (old / "checkpoint.json").write_text("unused checkpoint")
    before = {p: p.read_bytes() for p in old.iterdir()}
    root = tmp_path / "new"
    frozen = dict(
        runs=[dict(run_id="good", path=str(tmp_path / "raw/good"), files=summary["input_hashes"])]
    )

    class EndCheck(Exception):
        pass

    def prepare(frozen, out, config, budget, *, process_run):
        target = out / "canonical/good"
        assert process_run(frozen["runs"][0]["path"], target, summary["config"]) == summary
        assert (target / "frames.jsonl").read_bytes() == before[old / "frames.jsonl"]
        assert not (target / "checkpoint.json").exists()
        assert before == {p: p.read_bytes() for p in before}
        raise EndCheck

    from pretrain import assembly

    monkeypatch.setattr(assembly, "prepare", prepare)
    with pytest.raises(EndCheck):
        r.finalize_prepared(frozen, root, "prepared", 4, canonical_sources={"good": old})
