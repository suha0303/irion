"""New collections must not depend on retired experiment reports."""

from pretrain import collection
from pretrain.common import PRODUCER_MANIFEST, ROOT, read, sha, write


def test_new_plan_uses_current_preparation_without_historical_report(tmp_path, monkeypatch):
    retired = ROOT / "ai/experiments"

    def current_only(path):
        path = path if hasattr(path, "is_relative_to") else ROOT / path
        if path.is_relative_to(retired):
            raise FileNotFoundError("Retired experiment is unavailable")
        return sha(path)

    monkeypatch.setattr(collection, "sha", current_only)
    write(
        tmp_path / "config.json",
        dict(pipeline_overlap=False, continue_after_case_failure=True, unit_attempts_max=3),
    )
    rows = collection.cases(1_000_000, "balanced300")["cases"]
    folder = collection.plan(tmp_path, "collection", 16, rows, "fixture", export_workers=4)
    plan = read(folder / "plan.json")
    expected = PRODUCER_MANIFEST
    assert plan["producer_preparation"] == dict(path=str(expected), sha256=sha(expected))
    assert plan["isolation_evidence"]["path"] == str(folder / "isolation.json")
    assert plan["execution_authorized"] is True
    assert plan["execution_dir"] == str(folder / "execution")
    assert "gate_receipt" not in plan
    assert len(read(folder / "manifest.json")["cases"]) == 300
    assert plan["unit_attempts_max"] == 3 and plan["export_workers"] == 4
    assert all(
        not (collection.SIMULATION / name).is_relative_to(retired) for name in plan["source_hashes"]
    )
    assert plan["assessment_command"][:3] == [
        str(collection.SIMULATION / ".venv/bin/python"),
        "-B",
        str(collection.SIMULATION / "sim/tools/assess_run.py"),
    ]
    # Current readiness remains a real, mandatory, hash-bound input.
    assert plan["project_source_hashes"][str(expected.relative_to(ROOT))] == sha(expected)
