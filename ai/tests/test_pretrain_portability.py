"""Relocated CPU-only pipeline and immutable current-producer admission."""

import copy
import json
import os
import shutil
import subprocess

import pytest

from pretrain import cli, common


def test_current_inventory_rejects_tampering_and_path_escape(tmp_path, monkeypatch):
    native = tmp_path / "simulation"
    source = native / "config/producer-sources.json"
    common.write(source, dict(sources={}))
    binary = native / "build/current/libDronePowertrain.so"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic native output")
    seal = native / "artifacts/runtime-manifest.json"
    record = dict(
        source_manifest_sha256=common.sha(source),
        files={"build/current/libDronePowertrain.so": common.sha(binary)},
    )
    common.write(seal, record)
    monkeypatch.setattr(common, "SIMULATION", native)
    monkeypatch.setattr(common, "PRODUCER_MANIFEST", seal)
    monkeypatch.setattr(common, "producer_manifest", lambda: record)
    inventory = common.producer_sources()
    assert inventory["config/producer-sources.json"] == common.sha(source)
    assert inventory["artifacts/runtime-manifest.json"] == common.sha(seal)
    binary.write_bytes(b"changed native output")
    with pytest.raises(ValueError, match="runtime/source changed"):
        common.producer_sources()
    record["files"] = {"../outside": "0" * 64}
    with pytest.raises(ValueError, match="Unsafe producer"):
        common.producer_sources()


def test_systemd_cleanup_preserves_arguments_with_spaces_and_specifiers():
    command = cli.systemd_command(
        ["/tmp/clone with spaces/$name%/python", "-B", '/tmp/quote"name/task']
    )
    assert command == ('"/tmp/clone with spaces/$$name%%/python" "-B" "/tmp/quote\\"name/task"')


def test_relocated_preprocess_assemble_and_audit_without_torch(tmp_path):
    """Only exported collection code and storage helper exist in the other checkout."""
    clone = tmp_path / "another machine with spaces"
    source = common.ROOT / "Data/generation"
    shutil.copytree(
        source,
        clone / "Data/generation",
        ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", ".ruff_cache"),
    )
    (clone / "ai").mkdir()
    shutil.copy2(common.ROOT / "ai/x500_storage.py", clone / "ai/x500_storage.py")
    fixture = common.load("portable_raw_fixture", common.ROOT / "ai/tests/test_preprocessing.py")
    rows = []
    for index, role in enumerate(("normal_train", "normal_validation", "normal_calibration")):
        rid = f"{index + 1:032x}"
        run = clone / "Data/consumer" / rid
        metadata = copy.deepcopy(fixture.META)
        metadata["run_id"] = rid
        metadata["vehicle"].update(
            nominal_capacity_ah=12.0,
            nominal_parallel_cells=4,
            battery_profile_id="vtc6_4s4p",
            profile_revision="vtc6-4s4p",
        )
        common.write(run / "observations/metadata.json", metadata)
        common.write(
            run / "run.json", dict(run_id=rid, origin="synthetic_fixture", status="completed")
        )
        common.write(run / "reference/source_lock.json", dict(synthetic_fixture=True))
        events = list(fixture.events(72))
        for event in events:
            event["run_id"] = rid
        (run / "observations/streams.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        (run / "observations/watermarks.jsonl").write_text(
            json.dumps(
                dict(
                    schema="drone.raw.watermark/2.0",
                    run_id=rid,
                    clock_id="px4-boot",
                    through_available_time_us=72_000_000,
                    recorded_events=len(events),
                    closed=True,
                )
            )
            + "\n"
        )
        (run / "observations/diagnostics.jsonl").touch()
        files = {str(p.relative_to(run)): common.sha(p) for p in run.rglob("*") if p.is_file()}
        rows.append(dict(run_id=rid, role=role, path=str(run), files=files))
    common.write(clone / "rows.json", rows)
    script = r"""
import importlib.abc, json, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root / "Data/generation"), str(root / "ai")]
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "safetensors", "execution", "learning"}:
            raise RuntimeError("Training dependency loaded by CPU data pipeline: " + fullname)
sys.meta_path.insert(0, NoTraining())
from pretrain.common import Budget, PACKAGE, ROOT, STORE, package_imports, read, write
assert ROOT == root and STORE == root / "Data"
assert PACKAGE == root / "Data/generation/pretrain/resources/preprocessing"
action, index = sys.argv[2], int(sys.argv[3])
rows = read(root / "rows.json")
if action in {"preprocess", "audit"}:
    from pretrain.worker import execute
    row = rows[index]
    execute(dict(action=action, row=row, canonical=str(root / "canonical"),
        out=str(root / ("canonical" if action == "preprocess" else "audit") / row["run_id"]),
        status=str(root / "status" / (action + "-" + row["run_id"] + ".json"))))
else:
    from pretrain import assembly
    package_imports()
    from capstone_ai.cli import process
    from capstone_ai.preprocessing import Preprocessor
    from capstone_ai.source import DirectoryTail
    from pretrain.common import RESOURCES, load
    derived = load("portable_independent", RESOURCES / "admission/derived.py")
    def adopt(raw, target, config):
        source = root / "canonical" / Path(raw).name
        shutil.copytree(source, target)
        return read(target / "summary.json")
    dataset = assembly.prepare(dict(runs=rows), root / "dataset", read(PACKAGE / "preprocessing.json"), Budget(), process_run=adopt)
    report = assembly.inspect_dataset(dataset, dict(runs=rows), Budget(), inspect_run_fn=derived.inspect_run)
    assert report["train_only_scaler_verified"] is True
    assert set(report["by_role"]) == {"normal_train", "normal_validation", "normal_calibration"}
    assert all(Path(module.__file__).is_relative_to(root) for module in [sys.modules["pretrain"], sys.modules["capstone_ai"]])
    assert "torch" not in sys.modules and "safetensors" not in sys.modules
    write(root / "success.json", dict(passed=True, quality=report))
"""
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    env.pop("X500_STORAGE_ROOT", None)
    env.pop("X500_STORAGE_POLICY", None)
    (clone / "canonical").mkdir()
    (clone / "audit").mkdir()
    for action, count in (("preprocess", 3), ("audit", 3), ("assemble", 1)):
        for index in range(count):
            result = subprocess.run(
                [str(common.PYTHON), "-I", "-B", "-c", script, str(clone), action, str(index)],
                cwd=clone,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stdout + result.stderr
    assert common.read(clone / "success.json")["passed"] is True
