import json

import pytest
from execution.delivery import export_final as e
from execution.delivery.fetch_final import verify


def test_delivery_requires_release_not_just_training_completion(tmp_path):
    e.write(tmp_path / "output/plan.json", {})
    e.write(
        tmp_path / "output/state.json",
        {"status": "FINAL_VERIFICATION_COMPLETE_PENDING_RELEASE_CHECK"},
    )
    with pytest.raises(ValueError, match="release"):
        e.export(tmp_path)
    assert not (tmp_path / "delivery").exists()


def test_unsafe_paths_and_symlinks_are_rejected(tmp_path):
    for name in ("../outside", "/absolute"):
        with pytest.raises(ValueError):
            e.safe(tmp_path, name)
    (tmp_path / "link").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError):
        e.safe(tmp_path, "link")


def test_verify_detects_extra_and_corrupt_files(tmp_path):
    (tmp_path / "model").write_bytes(b"weights")
    manifest = {"files": {"model": e.sha(tmp_path / "model")}}
    verify(tmp_path, manifest)
    (tmp_path / "model").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path, manifest)
    (tmp_path / "extra").write_bytes(b"")
    with pytest.raises(ValueError, match="Unexpected"):
        verify(tmp_path, manifest)


def test_complete_export_contains_only_selected_models_and_final_evidence(
    tmp_path, monkeypatch
):
    base = tmp_path / "campaign"
    output = base / "output"
    inputs = base / "inputs"
    code = base / "code"
    old = tmp_path / "baseline"
    e.write(inputs / "main/scaler.npz", {"standin": "scaler"})
    e.write(inputs / "main/fit-manifest.json", {"train": 540})
    e.write(
        inputs / "cache.json",
        {"files": {"main/scaler.npz": e.sha(inputs / "main/scaler.npz")}},
    )
    final_inputs = base / "final-inputs"
    e.write(final_inputs / "cache.json", {"test": 729})
    e.write(code / "ai/example.py", {"source": "test"})
    plan = {
        "owner": "test",
        "baseline": str(old),
        "inputs": str(inputs),
        "final_inputs": str(final_inputs),
        "code": str(code),
        "source_sha256": {"ai/example.py": e.sha(code / "ai/example.py")},
        "cache_sha256": e.sha(inputs / "cache.json"),
        "final_cache_sha256": e.sha(final_inputs / "cache.json"),
        "models": [{"id": f"m{i}"} for i in range(16)],
    }
    state = {
        "status": "REPORT_READY_STOPPED",
        "allocation_released": True,
        "next_allocation_queued": False,
        "allocation": {"leader": 1, "components": {"DIS05": 1, "DIS06": 2}},
        "final_requests": {},
    }
    selected = {"models": {}, "final_test_used_for_selection": False}
    results = {}
    for item in plan["models"]:
        name = item["id"]
        training = old / name / "main"
        integrity = {}
        for filename in (
            "model.safetensors",
            "model-spec.json",
            "scaler.json",
            "history.json",
            "best.json",
            "result.json",
        ):
            e.write(training / filename, {"model": name})
            integrity[filename] = e.sha(training / filename)
        e.write(training / "last.pt", {"optimizer": "not for delivery"})
        integrity["last.pt"] = e.sha(training / "last.pt")
        e.write(training / "integrity.json", integrity)
        selected["models"][name] = {
            "folder": str(training),
            "lr": 0.1,
            "model_sha256": integrity["model.safetensors"],
        }
        final = output / name / "final"
        e.write(final / "calibration.json", {"threshold": 1})
        result = {
            "status": "FINAL_VERIFICATION_COMPLETE",
            "model_sha256": integrity["model.safetensors"],
            "calibration_sha256": e.sha(final / "calibration.json"),
            "final_used_for_selection": False,
        }
        e.write(final / "result.json", result)
        e.write(
            final / "integrity.json",
            {p.name: e.sha(p) for p in final.iterdir() if p.is_file()},
        )
        request = base / "requests" / f"{name}.json"
        e.write(request, {"output": str(final)})
        state["final_requests"][name] = {"request": str(request)}
        results[name] = result
    for name, value in [
        ("plan", plan),
        ("state", state),
        ("selected-models", selected),
        ("final-results", results),
    ]:
        e.write(output / (name + ".json"), value)
    monkeypatch.setattr(e.subprocess, "check_output", lambda *a, **kw: "")
    result = e.export(base)
    bundle = base / "delivery/final"
    manifest = e.read(bundle / "MANIFEST.json")
    assert (
        result["status"] == "FINAL_EXPORT_READY"
        and len(manifest["models"]) == 16
    )
    assert not any("last.pt" in name for name in manifest["files"])
    verify(bundle, manifest)
    assert e.export(base)["manifest_sha256"] == result["manifest_sha256"]
