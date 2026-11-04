"""Normal evidence can replay calibration without model inference or development truth."""

import shutil
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from execution import engine
from execution.common import read, sha, write


@pytest.mark.parametrize("preserve", [False, True])
def test_calibration_evidence_is_exact_normal_only(tmp_path, monkeypatch, preserve):
    upstream, out = tmp_path / "upstream", tmp_path / "compute"
    dataset = upstream / "dataset"
    write(dataset / "manifest.json", {"runs": [
        {"run_id": "val", "role": "normal_validation", "paired_case_id": "v"},
        {"run_id": "cal", "role": "normal_calibration", "paired_case_id": "c"},
        {"run_id": "dev", "role": "normal_dev_eval", "paired_case_id": "d"},
    ]})
    write(out / "models/persistence/integrity.json", {})
    values = {}
    for role, run in (("normal_validation", "val"), ("normal_calibration", "cal")):
        values[role] = dict(
            X=np.zeros((2, 100, 19)), C=np.zeros((2, 6)),
            Y=np.arange(80, dtype=np.float64).reshape(2, 10, 4) / 7,
            run_id=np.array([run, run]), segment=np.array([1, 1]),
            origin_us=np.array([1000000, 2000000]),
        )
    predictor = SimpleNamespace(
        scaler={"y_scale": np.array([1., 2., 3., 4.])},
        predict_scaled=lambda x, c: np.zeros((len(x), 10, 4)),
    )
    monkeypatch.setattr(engine, "Predictor", lambda *a, **k: predictor)
    monkeypatch.setattr(engine, "normalized_role", lambda dataset, role, scaler: values[role])
    monkeypatch.setattr(engine, "fit_calibration", lambda *a: {"status": "fixture"})
    replays = []
    monkeypatch.setattr(engine, "seal_development", lambda *args, **kw: replays.append(args[3]))
    progress = SimpleNamespace(check=lambda: None, update=lambda **k: None)
    engine.calibrate(upstream, out, progress, model_names=["persistence"],
                     backend_factory=lambda device: nullcontext(),
                     preserve_normal_residuals=preserve)
    folder = out / "calibration/persistence"
    evidence = folder / "normal-residuals.npz"
    assert evidence.exists() is preserve
    if preserve:
        metadata = read(folder / "normal-residuals.json")
        assert metadata["sha256"] == sha(evidence)
        assert metadata["truth_used"] is False
        assert metadata["paired_case_ids"] == {"val": "v", "cal": "c"}
        with np.load(evidence, allow_pickle=False) as saved:
            for label, role in (("validation", "normal_validation"),
                                ("calibration", "normal_calibration")):
                np.testing.assert_array_equal(saved[label + "_residual_scaled"], values[role]["Y"])
                np.testing.assert_array_equal(saved[label + "_run_id"], values[role]["run_id"])
            np.testing.assert_array_equal(saved["physical_target_scale"], [1., 2., 3., 16.])
        # A new attempt may reuse the completed normal fit, but must replay every
        # development run again. The old outputs are never overwritten.
        reused_out = tmp_path / "fresh-compute"
        write(reused_out / "models/persistence/integrity.json", {})
        copied = reused_out / "calibration/persistence"
        copied.mkdir(parents=True)
        hashes = {}
        for file in ("calibration.json", "normal-residuals.json", "normal-residuals.npz"):
            shutil.copyfile(folder / file, copied / file)
            hashes[file] = sha(copied / file)

        def no_refit(*args):
            raise AssertionError("Completed normal calibration must not be refitted")

        monkeypatch.setattr(engine, "fit_calibration", no_refit)
        monkeypatch.setattr(engine, "normalized_role", lambda dataset, role, scaler:
                            values[role] if role == "normal_validation" else no_refit())
        engine.calibrate(upstream, reused_out, progress, model_names=["persistence"],
                         reuse_calibration={"persistence": {"files": hashes}},
                         backend_factory=lambda device: nullcontext())
        assert replays == [folder / "dev", copied / "dev"]
        assert all(sha(folder / file) == sha(copied / file) == digest
                   for file, digest in hashes.items())
