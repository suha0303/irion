"""Read a new history19 export; historical context models are never loaded here."""

from pathlib import Path

import numpy as np
import torch
from execution.common import child, read, sha
from safetensors.torch import load_file

from .models import ModelSpec, make_model


class Predictor:
    def __init__(self, folder):
        self.folder = Path(folder)
        integrity = read(self.folder / "integrity.json")
        if not {
            "model.safetensors",
            "model-spec.json",
            "scaler.npz",
            "input-contract.json",
        } <= set(integrity):
            raise ValueError("Complete history19 export required")
        for name, digest in integrity.items():
            if sha(child(self.folder, name)) != digest:
                raise ValueError("History19 export changed")
        contract = read(self.folder / "input-contract.json")
        if contract != dict(
            feature_profile="common19_ctx0",
            input_shape=[100, 19],
            output_shape=[10, 4],
            optimizer="sgd",
        ):
            raise ValueError("History19 input contract mismatch")
        self.spec = ModelSpec(
            **read(self.folder / "model-spec.json")
        ).validate()
        self.model = make_model(self.spec).eval()
        self.model.load_state_dict(
            load_file(str(self.folder / "model.safetensors"))
        )
        with np.load(self.folder / "scaler.npz", allow_pickle=False) as source:
            self.scaler = dict(source)
        if set(self.scaler) != {"x_mean", "x_scale", "y_mean", "y_scale"}:
            raise ValueError("X/Y-only scaler required")

    def predict_scaled(self, history):
        x = np.asarray(history, dtype=np.float32)
        if (
            x.ndim != 3
            or x.shape[1:] != (100, 19)
            or len(x) == 0
            or not np.isfinite(x).all()
        ):
            raise ValueError("Finite history[N,100,19] required")
        from .training import backend

        with backend("cpu"), torch.no_grad():
            return self.model(torch.from_numpy(x)).numpy()

    def predict(self, history):
        normalized = (
            np.asarray(history) - self.scaler["x_mean"]
        ) / self.scaler["x_scale"]
        values = (
            self.predict_scaled(normalized) * self.scaler["y_scale"]
            + self.scaler["y_mean"]
        )
        values[..., 3] *= 4
        return values


def seal_export(folder, scaler):
    from execution.common import write

    folder = Path(folder)
    np.savez(folder / "scaler.npz", **scaler)
    write(
        folder / "input-contract.json",
        dict(
            feature_profile="common19_ctx0",
            input_shape=[100, 19],
            output_shape=[10, 4],
            optimizer="sgd",
        ),
    )
    write(
        folder / "integrity.json",
        {
            p.name: sha(p)
            for p in folder.iterdir()
            if p.is_file() and p.name != "integrity.json"
        },
    )
