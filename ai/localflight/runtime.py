"""Bounded FP32 CPU inference. No remote, CUDA, fitting or calibration side effects."""

import numpy as np
import torch
from safetensors.torch import load_file

from learning.history19.models import ModelSpec as ForecastSpec
from learning.history19.models import make_model as forecast_model
from learning.run_health19.models import ModelSpec as RunSpec
from learning.run_health19.models import make_model as run_model

from .catalog import Catalog


class Runtime:
    def __init__(self, model_id, *, catalog=None, threads=2):
        if type(threads) is not int or not 1 <= threads <= 8:
            raise ValueError("CPU threads must be 1..8")
        torch.set_num_threads(threads)
        self.catalog = catalog or Catalog()
        self.catalog.check_model_sources()
        self.entry = self.catalog.entry(model_id)
        e = self.entry
        spec_type, make = (
            (ForecastSpec, forecast_model) if e["task"] == "forecast" else (RunSpec, run_model)
        )
        self.model = make(spec_type(e["name"], e["hidden"], e["layers"], 4)).cpu()
        self.model.load_state_dict(
            load_file(str(e["files"]["model.safetensors"]), device="cpu"), strict=True
        )
        self.model.eval()
        with np.load(e["files"]["scaler.npz"], allow_pickle=False) as data:
            self.scaler = {
                key: data[key].copy() for key in ("x_mean", "x_scale", "y_mean", "y_scale")
            }
        for key, size in (("x_mean", 19), ("x_scale", 19), ("y_mean", 4), ("y_scale", 4)):
            a = self.scaler[key]
            if (
                a.shape != (size,)
                or not np.isfinite(a).all()
                or ("scale" in key and np.any(a <= 0))
            ):
                raise ValueError("Invalid frozen scaler")

    def predict_scaled(self, values, lengths=None):
        a = np.asarray(values)
        if a.dtype != np.float32 or a.ndim != 3 or a.shape[-1] != 19 or not np.isfinite(a).all():
            raise ValueError("Finite FP32 [batch,time,19] normalized input required")
        if self.entry["task"] == "forecast":
            if a.shape[1] != 100 or not 1 <= len(a) <= 32 or lengths is not None:
                raise ValueError("Forecast requires 1..32 windows of 100 frames")
            args = (torch.from_numpy(np.array(a, copy=True)),)
        else:
            if len(a) != 1 or not 101 <= a.shape[1] <= 10000:
                raise ValueError(
                    "Run health requires one contiguous completed run of 101..10000 frames"
                )
            n = np.asarray(lengths)
            if n.dtype != np.int64 or n.shape != (1,) or n[0] != a.shape[1]:
                raise ValueError("Exact unpadded run length required")
            args = (torch.from_numpy(np.array(a, copy=True)), torch.from_numpy(n.copy()))
        with torch.inference_mode():
            out = self.model(*args).cpu().numpy()
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite prediction; no health verdict")
        return out
