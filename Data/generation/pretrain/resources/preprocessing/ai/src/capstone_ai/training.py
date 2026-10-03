import copy
import hashlib
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .mapping import FEATURES, TARGETS
from .models import device_for, make_model
from .source import dump, read_json, sha


def normalized(dataset, split, scaler):
    a = np.load(Path(dataset) / (split + ".npz"))
    return {
        "X": (a["X"] - scaler["x_mean"]) / scaler["x_scale"],
        "C": (a["C"] - scaler["c_mean"]) / scaler["c_scale"],
        "Y": (a["Y"] - scaler["y_mean"]) / scaler["y_scale"],
        "run_id": a["run_id"],
        "origin_us": a["origin_us"],
        "segment": a["segment"],
    }


def loss(yhat, y):
    sq = (yhat - y).square()
    return 0.5 * sq[..., :3].mean() + 0.5 * sq[..., 3].mean()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


def tensor(a, device):
    return torch.as_tensor(a, dtype=torch.float32, device=device)


def neural_fit(name, cfg, train, val, folder, resume=None, stop_after=None):
    device = device_for(cfg["device"])
    seed_all(cfg["seed"])
    model = make_model(name, cfg["hidden"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
    data_hash = hashlib.sha256()
    for split in [train, val]:
        for key in ["X", "C", "Y"]:
            data_hash.update(np.ascontiguousarray(split[key]).tobytes())
    data_signature = data_hash.hexdigest()
    source_hashes = {n: sha(Path(__file__).with_name(n)) for n in ["training.py", "models.py"]}
    start, best, history = 0, float("inf"), []
    best_state = copy.deepcopy(model.state_dict())
    if resume:
        # Only our own locally generated checkpoint is accepted by this training API.
        # RNG byte tensors must remain on CPU; load_state_dict moves model/optimizer state.
        ck = torch.load(resume, map_location="cpu")
        if ck["config"] != cfg or ck["data_signature"] != data_signature:
            raise ValueError("resume config mismatch")
        if ck.get("source_hashes", source_hashes) != source_hashes:
            raise ValueError("resume source mismatch")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        if device.type == "cuda":
            if not ck.get("cuda_rng"):
                raise ValueError("CUDA resume requires saved CUDA RNG state")
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        start, best, history, best_state = (
            ck["epoch"] + 1,
            ck["best"],
            ck["history"],
            ck["best_state"],
        )
    x, c, y = [tensor(train[n], device) for n in ["X", "C", "Y"]]
    vx, vc, vy = [tensor(val[n], device) for n in ["X", "C", "Y"]]
    for epoch in range(start, cfg["epochs"]):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for idx in order.split(cfg["batch_size"]):
            optimizer.zero_grad()
            pred = model(x[idx], c[idx])
            value = loss(pred, y[idx])
            if not torch.isfinite(value):
                raise ValueError("nonfinite training loss")
            value.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("nonfinite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += value.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vloss = sum(
                loss(
                    model(vx[i : i + cfg["batch_size"]], vc[i : i + cfg["batch_size"]]),
                    vy[i : i + cfg["batch_size"]],
                ).item()
                * len(vx[i : i + cfg["batch_size"]])
                for i in range(0, len(vx), cfg["batch_size"])
            ) / len(vx)
        history.append({"epoch": epoch, "train_loss": total / len(x), "validation_loss": vloss})
        if vloss < best:
            best = vloss
            best_state = copy.deepcopy(model.state_dict())
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "epoch": epoch,
            "best": best,
            "best_state": best_state,
            "history": history,
            "config": cfg,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "source_hashes": source_hashes,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "data_order": order.tolist(),
            "data_signature": data_signature,
        }
        torch.save(checkpoint, folder / "last.pt")
        if vloss == best:
            torch.save(checkpoint, folder / "best.pt")
        print(name, history[-1], flush=True)
        if stop_after is not None and epoch + 1 >= stop_after:
            break
        best_epoch = min(history, key=lambda r: r["validation_loss"])["epoch"]
        if epoch - best_epoch >= cfg.get("early_stopping_patience", cfg["epochs"]):
            break
    model.load_state_dict(best_state)
    model.eval()
    save_file(
        {k: v.cpu().contiguous() for k, v in model.state_dict().items()},
        str(folder / "model.safetensors"),
    )
    restored = make_model(name, cfg["hidden"]).to(device)
    restored.load_state_dict(load_file(str(folder / "model.safetensors")))
    restored.eval()
    with torch.no_grad():
        before = model(vx[:2], vc[:2]).cpu()
        after = restored(vx[:2], vc[:2]).cpu()
    difference = (before - after).abs().max().item()
    if difference > 1e-6:
        raise ValueError("model save/load mismatch")
    return history, difference


def flat(data):
    return np.concatenate(
        [data["X"].reshape(len(data["X"]), -1), data["C"], np.ones((len(data["X"]), 1))], axis=1
    )


def train_models(config_path):
    cfg = read_json(config_path)
    dataset = Path(cfg["dataset"])
    out = Path(cfg["output"])
    if (out / "comparison_training.json").exists():
        raise ValueError("training revision exists; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    scaler = dict(np.load(dataset / "scaler.npz"))
    train, val = [normalized(dataset, s, scaler) for s in ["train", "validation"]]
    result = {}
    for name in ["persistence", "ridge_forecast", "lstm", "transformer"]:
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        history = []
        difference = 0.0
        if name in {"lstm", "transformer"}:
            history, difference = neural_fit(name, cfg, train, val, folder)
        elif name == "ridge_forecast":
            x = flat(train)
            y = train["Y"].reshape(len(x), -1)
            v = flat(val)
            best = float("inf")
            for alpha in [1.0, 10.0, 100.0]:
                coefficients = x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y)
                prediction = (v @ coefficients).reshape(val["Y"].shape)
                error = (prediction - val["Y"]) ** 2
                value = 0.5 * error[..., :3].mean() + 0.5 * error[..., 3].mean()
                history.append({"alpha": alpha, "validation_loss": float(value)})
                if value < best:
                    best = value
                    best_coefficients = coefficients
                    chosen = alpha
            np.savez(folder / "ridge.npz", coefficients=best_coefficients, alpha=chosen)
        shutil.copyfile(dataset / "scaler.npz", folder / "scaler.npz")
        shutil.copyfile(dataset / "manifest.json", folder / "training_manifest.json")
        dump(
            folder / "model.json",
            {
                "name": name,
                "hidden": cfg["hidden"],
                "task": "forecast",
                "target_order": TARGETS,
                "feature_order": FEATURES,
                "config": cfg,
            },
        )
        dump(
            folder / "preprocessing.json",
            read_json(cfg.get("preprocessing", "ai/configs/preprocessing.json")),
        )
        dump(
            folder / "forecast_contract.json",
            {
                "schema": "drone.forecast/3.0",
                "history": 100,
                "horizon": 10,
                "step_us": 100000,
                "cadence_us": 1000000,
                "require_issued_before_first_target": True,
                "teacher_forcing": False,
                "context": "60s + segment integral",
            },
        )
        dump(
            folder / "training.json",
            {
                "history": history,
                "seconds": time.perf_counter() - started,
                "save_load_max_abs": difference,
                "purpose": cfg["purpose"],
            },
        )
        dump(
            folder / "integrity.json",
            {
                p.name: sha(p)
                for p in folder.iterdir()
                if p.is_file() and p.name != "integrity.json"
            },
        )
        result[name] = read_json(folder / "training.json")
    dump(out / "comparison_training.json", result)
    return result


class Predictor:
    def __init__(self, folder, device="cpu"):
        self.folder = Path(folder)
        self.device = device_for(device)
        for name, expected in read_json(self.folder / "integrity.json").items():
            if sha(self.folder / name) != expected:
                raise ValueError("bundle hash mismatch: " + name)
        self.metadata = read_json(self.folder / "model.json")
        self.name = self.metadata["name"]
        if (
            self.metadata["task"] != "forecast"
            or self.metadata["feature_order"] != FEATURES
            or self.metadata["target_order"] != TARGETS
        ):
            raise ValueError("bundle forecast contract mismatch")
        self.scaler = dict(np.load(self.folder / "scaler.npz"))
        if self.name in {"lstm", "transformer"}:
            self.model = make_model(self.name, self.metadata["hidden"]).to(self.device)
            self.model.load_state_dict(load_file(str(self.folder / "model.safetensors")))
            self.model.eval()
        elif self.name == "ridge_forecast":
            self.coefficients = np.load(self.folder / "ridge.npz")["coefficients"]

    def predict_scaled(self, x, c):
        if self.name == "persistence":
            indices = [FEATURES.index(n) for n in TARGETS]
            return np.repeat(x[:, -1, indices][:, None, :], 10, axis=1)
        if self.name == "ridge_forecast":
            return (flat({"X": x, "C": c}) @ self.coefficients).reshape(-1, 10, 4)
        with torch.no_grad():
            batch = self.metadata["config"]["batch_size"]
            return np.concatenate(
                [
                    self.model(
                        tensor(x[i : i + batch], self.device), tensor(c[i : i + batch], self.device)
                    )
                    .cpu()
                    .numpy()
                    for i in range(0, len(x), batch)
                ]
            )

    def physical(self, scaled):
        values = scaled * self.scaler["y_scale"] + self.scaler["y_mean"]
        values[..., -1] *= 4
        return values
