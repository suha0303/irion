"""Load the delivered 16 models with their frozen preprocessing and calibration."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "ai/artifacts/training19-final-20261108"
MODEL_IDS = tuple(
    f"{task}_{name}_h{hidden}_l{layers}"
    for task in ("forecast", "run_health")
    for name in ("lstm", "transformer")
    for hidden in (64, 128)
    for layers in (2, 4)
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Catalog:
    def __init__(self, bundle=DEFAULT_BUNDLE):
        self.root = Path(bundle).absolute()
        self.manifest = read_json(self.root / "MANIFEST.json")
        if self.manifest.get("schema") != "drone-ai.final-delivery/1":
            raise ValueError("Unsupported model delivery")
        if set(self.manifest["models"]) != set(MODEL_IDS):
            raise ValueError(
                "Expected the frozen forecast 8 and run-health 8 model set"
            )

    def checked(self, relative):
        path = self.root / relative
        if relative not in self.manifest["files"] or not path.is_relative_to(
            self.root
        ):
            raise ValueError("File is not in the model manifest")
        if ".." in Path(relative).parts or any(
            p.is_symlink() for p in (path, *path.parents)
        ):
            raise ValueError("Symlink or traversal in model bundle")
        if digest(path) != self.manifest["files"][relative]:
            raise ValueError(f"Model bundle integrity mismatch: {relative}")
        return path

    def entry(self, model_id):
        if model_id not in MODEL_IDS:
            raise ValueError("Unknown model ID")
        prefix = f"models/{model_id}/"
        files = {
            name: self.checked(prefix + name)
            for name in (
                "model-spec.json",
                "model.safetensors",
                "scaler.npz",
                "scaler.json",
                "evaluation/calibration.json",
            )
        }
        spec = read_json(files["model-spec.json"])
        expected_id = (
            f"{spec['task']}_{spec['name']}_h{spec['hidden']}_l{spec['layers']}"
        )
        if model_id != expected_id:
            raise ValueError("Model ID/specification mismatch")
        calibration = read_json(files["evaluation/calibration.json"])
        if calibration["task"] != spec["task"]:
            raise ValueError("Calibration task mismatch")
        return dict(
            id=model_id,
            **spec,
            label=f"{spec['name'].upper()} · {spec['hidden']} · {spec['layers']}층",
            version="training19-20261108",
            device="cpu",
            files=files,
            weight_sha256=digest(files["model.safetensors"]),
            calibration=calibration["parameters"],
            validated_domain="X500 simulation only",
            real_aircraft_validated=False,
        )

    def list(self):
        return [
            {
                k: v
                for k, v in self.entry(mid).items()
                if k not in {"files", "calibration"}
            }
            for mid in MODEL_IDS
        ]

    def check_model_sources(self):
        for name in ("history19", "run_health19"):
            relative = f"code/ai/learning/{name}/models.py"
            frozen = self.checked(relative)
            if digest(frozen) != digest(ROOT / f"ai/learning/{name}/models.py"):
                raise ValueError(
                    f"Model implementation differs from delivered code: {name}"
                )
