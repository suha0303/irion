"""Local immutable adapted models, portable backups and explicit profile
selection.
"""

import hashlib
import json
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from execution.prepare import npz_headers

from .adapt import write
from .adapt_support import parameter_names
from .catalog import Catalog, digest, read_json
from .runtime import Runtime

FILES = ("result.json", "model.safetensors", "scaler.npz")
MAX_ARCHIVE = 64 * 1024 * 1024


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def checked_id(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Invalid registered model ID")
    return value


def validate(folder, bundle):
    folder = Path(folder)
    for name in FILES:
        path = folder / name
        if (
            any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file()
            or path.stat().st_size > MAX_ARCHIVE
        ):
            raise ValueError("Invalid model package member")
    result = read_json(folder / "result.json")
    if (
        result["status"] != "experimental_adaptation_complete"
        or result["calibration_status"] != "not_calibrated"
    ):
        raise ValueError("Only complete uncalibrated adaptations are supported")
    runtime = Runtime(result["base_model_id"], catalog=Catalog(bundle))
    if result["binding"]["base_sha256"] != runtime.entry["weight_sha256"]:
        raise ValueError("Adapted model base mismatch")
    if digest(folder / "model.safetensors") != result["model_sha256"]:
        raise ValueError("Adapted weights changed")
    headers = npz_headers(folder / "scaler.npz")
    if set(headers) != set(runtime.scaler) or any(
        headers[k]["shape"] != v.shape or headers[k]["bytes"] > v.nbytes + 1024
        for k, v in runtime.scaler.items()
    ):
        raise ValueError("Invalid scaler array dimensions")
    with np.load(folder / "scaler.npz", allow_pickle=False) as scaler:
        if set(scaler.files) != set(runtime.scaler) or any(
            not np.array_equal(scaler[k], v) for k, v in runtime.scaler.items()
        ):
            raise ValueError("Frozen scaler mismatch")
    weights = load_file(str(folder / "model.safetensors"), device="cpu")
    original = runtime.model.state_dict()
    permitted = parameter_names(
        runtime.model, result["binding"].get("trainable", "head")
    )
    declared = result["binding"].get("trainable_names")
    if declared is not None and set(declared) != permitted:
        raise ValueError("Adaptation parameter scope mismatch")
    if set(weights) != set(original):
        raise ValueError("Model tensor names mismatch")
    for key, tensor in weights.items():
        if (
            tensor.shape != original[key].shape
            or tensor.dtype != original[key].dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Invalid model tensor")
        if key not in permitted and not torch.equal(tensor, original[key]):
            raise ValueError("Frozen model encoder changed")
    runtime.model.load_state_dict(weights, strict=True)
    if runtime.entry["task"] == "forecast":
        runtime.predict_scaled(np.zeros((1, 100, 19), dtype="f4"))
    else:
        runtime.predict_scaled(
            np.zeros((1, 110, 19), dtype="f4"), np.array([110], dtype="i8")
        )
    return result


def model_path(root, model_id):
    return Path(root) / "models" / checked_id(model_id)


def register(root, source, *, bundle):
    result = validate(source, bundle)
    portable = {k: v for k, v in result.items() if k != "output"}
    parent = Path(root) / "models"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".register-", dir=parent) as tmp:
        stage = Path(tmp)
        for name in FILES[1:]:
            shutil.copyfile(Path(source) / name, stage / name)
        write(stage / "result.json", portable)
        hashes = {name: digest(stage / name) for name in FILES}
        model_id = fingerprint(hashes)
        descriptor = dict(
            schema="drone.local-adapted/1",
            id=model_id,
            files=hashes,
            base_model_id=result["base_model_id"],
            profile=result["binding"]["profile"],
            model_sha256=result["model_sha256"],
            origin=result["origin"],
            calibration_status="not_calibrated",
        )
        write(stage / "package.json", descriptor)
        destination = parent / model_id
        if destination.exists():
            read_model(root, model_id, bundle=bundle)
        else:
            stage.rename(destination)
    return descriptor


def read_model(root, model_id, *, bundle, full=True):
    folder = model_path(root, model_id)
    descriptor = read_json(folder / "package.json")
    if (
        descriptor.get("schema") != "drone.local-adapted/1"
        or descriptor["id"] != model_id
        or set(descriptor["files"]) != set(FILES)
    ):
        raise ValueError("Invalid model package manifest")
    if fingerprint(descriptor["files"]) != model_id or any(
        digest(folder / k) != h for k, h in descriptor["files"].items()
    ):
        raise ValueError("Model package integrity mismatch")
    result = (
        validate(folder, bundle) if full else read_json(folder / "result.json")
    )
    if (
        any(
            descriptor[k] != result[k]
            for k in ("base_model_id", "model_sha256", "origin")
        )
        or descriptor["profile"] != result["binding"]["profile"]
    ):
        raise ValueError("Model package metadata mismatch")
    return descriptor


def list_models(root, *, bundle):
    models = [
        read_model(root, p.name, bundle=bundle, full=False)
        for p in sorted((Path(root) / "models").glob("*"))
        if p.is_dir() and not p.name.startswith(".")
    ]
    state = Path(root) / "model-selection.json"
    return dict(
        models=models,
        selection=read_json(state)
        if state.exists()
        else dict(active={}, history=[]),
    )


def export_model(root, model_id, output, *, bundle):
    read_model(root, model_id, bundle=bundle)
    folder = model_path(root, model_id)
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in (*FILES, "package.json"):
            archive.write(folder / name, name)
    return dict(path=str(output), sha256=digest(output), model_id=model_id)


def import_model(root, source, *, bundle):
    source = Path(source)
    if any(p.is_symlink() for p in (source, *source.parents)):
        raise ValueError("Symlink model archives are forbidden")
    if source.stat().st_size > MAX_ARCHIVE:
        raise ValueError("Model archive exceeds 64 MiB")
    with (
        zipfile.ZipFile(source) as archive,
        tempfile.TemporaryDirectory(prefix="drone-model-") as tmp,
    ):
        entries = archive.infolist()
        names = [e.filename for e in entries]
        if (
            len(names) != 4
            or set(names) != {*FILES, "package.json"}
            or sum(e.file_size for e in entries) > MAX_ARCHIVE
        ):
            raise ValueError(
                "Unexpected model archive entries or uncompressed size"
            )
        folder = Path(tmp)
        for entry in entries:
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are forbidden in model archives")
            with (
                archive.open(entry) as src,
                (folder / entry.filename).open("xb") as dst,
            ):
                shutil.copyfileobj(src, dst)
        descriptor = read_json(folder / "package.json")
        hashes = {name: digest(folder / name) for name in FILES}
        if descriptor.get("files") != hashes or descriptor.get(
            "id"
        ) != fingerprint(hashes):
            raise ValueError("Imported package hash mismatch")
        # No code, pickle or raw flight logs are loaded from the archive.
        return register(root, folder, bundle=bundle)


def selection_key(profile, base_model_id):
    return fingerprint(dict(profile=profile, base_model_id=base_model_id))


def select_model(root, model_id, comparison, *, bundle):
    descriptor = read_model(root, model_id, bundle=bundle)
    measured = comparison.get("adapted", {})
    if (
        comparison.get("independent") is not True
        or measured.get("adapted_sha256") != descriptor["model_sha256"]
        or measured.get("profile_version") != fingerprint(descriptor["profile"])
    ):
        raise ValueError(
            (
                "An independent local comparison of this model and profile is "
                "required"
            )
        )
    result = read_json(model_path(root, model_id) / "result.json")
    if result["binding"].get("promotion_policy"):
        review = comparison.get("promotion_check", {})
        if (
            review.get("status") != "passed"
            or review.get("model_sha256") != descriptor["model_sha256"]
            or review.get("policy") != result["binding"]["promotion_policy"]
        ):
            raise ValueError(
                "정상·이상 퇴행 검증을 통과해야 새 모델을 선택할 수 있습니다."
            )
    return change_selection(
        root,
        descriptor["profile"],
        descriptor["base_model_id"],
        model_id,
        comparison,
    )


def change_selection(root, profile, base_model_id, model_id, comparison=None):
    path = Path(root) / "model-selection.json"
    state = read_json(path) if path.exists() else dict(active={}, history=[])
    key = selection_key(profile, base_model_id)
    old = state["active"].get(key)
    event = dict(
        id=uuid.uuid4().hex,
        time=time.time(),
        key=key,
        previous=old,
        selected=model_id,
        base_model_id=base_model_id,
        profile_id=profile["id"],
        comparison=comparison,
    )
    state["active"][key] = model_id
    state["history"].append(event)
    write(path, state)
    return event


def rollback(root, event_id, *, bundle):
    state = read_json(Path(root) / "model-selection.json")
    event = next(
        (e for e in reversed(state["history"]) if e["id"] == event_id), None
    )
    if event is None or state["active"].get(event["key"]) != event["selected"]:
        raise ValueError(
            "Selection changed since this event; choose the latest event"
        )
    # Only undo the latest change for this key, including repeated selections.
    if (
        next(e for e in reversed(state["history"]) if e["key"] == event["key"])[
            "id"
        ]
        != event_id
    ):
        raise ValueError("Only the latest selection event can be rolled back")
    old = event["previous"]
    if old:
        read_model(root, old, bundle=bundle)
    descriptor = read_model(root, event["selected"] or old, bundle=bundle)
    return change_selection(
        root, descriptor["profile"], descriptor["base_model_id"], old
    )


def active_folder(root, profile, base_model_id, *, bundle, fill_policy=None):
    state = Path(root) / "model-selection.json"
    if not state.exists():
        return None
    model_id = read_json(state)["active"].get(
        selection_key(profile, base_model_id)
    )
    if not model_id:
        return None
    descriptor = read_model(root, model_id, bundle=bundle)
    if (
        descriptor["profile"] != profile
        or descriptor["base_model_id"] != base_model_id
    ):
        raise ValueError("Selected model profile mismatch")
    folder = model_path(root, model_id)
    if (
        read_json(folder / "result.json")["binding"].get("fill_policy")
        != fill_policy
    ):
        return None
    return folder
