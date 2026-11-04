"""Finite normal-only learning-rate selection for the sixteen-model study."""

import copy
import hashlib
import json
import math

import numpy as np

from execution.common import child, read, sha, write

from .models import STUDY_REVISION, StudySpec
from .optimization import CANDIDATES, POLICY, policy_for

REVISION = "normalization_optimizer_study"
VARIANTS = [
    dict(id=f"{name}_h{hidden}_l{layers}_{optimizer}_s0", name=name, hidden=hidden,
         layers=layers, heads=heads, optimizer=optimizer)
    for hidden, layers, heads in ((64, 2, 4), (128, 2, 4), (128, 4, 4), (256, 4, 8))
    for name in ("lstm", "transformer")
    for optimizer in ("sgd", "adam")
]
MODELS = [v["id"] for v in VARIANTS]
PROBE = dict(epochs=3, train_windows_max=8192, validation_windows_max=4096,
             selection="lowest final validation loss; ties use declared candidate order",
             origin_selection="evenly spaced indices over each frozen normal role",
             limitations="Short normal-only proxy; not guaranteed full-training optimum")


def configure(cfg):
    cfg = dict(cfg)
    cfg.update(revision=REVISION, variants=copy.deepcopy(VARIANTS), models=list(MODELS),
               model_revision=STUDY_REVISION, weight_decay=0.0, learning_rate=None,
               optimization_policy=copy.deepcopy(POLICY),
               learning_rate_candidates=copy.deepcopy(CANDIDATES),
               learning_rate_probe=copy.deepcopy(PROBE))
    cfg.pop("dropout", None)
    return cfg


def spec_for(entry):
    return StudySpec(entry["name"], entry["hidden"], entry["layers"], entry["heads"])


def _verified_json(path, expected_hash):
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_hash:
        raise ValueError("Learning-rate evidence changed: " + str(path))
    return json.loads(body)


def validated_selection(cfg, name, out):
    entry = next(v for v in cfg["variants"] if v["id"] == name)
    record = read(child(out, "learning-rate-selection.json"))
    if record.get("status") != "SELECTED_NORMAL_VALIDATION_ONLY":
        raise ValueError("Complete normal-only rate selection required")
    if record.get("protocol") != PROBE or any(record.get(key) is not False for key in (
        "event_truth_used", "development_data_used", "final_test_used", "probe_weights_reused_for_main"
    )):
        raise ValueError("Frozen normal-only learning-rate protocol changed")
    if record.get("candidate_grid") != CANDIDATES or set(record["models"]) != set(MODELS):
        raise ValueError("Rate selection does not cover the frozen sixteen configurations")
    if record.get("binding") != cfg["binding"]:
        raise ValueError("Rate selection data/scaler/source binding changed")
    chosen = record["models"][name]
    if [t["rate"] for t in chosen["trials"]] != CANDIDATES[entry["optimizer"]]:
        raise ValueError("Rate trials differ from the frozen three candidates")
    initial_hash = None
    for index, trial in enumerate(chosen["trials"]):
        if trial["directory"] != f"candidate-{index + 1}":
            raise ValueError("Unexpected learning-rate trial directory")
        path = child(out, "learning-rate-probe/" + name + "/" + trial["directory"])
        history = _verified_json(child(path, "history.json"), trial["history_sha256"])
        initial = _verified_json(child(path, "initialization.json"), trial["initialization_sha256"])
        if (initial.get("seed") != cfg["seed"]
                or initial.get("model_spec") != spec_for(entry).to_dict()
                or initial.get("weights_from_previous_training") is not False
                or not isinstance(initial.get("state_sha256"), str)):
            raise ValueError("Rate-trial initialization contract changed")
        initial_hash = initial_hash or initial["state_sha256"]
        if initial["state_sha256"] != initial_hash:
            raise ValueError("Rate candidates started from different initial weights")
        if (len(history) != PROBE["epochs"]
                or not math.isfinite(history[-1]["validation_loss"])
                or history[-1]["validation_loss"] != trial["final_validation_loss"]):
            raise ValueError("Rate-trial loss changed")
    best = min(chosen["trials"], key=lambda t: t["final_validation_loss"])
    if chosen["peak_learning_rate"] != best["rate"]:
        raise ValueError("Selected rate differs from frozen selection rule")
    return dict(learning_rate=chosen["peak_learning_rate"],
                optimization=policy_for(entry["optimizer"]), initial_state_sha256=initial_hash)


def selected_settings(cfg, name, out):
    selected = validated_selection(cfg, name, out)
    return {key: selected[key] for key in ("learning_rate", "optimization")}


def select_rates(cfg, out, train, validation, progress):
    from .training import run_fit

    output = out / "learning-rate-probe"
    output.mkdir(exist_ok=False)
    indices = {
        role: np.linspace(0, len(arrays["X"]) - 1,
                          min(PROBE[key], len(arrays["X"])), dtype=np.int64)
        for role, arrays, key in (("train", train, "train_windows_max"),
                                  ("validation", validation, "validation_windows_max"))
    }
    samples = {
        role: {k: np.asarray(arrays[k][indices[role]]) for k in ("X", "C", "Y")}
        for role, arrays in (("train", train), ("validation", validation))
    }
    record = dict(status="RUNNING", candidate_grid=copy.deepcopy(CANDIDATES), protocol=copy.deepcopy(PROBE),
                  binding=cfg["binding"], models={}, event_truth_used=False,
                  development_data_used=False, final_test_used=False,
                  probe_weights_reused_for_main=False,
                  selected_indices={k: v.tolist() for k, v in indices.items()})
    initialization = {}
    for entry in cfg["variants"]:
        name = entry["id"]
        trials = []
        for index, rate in enumerate(CANDIDATES[entry["optimizer"]]):
            progress.check()
            progress.update(stage="normal_learning_rate_probe", model=name,
                            learning_rate=rate, optimizer=entry["optimizer"],
                            rate_candidate=index + 1, rate_candidates=3)
            folder = output / name / f"candidate-{index + 1}"
            history = run_fit(spec_for(entry), samples["train"], samples["validation"], folder,
                              binding=cfg["binding"], device="cuda", gpu_authorized=True,
                              seed=cfg["seed"], batch_size=cfg["batch_size"],
                              epochs=PROBE["epochs"], patience=cfg["early_stopping_patience"],
                              learning_rate=rate, optimization=policy_for(entry["optimizer"]),
                              progress=progress, model_id=name)
            score = history[-1]["validation_loss"]
            if len(history) != PROBE["epochs"] or not math.isfinite(score):
                raise ValueError("Incomplete/nonfinite learning-rate trial; no fallback")
            initial = read(folder / "initialization.json")["state_sha256"]
            key = (entry["name"], entry["hidden"], entry["layers"], entry["heads"])
            if initial != initialization.setdefault(key, initial):
                raise ValueError("Candidates and optimizers must start from identical weights")
            trials.append(dict(rate=rate, directory=folder.name, final_validation_loss=score,
                               history_sha256=sha(folder / "history.json"),
                               initialization_sha256=sha(folder / "initialization.json")))
        record["models"][name] = dict(
            peak_learning_rate=min(trials, key=lambda t: t["final_validation_loss"])["rate"],
            trials=trials,
        )
        write(out / "learning-rate-selection-progress.json", record)
    record["status"] = "SELECTED_NORMAL_VALIDATION_ONLY"
    signature = hashlib.sha256()
    for role in ("train", "validation"):
        for key in ("X", "C", "Y"):
            signature.update(memoryview(np.ascontiguousarray(samples[role][key])).cast("B"))
    record["sample_sha256"] = signature.hexdigest()
    write(out / "learning-rate-selection.json", record)
