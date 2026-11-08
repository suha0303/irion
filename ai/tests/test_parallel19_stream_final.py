"""Check final streaming preparation and sealed source bindings."""

import copy

import pytest

from learning.parallel19.control import read, sha, write
from learning.parallel19.final_verify import selection_binding
from learning.parallel19.refine import (
    accept_final,
    adopted_occupancy,
    enqueue_final,
)


def fixture(root):
    fit = root / "fit"
    for name in ("model.safetensors", "result.json", "scaler.json"):
        write(fit / name, {"fixture": True})
    write(fit / "integrity.json", {p.name: sha(p) for p in fit.iterdir()})
    best = {
        "status": "FIT_COMPLETE_STOPPED",
        "folder": str(fit),
        "lr": 0.1,
        "best_validation_loss": 0.2,
    }
    item = {
        "status": "LOCAL_NO_MEANINGFUL_IMPROVEMENT",
        "best": best,
        "tested": {"a": best},
    }
    return item, {str(fit): {"status": "DEVELOPMENT_EVALUATION_COMPLETE"}}


def test_final_waits_for_search_and_development_then_freezes_once(tmp_path):
    item, dev = fixture(tmp_path)
    pending = []
    state = {}
    root = tmp_path / "output"
    item["status"] = "SEARCHING"
    enqueue_final(root, {"m": item}, pending, dev, state)
    assert not pending
    item["status"] = "LOCAL_NO_MEANINGFUL_IMPROVEMENT"
    enqueue_final(root, {"m": item}, pending, {}, state)
    assert not pending
    enqueue_final(root, {"m": item}, pending, dev, state)
    enqueue_final(root, {"m": item}, pending, dev, state)
    assert len(pending) == 1 and pending[0]["kind"] == "final"
    req = read(pending[0]["request"])
    selection_binding(
        {"output": str(root), "final_per_model_when_search_stopped": True},
        req,
        "final",
        "m",
    )
    item["best"] = {**item["best"], "lr": 0.2}
    with pytest.raises(ValueError, match="selection changed"):
        enqueue_final(root, {"m": item}, pending, dev, state)


def test_final_score_cannot_change_search_selection(tmp_path):
    item, dev = fixture(tmp_path)
    pending = []
    state = {}
    root = tmp_path / "output"
    enqueue_final(root, {"m": item}, pending, dev, state)
    before = copy.deepcopy(item)
    result = {
        "status": "FINAL_VERIFICATION_COMPLETE",
        "evaluation_cohort": "final",
        "model_sha256": state["frozen_models"]["m"]["model_sha256"],
        "final_used_for_selection": False,
        "summary": {"recall": 0.0},
    }
    accept_final(root, state, pending[0], result)
    assert item == before and state["final_results"]["m"] == result
    result["model_sha256"] = "wrong"
    with pytest.raises(ValueError, match="binding"):
        accept_final(root, state, pending[0], result)


def test_final_request_cannot_select_another_weight_or_unfinished_model(
    tmp_path,
):
    item, dev = fixture(tmp_path)
    pending = []
    state = {}
    root = tmp_path / "output"
    enqueue_final(root, {"m": item}, pending, dev, state)
    req = read(pending[0]["request"])
    plan = {"output": str(root), "final_per_model_when_search_stopped": True}
    req["selected"] = {**req["selected"], "lr": 0.9}
    with pytest.raises(ValueError, match="frozen receipt"):
        selection_binding(plan, req, "final", "m")
    req = read(pending[0]["request"])
    p = req["selection_manifest"]
    receipt = read(p)
    receipt["search_status"] = "SEARCHING"
    write(p, receipt)
    req["selected_manifest_sha256"] = sha(p)
    with pytest.raises(ValueError, match="search must be frozen"):
        selection_binding(plan, req, "final", "m")


def test_adopt_completed_development_as_development_not_fit(tmp_path):
    output = tmp_path / "dev"
    write(output / "result.json", {"status": "DEVELOPMENT_EVALUATION_COMPLETE"})
    req = tmp_path / "request.json"
    write(req, {"output": str(output)})
    plan = {
        "adopted_allocation": 1,
        "allocation": {"leader": 1},
        "adopted_steps": [
            {
                "pid": 999999999,
                "node": "DIS05",
                "kind": "development",
                "request": str(req),
            }
        ],
    }
    state = {}
    busy, live = adopted_occupancy(plan, state)
    assert (
        busy == {"DIS05": 0, "DIS06": 0}
        and not live
        and state["adopted_complete"] == [999999999]
    )


def test_preserved_development_binding_is_explicit(tmp_path):
    old = tmp_path / "old"
    selection = old / "development-selections/m.json"
    chosen = {"lr": 0.1}
    write(selection, {"selected": chosen})
    req = {
        "selection_manifest": str(selection),
        "selected_manifest_sha256": sha(selection),
        "selected": chosen,
    }
    plan = {"output": str(tmp_path / "new"), "preserved_outputs": [str(old)]}
    selection_binding(plan, req, "development", "m")
    plan.pop("preserved_outputs")
    with pytest.raises(ValueError, match="development binding"):
        selection_binding(plan, req, "development", "m")
