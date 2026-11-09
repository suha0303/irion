"""Check bounded learning-rate plans and independent continuation evidence."""

import copy

import pytest

from learning.parallel19.control import read, write
from learning.parallel19.handover_upgrade import preserve_final_selections
from learning.parallel19.refine import (
    candidates,
    finish_round,
    key,
    widen_open_rounds,
)


def item():
    best = {
        "lr": 0.27712812921102037,
        "best_validation_loss": 1.0,
        "initial": "seed0",
        "status": "FIT_COMPLETE_STOPPED",
    }
    return {
        "best": best,
        "tested": {
            key(r): {"lr": r}
            for r in [
                0.00866025403784,
                0.0173205080757,
                0.0346410161514,
                0.0692820323028,
                0.138564064606,
                best["lr"],
            ]
        },
        "status": "SEARCHING",
        "round_open": True,
        "rounds": 4,
        "round_results": [],
        "factor": 2.0,
        "no_improvement": 0,
    }


def test_parallel_grid_is_unique_bounded_and_does_not_repeat_completed():
    m = item()
    m["candidate_width"] = 5
    r = candidates(m)
    assert len(r) == 5 and len(set(map(key, r))) == 5
    assert all(1e-5 <= x <= 1 and key(x) not in m["tested"] for x in r)
    assert key(r[0]) == key(m["best"]["lr"] * 2)


def test_widen_preserves_active_request_and_completed_models(tmp_path):
    m = item()
    done = copy.deepcopy(m)
    done["status"] = "LOCAL_NO_MEANINGFUL_IMPROVEMENT"
    before = copy.deepcopy(done)
    req = tmp_path / "active.json"
    write(
        req, {"lr": m["best"]["lr"] * 2, "node": "DIS05", "output": "existing"}
    )
    original = req.read_bytes()
    pending = [{"model": "remaining", "request": str(req), "attempt": 1}]
    plan = {"parallel_lr_candidate_widths": {"remaining": 5, "finished": 4}}
    models = {"remaining": m, "finished": done}
    widen_open_rounds(plan, tmp_path, models, pending)
    widen_open_rounds(plan, tmp_path, models, pending)
    assert len(pending) == 5 and req.read_bytes() == original and done == before
    rates = [read(j["request"])["lr"] for j in pending]
    assert len(set(map(key, rates))) == 5
    assert models["remaining"]["best"]["lr"] == m["best"]["lr"]


def test_wide_comparison_uses_validation_loss_and_same_stop_rule():
    m = item()
    m["candidate_width"] = 4
    for _ in range(2):
        rates = candidates(m)
        finish_round(
            m,
            [
                {
                    "status": "FIT_COMPLETE_STOPPED",
                    "lr": r,
                    "best_validation_loss": 1.01,
                    "final_recall": 1.0,
                }
                for r in rates
            ],
        )
    assert (
        m["status"] == "LOCAL_NO_MEANINGFUL_IMPROVEMENT"
        and m["best"]["best_validation_loss"] == 1.0
    )


def test_widen_does_not_duplicate_current_round_receipts(tmp_path):
    m = item()
    m["round_results"] = [
        {"lr": m["best"]["lr"] * 2, "status": "FIT_COMPLETE_STOPPED"}
    ]
    pending = []
    widen_open_rounds(
        {"parallel_lr_candidate_widths": {"m": 4}}, tmp_path, {"m": m}, pending
    )
    assert len(pending) == 3
    assert all(
        key(read(j["request"])["lr"]) != key(m["round_results"][0]["lr"])
        for j in pending
    )


def test_remaining_model_can_use_all_nine_candidate_slots(tmp_path):
    models = {"a": item(), "b": item()}
    pending = []
    plan = {
        "parallel_lr_candidate_widths": {"a": 5, "b": 4},
        "parallel_lr_total_candidates": 9,
    }
    widen_open_rounds(plan, tmp_path, models, pending)
    assert len(pending) == 9
    models["b"]["status"] = "LOCAL_NO_MEANINGFUL_IMPROVEMENT"
    widen_open_rounds(plan, tmp_path, models, pending)
    assert models["a"]["candidate_width"] == 9
    assert len([j for j in pending if j["model"] == "a"]) == 9


def test_handover_preserves_selections_from_paused_snapshot(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    selection = {"lr": 0.1, "model_sha256": "same"}
    source = old / "output/final-selections/m.json"
    write(source, {"selected": selection})
    state = {"frozen_models": {"m": selection}}
    preserve_final_selections(old, new, state)
    assert (new / "final-selections/m.json").read_bytes() == source.read_bytes()
    write(source, {"selected": {"lr": 0.9}})
    with pytest.raises(ValueError, match="selection changed"):
        preserve_final_selections(old, new, state)
