import copy
from pathlib import Path

import numpy as np
import pytest
from test_preprocessing import CONFIG, META, events, run

from capstone_ai.cli import compare
from capstone_ai.mapping import validate_metadata
from capstone_ai.preprocessing import Preprocessor
from capstone_ai.source import dump, resolve_profile
from capstone_ai.training import Predictor


def test_truth_fields_do_not_change_features():
    original = list(events())
    changed = copy.deepcopy(original)
    for e in changed:
        e["fields"]["true_soc"] = 0.01
        e["fields"]["fault_kind"] = 9000
        e["fields"]["temperature"] = 500
    a, b = run(original), run(changed)
    assert a.frames == b.frames
    assert a.windows == b.windows


def test_watermark_boundaries_and_clock_reset_reject():
    e = next(events(1))
    p = Preprocessor(META, CONFIG)
    e["sample_time_us"] = 1000000
    e["available_time_us"] = 0
    p.ingest(e)
    p.advance("px4-boot", 500000)
    assert not p.frames
    p.advance("px4-boot", 1000000)
    assert len(p.frames) == 1
    future = copy.deepcopy(e)
    future["seq"] = 2
    with pytest.raises(ValueError, match="committed watermark"):
        p.ingest(future)


def test_capability_fail_and_explicit_common15():
    m = copy.deepcopy(META)
    del m["semantic_bindings"]["climb_rate_target"]
    with pytest.raises(ValueError, match="profile_incomplete"):
        validate_metadata(m, CONFIG)
    c = {**CONFIG, "feature_profile": "common15_ctx6"}
    p = Preprocessor(m, c)
    assert len(p.names) == 15
    m = copy.deepcopy(META)
    m["conversion_history"] = ["converted_before"]
    with pytest.raises(ValueError, match="already converted"):
        validate_metadata(m, CONFIG)


def test_comparison_empty_is_not_pass(tmp_path):
    summary = {
        "run_id": "r",
        "origin": "px4_sitl",
        "config": CONFIG,
        "final_watermark": {},
        "input_hashes": {},
        "valid_frames": 0,
        "frames": 0,
        "windows": 0,
        "closed": True,
        "evidence": {"live_verified": True},
    }
    for name in ["a", "b"]:
        d = tmp_path / name
        d.mkdir()
        dump(d / "summary.json", summary)
        for file in ["frames", "windows", "exclusions"]:
            (d / (file + ".jsonl")).write_text("")
    result = compare(tmp_path / "a", tmp_path / "b", CONFIG)
    assert not result["passed"]
    assert result["C9_live"] == "not_executed"


def test_own_bundle_integrity_and_persistence():
    root = Path(__file__).resolve().parents[2]
    folder = root / "ai/artifacts/models/persistence"
    if not folder.exists():
        pytest.skip("Run local training to create real bundle first")
    p = Predictor(folder)
    x = np.zeros((2, 100, 19))
    c = np.zeros((2, 6))
    x[:, -1, 4:7] = [1, 2, 3]
    x[:, -1, 17] = 4
    result = p.predict_scaled(x, c)
    assert np.array_equal(result[0], np.tile([1, 2, 3, 4], (10, 1)))


def test_source_profile():
    assert (
        resolve_profile("ai/configs/sources/sitl_tail.json").name
        == "bfa6b58eff354d42a7ea8e50fa87064f"
    )
