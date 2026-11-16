"""Training variants preserve flight separation and bound their influence."""

import json

import numpy as np
import pytest
from test_localflight_adapt import flight

from localflight.adapt import fit
from localflight.adapt_data import forecast_training, training_groups
from localflight.catalog import DEFAULT_BUNDLE, digest
from localflight.prepared import load, windows
from localflight.registry import validate
from localflight.runtime import Runtime


def variant(root, parent, name):
    path = flight(root, name, "adaptation")
    metadata = json.loads(parent.read_text())
    other = json.loads(path.read_text())
    other["source_sha256"] = metadata["source_sha256"]
    other["training_augmentation"] = dict(
        parent_manifest_sha256=digest(parent),
        recipe_sha256="1" * 64,
        generator_sha256="2" * 64,
    )
    path.write_text(json.dumps(other))
    return path


def test_views_cannot_leak_development_or_change_identity(tmp_path):
    train = flight(tmp_path, "train", "adaptation")
    dev = flight(tmp_path, "dev", "development")
    altered = variant(tmp_path, train, "view")
    source = load(train)[0]["source_sha256"]
    with pytest.raises(ValueError, match="belong"):
        training_groups(
            [train], [load(train)], {load(dev)[0]["source_sha256"]: [dev]}
        )
    with pytest.raises(ValueError, match="Duplicate"):
        training_groups([train], [load(train)], {source: [altered, altered]})
    metadata = json.loads(altered.read_text())
    metadata["role"] = "development"
    altered.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identity/provenance"):
        training_groups([train], [load(train)], {source: [altered]})


def test_window_budget_balances_flights_and_preserves_endpoints(tmp_path):
    path = flight(tmp_path, "train", "adaptation")
    m, v, _ = load(path)
    v = np.tile(v, (30, 1))
    t = np.arange(len(v), dtype="i8") * 100000
    scaler = Runtime("forecast_lstm_h64_l2").scaler
    data = (m, v, t)
    x, y, counts = forecast_training([[data] * 8] * 8, scaler, bounded=True)
    assert len(x) == len(y) == 3000
    assert {
        sum(c["selected"] for c in counts if c["flight"] == i) for i in range(8)
    } == {375}
    expected = windows(*data, scaler)
    np.testing.assert_array_equal(x[0], expected[0][0])
    np.testing.assert_array_equal(y[counts[0]["selected"] - 1], expected[1][-1])


@pytest.mark.parametrize(
    "model_id", ["forecast_lstm_h64_l2", "run_health_transformer_h64_l2"]
)
def test_grouped_fit_keeps_original_identity_and_restores(tmp_path, model_id):
    train = flight(tmp_path, "train", "adaptation")
    dev = flight(tmp_path, "dev", "development")
    altered = variant(tmp_path, train, "view")
    source = load(train)[0]["source_sha256"]
    out = tmp_path / "fitted"
    result = fit(
        model_id,
        [train],
        [dev],
        out,
        bundle=DEFAULT_BUNDLE,
        epochs=3,
        strategy="staged_last_blocks",
        training_views={source: [altered]},
    )
    assert result["binding"]["training_view_counts"] == [2]
    assert len(result["binding"]["source_sha256"]) == 2
    assert result["binding"]["training_views"][0]["source_sha256"] == source
    assert (
        validate(out, DEFAULT_BUNDLE)["model_sha256"] == result["model_sha256"]
    )
