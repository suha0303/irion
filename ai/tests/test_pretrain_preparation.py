"""Machine readiness must remain hash-bound without a past round report."""

import copy

import pytest

from pretrain import collection


@pytest.fixture
def readiness(monkeypatch):
    record = dict(
        schema="x500.producer-runtime/1",
        revision="X500_CURRENT",
        configuration="collection",
        profile_id="test-profile",
        source_manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(collection, "producer_manifest", lambda: copy.deepcopy(record))
    return record


def test_current_machine_preparation_is_admitted(readiness):
    assert collection.preparation_assertions(readiness, "test-profile") == readiness


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unverified"),
        ("revision", "retired"),
        ("configuration", "other"),
        ("profile_id", "other"),
        ("source_manifest_sha256", "0" * 64),
    ],
)
def test_wrong_identity_or_source_binding_rejected(readiness, field, value):
    readiness = dict(readiness, **{field: value})
    with pytest.raises(ValueError, match=field + " mismatch"):
        collection.preparation_assertions(readiness, "test-profile")
