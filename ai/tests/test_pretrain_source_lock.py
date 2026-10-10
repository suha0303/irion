"""Source identity is independent of dictionary order, never file content."""

import copy
import hashlib
import json

import pytest

from pretrain.common import RESOURCES, load, package_imports


def freezer():
    package_imports()
    return load("collection_source_lock_test", RESOURCES / "admission/freeze.py")


def digest(value, *, sorted_keys=False):
    return hashlib.sha256(
        (json.dumps(value, indent=2, sort_keys=sorted_keys, allow_nan=False) + "\n").encode()
    ).hexdigest()


SOURCE = {
    "local_sources": {"sim/b.py": "b" * 64, "sim/a.py": "a" * 64},
    "runtime_binary_sha256": {"px4": "c" * 64, "plugin": "d" * 64},
    "metadata": {"revision": "r3", "versions": [3, 12]},
}


def test_order_only_difference_is_accepted():
    other = {
        key: dict(reversed(list(value.items()))) for key, value in reversed(list(SOURCE.items()))
    }
    assert other == SOURCE and digest(other) != digest(SOURCE)
    revision = dict(
        producer_lock_hash_format="sorted-json-v1",
        producer_lock_sha256=digest(SOURCE, sorted_keys=True),
    )
    freezer().verify_source_lock(other, digest(other), revision)


@pytest.mark.parametrize(
    "damage", ["source", "binary", "missing", "extra", "metadata", "list_order"]
)
def test_content_changes_are_rejected(damage):
    changed = copy.deepcopy(SOURCE)
    if damage == "source":
        changed["local_sources"]["sim/a.py"] = "0" * 64
    elif damage == "binary":
        changed["runtime_binary_sha256"]["px4"] = "0" * 64
    elif damage == "missing":
        del changed["local_sources"]["sim/b.py"]
    elif damage == "extra":
        changed["local_sources"]["sim/c.py"] = "f" * 64
    elif damage == "metadata":
        changed["metadata"]["revision"] = "r4"
    else:
        changed["metadata"]["versions"].reverse()
    revision = dict(
        producer_lock_hash_format="sorted-json-v1",
        producer_lock_sha256=digest(SOURCE, sorted_keys=True),
    )
    with pytest.raises(ValueError, match="source-lock"):
        freezer().verify_source_lock(changed, digest(changed), revision)


def test_legacy_revision_still_requires_original_bytes_hash():
    revision = dict(producer_lock_sha256=digest(SOURCE))
    freezer().verify_source_lock(SOURCE, digest(SOURCE), revision)
    with pytest.raises(ValueError, match="source-lock"):
        freezer().verify_source_lock(SOURCE, "0" * 64, revision)


def test_unknown_hash_format_is_rejected():
    with pytest.raises(ValueError, match="format"):
        freezer().verify_source_lock(
            SOURCE,
            digest(SOURCE),
            dict(producer_lock_hash_format="skip", producer_lock_sha256=digest(SOURCE)),
        )
