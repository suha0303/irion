"""Admission must distinguish reclaimable read cache from real RAM pressure."""

import json
from pathlib import Path

import pytest

from execution import common
from learning import lifecycle

GIB = 1024**3


def fixture(tmp_path, version=2, host=128, mount_root="/"):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemAvailable: {host * 1024**2} kB\n")
    mount = tmp_path / "memory mount"
    mount.mkdir()
    group = mount / "job/step"
    group.mkdir(parents=True)
    member = str(Path(mount_root) / "job/step")
    membership = f"0::{member}\n" if version == 2 else f"7:memory:{member}\n"
    (proc / "self/cgroup").write_text(membership)
    encoded = str(mount).replace(" ", "\\040")
    suffix = "cgroup2 cgroup2 rw" if version == 2 else "cgroup cgroup rw,memory"
    (proc / "self/mountinfo").write_text(f"1 0 0:1 {mount_root} {encoded} rw - {suffix}\n")
    return proc, mount, group


def counters(
    group,
    version=2,
    limit=64,
    used=63,
    inactive=0,
    file=0,
    dirty=0,
    writeback=0,
    shmem=0,
    hierarchy="1",
):
    maximum, current = (
        ("memory.max", "memory.current")
        if version == 2
        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    )
    (group / maximum).write_text("max" if limit is None else str(limit * GIB))
    (group / current).write_text(str(used * GIB))
    if version == 2:
        values = dict(
            inactive_file=inactive,
            file=file,
            file_dirty=dirty,
            file_writeback=writeback,
            shmem=shmem,
        )
    else:
        values = dict(
            inactive_file=inactive, cache=file, dirty=dirty, writeback=writeback, shmem=shmem
        )
        values.update({"total_" + k: v for k, v in list(values.items())})
        (group / "memory.use_hierarchy").write_text(hierarchy)
    (group / "memory.stat").write_text("".join(f"{k} {v * GIB}\n" for k, v in values.items()))


def test_inactive_read_cache_preserves_64g_limit_without_false_1g_headroom(tmp_path):
    proc, _, group = fixture(tmp_path)
    counters(group, inactive=59, file=60)
    result = common.memory_snapshot(proc)
    assert result["errors"] == []
    assert result["available_bytes"] == 60 * GIB
    row = result["cgroups"][0]
    assert row["raw_headroom_bytes"] == GIB
    assert row["credited_cache_bytes"] == 59 * GIB
    assert result["available_bytes"] <= row["limit_bytes"]


def test_dirty_shared_and_writeback_are_not_credited(tmp_path):
    proc, _, group = fixture(tmp_path)
    counters(group, inactive=59, file=62, dirty=10, writeback=8, shmem=6)
    result = common.memory_snapshot(proc)
    assert result["available_bytes"] == 36 * GIB


def test_anonymous_pressure_is_not_excused_by_file_cache_policy(tmp_path):
    proc, _, group = fixture(tmp_path)
    counters(group, inactive=0, file=1)
    result = common.memory_snapshot(proc)
    assert result["available_bytes"] == GIB
    assert result["cgroups"][0]["credited_cache_bytes"] == 0


@pytest.mark.parametrize("host,parent_limit,expected", [(128, 20, 10), (8, 100, 8)])
def test_host_and_every_parent_cap_still_apply(tmp_path, host, parent_limit, expected):
    proc, _, group = fixture(tmp_path, host=host)
    counters(group, inactive=59, file=60)
    counters(group.parent, limit=parent_limit, used=10)
    result = common.memory_snapshot(proc)
    assert result["available_bytes"] == expected * GIB
    assert len(result["cgroups"]) == 2


def test_rebased_mount_root_and_unlimited_leaf_still_use_parent(tmp_path):
    proc, _, group = fixture(tmp_path, mount_root="/slurm")
    counters(group, limit=None, used=1)
    counters(group.parent, limit=16, used=10)
    result = common.memory_snapshot(proc)
    assert result["errors"] == []
    assert result["available_bytes"] == 6 * GIB


@pytest.mark.parametrize(
    "change", ["absent", "missing_counter", "malformed", "negative", "duplicate"]
)
def test_missing_or_invalid_statistics_never_assume_reclaimable_cache(tmp_path, change):
    proc, _, group = fixture(tmp_path)
    counters(group, inactive=59, file=60)
    path = group / "memory.stat"
    if change == "absent":
        path.unlink()
    elif change == "missing_counter":
        path.write_text("inactive_file 60000000000\nfile 60000000000\n")
    elif change == "malformed":
        path.write_text(path.read_text() + "bad not_an_int\n")
    elif change == "negative":
        path.write_text(path.read_text().replace("shmem 0", "shmem -1"))
    else:
        path.write_text(path.read_text() + "shmem 0\n")
    result = common.memory_snapshot(proc)
    assert result["available_bytes"] == GIB
    assert result["cgroups"][0]["credited_cache_bytes"] == 0
    assert "stats_error" in result["cgroups"][0]


def test_v1_hierarchical_usage_uses_total_statistics(tmp_path):
    proc, _, group = fixture(tmp_path, version=1)
    counters(group, version=1, inactive=59, file=60)
    path = group / "memory.stat"
    path.write_text(
        path.read_text().replace("inactive_file " + str(59 * GIB) + "\n", "inactive_file 0\n", 1)
    )
    result = common.memory_snapshot(proc)
    assert result["available_bytes"] == 60 * GIB
    assert result["cgroups"][0]["stats_scope"] == "hierarchical"


def test_v1_nonhierarchical_usage_never_credits_child_cache(tmp_path):
    proc, _, group = fixture(tmp_path, version=1)
    counters(group, version=1, hierarchy="0")
    path = group / "memory.stat"
    path.write_text(
        path.read_text()
        .replace("total_inactive_file 0", f"total_inactive_file {60 * GIB}")
        .replace("total_cache 0", f"total_cache {60 * GIB}")
    )
    result = common.memory_snapshot(proc)
    assert result["available_bytes"] == GIB
    assert result["cgroups"][0]["stats_scope"] == "local"


def test_unknown_v1_hierarchy_gives_no_cache_credit(tmp_path):
    proc, _, group = fixture(tmp_path, version=1)
    counters(group, version=1, inactive=59, file=60)
    (group / "memory.use_hierarchy").unlink()
    assert common.memory_snapshot(proc)["available_bytes"] == GIB


def test_unreadable_limit_pair_is_not_unrestricted_host_memory(tmp_path):
    proc, _, group = fixture(tmp_path)
    counters(group)
    (group / "memory.current").unlink()
    result = common.memory_snapshot(proc)
    assert result["available_bytes"] == 0 and result["errors"]


def test_cache_credit_never_exceeds_usage_or_makes_headroom_exceed_limit(tmp_path):
    proc, _, group = fixture(tmp_path)
    counters(group, used=2, inactive=1000, file=1000)
    result = common.memory_snapshot(proc)
    assert result["cgroups"][0]["credited_cache_bytes"] == 2 * GIB
    assert result["available_bytes"] == 64 * GIB


def test_progress_failure_saves_the_same_snapshot_once(tmp_path, monkeypatch):
    calls = []
    snapshot = {"available_bytes": GIB - 1, "node_available_bytes": 100 * GIB, "cgroups": []}
    monkeypatch.setattr(common, "memory_snapshot", lambda: calls.append(1) or snapshot)
    progress = common.Progress(tmp_path / "progress.json")
    with pytest.raises(MemoryError):
        progress.check()
    proof = common.read(tmp_path / "progress-memory-failure.json")
    assert proof["snapshot"] == snapshot and proof["required_bytes"] == GIB
    assert proof["passed"] is False and len(calls) == 1


def test_lifecycle_rejection_records_snapshot_before_failing(tmp_path, monkeypatch):
    cfg = {
        "revision": lifecycle.comparison.REVISION,
        "wall_hours": 48,
        "memory": {"estimated_host_peak_bytes": 48 * GIB},
        "upstream_files": {},
    }
    common.write(tmp_path / "input/inventory.json", {})
    monkeypatch.setattr(
        lifecycle, "paths", lambda root: (root, cfg, root / "upstream", root / "compute")
    )
    monkeypatch.setattr(lifecycle, "verify", lambda *a: None)
    monkeypatch.setattr(lifecycle, "check", lambda *a, **k: {"fixture": True})
    calls = []
    snapshot = {"available_bytes": 47 * GIB, "node_available_bytes": 100 * GIB, "cgroups": []}
    monkeypatch.setattr(lifecycle, "memory_snapshot", lambda: calls.append(1) or snapshot)
    monkeypatch.setattr(lifecycle.signal, "signal", lambda *a: None)
    monkeypatch.setattr(
        lifecycle, "Telemetry", lambda *a: pytest.fail("No GPU benchmark or training")
    )
    with pytest.raises(MemoryError, match="available"):
        lifecycle.fit_worker(tmp_path)
    proof = json.loads((tmp_path / "compute/host-memory-admission.json").read_text())
    assert proof["snapshot"] == snapshot and proof["required_bytes"] == 49 * GIB
    assert proof["passed"] is False and len(calls) == 1
