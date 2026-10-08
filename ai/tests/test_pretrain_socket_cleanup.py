"""Real temporary UNIX sockets; no PX4, flight, GPU or service is started."""

import fcntl
import os
import socket
import subprocess
import sys

import pytest

from pretrain import socket_cleanup as cleanup
from pretrain.common import read, write


@pytest.fixture
def layout(tmp_path, monkeypatch):
    root = tmp_path / "root"
    home = root / "Data/operations/pretrain"
    task = home / "tasks/fixture"
    (task / "operator").mkdir(parents=True)
    (root / "Data/operations").mkdir(parents=True, exist_ok=True)
    (root / "simulation/.cache/instances").mkdir(parents=True)
    sockets = tmp_path / "sockets"
    sockets.mkdir()
    monkeypatch.setattr(cleanup, "ROOT", root)
    monkeypatch.setattr(cleanup, "HOME", home)
    monkeypatch.setattr(cleanup, "SOCKET_ROOT", sockets)
    return root, home, task, sockets


@pytest.fixture
def owned(layout):
    root, _, task, sockets = layout
    run = root / "runs" / ("a" * 32)
    (run / "px4").mkdir(parents=True)
    path = sockets / "px4-sock-0"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import socket,sys; s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[1]); print('ready',flush=True); sys.stdin.read()",
            str(path),
        ],
        cwd=run / "px4",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        write(
            run / "run.json",
            dict(run_id=run.name, instance=0, owned_process_groups=dict(px4=child.pid)),
        )
        receipt = task / "collection/execution/case/socket-claim.json"
        cleanup.claim(run, 0, receipt)
        assert receipt.exists()
        data = read(receipt)
        # Give the isolated fixture the service binding normally provided by systemd.
        data["cgroup"] = "0::/test/x500-pretrain-fixture.service\n"
        write(receipt, data)
        yield child, path, receipt
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()


def test_live_owner_preserved_then_forced_exit_cleaned(layout, owned):
    child, path, receipt = owned
    data = read(receipt)
    with pytest.raises(ValueError, match="still running"):
        cleanup.release(data, "x500-pretrain-fixture.service")
    assert path.exists()
    child.kill()
    child.wait(timeout=5)
    assert cleanup.release(data, "x500-pretrain-fixture.service") == "removed_owned_stale_socket"
    assert not path.exists()
    assert cleanup.release(data, "x500-pretrain-fixture.service") == "already_removed"


@pytest.mark.parametrize("replacement", ["socket", "file", "symlink"])
def test_replaced_path_preserved(layout, owned, replacement):
    child, path, receipt = owned
    child.kill()
    child.wait(timeout=5)
    path.unlink()
    if replacement == "socket":
        with socket.socket(socket.AF_UNIX) as new:
            new.bind(str(path))
    elif replacement == "file":
        path.write_text("must survive")
    else:
        target = path.with_name("target")
        target.write_text("must survive")
        path.symlink_to(target)
    with pytest.raises(ValueError):
        cleanup.release(read(receipt), "x500-pretrain-fixture.service")
    assert path.exists()


def test_other_service_receipt_preserved(layout, owned):
    child, path, receipt = owned
    child.kill()
    child.wait(timeout=5)
    with pytest.raises(ValueError, match="different service"):
        cleanup.release(read(receipt), "x500-pretrain-other.service")
    assert path.exists()


def test_stop_hook_keeps_old_state_and_records_oom(layout, owned, monkeypatch):
    _, _, task, sockets = layout
    child, path, _ = owned
    child.kill()
    child.wait(timeout=5)
    write(
        task / "operator/state.json", dict(status="RUNNING", stage="collection", done=7, total=50)
    )
    before = (task / "operator/state.json").read_bytes()
    unknown = sockets / "px4-sock-1"
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(unknown))
    monkeypatch.setenv("SERVICE_RESULT", "oom-kill")
    result = cleanup.cleanup(task)
    assert not path.exists() and unknown.exists()
    assert result["status"] == "CLEANUP_REQUIRES_REVIEW"
    assert result["actions"][0]["result"] == "removed_owned_stale_socket"
    assert (task / "operator/state-before-stop.json").read_bytes() == before
    state = read(task / "operator/state.json")
    assert state["status"] == "FAILED_STOPPED" and state["done"] == 7
    assert "oom-kill" in state["reason"] and not state["next_stage_authorized"]


def test_locked_project_preserves_socket_and_state(layout, owned):
    _, home, task, _ = layout
    child, path, _ = owned
    child.kill()
    child.wait(timeout=5)
    write(task / "operator/state.json", dict(status="RUNNING"))
    before = (task / "operator/state.json").read_bytes()
    with (home / "pretrain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = cleanup.cleanup(task)
    assert result["errors"] and path.exists()
    assert (task / "operator/state.json").read_bytes() == before


def test_bound_socket_preserved_even_if_owner_check_says_dead(layout, owned, monkeypatch):
    _, path, receipt = owned
    monkeypatch.setattr(cleanup, "process", lambda pid: None)
    with pytest.raises(ValueError, match="still bound"):
        cleanup.release(read(receipt), "x500-pretrain-fixture.service")
    assert path.exists()


def test_claim_tolerates_closed_enumeration_fd(layout, monkeypatch):
    root, _, task, sockets = layout
    run = root / "runs" / ("b" * 32)
    (run / "px4").mkdir(parents=True)
    monkeypatch.chdir(run / "px4")
    write(
        run / "run.json",
        dict(run_id=run.name, instance=0, owned_process_groups=dict(px4=os.getpid())),
    )
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(sockets / "px4-sock-0"))
        receipt = task / "collection/execution/self/socket-claim.json"
        cleanup.claim(run, 0, receipt)
        assert read(receipt)["owner"]["pid"] == os.getpid()


def test_reused_lane_cleans_latest_claim_before_old_receipt(layout, owned):
    _, _, task, _ = layout
    child, path, receipt = owned
    child.kill()
    child.wait(timeout=5)
    old = read(receipt)
    path.unlink()
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(path))
    current = dict(old, identity=cleanup.socket_identity(path))
    latest = task / "collection/execution/new-case/socket-claim.json"
    write(latest, current)
    os.utime(receipt, ns=(1, 1))
    os.utime(latest, ns=(2, 2))
    result = cleanup.cleanup(task)
    assert not path.exists() and not result["errors"]
    assert [item["result"] for item in result["actions"]] == [
        "removed_owned_stale_socket",
        "already_removed",
    ]
