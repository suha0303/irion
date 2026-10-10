"""One-shot systemd stop hook; remove only durably claimed, dead PX4 sockets."""

import fcntl
import os
import stat
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from .common import HOME, ROOT, now, read, service_unit, write

SOCKET_ROOT = Path("/tmp")


def process(pid):
    try:
        folder = Path(f"/proc/{pid}")
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, start=fields[19], state=fields[0], uid=folder.stat().st_uid)
    except FileNotFoundError:
        return None


def socket_identity(path):
    info = path.lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("Not a socket owned by this user")
    return dict(device=info.st_dev, inode=info.st_ino, uid=info.st_uid, ctime_ns=info.st_ctime_ns)


def bound_inodes(path):
    return {
        fields[6]
        for line in Path("/proc/net/unix").read_text().splitlines()[1:]
        if len(fields := line.split()) >= 8 and fields[7] == str(path)
    }


def claim(run, instance, destination):
    """Called by the existing collector check; no new monitor or process."""
    destination = Path(destination)
    if destination.exists():
        return
    path = SOCKET_ROOT / f"px4-sock-{instance}"
    try:
        meta = read(run / "run.json")
        pid = meta.get("owned_process_groups", {}).get("px4")
        if not pid or meta.get("instance") != instance or meta.get("run_id") != run.name:
            return
        owner = process(pid)
        if not owner or owner["uid"] != os.getuid() or owner["state"] == "Z":
            return
        group = Path(f"/proc/{pid}/cgroup").read_text()
        if group != Path("/proc/self/cgroup").read_text():
            return
        if Path(f"/proc/{pid}/cwd").resolve() != run / "px4":
            return
        identity = socket_identity(path)
        inodes = bound_inodes(path)
        fds = set()
        for fd in Path(f"/proc/{pid}/fd").iterdir():
            try:
                fds.add(os.readlink(fd))
            except FileNotFoundError:
                continue  # File descriptors can close while enumerating a live process.
        if not any(f"socket:[{inode}]" in fds for inode in inodes):
            return
        current = process(pid)
        if (
            socket_identity(path) != identity
            or not current
            or any(current[key] != owner[key] for key in ("pid", "start", "uid"))
        ):
            return
    except FileNotFoundError:
        return  # PX4 is still starting or has already exited; never guess ownership.
    write(
        destination,
        dict(
            path=str(path),
            instance=instance,
            run_id=run.name,
            identity=identity,
            owner=owner,
            cgroup=group,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            recorded_utc=now(),
        ),
    )


def release(receipt, unit):
    instance = receipt["instance"]
    if type(instance) is not int or not 0 <= instance <= 15:
        raise ValueError("Invalid instance")
    path = SOCKET_ROOT / f"px4-sock-{instance}"
    if receipt["path"] != str(path) or not any(
        line.startswith("0::") and line[3:].endswith("/" + unit)
        for line in receipt["cgroup"].splitlines()
    ):
        raise ValueError("Socket belongs to a different service")
    if receipt["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise ValueError("Boot identity changed")
    if not os.path.lexists(path):
        return "already_removed"
    owner = process(receipt["owner"]["pid"])
    if (
        owner
        and owner["state"] != "Z"
        and all(owner[key] == receipt["owner"][key] for key in ("pid", "start", "uid"))
    ):
        raise ValueError("Owner still running")
    if socket_identity(path) != receipt["identity"]:
        raise ValueError("Socket identity changed")
    if bound_inodes(path):
        raise ValueError("Socket still bound")
    if socket_identity(path) != receipt["identity"]:
        raise ValueError("Socket replaced during check")
    path.unlink()
    return "removed_owned_stale_socket"


def cleanup(task):
    task = Path(task).resolve()
    if task.parent != HOME / "tasks":
        raise ValueError("Registered pretrain task required")
    unit = service_unit(task)
    report = dict(
        updated_utc=now(),
        service_result=os.environ.get("SERVICE_RESULT", "unknown"),
        actions=[],
        errors=[],
        retry=False,
    )
    locked = False
    with ExitStack() as stack:
        try:
            for path in (
                HOME / "pretrain.lock",
                ROOT / "Data/operations/collection.lock",
            ):
                handle = stack.enter_context(path.open("a"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            # A lane can be reused within one campaign. Clean its latest claim
            # first so older receipts see an already-removed pathname.
            receipts = sorted(
                task.glob("*/execution/*/socket-claim.json"),
                key=lambda path: path.lstat().st_mtime_ns,
                reverse=True,
            )
            for source in receipts:
                try:
                    if any(p.is_symlink() for p in (source, *source.parents)):
                        raise ValueError("Symlink receipt preserved")
                    receipt = read(source)
                    instance = receipt["instance"]
                    if type(instance) is not int or not 0 <= instance <= 15:
                        raise ValueError("Invalid instance")
                    with (ROOT / f"simulation/.cache/instances/{instance}.lock").open("a") as lease:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        result = release(receipt, unit)
                    report["actions"].append(
                        dict(receipt=str(source), path=receipt["path"], result=result)
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    report["errors"].append(dict(receipt=str(source), reason=str(exc)))
        except OSError as exc:
            report["errors"].append(dict(reason="Project lock unavailable; preserved: " + str(exc)))
        claimed = {action["path"] for action in report["actions"]}
        report["remaining_unremoved_sockets"] = [
            str(path) for i in range(16) if os.path.lexists(path := SOCKET_ROOT / f"px4-sock-{i}")
        ]
        report["status"] = (
            "CLEANUP_REQUIRES_REVIEW"
            if report["errors"]
            or any(p not in claimed for p in report["remaining_unremoved_sockets"])
            else "CLEANUP_COMPLETE"
        )
        write(task / "operator/socket-cleanup.json", report)
        state_path = task / "operator/state.json"
        if locked and state_path.exists():
            state = read(state_path)
            if state.get("status") in {"RUNNING", "DRAINING"}:
                backup = task / "operator/state-before-stop.json"
                if not backup.exists():
                    backup.write_bytes(state_path.read_bytes())
                state.update(
                    status="FAILED_STOPPED",
                    finished=time.time(),
                    updated_utc=now(),
                    reason="Service exited: " + report["service_result"],
                    next_stage_authorized=False,
                    active=[],
                )
                write(state_path, state)
    return report


if __name__ == "__main__":
    cleanup(sys.argv[1])
