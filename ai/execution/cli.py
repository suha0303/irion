"""Operator CLI: one finite local controller, independent Slurm job, read-only display."""

import argparse
import fcntl
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .common import (
    HANDOFF,
    HOME,
    REMOTE_BASE,
    ROOT,
    STORE,
    Progress,
    config_check,
    python_for,
    read,
    runtime_profile,
    sha,
    verify,
    write,
)


def connection():
    selected = runtime_profile(required=True)["ssh"]
    if not selected:
        raise ValueError("SSH connection must be explicitly selected in the profile")
    return selected


def destination():
    selected = connection()
    return (selected["user"] + "@" if selected.get("user") else "") + selected["host"]


def ssh_argv():
    return [
        "ssh",
        "-p",
        str(connection()["port"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        destination(),
    ]


def ssh(command, timeout=60):
    return subprocess.check_output([*ssh_argv(), command], text=True, timeout=timeout).strip()


def remote(root, action):
    command = (
        "PYTHONPATH="
        + shlex.quote(str(root) + "/ai:" + str(root) + "/ai/src")
        + " "
        + shlex.join(
            [python_for(required=True), "-B", "-m", "execution.remote", action, "--root", str(root)]
        )
    )
    return ssh(command)


def connect():
    # Human terminal only. The selected tmux session survives SSH disconnection.
    profile = runtime_profile(required=True)
    subprocess.run(
        [
            "ssh",
            "-tt",
            "-o",
            "StrictHostKeyChecking=yes",
            "-p",
            str(connection()["port"]),
            destination(),
            "TERM=xterm-256color tmux new-session -A -s drone-ai -c "
            + shlex.quote(profile["project_root"]),
        ],
        check=True,
    )


def prepared_input(cfg):
    receipt = read(HOME / "prepared-input.json")
    if receipt["plan_sha256"] != sha(HOME / "plan.json"):
        raise ValueError("Prepared input belongs to a different plan")
    root = Path(receipt["root"])
    output = Path(receipt["output"])
    if not root.is_relative_to(REMOTE_BASE) or root.name != "input":
        raise ValueError("Prepared server input must be inside the selected managed root")
    if not output.is_relative_to(STORE / "gpu104"):
        raise ValueError("Prepared local input must use external storage")
    if read(output / "input/config.json") != cfg:
        raise ValueError("Prepared input configuration changed")
    if sha(output / "input/inventory.json") != receipt["inventory_sha256"]:
        raise ValueError("Prepared input inventory changed")
    if (output / "execution.claim").exists():
        raise ValueError("Prepared attempt already claimed; use recover, never resubmit")
    return dict(
        root=str(root),
        output=str(output),
        prepared=True,
        inventory_sha256=receipt["inventory_sha256"],
    )


def plan():
    path = HOME / "plan.json"
    if sha(path) != (HOME / "plan.sha256").read_text().strip():
        raise ValueError("Plan changed; prepare a reviewed revision")
    cfg = read(path)
    config_check(cfg)
    verify(ROOT, cfg["source_sha256"])
    if sha(HANDOFF) != cfg["handoff_sha256"]:
        raise ValueError("Prepared handoff changed")
    return cfg


def launch(action="run", existing=None):
    cfg = plan()
    HOME.mkdir(parents=True, exist_ok=True)
    with (HOME / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if action == "run":
        name = time.strftime("gpu104-%Y%m%d-%H%M%S")
        task = HOME / "tasks" / name
        task.mkdir(parents=True, exist_ok=False)
        write(task / "config.json", cfg)
        metadata = (
            prepared_input(cfg)
            if (HOME / "prepared-input.json").exists()
            else dict(root=REMOTE_BASE + "/" + name + "/input", output=str(STORE / "gpu104" / name))
        )
        write(task / "remote.json", metadata)
    else:
        task = Path(existing).resolve()
        if not task.is_relative_to(HOME / "tasks") or not (task / "remote.json").exists():
            raise ValueError("Existing managed task required")
        # Only a result recovery controller is started; it never submits a Slurm job.
        name = task.name + "-recover-" + time.strftime("%H%M%S")
    state = task / "controller.json"
    if action == "run":
        write(state, dict(status="START_REQUESTED", action=action, updated=time.time()))
    unit = "x500-gpu104-" + name
    cmd = [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--property=Type=exec",
        "--property=Restart=no",
        "--property=KillMode=control-group",
        "--property=TimeoutStopSec=30",
        "--property=StandardOutput=append:" + str(task / "controller.log"),
        "--property=StandardError=append:" + str(task / "controller.log"),
        "--setenv=PYTHONPATH=" + str(ROOT / "ai") + ":" + str(ROOT / "ai/src"),
        "--setenv=PYTHONDONTWRITEBYTECODE=1",
        "--setenv=OPENBLAS_NUM_THREADS=2",
        "--setenv=OMP_NUM_THREADS=2",
        "--setenv=MKL_NUM_THREADS=2",
        "--working-directory=" + str(ROOT),
        str(ROOT / "ai/.venv/bin/python"),
        "-B",
        "-m",
        "execution.cli",
        "_worker",
        "--task",
        str(task),
        "--action",
        action,
    ]
    subprocess.run(cmd, check=True)
    write(task / "service.json", dict(unit=unit, argv=cmd, automatic_restart=False))
    write(HOME / "latest.json", dict(task=str(task)))
    print("실행 요청:", task)
    print("화면: " + shlex.join([str(ROOT / "ai/drone-ai"), "watch", "--root", str(task)]))
    return task


def upload(root, workspace, progress):
    job = str(root).rsplit("/", 1)[0]
    # A fresh path and no overwrite: existing remote evidence is never replaced.
    ssh("umask 077; mkdir -p " + shlex.quote(REMOTE_BASE) + " && mkdir " + shlex.quote(job))
    needed = 2 * sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file()) + 20_000_000_000
    available = int(
        ssh(
            shlex.join(
                [
                    python_for(required=True),
                    "-B",
                    "-c",
                    "import shutil; print(shutil.disk_usage(" + repr(job) + ").free)",
                ]
            )
        )
    )
    if available < needed:
        raise ValueError(f"Remote disk insufficient: {available} < {needed} bytes")
    archive = workspace.parent / "input.tar.gz"
    files = list(workspace.rglob("*"))
    progress.update(stage="packing", done=0, total=len(files))
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for index, path in enumerate(files):
            progress.check()
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(workspace)), recursive=False)
            progress.update(force=False, done=index + 1, total=len(files))
    progress.update(
        stage="upload",
        done=0,
        total=archive.stat().st_size,
        progress_kind="transfer size; no measured byte percentage",
    )
    subprocess.run(
        [
            "scp",
            "-P",
            str(connection()["port"]),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            str(archive),
            destination() + ":" + job + "/input.tar.gz",
        ],
        check=True,
        timeout=14400,
    )
    digest = sha(archive)
    check = ssh("sha256sum " + shlex.quote(job + "/input.tar.gz"), timeout=600).split()[0]
    if check != digest:
        raise ValueError("Remote archive hash mismatch")
    # This archive was generated solely from validated regular relative paths above.
    ssh(
        "mkdir "
        + shlex.quote(str(root))
        + " && tar -xzf "
        + shlex.quote(job + "/input.tar.gz")
        + " -C "
        + shlex.quote(str(root)),
        timeout=600,
    )
    progress.update(done=archive.stat().st_size)


def wait_and_fetch(task, cfg, meta, progress):
    root = meta["root"]
    output = Path(meta["output"])
    deadline = time.monotonic() + (cfg["queue_hours"] + cfg["wall_hours"] + 1) * 3600
    accounting_gaps = 0
    while True:
        progress.check()
        snap = json.loads(remote(root, "snapshot"))
        write(task / "remote-snapshot.json", snap)
        progress.update(
            stage="remote_job",
            job_id=snap["job_id"],
            remote_progress=snap["progress"],
            queue=snap["queue"],
        )
        if snap.get("released"):
            write(output / "release.json", snap)
            break
        if snap.get("status") == "UNKNOWN_ACCOUNTING":
            accounting_gaps += 1
            if accounting_gaps >= 5:
                raise RuntimeError(
                    "Job not in queue but accounting remains unknown; no submission retry. Use recover later."
                )
        else:
            accounting_gaps = 0
        if time.monotonic() > deadline:
            raise TimeoutError(
                "Controller wait budget exhausted; remote job is not cancelled. Use recover."
            )
        time.sleep(30)  # Program only; no GPT/API/LLM calls.
    progress.update(stage="download", remote_progress=None)
    download = output / "download"
    if download.exists():
        # Recovery may reuse only an already verified complete download.
        seal = read(download / "result-seal.json")
        verify(download, seal["files"])
    else:
        pending = output / ("download-partial-" + str(int(time.time())))
        pending.mkdir()
        subprocess.run(
            [
                "scp",
                "-r",
                "-P",
                str(connection()["port"]),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                destination() + ":" + root.rsplit("/", 1)[0] + "/output/.",
                str(pending),
            ],
            check=True,
            timeout=14400,
        )
        if snap.get("slurm_state") != "COMPLETED" or snap.get("exit_code") != "0:0":
            raise RuntimeError(
                "GPU job failed; evidence downloaded to " + str(pending) + "; no automatic retry"
            )
        seal = read(pending / "result-seal.json")
        verify(pending, seal["files"])
        pending.rename(download)
    seal = read(download / "result-seal.json")
    if seal.get("config_sha256") != sha(output / "input/config.json"):
        raise ValueError("Downloaded results belong to different input configuration")
    if snap.get("slurm_state") != "COMPLETED" or snap.get("exit_code") != "0:0":
        raise ValueError("Job not successfully released")
    if (output / "evaluation").exists():
        raise ValueError(
            "Evaluation output exists; preserve it and do not repeat evaluation automatically"
        )
    if snap.get("remaining_own_job_ids"):
        raise ValueError("Other own jobs remain; preserve results pending review")
    verify(ROOT, cfg["source_sha256"])
    from .evaluate import evaluate

    evaluate(cfg, download, output / "evaluation", progress)


def worker(task, action):
    task = Path(task).resolve()
    HOME.mkdir(parents=True, exist_ok=True)
    # A rejected duplicate must never overwrite the active controller's state.
    with (HOME / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _worker_locked(task, action)


def _worker_locked(task, action):
    progress = Progress(task / "controller.json", hours=48)
    try:
        cfg = read(task / "config.json")
        config_check(cfg)
        if cfg != plan():
            raise ValueError("Task/plan binding changed")
        meta = read(task / "remote.json")
        output = Path(meta["output"])
        from x500_storage import guard

        guard(output)
        if action == "run":
            with (task / "started.claim").open("x") as f:
                f.write("No automatic resubmission\n")
            progress.update(stage="login_preflight")
            queue = ssh('squeue -h -u "$(id -u)" -o "%A %P %j %T %N"')
            if queue:
                raise ValueError("기존 작업이 남아 있어 제출하지 않았습니다: " + queue)
            if not meta.get("prepared"):
                output.mkdir(parents=True, exist_ok=False)
            if shutil.disk_usage(output).free < 50_000_000_000:
                raise ValueError("Selected output free disk below50GB")
            from .prepare import admission, stage

            admission(Path(cfg["handoff"]))
            if meta.get("prepared"):
                if read(output / "input/config.json") != cfg:
                    raise ValueError("Prepared configuration changed")
                if sha(output / "input/inventory.json") != meta["inventory_sha256"]:
                    raise ValueError("Prepared inventory changed")
                verify(output / "input", read(output / "input/inventory.json"))
                progress.update(stage="verify_prepared_remote")
                ready = json.loads(remote(meta["root"], "verify"))
                if ready["inventory_sha256"] != meta["inventory_sha256"]:
                    raise ValueError("Remote prepared inventory changed")
            else:
                stage(cfg, output / "input", progress)
                upload(meta["root"], output / "input", progress)
            with (output / "execution.claim").open("x") as claim:
                claim.write(str(task) + "\n")
            progress.update(stage="submit_once")
            submission = json.loads(remote(meta["root"], "submit"))
            write(task / "submission.json", submission)
        elif action != "recover":
            raise ValueError("Unknown action")
        wait_and_fetch(task, cfg, meta, progress)
    except BaseException as exc:
        # Never cancel an independently running Slurm job on a local transport failure.
        progress.update(
            status="FAILED_OR_UNKNOWN_STOPPED",
            reason=f"{type(exc).__name__}: {exc}",
            remote_job_cancelled=False,
            automatic_retry=False,
            recovery="recover polls/fetches the same remote job only; never submits",
        )
        raise


def resolve_task(value=None):
    return Path(value) if value else Path(read(HOME / "latest.json")["task"])


def duration(seconds):
    if seconds is None:
        return "미확인"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def render(task):
    state = read(task / "controller.json")
    remote_state = state.get("remote_progress") or {}
    current = remote_state or state
    done, total = current.get("done", 0), current.get("total", 0)
    percentage = 100 * done / total if total else None
    stale = time.time() - state.get("updated", 0) > 120 and state.get("status") == "RUNNING"
    lines = [
        "X500 / GPU104 / LSTM + Transformer + Persistence + Ridge / seed17",
        f"상태 {state.get('status')} {'(기록 오래됨: 현재 실행 상태 미확인)' if stale else ''}",
        f"job {state.get('job_id', '-')} · p02 / DIS01 / GPU1 · 경과 {duration(state.get('elapsed_s'))}",
        f"단계 {current.get('stage', '-')} · 모델 {current.get('model') or '-'}",
    ]
    if current.get("stage") == "upload":
        lines.append(f"전송 대상 {total / 1024**3:.2f} GiB · 전송 중 바이트 진행률은 미측정")
    elif percentage is not None:
        label = (
            "완료 건수"
            if current.get("stage") not in {"training", "validation"}
            else "현재 학습/검증 진행"
        )
        lines.append(f"{label} {done:,}/{total:,} ({percentage:.1f}%) — 전체 시간 진행률 아님")
    if current.get("epoch") and current.get("stage") in {"training", "validation", "checkpoint"}:
        lines.append(
            f"epoch {current['epoch']}/{current['epochs']} · batch {current.get('batch', '-')}/{current.get('batches', '-')}"
        )
    lines.append(
        "현재 단계 잔여 "
        + duration(current.get("phase_eta_s"))
        + " · 다른 모델/보정/평가 시간 제외"
    )
    if current.get("train_loss") is not None and current.get("stage") in {
        "training",
        "validation",
        "checkpoint",
    }:
        lines.append(
            f"train loss {current['train_loss']:.6g} · validation {current.get('validation_loss', '미측정')} · best epoch {current.get('best_epoch', '-')}"
        )
    if current.get("steps_per_s") is not None:
        lines.append(f"학습 처리율 {current['steps_per_s']:.2f} step/s")
    if current.get("gpu_reserved_bytes") is not None:
        lines.append(f"PyTorch GPU 예약 메모리 {current['gpu_reserved_bytes'] / 1024**3:.2f} GiB")
    if state.get("queue"):
        lines.append("Slurm " + state["queue"])
    if state.get("reason"):
        lines.append("사유 " + state["reason"])
    if remote_state.get("reason"):
        lines.append("서버 사유 " + remote_state["reason"])
    if state.get("report"):
        lines.append("결과 " + state["report"])
    lines += [
        "로그 " + str(task / "controller.log"),
        "Ctrl+C: 화면만 종료 · 학습은 독립 실행 · 자동 재학습/LLM 호출 없음",
    ]
    return "\n".join(lines)


def watch(task, once=False):
    try:
        while True:
            if not once and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(render(task), flush=True)
            if once:
                return
            time.sleep(2)
    except KeyboardInterrupt:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        nargs="?",
        default="menu",
        choices=[
            "menu",
            "connect",
            "plan",
            "start",
            "watch",
            "status",
            "recover",
            "stop",
            "_worker",
        ],
    )
    p.add_argument("--task")
    p.add_argument("--action", choices=["run", "recover"], default="run")
    a = p.parse_args()
    if a.command == "menu":
        print(
            "1 학습 시작 / 2 진행 화면 / 3 기존 job 결과 인수 재개 / 4 해당 job 취소 / 5 SSH 연결 / 0 종료"
        )
        choice = input("선택: ").strip()
        a.command = {
            "1": "start",
            "2": "watch",
            "3": "recover",
            "4": "stop",
            "5": "connect",
            "0": "exit",
        }.get(choice, "invalid")
    if a.command == "connect":
        connect()
    elif a.command == "plan":
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
    elif a.command == "start":
        watch(launch())
    elif a.command == "recover":
        watch(launch("recover", resolve_task(a.task)))
    elif a.command == "stop":
        print(remote(read(resolve_task(a.task) / "remote.json")["root"], "cancel"))
    elif a.command == "_worker":
        worker(a.task, a.action)
    elif a.command in {"watch", "status"}:
        watch(resolve_task(a.task), once=a.command == "status")
    elif a.command != "exit":
        raise ValueError("잘못된 선택")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOP:", exc, file=sys.stderr)
        sys.exit(1)
