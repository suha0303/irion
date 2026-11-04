"""Finite server-side coordinator and read-only terminal. No LLM or automatic retries."""

import argparse
import fcntl
import json
import os
import pwd
import shlex
import subprocess
import sys
import time
from pathlib import Path

from execution.common import (
    SUBMIT_LOCK,
    child,
    python_for,
    read,
    runtime_profile,
    sha,
    verify,
    write,
)

from . import comparison

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}
LOCK = SUBMIT_LOCK
FILE_VISIBILITY_SECONDS = 120
FILE_VISIBILITY_INTERVAL_SECONDS = 5


def command(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.PIPE, timeout=30).strip()


def own_queue():
    return command(["squeue", "-h", "-u", str(os.getuid()), "-o", "%A|%j|%T|%N"])


def parse_snapshot(identifier, queue, accounting, owner):
    active = [line for line in queue.splitlines() if line.split("|")[0] == identifier]
    result = dict(job_id=identifier, released=False, queue=active)
    if active:
        return result
    entries = [
        line.split("|") for line in accounting.splitlines() if line.split("|")[0] == identifier
    ]
    if len(entries) == 1 and entries[0][2].split()[0] in TERMINAL:
        row = entries[0]
        if row[1] != owner:
            raise ValueError("Slurm accounting owner mismatch")
        result.update(
            released=True, slurm_state=row[2], exit_code=row[3], remaining_own_queue=queue
        )
    else:
        result["status"] = "UNKNOWN_ACCOUNTING"
    return result


def snapshot(record):
    identifier = record["job_id"]
    if not identifier.isdigit() or record["uid"] != os.getuid():
        raise ValueError("Job identity mismatch")
    queue = own_queue()
    accounting = (
        ""
        if any(line.split("|")[0] == identifier for line in queue.splitlines())
        else command(
            ["sacct", "-n", "-P", "-j", identifier, "--format=JobIDRaw,User,State,ExitCode"]
        )
    )
    return parse_snapshot(identifier, queue, accounting, pwd.getpwuid(os.getuid()).pw_name)


def submit(root, phase, cfg):
    if phase not in {"fit", "evaluate"}:
        raise ValueError("Unknown approved phase")
    if phase == "evaluate" and comparison.execution_scope(cfg) == "training_only":
        raise ValueError("Evaluation submission is not authorized for this training-only attempt")
    if comparison.is_comparison(cfg):
        comparison.config_check(cfg)
    profile = runtime_profile(root, required=True)
    if any(
        profile["slurm"].get(k) != v
        for k, v in dict(partition="p02", node="DIS01", nodes=1, ntasks=1, gpus=1).items()
    ):
        raise ValueError("Only the currently approved p02/DIS01 single-GPU allocation is allowed")
    receipt = read(root / "prepared.json")
    if sha(root / "server-profile.json") != receipt["server_profile_sha256"]:
        raise ValueError("Selected server profile changed")
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if own_queue():
            raise ValueError("Own job already pending/running; no concurrent submission")
        verify(root / "input", read(root / "input/inventory.json"))
        if phase == "evaluate":
            require_success(read(root / "release.json"))
            if (
                read(root / "compute/result-seal.json")["status"]
                != "TRAIN_CALIBRATION_INFERENCE_COMPLETE"
            ):
                raise ValueError("Complete sealed compute required before evaluation")
        with (root / (phase + "-submitted.claim")).open("x") as stream:
            stream.write("One submission; unknown response must not be retried.\n")
        script = root / (phase + "-job.sh")
        script.write_text(
            "#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH="
            + shlex.quote(str(root / "input/ai") + ":" + str(root / "input/ai/src"))
            + "\n"
            + "export DRONE_AI_PROFILE="
            + shlex.quote(str(root / "server-profile.json"))
            + "\n"
            + "export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
            + "export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2\nexec "
            + shlex.join(
                [
                    python_for(root, required=True),
                    "-B",
                    "-m",
                    "learning.lifecycle",
                    phase,
                    "--root",
                    str(root),
                ]
            )
            + "\n"
        )
        argv = [
            "sbatch",
            "--parsable",
            "--partition=p02",
            "--nodelist=DIS01",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            "--mem=" + str(cfg["memory_gib"]) + "G"
            if comparison.is_comparison(cfg)
            else "--mem=16G",
            "--time="
            + (
                str(cfg["wall_hours"] if phase == "fit" else cfg["evaluation_hours"]) + ":00:00"
                if comparison.is_comparison(cfg)
                else ("06:00:00" if phase == "fit" else "02:00:00")
            ),
            "--no-requeue",
            "--job-name=drone-ai-" + phase,
            "--output=" + str(root / (phase + "-slurm.log")),
            "--error=" + str(root / (phase + "-slurm.log")),
            "--chdir=" + str(root),
            str(script),
        ]
        write(
            root / (phase + "-submission-intent.json"),
            dict(argv=argv, time=time.time(), automatic_retry=False),
        )
        response = command(argv)
        identifier = response.split(";")[0]
        if not identifier.isdigit():
            raise ValueError("Unknown submission outcome; no retry: " + response)
        record = dict(job_id=identifier, uid=os.getuid(), phase=phase, submitted=time.time())
        write(root / (phase + "-submission.json"), record)
        return record


def require_success(release):
    if (
        not release.get("released")
        or release.get("slurm_state") != "COMPLETED"
        or release.get("exit_code") != "0:0"
    ):
        raise RuntimeError("Phase did not finish successfully: " + json.dumps(release))
    if release.get("remaining_own_queue"):
        raise RuntimeError("Another own job remains; stop before next stage")


def wait_release(root, record, state, cfg):
    deadline = time.monotonic() + (cfg["queue_hours"] + cfg["wall_hours"] + 1) * 3600
    gaps = 0
    while True:
        result = snapshot(record)
        write(root / (record["phase"] + "-snapshot.json"), result)
        state.update(updated=time.time(), job_id=record["job_id"], queue=result["queue"])
        write(root / "controller.json", state)
        if result["released"]:
            return result
        gaps = gaps + 1 if result.get("status") == "UNKNOWN_ACCOUNTING" else 0
        if gaps >= 5 or time.monotonic() > deadline:
            raise RuntimeError("Job release unconfirmed; no continuation or resubmission")
        time.sleep(30)  # Finite program wait; never calls an LLM.


def phase_files_visible(root, phase, cfg):
    """Read metadata and open expected files; leave hash checks to existing verification."""
    if phase == "fit":
        seal = read(root / "compute/result-seal.json")
        training_only = comparison.execution_scope(cfg) == "training_only"
        expected = "TRAINING_COMPLETE" if training_only else "TRAIN_CALIBRATION_INFERENCE_COMPLETE"
        if seal["status"] != expected:
            raise ValueError("Complete sealed compute required for the approved scope")
        if training_only and any(
            seal.get(key) is not False
            for key in ("calibration_started", "inference_started", "evaluation_started")
        ):
            raise ValueError("Training-only result contains unauthorized post-training phases")
        for name in seal["files"]:
            with child(root / "compute", name).open("rb"):
                pass
    elif phase == "evaluate":
        if comparison.execution_scope(cfg) == "training_only":
            raise ValueError("Evaluation result is outside this attempt's approved scope")
        evaluation = read(root / "evaluation/report.json")
        if evaluation["status"] != "REPORT_READY_STOPPED":
            raise ValueError("Final report status verification failed")
        read(root / "evaluation/metrics.json")
        for name in cfg["models"]:
            read(child(root / "compute/models", name) / "training.json")
    else:
        raise ValueError("Unknown released phase")


def wait_phase_files(root, phase, state, cfg):
    """A released allocation can precede NFS visibility; retry only ENOENT reads."""
    started = time.monotonic()
    deadline = started + FILE_VISIBILITY_SECONDS
    state.update(
        allocation_released=True,
        file_visibility_phase=phase,
        file_visibility="CHECKING",
        file_visibility_waited_s=0,
        file_visibility_limit_s=FILE_VISIBILITY_SECONDS,
        updated=time.time(),
    )
    write(root / "controller.json", state)
    while True:
        try:
            phase_files_visible(root, phase, cfg)
        except FileNotFoundError as exc:
            elapsed = time.monotonic() - started
            remaining = deadline - time.monotonic()
            state.update(
                allocation_released=True,
                file_visibility_phase=phase,
                file_visibility="WAITING" if remaining > 0 else "TIMED_OUT",
                file_visibility_waited_s=elapsed,
                file_visibility_limit_s=FILE_VISIBILITY_SECONDS,
                file_visibility_last_error=f"{type(exc).__name__}: {exc}",
                updated=time.time(),
            )
            write(root / "controller.json", state)
            if remaining <= 0:
                raise FileNotFoundError(
                    f"Released {phase} artifacts stayed unavailable for "
                    f"{FILE_VISIBILITY_SECONDS}s; no job retry or resubmission"
                ) from exc
            time.sleep(min(FILE_VISIBILITY_INTERVAL_SECONDS, remaining))
        else:
            state.update(
                allocation_released=True,
                file_visibility_phase=phase,
                file_visibility="VISIBLE",
                file_visibility_waited_s=time.monotonic() - started,
                file_visibility_limit_s=FILE_VISIBILITY_SECONDS,
                updated=time.time(),
            )
            state.pop("file_visibility_last_error", None)
            write(root / "controller.json", state)
            return


def report(root, cfg):
    metrics = read(root / "evaluation/metrics.json")
    lines = [
        "X500 " + cfg["revision"],
        "Development evaluation only; final test not used.",
        "Model | events detected/total | normal union alarms/hour | selected epoch",
    ]
    for name in cfg["models"]:
        m = metrics["models"][name]
        training = read(root / "compute/models" / name / "training.json")
        lines.append(
            f"{name} | {m['events']['detected']}/{m['events']['events']} | "
            f"{m['normal_episodes_per_hour']['union']} | {training.get('selected_epoch', 'n/a')}"
        )
    lines += [
        "",
        "Forecast errors, onset delays, coverage and unavailable cases: evaluation/metrics.json",
        f"One model seed{cfg.get('seed', 17)}. Development groups only; poor results never trigger retraining.",
        "Models: compute/models/<model>/{model.safetensors or ridge.npz, scaler.npz, model.json}",
        "Neural best.pt and last.pt are separate from exported best weights. No cross-round accumulation.",
        "REPORT_READY_STOPPED: both own allocations released; no next round authorized.",
    ]
    (root / "report.txt").write_text("\n".join(lines) + "\n")


def training_report(root, cfg):
    lines = [
        "X500 " + cfg["revision"],
        "Training and model export only; calibration, inference and development evaluation not run.",
        "Model | selected epoch | training seconds",
    ]
    for name in cfg["models"]:
        trained = read(root / "compute/models" / name / "training.json")
        lines.append(
            f"{name} | {trained.get('selected_epoch', 'n/a')} | {trained.get('seconds', 'n/a')}"
        )
    lines.extend(
        [
            "Models: compute/models/<model>; best and last checkpoints are preserved separately.",
            "TRAINING_READY_STOPPED_PENDING_EVALUATION_APPROVAL: own allocation released; no next work scheduled.",
        ]
    )
    (root / "training-report.txt").write_text("\n".join(lines) + "\n")


def control(root):
    root = Path(root).resolve()
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Coordinator runs in login-node tmux; computations run in Slurm")
    if Path(sys.executable).resolve() != Path(python_for(root, required=True)).resolve():
        raise ValueError("Approved Python required")
    cfg = read(root / "input/config.json")
    if comparison.is_comparison(cfg):
        comparison.config_check(cfg)
        if not os.environ.get("TMUX"):
            raise ValueError("Comparison controller must run inside server tmux")
        verify(root / "input", read(root / "input/inventory.json"))
    with (root / "controller.claim").open("x") as stream:
        stream.write("One finite approved lifecycle; no automatic retry.\n")
    state = dict(
        status="RUNNING",
        stage="submission",
        round=cfg["revision"],
        pid=os.getpid(),
        started=time.time(),
        llm_calls=0,
        retries=0,
        allocation_released=False,
        next_round_authorized=False,
    )
    record = None
    try:
        training_only = comparison.execution_scope(cfg) == "training_only"
        for phase in ("fit",) if training_only else ("fit", "evaluate"):
            record = submit(root, phase, cfg)
            state.update(stage=phase, job_id=record["job_id"], allocation_released=False)
            write(root / "controller.json", state)
            release = wait_release(root, record, state, cfg)
            write(root / ("release.json" if phase == "fit" else "evaluation-release.json"), release)
            require_success(release)
            wait_phase_files(root, phase, state, cfg)
        verify(root / "compute", read(root / "compute/result-seal.json")["files"])
        if training_only:
            seal = read(root / "compute/result-seal.json")
            if (
                seal.get("config_sha256") != sha(root / "input/config.json")
                or seal.get("models") != cfg["models"]
            ):
                raise ValueError("Training-only output binding mismatch")
            if own_queue():
                raise ValueError("Own job still pending/running at training completion check")
            training_report(root, cfg)
            status = "TRAINING_READY_STOPPED_PENDING_EVALUATION_APPROVAL"
            write(
                root / "completion.json",
                dict(
                    status=status,
                    finished=time.time(),
                    fit_release=read(root / "release.json"),
                    compute_seal_sha256=sha(root / "compute/result-seal.json"),
                    report_sha256=sha(root / "training-report.txt"),
                    execution_scope="training_only",
                    calibration_started=False,
                    evaluation_started=False,
                    next_round_authorized=False,
                    automatic_retry=False,
                    llm_calls=0,
                ),
            )
            state.update(
                status=status,
                stage="stopped",
                finished=time.time(),
                report=str(root / "training-report.txt"),
            )
            write(root / "controller.json", state)
            return
        evaluation = read(root / "evaluation/report.json")
        if (
            evaluation["status"] != "REPORT_READY_STOPPED"
            or sha(root / "evaluation/metrics.json") != evaluation["metrics_sha256"]
        ):
            raise ValueError("Final report verification failed")
        if own_queue():
            raise ValueError("Own job still pending/running at final check")
        report(root, cfg)
        write(
            root / "completion.json",
            dict(
                status="REPORT_READY_STOPPED",
                finished=time.time(),
                fit_release=read(root / "release.json"),
                evaluation_release=read(root / "evaluation-release.json"),
                report_sha256=sha(root / "report.txt"),
                metrics_sha256=evaluation["metrics_sha256"],
                next_round_authorized=False,
                automatic_retry=False,
                llm_calls=0,
            ),
        )
        state.update(
            status="REPORT_READY_STOPPED",
            stage="stopped",
            finished=time.time(),
            report=str(root / "report.txt"),
        )
        write(root / "controller.json", state)
    except BaseException as exc:
        released = False
        if record is not None:
            try:
                released = snapshot(record).get("released", False)
            except Exception:
                pass
        # A timed-out sbatch can have succeeded before its receipt was written.
        # Never label such an allocation stopped solely because submission raised.
        try:
            queue_uncertain = bool(own_queue())
        except Exception:
            queue_uncertain = True
        state.update(
            status="FAILED_STOPPED"
            if (record is None or released) and not queue_uncertain
            else "CONTROLLER_STOPPED_JOB_STATE_UNCONFIRMED",
            reason=f"{type(exc).__name__}: {exc}",
            automatic_retry=False,
            updated=time.time(),
        )
        write(root / "controller.json", state)
        raise


def optional(path):
    return read(path) if path.exists() else {}


def render(root):
    state = optional(root / "controller.json")
    preparing = not state and (root / "prepare-progress.json").exists()
    if preparing:
        state = optional(root / "prepare-progress.json")
        receipt = optional(root / "prepared.json")
        state["status"] = receipt.get("status", state.get("status"))
    phase = state.get("stage")
    progress = (
        state
        if preparing
        else optional(
            root / ("evaluation-progress.json" if phase == "evaluate" else "compute/progress.json")
        )
    )
    gpu = optional(root / "compute/gpu-live.json")
    age = time.time() - gpu.get("timestamp", 0)
    cfg = optional(root / "input/config.json")
    final = cfg.get("execution_scope") == "final_test_evaluation"
    is_comparison = comparison.is_comparison(cfg) or state.get("round") == comparison.REVISION
    neural_count = (len(cfg.get("variants", [])) or
                    sum(n.startswith(("lstm_", "transformer_")) for n in cfg.get("models", [])) or 8)
    model_count = len(cfg.get("models", [])) or 10
    lines = [
        (
            "Drone AI / 최종 시험324run · 학습 완료 신경망16 + baseline2 · batch256 · seed0"
            if final else f"Drone AI / 1200run · 신경망{neural_count} · 전체{model_count} · batch256 · seed0"
            if is_comparison
            else "Drone AI / 신경망2 + baseline2 · hidden128 · 2층 · batch256 · seed17"
        ),
        f"{state.get('status', 'NOT_STARTED')} | 단계 {phase} / {progress.get('stage', 'allocation')} | job {state.get('job_id', '미확인')}",
        "─" * 90,
        (
            "흐름: 고정 모델·scaler·임계값 확인 → 최종 추론 → 별도 최종 평가·기록 → 반납·정지 (학습·보정·분석 없음)"
            if final else f"흐름: {neural_count}구성 메모리·복원·처리율 → {model_count}종 학습·저장 → 반납·정지 (보정·평가 승인 대기)"
            if cfg.get("execution_scope") == "training_only"
            else f"흐름: {model_count}종 공통 수치 검사 → 새 정상 보정·개발 추론 → 별도 평가 → 반납·정지 (학습 없음)"
            if cfg.get("execution_scope") == "post_training_evaluation"
            else "흐름: 8구성 메모리·복원 → 10종 비교 → 정상 보정·개발 추론 → 별도 평가 → 정지"
            if is_comparison
            else "흐름: batch256 메모리·복원 → 4종 학습 → 정상 보정·개발 추론 → 별도 평가 → 정지"
        ),
    ]
    if progress.get("model"):
        lines.append("모델: " + progress["model"])
    if final:
        final_completed = optional(root / "compute/progress.json").get("models_completed", 0)
        lines.append(f"최종 추론 완료 모델: {final_completed}/18 · CPU8병렬 · 결과 비교/해석 자동 실행 없음")
    if progress.get("parallel_benchmark_active"):
        lines.append("2·4·8 CPU 병렬 사전시험 · 정상 검증 비행1개 반복 · 본 개발 추론 진행률 아님")
    elif progress.get("selected_cpu_workers"):
        lines.append(f"실측 선택 CPU 작업자: {progress['selected_cpu_workers']}개 · 전 모델 동일 적용")
    if state.get("file_visibility"):
        lines.append(
            f"파일 가시성: {state['file_visibility_phase']} / {state['file_visibility']} "
            f"| 대기 {state['file_visibility_waited_s']:.1f}/{state['file_visibility_limit_s']}초 "
            f"| 현재 할당 반납 {'확인' if state.get('allocation_released') else '대기'}"
        )
    if is_comparison and "models_completed" in progress:
        lines.append(f"모델 산출물: {len(progress['models_completed'])}/{model_count} (보정·평가는 별도)")
    if progress.get("gate_phase"):
        lines.append("복원 검사: " + progress["gate_phase"])
    if progress.get("epoch"):
        lines.append(
            f"epoch {progress['epoch']}/{progress.get('epochs', 60)} (조기 종료 가능) | batch {progress.get('batch', 0)}/{progress.get('batches', '?')}"
        )
    total = progress.get("total", 0)
    done = progress.get("done", 0)
    if total:
        percent = max(0, min(100, 100 * done / total))
        lines.append(
            f"현재 단계 [{'█' * int(percent / 5):·<20}] {percent:5.1f}%  {done}/{total} (전체 시간 진행률 아님)"
        )
    if progress.get("cpu_workers") and str(progress.get("stage", "")).startswith("development_"):
        lines.append(
            f"병렬 처리: GPU 추론1 · CPU 경보/저장{progress['cpu_workers']} · 입력 미리 읽기1 "
            f"| CPU 미완료 {progress.get('cpu_pending', 0)}건 · 완료 건수는 seal 작성 기준"
        )
    for key in ("train_loss", "validation_loss", "best_epoch"):
        if progress.get(key) is not None:
            lines.append(f"{key}: {progress[key]}")
    if gpu.get("status") == "SAMPLED":
        label = "실시간" if age < 10 and phase in {"fit", "calibrate", "diagnose", "infer"} else f"마지막 기록 ({age:.0f}초 전)"
        used = gpu.get("used_mib")
        capacity = gpu.get("total_mib")
        memory = (
            f"{used / 1024:.2f}/{capacity / 1024:.2f} GiB"
            if used is not None and capacity
            else "미확인"
        )
        lines.append(
            f"GPU {label}: 연산 {gpu.get('utilization_pct')}% | VRAM {memory} | {gpu.get('power_w')}W | {gpu.get('temperature_c')}°C"
        )
    else:
        lines.append("GPU 사용률: " + gpu.get("reason", "할당·표본 대기 / 미확인"))
    if state.get("reason") or progress.get("reason"):
        lines.append("사유: " + str(state.get("reason") or progress["reason"]))
    lines += [
        "─" * 90,
        "모델: " + str(root / "compute/models"),
        "결과: "
        + str(
            root
            / (
                "training-report.txt"
                if cfg.get("execution_scope") == "training_only"
                else "evaluation/metrics.json"
            )
        ),
        "q+Enter / Ctrl+C: 화면만 닫기 · 서버 작업 유지 · 자동 재학습·LLM 호출 없음",
    ]
    if (root / "completion.json").exists():
        completion = optional(root / "completion.json")
        if (
            completion.get("status") == "TRAINING_READY_STOPPED_PENDING_EVALUATION_APPROVAL"
            and completion.get("fit_release", {}).get("released") is True
        ):
            lines.append("학습 산출물·실제 할당 반납 확인됨. 보정·평가는 미실행·별도 승인 대기.")
        elif (
            completion.get("status") == "REPORT_READY_STOPPED"
            and completion.get("fit_release", completion.get("calibration_release", completion.get("inference_release", {}))).get("released") is True
            and completion.get("evaluation_release", {}).get("released") is True
        ):
            lines.append("실제 할당 반납·평가 완료 확인됨. 다음 회차 실행 없음.")
    return "\n".join(lines)


def display(root, once=False):
    import select

    root = Path(root)
    while True:
        if not once and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(render(root), flush=True)
        if once:
            return
        ready, _, _ = select.select([sys.stdin], [], [], 2)
        if ready:
            value = sys.stdin.readline()
            if not value or value.strip().lower() == "q":
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["control", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.action == "control":
        control(args.root)
    else:
        try:
            display(args.root, args.once)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
