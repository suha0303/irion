"""Read-only model and Slurm progress, following the existing status/watch UI."""

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = "/abr/coss43/.local/state/drone-ai/new19-20261120-fit01"
MODELS = tuple(
    f"{task}-{kind}-{size}"
    for task in ("inflight", "whole-run")
    for kind in ("lstm", "transformer")
    for size in ("64x2", "128x4")
)
CORRECTED = "inflight-transformer-128x4"
LSTM_COMPARISON = ("inflight-lstm-64x2", "inflight-lstm-128x4")
TRANSFORMER_COMPARISON = ("range-suggestion", "twice-range-suggestion")
PHASES = {
    "starting": "준비",
    "lr_range": "LR 범위 탐색",
    "lr_recheck_pending": "LR 재검증 대기",
    "recovery_pending": "복구 작업 대기",
    "resource_wait": "유휴 GPU 대기",
    "waiting_for_fit": "재학습 완료 대기",
    "waiting_for_evaluation": "평가 대기",
    "fit": "학습·검증",
    "calibration": "정상 보정",
    "sealed": "보정 완료 · 임계값 고정",
    "final_evaluation": "최종 평가",
    "complete": "완료",
    "failed": "실패",
    "interrupted": "할당 종료",
}


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def read_text(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def schedule_key(name, plan):
    manifest = plan.get("manifest_sha256")
    interval = plan.get("train_origin_interval_us")
    source = plan.get("sources", {}).get("data.py")
    if manifest and interval and source:
        return (name.rsplit("-", 2)[0], manifest, interval, source)
    return None


def training_item_counts(root):
    counts = {}
    for name in MODELS:
        fit = root / "run" / name / "fit"
        result = read(fit / "result.json") or {}
        plan = read(fit / "fit-plan.json") or {}
        key = schedule_key(name, plan)
        items = result.get("train_items")
        if key is not None and isinstance(items, int) and items > 0:
            counts.setdefault(key, set()).add(items)
    return {
        key: next(iter(values))
        for key, values in counts.items()
        if len(values) == 1
    }


def epoch_learning_rate(name, plan, epoch, train_items):
    """Rate at the last optimizer step of a completed epoch."""
    batch = plan.get("batch_size")
    epochs = plan.get("epochs")
    peak = plan.get("rate")
    if not (
        batch
        and epochs
        and peak
        and train_items
        and epoch
        and 1 <= epoch <= epochs
    ):
        return None
    # Match the frozen training.lr_at schedule on the Torch-free status path.
    steps = (train_items + batch - 1) // batch
    total = steps * epochs
    step = steps * epoch - 1
    architecture = name.rsplit("-", 2)[1]
    fraction = 0.02 if architecture == "lstm" else 0.07
    warmup = min(total - 1, math.floor(total * fraction))
    if step < warmup:
        return peak * (0.1 + 0.9 * step / warmup)
    progress = (step - warmup) / max(1, total - 1 - warmup)
    return peak * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)


def latest_batch(path):
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 4096))
            data = stream.read()
    except FileNotFoundError:
        return None
    if b"\n" not in data:
        return None
    rows = data[: data.rindex(b"\n")].splitlines()
    if not rows:
        return None
    try:
        step, rate, loss, norm = rows[-1].split(b",")
        return dict(
            step=int(step),
            rate=float(rate),
            loss=float(loss),
            gradient_norm=float(norm),
        )
    except ValueError:
        return None


def comparison_progress(experiment, names, allocations, *, previous=None):
    job = read_text(experiment / "job-id.txt")
    if job is None:
        return []
    active = any(row["job"] == job for row in allocations)
    waiting = (experiment / "waiting-at.txt").is_file() and not (
        experiment / "started-at.txt"
    ).is_file()
    models = []
    for name in names:
        folder = experiment / "run" / name
        status = read(folder / "status.json") or {
            "phase": "resource_wait" if waiting else "starting"
        }
        progress = read(folder / "fit/progress.json") or {}
        plan = read(folder / "comparison-plan.json") or {}
        if previous is not None:
            initial = previous / "run" / name
            progress = progress or read(initial / "fit/progress.json") or {}
            plan = plan or read(initial / "comparison-plan.json") or {}
        result = read(folder / "result.json")
        phase = status["phase"]
        error = status.get("error")
        exit_code = read_text(experiment / f"{name}.exit")
        batch = latest_batch(folder / "batch-trace.csv")
        if batch and plan.get("trace_step_offset"):
            batch["step"] += plan["trace_step_offset"]
        if exit_code not in (None, "0") and error is None:
            phase = "failed"
            error = f"Slurm 종료 코드 {exit_code}"
        elif not active and phase in {"starting", "fit", "resource_wait"}:
            phase = "interrupted"
        models.append(
            dict(
                model=name,
                job=job,
                phase=phase,
                epoch=progress.get("epoch", 0),
                max_epochs=plan.get("epochs"),
                lr=status.get("rate", plan.get("rate")),
                train_loss=progress.get("train_loss"),
                validation_loss=progress.get("validation_loss"),
                lr_at_batch=batch["rate"] if batch else None,
                batch=batch,
                result=result,
                error=error,
            )
        )
    return models


def evaluation_progress(evaluation, jobs, comparison, *, launched, active_jobs):
    if not launched:
        return []
    finished_fits = {row["model"] for row in comparison if row["result"]}
    rows = []
    for name, job in jobs.items():
        folder = evaluation / "run" / name
        status = read(folder / "status.json") or {}
        result = read(folder / "final/result.json")
        error = status.get("error") or read_text(evaluation / f"{name}.blocked")
        for stage in ("calibrate", "evaluate"):
            worker = read_text(evaluation / "logs" / f"{name}.{stage}.exit")
            if worker not in (None, "0"):
                error = error or f"Slurm 종료 코드 {worker}"
        if error:
            phase = "failed"
        elif result is None and job is not None and job not in active_jobs:
            phase = "interrupted"
        elif "phase" in status:
            phase = status["phase"]
        elif job is not None or name in finished_fits:
            phase = "waiting_for_evaluation"
        else:
            phase = "waiting_for_fit"
        rows.append(
            dict(
                model=name,
                job=job,
                phase=phase,
                prediction_files=sum(
                    1 for _ in (folder / "final/predictions").glob("*.npz")
                )
                if phase == "final_evaluation"
                else None,
                final=result["summary"] if result is not None else None,
                error=error,
            )
        )
    return rows


def snapshot(root):
    snapshot_time = datetime.now().astimezone().isoformat(timespec="seconds")
    retry = root / "lr-correction"
    correction_job = read_text(retry / "job-id.txt")
    correction_deferred = read_text(retry / "deferred.txt")
    recovery = root / "recovery-02"
    recovery_job = read_text(recovery / "job-id.txt")
    evaluation = root / "lstm-half-lr-eval-01"
    initial_transformer = root / "transformer128-higher-lr-01"
    resumed_transformer = root / "transformer128-higher-lr-02"
    transformer_job = read_text(resumed_transformer / "job-id.txt")
    transformer_eval = root / "transformer128-higher-lr-eval-01"
    transformer_eval_job = read_text(transformer_eval / "job-id.txt")
    evaluation_jobs = {
        name: read_text(evaluation / f"{name}.job") for name in LSTM_COMPARISON
    }
    queue = subprocess.check_output(
        ["squeue", "-h", "-u", "coss43", "-o", "%i|%T|%R|%b"],
        text=True,
        timeout=8,
    )
    allocations = []
    for line in queue.splitlines():
        job, state, node, gpus = line.split("|", 3)
        if job.split("+")[0] == "59830" or job in {
            correction_job,
            recovery_job,
            transformer_job,
            transformer_eval_job,
            *evaluation_jobs.values(),
        }:
            allocations.append(dict(job=job, state=state, node=node, gpus=gpus))
    models = []
    train_counts = training_item_counts(root)
    for name in MODELS:
        folder = root / "run" / name
        replacement = retry / "run" / name
        repaired_state = (
            read(replacement / "status.json")
            if correction_deferred is None
            else None
        )
        queued_retry = (
            name == CORRECTED
            and bool(correction_job)
            and correction_deferred is None
        )
        if name == CORRECTED and repaired_state is not None:
            folder = replacement
        recovering = recovery_job is not None and name.startswith("inflight-")
        recovery_folder = recovery / "run" / name
        recovered_state = (
            read(recovery_folder / "status.json") if recovering else None
        )
        if recovered_state is not None:
            folder = recovery_folder
        current = (
            recovered_state
            or ({"phase": "recovery_pending"} if recovering else None)
            or read(folder / "status.json")
            or {"phase": "starting"}
        )
        progress = read(folder / "fit/progress.json") or {}
        fit_plan = read(folder / "fit/fit-plan.json") or {}
        fit_result = read(folder / "fit/result.json") or {}
        allocation = read(folder / "environment.json") or {}
        ranges = sorted(folder.glob("range-*/lr-range.json"))
        lr = (read(ranges[-1]) or {}) if ranges else {}
        result = read(folder / "final/result.json")
        if (
            not recovering
            and name == CORRECTED
            and queued_retry
            and repaired_state is None
        ):
            current = {**current, "phase": "lr_recheck_pending"}
        if recovering:
            worker_exit = read_text(recovery / "logs" / f"{name}.exit")
        elif folder == root / "run" / name:
            worker_exit = read_text(root / "logs" / f"{name}.exit")
        else:
            worker_exit = None
        if worker_exit not in (None, "0") and current["phase"] != "failed":
            current = dict(
                phase="failed", error=f"Slurm 종료 코드 {worker_exit}"
            )
        row = dict(
            model=name,
            node=allocation.get("node"),
            job=recovery_job if recovering else allocation.get("component"),
            phase=current["phase"],
            epoch=progress.get("epoch"),
            max_epochs=fit_plan.get("epochs"),
            patience=fit_plan.get("patience"),
            fit_status=fit_result.get("status"),
            train_loss=progress.get("train_loss"),
            validation_loss=progress.get("validation_loss"),
            lr=current.get("lr", lr.get("candidate", fit_plan.get("rate"))),
            lr_at_epoch=epoch_learning_rate(
                name,
                fit_plan,
                progress.get("epoch"),
                fit_result.get("train_items")
                or train_counts.get(schedule_key(name, fit_plan)),
            ),
            bad_epochs=progress.get("bad_epochs"),
            final=result["summary"] if result is not None else None,
            prediction_files=len(
                list((folder / "final/predictions").glob("*.npz"))
            )
            if current["phase"] == "final_evaluation"
            else None,
            error=current.get("error"),
        )
        if name == CORRECTED:
            row["lr_recheck_job"] = correction_job
        models.append(row)
    lstm_comparison = comparison_progress(
        root / "lstm-half-lr-01", LSTM_COMPARISON, allocations
    )
    active_jobs = {item["job"] for item in allocations}
    return dict(
        checked_at=snapshot_time,
        allocations=allocations,
        models=models,
        lr_recheck_deferred_at=correction_deferred,
        recovery_job=recovery_job,
        lstm_comparison=lstm_comparison,
        lstm_evaluation=evaluation_progress(
            evaluation,
            evaluation_jobs,
            lstm_comparison,
            launched=(evaluation / "queued-at.txt").is_file(),
            active_jobs=active_jobs,
        ),
        transformer_comparison=comparison_progress(
            resumed_transformer
            if transformer_job is not None
            else initial_transformer,
            TRANSFORMER_COMPARISON,
            allocations,
            previous=initial_transformer
            if transformer_job is not None
            else None,
        ),
        transformer_evaluation=evaluation_progress(
            transformer_eval,
            dict.fromkeys(TRANSFORMER_COMPARISON, transformer_eval_job),
            (),
            launched=transformer_eval_job is not None,
            active_jobs=active_jobs,
        ),
    )


def display(value):
    print(f"X500 new19 · {value['checked_at']}")
    for row in value["allocations"]:
        print(
            f"Slurm {row['job']}  {row['state']}  {row['node']}  {row['gpus']}"
        )
    print()
    print(
        f"{'MODEL':30} {'NODE':6} {'EPOCH':>8} {'PEAK LR':>10} {'LR@EPOCH':>10}"
        f" {'TRAIN':>10} {'VAL':>10}  STATUS"
    )
    for row in value["models"]:
        epoch = (
            f"{row['epoch']}/{row['max_epochs'] or '?'}"
            if row["epoch"] is not None
            else "-"
        )
        last = row["phase"]
        status = PHASES.get(last, last)
        if row["fit_status"] == "early_stopped":
            status += " · 학습 조기 종료"
        if row["final"]:
            normal, anomaly = (
                row["final"][r] for r in ("normal_test", "anomaly_test")
            )
            status += f" · 오경보 {normal['alarms']}/{normal['runs']} · 검출 {anomaly['alarms']}/{anomaly['runs']}"
        if row["error"]:
            status += f" · {row['error']}"
        elif last == "fit" and row["bad_epochs"] is not None:
            status += f" · 미개선 {row['bad_epochs']}/{row['patience'] or '?'}"
        elif last == "final_evaluation" and row["prediction_files"] is not None:
            status += f" · 예측 파일 {row['prediction_files']}/450"
        cells = [
            f"{row[k]:.5g}" if row[k] is not None else "-"
            for k in ("lr", "lr_at_epoch", "train_loss", "validation_loss")
        ]
        print(
            f"{row['model']:30} {row['node'] or '-':6} {epoch:>8} "
            f"{' '.join(cell.rjust(10) for cell in cells)}  {status}"
        )
    print("LR@EPOCH · 마지막 완료 epoch의 마지막 업데이트 학습률")
    for title, rows in (
        ("LSTM 재학습 · 최고 학습률 절반", value["lstm_comparison"]),
        (
            "Transformer 128×4 · 높은 학습률 비교",
            value["transformer_comparison"],
        ),
    ):
        if not rows:
            continue
        print(f"\n{title}")
        print(
            f"{'MODEL':30} {'EPOCH':>8} {'PEAK LR':>10} {'LR@BATCH':>10}"
            f" {'TRAIN':>10} {'VAL':>10}  STATUS"
        )
        for row in rows:
            epoch = f"{row['epoch']}/{row['max_epochs'] or '?'}"
            cells = [
                f"{row[key]:.5g}" if row[key] is not None else "-"
                for key in (
                    "lr",
                    "lr_at_batch",
                    "train_loss",
                    "validation_loss",
                )
            ]
            detail = PHASES.get(row["phase"], row["phase"])
            if row["error"]:
                detail += f" · {row['error']}"
            elif result := row["result"]:
                detail += (
                    f" · 검증 최저 {result['best_validation_loss']:.5g}"
                    f" (epoch {result['best_epoch']})"
                )
            elif batch := row["batch"]:
                detail += (
                    f" · 배치 {batch['step']:,}"
                    f" · gradient norm {batch['gradient_norm']:.4g}"
                )
            print(
                f"{row['model']:30} {epoch:>8} "
                f"{' '.join(cell.rjust(10) for cell in cells)}  {detail}"
            )
        print("LR@BATCH · 배치 추적 파일에 기록된 마지막 업데이트 학습률")
    for title, rows in (
        ("LSTM 재학습 · 보정 및 최종 평가", value["lstm_evaluation"]),
        (
            "Transformer 128×4 후보 · 보정 및 최종 평가",
            value["transformer_evaluation"],
        ),
    ):
        if not rows:
            continue
        print(f"\n{title}")
        for row in rows:
            detail = PHASES.get(row["phase"], row["phase"])
            if row["error"]:
                detail += f" · {row['error']}"
            elif final := row["final"]:
                normal, anomaly = final["normal_test"], final["anomaly_test"]
                detail += (
                    f" · 오경보 {normal['alarms']}/{normal['runs']}"
                    f" · 검출 {anomaly['alarms']}/{anomaly['runs']}"
                )
            elif row["prediction_files"] is not None:
                detail += f" · 예측 파일 {row['prediction_files']}/450"
            print(f"{row['model']:30} {row['job'] or '-':>8}  {detail}")


def arguments(argv):
    parser = argparse.ArgumentParser(prog="drone-ai train new19 status")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true", help="10초마다 새로 표시")
    mode.add_argument(
        "--json", action="store_true", help="한 번 조회하고 JSON 출력"
    )
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    return parser.parse_args(argv)


def client(argv):
    args = arguments(argv)
    flags = ["--watch"] if args.watch else ["--json"] if args.json else []
    remote = shlex.join(
        [
            "env",
            "TZ=Asia/Seoul",
            "python3",
            "-u",
            "-",
            "--root",
            str(args.root),
            *flags,
        ]
    )
    ssh = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "knu",
        remote,
    ]
    try:
        with Path(__file__).open("rb") as source:
            return subprocess.run(ssh, stdin=source, check=False).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    args = arguments(sys.argv[1:])
    try:
        while True:
            value = snapshot(args.root)
            if args.watch:
                print("\033[2J\033[H", end="")
            if args.json:
                print(json.dumps(value, ensure_ascii=False, indent=2))
            else:
                display(value)
            if not args.watch:
                break
            print("\nCtrl+C로 조회 화면 종료")
            time.sleep(10)
    except (KeyboardInterrupt, BrokenPipeError):
        pass
