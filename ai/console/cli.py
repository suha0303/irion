"""Portable AI console: configure, diagnose, smoke, and explicit comparison controls."""

import argparse
import json
import os
import select
import shlex
import subprocess
import sys
import time
from pathlib import Path

from . import profile


def configure(args):
    def value(key, prompt, default=None, *, numeric=False):
        found = getattr(args, key, None)
        if found is None:
            if args.non_interactive:
                if default is None:
                    raise ValueError("Missing explicit configuration: --" + key.replace("_", "-"))
                found = default
            else:
                suffix = f" [{default}]" if default is not None else ""
                found = input(prompt + suffix + ": ").strip() or default
        if found is None:
            raise ValueError("A value is required: " + key)
        return int(found) if numeric else str(found)

    name = value("name", "프로필 이름", "default")
    ssh_host = value("host", "SSH 서버 (서버에서 직접 사용할 때 빈칸)", "")
    ssh = None if not ssh_host else {
        "host": ssh_host, "user": value("user", "SSH 사용자"),
        "port": value("port", "SSH 포트", 22, numeric=True),
    }
    settings = {
        "schema": profile.SCHEMA, "name": name, "ssh": ssh,
        "project_root": value("project_root", "서버의 클론 경로", str(profile.ROOT)),
        "python": value("python", "기존 AI Python 실행 파일의 절대 경로"),
        "runtime_profile": value("runtime_profile", "런타임 (torch21/mamba25)", "mamba25"),
        "dataset_root": value("dataset_root", "검증된 데이터 경로 (dataset/·evidence/의 상위; 기존 봉인 input은 기능시험 전용)"),
        "output_root": value("output_root", "모델·시험·로그 출력 상위 경로"),
        "slurm": {
            "partition": value("partition", "Slurm partition", "p02"),
            "node": value("node", "Slurm node", "DIS01"),
            "nodes": 1, "ntasks": 1, "gpus": 1,
            "cpus": value("cpus", "짧은 GPU 시험 CPU 수", 4, numeric=True),
            "memory_gib": value("memory_gib", "짧은 GPU 시험 RAM 요청 (GiB)", 16, numeric=True),
        },
    }
    destination = args.profile or name
    path = profile.write(destination, settings, replace=args.replace)
    print("설정 저장: " + str(path))
    print("비밀번호·개인키를 저장하지 않았습니다. 설치·GPU 작업은 시작하지 않았습니다.")
    return 0


def environment(settings, profile_file=None):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ":".join(str(profile.ROOT / p) for p in (
        "ai", "ai/src", "Data/generation"
    ))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("DRONE_AI_PROFILE", None)
    if profile_file is not None:
        environment["DRONE_AI_PROFILE"] = str(profile.profile_path(profile_file))
    return environment


def selected_python(settings, module, arguments, *, profile_file=None):
    if profile.absolute_path(settings["project_root"]) != profile.ROOT:
        raise ValueError("Run this command on the configured server clone; use connect to log in")
    executable = Path(settings["python"])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Configured Python does not exist or is not executable")
    return subprocess.run(
        [str(executable), "-B", "-m", module, *arguments],
        env=environment(settings, profile_file), check=False
    ).returncode


def snapshot(root):
    root = profile.absolute_path(root)
    records = {}
    for relative in (
        "control.json", "controller.json", "progress.json", "prepare-progress.json", "prepared.json",
        "compute/progress.json", "compute/result.json", "fit/progress.json",
        "evaluation/progress.json", "evaluation-progress.json", "result.json",
        "input/config.json", "completion.json", "release.json", "evaluation-release.json",
        "reconciliation.json",
    ):
        path = root / relative
        if path.is_file() and not path.is_symlink():
            try:
                records[relative] = json.loads(path.read_text())
            except (OSError, ValueError):
                records[relative] = {"status": "UNREADABLE"}
    if "reconciliation.json" in records:
        from .control import file_sha

        reconciliation = records["reconciliation.json"]
        reconciliation["binding_valid"] = all(
            (root / name).is_file() and not (root / name).is_symlink()
            and file_sha(root / name) == reconciliation.get(key)
            for name, key in (
                ("control.json", "original_control_sha256"), ("request.json", "request_sha256"),
                ("server-profile.json", "profile_sha256"),
            )
        )
    gpu_files = [p for p in root.glob("**/gpu-live.json") if len(p.relative_to(root).parts) <= 5]
    if gpu_files:
        latest = max(gpu_files, key=lambda p: p.stat().st_mtime)
        try:
            records["gpu"] = json.loads(latest.read_text())
            records["gpu"]["source"] = str(latest)
        except (OSError, ValueError):
            records["gpu"] = {"status": "UNREADABLE"}
    return {"root": str(root), "records": records, "read_at": time.time()}


def display(value):
    records = value["records"]
    main_comparison = "controller.json" in records or "input/config.json" in records
    control = records.get("control.json", records.get("controller.json", {}))
    reconciliation = records.get("reconciliation.json", {})
    reconciled = reconciliation.get("status") == "SMOKE_READY_STOPPED" and (
        reconciliation.get("binding_valid") is True
    )
    if reconciled:
        control = reconciliation
    phase = control.get("stage")
    if phase == "evaluate":
        progress = records.get("evaluation-progress.json", records.get("evaluation/progress.json", {}))
    else:
        progress = records.get("compute/progress.json", records.get("progress.json", {}))
    if not control and "prepare-progress.json" in records:
        progress = records["prepare-progress.json"]
        control = records.get("prepared.json", {})
    status = control.get("status", progress.get("status", "UNKNOWN"))
    print("DRONE AI  /  학습·복원 시험 및 모델 비교·평가")
    print("상태: " + status + "  |  " + time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    if reconciled:
        print("기존 종료 작업의 검산 결과 · 원래 control.json 실패 기록 보존")
    print("작업: " + value["root"])
    print("Job: " + str(control.get("job_id", "미확인"))
          + "  |  Slurm: " + str(control.get("slurm_state", "미확인")))
    if main_comparison:
        cfg = records.get("input/config.json", {})
        description = ("비교: " + str(len(cfg.get("models", []))) + "종  |  seed "
                       + str(cfg.get("seed", "미확인")))
        if cfg.get("execution_scope") != "post_training_evaluation":
            description += "  |  batch " + str(cfg.get("batch_size", "미확인"))
        print(description)
        if cfg.get("execution_scope") == "training_only":
            print("승인 범위: 구성별 검사 → 학습·모델 저장 → 반납·정지 (보정·평가 별도 승인)")
        elif cfg.get("execution_scope") == "post_training_evaluation":
            print("승인 범위: 전 모델 정상 수치 진단 → 반납·정지 (보정·개발 추론 없음)"
                  if cfg.get("numerical_diagnostic_only") else
                  "승인 범위: 전 모델 수치 검사 → 정상 보정·개발 추론 → 독립 평가 → 반납·정지")
            print("재학습·최종 시험·자동 재시도 없음")
            if cfg.get("numerical_diagnostic_only"):
                print("공통 입력 257창 · batch 1 대비 256 · batch 256 반복 재현성")
            elif cfg.get("all_model_replay_fresh"):
                print("개발 추론: 전 모델 새로 실행 · batch " + str(cfg["prediction_batch_size"])
                      + (" · 정상 보정도 전 모델 새로 계산"
                         if cfg.get("all_model_calibration_fresh") else
                         " · 완료된 정상 보정 재사용 " + str(len(cfg.get("reuse_calibration", {}))) + "종"))
            if cfg.get("sensitivity_policy"):
                print("선행 검사: 전 모델 연속 정상 구간 점수·경보 민감도 · 모두 통과해야 본 보정 시작")
                print("진단용 보정은 본 평가에 재사용하지 않음 · 기준 변경/자동 재시도 없음")
    if progress.get("model"):
        print("모델: " + str(progress["model"]))
    if progress.get("models_completed") is not None:
        print("모델 산출물: " + str(len(progress["models_completed"]))
              + "개 (정상 보정·개발 평가 완료와 별개)")
    if progress.get("gate_phase"):
        print("복원 검사: " + str(progress["gate_phase"]))
    if progress.get("epoch") is not None:
        print(f"epoch {progress['epoch']}/{progress.get('epochs', '미확인')} (조기 종료 가능)"
              f"  |  batch {progress.get('batch', 0)}/{progress.get('batches', '미확인')}")
    if progress:
        done, total = progress.get("done"), progress.get("total")
        count = "미확인"
        if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
            count = f"{done}/{total} ({100 * done / total:.1f}%, 시간 진행률 아님)"
        print("단계: " + str(progress.get("stage", "미확인")) + "  |  처리: " + count)
    for key in ("train_loss", "validation_loss", "best_epoch", "optimizer", "learning_rate"):
        if progress.get(key) is not None:
            print(key + ": " + str(progress[key]))
    gpu = records.get("gpu", {})
    if gpu:
        age = time.time() - gpu.get("timestamp", 0)
        freshness = "현재 기록" if age < 10 and (
            not main_comparison or phase in {"fit", "calibrate", "diagnose", "benchmark"}
        ) else "마지막 기록 (실시간 아님)"
        print(f"GPU: {gpu.get('utilization_pct', '미확인')}%  |  VRAM: "
              f"{gpu.get('used_mib', '미확인')}/{gpu.get('total_mib', '미확인')} MiB  |  {freshness}")
    reason = control.get("reason") or progress.get("reason")
    if reason:
        print("사유: " + str(reason))
    released = control.get("allocation_released", "미확인")
    if main_comparison:
        completion = records.get("completion.json", {})
        if cfg.get("execution_scope") == "training_only":
            released = bool(
                completion.get("status") == "TRAINING_READY_STOPPED_PENDING_EVALUATION_APPROVAL"
                and completion.get("fit_release", {}).get("released") is True
            )
            print("학습 기록: " + str(Path(value["root"]) / "training-report.txt"))
            print("보정·개발 평가: 미실행 · 별도 승인 필요")
        elif cfg.get("numerical_diagnostic_only"):
            released = bool(completion.get("status") == "DIAGNOSTIC_READY_STOPPED"
                            and completion.get("diagnostic_release", {}).get("released") is True)
            print("진단: " + str(Path(value["root"]) / "compute/batch-checks/preflight.json"))
        else:
            release_key = "calibration_release" if cfg.get("execution_scope") == "post_training_evaluation" else "fit_release"
            released = bool(
                completion.get("status") == "REPORT_READY_STOPPED"
                and completion.get(release_key, {}).get("released") is True
                and completion.get("evaluation_release", {}).get("released") is True
            )
            print("결과: " + str(Path(value["root"]) / "evaluation/metrics.json"))
        print("모델: " + str(Path(value["root"]) / "compute/models"))
    print("할당 반납 확인: " + str(released))
    if control.get("scope") and not main_comparison:
        print("범위: 짧은 학습·저장·복원 기능시험 / 본학습 아님")
    print("화면만 갱신 · q/Ctrl+C: 화면 종료 · 작업은 tmux/Slurm에서 유지 · LLM 호출 없음")


def watch(root):
    import termios
    import tty

    terminal = sys.stdin.isatty()
    original = termios.tcgetattr(sys.stdin) if terminal else None
    try:
        if terminal:
            tty.setcbreak(sys.stdin.fileno())
        while True:
            if sys.stdout.isatty():
                print("\033[H\033[2J", end="")
            display(snapshot(root))
            if terminal:
                ready, _, _ = select.select([sys.stdin], [], [], 2)
                if ready and sys.stdin.read(1).lower() == "q":
                    break
            else:
                time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        if original is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)


def menu():
    print("DRONE AI  /  서버의 클론 폴더에서 실행")
    print("1. 서버·환경·데이터 경로 설정\n2. 환경 준비·진단 (읽기 전용)")
    print("3. 짧은 GPU 학습·저장·복원 시험\n4. 기존 작업 화면\n0. 종료")
    choice = input("선택: ").strip()
    if choice == "0":
        return 0
    if choice == "1":
        return main(["configure"])
    selected = input("프로필 이름 또는 절대 경로 [default]: ").strip() or "default"
    if choice == "2":
        return main(["doctor", "--profile", selected])
    if choice in {"3", "4"}:
        root = input("새 시험 출력 경로" if choice == "3" else "기존 작업 경로").strip()
        return main(["smoke" if choice == "3" else "watch", "--profile", selected,
                     "--root", root])
    raise ValueError("Unknown menu selection")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="action")
    setup = sub.add_parser("configure", help="서버·경로 직접 입력; 인증 비밀 저장 금지")
    setup.add_argument("--profile")
    setup.add_argument("--non-interactive", action="store_true")
    setup.add_argument("--replace", action="store_true")
    for key in ("name", "host", "user", "project-root", "python", "runtime-profile",
                "dataset-root", "output-root", "partition", "node"):
        setup.add_argument("--" + key)
    for key in ("port", "cpus", "memory-gib"):
        setup.add_argument("--" + key, type=int)
    for name in ("setup", "doctor", "connect", "smoke", "status", "watch"):
        command = sub.add_parser(name)
        command.add_argument("--profile", default="default")
        if name in {"smoke", "status", "watch"}:
            command.add_argument("--root", type=Path, required=True)
        if name == "status":
            command.add_argument("--json", action="store_true")
    reconcile = sub.add_parser("reconcile", help="기존 종료 smoke 검산; 제출·재시작 없음")
    reconcile.add_argument("--root", type=Path, required=True)
    for name, help_text in (
        ("train", "비교 준비·승인·시작 CLI; 자동 본학습 없음"),
        ("evaluate", "완료 모델의 정상 보정·개발 평가; 재학습 없음"),
        ("reassess", "봉인된 예측의 다른 기준 재집계; 새 출력 필요"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--profile", default="default")
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    sub.add_parser("menu")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action in {None, "menu"}:
            return menu()
        if args.action == "configure":
            return configure(args)
        if args.action == "reconcile":
            from .control import reconcile

            print(json.dumps(reconcile(args.root), indent=2, ensure_ascii=False))
            return 0
        if args.action in {"status", "watch"}:
            if args.action == "watch":
                watch(args.root)
            elif args.json:
                print(json.dumps(snapshot(args.root), indent=2, ensure_ascii=False))
            else:
                display(snapshot(args.root))
            return 0
        settings = profile.load(args.profile)
        profile_file = profile.profile_path(args.profile)
        if args.action in {"setup", "doctor"}:
            return selected_python(settings, "console.environment", ["--profile", str(profile_file)],
                                   profile_file=profile_file)
        if args.action == "connect":
            body = "cd -- " + shlex.quote(settings["project_root"]) + "; exec tmux new-session -A -s drone-ai"
            ssh = profile.ssh_command(settings)
            return subprocess.run([*ssh[:-1], "-t", ssh[-1], body], check=False).returncode
        if args.action == "smoke":
            from .control import start

            code = selected_python(settings, "console.environment", ["--profile", str(profile_file)],
                                   profile_file=profile_file)
            if code:
                return code
            print(json.dumps(start(profile_file, args.root), indent=2, ensure_ascii=False))
            return 0
        if args.action in {"train", "evaluate", "reassess"}:
            arguments = list(args.arguments)
            if arguments[:1] == ["--"]:
                arguments = arguments[1:]
            if not arguments:
                arguments = ["--help"]
            if args.action != "reassess" and arguments[0] == "prepare":
                arguments += ["--profile", str(profile_file)]
            module = {"train": "learning.prepare_compare", "evaluate": "learning.post_training",
                      "reassess": "learning.reassess"}[args.action]
            return selected_python(settings, module, arguments,
                                   profile_file=profile_file)
        raise ValueError("Unsupported action")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
