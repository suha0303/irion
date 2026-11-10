"""Explicit commands for the fresh19-input SGD implementation."""

import argparse
import json
import tempfile
from pathlib import Path

from learning.history19 import pilot, preprocessing, short_cycle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser(
        "validate", help="새 데이터셋 형식·분할·scaler 검증; 학습 없음"
    )
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--producer-binding", type=Path)
    validate.add_argument(
        "--protection", type=Path, help="Frozen global identity exclusions"
    )
    inspect = sub.add_parser(
        "inspect-run", help="예비/개발/최종 run 입력 검산; scaler·학습 없음"
    )
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--producer-binding", type=Path)
    inspect.add_argument(
        "--protection", type=Path, help="Frozen global identity exclusions"
    )
    inspect.add_argument(
        "--usage",
        choices=(
            "input_check",
            "pilot_input_check",
            "short_cycle_input_check",
            "final_test",
        ),
        default="input_check",
    )
    check = sub.add_parser(
        "check-input",
        help="P27 one-shot prerequisite/case receipt; no model/scaler",
    )
    check.add_argument("--request", type=Path, required=True)
    check.add_argument("--output", type=Path)
    short = sub.add_parser(
        "check-short-input",
        help="Explicit180s no-fit short-cycle gate; no development6",
    )
    short.add_argument("--request", type=Path, required=True)
    short.add_argument("--output", type=Path)
    fit = sub.add_parser(
        "fit", help="할당된 단일GPU에서 유한SGD 검사·학습; 명시 승인 필요"
    )
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--approve-training", action="store_true")
    replay = sub.add_parser(
        "preprocess", help="새19입력 전처리; source는 읽기 전용, 새 출력만"
    )
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("preprocessing.json"),
    )
    args = parser.parse_args(argv)
    if args.command == "check-short-input":
        return short_cycle.main(args.request, args.output)
    if args.command == "check-input":
        return pilot.main(args.request, args.output)
    elif args.command == "inspect-run":
        from .data import inspect_run

        expected = (
            json.loads(args.producer_binding.read_text(encoding="utf-8"))
            if args.producer_binding
            else None
        )
        protection = (
            json.loads(args.protection.read_text(encoding="utf-8"))
            if args.protection
            else None
        )
        print(
            json.dumps(
                inspect_run(
                    args.manifest,
                    expected_producer=expected,
                    protection=protection,
                    usage=args.usage,
                )
            )
        )
    elif args.command == "validate":
        from .data import load

        with tempfile.TemporaryDirectory(prefix="history19-validate-") as temp:
            expected = (
                json.loads(args.producer_binding.read_text(encoding="utf-8"))
                if args.producer_binding
                else None
            )
            arrays, _, binding = load(
                args.manifest,
                cache=Path(temp) / "normalized",
                expected_producer=expected,
                protection=json.loads(
                    args.protection.read_text(encoding="utf-8")
                )
                if args.protection
                else None,
            )
            print(
                json.dumps(
                    dict(
                        status="INPUT_VALIDATED_NOT_TRAINED",
                        binding=binding,
                        shapes={
                            r: {k: list(v.shape) for k, v in a.items()}
                            for r, a in arrays.items()
                        },
                    )
                )
            )
    elif args.command == "preprocess":
        print(
            json.dumps(
                preprocessing.replay(
                    args.run,
                    args.output,
                    json.loads(args.config.read_text(encoding="utf-8")),
                )
            )
        )
    else:
        from .runner import fit as run

        run(args.manifest, args.output, approved=args.approve_training)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
