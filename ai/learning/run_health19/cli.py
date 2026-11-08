"""Local bounded tests and new run-dataset inspection. No GPU or production
launcher.
"""

import argparse
import json
from pathlib import Path

from .data import iter_manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "validate", help="Read-only check of a new full-run manifest"
    )
    inspect.add_argument(
        "--manifest", required=True, type=Path, action="append"
    )
    inspect.add_argument(
        "--producer-binding",
        type=Path,
        help="Externally frozen expected producer JSON",
    )
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
    check = commands.add_parser(
        "check-input", help="P27 one-shot prerequisite/case receipt; no fit"
    )
    check.add_argument("--request", type=Path, required=True)
    check.add_argument("--output", type=Path)
    short = commands.add_parser(
        "check-short-input", help="Explicit180s no-fit short-cycle gate"
    )
    short.add_argument("--request", type=Path, required=True)
    short.add_argument("--output", type=Path)
    smoke = commands.add_parser(
        "smoke", help="Synthetic normal-only bounded CPU train/restore test"
    )
    smoke.add_argument(
        "--model", choices=("lstm", "transformer"), required=True
    )
    smoke.add_argument("--hidden", type=int, choices=(64, 128), default=64)
    smoke.add_argument("--layers", type=int, choices=(2, 4), default=2)
    smoke.add_argument("--output", required=True, type=Path)
    smoke.add_argument("--resume", type=Path)
    smoke.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args(argv)
    if args.command == "check-short-input":
        from learning.history19.short_cycle import main as check_short

        return check_short(args.request, args.output)
    if args.command == "check-input":
        from learning.history19.pilot import main as check_input

        return check_input(args.request, args.output)
    if args.command == "validate":
        expected = (
            json.loads(args.producer_binding.read_text(encoding="utf-8"))
            if args.producer_binding
            else None
        )
        count, frames = 0, 0
        protection = (
            json.loads(args.protection.read_text(encoding="utf-8"))
            if args.protection
            else None
        )
        for run in iter_manifests(
            args.manifest,
            expected_producer=expected,
            protection=protection,
            usage=args.usage,
        ):
            count += 1
            frames += len(run.values)
            del run
        print(
            json.dumps(
                dict(
                    status="schema_validated",
                    runs=count,
                    frames=frames,
                    scope="supplied_manifests_only",
                    model_training_started=False,
                    physical_eligibility_verified=False,
                )
            )
        )
        return 0
    from .fixtures import normal_runs
    from .models import ModelSpec
    from .training import fit_cpu, restore

    result = fit_cpu(
        ModelSpec(args.model, args.hidden, args.layers),
        normal_runs("normal_train", 4),
        normal_runs("normal_validation", 2, seed=1),
        args.output,
        resume=args.resume,
        stop_after_epoch=args.stop_after_epoch,
    )
    restore(args.output)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
