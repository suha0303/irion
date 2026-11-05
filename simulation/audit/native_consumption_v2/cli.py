"""Read-only full assessment or native receipt, written only outside original raw."""
import argparse
import json
from pathlib import Path

from . import api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["assess", "joint-check", "pair-check"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--normal-run", type=Path)
    parser.add_argument("--anomaly-run", type=Path)
    parser.add_argument("--instance", type=int)
    parser.add_argument("--duration", type=float, default=480)
    parser.add_argument("--profile-id", default="vtc6_4s4p")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    inputs = [path.resolve() for path in (args.run, args.normal_run, args.anomaly_run) if path is not None]
    if (output.exists() or any(path.is_symlink() for path in (output, *output.parents))
            or any(output.resolve().is_relative_to(path) for path in inputs)):
        raise ValueError("Audit output must be a new external regular file")
    try:
        if args.action == "assess":
            if args.run is None or args.instance is None:
                raise ValueError("Actual run and instance required")
            result = api.assess(args.run, args.instance, args.duration, args.profile_id)
            passed = result["audit_full_gate_passed"]
        elif args.action == "joint-check":
            if args.run is None:
                raise ValueError("Actual run required")
            result = api.verify(args.run)
            passed = result["status"] == "passed"
        else:
            result = api.verify_pair(args.normal_run, args.anomaly_run, output)
            print(json.dumps(result, allow_nan=False))
            return 0 if result["status"] == "passed" else 4
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(schema="drone.native-consumption-audit-error/1", status="failed", error=str(error))
        passed = False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, allow_nan=False))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
