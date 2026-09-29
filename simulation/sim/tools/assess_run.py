"""Print one read-only Collection assessment JSON; no simulation or automatic retries."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim/src"))
from capstone_sim.run_assessment import assess  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--instance", type=int, required=True)
parser.add_argument("--duration", type=float, required=True)
parser.add_argument("--profile-id", required=True)
args = parser.parse_args()
try:
    result = assess(args.run, args.instance, args.duration, args.profile_id)
except (OSError, ValueError, KeyError, TypeError) as exc:
    result = dict(schema="drone.run-assessment/1", run_id=Path(args.run).name,
                  instance=args.instance, gz_partition="x500_"+Path(args.run).name,
                  profile_id=args.profile_id, contract_passed=False, instance_isolation_passed=False,
                  missing_data_passed=False, physical_gate_passed=False, observed_valid_seconds=0,
                  errors=[str(exc)])
print(json.dumps(result, indent=2, allow_nan=False))
raise SystemExit(0 if all(result[k] for k in ["contract_passed", "instance_isolation_passed",
                                          "missing_data_passed", "physical_gate_passed"]) else 4)
