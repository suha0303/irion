import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capstone_sim.profile import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the selected vehicle's compiled physics header"
    )
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.profile is None:
        generate(target=args.output)
    else:
        generate(args.profile, args.output)
