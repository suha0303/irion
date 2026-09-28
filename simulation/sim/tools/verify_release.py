"""Record actual command exit codes; stop short of claiming AI C9."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parents[2]
env = os.environ.copy()
env["PYTHONPATH"] = str(root / "sim/src")
env["PATH"] = str(root / ".venv/bin") + ":" + env["PATH"]
commands = [
    ("profile", ["./capstone-sim", "profile", "validate", "profiles/x500_reference.yaml"]),
    ("unit-suite", ["./capstone-sim", "verify", "--suite", "unit"]),
    ("physics-details", ["build/powertrain_test"]),
    (
        "cppcheck",
        [
            "cppcheck",
            "--enable=warning,performance,portability",
            "--error-exitcode=1",
            "--std=c++17",
            "--suppress=missingIncludeSystem",
            "sim/physics",
        ],
    ),
    (
        "cpp-format",
        [
            "clang-format",
            "--dry-run",
            "--Werror",
            *[
                str(p.relative_to(root))
                for p in (root / "sim").rglob("*")
                if p.suffix in [".hpp", ".cpp"]
            ],
        ],
    ),
    ("rc-comparison", [sys.executable, "sim/tools/compare_rc.py"]),
    ("sensitivity", [sys.executable, "sim/tools/sensitivity.py"]),
    ("causal-evidence", [sys.executable, "sim/tools/causal_evidence.py"]),
    ("integration", ["./capstone-sim", "verify", "--suite", "integration"]),
    ("end-to-end", ["./capstone-sim", "verify", "--suite", "end-to-end"]),
]
results = []
for name, cmd in commands:
    path = root / "artifacts" / f"release-{name}.log"
    start = datetime.now(UTC).isoformat()
    with path.open("w") as out:
        p = subprocess.run(
            cmd, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, check=False
        )
    results.append(
        dict(name=name, command=cmd, start_utc=start, exit_code=p.returncode, log=str(path))
    )
    (root / "artifacts/release-verification.json").write_text(
        json.dumps(dict(commands=results, complete=False), indent=2) + "\n"
    )
report = dict(
    commands=results,
    complete=True,
    passed=all(r["exit_code"] == 0 for r in results),
    C9="deferred_ai_integration",
)
(root / "artifacts/release-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 4)
