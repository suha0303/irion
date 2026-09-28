"""Acceptance evidence aggregation. C9 is deliberately outside producer acceptance."""

import json
from pathlib import Path

from .recording import dump, verify
from .runner import ROOT


def acceptance(suite):
    checks = []
    for name in ["campaign-v2.json", "campaign-final.json"]:
        path = ROOT / "artifacts" / name
        if not path.exists():
            checks.append(dict(name=name, status="not_run"))
            continue
        rows = json.loads(path.read_text())
        expected = 16 if name == "campaign-v2.json" else 10
        for index, row in rows.items():
            state = json.loads((Path(row["run"]) / "run.json").read_text())
            result = verify(row["run"])
            checks.append(
                dict(
                    name=f"{name}:{index}",
                    run=row["run"],
                    status="passed"
                    if state["status"] == "completed" and result["status"] == "passed"
                    else "failed",
                )
            )
        if len(rows) != expected:
            checks.append(dict(name=f"{name}:coverage", status="not_run"))
    for name, key in [
        ("rc-comparison.json", "passed"),
        ("propulsion-comparison.json", "passed"),
        ("battery-fit.json", "comparison_status"),
        ("live-control.json", "passed"),
        ("process-recovery.json", "passed"),
        ("instance-lease.json", "passed"),
        ("diagnostics-check.json", "passed"),
        ("sensitivity.json", "passed"),
        ("causal-evidence.json", "passed"),
    ]:
        path = ROOT / "artifacts" / name
        if not path.exists():
            checks.append(dict(name=name, status="not_run"))
        else:
            value = json.loads(path.read_text())[key]
            checks.append(
                dict(name=name, status="passed" if value is True or value == "passed" else "failed")
            )
    result = dict(
        suite=suite,
        scope="producer simulation acceptance; no AI consumer",
        status="passed" if all(c["status"] == "passed" for c in checks) else "failed",
        checks=checks,
        C9="deferred_ai_integration",
        hardware_validation="not_in_scope",
        clean_OS_install="not_run_existing_supported_environment_fresh_upstream_source_build",
    )
    dump(ROOT / "artifacts" / f"{suite}-acceptance.json", result)
    return result
