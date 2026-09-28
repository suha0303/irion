"""Produce evidence summaries from actual runs. Does not certify full acceptance."""

import json
from pathlib import Path

import numpy as np
from capstone_sim.recording import dump, verify

ROOT = Path(__file__).resolve().parents[2]
rows = []
for r in sorted((ROOT / "runs").glob("*/run.json")):
    run = json.loads(r.read_text())
    path = r.parent
    if run["status"] not in ["completed", "failed"]:
        continue
    scenario = json.loads((path / "evaluation/scenario.json").read_text())
    result = dict(
        run_id=run["run_id"],
        run=str(path),
        scenario=scenario["name"],
        seed=scenario["seed"],
        status=run["status"],
        reason=run.get("termination_reason"),
        process_exit_codes=run.get("process_exit_codes"),
        parent_attempt_id=run.get("parent_attempt_id"),
        origin=run["origin"],
    )
    try:
        result["contract"] = verify(path)
    except (OSError, ValueError, KeyError) as exc:
        result["contract"] = dict(status="failed", errors=[str(exc)])
    truth_file = path / "evaluation/truth.jsonl"
    if truth_file.exists() and truth_file.stat().st_size:
        truth = [json.loads(line) for line in truth_file.read_text().splitlines()]
        t = np.array([v["time_us"] / 1e6 for v in truth])
        alt = np.array([v["altitude_m"] for v in truth])
        stable = [v for v in truth if 30 <= v["time_us"] / 1e6 <= 90 and v["altitude_m"] > 2]
        result["physics"] = dict(
            samples=len(truth),
            simulated_seconds=float(t[-1] - t[0]),
            max_altitude_m=float(alt.max()),
            hover_seconds=float(sum(np.diff(t)[alt[:-1] > 2])),
            usable_after_60s_seconds=float(max(0, sum(np.diff(t)[alt[:-1] > 2]) - 60)),
            max_bus_residual_v=max(abs(v["solver_residual_v"]) for v in truth),
            max_motor_energy_residual_fraction=max(
                (
                    abs(v["energy_residual_j"]) / v["bus_energy_j"]
                    for v in truth
                    if v["bus_energy_j"] > 1
                ),
                default=None,
            ),
            soc_used=truth[0]["soc"] - truth[-1]["soc"],
            charge_used_ah=(truth[0]["soc"] - truth[-1]["soc"]) * truth[0]["capacity_ah"],
            nominal_charge_fraction=(truth[0]["soc"] - truth[-1]["soc"])
            * truth[0]["capacity_ah"]
            / 6,
            real_time_factor=float((t[-1] - t[0]) / run["wall_duration_s"]),
        )
        if stable:
            result["physics"]["steady_mean"] = {
                key: float(np.mean([v[key] for v in stable]))
                for key in ["voltage_v", "current_a", "altitude_m"]
            }
        if stable:
            result["physics"]["steady_mean"]["esc_request"] = float(
                np.mean([v["command"] for v in stable])
            )
        events = [
            json.loads(line) for line in (path / "evaluation/events.jsonl").read_text().splitlines()
        ]
        result["events"] = events
        result["physics_checks"] = dict(
            bus_solver=result["physics"]["max_bus_residual_v"] < 1e-6,
            motor_energy=result["physics"]["max_motor_energy_residual_fraction"] is not None
            and result["physics"]["max_motor_energy_residual_fraction"] < 0.001,
            airborne=result["physics"]["max_altitude_m"] > 2.5,
            complete_landing=run["status"] == "completed",
        )
    rows.append(result)
dump(ROOT / "artifacts/run-summary.json", rows)
lines = [
    "# Actual run evidence",
    "",
    "| run | scenario | flight | producer contract | sim s | hover s | result |",
    "|---|---|---|---|---:|---:|---|",
]
for r in rows:
    phy = r.get("physics", {})
    lines.append(
        f"| {r['run_id']} | {r['scenario']} | {r['status']} | {r['contract']['status']} | {phy.get('simulated_seconds', 0):.1f} | {phy.get('hover_seconds', 0):.1f} | {r['reason']} |"
    )
lines += [
    "",
    "C9: deferred_ai_integration. Table is actual evidence, not a full acceptance declaration.",
    "Process exit codes and individual numeric checks: `artifacts/run-summary.json`.",
]
(ROOT / "artifacts/RUN_REPORT.md").write_text("\n".join(lines) + "\n")
print(
    json.dumps(
        dict(
            runs=len(rows),
            completed=sum(r["status"] == "completed" for r in rows),
            producer_passed=sum(r["contract"]["status"] == "passed" for r in rows),
        )
    )
)
