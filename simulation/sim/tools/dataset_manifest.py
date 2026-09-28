"""Evaluation/selection manifest only; never creates AI train/test splits."""

import json
from pathlib import Path

from capstone_sim.recording import dump

ROOT = Path(__file__).resolve().parents[2]
rows = []
for name in ["campaign-v2.json", "campaign-final.json"]:
    checkpoint = json.loads((ROOT / "artifacts" / name).read_text())
    for index, row in sorted(checkpoint.items(), key=lambda kv: int(kv[0])):
        run = Path(row["run"])
        scenario = json.loads((run / "evaluation/scenario.json").read_text())
        metadata = json.loads((run / "observations/metadata.json").read_text())
        state = json.loads((run / "run.json").read_text())
        rows.append(
            dict(
                run_id=run.name,
                run_directory=str(run),
                origin="px4_sitl",
                campaign=name,
                case_index=int(index),
                scenario=scenario["name"],
                seed=scenario["seed"],
                health_eligibility=scenario["health_eligibility"],
                paired_case_id=scenario["paired_case_id"],
                parent_attempt_id=state["parent_attempt_id"],
                continuity="independent_reset",
                asset_ids={
                    k: scenario[k]
                    for k in [
                        "vehicle_instance_id",
                        "battery_instance_id",
                        "propulsion_assembly_ids",
                    ]
                },
                requested_mission_seconds=scenario["requested_duration_s"],
                producer_status=row["contract"]["status"],
                actual_vertical_target_captured=metadata["semantic_bindings"]["climb_rate_target"][
                    "topic"
                ]
                == "vehicle_local_position_setpoint",
                selection_candidate=state["status"] == "completed"
                and row["contract"]["status"] == "passed",
                split="not_assigned_AI_responsibility",
            )
        )
dump(
    ROOT / "artifacts/dataset-manifest.json",
    dict(
        schema="x500.evaluation.dataset/1",
        usage="selection_and_evaluation_only_not_neural_features",
        C9="deferred_ai_integration",
        runs=rows,
    ),
)
print(
    json.dumps(
        dict(
            runs=len(rows),
            healthy=sum(r["health_eligibility"] == "confirmed_healthy" for r in rows),
            full_target_capture=sum(r["actual_vertical_target_captured"] for r in rows),
        )
    )
)

latest = ROOT / "artifacts/diagnostics-check.json"
if latest.exists():
    manifest_path = ROOT / "artifacts/dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["recommended_additional_healthy_run"] = json.loads(latest.read_text())["run"]
    dump(manifest_path, manifest)
