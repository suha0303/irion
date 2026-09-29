"""Producer-only asset identity fixtures. No trend estimator or AI split."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "artifacts/remediation-20261014/asset-fixtures"
out.mkdir(exist_ok=True)
records = [
    dict(
        run_id="asset-a",
        vehicle_instance_id="x500-asset-fixture",
        battery_instance_id="pack-a",
        propulsion_assembly_ids=["motor-a", "motor-b", "motor-c", "motor-d"],
        continuity="independent_reset",
        replacement_event=None,
        comparable_operation={"mission": "hover", "ambient_temperature_k": 298.15},
    ),
    dict(
        run_id="asset-b",
        vehicle_instance_id="x500-asset-fixture",
        battery_instance_id="pack-b",
        propulsion_assembly_ids=["motor-a2", "motor-b", "motor-c", "motor-d"],
        continuity="independent_reset",
        replacement_event={
            "battery": {"removed": "pack-a", "installed": "pack-b"},
            "front_right": {"removed": "motor-a", "installed": "motor-a2"},
        },
        comparable_operation={"mission": "hover", "ambient_temperature_k": 298.15},
    ),
    dict(
        run_id="asset-c",
        vehicle_instance_id="x500-asset-fixture",
        battery_instance_id="pack-b",
        propulsion_assembly_ids=["motor-a2", "motor-b", "motor-c", "motor-d"],
        continuity="unknown",
        replacement_event=None,
        comparable_operation={"mission": "hover", "ambient_temperature_k": None},
    ),
]
fixture = dict(
    origin="synthetic_contract_fixture",
    usage="asset_selection_and_evaluation_only_not_features",
    records=records,
    expected={
        "asset-a_to_asset-b": "break_at_replacement_and_independent_reset",
        "asset-b_to_asset-c": "insufficient_continuity_and_temperature_no_trend_claim",
    },
)
(out / "asset-registry.json").write_text(json.dumps(fixture, indent=2) + "\n")
assert records[0]["battery_instance_id"] == records[1]["replacement_event"]["battery"]["removed"]
assert records[1]["battery_instance_id"] == records[1]["replacement_event"]["battery"]["installed"]
assert (
    records[1]["propulsion_assembly_ids"][0]
    == records[1]["replacement_event"]["front_right"]["installed"]
)
assert records[2]["comparable_operation"]["ambient_temperature_k"] is None
(out / "producer-check.json").write_text(
    json.dumps(
        dict(
            passed=True,
            origin="synthetic_contract_fixture",
            consumer_trend_test="AI_responsibility_not_claimed",
        ),
        indent=2,
    )
    + "\n"
)
print(out)
