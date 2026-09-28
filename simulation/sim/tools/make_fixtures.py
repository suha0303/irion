"""Permanent producer fixtures. Does not implement AI framing or preprocessing."""

import json
from pathlib import Path

from capstone_sim.recording import TOPICS, Recorder, dump, integrity

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/fixtures"
BASE.mkdir(exist_ok=True)
run = BASE / "normal90"
for d in ["observations", "evaluation", "reference", "native_logs"]:
    (run / d).mkdir(parents=True, exist_ok=True)
dump(
    run / "run.json", dict(run_id=run.name, origin="synthetic_contract_fixture", status="completed")
)
rec = Recorder(run, "synthetic_contract_fixture")
with (run / "native_logs/capture.jsonl").open("w") as out:
    for t in range(0, 90000000, 100000):
        for topic, (_, units, _) in TOPICS.items():
            fields = {k: 0.0 for k in units}
            if topic == "actuator_outputs":
                fields = dict(output=[500.0] * 4)
            elif topic in ["vehicle_angular_velocity", "vehicle_acceleration"]:
                fields = dict(xyz=[0.0, 0.0, -9.80665 if topic == "vehicle_acceleration" else 0.0])
            elif topic == "vehicle_attitude":
                fields = dict(q=[1.0, 0.0, 0.0, 0.0])
            elif topic == "trajectory_setpoint":
                fields = dict(velocity=[None] * 3)
            elif topic == "battery_status":
                fields = dict(voltage_v=15.0, current_a=20.0)
            out.write(
                json.dumps(
                    dict(topic=topic, sample_time_us=t, available_time_us=t + 5000, fields=fields)
                )
                + "\n"
            )
        if t % 1000000 == 0:
            out.flush()
            rec.consume()
rec.close()
integrity(run)
# These focused vectors specify input/expected semantics. Only normal90 is a complete run.
rows = [json.loads(s) for s in (run / "observations/streams.jsonl").read_text().splitlines()]
battery = [e for e in rows if e["topic"] == "battery_status"]
variants = {}
late = [dict(battery[0], available_time_us=250000), dict(battery[1], available_time_us=105000)]
variants["delay_reordering"] = (
    late,
    "availability order is seq1 then seq0; sample time is not arrival order",
)
gap = [
    battery[0],
    dict(
        battery[1],
        seq=2,
        sample_time_us=30000000,
        available_time_us=30005000,
        fields=dict(voltage_v=None, current_a=None),
        field_quality=dict(voltage_v="unavailable", current_a="unavailable"),
    ),
]
variants["missing_long_gap"] = (
    gap,
    "one missing producer sequence and 30s gap; do not impute from truth",
)
reset = [battery[0], dict(battery[0], clock_id="px4-boot-2", instance=1)]
variants["clock_reset_instance"] = (reset, "new clock/instance requires a new consumer segment")
for name, (events, expected) in variants.items():
    p = BASE / name
    p.mkdir(exist_ok=True)
    (p / "streams.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    dump(p / "fixture.json", dict(origin="synthetic_contract_fixture", expected=expected))
p = BASE / "partial_end"
p.mkdir(exist_ok=True)
(p / "streams.jsonl").write_text(json.dumps(battery[0]) + "\n" + json.dumps(battery[1])[:70])
dump(
    p / "fixture.json",
    dict(
        origin="synthetic_contract_fixture",
        expected="recover 1 row; retain incomplete trailing bytes separately",
    ),
)
p = BASE / "separate_truth"
p.mkdir(exist_ok=True)
(p / "streams.jsonl").write_text(json.dumps(battery[0]) + "\n")
e = dict(
    schema="drone.evaluation.event/2.1",
    run_id="normal90",
    event_id="fixture-r0",
    kind="battery_resistance_increase",
    target="battery_pack",
    onset_kind="pre_existing",
    physical_onset_time_us=None,
    requested_time_us=0,
    applied_time_us=0,
    end_time_us=None,
    profile_revision=1,
    parameters=dict(r0_multiplier=1.4),
)
(p / "evaluation-events.jsonl").write_text(json.dumps(e) + "\n")
dump(
    p / "fixture.json",
    dict(
        origin="synthetic_contract_fixture",
        expected="observation bytes unchanged when truth is replaced; no known physical onset",
    ),
)
dump(
    BASE / "manifest.json",
    dict(
        origin="synthetic_contract_fixture",
        producer="capstone_sim",
        C9="deferred_ai_integration",
        fixtures=["normal90", *variants, "partial_end", "separate_truth"],
    ),
)
print(BASE)
