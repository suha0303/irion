"""CMU snapshot selection and immutable per-run physics inputs (no AI features)."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "profiles/battery-aging/cmu-vah01.json"
# combined_existing remains here only to interpret sealed historical battery selections.
# It is absent from the runner's executable scenarios; use a propulsion scenario + rpt_16.
SCENARIOS = {"battery_aging_existing", "battery_aging_step", "battery_aging_ramp", "combined_existing"}
DYNAMIC = {"battery_aging_step", "battery_aging_ramp"}
SOURCE_SHA256 = "596ab1e1e80eeac3509f789609f751527c2073a4310deec8274cbfaf4fef39c4"
NORMAL_STAGE = "rpt_00"
AGED_STAGE = "rpt_16"
FLIGHT_POLICY = {
    "revision": "cmu-vah01-rpt16-only/1",
    "normal_stage": NORMAL_STAGE,
    "aged_stage": AGED_STAGE,
}


def finite(value, *, positive=True):
    return (type(value) in (float, int) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def load_catalog(path=None):
    path = Path(path) if path is not None else CATALOG
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Battery aging inputs cannot use symlinks")
    catalog = json.loads(path.read_text())
    if (catalog.get("schema") != "drone.battery-aging/2"
            or catalog.get("holdout_status") != "passed"
            or catalog.get("source", {}).get("sha256") != SOURCE_SHA256):
        raise ValueError("Unverified CMU battery aging catalog")
    stages = catalog.get("stages", [])
    if "flight_policy" in catalog and catalog["flight_policy"] != FLIGHT_POLICY:
        raise ValueError("Unsupported CMU battery flight policy")
    if not stages or len({s["id"] for s in stages}) != len(stages):
        raise ValueError("Missing or duplicate battery aging snapshots")
    for index, stage in enumerate(stages):
        cell = stage["cell"]
        curve = cell.get("ocv_cell_v", [])
        if (stage.get("id") != f"rpt_{index:02d}"
                or stage.get("holdout_status") != "passed"
                or not all(finite(cell.get(k)) for k in
                           ("r0_ohm", "r1_ohm", "c1_f", "capacity_ah"))
                or len(curve) != 31 or not all(finite(v) and v < 5 for v in curve)
                or any(a > b for a, b in zip(curve, curve[1:]))):
            raise ValueError("Invalid or failed CMU battery snapshot")
    return catalog


def validate_selection(scenario, stage, transition_s, configuration):
    motor = scenario.startswith(("motor_", "propulsion_", "propeller_"))
    if stage is not None and stage != AGED_STAGE:
        raise ValueError("Aged flights use rpt_16 only; omit --battery-stage for normal health")
    if scenario not in SCENARIOS:
        if transition_s is not None or (stage is not None and not motor):
            raise ValueError("Stage requires a battery or motor-health scenario; transition requires battery_aging")
        return
    if configuration not in (None, "collection"):
        raise ValueError("CMU aging requires the built-in X500")
    if scenario not in DYNAMIC:
        if transition_s is not None:
            raise ValueError("An existing battery snapshot is fixed for the entire flight")
    elif transition_s is not None and not finite(transition_s):
        raise ValueError("Battery transition duration must be positive and finite")


def pack_snapshot(catalog, stage_id, nominal_pack):
    stage = next((s for s in catalog["stages"] if s["id"] == stage_id), None)
    if stage is None:
        raise ValueError("Unknown measured battery stage: " + str(stage_id))
    cell = stage["cell"]
    ns, parallel = nominal_pack["ns"], nominal_pack["np"]
    wire = nominal_pack.get("interconnect_resistance_ohm", 0.0)
    return {
        "r0_ohm": ns / parallel * cell["r0_ohm"] + wire,
        "r1_ohm": ns / parallel * cell["r1_ohm"],
        "c1_f": parallel / ns * cell["c1_f"],
        "capacity_ah": parallel * cell["capacity_ah"],
        "ocv_cell_v": list(cell["ocv_cell_v"]),
    }


def resolve(scenario, stage_id, transition_s, nominal_pack):
    validate_selection(scenario, stage_id, transition_s, None)
    catalog = load_catalog()
    if catalog.get("flight_policy") != FLIGHT_POLICY:
        raise ValueError("Current X500 requires the rpt_16-only battery flight policy")
    selected_stage = AGED_STAGE if scenario in SCENARIOS else stage_id or NORMAL_STAGE
    initial = pack_snapshot(catalog, NORMAL_STAGE, nominal_pack)
    target = pack_snapshot(catalog, selected_stage, nominal_pack)
    dynamic = scenario in DYNAMIC
    seconds = (transition_s if transition_s is not None
               else 1.0 if scenario.endswith("step") else 20.0) if dynamic else 0.0
    return {
        "schema": "drone.battery-aging-selection/1",
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "catalog_revision": catalog["revision"],
        "flight_policy": dict(FLIGHT_POLICY),
        "source": catalog["source"],
        "initial_stage": NORMAL_STAGE if dynamic else selected_stage,
        "target_stage": selected_stage,
        "initial": initial if dynamic else target,
        "target": target,
        "transition_s": seconds,
        "interpolation": "linear in R0, R1, C1, Q and every OCV knot",
        "fixed_for_flight": not dynamic,
        "state_mapping": "SOC and polarization voltage continuous; accessible charge/energy changes ledgered",
        "nominal_capacity_ah": nominal_pack["capacity_ah"],
        "ns": nominal_pack["ns"], "np": nominal_pack["np"],
        "limitations": catalog["limitations"],
    }


def write_runtime(path, pack):
    values = [pack[k] for k in ("r0_ohm", "r1_ohm", "c1_f", "capacity_ah")]
    Path(path).write_text(" ".join(map(str, values + pack["ocv_cell_v"])) + "\n")


def verify_selection(selection, catalog_path, nominal_pack, scenario, expected_hash):
    catalog_path = Path(catalog_path)
    if (hashlib.sha256(catalog_path.read_bytes()).hexdigest() != expected_hash
            or selection.get("catalog_sha256") != expected_hash):
        raise ValueError("CMU run catalog differs from its sealed producer")
    catalog = load_catalog(catalog_path)
    normal = scenario.endswith("normal")
    dynamic = scenario in DYNAMIC
    initial_id = selection.get("initial_stage")
    target_id = selection.get("target_stage")
    # Older sealed catalogs retain their original multi-stage interpretation.
    # In new catalogs the hash also binds the user's rpt_16-only flight policy.
    if "flight_policy" in catalog and (
        selection.get("flight_policy") != catalog["flight_policy"]
        or target_id not in (NORMAL_STAGE, AGED_STAGE)
        or (scenario in SCENARIOS and target_id != AGED_STAGE)
    ):
        raise ValueError("CMU battery selection violates the rpt_16-only flight policy")
    if ((normal and (initial_id != "rpt_00" or target_id != "rpt_00"))
            or (not dynamic and initial_id != target_id)
            or (scenario in SCENARIOS and target_id == "rpt_00")
            or (dynamic and initial_id != "rpt_00")
            or selection.get("fixed_for_flight") != (not dynamic)):
        raise ValueError("CMU battery selection does not match its scenario")
    for key, identity in (("initial", initial_id), ("target", target_id)):
        if selection.get(key) != pack_snapshot(catalog, identity, nominal_pack):
            raise ValueError("CMU battery parameters were changed after selection")
    if (selection.get("catalog_revision") != catalog["revision"]
            or selection.get("nominal_capacity_ah") != nominal_pack["capacity_ah"]):
        raise ValueError("CMU battery provenance/nameplate mismatch")


def verify_truth(path, selection):
    """Check native snapshots against the requested linear path and final endpoint."""
    count = 0
    last_fraction = 0.0
    first_time = None
    onset = None
    dynamic = not selection["fixed_for_flight"]
    keys = ("r0_ohm", "r1_ohm", "c1_f", "capacity_ah")
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            fraction = row.get("battery_stage_fraction")
            if not finite(fraction, positive=False) or not 0 <= fraction <= 1:
                raise ValueError("Missing or invalid native battery stage fraction")
            if fraction + 1e-10 < last_fraction:
                raise ValueError("Battery stage moved backwards")
            last_fraction = fraction
            first_time = row["time_us"] if first_time is None else first_time
            if dynamic:
                scheduled = row.get("battery_stage_onset_us", -1)
                expected_fraction = (max(0.0, min(1.0, (row["time_us"] - scheduled)
                                                / (selection["transition_s"] * 1e6)))
                                     if scheduled >= 0 else 0.0)
                # Native truth timestamps are integer microseconds, while the
                # schedule evaluates double simulation seconds before logging.
                tolerance = max(1e-8, 2e-6 / selection["transition_s"])
                if not math.isclose(fraction, expected_fraction, abs_tol=tolerance):
                    raise ValueError("Battery profile is not linear in scheduled simulation time")
            elif fraction != 0:
                raise ValueError("A fixed battery profile changed during flight")
            if dynamic and row.get("fault_applied"):
                onset = row["fault_applied_time_us"] if onset is None else onset
            for key in keys:
                start = selection["initial"][key]
                expected = start + fraction * (selection["target"][key] - start)
                if not math.isclose(row[key], expected, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("Native battery parameter differs from selected stage: " + key)
            expected_ocv = [a + fraction * (b - a) for a, b in
                            zip(selection["initial"]["ocv_cell_v"],
                                selection["target"]["ocv_cell_v"])]
            observed = row.get("battery_ocv_cell_v", [])
            if len(observed) != 31 or any(not math.isclose(a, b, abs_tol=1e-9)
                                         for a, b in zip(observed, expected_ocv)):
                raise ValueError("Native battery OCV differs from selected stage")
    if not count or (dynamic and (onset is None or last_fraction < 1 - 1e-9)):
        raise ValueError("Battery stage endpoint was not observed before run end")
    return {"status": "passed", "samples": count, "fixed_for_flight": not dynamic,
            "target_fraction": last_fraction, "first_time_us": first_time}
