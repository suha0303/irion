"""Two joint propulsion profiles, with fixed or simultaneous linear transitions."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "profiles/propulsion-health.json"
SCENARIOS = {"propulsion_loss_existing", "propulsion_loss_step", "propulsion_loss_ramp"}
DYNAMIC = SCENARIOS - {"propulsion_loss_existing"}
TRAINING_STAGES = (4, 8, 12, 16)
KEYS = ("kt_prop", "kq_prop", "friction", "coulomb", "esc_efficiency", "resistance")
TRUTH = ("kt_prop", "kq_prop", "friction_nms", "motor_coulomb_nm",
         "esc_efficiency", "motor_resistance_ohm")


def validate_snapshot(snapshot):
    if set(snapshot) != set(KEYS) or any(
        type(v) not in (float, int) or not math.isfinite(v) for v in snapshot.values()
    ):
        raise ValueError("Invalid propulsion snapshot fields")
    if (any(snapshot[k] <= 0 for k in ("kt_prop", "kq_prop", "resistance", "esc_efficiency"))
            or snapshot["friction"] < 0 or snapshot["coulomb"] < 0
            or snapshot["esc_efficiency"] > 1):
        raise ValueError("Invalid propulsion snapshot parameters")


def load_catalog(path=CATALOG):
    catalog = json.loads(Path(path).read_text())
    if (catalog.get("schema") != "drone.propulsion-health/1"
            or set(catalog.get("states", {})) != {"normal", "abnormal"}):
        raise ValueError("Exactly normal and abnormal propulsion profiles required")
    a, b = (catalog["states"][k] for k in ("normal", "abnormal"))
    validate_snapshot(a)
    validate_snapshot(b)
    if (b["resistance"] <= a["resistance"] or b["friction"] <= a["friction"]
            or b["kt_prop"] >= a["kt_prop"] or b["kq_prop"] <= a["kq_prop"]
            or any(a[k] != b[k] for k in ("coulomb", "esc_efficiency"))):
        raise ValueError("All three propulsion impairments must be applied together")
    return catalog


def read_runtime(path):
    values = list(map(float, Path(path).read_text().split()))
    if len(values) != len(KEYS):
        raise ValueError("Six propulsion runtime values required")
    snapshot = dict(zip(KEYS, values, strict=True))
    validate_snapshot(snapshot)
    return snapshot


def write_runtime(path, snapshot):
    validate_snapshot(snapshot)
    Path(path).write_text(" ".join(str(snapshot[k]) for k in KEYS) + "\n")


def resolve(scenario, normal=None, transition_s=None, *, catalog_path=CATALOG, mavlink_fixed_stage=None):
    catalog = load_catalog(catalog_path)
    baseline = catalog["states"]["normal"]
    target = dict(catalog["states"]["abnormal"])
    normal = dict(baseline if normal is None else normal)
    validate_snapshot(normal)
    if normal != baseline:
        # Keep prepared custom aircraft's healthy model intact. This transfer is an
        # engineering experiment, not validation of another motor's aging behavior.
        for key in ("resistance", "kt_prop", "kq_prop"):
            target[key] = normal[key] * target[key] / baseline[key]
        target["friction"] = normal["friction"] + (
            target["friction"] - baseline["friction"]) * normal["kq_prop"] / baseline["kq_prop"]
        for key in ("coulomb", "esc_efficiency"):
            target[key] = normal[key]
    dynamic = scenario in DYNAMIC
    if transition_s is not None and (not dynamic or type(transition_s) not in (int, float)
                                    or not math.isfinite(transition_s) or not 0 < transition_s <= 1800):
        raise ValueError("Propulsion transition requires a dynamic scenario and positive duration")
    abnormal = scenario in SCENARIOS
    target = target if abnormal else normal
    if mavlink_fixed_stage is not None:
        if (type(mavlink_fixed_stage) is not int or mavlink_fixed_stage not in TRAINING_STAGES
                or scenario != "propulsion_loss_existing"):
            raise ValueError("MAVLink propulsion stages require pre-existing training wear")
        if mavlink_fixed_stage != 16:
            fraction = mavlink_fixed_stage / 16
            target = {key: normal[key] + fraction * (target[key] - normal[key]) for key in KEYS}
    seconds = (transition_s if transition_s is not None else
               catalog["transition_defaults_s"]["step" if scenario.endswith("step") else "ramp"]
               ) if dynamic else 0.0
    return dict(schema="drone.propulsion-selection/1", catalog_revision=catalog["revision"],
                catalog_sha256=hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(),
                normal=normal, initial=normal if dynamic else target, target=target,
                initial_state="normal" if dynamic or not abnormal else "abnormal",
                target_state="abnormal" if abnormal else "normal", transition_s=seconds,
                **({"mavlink_fixed_stage": mavlink_fixed_stage} if mavlink_fixed_stage is not None else {}),
                fixed_for_flight=not dynamic, interpolation="linear simultaneously in Rm,b,kF,kQ",
                basis=catalog["abnormal_basis"] if abnormal else catalog["normal_basis"],
                limitations=catalog["limitations"])


def verify_selection(selection, catalog_path, scenario, expected_hash):
    if hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Propulsion catalog differs from sealed producer")
    # Collection uses the exact X500 pair; no arbitrary per-run health values.
    expected = resolve(scenario, transition_s=selection.get("transition_s")
                       if scenario in DYNAMIC else None, catalog_path=catalog_path)
    if selection != expected:
        raise ValueError("Propulsion selection differs from its two-state policy")


def verify_truth(path, selection):
    count = 0
    last = 0.0
    dynamic = not selection["fixed_for_flight"]
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            fraction = row["propulsion_profile_fraction"]
            onset = row["propulsion_profile_onset_us"]
            expected = max(0.0, min(1.0, (row["time_us"] - onset) /
                           (selection["transition_s"] * 1e6))) if dynamic and onset >= 0 else 0.0
            tolerance = max(1e-8, 2e-6 / selection["transition_s"]) if dynamic else 1e-10
            if (not math.isfinite(fraction) or not 0 <= fraction <= 1 or fraction < last - 1e-10
                    or not math.isclose(fraction, expected, abs_tol=tolerance)):
                raise ValueError("Propulsion profile is not the prescribed linear path")
            for key, truth_key in zip(KEYS, TRUTH, strict=True):
                a, b = selection["initial"][key], selection["target"][key]
                if not math.isclose(row[truth_key], a + fraction * (b-a), rel_tol=1e-9, abs_tol=1e-14):
                    raise ValueError("Propulsion parameter differs from its profile: " + key)
            count += 1
            last = fraction
    if not count or (dynamic and last < 1 - 1e-9):
        raise ValueError("Propulsion profile endpoint was not observed")
    return dict(status="passed", samples=count, final_fraction=last,
                initial_state=selection["initial_state"], target_state=selection["target_state"])
