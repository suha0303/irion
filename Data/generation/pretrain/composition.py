"""Explicit collection profiles; retain the sealed per-condition safeguards."""

from collections import Counter

from .common import package_imports

SIZES = {"legacy50": 50, "train104": 104, "diverse104": 104, "balanced300": 300, "final324": 324}


def seed_stride(profile):
    return 10000 if profile == "final324" else 1000 if profile == "balanced300" else 100


def validation_subsets(cases, profile):
    if profile == "final324":
        raise ValueError("Final holdout cannot use training/development subset admission")
    if profile not in SIZES or len(cases) != SIZES[profile]:
        raise ValueError("Collection profile/count mismatch")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case identity")
    if profile == "legacy50":
        return [cases]
    if profile == "balanced300":
        from .balanced import validate

        validate(cases)
        normal = [c for c in cases if c["mission_seconds"] in (180, 900)]
        groups = {
            (role, scenario): sorted(
                [c for c in normal if c["role"] == role and c["scenario"] == scenario],
                key=lambda c: c["seed"],
            )
            for role in (
                "normal_train",
                "normal_validation",
                "normal_calibration",
                "normal_dev_eval",
            )
            for scenario in sorted({c["scenario"] for c in normal})
        }
        controls = sorted(
            [c for c in cases if c["role"] == "normal_dev_eval" and c["mission_seconds"] == 480],
            key=lambda c: c["seed"],
        )
        result = []
        for i in range(20):
            part = [v[i % len(v)] for v in groups.values()]
            pair = controls[2 * (i % 3) : 2 * (i % 3) + 2]
            seeds = {c["seed"] for c in pair}
            part += pair + [c for c in cases if c["role"] == "anomaly_dev" and c["seed"] in seeds]
            result.append(part)
        if {c["case_id"] for subset in result for c in subset} != {c["case_id"] for c in cases}:
            raise ValueError("Balanced300 legacy guard coverage incomplete")
        return result
    train = [c for c in cases if c["role"] == "normal_train"]
    other = [c for c in cases if c["role"] != "normal_train"]
    if len(train) != 63 or len(other) != 41:
        raise ValueError("train104 requires roles63/9/9/11/12")
    normal = [c for c in cases if c["role"] != "anomaly_dev" and c["mission_seconds"] in {180, 900}]
    if len(normal) != 90 or any(len({c[k] for c in normal}) != 90 for k in ("seed", "group_id")):
        raise ValueError("Independent normal runs require90 distinct seeds/groups")
    conditions = sorted({c["scenario"] for c in train})
    if len(conditions) != 9 or set(Counter(c["scenario"] for c in train).values()) != {7}:
        raise ValueError("Seven independent training seeds per normal condition required")
    groups = {
        s: sorted([c for c in train if c["scenario"] == s], key=lambda c: c["seed"])
        for s in conditions
    }
    return [[groups[s][i] for s in conditions] + other for i in range(7)]


def validate_cases(cases, profile):
    if profile == "final324":
        from .final_dataset import validate

        validate(cases)
        return
    package_imports()
    from preprocessing_tools.control import validate_cases as original

    for subset in validation_subsets(cases, profile):
        original(subset)
    if profile in {"diverse104", "balanced300"}:
        import sys

        from .common import ROOT

        sys.path.insert(0, str(ROOT / "simulation/sim/src"))
        from capstone_sim.normal_variation import sample, validate

        for case in cases:
            varied = case["role"] != "anomaly_dev" and case["mission_seconds"] in {180, 900}
            if varied:
                validate(case.get("normal_variation", {}))
                if profile == "diverse104" and case["normal_variation"] != sample(
                    case["scenario"], case["seed"]
                ):
                    raise ValueError("Normal variation differs from frozen seed and condition")
            elif "normal_variation" in case:
                raise ValueError("Paired anomaly/control definitions must remain matched")


def validate_roles(rows, cases, protected, profile, *, case_sets=None):
    package_imports()
    from preprocessing_tools.control import validate_roles as original

    if len(rows) != len(cases) or {r["case_id"] for r in rows} != {c["case_id"] for c in cases}:
        raise ValueError("Collection membership mismatch")
    for key in ("run_id", "raw_sha256", "raw_flight_id"):
        if any(not r.get(key) for r in rows) or len({r[key] for r in rows}) != len(rows):
            raise ValueError("Missing/duplicate global identity: " + key)
    by_id = {r["case_id"]: r for r in rows}
    if case_sets is None:
        case_sets = [[c["case_id"] for c in cases]]
    ids = [case_id for group in case_sets for case_id in group]
    if not case_sets or len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError("Missing/duplicate collection set membership")
    expected = {c["case_id"]: c for c in cases}
    if len(expected) != len(cases):
        raise ValueError("Duplicate case identity")
    if profile in {"diverse104", "balanced300"}:
        for row in rows:
            case = expected[row["case_id"]]
            spec = case.get("normal_variation")
            if row.get("scenario_parameters_snapshot", {}).get("normal_variation") != spec:
                raise ValueError("Executed conditions/path differ from the frozen case")
    # Check across sets too: independent subsets must not hide shared lineage.
    from preprocessing_core.control import FIT_ROLES, lineage

    owners = {}
    for index, group in enumerate(case_sets):
        for case_id in group:
            row = by_id[case_id]
            partition = row["role"] if row["role"] in FIT_ROLES else "development"
            keys = lineage(row) | {f"seed:{row['seed']}", f"group:{row['group_id']}"}
            for key in keys:
                owner = (index, partition)
                if key in owners and owners[key] != owner:
                    raise ValueError("Cross-set identity or split leakage: " + key)
                owners[key] = owner
        for subset in validation_subsets([expected[key] for key in group], profile):
            original([by_id[c["case_id"]] for c in subset], subset, protected)
