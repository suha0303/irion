"""Data projections for the AI-owned one-shot gates; no models or native-field guesses."""

import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

from . import ai19
from .common import ROOT, read, sha, verify, write
from .pilot19 import COHORT, FINAL16_COHORT, SHORT_COHORT


class Pending(RuntimeError):
    pass


def ref(path):
    path = ai19.no_links(path)
    return dict(path=str(path), sha256=sha(path))


def dependencies(path):
    """Only owner-delivered final paths/CLI templates can enable the pipeline."""
    if not Path(path).is_file():
        raise Pending("AI/native final handoff missing: " + str(path))
    value = read(path)
    if value.get("schema") == "drone-ai.final16.data-handoff/1":
        from .final16 import dependencies as final16_dependencies

        return final16_dependencies(value)
    if value.get("schema") == "drone-ai.short-cycle.data-handoff/1":
        return short_dependencies(value)
    if (
        value.get("schema") != "drone-ai.pilot45.data-handoff/1"
        or value.get("task_id") != "P27_PILOT45_20261107"
    ):
        raise ValueError("Reviewed AI/native handoff schema required")
    if value.get("status") in {"FAILED", "STOPPED"}:
        raise ValueError("Final dependency failed; no blind wait/retry")
    if value.get("status") != "READY":
        raise Pending(value.get("reason", "Final owner readiness not confirmed"))
    paths = value["paths"]
    for key in ("native_completion", "native_policy", "paired_policy", "development_receipt"):
        actual = ref(paths[key])
        if actual["sha256"] != value["sha256"][key]:
            raise ValueError("Final handoff evidence changed: " + key)
    ai19.ai_modules()
    from learning.history19.pilot import check_native_completion

    check_native_completion(read(paths["native_completion"]))
    binding = ai19.producer_binding()
    from learning.admission19 import check_producer

    check_producer(value["producer"], binding)
    native = read(paths["native_completion"])
    if (
        native["source_sha256"] != binding["source_manifest_sha256"]
        or native["runtime_sha256"] != binding["runtime_sha256"]
    ):
        raise ValueError("Actual native6 completion belongs to another producer")
    for rid in native["actual_run_ids"]:
        if read(ROOT / "simulation/runs" / rid / "run.json").get("status") != "completed":
            raise ValueError("Actual development raw missing/not completed")
    member = Path(value["origin_relative_path"])
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("Reviewed native origin relative member required")
    if not isinstance(value.get("paired_cli_options"), list) or not value["paired_cli_options"]:
        raise Pending("Reviewed new-policy flight CLI options missing")
    help_text = subprocess.run(
        [str(ROOT / "simulation/capstone-sim"), "run", "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    for token in value["paired_cli_options"]:
        if token.startswith("--") and (
            token not in help_text
            or not any(x in token for x in ("phase", "origin", "paired", "schedule"))
        ):
            raise ValueError("Only actual reviewed origin/phase CLI options allowed")
    if not value.get("pair_verifier_argv"):
        raise Pending("Actual native pair verifier CLI template missing")
    return value


def short_dependencies(value):
    """Validate the owner's separate static180 contract, never historical native6 READY."""
    if value.get("task_id") != "SHORT_CYCLE_20261107":
        raise ValueError("Explicit short-cycle task required")
    if value.get("status") in {"FAILED", "STOPPED"}:
        raise ValueError("Short static handoff failed; no blind retry")
    if value.get("status") != "READY":
        raise Pending("Final short static owner handoff is not READY")
    if (
        value.get("code_frozen") is not True
        or value.get("duration_profile") != "short180"
        or value.get("mission_seconds") != 180
        or value.get("expected_cases") != 45
        or value.get("expected_pairs") != 9
        or value.get("limits")
        != dict(
            flight_workers=16, RTF=2, aggregate_nonflight_workers=4, max_attempts_per_logical_case=3
        )
        or value.get("no_extra_development6") is not True
        or value.get("no_LR108_or_main_training_link") is not True
    ):
        raise ValueError("Explicit frozen short180/45/9pair/16flight/RTF2/CPU4/max3 scope required")
    frozen_ref = value["source_freeze"]
    if ref(frozen_ref["path"])["sha256"] != frozen_ref["sha256"]:
        raise ValueError("Owner AI source freeze changed")
    frozen = read(frozen_ref["path"])
    if frozen.get("code_frozen") is not True or frozen.get("task_id") != value["task_id"]:
        raise ValueError("Explicit final owner AI source freeze required")
    for name, digest in frozen["files"].items():
        if not name.startswith("ai/") or ".." in Path(name).parts:
            raise ValueError("Only actual AI source paths in owner freeze")
        verify({str(ai19.no_links(ROOT / name)): digest})
    for item in value["simulation_evidence"].values():
        if ref(item["path"])["sha256"] != item["sha256"]:
            raise ValueError("Owner simulation handoff/contract changed")
    paths = value["paths"]
    if not {"native_policy", "paired_policy"} <= set(paths) or any(
        k in paths for k in ("native_completion", "development_receipt")
    ):
        raise ValueError("New short policies required without historical development6 dependency")
    for name, path in paths.items():
        if ref(path)["sha256"] != value["sha256"][name]:
            raise ValueError("Final short handoff evidence changed: " + name)
    ai19.ai_modules()
    from learning.admission19 import check_producer

    binding = ai19.producer_binding()
    check_producer(value["producer"], binding)
    policy = read(paths["paired_policy"])
    native = read(paths["native_policy"])
    check_producer(policy["producer"], binding)
    if (
        policy.get("approved_for_short_cycle") is not True
        or policy.get("approved_for_pilot") is not False
        or policy.get("scope_task") != "SHORT_CYCLE_20261107"
        or policy.get("purpose") != SHORT_COHORT
        or policy.get("duration_profile") != "short180"
        or policy.get("fixture_only") is not False
        or policy.get("native_adapter") != "joint-native-consumption-c2-v2"
        or policy.get("native_adapter") != native.get("revision")
        or policy.get("native_policy_sha256") != sha(paths["native_policy"])
        or set(native.get("supported_duration_us", [])) != {180000000, 480000000}
    ):
        raise ValueError("Actual v2 short180 policy/current generation binding required")
    member = Path(value["origin_relative_path"])
    if member.is_absolute() or ".." in member.parts or str(member) != policy.get("mission_source"):
        raise ValueError("Actual native short mission-origin member required")
    options = value.get("paired_cli_options")
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("Reviewed phase and pair options only")
    parsed = dict(zip(options[::2], options[1::2], strict=True))
    if parsed != {
        "--phase-policy": native["revision"],
        "--paired-case-id": "{paired_case_id}",
    }:
        raise ValueError("Actual v2 phase and independent pair identity required")
    argv = value.get("pair_verifier_argv", [])
    if (
        len(argv) != 8
        or ai19.no_links(argv[0]) != ROOT / "simulation/capstone-sim"
        or argv[1:]
        != [
            "pair-check",
            "--normal-run",
            "{normal_run}",
            "--anomaly-run",
            "{anomaly_run}",
            "--output",
            "{output}",
        ]
    ):
        raise ValueError("Actual native read-only pair-check argv required")
    help_text = subprocess.run(
        [str(ROOT / "simulation/capstone-sim"), "run", "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    if not all(option in help_text for option in parsed):
        raise ValueError("Final short flight CLI options are not available")
    return value


def policy_bundle(spec):
    return {k: spec["paths"][k] for k in ("paired_policy", "native_policy")} | dict(
        origin_relative_path=spec["origin_relative_path"]
    )


def qualified(task, case):
    return read(task / "config.json").get("lineage_namespace", task.name) + ":" + case["case_id"]


def plan(task):
    cfg = read(task / "config.json")
    rows = []
    for c in read(task / "manifest.json")["cases"]:
        namespace = cfg.get("lineage_namespace", task.name)
        rows.append(
            dict(
                case_id=qualified(task, c),
                case_kind={
                    "standalone_normal": "normal_base",
                    "paired_normal": "paired_normal",
                    "anomaly": "anomaly",
                }[c["category"]],
                role=c["role"],
                seed=c["seed"],
                motion=c["kind"],
                environment=c["environment"],
                operation_sha256=hashlib.sha256(
                    json.dumps(
                        c["operation_variation"], sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                pair_id=c["paired_case_id"],
                health_family=c["anomaly_family"],
                timing=c["anomaly_timing"],
                lineage=[
                    "pair:" + c["paired_case_id"],
                    "group:" + namespace + ":" + c["group_id"],
                    "retry:" + namespace + ":" + c["retry_family_id"],
                ],
            )
        )
    return dict(
        schema="drone-ai.final16.manifest/1"
        if cfg.get("final16")
        else "drone-ai.pilot45.manifest/1",
        task_id=cfg["task_id"],
        cohort=FINAL16_COHORT
        if cfg.get("final16")
        else SHORT_COHORT
        if cfg.get("short_cycle")
        else COHORT,
        purpose=FINAL16_COHORT
        if cfg.get("final16")
        else SHORT_COHORT
        if cfg.get("short_cycle")
        else COHORT,
        no_fit=True,
        status="FROZEN",
        cases=rows,
    )


def snapshot(task, directory):
    manifest = read(task / "ai-manifest.json")
    local = read(task / "ledger.json")
    registry = copy.deepcopy(read(task / "global-protection.json"))
    by_id = {qualified(task, c): c for c in read(task / "manifest.json")["cases"]}
    rows = []
    for case in manifest["cases"]:
        source = local["cases"].get(by_id[case["case_id"]]["case_id"], {})
        attempts = []
        for a in source.get("attempts", []):
            state = (
                "completed"
                if a["state"].startswith("COMPLETED")
                else "running"
                if a["state"] == "RUNNING"
                else "failed"
            )
            attempts.append(
                dict(
                    index=a["number"],
                    status=state,
                    run_id=a.get("run_id"),
                    seed=case["seed"],
                    lineage=case["lineage"],
                    parent_attempt_id=a["parent_attempt_id"],
                    failure_reason=a.get("error"),
                )
            )
            if a.get("quality_rejection"):
                if not read(task / "config.json").get("short_cycle") or a["state"] != "COMPLETED":
                    raise ValueError(
                        "Quality rejection projection requires short physical COMPLETED"
                    )
                attempts[-1].update(
                    status="quality_rejected",
                    physical_status="completed",
                    quality_rejection=a["quality_rejection"],
                )
            if a.get("run_id"):
                registry["entries"].append(
                    dict(
                        identities=["run:" + a["run_id"]],
                        forbidden_uses=[
                            "fit",
                            "input_check",
                            "calibration",
                            "final",
                            "final_test",
                            "final_input_check",
                        ],
                        purpose=FINAL16_COHORT
                        if read(task / "config.json").get("final16")
                        else SHORT_COHORT
                        if read(task / "config.json").get("short_cycle")
                        else COHORT,
                    )
                )
        status = (
            "completed"
            if source.get("status") == "COMPLETE"
            else "running"
            if attempts
            else "pending"
        )
        item = dict(case_id=case["case_id"], status=status, attempts=attempts)
        if source.get("status") == "COMPLETE" and not (
            source["online_available"] and source["run_available"]
        ):
            item.update(
                status="unavailable",
                unavailable_reasons=read(source["receipt"]).get("run_health", {}).get("reasons", [])
                + ["one_or_both_AI_views_unavailable"],
            )
        rows.append(item)
    write(directory / "registry.json", registry)
    write(
        directory / "ledger.json",
        dict(
            schema="drone-ai.final16.ledger/1"
            if read(task / "config.json").get("final16")
            else "drone-ai.pilot45.ledger/1",
            task_id=manifest["task_id"],
            cases=rows,
        ),
    )


def capsule(task, spec):
    ai19.ai_modules()
    from learning.history19.pilot import CRITICAL

    if read(task / "config.json").get("final16"):
        from learning.history19.final16 import CRITICAL

    files = {name: sha(ROOT / name) for name in CRITICAL}
    files.update(
        {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "Data/generation/pretrain").glob("*.py")}
    )
    return dict(
        schema="drone-ai.consumer-capsule/1",
        producer=read(task / "expected-producer.json"),
        paired_policy_sha256=sha(spec["paths"]["paired_policy"]),
        Data_entrypoint="Data/generation/pretrain/pilot45.py",
        files=files,
    )


def check(task, spec, *, case_id=None, pair_path=None, phase=None):
    ai19.ai_modules()
    from learning.history19.pilot import main
    from learning.history19.pilot_connector import prepare_request

    short = read(task / "config.json").get("short_cycle") is True
    final16 = read(task / "config.json").get("final16") is True
    phase = phase or ("prerequisites" if case_id is None else "case")

    directory = task / "pilot/admission" / (str(time.time_ns()) + "-" + phase)
    directory.mkdir(parents=True)
    snapshot(task, directory)
    paths = dict(
        spec["paths"],
        expected_producer=str(task / "expected-producer.json"),
        source_manifest=str(ROOT / "simulation/config/producer-sources.json"),
        runtime_manifest=str(ROOT / "simulation/artifacts/runtime-manifest.json"),
        capsule=str(task / "consumer-capsule.json"),
        registry=str(directory / "registry.json"),
        pilot_manifest=str(task / "ai-manifest.json"),
        ledger=str(directory / "ledger.json"),
    )
    bundle = dict(
        schema="drone-ai.pilot45.connector/1",
        task_id=read(task / "config.json")["task_id"],
        phase=phase,
        paths=paths,
    )
    if case_id is not None:
        entry = read(task / "ledger.json")["cases"][case_id]
        raw, output = Path(entry["run"]), Path(entry["output"])
        case = entry["case"]
        bundle.update(
            case_id=qualified(task, case),
            case_origin=dict(
                run=str(raw / "run.json"),
                execution=str(raw / "evaluation/execution.json"),
                origin=str(raw / spec["origin_relative_path"]),
            ),
        )
        paths.update(
            scenario=str(raw / "evaluation/scenario.json"),
            raw_integrity=str(raw / "integrity.json"),
            battery_selection=str(raw / "evaluation/battery-selection.json"),
            propulsion_selection=str(raw / "evaluation/propulsion-selection.json"),
            canonical_frames=str(output / "frames.jsonl"),
            online_manifest=str(output / "online.json"),
            run_manifest=str(output / "run.json"),
        )
        if pair_path is not None:
            paths["pair_receipt"] = str(pair_path)
    if final16:
        from learning.history19.final16 import main, prepare_request

        bundle["schema"] = "drone-ai.final16.connector/1"
    if short:
        from learning.history19.short_cycle import main, prepare_request

        bundle["schema"] = "drone-ai.short-cycle.connector/1"
        if case_id is not None and entry.get("external_audit") is not None:
            bundle["external_audit"] = entry["external_audit"]
    if short or final16:
        if phase == "final":
            receipts = []
            for entry in read(task / "ledger.json")["cases"].values():
                checked = read(entry["ai_receipt"])
                actual = ref(checked["receipt"])
                if actual["sha256"] != checked["sha256"]:
                    raise ValueError("Actual scoped case receipt changed before final check")
                receipts.append(actual["path"])
            bundle["case_receipts"] = receipts
    elif phase == "final":
        raise ValueError("The explicit short final gate must not alter480s pilot contracts")
    write(directory / "bundle.json", bundle)
    prepare_request(directory / "bundle.json", directory / "request.json")
    code = main(directory / "request.json", directory / "receipt.json")
    result = read(directory / "receipt.json")
    if code != 0 or result.get("fixture_only") or result.get("actual_data_verified") is not True:
        raise ValueError("AI input gate did not pass: " + str(directory / "receipt.json"))
    return dict(
        receipt=str(directory / "receipt.json"),
        sha256=sha(directory / "receipt.json"),
        result=result,
    )


def pair(task, spec, normal_id, anomaly_id):
    """Execute an owner-delivered native verifier, then map its fields by AI policy."""
    ledger = read(task / "ledger.json")["cases"]
    normal, anomaly = ledger[normal_id], ledger[anomaly_id]
    directory = task / "pilot/pairs" / normal["case"]["group_id"]
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "native.json"
    values = dict(normal_run=normal["run"], anomaly_run=anomaly["run"], output=str(output))
    argv = [part.format(**values) for part in spec["pair_verifier_argv"]]
    executable = ai19.no_links(argv[0])
    if (
        executable != ROOT / "simulation/capstone-sim"
        or len(argv) < 2
        or not any(x in argv[1] for x in ("verify", "check"))
    ):
        raise ValueError(
            "Reviewed Sim pair verification CLI required; flight/control commands forbidden"
        )
    if not output.exists():
        with (directory / "verify.log").open("x") as log:
            subprocess.run(
                argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=True
            )
    ai19.ai_modules()
    from learning.history19.pilot_connector import project_pair_receipt

    projected = project_pair_receipt(
        output,
        policy_path=spec["paths"]["paired_policy"],
        normal_run=normal["run"],
        anomaly_run=anomaly["run"],
    )
    result = dict(
        projected,
        schema="drone-ai.pilot-pair-check/1",
        producer=read(task / "expected-producer.json"),
        paired_policy_sha256=sha(spec["paths"]["paired_policy"]),
        pair_id=normal["case"]["paired_case_id"],
        native_evidence=ref(output),
    )
    if (
        result["normal_run_id"] != Path(normal["run"]).name
        or result["anomaly_run_id"] != Path(anomaly["run"]).name
    ):
        raise ValueError("Native pair verifier returned another pair")
    write(directory / "projection.json", result)
    return directory / "projection.json"
