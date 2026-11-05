"""One-shot CPU publisher from actual final native6 evidence to Data readiness.

No flight, model, fit, subprocess, wait, polling or callback. Run after the Sim
owner supplies completed evidence; Data may invoke it from its finite program.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from learning.admission19 import MISSION_POLICY, PILOT, check_producer

from .joint_native import ADAPTER, ORIGIN, WIND_CLOCK, check_policy, origin, pair
from .pilot import (
    RECOVERY_AUTH,
    ROOT,
    TASK,
    Evidence,
    Pending,
    check_native_completion,
    development_check,
    json_file,
    policy_check,
    regular,
    sha,
)


def reference(path):
    path = regular(path)
    return dict(path=str(path), sha256=sha(path))


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


class PublicationPathError(ValueError):
    pass


def pointer(path, value):
    """Atomic owned pointer update; archive previous content instead of erasing it."""
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Regular AI-owned handoff path required")
    if path.exists():
        previous = json_file(path)
        if previous == value:
            return
        archive = path.with_name(path.stem + ".history-" + sha(path) + ".json")
        if not archive.exists():
            write_new(archive, previous)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def check_outputs(output_dir, handoff, *, fixture=False):
    paths = tuple(Path(p).absolute() for p in (output_dir, handoff))
    if not fixture and any(not p.is_relative_to(ROOT / "ai") for p in paths):
        raise PublicationPathError("Publication writes only inside the AI project")
    if any(".." in p.parts or any(q.is_symlink() for q in (p, *p.parents)) for p in paths):
        raise PublicationPathError("No symlink/traversal publication outputs")
    if paths[1].is_relative_to(paths[0]):
        raise PublicationPathError(
            "Mutable handoff must be outside immutable publication snapshots"
        )
    return paths


def publish(
    *,
    native_completion,
    native_policy,
    pair_checks,
    producer_binding,
    source_manifest,
    runtime_manifest,
    output_dir,
    handoff,
    fixture_only=False,
):
    """Return (exit_code, receipt); status READY is reserved for actual verified native6.

    Pair report paths are explicit owner inputs. There is no directory scan or
    guessed filename. Every output snapshot is new; old attempts are preserved.
    """
    output_dir, handoff = check_outputs(output_dir, handoff, fixture=fixture_only)
    input_paths = [
        native_completion,
        native_policy,
        producer_binding,
        source_manifest,
        runtime_manifest,
        *(pair_checks or []),
    ]
    if any(handoff == Path(p).absolute() for p in input_paths if p is not None):
        raise PublicationPathError("Handoff must never overwrite a publication input")
    base = dict(
        schema="drone-ai.pilot45.data-handoff/1",
        task_id=TASK,
        ai_software_ready=True,
        actual_input_admitted=False,
        mutable_dependency_pointer=True,
        model_forward_steps=0,
        optimizer_steps=0,
        GPU_jobs=0,
        fixture_only=fixture_only,
    )
    try:
        authorization = json_file(RECOVERY_AUTH)
        if (
            authorization.get("task_id") != TASK
            or authorization.get("authorized", {}).get("pilot_runs") != 45
        ):
            raise Pending("Finite recovery authorization unavailable")
        native_completion, native_policy = regular(native_completion), regular(native_policy)
        complete = json_file(native_completion)
        if str(complete.get("status", "")).upper().startswith(("FAILED", "STOPPED")):
            raise ValueError(
                "Native development stopped/failed; preserve evidence and stop the finite flow"
            )
        check_native_completion(complete, fixture=fixture_only)
        native = json_file(native_policy)
        check_policy(native)
        source_manifest, runtime_manifest = regular(source_manifest), regular(runtime_manifest)
        if not fixture_only:
            if (
                source_manifest != ROOT / "simulation/config/producer-sources.json"
                or runtime_manifest != ROOT / "simulation/artifacts/runtime-manifest.json"
            ):
                raise ValueError("Publish against current producer files, not a historical copy")
            installed = ROOT / "simulation/profiles/collection/joint-phase-policy.json"
            if sha(regular(installed)) != sha(native_policy):
                raise ValueError("Native policy is not the installed current producer policy")
        producer = json_file(producer_binding)
        check_producer(producer, producer)
        source_hash, runtime_hash, policy_hash = (
            sha(source_manifest),
            sha(runtime_manifest),
            sha(native_policy),
        )
        if (
            source_hash != producer["source_manifest_sha256"]
            or runtime_hash != producer["runtime_sha256"]
            or json_file(runtime_manifest).get("source_manifest_sha256") != source_hash
            or complete.get("source_sha256") != source_hash
            or complete.get("runtime_sha256") != runtime_hash
            or complete.get("policy_sha256") != policy_hash
        ):
            raise ValueError(
                "Final native completion/current producer/source/runtime/policy mismatch"
            )
        source = json_file(source_manifest)
        for member in (
            "sim/src/capstone_sim/joint_phase.py",
            "sim/src/capstone_sim/cli.py",
            "profiles/collection/joint-phase-policy.json",
        ):
            if member not in source.get("files", {}):
                raise ValueError("Native interface source is absent from final producer manifest")
            if (
                not fixture_only
                and sha(regular(ROOT / "simulation" / member)) != source["files"][member]
            ):
                raise ValueError("Current native interface source differs from seal: " + member)
        if not fixture_only:
            interface = (ROOT / "simulation/sim/src/capstone_sim/cli.py").read_text()
            if any(
                token not in interface
                for token in (
                    "--phase-policy",
                    "--paired-case-id",
                    "pair-check",
                    "--normal-run",
                    "--anomaly-run",
                )
            ):
                raise ValueError("Reviewed official phase/pair CLI is absent from installed source")
        if pair_checks is None:
            raise Pending("Owner-provided three actual native pair report paths are missing")
        if len(pair_checks) != 3:
            raise ValueError("Exactly three independent native pair reports required")
        policy = dict(
            schema="drone-ai.paired-source-clock-policy/1",
            status="FROZEN",
            approved_for_pilot=True,
            producer=producer,
            policy_id=ADAPTER,
            alignment_mode="versioned_joint_phase",
            native_adapter=ADAPTER,
            mission_policy=MISSION_POLICY,
            posthoc_adjustment=False,
            native_policy_sha256=policy_hash,
            native_receipt_schema="drone.joint-phase-native-receipt/1",
            wind_model_clock=WIND_CLOCK,
            mission_source=ORIGIN,
            fixture_only=fixture_only,
            limits=dict(
                origin_delta_us=0,
                wind_rms_mps=native["acceptance"]["wind_rms_mps"],
                wind_peak_mps=native["acceptance"]["wind_max_mps"],
                receipt_min_offset_us=0,
                receipt_max_offset_us=native["maximum_origin_notice_delay_us"],
            ),
            origin_delta_meaning="Actual weather origin minus actual path-consumption origin within each run; not equality of absolute starts across runs",
        )
        members, comparisons, ids = [], [], set()
        for report_path in pair_checks:
            projected = pair(report_path, policy, fixture=fixture_only)
            report = json_file(report_path)
            for role in ("normal", "anomaly"):
                raw = Path(report[role + "_run"])
                identity = origin(raw / "run.json", policy, fixture=fixture_only)["run_id"]
                if identity in ids:
                    raise ValueError(
                        "Development pair control/raw reused; six distinct runs required"
                    )
                ids.add(identity)
                members.append(
                    dict(
                        run=reference(raw / "run.json"),
                        execution=reference(raw / "evaluation/execution.json"),
                        origin=reference(raw / ORIGIN),
                    )
                )
            comparisons.append(dict(**projected, native_evidence=reference(report_path)))
        if ids != set(complete["actual_run_ids"]):
            raise ValueError("Native completion membership differs from the three actual pairs")
        if sha(source_manifest) != source_hash or sha(runtime_manifest) != runtime_hash:
            raise ValueError("Producer changed during one-shot publication")
        output_dir.mkdir(parents=True, exist_ok=False)
        for name, path in (
            ("native-completion.json", native_completion),
            ("native-policy.json", native_policy),
        ):
            with (output_dir / name).open("xb") as stream:
                stream.write(path.read_bytes())
        for name, value in (("expected-producer.json", producer), ("paired-policy.json", policy)):
            write_new(output_dir / name, value)
        development = dict(
            schema="drone-ai.paired-development-evidence/1",
            status="PASSED",
            producer=producer,
            paired_policy_sha256=sha(output_dir / "paired-policy.json"),
            members=members,
            comparisons=comparisons,
        )
        write_new(output_dir / "development-receipt.json", development)
        refs = dict(
            expected_producer=reference(output_dir / "expected-producer.json"),
            paired_policy=reference(output_dir / "paired-policy.json"),
            native_policy=reference(output_dir / "native-policy.json"),
            native_completion=reference(output_dir / "native-completion.json"),
            development_receipt=reference(output_dir / "development-receipt.json"),
        )
        request = dict(
            schema="drone-ai.pilot45.check/1",
            task_id=TASK,
            purpose=PILOT,
            phase="prerequisites",
            fixture_only=fixture_only,
            refs=refs,
        )
        write_new(output_dir / "native-binding-request.json", request)
        ctx = Evidence(output_dir / "native-binding-request.json")
        checked_policy = policy_check(ctx, producer)
        development_check(ctx, producer, checked_policy)
        result = dict(
            base,
            status="SYNTHETIC_READY" if fixture_only else "READY",
            producer=producer,
            native_development_verified=not fixture_only,
            official_cli_help_checked=False,
            requires_Data_cli_help_check=True,
            requires_Data_prerequisites_and_case_gates=True,
            paths={
                k: str(output_dir / name)
                for k, name in (
                    ("native_completion", "native-completion.json"),
                    ("native_policy", "native-policy.json"),
                    ("paired_policy", "paired-policy.json"),
                    ("development_receipt", "development-receipt.json"),
                )
            },
            origin_relative_path=ORIGIN,
            paired_cli_options=["--phase-policy", ADAPTER, "--paired-case-id", "{paired_case_id}"],
            pair_verifier_argv=[
                str(ROOT / "simulation/capstone-sim"),
                "pair-check",
                "--normal-run",
                "{normal_run}",
                "--anomaly-run",
                "{anomaly_run}",
                "--output",
                "{output}",
            ],
            publication_receipt=str(output_dir / "publication-receipt.json"),
            evidence_sha256=ctx.hashes,
            checked_stages=ctx.checked,
        )
        result["sha256"] = {k: sha(Path(v)) for k, v in result["paths"].items()}
        write_new(output_dir / "publication-receipt.json", result)
        pointer(handoff, result)
        return 0, result
    except (Pending, FileNotFoundError) as error:
        result, code = dict(base, status="WAITING", reason=str(error)), 2
    except (ValueError, KeyError, TypeError, OSError) as error:
        result, code = dict(base, status="FAILED", reason=str(error)), 1
    pointer(handoff, result)
    return code, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, help="Owner-provided actual native publication paths")
    for name in (
        "native-completion",
        "native-policy",
        "producer-binding",
        "source-manifest",
        "runtime-manifest",
        "output-dir",
        "handoff",
    ):
        parser.add_argument("--" + name, required=name in {"output-dir", "handoff"}, type=Path)
    parser.add_argument("--pair-check", action="append", type=Path)
    args = parser.parse_args(argv)
    try:
        check_outputs(args.output_dir, args.handoff)
    except ValueError as error:
        print(json.dumps(dict(status="FAILED", reason=str(error), exit_code=1)))
        return 1
    try:
        values = {
            name: getattr(args, name)
            for name in (
                "native_completion",
                "native_policy",
                "producer_binding",
                "source_manifest",
                "runtime_manifest",
            )
        }
        values["pair_checks"] = args.pair_check
        if args.spec:
            spec = json_file(args.spec)
            if (
                spec.get("schema") != "drone-ai.pilot45.publish-request/1"
                or spec.get("task_id") != TASK
            ):
                raise ValueError("Explicit owner publication request required")
            paths = spec.get("paths", {})
            for name in values:
                supplied = paths.get(name)
                if supplied is None:
                    raise Pending("Native publication input is not delivered: " + name)

                def path(value):
                    p = Path(value)
                    return p if p.is_absolute() else args.spec.absolute().parent / p

                values[name] = (
                    [path(v) for v in supplied] if name == "pair_checks" else path(supplied)
                )
        if any(value is None for value in values.values()):
            raise Pending("Supply --spec or every explicit native publication input")
        code, result = publish(**values, output_dir=args.output_dir, handoff=args.handoff)
    except (Pending, FileNotFoundError) as error:
        code, result = (
            2,
            dict(
                schema="drone-ai.pilot45.data-handoff/1",
                task_id=TASK,
                status="WAITING",
                reason=str(error),
            ),
        )
        pointer(args.handoff, result)
    except PublicationPathError as error:
        code, result = (
            1,
            dict(
                schema="drone-ai.pilot45.data-handoff/1",
                task_id=TASK,
                status="FAILED",
                reason=str(error),
            ),
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        code, result = (
            1,
            dict(
                schema="drone-ai.pilot45.data-handoff/1",
                task_id=TASK,
                status="FAILED",
                reason=str(error),
            ),
        )
        pointer(args.handoff, result)
    print(json.dumps(dict(result, exit_code=code), allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
