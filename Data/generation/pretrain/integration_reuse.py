"""Read-only acceptance of a completed integration; never launch or rewrite
it.
"""

import argparse
import json
from pathlib import Path

from pretrain import pilot45

from . import ai19
from .common import ROOT, identity, now, read, sha, verify, write
from .pilot19 import audit_one

CONTROL_FILES = {
    str(ROOT / "Data/generation/pretrain" / name)
    for name in ("pilot45.py", "pilot_flow.py")
}


def compatible_sources(task):
    """Only the reviewed controller/termination fix may differ from the old
    task.
    """
    sources = read(task / "config.json")["source_sha256"]
    changed = {}
    for name, old in sources.items():
        current = sha(ai19.no_links(name))
        if current != old:
            if name not in CONTROL_FILES:
                raise ValueError(
                    "Integration producer/preprocessor dependency changed: "
                    + name
                )
            changed[name] = dict(old_sha256=old, current_sha256=current)
    return dict(verified_entries=len(sources), reviewed_control_changes=changed)


def historical_termination(task):
    """An unloaded unit's default Result is deliberately not an input here."""
    cleanup = read(task / "operator/socket-cleanup.json")
    state = read(task / "operator/state.json")
    unit = read(task / "operator/service.json")["unit"]
    if {
        "status",
        "service_result",
        "errors",
        "remaining_unremoved_sockets",
    } - set(cleanup):
        raise ValueError("Historical cleanup fields missing")
    if (
        cleanup.get("status") != "CLEANUP_COMPLETE"
        or cleanup.get("service_result") != "success"
        or cleanup.get("errors")
        or cleanup.get("remaining_unremoved_sockets")
    ):
        raise ValueError(
            "Historical actual SERVICE_RESULT/cleanup evidence does not pass"
        )
    if (
        state.get("status") != "INTEGRATION19_QUALITY_PASSED_STOPPED"
        or state.get("service_unit") != unit
    ):
        raise ValueError(
            "Historical integration completion/owned unit does not match"
        )
    if identity(state["pid"]) == state["pid_start_ticks"]:
        raise ValueError("Integration worker is still alive")
    cgroup = state["cgroup"].strip().removeprefix("0::/")
    if not cgroup.endswith("/" + unit):
        raise ValueError("Historical owned cgroup identity mismatch")
    group = Path("/sys/fs/cgroup") / cgroup
    if (
        group.exists()
        and (group / "cgroup.procs").read_text(encoding="utf-8").strip()
    ):
        raise ValueError("Historical integration cgroup still has processes")
    processes = []
    for stage in ("flight", "prepare", "audit"):
        paths = sorted(
            (task / "pilot/execution").glob("*/" + stage + "-*.process.json")
        )
        if len(paths) != 1:
            raise ValueError(
                "Exactly one successful integration stage receipt required: "
                + stage
            )
        p = read(paths[0])
        if (
            p.get("exit_code") != 0
            or p.get("owner_uid") != state["owner_uid"]
            or p.get("cgroup") != state["cgroup"]
        ):
            raise ValueError(
                "Historical child exit/owner/cgroup evidence failed: " + stage
            )
        if identity(p["pid"]) == p["pid_start_ticks"]:
            raise ValueError("Historical stage still running")
        processes.append(str(paths[0]))
    return dict(
        basis=(
            "actual ExecStopPost SERVICE_RESULT plus closed child "
            "exits/owner/quality/cgroup"
        ),
        unit=unit,
        service_result=cleanup["service_result"],
        process_receipts=processes,
        unloaded_Result_not_used=True,
    )


def accept(task, output):
    pilot45.recovery_authorization()
    task = pilot45.task_path(task)
    verify(
        {
            str(task / "config.json"): (task / "config.sha256")
            .read_text(encoding="utf-8")
            .strip(),
            str(task / "manifest.json"): (task / "manifest.sha256")
            .read_text(encoding="utf-8")
            .strip(),
        }
    )
    source = compatible_sources(task)
    termination = historical_termination(task)
    quality = read(task / "quality.json")
    if (
        quality.get("completed_raw") != 1
        or quality.get("prepared_and_audited") != 1
        or quality.get("online_available") != 1
        or quality.get("run_available") != 1
    ):
        raise ValueError(
            "A fully available and audited single integration is required"
        )
    ledger = read(task / "ledger.json")
    if len(ledger["cases"]) != 1:
        raise ValueError("Exactly one completed integration case required")
    entry = next(iter(ledger["cases"].values()))
    if entry.get("status") != "COMPLETE":
        raise ValueError("Integration case is not complete")
    verify(entry["files"])
    raw = ai19.no_links(entry["run"])
    if (
        raw.parent != ROOT / "simulation/runs"
        or read(raw / "run.json").get("status") != "completed"
    ):
        raise ValueError("Completed actual integration raw required")
    for pid in read(raw / "run.json").get("owned_process_groups", {}).values():
        if identity(pid) is not None:
            raise ValueError(
                "Recorded native process ID is still live; review ownership"
            )
    expected = read(task / "expected-producer.json")
    ai19.ai_modules()
    from learning.admission19 import check_producer

    check_producer(ai19.producer_binding(), expected)
    export = Path(entry["output"])
    binding = read(export / "binding.json")
    verify(binding["consumer_sources"])
    if binding["raw_integrity_sha256"] != sha(raw / "integrity.json"):
        raise ValueError("Integration raw closing integrity changed")
    raw_files = {}
    for name, item in read(raw / "integrity.json").items():
        p = ai19.no_links(raw / name)
        if p.stat().st_size != item["size"] or sha(p) != item["sha256"]:
            raise ValueError("Integration raw member changed: " + name)
        raw_files[str(p)] = item["sha256"]
    audit = audit_one(export, task / "expected-producer.json")
    evidence = {str(p): sha(p) for p in task.rglob("*") if p.is_file()}
    evidence.update(entry["files"])
    evidence.update(raw_files)
    result = dict(
        schema="drone-data.reused-integration/1",
        status="ACCEPTED_EXISTING_INTEGRATION_NO_FLIGHT",
        task=str(task),
        raw=str(raw),
        producer=expected,
        source_compatibility=source,
        termination=termination,
        quality=quality,
        audit=audit,
        evidence_sha256=evidence,
        checked_utc=now(),
        new_flights=0,
    )
    write(output, result)
    return result


def validate(task, receipt):
    value = read(receipt)
    if (
        value.get("schema") != "drone-data.reused-integration/1"
        or value.get("status") != "ACCEPTED_EXISTING_INTEGRATION_NO_FLIGHT"
        or value.get("task") != str(task)
    ):
        raise ValueError("Actual integration reuse receipt required")
    verify(value["evidence_sha256"])
    if compatible_sources(task) != value["source_compatibility"]:
        raise ValueError(
            "Integration source compatibility changed since acceptance"
        )
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cli_result = accept(args.task, args.output)
    print(
        json.dumps(
            dict(
                status=cli_result["status"],
                raw=cli_result["raw"],
                new_flights=0,
            )
        )
    )
