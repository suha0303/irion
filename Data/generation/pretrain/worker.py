"""One bounded CPU unit; progress files are consumed by the local CLI only."""

import argparse
import os
import resource
import time
from pathlib import Path

from .common import (
    ADMISSION,
    PACKAGE,
    RESOURCES,
    Budget,
    load,
    now,
    package_imports,
    read,
    sha,
    verify,
    write,
)
from .retry import shared_failure


def execute(request):
    package_imports()
    from preprocessing_tools.control import verify_inventory

    row = request["row"]
    action = request["action"]
    out = Path(request["out"])
    status = Path(request["status"])
    state = dict(
        status="RUNNING",
        action=action,
        attempt=request.get("attempt", 1),
        attempts_max=request.get("attempts_max", 1),
        run_id=row["run_id"],
        pid=os.getpid(),
        started=time.time(),
        updated_utc=now(),
        bytes_done=0,
        bytes_total=0,
    )
    write(status, state)

    def update(**fields):
        state.update(fields, updated_utc=now())
        write(status, state)

    try:
        if action == "extract":
            from .archive import extract_consumer

            expected = sum(row["archive"]["members"][name]["bytes"] for name in row["files"])

            class ExtractionBudget(Budget):
                last = 0

                def check(self):
                    super().check()
                    if time.monotonic() - self.last > 2:
                        files = [Path(row["path"]) / name for name in row["files"]]
                        update(
                            bytes_done=sum(p.stat().st_size for p in files if p.exists()),
                            bytes_total=expected,
                            phase="extract_and_verify",
                        )
                        self.last = time.monotonic()

            result = extract_consumer(row, ExtractionBudget())
            update(bytes_done=result["consumer_bytes"], bytes_total=result["consumer_bytes"])
        elif action == "preprocess":
            from capstone_ai.cli import output
            from capstone_ai.source import DirectoryTail

            from .preprocessing_compat import make_preprocessor

            verify_inventory(row["path"], row["files"])
            config = read(PACKAGE / "preprocessing.json")
            run = Path(row["path"])
            engine, metadata_alias = make_preprocessor(
                read(run / "observations/metadata.json"),
                config,
                row["files"]["observations/metadata.json"],
            )
            tail = DirectoryTail(run, engine, 1048576)
            total = (run / "observations/streams.jsonl").stat().st_size
            evidence = dict(
                initial_status=read(run / "run.json")["status"],
                started_utc=now(),
                source_grew_while_attached=False,
                live_verified=False,
            )
            if metadata_alias is not None:
                evidence["metadata_validation_alias"] = metadata_alias
            previous = 0
            budget = Budget()
            while not tail.closed:
                budget.check()
                if not tail.poll():
                    break
                if time.monotonic() - previous > 2:
                    update(
                        bytes_done=tail.stream.offset,
                        bytes_total=total,
                        frames=len(engine.frames),
                        windows=len(engine.windows),
                        phase="reading",
                    )
                    previous = time.monotonic()
            if not tail.closed:
                raise ValueError("Unclosed stream; whole task stops")
            update(phase="writing_canonical", bytes_done=tail.stream.offset, bytes_total=total)
            result = output(engine, tail, out, "replay", evidence)
            verify_inventory(run, row["files"])
            if result["input_hashes"] != row["files"]:
                raise ValueError("Preprocess input inventory mismatch")
            reference = request.get("reference")
            if reference:
                expected = read(Path(reference) / row["run_id"] / "summary.json")
                if result["output_hashes"] != expected["output_hashes"]:
                    raise ValueError("Parallel canonical result differs from prior serial result")
            result = dict(
                frames=result["frames"],
                windows=result["windows"],
                equivalence_passed=bool(reference),
                output_hashes=result["output_hashes"],
            )
        elif action == "audit":
            import numpy as np

            derived = load("parallel_derived", ADMISSION / "derived.py")
            derived.fence(read(RESOURCES / "protected-inventory.json")["deny_paths"])
            canonical = Path(request["canonical"]) / row["run_id"]
            summary = read(canonical / "summary.json")
            if (
                summary["config"] != read(PACKAGE / "preprocessing.json")
                or summary["input_hashes"] != row["files"]
            ):
                raise ValueError("Canonical input/config mismatch")
            if (
                summary["contract_sha256"]
                != read(PACKAGE / "PACKAGE.json")["files"][
                    "CODEX_REBUILD_PACKAGE_V4_2/00_SHARED_DATA_CONTRACT.md"
                ]
            ):
                raise ValueError("Canonical contract mismatch")
            verify(
                {
                    str(canonical / (name + ".jsonl")): digest
                    for name, digest in summary["output_hashes"].items()
                }
            )
            report, samples, selected = derived.inspect_run(canonical, row)
            out.mkdir(exist_ok=False)
            arrays = (
                dict(zip(("X", "C", "Y", "run_id", "origin_us", "segment"), zip(*samples)))
                if samples
                else {}
            )
            np.savez_compressed(
                out / "numeric.npz",
                **{k: np.asarray(v) for k, v in arrays.items()},
                train_frames=np.asarray(selected),
            )
            write(out / "report.json", report)
            result = dict(
                report=report,
                numeric_sha256=sha(out / "numeric.npz"),
                report_sha256=sha(out / "report.json"),
            )
        elif action == "verify":
            verify_inventory(row["path"], row["files"])
            result = dict(
                verified=True,
                bytes=sum((Path(row["path"]) / n).stat().st_size for n in row["files"]),
            )
        else:
            raise ValueError("Unknown CPU-only action")
        update(
            status="COMPLETE",
            result=result,
            finished=time.time(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
    except BaseException as exc:
        fatal = shared_failure(exc)
        update(
            status="FAILED_STOPPED",
            reason=f"{type(exc).__name__}: {exc}",
            finished=time.time(),
            failure_scope="shared" if fatal else "unit",
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    execute(read(parser.parse_args().request))
