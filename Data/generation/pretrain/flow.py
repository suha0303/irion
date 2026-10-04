"""Length-homogeneous flights overlap one CPU group; bounded raw backlog, no LLM."""

import copy
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from .common import write


def homogeneous_batches(cases, width):
    buckets = {}
    for case in cases:
        buckets.setdefault(case.get("mission_seconds", 900), []).append(case)
    return [
        rows[index : index + width]
        for duration, rows in sorted(buckets.items(), reverse=True)
        for index in range(0, len(rows), width)
    ]


def processing_runner(runner):
    cpu = copy.copy(runner)
    cpu.state = copy.deepcopy(runner.state)
    cpu.state.update(
        completed=[], stages=[], active=[], stage="waiting-for-first-flight", collection_ledger=None
    )
    cpu.state_path = runner.task / "operator/processing-state.json"
    cpu.peaks = runner.peaks.copy()
    cpu.update()
    return cpu


def collect_overlapping(runner, pipeline, planned, groups):
    from .collection import IncompleteCollection
    from .streaming import fragment_freeze

    cpu = processing_runner(runner)
    pipeline.r = cpu
    stop = runner.task / "operator/processing-stop.json"
    if stop.exists():
        raise ValueError("Existing processing failure signal; never restart a claimed task")
    runner.update(pipeline_overlap=True, bounded_raw_runs=2 * runner.config["flight_workers"])
    pending, outcomes = [], {}

    def process(label, folder, approved, count):
        if stop.exists():
            raise RuntimeError("Previous processing failure; pending group preserved")
        try:
            cpu.begin(label + "-freeze", total=count, workers=1)
            output, rows = fragment_freeze(cpu, folder, approved)
            cpu.finish(
                label + "-freeze", dict(verified_runs=len(rows), whole_dataset_admitted=False)
            )
            if rows:
                rows = pipeline.process_chunk(rows, label, folder)
            return output, rows
        except BaseException as exc:
            write(
                stop,
                dict(
                    status="FAILED_STOPPED",
                    reason=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                    policy="close flight admission; drain owned peers; preserve all pending groups",
                ),
            )
            cpu.update(
                status="FAILED_STOPPED", reason=f"{type(exc).__name__}: {exc}", finished=time.time()
            )
            raise

    def receive():
        label, future = pending.pop(0)
        outcomes[label] = future.result()  # Program wait, not a Codex/LLM loop.

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        for set_index, batches in enumerate(groups):
            for index, cases in enumerate(batches):
                # At most one older CPU group plus the next flight group retain raw outputs.
                if len(pending) >= 2:
                    runner.update(stage="waiting-for-processing-capacity", collection_ledger=None)
                    receive()
                if stop.exists():
                    while pending:
                        receive()  # Re-raise the original background failure before any new flight.
                    raise RuntimeError("Processing failure signal; no new flight admitted")
                label = f"stream-{set_index + 1:03}-{index + 1:03}"
                try:
                    runner.flight("collection", runner.config["flight_workers"], cases, label)
                except IncompleteCollection:
                    pass
                folder = runner.task / label
                pending.append(
                    (label, pool.submit(process, label, folder, planned[set_index], len(cases)))
                )
        runner.update(stage="waiting-for-processing", collection_ledger=None)
        while pending:
            receive()
        cpu.update(status="PROCESSING_COMPLETE", finished=time.time())
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        # Final assembly uses the main coordinator again, after every CPU group is finished.
        pipeline.r = runner
        runner.update(
            stream_completed=cpu.state.get("stream_completed", 0),
            stream_retired_paths=cpu.state.get("stream_retired_paths", 0),
        )
    return outcomes
