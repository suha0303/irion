"""Finite local task runner. All waiting and progress accounting happen here, without an LLM."""

import copy
import fcntl
import json
import os
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

from .common import (
    ADMISSION,
    CPU_WORKER_BYTES,
    HOME,
    PACKAGE,
    PYTHON,
    RESOURCES,
    ROOT,
    STORE,
    TASK_WALL_SECONDS,
    Budget,
    admission,
    collector,
    external,
    identity,
    load,
    now,
    overlap,
    package_imports,
    pythonpath,
    read,
    runs_root,
    sha,
    verify,
    verify_runtime,
    write,
)
from .composition import seed_stride
from .retry import attempts_limit


class ResourceBlocked(RuntimeError):
    pass


class Runner:
    def __init__(self, task):
        self.task = Path(task)
        self.config = read(self.task / "config.json")
        self.state_path = self.task / "operator/state.json"
        self.state = dict(
            status="RUNNING",
            task=self.task.name,
            mode=self.config["mode"],
            pid=os.getpid(),
            pid_start_ticks=identity(os.getpid()),
            started=time.time(),
            updated_utc=now(),
            completed=[],
            stage="admission",
            stages=[],
            training_authorized=False,
            unit_attempts_max=attempts_limit(self.config),
            whole_task_retry=False,
        )
        self.c = collector()
        self.peaks = CPU_WORKER_BYTES.copy()

    def update(self, **fields):
        self.state.update(fields, updated_utc=now())
        write(self.state_path, self.state)

    def begin(self, name, **fields):
        verify(self.config["source_sha256"])
        self.update(
            **dict(
                dict(
                    stage=name,
                    stage_started=time.time(),
                    done=0,
                    total=0,
                    active=[],
                    reason=None,
                    admission=None,
                ),
                **fields,
            )
        )

    def finish(self, name, result):
        result = copy.deepcopy(result)
        result.setdefault("elapsed_s", time.time() - self.state["stage_started"])
        self.state["completed"].append(name)
        self.state["stages"].append(dict(name=name, **result))
        self.update(active=[], done=result.get("completed", self.state.get("done", 0)))

    def resources(self):
        external(STORE)
        snapshot = self.c.memory_snapshot()
        snapshot["external_free_bytes"] = shutil.disk_usage(STORE).free
        self.state["resources"] = snapshot
        peaks = self.state.setdefault("resource_peaks", {})
        peaks["max_owned_rss_bytes"] = max(
            peaks.get("max_owned_rss_bytes", 0), snapshot["owned_tree_rss_bytes"]
        )
        peaks["min_available_bytes"] = min(
            peaks.get("min_available_bytes", snapshot["available_bytes"]),
            snapshot["available_bytes"],
        )

        def blocked(reason):
            self.state.setdefault(
                "first_resource_violation",
                dict(
                    stage=self.state["stage"],
                    observed_utc=now(),
                    reason=reason,
                    snapshot=snapshot.copy(),
                ),
            )
            raise ResourceBlocked(reason)

        initial = self.state.setdefault(
            "initial_external_free_bytes", snapshot["external_free_bytes"]
        )
        storage_cap = self.config.get("external_additional_bytes_max", 80_000_000_000)
        if initial - snapshot["external_free_bytes"] > storage_cap:
            blocked(f"Task external additional-space cap{storage_cap / 1e9:g}GB reached")
        # Working-set RSS drives CPU admission; cgroup cache remains visible separately.
        floor_gb = self.config.get("min_available_ram_gb", 4)
        expanded = self.config.get("use_available_ram", False)
        if snapshot["available_bytes"] < floor_gb * 1_000_000_000 or (
            not expanded and snapshot["owned_tree_rss_bytes"] > 8_000_000_000
        ):
            blocked(
                f"RAM limit: available={snapshot['available_bytes'] / 1e9:.3f}GB (min{floor_gb}), own RSS={snapshot['owned_tree_rss_bytes'] / 1e9:.3f}GB ({'fixed cap disabled' if expanded else 'max8'}); no new units admitted"
            )
        if snapshot["external_free_bytes"] < 50_000_000_000:
            blocked("External free-space floor50GB")
        if time.time() - self.state["started"] > self.config.get("wall_seconds", TASK_WALL_SECONDS):
            blocked("Finite task wall cap reached")
        return snapshot

    def cpu_admission(self, workers, action, available):
        estimate = admission(workers, self.peaks[action], available)
        estimate["estimate_only_override"] = bool(
            (self.config["mode"] == "test" and self.config.get("ignore_memory_estimate"))
            or self.config.get("use_available_ram")
        )
        estimate["allowed"] = estimate["passed"] or estimate["estimate_only_override"]
        self.update(admission=estimate)
        return estimate

    def batch(self, name, action, rows, workers, out, *, reference=None, canonical=None):
        self.begin(name, total=len(rows), workers=workers)
        snapshot = self.resources()
        estimate = self.cpu_admission(workers, action, snapshot["available_bytes"])
        if not estimate["allowed"]:
            raise ResourceBlocked(
                f"{name}: requested{workers} workers exceeds memory admission; no silent downgrade"
            )
        folder = self.task / "units" / name
        folder.mkdir(parents=True)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        active, finished, pending = {}, [], deque(rows)
        limit = attempts_limit(self.config)
        attempts, history = {}, {}
        failure = None
        unit_failures = []
        resource_failure = False
        env = dict(
            os.environ,
            PYTHONPATH=pythonpath(),
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONFAULTHANDLER="1",
        )
        exhausted = False
        stopping = {}
        while active or not exhausted:
            try:
                self.resources()
            except Exception as exc:
                if failure is None:
                    failure = str(exc)
                    resource_failure = isinstance(exc, ResourceBlocked)
                if (
                    isinstance(exc, ResourceBlocked)
                    and self.config.get("use_available_ram")
                    and self.state.get("resources", {}).get("available_bytes", 10**12)
                    < 1_000_000_000
                ):
                    resource_failure = True
                    self.update(
                        reason=failure,
                        status="DRAINING",
                        memory_stop_policy="terminate owned CPU workers below1GB; kill after5s; no retry",
                    )
                    for key, (process, _, _) in active.items():
                        if key not in stopping and process.poll() is None:
                            stopping[key] = time.monotonic()
                            process.terminate()
            for key, stopped_at in stopping.items():
                if key in active and time.monotonic() - stopped_at >= 5:
                    process = active[key][0]
                    if process.poll() is None:
                        process.kill()
            while len(active) < workers and not exhausted and not failure:
                row = pending.popleft() if pending else None
                if row is None:
                    exhausted = True
                    break
                key = row["run_id"]
                attempts[key] = attempts.get(key, 0) + 1
                if action == "extract" and Path(row["path"]).exists():
                    failure = f"Refuse to overwrite existing extraction destination: {row['path']}"
                    break
                if attempts[key] > 1:
                    verify(self.config["source_sha256"])
                request = dict(
                    attempt=attempts[key],
                    attempts_max=limit,
                    action=action,
                    row=row,
                    out=str(out / key),
                    status=str(folder / (key + ".status.json")),
                    reference=str(reference) if reference else None,
                    canonical=str(canonical) if canonical else None,
                )
                path = folder / (key + ".request.json")
                write(path, request)
                log = (folder / (key + ".log")).open("x")
                try:
                    process = subprocess.Popen(
                        [str(PYTHON), "-B", "-m", "pretrain.worker", str(path)],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except OSError as exc:
                    log.close()
                    failure = f"Launch failed for {key}: {exc}; drain active workers"
                    write(request["status"], dict(status="FAILED_STOPPED", reason=failure))
                    break
                active[key] = (process, log, request)
            if failure:
                exhausted = (
                    True  # Drain already launched units; never retry or submit queued units.
                )
            for key, (process, log, request) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                status_file = Path(request["status"])
                try:
                    result = read(status_file) if status_file.exists() else {}
                    if not isinstance(result, dict):
                        result = dict(reason="Invalid worker status shape")
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    result = dict(reason=f"Invalid worker status: {exc}")
                if code != 0 or result.get("status") != "COMPLETE":
                    error = f"{key}: exit{code}: {result.get('reason', 'missing completion proof')}"
                    evidence = folder / "attempts" / key / f"attempt-{attempts[key]:02d}"
                    evidence.mkdir(parents=True, exist_ok=False)
                    for suffix in ("request.json", "status.json", "log"):
                        source = folder / (key + "." + suffix)
                        if source.exists():
                            shutil.copy2(source, evidence / source.name)
                    entry = dict(
                        attempt=attempts[key], error=error, exit_code=code, evidence=str(evidence)
                    )
                    history.setdefault(key, []).append(entry)
                    write(evidence / "failure.json", entry)
                    shared = result.get("failure_scope") == "shared" or code in {
                        -9,
                        -15,
                        -2,
                        137,
                        143,
                        130,
                    }
                    if shared:
                        failure = failure or error
                    if not failure and attempts[key] < limit:
                        # Preserve only this failed worker's owned partial destinations.
                        paths = [Path(request["out"])]
                        if action == "extract":
                            paths.append(Path(request["row"]["path"]))
                        try:
                            for source in paths:
                                if source.exists():
                                    target = (
                                        source.parent
                                        / ".failed"
                                        / key
                                        / f"attempt-{attempts[key]:02d}"
                                    )
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    if target.exists():
                                        raise FileExistsError(target)
                                    source.rename(target)
                            for suffix in ("request.json", "status.json", "log"):
                                (folder / (key + "." + suffix)).unlink(missing_ok=True)
                            pending.appendleft(request["row"])
                            exhausted = False
                        except OSError as exc:
                            failure = f"Cannot preserve failed output safely: {exc}"
                    else:
                        unit_failures.append(
                            dict(
                                run_id=key,
                                error=error,
                                exit_code=code,
                                attempts=attempts[key],
                                history=history[key],
                            )
                        )
                        if not self.config.get("continue_after_case_failure"):
                            failure = failure or error
                else:
                    result.update(attempts=attempts[key], previous_failures=history.get(key, []))
                    write(status_file, result)
                    finished.append(result)
                    self.peaks[action] = max(self.peaks[action], result["peak_rss_bytes"])
                del active[key]
            self.update(
                status="DRAINING" if failure else "RUNNING",
                done=len(finished) + len(unit_failures),
                succeeded=len(finished),
                failed=len(unit_failures),
                active=[
                    dict(
                        run_id=k,
                        pid=p.pid,
                        status_file=r["status"],
                        attempt=attempts[k],
                        attempts_max=limit,
                    )
                    for k, (p, _, r) in active.items()
                ],
                reason=failure or (unit_failures[-1]["error"] if unit_failures else None),
            )
            if active:
                time.sleep(1)  # Program polling only. The assistant does not wait on this loop.
        failure = failure or (unit_failures[0]["error"] if unit_failures else None)
        result = dict(
            action=action,
            workers=workers,
            total=len(rows),
            completed=len(finished),
            failed=len(unit_failures),
            failure_records=unit_failures,
            elapsed_s=time.time() - self.state["stage_started"],
            max_overlap=overlap(finished),
            peak_worker_rss_bytes=max((r["peak_rss_bytes"] for r in finished), default=0),
            failure=failure,
            failure_kind="resource" if resource_failure else "worker" if failure else None,
            resource_peaks=self.state.get("resource_peaks"),
            first_resource_violation=self.state.get("first_resource_violation"),
            unit_records=finished,
            unit_attempts_max=limit,
            attempt_counts=attempts,
            attempt_failures=history,
        )
        write(folder / "result.json", result)
        if failure:
            if resource_failure:
                raise ResourceBlocked(failure)
            raise RuntimeError(failure)
        self.finish(name, {k: v for k, v in result.items() if k != "unit_records"})
        return result

    def rows(self, frozen, label):
        result = copy.deepcopy(frozen)
        for row in result["runs"]:
            row["path"] = str(STORE / "consumer/pretrain" / self.task.name / label / row["run_id"])
        return result

    def prepare(self, frozen, label, workers, reference=None, full=True, cases_path=None):
        package_imports()
        from .composition import validate_roles

        if full:
            cases = (
                read(cases_path or self.task / "cases.json")["cases"]
                if (cases_path or self.task / "cases.json").exists()
                else read(RESOURCES / "case_manifest.json")["cases"]
            )
            validate_roles(
                frozen["runs"],
                cases,
                read(RESOURCES / "protected-inventory.json")["runs"],
                frozen.get("collection_profile", "legacy50"),
                case_sets=frozen.get("collection_set_case_ids"),
            )
        # Reject an impossible CPU request before repeating archive extraction.
        for action, count in (
            ("extract", self.config["io_workers"] if full else workers),
            ("preprocess", workers),
            ("audit", self.config["audit_workers"] if full else workers),
        ):
            self.begin(label + "-" + action + "-admission", workers=count)
            estimate = self.cpu_admission(count, action, self.resources()["available_bytes"])
            if not estimate["allowed"]:
                raise ResourceBlocked(
                    f"{action}: requested{count}, estimate={estimate['estimated_bytes'] / 1e9:.2f}GB; stopped before extraction, no silent downgrade"
                )
        frozen = self.rows(frozen, label)
        root = external(STORE / "processed/pretrain" / self.task.name / label)
        root.mkdir(parents=True, exist_ok=False)
        consumer = Path(frozen["runs"][0]["path"]).parent
        consumer.mkdir(parents=True, exist_ok=False)
        write(root / "frozen-manifest.json", frozen)
        self.batch(
            label + "-extract",
            "extract",
            frozen["runs"],
            self.config["io_workers"] if full else workers,
            root / "extract-evidence",
        )
        self.batch(
            label + "-preprocess",
            "preprocess",
            frozen["runs"],
            workers,
            root / "canonical-work",
            reference=reference,
        )
        return self.finalize_prepared(frozen, root, label, workers, full=full)

    def finalize_prepared(
        self, frozen, root, label, workers, *, full=True, canonical_sources=None, preaudited=None
    ):
        package_imports()
        canonical_sources = canonical_sources or {}
        self.begin(label + "-assemble", total=len(frozen["runs"]), workers=1)
        from . import assembly as pipeline

        class AssemblyBudget(Budget):
            last_check = 0.0

            def check(inner):
                super().check()
                if time.monotonic() - inner.last_check >= 1:
                    self.resources()
                    inner.last_check = time.monotonic()

        def adopt(raw, target, config):
            source = canonical_sources.get(Path(raw).name, root / "canonical-work" / Path(raw).name)
            summary = read(source / "summary.json")
            row = next(r for r in frozen["runs"] if r["run_id"] == Path(raw).name)
            if (
                summary["config"] != config
                or summary["input_hashes"] != row["files"]
                or not summary["closed"]
            ):
                raise ValueError("Completed canonical binding mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            if Path(raw).name in canonical_sources:
                # Preserve the successful prior results; no preprocessing is repeated.
                shutil.copytree(source, target, ignore=shutil.ignore_patterns("checkpoint.json"))
            else:
                source.rename(target)
            verify(
                {
                    str(target / (name + ".jsonl")): digest
                    for name, digest in summary["output_hashes"].items()
                }
            )
            self.update(done=self.state["done"] + 1)
            return summary

        dataset = pipeline.prepare(
            frozen,
            root / "dataset",
            read(PACKAGE / "preprocessing.json"),
            AssemblyBudget(),
            process_run=adopt,
        )
        self.finish(
            label + "-assemble", dict(workers=1, reason="fixed manifest order; train-only scaler")
        )
        if preaudited is None:
            self.batch(
                label + "-audit",
                "audit",
                frozen["runs"],
                self.config["audit_workers"] if full else workers,
                root / "audit-work",
                canonical=dataset / "canonical",
            )
        else:
            self.begin(label + "-audit", total=len(preaudited), workers=1)
            self.finish(
                label + "-audit", dict(passed=True, verified_saved_run_audits=len(preaudited))
            )
        return self.finish_quality(
            frozen, root, label, dataset, full=full, audit_sources=preaudited
        )

    def finish_quality(self, frozen, root, label, dataset, *, full=True, audit_sources=None):
        package_imports()
        self.begin(label + "-quality", workers=1)
        import numpy as np

        derived = load("pretrain_derived", ADMISSION / "derived.py")

        def audited(folder, row):
            if audit_sources is None:
                source = root / "audit-work" / row["run_id"]
                status = read(
                    self.task / "units" / (label + "-audit") / (row["run_id"] + ".status.json")
                )
            else:
                source, status_path = audit_sources[row["run_id"]]
                status = read(status_path)
                summary = read(folder / "summary.json")
                verify(
                    {
                        str(folder / (name + ".jsonl")): digest
                        for name, digest in summary["output_hashes"].items()
                    }
                )
            if status.get("status") != "COMPLETE" or status.get("run_id") != row["run_id"]:
                raise ValueError("Completed matching audit evidence required")
            verify(
                {
                    str(source / "numeric.npz"): status["result"]["numeric_sha256"],
                    str(source / "report.json"): status["result"]["report_sha256"],
                }
            )
            with np.load(source / "numeric.npz", allow_pickle=False) as data:
                samples = (
                    list(zip(*(data[k] for k in ("X", "C", "Y", "run_id", "origin_us", "segment"))))
                    if "X" in data
                    else []
                )
                return read(source / "report.json"), samples, list(data["train_frames"])

        if self.config.get("collection_profile") == "balanced300":
            from . import assembly

            class QualityBudget(Budget):
                last_check = 0.0

                def check(inner):
                    super().check()
                    if time.monotonic() - inner.last_check >= 1:
                        self.resources()
                        inner.last_check = time.monotonic()

            quality = assembly.inspect_dataset(
                dataset, frozen, QualityBudget(), inspect_run_fn=audited
            )
        else:
            quality = derived.inspect_dataset(dataset, frozen, inspect_run_fn=audited)
        seal = {
            str(p.relative_to(dataset)): sha(p) for p in sorted(dataset.rglob("*")) if p.is_file()
        }
        write(root / "dataset-seal.json", seal)
        write(
            root / "quality.json",
            dict(
                status="PRETRAIN_QUALITY_PASSED",
                passed=True,
                training_admission=False,
                final_test_used=False,
                **quality,
            ),
        )
        write(
            root / "handoff.json",
            dict(
                status="BENCHMARK_ONLY_STOPPED"
                if not full
                else "PRETRAIN_READY_PENDING_USER_REVIEW",
                training_eligible=full,
                dataset=str(dataset),
                frozen_manifest_sha256=sha(root / "frozen-manifest.json"),
                dataset_seal_sha256=sha(root / "dataset-seal.json"),
                quality_sha256=sha(root / "quality.json"),
                package_sha256=sha(PACKAGE / "PACKAGE.json"),
                runner_config_sha256=sha(self.task / "config.json"),
                training_authorized=False,
                requires_new_training_package_admission=True,
                scaler_scope="all_selected_normal_train",
                collection_sets=frozen.get("collection_sets", 1),
                run_count=len(frozen["runs"]),
            ),
        )
        self.finish(label + "-quality", dict(passed=True, handoff=str(root / "handoff.json")))
        if full and self.config.get("storage_profile") in {"compact", "stream_compact"}:
            from .storage import pack

            self.begin(label + "-portable", total=1, workers=1)
            portable = root.parent / (root.name + "-portable")

            class PackagingBudget(Budget):
                last_check = 0.0

                def check(inner):
                    super().check()
                    if time.monotonic() - inner.last_check >= 1:
                        self.resources()
                        inner.last_check = time.monotonic()

            receipt = pack(
                root, portable, progress=lambda **kw: self.update(**kw), budget=PackagingBudget()
            )
            self.finish(
                label + "-portable",
                dict(
                    passed=True,
                    output=str(portable),
                    bytes=receipt["dataset"]["bytes"],
                    raw_deleted=False,
                ),
            )
            self.update(portable_dataset=str(portable), raw_deleted=False)
        return root

    def flight(self, phase, count, rows, label):
        if self.config["mode"] in {
            "resume_preprocess",
            "finish_preprocess",
            "finish_audit",
            "merge_prepared",
            "recover_capsules",
        }:
            raise ValueError("Recollection forbidden in preprocessing recovery")
        from .collection import plan, run

        self.begin(label, workers=count, total=len(rows))
        # Long-run16 admission uses actual attempt07 8-lane peak, not a short120s projection.
        measurements = read(RESOURCES / "memory-reference.json")
        per_lane = measurements[
            "benchmark_per_lane_bytes" if phase == "benchmark" else "collection_per_lane_bytes"
        ]
        estimate = admission(count, per_lane, self.resources()["available_bytes"])
        estimate["estimate_only_override"] = bool(self.config.get("use_available_ram"))
        self.update(admission=estimate)
        if not estimate["passed"] and not estimate["estimate_only_override"]:
            raise ResourceBlocked(
                f"{label}: projected RAM does not fit; requested{count}, not executed"
            )
        exporters = count if self.config["mode"] == "test" else self.config["export_workers"]
        folder = plan(self.task, phase, count, rows, label, exporters)
        self.update(collection_ledger=str(folder / "execution/ledger.json"))
        result = run(folder)
        intervals = {}
        for key in ("flight", "verification", "archive"):
            intervals[key] = overlap(
                [
                    dict(
                        started=r[key + "_started_monotonic"],
                        finished=r[key + "_finished_monotonic"],
                    )
                    for r in result["cases"].values()
                ]
            )
        self.finish(
            label,
            dict(
                workers=count,
                export_workers=exporters,
                completed=len(result["cases"]),
                elapsed_s=result["elapsed_s"],
                measured_overlap=intervals,
                resources=result["resources"],
            ),
        )
        self.update(collection_ledger=None)
        return folder

    def benchmark(self):
        if self.config.get("speed_test"):
            from .speed_test import run

            run(self)
            return
        from .collection import freeze_benchmark

        outcomes = []
        for count in self.config["levels"]:
            try:
                # At least three independent runs exercise all scaler partitions.
                rows = [
                    dict(
                        case_id=f"probe-{count}-{i}",
                        scenario="hover_normal",
                        role="benchmark",
                        seed=self.config["seed_base"] + count * 100 + i,
                        group_id=f"probe-{count}-{i}",
                        mission_seconds=120,
                        retries=0,
                    )
                    for i in range(max(3, count))
                ]
                if self.config.get("collection_profile") == "diverse104":
                    from .collection import cases

                    templates = [
                        c
                        for c in cases(self.config["seed_base"], "diverse104")["cases"]
                        if c["role"] == "normal_train"
                    ]
                    names = sorted({c["scenario"] for c in templates})
                    from capstone_sim.normal_variation import sample

                    for i, row in enumerate(rows):
                        row["scenario"] = names[i % len(names)]
                        row["normal_variation"] = sample(row["scenario"], row["seed"])
                elif self.config.get("collection_profile") == "balanced300":
                    from .balanced import cases

                    templates = [
                        c
                        for c in cases(self.config["seed_base"] + count * 1000)
                        if c.get("path_mode")
                    ]
                    # Keep the exact program/seed binding;120s tests exercise only its initial part.
                    for row, template in zip(rows, templates):
                        for key in ("scenario", "seed", "normal_variation", "path_mode"):
                            row[key] = copy.deepcopy(template[key])
                folder = self.flight("benchmark", count, rows, f"flight-test-{count}")
                frozen = freeze_benchmark(folder)
                root = self.prepare(frozen, f"test-{count}", count, full=False)
                batches = [
                    r
                    for r in self.state["stages"]
                    if r["name"].startswith(f"test-{count}-") and "max_overlap" in r
                ]
                if len(batches) != 3 or any(r["max_overlap"] < count for r in batches):
                    raise RuntimeError("Requested worker overlap not observed; no parallel pass")
                outcomes.append(
                    dict(
                        workers=count,
                        status="PASSED",
                        output=str(root),
                        source=str(folder / "benchmark-manifest.json"),
                        fresh_runs=len(rows),
                        mission_seconds=120,
                        training_eligible=False,
                        cache="fresh files may remain in filesystem cache; cold-device speed not established",
                    )
                )
                write(self.task / "parallel-results.json", outcomes)
            except Exception as exc:
                outcomes.append(
                    dict(
                        workers=count,
                        status="BLOCKED_NOT_EXECUTED"
                        if isinstance(exc, ResourceBlocked)
                        else "FAILED_STOPPED",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
                write(self.task / "parallel-results.json", outcomes)
                raise
        self.update(result=str(self.task / "parallel-results.json"))

    def run(self):
        verify({str(self.task / "config.json"): (self.task / "config.sha256").read_text().strip()})
        verify(self.config["source_sha256"])
        verify_runtime(self.config)
        allocator = self.config.get("python_allocator")
        if allocator and (allocator != "debug" or os.environ.get("PYTHONMALLOC") != allocator):
            raise ValueError("Task Python allocator mismatch; start through drone-pretrain start")
        external(STORE).mkdir(parents=True, exist_ok=True)
        if self.config["mode"] in {"collect", "test"}:
            runs_root(ROOT).mkdir(parents=True, exist_ok=True)
        self.resources()
        with (self.task / "operator/claimed").open("x") as f:
            f.write(now())
        self.update()
        if self.config["mode"] == "test":
            self.benchmark()
        elif self.config["mode"] == "merge_prepared":
            from .merge import run

            run(self)
        elif self.config["mode"] == "recover_capsules":
            from .capsule_recovery import run

            run(self)
        elif self.config["mode"] == "resume_preprocess":
            from .recovery import validated_collections

            folders = validated_collections(Path(self.config["resume_source_task"]))
            if [str(p) for p in folders] != self.config["resume_collections"]:
                raise ValueError("Recovery input collection list changed")
            self.prepare_collections(folders)
        elif self.config["mode"] == "finish_preprocess":
            from .partial import finish_partial

            finish_partial(self)
        elif self.config["mode"] == "finish_audit":
            from .audit_recovery import finish_audit

            finish_audit(self)
        else:
            if self.config["mode"] == "collect":
                count = self.config["flight_workers"]
                gate_seed = self.config["seed_base"] + seed_stride(
                    self.config.get("collection_profile", "legacy50")
                ) * self.config.get("collection_sets", 1)
                if self.config["gates"]:
                    rows = [
                        dict(
                            case_id=f"optional-gate-{i}",
                            scenario="hover_normal",
                            role="benchmark",
                            seed=gate_seed + 2000 + i,
                            group_id=f"optional-gate-{i}",
                            mission_seconds=120,
                            retries=0,
                        )
                        for i in range(count)
                    ]
                    self.flight("benchmark", count, rows, "optional-flight-gate")
                    rows = [
                        dict(
                            case_id=f"optional-endurance-{i}",
                            scenario=scenario,
                            role="endurance",
                            seed=gate_seed + 3000 + i,
                            group_id=f"optional-endurance-{i}",
                            mission_seconds=seconds,
                            retries=0,
                        )
                        for i, (scenario, seconds) in enumerate(
                            (
                                ("hover_normal", 900),
                                ("payload_normal", 900),
                                ("low_soc_normal", 180),
                            )
                        )
                    ]
                    self.flight("endurance", count, rows, "optional-endurance-gate")
                if self.config.get("storage_profile") == "stream_compact":
                    from .streaming import run

                    run(self)
                else:
                    self.collection_sets()
            else:
                raise ValueError("Only new collection and fresh tests may execute")
        verify(self.config["source_sha256"])
        self.update(
            status="TEST_COMPLETE_STOPPED"
            if self.config["mode"] == "test"
            else "PRETRAIN_COMPACT_READY_STOPPED_RESTORE_REQUIRED"
            if self.state.get("dataset_requires_restore")
            else "PRETRAIN_READY_STOPPED_PENDING_USER_REVIEW",
            finished=time.time(),
            active=[],
            next_stage_authorized=False,
        )

    def collection_sets(self):
        from .collection import IncompleteCollection, cases

        sets = self.config.get("collection_sets", 1)
        folders = []
        incomplete = []
        planned = (
            read(self.config["planned_cases"])["sets"] if self.config.get("planned_cases") else None
        )
        if planned is not None and len(planned) != sets:
            raise ValueError("Frozen planned collection count differs")
        for index in range(sets):
            label = "collection" if sets == 1 else f"collection-set-{index + 1:03}"
            rows = (
                planned[index]
                if planned is not None
                else cases(
                    self.config["seed_base"]
                    + seed_stride(self.config.get("collection_profile", "legacy50")) * index,
                    self.config.get("collection_profile", "legacy50"),
                )["cases"]
            )
            try:
                folder = self.flight("collection", self.config["flight_workers"], rows, label)
            except IncompleteCollection as exc:
                incomplete.append(str(exc))
                self.update(incomplete_collections=incomplete)
                continue
            folders.append(folder)
            self.update(collected_sets=[str(p) for p in folders])
        if incomplete:
            write(
                self.task / "collection-errors.json",
                dict(
                    errors=incomplete,
                    completed_sets=[str(p) for p in folders],
                    preprocessing_started=False,
                    training_authorized=False,
                ),
            )
            raise RuntimeError(
                "Finite collection finished with run errors; no incomplete dataset admitted: "
                + "; ".join(incomplete)
            )
        self.prepare_collections(folders)

    def prepare_collections(self, folders):
        from .collection import combine_frozen, freeze_collection

        parts = []
        sets = len(folders)
        for index, folder in enumerate(folders):
            stage = "freeze" if sets == 1 else f"freeze-set-{index + 1:03}"
            self.begin(stage, workers=1, total=len(read(folder / "manifest.json")["cases"]))
            output = (
                self.task / ("recovery-freeze-" + str(index + 1))
                if self.config["mode"] == "resume_preprocess"
                else folder / "freeze"
            )
            frozen = freeze_collection(
                self.task, folder, output=output, progress=lambda done: self.update(done=done)
            )
            self.finish(stage, dict(passed=True))
            parts.append((output, frozen))
        self.begin("combine", workers=1)
        frozen = combine_frozen(self.task, parts)
        self.finish("combine", dict(passed=True, completed=len(frozen["runs"])))
        root = self.prepare(
            frozen, "prepared", self.config["workers"], cases_path=self.task / "cases.json"
        )
        self.update(result=str(root / "handoff.json"))


def worker(task):
    task = Path(task)
    # Existing task claim is checked before state writes; duplicate commands preserve evidence.
    if (task / "operator/claimed").exists():
        raise RuntimeError("STOP: Attempt already claimed; no automatic retry")
    lock = (HOME / "pretrain.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from x500_storage import environment

    expected = environment(read(task / "config.json"))
    if expected["X500_STORAGE_ROOT"] != str(STORE):
        raise ValueError("Worker storage mismatch; start through drone-pretrain start")
    os.environ.update(expected)
    runner = Runner(task)
    try:
        runner.run()
    except BaseException as exc:
        runner.update(
            status="BLOCKED_STOPPED" if isinstance(exc, ResourceBlocked) else "FAILED_STOPPED",
            reason=f"{type(exc).__name__}: {exc}",
            finished=time.time(),
            next_stage_authorized=False,
        )
        raise
