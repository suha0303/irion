"""Finite operator-run collection; archive-only storage; bounded per-case retries, no campaign restart."""

from __future__ import annotations

import argparse
import errno
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import socket
import stat
import tarfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def unit_attempts(plan):
    limit = plan.get("unit_attempts_max", 1)
    if type(limit) is not int or not 1 <= limit <= 3:
        raise Stop("Per-case attempt limit must be1..3")
    return limit


def shared_failure(exc):
    return (
        isinstance(exc, (MemoryError, KeyboardInterrupt, SystemExit))
        or (
            isinstance(exc, OSError)
            and exc.errno
            in {errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EIO, errno.ENOMEM, errno.ENODEV}
        )
        or bool(re.search(r"Command exit(?:=|:| )?(-9|-15|-2|137|143|130)(?:\D|$)", str(exc)))
    )


BLOCK = 1024 * 1024
HELPER_SHA = "ecfb0265876e5f7316094ccd7655ac1e258cc87cf2fa7806b70b5ff891c97885"
ROLES = {
    "normal_train",
    "normal_validation",
    "normal_calibration",
    "normal_dev_eval",
    "anomaly_dev",
    "normal_test",
    "anomaly_test",
    "benchmark",
    "endurance",
}

COLLECTION_PROTOCOL = "DRONE_COLLECTION"
# Historical plans are read without rewriting their identities or evidence.
LEGACY_COLLECTION_PROTOCOL = "R3_LONG15"
COLLECTION_PROTOCOLS = {COLLECTION_PROTOCOL, LEGACY_COLLECTION_PROTOCOL}
COLLECTION_CONFIGURATIONS = {"collection", "r3_long15"}
SPEED_TEST_ROOT = (
    Path(os.environ.get("X500_STORAGE_ROOT", str(Path(__file__).resolve().parents[2])))
    / "speed-tests"
)


def reference(ref):
    if sha(ref["path"]) != ref["sha256"]:
        raise Stop("Evidence hash mismatch: " + ref["path"])
    return read(ref["path"])


def duration_for(plan, case):
    return case.get("mission_seconds", plan["mission_seconds"])


def validate_case_conditions(plan, case):
    """Keep the new final-test operating axes separate from fault parameters."""
    variation = case.get("operation_variation")
    onset = case.get("fault_after_mission_s")
    final = plan.get("collection_profile") == "final324"
    if variation is not None and case.get("normal_variation") is not None:
        raise Stop("Normal and operation variation are mutually exclusive")
    if onset is not None and case.get("fault_at_s") is not None:
        raise Stop("Boot and mission-relative fault schedules are mutually exclusive")
    if not final:
        if variation is not None or onset is not None:
            raise Stop("Operating variation and mission-relative faults require final324")
        return
    if (
        plan.get("protocol") != COLLECTION_PROTOCOL
        or plan.get("phase") != "collection"
        or plan.get("dataset_purpose") != "locked_final_test"
        or case["role"] not in {"normal_test", "anomaly_test"}
    ):
        raise Stop("Final operating conditions require the locked collection test partition")
    fields = {"soc", "temperature", "payload", "wind_e", "wind_n", "gust_sigma", "path"}
    if not isinstance(variation, dict) or set(variation) != fields:
        raise Stop("Exact operation variation fields required; health/fault overrides forbidden")
    if any(
        type(variation[key]) not in (int, float) or not math.isfinite(variation[key])
        for key in fields - {"path"}
    ) or not isinstance(variation["path"], dict):
        raise Stop("Finite explicit operating conditions/path required")
    if not (
        0.4 <= variation["soc"] <= 0.95
        and 283.15 <= variation["temperature"] <= 303.15
        and 0 <= variation["payload"] <= 0.2
        and 0 <= variation["gust_sigma"] <= 0.4
        and math.hypot(variation["wind_e"], variation["wind_n"]) <= 2.000001
    ):
        raise Stop("Operating conditions exceed the existing engineering envelope")
    seconds = duration_for(plan, case)
    if seconds != (180 if variation["soc"] < 0.6 else 480):
        raise Stop("Final low-SOC operation requires180s; other conditions require480s")
    dynamic = case["scenario"] in {
        "battery_resistance_step",
        "battery_resistance_ramp",
        "propulsion_loss_step",
        "propulsion_loss_ramp",
    }
    if case.get("fault_at_s") is not None:
        raise Stop("Final dynamic anomalies require a mission-relative fault schedule")
    if dynamic:
        if (
            case["role"] != "anomaly_test"
            or type(onset) not in (int, float)
            or not math.isfinite(onset)
            or not 0 < onset < seconds
        ):
            raise Stop("Final dynamic anomaly requires a finite in-mission fault onset")
    elif onset is not None:
        raise Stop("Mission-relative fault onset only applies to dynamic anomalies")


def memory_used(snapshot):
    return max(
        snapshot["owned_tree_rss_bytes"],
        snapshot.get("cgroup_total_rss_bytes") or 0,
        snapshot.get("cgroup_memory_current_bytes") or 0,
    )


def expanded_memory(plan):
    return plan.get("pretrain_memory_override") or plan.get("test_memory_override")


def budget_memory_used(plan, snapshot):
    # Reclaimable file cache is accounted by the OS cap, not an RSS violation.
    return snapshot["owned_tree_rss_bytes"] if expanded_memory(plan) else memory_used(snapshot)


class MissionProgress:
    """Read tiny producer metadata only; bind a unique new UUID to this case/lane."""

    def __init__(self, root, baseline_dirs, case, lane, duration, *, runs_root=None):
        self.runs = Path(runs_root) if runs_root is not None else Path(root) / "runs"
        self.baseline = baseline_dirs
        self.case, self.lane, self.duration = case, lane, duration
        self.classified, self.matches = set(), set()
        self.previous = None

    @staticmethod
    def small_json(path):
        no_links(path)
        with path.open("rb") as stream:
            data = stream.read(256 * 1024 + 1)
        if len(data) > 256 * 1024:
            raise ValueError("Progress metadata exceeds size bound")
        return json.loads(data)

    def sample(self, now_utc=None, monotonic_now=None):
        result = dict(
            observed_simulation_elapsed_s=None,
            live_rtf=None,
            simulation_clock_basis="Producer progress unavailable or identity ambiguous",
            producer_run_id=None,
            producer_progress_updated_utc=None,
            producer_px4_boot_s=None,
            producer_target_s=None,
        )
        now_utc = now_utc or datetime.now(timezone.utc)
        monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
        try:
            # Names only. Once classified, unrelated metadata is never rescanned.
            for path in self.runs.iterdir():
                if (
                    path.name in self.baseline
                    or path.name in self.classified
                    or not re.fullmatch("[0-9a-f]{32}", path.name)
                ):
                    continue
                try:
                    state = self.small_json(path / "run.json")
                    scenario = self.small_json(path / "evaluation/scenario.json")
                    if "instance" not in state or "configuration" not in state:
                        continue  # Producer creates initial run.json before adding collection fields.
                    if state.get("configuration") in COLLECTION_CONFIGURATIONS and (
                        state.get("run_id"),
                        state.get("instance"),
                        scenario.get("name"),
                        scenario.get("seed"),
                    ) == (
                        path.name,
                        self.lane["instance"],
                        self.case["scenario"],
                        self.case["seed"],
                    ):
                        self.matches.add(path.name)
                    self.classified.add(path.name)
                except (OSError, ValueError, KeyError, TypeError, Stop):
                    continue
            if len(self.matches) != 1:
                self.previous = None
                return result
            run_id = next(iter(self.matches))
            state = self.small_json(self.runs / run_id / "run.json")
            if (
                state.get("run_id") != run_id
                or state.get("instance") != self.lane["instance"]
                or state.get("configuration") not in COLLECTION_CONFIGURATIONS
                or state.get("gz_partition") != "x500_" + run_id
                or state.get("clock")
                != {"id": "px4-boot", "epoch": "simulation_boot", "unit": "us"}
            ):
                raise ValueError("Producer identity or clock mismatch")
            progress = state["progress"]
            elapsed, target, boot = (
                progress[key] for key in ("sim_elapsed_s", "target_s", "px4_boot_s")
            )
            if (
                any(
                    type(x) not in (int, float) or not math.isfinite(x)
                    for x in (elapsed, target, boot)
                )
                or not 0 <= elapsed <= boot
                or target != self.duration
            ):
                raise ValueError("Invalid producer mission clock")
            updated = datetime.fromisoformat(progress["last_update_utc"])
            if updated.tzinfo is None or not 0 <= (now_utc - updated).total_seconds() <= 10:
                raise ValueError("Stale/future/unspecified producer clock")
            current = (updated, elapsed, boot, monotonic_now)
            rtf = None
            if self.previous:
                old_utc, old_elapsed, old_boot, old_monotonic = self.previous
                delta_wall = (updated - old_utc).total_seconds()
                if delta_wall < 0 or elapsed < old_elapsed or boot < old_boot:
                    raise ValueError("Producer clock moved backwards")
                if delta_wall == 0 and (elapsed != old_elapsed or boot != old_boot):
                    raise ValueError("Changed progress without producer timestamp")
                if delta_wall > 0:
                    if abs(delta_wall - (monotonic_now - old_monotonic)) > 2:
                        raise ValueError(
                            "Producer wall clock disagrees with local monotonic interval"
                        )
                    rtf = (elapsed - old_elapsed) / delta_wall
                    self.previous = current
            else:
                self.previous = current
            result.update(
                observed_simulation_elapsed_s=elapsed,
                live_rtf=rtf,
                simulation_clock_basis="run.json PX4 mission elapsed; interval RTF = delta sim / delta producer UTC; first/repeated sample RTF unknown",
                producer_run_id=run_id,
                producer_progress_updated_utc=progress["last_update_utc"],
                producer_px4_boot_s=boot,
                producer_target_s=target,
            )
        except (OSError, ValueError, KeyError, TypeError, Stop):
            self.previous = None
        return result

    def terminal_evidence(self):
        """Recover failure context even when the command exited before result parsing."""
        try:
            self.sample()  # A stopped run's live clock can be stale; identity is still useful.
            if len(self.matches) != 1:
                return {}
            run_id = next(iter(self.matches))
            path = self.runs / run_id / "run.json"
            state = self.small_json(path)
            scenario = self.small_json(self.runs / run_id / "evaluation/scenario.json")
            if (
                state.get("run_id"),
                state.get("instance"),
                state.get("gz_partition"),
                scenario.get("name"),
                scenario.get("seed"),
                scenario.get("requested_duration_s"),
            ) != (
                run_id,
                self.lane["instance"],
                "x500_" + run_id,
                self.case["scenario"],
                self.case["seed"],
                self.duration,
            ) or state.get("configuration") not in COLLECTION_CONFIGURATIONS:
                return {}
            if state.get("status") not in {"failed", "completed"}:
                return {}
            failure = state.get("primary_failure") or {}
            reason = state.get("termination_reason")
            stage = failure.get("stage")
            if (
                not isinstance(reason, str)
                or not reason
                or (stage is not None and not isinstance(stage, str))
            ):
                return {}
            return dict(
                producer_run_id=run_id,
                producer_status=state["status"],
                producer_stage=stage,
                producer_termination_reason=reason,
                producer_evidence=str(path),
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError, Stop):
            # Diagnostic reads must never replace the original execution failure.
            return {}


class Stop(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(BLOCK), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temp.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def no_links(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise Stop("Symlink path: " + str(path))
    return path


def files(root, missing_ok=False):
    try:
        entries = sorted(Path(root).iterdir())
    except FileNotFoundError:
        if missing_ok:
            return
        raise
    for path in entries:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            if missing_ok:
                continue
            raise
        yield path, mode
        if stat.S_ISDIR(mode):
            yield from files(path, missing_ok)


def bytes_under(root):
    total = 0
    if root.exists():
        for path, mode in files(root, missing_ok=True):
            if stat.S_ISREG(mode):
                try:
                    total += path.lstat().st_size
                except FileNotFoundError:
                    pass  # Atomic progress-file replacement during approximate accounting.
    return total


class Progress:
    """Only increasing byte counters reset the stall clock. State writes do not."""

    def __init__(self, path, stage, wall_s, stall_s, check=lambda: None, clock=time.monotonic):
        self.path, self.stage, self.clock, self.check_external = path, stage, clock, check
        self.started = self.last_progress = self.last_write = clock()
        self.last_external = self.started - 2
        self.wall_s, self.stall_s, self.bytes = wall_s, stall_s, 0
        self.persist("RUNNING")

    def check(self):
        now = self.clock()
        if now - self.last_external >= 1:
            self.check_external()
            self.last_external = now
        if now - self.started >= self.wall_s:
            raise Stop(self.stage + " wall budget exceeded")
        if now - self.last_progress >= self.stall_s:
            raise Stop(self.stage + " byte-progress timeout")

    def advance(self, count):
        self.check()
        if count > 0:
            self.bytes += count
            self.last_progress = self.clock()
        if self.clock() - self.last_write >= 1:
            self.persist("RUNNING")

    def persist(self, status):
        atomic(
            self.path,
            dict(
                stage=self.stage,
                status=status,
                bytes_processed=self.bytes,
                counter_semantics="byte operations, may count raw and compressed passes; not unique raw pool bytes",
                elapsed_s=self.clock() - self.started,
                byte_progress_age_s=self.clock() - self.last_progress,
            ),
        )
        self.last_write = self.clock()

    def finish(self):
        self.check()
        self.persist("COMPLETED")
        return self.clock() - self.started


class HashIO:
    def __init__(self, stream, progress):
        self.stream, self.progress, self.digest, self.count = stream, progress, hashlib.sha256(), 0

    def read(self, n=-1):
        self.progress.check()
        data = self.stream.read(BLOCK if n < 0 else min(n, BLOCK))
        self.digest.update(data)
        self.count += len(data)
        self.progress.advance(len(data))
        return data

    def write(self, data):
        self.progress.check()
        n = self.stream.write(data)
        if n != len(data):
            raise Stop("Short archive write")
        self.digest.update(data)
        self.count += n
        self.progress.advance(n)
        return n

    def flush(self):
        self.stream.flush()


def archive_run(run, destination, stage_dir, budgets, check):
    """One streaming gzip tar, then streaming stored-byte and member verification."""
    run, destination = Path(run), Path(destination)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() or partial.exists():
        raise Stop("Archive already exists; no overwrite/resume")
    inventory = {}
    progress = Progress(
        stage_dir / "archive-progress.json",
        "archive_transfer",
        budgets["archive_wall_s"],
        budgets["byte_stall_s"],
        check,
    )
    try:
        with partial.open("xb") as output:
            sink = HashIO(output, progress)
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=sink, compresslevel=1, mtime=0
            ) as gz:
                with tarfile.open(fileobj=gz, mode="w|", dereference=False) as archive:
                    for path, mode in files(run):
                        progress.check()
                        name = str(path.relative_to(run))
                        member = archive.gettarinfo(path, arcname=name)
                        if stat.S_ISREG(mode):
                            if not member.isfile():
                                raise Stop("Hardlink/special tar member requires review")
                            with path.open("rb") as source:
                                wrapped = HashIO(source, progress)
                                archive.addfile(member, wrapped)
                            if wrapped.count != member.size or path.stat().st_size != member.size:
                                raise Stop("Source file changed during archive")
                            inventory[name] = dict(
                                kind="file", bytes=member.size, sha256=wrapped.digest.hexdigest()
                            )
                        elif stat.S_ISDIR(mode):
                            archive.addfile(member)
                            inventory[name] = dict(kind="directory")
                        elif stat.S_ISLNK(mode) and path.is_dir():
                            archive.addfile(member)
                            inventory[name] = dict(kind="symlink", target=os.readlink(path))
                        else:
                            raise Stop("File/broken symlink or special file requires review")
            output.flush()
            os.fsync(output.fileno())
            digest, compressed_bytes = sink.digest.hexdigest(), sink.count
        archive_s = progress.finish()
        integrity = read(run / "integrity.json")
        for name, expected in integrity.items():
            item = inventory.get(name, {})
            if (item.get("bytes"), item.get("sha256")) != (expected["size"], expected["sha256"]):
                raise Stop("Source integrity mismatch: " + name)
        verify = Progress(
            stage_dir / "archive-verify-progress.json",
            "archive_verify",
            budgets["archive_verify_wall_s"],
            budgets["byte_stall_s"],
            check,
        )
        actual = {}
        try:
            with partial.open("rb") as source:
                wrapped = HashIO(source, verify)
                with gzip.GzipFile(mode="rb", fileobj=wrapped) as gz:
                    with tarfile.open(fileobj=gz, mode="r|") as archive:
                        for member in archive:
                            verify.check()
                            if member.name in actual:
                                raise Stop("Duplicate archive member")
                            if member.isfile():
                                h, count = hashlib.sha256(), 0
                                stream = archive.extractfile(
                                    member
                                )  # streaming, no disk extraction
                                for chunk in iter(lambda: stream.read(BLOCK), b""):
                                    h.update(chunk)
                                    count += len(chunk)
                                    verify.advance(len(chunk))
                                actual[member.name] = dict(
                                    kind="file", bytes=count, sha256=h.hexdigest()
                                )
                            elif member.isdir():
                                actual[member.name] = dict(kind="directory")
                            elif member.issym():
                                actual[member.name] = dict(kind="symlink", target=member.linkname)
                            else:
                                raise Stop("Unexpected archive member type")
                    while gz.read(BLOCK):  # consume gzip footer and verify CRC
                        verify.check()
                while wrapped.read(BLOCK):
                    pass
                if (
                    wrapped.digest.hexdigest() != digest
                    or wrapped.count != compressed_bytes
                    or actual != inventory
                ):
                    raise Stop("Stored archive/member hash mismatch")
            verify_s = verify.finish()
        except BaseException:
            verify.persist("FAILED")
            raise
        check()
        if destination.exists():
            raise Stop("Archive publish collision")
        os.rename(partial, destination)  # same filesystem, exclusive campaign writer
    except BaseException:
        progress.persist("FAILED")
        raise
    return dict(
        archive=str(destination),
        sha256=digest,
        bytes=compressed_bytes,
        archive_wall_s=archive_s,
        archive_verify_wall_s=verify_s,
        members=inventory,
    )


def memory_snapshot():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        info[key] = int(value.split()[0]) * 1024
    # Current CLI plus descendants, never other users' unrelated processes.
    processes = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
            processes[int(p.name)] = (int(fields[1]), int(fields[21]) * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            continue
    selected = {os.getpid()}
    while True:
        larger = selected | {pid for pid, (parent, _) in processes.items() if parent in selected}
        if larger == selected:
            break
        selected = larger
    result = {
        "available_bytes": info["MemAvailable"],
        "owned_tree_rss_bytes": sum(processes.get(pid, (0, 0))[1] for pid in selected),
        "cgroup_total_rss_bytes": None,
        "cgroup_memory_current_bytes": None,
        "cgroup_path": None,
    }
    try:
        group = next(
            line[3:]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        base = Path("/sys/fs/cgroup") / group.lstrip("/")
        pids = set()
        for file in base.rglob("cgroup.procs"):
            pids.update(int(pid) for pid in file.read_text().split())
        result.update(
            cgroup_path=group,
            cgroup_total_rss_bytes=sum(processes.get(pid, (0, 0))[1] for pid in pids),
            cgroup_memory_current_bytes=int((base / "memory.current").read_text()),
        )
    except (OSError, ValueError, StopIteration):
        pass  # Explicit null means unavailable, never zero or successful cgroup isolation.
    return result


def collection_prerequisites(plan, dry_run=False):
    """Bind the selected collection identity to its producer, gate and resource checks."""
    phase = plan.get("phase")
    legacy = plan.get("protocol") == LEGACY_COLLECTION_PROTOCOL
    expected_pool = "ROUND_001/R3_LONG15" if legacy else "collection/x500"
    expected_configuration = "r3_long15" if legacy else "collection"
    if plan.get("data_pool_id") != expected_pool:
        raise Stop("Collection requires its separate data pool")
    if phase not in {"benchmark", "endurance", "collection"}:
        raise Stop("Collection phase required")
    if not plan.get("profile_id") or not plan.get("assessment_command"):
        raise Stop("Collection profile and Simulation assessment command required")
    if plan.get("producer_args") != ["--configuration", expected_configuration]:
        raise Stop("Collection requires explicit new producer configuration")
    if not plan.get("assessment_code_paths") or any(
        p not in plan["source_hashes"] for p in plan["assessment_code_paths"]
    ):
        raise Stop("Assessment implementation must be in source lock")
    preparation = reference(plan["producer_preparation"])
    # The coordinator supplies exact JSON top-level assertions from Simulation's report.
    assertions = plan.get("preparation_assertions", {})
    if not assertions or any(preparation.get(k) != v for k, v in assertions.items()):
        raise Stop("Producer preparation assertions not satisfied")
    override = expanded_memory(plan)
    if override:
        config = reference(override["config"])
        memory = reference(override["memory"])
        allowed_phase = (config.get("mode") == "test" and plan["phase"] == "benchmark") or (
            config.get("mode") == "collect"
            and (
                plan["phase"] == "collection"
                or (config.get("gates") is True and plan["phase"] in {"benchmark", "endurance"})
            )
        )
        if (
            plan.get("gate_policy") != "optional_pretrain_cli_v1"
            or plan.get("pretrain_cli_approval") != override["config"]["path"]
            or not allowed_phase
            or config.get("use_available_ram") is not True
            or config.get("min_available_ram_gb") != 1
            or memory.get("reserve_bytes") != 1_000_000_000
            or memory["max_bytes"] != memory["available_bytes"] - 1_000_000_000
            or plan["budgets"]["min_available_ram_bytes"] != 1_000_000_000
            or plan["budgets"]["memory_soft_cap_bytes"] != memory["max_bytes"]
        ):
            raise Stop("Invalid finite-task memory override")
    elif (
        plan["budgets"]["min_available_ram_bytes"] < 4_000_000_000
        or plan["budgets"]["memory_soft_cap_bytes"] > 8_000_000_000
    ):
        raise Stop("Collection requires RAM headroom >=4GB and RSS cap <=8GB")
    thresholds = plan["quality_thresholds"]
    for key in ("min_effective_fraction", "min_effective_rtf"):
        if (
            not isinstance(thresholds[key], (int, float))
            or not math.isfinite(thresholds[key])
            or thresholds[key] <= 0
        ):
            raise Stop("Finite positive quality threshold required")
    if thresholds["min_effective_fraction"] > 1:
        raise Stop("Effective fraction must be <=1")
    optional = plan.get("gate_policy") == "optional_pretrain_cli_v1"
    if optional and not plan.get("pretrain_cli_approval"):
        raise Stop("Explicit standalone pretrain CLI approval required")
    if not dry_run and phase == "collection" and not optional:
        receipt = reference(plan["gate_receipt"])
        if (
            receipt.get("status") != "ENDURANCE_PASSED"
            or receipt.get("profile_id") != plan["profile_id"]
            or receipt.get("source_hashes") != plan["source_hashes"]
            or receipt.get("collector_sha256") != sha(__file__)
            or receipt.get("producer_preparation") != plan["producer_preparation"]
            or receipt.get("selected_concurrency") != plan["concurrency"]
        ):
            raise Stop("Matching benchmark/endurance receipt required before collection")
        benchmark = reference(receipt["benchmark_receipt"])
        if (
            benchmark.get("status") not in {"BENCHMARK_PASSED", "BENCHMARK_8_NOT_TESTED"}
            or plan["concurrency"] not in benchmark.get("eligible_concurrency", [])
            or any(
                benchmark.get(k) != receipt.get(k)
                for k in (
                    "source_hashes",
                    "profile_id",
                    "collector_sha256",
                    "driver_sha256",
                    "producer_preparation",
                )
            )
        ):
            raise Stop("Endurance benchmark chain mismatch")
    if phase in {"benchmark", "endurance"} and not dry_run:
        raise Stop("Collection gates must use the finite gate driver, not direct collector CLI")


def bound(plan_path, manifest_path, approval_path, dry_run=False):
    plan, manifest, approval = map(read, (plan_path, manifest_path, approval_path))
    for key, path in [
        ("plan_sha256", plan_path),
        ("manifest_sha256", manifest_path),
        ("collector_sha256", __file__),
    ]:
        if approval.get(key) != sha(path):
            raise Stop("Approval hash mismatch: " + key)
    if not dry_run and (
        plan.get("execution_authorized") is not True
        or approval.get("authorized") is not True
        or approval.get("action") != "START_LONG_COLLECTION"
    ):
        raise Stop("Explicit finite collection approval required")
    if not re.fullmatch("[A-Za-z0-9_-]+", plan["campaign_id"]):
        raise Stop("Unsafe campaign ID")
    collection = plan.get("protocol") in COLLECTION_PROTOCOLS
    duration, concurrency = (
        plan.get("mission_seconds", 900 if collection else 1200),
        plan.get("concurrency", 2),
    )
    allowed = {120} if collection and plan.get("phase") == "benchmark" else {180, 480, 900}
    if (
        type(duration) is not int
        or (duration not in allowed if collection else not 1200 <= duration <= 1800)
        or type(concurrency) is not int
        or not 1 <= concurrency <= (16 if collection else 2)
    ):
        raise Stop("Require legacy 1200..1800/concurrency1..2 or collection duration/lanes")
    lanes = plan["lanes"]
    if len(lanes) != concurrency or len({x["instance"] for x in lanes}) != concurrency:
        raise Stop("Distinct instance per lane required")
    ports = [p for lane in lanes for p in lane["ports"]]
    if len(ports) != len(set(ports)) or any(
        type(p) is not int or not 1 <= p <= 65535 for p in ports
    ):
        raise Stop("Port overlap/invalid port")
    for lane in lanes:
        if (
            type(lane["instance"]) is not int
            or not 0 <= lane["instance"] <= (15 if collection else 9)
            or 14540 + lane["instance"] not in lane["ports"]
        ):
            raise Stop("Invalid producer instance/observation port")
        if collection and not {
            base + lane["instance"] for base in (14540, 14580, 18570, 14280, 13030)
        }.issubset(lane["ports"]):
            raise Stop("Collection requires every instance-local port")
    evidence = read(plan["isolation_evidence"]["path"])
    if sha(plan["isolation_evidence"]["path"]) != plan["isolation_evidence"]["sha256"]:
        raise Stop("Isolation evidence hash mismatch")
    if not dry_run and any(
        evidence.get(k) is not True
        for k in ("parallel_safe", "uuid_partitioned", "immutable_shared_inputs")
    ):
        raise Stop("Simulation isolation review required")
    limit = unit_attempts(plan)
    if limit > 1:
        config = read(plan["pretrain_cli_approval"])
        if config.get("unit_attempts_max") != limit:
            raise Stop("Retry policy must match the task configuration")
    if plan.get("continue_after_case_failure"):
        config = read(plan["pretrain_cli_approval"])
        if config.get("continue_after_case_failure") is not True:
            raise Stop("Continue policy must match the task configuration")
    if evidence.get("lanes") != lanes or evidence.get("source_hashes") != plan["source_hashes"]:
        raise Stop("Isolation review does not cover lanes/source revision")
    if not plan["source_hashes"]:
        raise Stop("Nonempty source lock required")
    source_entries = list(plan["source_hashes"].items()) + list(
        plan.get("project_source_hashes", {}).items()
    )
    if plan.get("project_source_hashes") and not plan.get("project_root"):
        raise Stop("Project sources require a separate project root")
    if evidence.get("project_source_hashes", {}) != plan.get("project_source_hashes", {}):
        raise Stop("Isolation review does not cover project sources")
    for name, digest in source_entries:
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not re.fullmatch("[0-9a-f]{64}", digest)
        ):
            raise Stop("Unsafe source lock")
    cases = manifest["cases"]
    if (
        (not cases and not dry_run)
        or len(cases) != plan["attempts_max"]
        or len({c["case_id"] for c in cases}) != len(cases)
    ):
        raise Stop("Explicit finite unique cases required")
    signatures, partitions = set(), {}
    for case in cases:
        if (
            not re.fullmatch("[A-Za-z0-9_-]+", case["case_id"])
            or case["role"] not in ROLES
            or type(case["seed"]) is not int
        ):
            raise Stop("Invalid case/role/seed")
        if (
            (
                type(case.get("mission_seconds", duration)) is not int
                or case.get("mission_seconds", duration) not in allowed
                if collection
                else case.get("mission_seconds", duration) != duration
            )
            or case.get("parent_attempt_id")
            or case.get("retries", 0)
        ):
            raise Stop("Duration drift/retry forbidden")
        validate_case_conditions(plan, case)
        if (
            collection
            and plan["phase"] != "benchmark"
            and plan.get("collection_profile") != "final324"
        ):
            seconds = case.get("mission_seconds", duration)
            if (case["scenario"] == "low_soc_normal") != (seconds == 180):
                raise Stop("Collection low-SOC requires180s; other scenarios cannot use180s")
        signature = (case["scenario"], case["seed"])
        partition = (
            case["role"]
            if case["role"] in {"normal_train", "normal_validation", "normal_calibration"}
            else "test"
            if case["role"] in {"normal_test", "anomaly_test"}
            else "development"
        )
        for key in (("seed", case["seed"]), ("group", case["group_id"])):
            if key in partitions and partitions[key] != partition:
                raise Stop("Seed/group split leakage")
            partitions[key] = partition
        if signature in signatures:
            raise Stop("Duplicate seed/scenario claim")
        signatures.add(signature)
    budgets = plan["budgets"]
    if (
        type(plan.get("export_workers", 1)) is not int
        or not 1 <= plan.get("export_workers", 1) <= concurrency
    ):
        raise Stop("Export workers must be within1..concurrency")
    for key in (
        "flight_wall_s",
        "verify_wall_s",
        "archive_wall_s",
        "archive_verify_wall_s",
        "byte_stall_s",
        "campaign_wall_s",
        "local_bytes_cap",
        "external_bytes_cap",
        "min_free_bytes",
        "min_available_ram_bytes",
        "memory_soft_cap_bytes",
    ):
        if (
            type(budgets[key]) not in (int, float)
            or not math.isfinite(budgets[key])
            or budgets[key] <= 0
        ):
            raise Stop("Positive stage budgets required")
    if budgets["flight_wall_s"] < max(
        [duration] + [case.get("mission_seconds", duration) for case in cases]
    ):
        raise Stop("Flight wall budget shorter than mission")
    plan = dict(
        plan, mission_seconds=duration, concurrency=concurrency, timing=plan.get("timing", False)
    )
    if collection:
        if manifest.get("data_pool_id") != plan.get("data_pool_id"):
            raise Stop("Collection manifest pool mismatch")
        collection_prerequisites(plan, dry_run)
        for case in cases:
            if (plan["phase"] in {"benchmark", "endurance"}) != (case["role"] == plan["phase"]):
                raise Stop("Gate cases cannot enter collection splits")
    return plan, cases


def helper(root):
    path = Path(__file__).resolve().with_name("collection_support.py")
    if sha(path) != HELPER_SHA:
        raise Stop("Frozen process cleanup helper mismatch")
    spec = importlib.util.spec_from_file_location("_long_cleanup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_instance_socket(instance):
    path = Path("/tmp") / f"px4-sock-{instance}"
    if os.path.lexists(path):
        raise Stop("Existing PX4 instance socket; never removed: " + str(path))


def verify_run_source_snapshot(plan, source):
    for name, digest in plan["source_hashes"].items():
        if (
            name.startswith(("sim/", "profiles/", "contracts/"))
            and source.get("local_sources", {}).get(name) != digest
        ):
            raise Stop("Run source snapshot mismatch: " + name)
    if plan.get("protocol") in COLLECTION_PROTOCOLS:
        # The installed native runtime is sealed per host from the tracked source recipe.
        names = (
            "vendor/PX4-Autopilot/build/px4_sitl_default/bin/px4",
            "build/current/libDronePowertrain.so",
        )
        transport = "runtime/gazebo-transport/lib/libgz-transport13.so.13.6.0"
        if transport in plan["source_hashes"]:
            if "artifacts/runtime-manifest.json" not in plan["source_hashes"]:
                raise Stop("Private transport requires the installed producer runtime seal")
            names += (transport,)
        if any(name not in plan["source_hashes"] for name in names):
            raise Stop("Collection active runtime binary hashes missing from plan")
        expected = {name: plan["source_hashes"][name] for name in names}
        if source.get("runtime_binary_sha256") != expected:
            raise Stop("Run collection runtime binary snapshot mismatch (PX4 / collection plugin)")
    else:
        for name, digest in plan["source_hashes"].items():
            if (
                name.startswith(("vendor/", "build/"))
                and source.get("runtime_binary_sha256", {}).get(name) != digest
            ):
                raise Stop("Run runtime binary snapshot mismatch")


def prepare_external_parent(external, process_helper, mount=Path("/media/sa/X31")):
    """Create output parents only on the mounted device; never recreate the mount."""
    external = no_links(external)
    if not external.is_relative_to(mount / "Data") or not mount.is_mount():
        raise Stop("External mount absent/outside scope; no internal-disk fallback")
    device = mount.stat().st_dev
    for part in (external.parent, *external.parent.parents):
        if part.is_relative_to(mount) and part.exists():
            if not part.is_dir() or part.stat().st_dev != device:
                raise Stop("External output parent is not a directory on the mounted device")
    external.parent.mkdir(parents=True, exist_ok=True)
    process_helper.check_mount(external.parent)


def validate_storage_selection(plan):
    if plan.get("storage_policy") is None:
        return  # Old frozen plans retain their original mandatory X31 mount checks.
    config = reference(plan["storage_config"])
    if (
        config.get("storage_policy") != plan["storage_policy"]
        or config.get("external_storage") != plan["storage_policy"]["root"]
        or plan["storage_config"]["path"] != plan.get("pretrain_cli_approval")
    ):
        raise Stop("Selected storage differs from frozen task configuration")


def check_output_storage(plan, external, process_helper):
    policy = plan.get("storage_policy")
    if policy is None:
        return process_helper.check_mount(external.parent)
    from x500_storage import guard

    return guard(external.parent, policy)


def preflight(plan, process_helper):
    validate_storage_selection(plan)
    root, external = Path(plan["root"]), no_links(plan["external_output"])
    local = no_links(plan["execution_dir"])
    if external.exists() or local.exists():
        raise Stop("New exclusive output directories required")
    project = no_links(Path(plan.get("project_root", root)))
    if plan.get("project_root") and root != project / "simulation":
        raise Stop("Native/project root mismatch")
    if not (
        local.is_relative_to(project / "Data/operations")
        or (not plan.get("project_root") and local.is_relative_to(project / "ai"))
    ):
        raise Stop("Output scope mismatch")
    if plan.get("storage_policy"):
        from x500_storage import guard

        guard(external, plan["storage_policy"])
        if not external.is_relative_to(Path(plan["storage_policy"]["root"]) / "archives"):
            raise Stop("Output archive category mismatch")
        external.parent.mkdir(parents=True, exist_ok=True)
        check_output_storage(plan, external, process_helper)
    else:
        if not external.is_relative_to(Path("/media/sa/X31/Data")):
            raise Stop("Legacy output scope mismatch")
        prepare_external_parent(external, process_helper)
    budgets = plan["budgets"]
    if (
        min(shutil.disk_usage(root).free, shutil.disk_usage(external.parent).free)
        < budgets["min_free_bytes"]
    ):
        raise Stop("Disk preflight failed")
    mem = memory_snapshot()
    if (
        mem["available_bytes"] < budgets["min_available_ram_bytes"]
        or budget_memory_used(plan, mem) > budgets["memory_soft_cap_bytes"]
    ):
        raise Stop("RAM preflight failed")
    for lane in plan["lanes"]:
        check_instance_socket(lane["instance"])
        for port in lane["ports"]:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind(("127.0.0.1", port))
    return root, external, local


def assess_run(plan, case, lane, run, folder, process_helper, check, flight_wall):
    """Consume Simulation's assessment; never infer safety from completed status."""
    values = {
        "run": str(run),
        "instance": str(lane["instance"]),
        "duration": str(duration_for(plan, case)),
        "profile_id": plan["profile_id"],
    }
    args = [arg.format(**values) for arg in plan["assessment_command"]]
    process_helper.command(
        args,
        Path(plan["root"]),
        folder / "assessment.json",
        check,
        plan["budgets"]["verify_wall_s"],
    )
    result = read(folder / "assessment.json")
    if (
        result.get("schema") not in {"drone.run-assessment/1", "x500.r3.run-assessment/1"}
        or result.get("run_id") != run.name
        or result.get("instance") != lane["instance"]
        or result.get("gz_partition") != "x500_" + run.name
        or result.get("profile_id") != plan["profile_id"]
    ):
        raise Stop("Simulation assessment identity/profile/instance mismatch")
    for key in (
        "contract_passed",
        "instance_isolation_passed",
        "missing_data_passed",
        "physical_gate_passed",
    ):
        if result.get(key) is not True:
            raise Stop("Simulation assessment failed: " + key)
    seconds = result.get("observed_valid_seconds")
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 < seconds <= duration_for(plan, case)
        or flight_wall <= 0
    ):
        raise Stop("Measured observation duration/wall required")
    result["flight_command_wall_s"] = flight_wall
    result["effective_rtf"] = seconds / flight_wall
    result["effective_fraction"] = seconds / duration_for(plan, case)
    thresholds = plan["quality_thresholds"]
    if (
        result["effective_fraction"] < thresholds["min_effective_fraction"]
        or result["effective_rtf"] < thresholds["min_effective_rtf"]
    ):
        raise Stop("Observation coverage or effective RTF gate failed")
    return result


def execute(plan, cases, process_helper):
    """One project collection lock shared by optional gates and production collection."""
    lock_path = Path(plan.get("project_root", plan["root"])) / "Data/operations/collection.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Stop("Project collection lock held; no new processes") from exc
        try:
            return _execute_locked(plan, cases, process_helper)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def candidate_baseline(config):
    """Only an explicitly approved single3x/8x probe may reuse this stopped test seed."""
    speed = config.get("speed_test", {})
    if (
        config.get("mode") != "test"
        or speed.get("factors") not in ([3], [8])
        or speed.get("protocol") != "information-v2"
        or speed.get("candidate_only") is not True
        or speed.get("continue_after_lower_speed_failure") is not True
    ):
        raise Stop("Explicit finite single-candidate probe required")
    baseline = reference(speed["baseline_review"])
    source = Path(baseline["source_task"])
    old_config = reference(speed["baseline_config"])
    if (
        Path(speed["baseline_config"]["path"]) != source / "config.json"
        or sha(source / "config.json") != (source / "config.sha256").read_text().strip()
        or read(source / "operator/state.json")["status"]
        not in {"FAILED_STOPPED", "TEST_COMPLETE_STOPPED"}
        or old_config.get("mode") != "test"
        or old_config.get("speed_test", {}).get("factors") != [2, 2, 4, 8]
        or old_config.get("seed_base") != config.get("seed_base")
        or baseline.get("protocol") != "information-v2"
        or len(baseline.get("results", [])) != 3
        or [r["requested_factor"] for r in baseline["results"]] != [2, 2, 4]
    ):
        raise Stop("Stopped matched baseline configuration required")
    original = reference(
        dict(
            path=str(SPEED_TEST_ROOT / source.name / "results.json"),
            sha256=baseline["original_results_sha256"],
        )
    )
    ids = [r["measured"]["raw_run"] for r in baseline["results"]]
    if ids != [r["measured"]["raw_run"] for r in original["results"]] or len(set(ids)) != 3:
        raise Stop("Baseline evidence identity mismatch")
    for i, rid in enumerate(ids):
        ledger = read(source / f"speed-{i + 1:02}/execution/ledger.json")
        entries = list(ledger["cases"].values())
        if (
            ledger.get("status") != "COLLECTED_PENDING_QUALITY_STOPPED"
            or len(entries) != 1
            or entries[0]["run_id"] != rid
            or entries[0]["case"]["scenario"] != "hover_normal"
            or entries[0]["case"]["seed"] != config["seed_base"] + 2000
        ):
            raise Stop("Baseline ledger/seed mismatch")
    allowed = set(ids)
    for previous in speed.get("previous_candidates", []):
        prior_config = reference(previous["config"])
        prior_results = reference(previous["results"])
        prior_task = Path(previous["config"]["path"]).parent
        prior_speed = prior_config.get("speed_test", {})
        if (
            Path(previous["results"]["path"]) != SPEED_TEST_ROOT / prior_task.name / "results.json"
            or sha(prior_task / "config.json") != (prior_task / "config.sha256").read_text().strip()
            or read(prior_task / "operator/state.json")["status"]
            not in {"FAILED_STOPPED", "TEST_COMPLETE_STOPPED"}
            or prior_config.get("mode") != "test"
            or prior_config.get("seed_base") != config["seed_base"]
            or prior_speed.get("candidate_only") is not True
            or prior_speed.get("factors") not in ([3], [8])
            or prior_speed.get("baseline_review") != speed["baseline_review"]
            or prior_speed.get("baseline_config") != speed["baseline_config"]
            or len(prior_results.get("results", [])) != 1
            or [r["requested_factor"] for r in prior_results["results"]] != prior_speed["factors"]
        ):
            raise Stop("Previous candidate evidence mismatch")
        rid = prior_results["results"][0]["measured"]["raw_run"]
        ledger = read(prior_task / "speed-01/execution/ledger.json")
        entries = list(ledger["cases"].values())
        if (
            ledger.get("status") != "COLLECTED_PENDING_QUALITY_STOPPED"
            or len(entries) != 1
            or entries[0]["run_id"] != rid
            or entries[0]["case"]["scenario"] != "hover_normal"
            or entries[0]["case"]["seed"] != config["seed_base"] + 2000
        ):
            raise Stop("Previous candidate ledger/seed mismatch")
        allowed.add(rid)
    return baseline, allowed


def speed_comparison_scope(plan, cases):
    scope = plan.get("speed_comparison")
    if scope is None:
        return None
    override = expanded_memory(plan)
    if not override:
        raise Stop("Speed comparison requires frozen test configuration")
    config = reference(override["config"])
    index = scope.get("index")
    single = config.get("speed_test", {}).get("candidate_only", False)
    factors = config.get("speed_test", {}).get("factors")
    if (
        scope.get("protocol") not in {"information-v1", "information-v2"}
        or scope.get("protocol") != config.get("speed_test", {}).get("protocol", "information-v1")
        or config.get("mode") != "test"
        or (factors not in ([3], [8]) if single else factors != [2, 2, 4, 8])
        or type(index) is not int
        or index not in range(len(factors))
        or scope.get("factor") != config["speed_test"]["factors"][index]
        or plan.get("phase") != "benchmark"
        or plan.get("concurrency") != 1
        or unit_attempts(plan) != 1
        or len(cases) != 1
        or cases[0].get("role") != "benchmark"
        or cases[0].get("scenario") != "hover_normal"
        or cases[0].get("seed") != config["seed_base"] + 2000
    ):
        raise Stop("Invalid controlled speed comparison; no seed reuse outside this finite test")
    if single:
        return candidate_baseline(config)[1]
    previous = set()
    task = Path(plan["pretrain_cli_approval"]).parent
    for i in range(index):
        folder = task / f"speed-{i + 1:02}"
        old_plan = read(folder / "plan.json")
        ledger = read(folder / "execution/ledger.json")
        if (
            old_plan.get("pretrain_cli_approval") != plan["pretrain_cli_approval"]
            or ledger.get("status") != "COLLECTED_PENDING_QUALITY_STOPPED"
        ):
            raise Stop("Earlier controlled comparison did not complete")
        previous.update(r["run_id"] for r in ledger["cases"].values())
    return previous


def _execute_locked(plan, cases, process_helper):
    limit = unit_attempts(plan)
    root, external, local = preflight(plan, process_helper)
    processing_signal = plan.get("processing_stop_signal")
    if processing_signal:
        expected = (
            Path(plan.get("pretrain_cli_approval", "")).parent / "operator/processing-stop.json"
        )
        processing_signal = Path(processing_signal)
        if processing_signal != expected or plan.get("phase") != "collection":
            raise Stop("Invalid task processing stop signal")
    runs = no_links(plan.get("runs_root", root / "runs"))
    expected_runs = root / ("runs" if plan.get("project_root") else "simulation/runs")
    if plan.get("runs_root") is not None and runs != expected_runs:
        raise Stop("Unrecognized simulator raw-run scope")
    runs.mkdir(parents=True, exist_ok=True)
    budgets = plan["budgets"]
    claim_root = Path(plan.get("project_root", root)) / "Data/operations/claims"
    claim_root.mkdir(parents=True, exist_ok=True)
    # Persistent claims prevent whole-task resubmission; only bounded in-task attempts repeat.
    with (claim_root / (plan["campaign_id"] + ".claim")).open("x") as f:
        json.dump(plan, f)
    existing = set()
    comparison_previous = speed_comparison_scope(plan, cases)
    for p in runs.glob("*/evaluation/scenario.json"):
        try:
            row = read(p)
        except FileNotFoundError:
            if processing_signal:
                continue  # Verified previous groups may retire; persistent seed claims still guard reuse.
            raise
        if comparison_previous is not None and p.parents[1].name in comparison_previous:
            continue  # Same test's explicit matched-seed replicate, never a training run/retry.
        existing.add((row.get("name"), row.get("seed")))
    if any((c["scenario"], c["seed"]) in existing for c in cases):
        raise Stop("Existing raw seed/scenario reuse forbidden")
    local.mkdir(parents=True)
    external.mkdir(parents=True)
    halt, admission, state_lock = threading.Event(), threading.Lock(), threading.Lock()
    memory_halt = threading.Event()
    exporter = threading.Semaphore(plan.get("export_workers", 1))
    queue = iter(cases)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    planned_mission_seconds = sum(duration_for(plan, case) for case in cases)
    records, run_ids = {}, set()
    known_runs = []
    baseline_dirs = {p.name for p in runs.iterdir() if p.is_dir()}
    locked_sources = {root / name: digest for name, digest in plan["source_hashes"].items()}
    project = Path(plan.get("project_root", root))
    locked_sources.update(
        {project / name: digest for name, digest in plan.get("project_source_hashes", {}).items()}
    )
    source_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in locked_sources}
    watch_lock = threading.Lock()
    last_watch = [0.0]
    stop_reason = []
    resources = {"samples": 0, "peak_memory_used_bytes": 0, "min_available_bytes": None}

    def stop_new(reason):
        halt.set()
        with state_lock:
            if not stop_reason:
                stop_reason.append(reason)
                atomic(
                    local / "stop-new-cases.json",
                    {"reason": reason, "policy": "in-flight peers drain; no retries"},
                )

    def sources():
        for path, digest in locked_sources.items():
            if sha(path) != digest:
                raise Stop("Source revision changed: " + str(path))

    def space_check():
        check_output_storage(plan, external, process_helper)
        if (
            min(shutil.disk_usage(root).free, shutil.disk_usage(external).free)
            < budgets["min_free_bytes"]
        ):
            raise Stop("Free-space floor")

    def write_record(case_id, value):
        with state_lock:
            records[case_id] = json.loads(json.dumps(value))
            atomic(
                local / "ledger.json",
                {
                    "status": "DRAINING" if halt.is_set() else "RUNNING",
                    "cases": records,
                    "started_utc": started_utc,
                    "started_monotonic": started,
                    "elapsed_s": time.monotonic() - started,
                    "planned_mission_seconds_total": planned_mission_seconds,
                },
            )

    def observe_limits():
        # Check on every admission, even inside the resource-sampling interval.
        if processing_signal and os.path.lexists(processing_signal):
            stop_new("Processing failure: close new flight admission; drain peers")
        # One-second RAM sampling; byte/space accounting every ten seconds.
        with watch_lock:
            if time.monotonic() - last_watch[0] < 1:
                return
            last_watch[0] = time.monotonic()
            try:
                if any(
                    (n.stat().st_size, n.stat().st_mtime_ns) != v
                    for n, v in source_stats.items()
                ):
                    stop_new("Source metadata changed; drain active cases")
                space_check()
                mem = memory_snapshot()
                if (
                    mem["available_bytes"] < budgets["min_available_ram_bytes"]
                    or budget_memory_used(plan, mem) > budgets["memory_soft_cap_bytes"]
                ):
                    if expanded_memory(plan):
                        memory_halt.set()
                        resources.setdefault("first_memory_violation", dict(mem))
                    stop_new("RAM admission/soft cap reached; drain active cases")
                if resources["samples"] % 10 == 0:
                    new_dirs = [
                        p
                        for p in runs.iterdir()
                        if p.is_dir() and p.name not in baseline_dirs
                    ]
                    resources.update(
                        local_bytes=bytes_under(local) + sum(bytes_under(p) for p in new_dirs),
                        external_bytes=bytes_under(external),
                    )
                resources["samples"] += 1
                resources["peak_memory_used_bytes"] = max(
                    resources["peak_memory_used_bytes"], memory_used(mem)
                )
                resources["min_available_bytes"] = min(
                    resources["min_available_bytes"] or mem["available_bytes"],
                    mem["available_bytes"],
                )
                resources.update(mem)
                usage = dict(
                    resources,
                    elapsed_s=time.monotonic() - started,
                    local_free_bytes=shutil.disk_usage(root).free,
                    external_free_bytes=shutil.disk_usage(external).free,
                )
                resources.update(usage)
                atomic(local / "resource-progress.json", usage)
                if (
                    usage["local_bytes"] >= budgets["local_bytes_cap"]
                    or usage["external_bytes"] >= budgets["external_bytes_cap"]
                ):
                    stop_new("Output admission cap reached; drain active cases")
                if time.monotonic() - started >= budgets["campaign_wall_s"]:
                    stop_new("Campaign admission deadline reached; drain active cases")
            except Exception as exc:
                stop_new(str(exc))

    def worker(lane):
        while True:
            with admission:
                if halt.is_set():
                    return
                try:
                    sources()
                    space_check()
                    observe_limits()
                    if halt.is_set():
                        return
                    if time.monotonic() - started >= budgets["campaign_wall_s"]:
                        raise Stop("Campaign admission deadline; draining active cases")
                    if (
                        bytes_under(local) + sum(bytes_under(p) for p in known_runs)
                        >= budgets["local_bytes_cap"]
                        or bytes_under(external) >= budgets["external_bytes_cap"]
                    ):
                        raise Stop("Output cap; no new cases")
                    case = next(queue, None)
                    if case is None:
                        return
                    claim = hashlib.sha256(
                        json.dumps(
                            [
                                case["scenario"],
                                case["seed"],
                                plan["campaign_id"],
                                plan["speed_comparison"],
                            ]
                            if comparison_previous is not None
                            else [case["scenario"], case["seed"]]
                        ).encode()
                    ).hexdigest()
                    with (claim_root / (claim + ".claim")).open("x") as f:
                        json.dump(case, f)
                except BaseException as exc:
                    stop_new(str(exc))
                    return
            case_root = local / case["case_id"]
            history = []
            for attempt in range(1, limit + 1):
                folder = case_root if attempt == 1 else case_root / f"attempt-{attempt:02d}"
                folder.mkdir()
                record = {
                    "case": case,
                    "lane": lane,
                    "status": "FLYING",
                    "timings_s": {},
                    "attempt": attempt,
                    "attempts_max": limit,
                    "previous_attempts": list(history),
                }
                write_record(case["case_id"], record)
                last_flight_progress = [0.0]
                mission_progress = MissionProgress(
                    root,
                    {p.name for p in runs.iterdir() if p.is_dir()},
                    case,
                    lane,
                    duration_for(plan, case),
                    runs_root=runs,
                )

                def flight_progress(force=False):
                    now = time.monotonic()
                    if (
                        record["status"] == "FLYING"
                        and "flight_started_monotonic" in record
                        and (force or now - last_flight_progress[0] >= 1)
                    ):
                        last_flight_progress[0] = now
                        atomic(
                            folder / "flight-progress.json",
                            {
                                "case_id": case["case_id"],
                                "scenario": case["scenario"],
                                "instance": lane["instance"],
                                "stage": "FLYING",
                                "started_utc": record["flight_started_utc"],
                                "started_monotonic": record["flight_started_monotonic"],
                                "updated_utc": datetime.now(timezone.utc).isoformat(),
                                "updated_monotonic": now,
                                "wall_elapsed_s": now - record["flight_started_monotonic"],
                                "target_mission_seconds": duration_for(plan, case),
                                **mission_progress.sample(monotonic_now=now),
                                "campaign_process_metrics": resources.copy(),
                            },
                        )
                        if plan.get("pretrain_cli_approval") and len(mission_progress.matches) == 1:
                            from pretrain.socket_cleanup import claim

                            run_path = runs / next(iter(mission_progress.matches))
                            claim(run_path, lane["instance"], folder / "socket-claim.json")

                def active_check():
                    observe_limits()
                    flight_progress()
                    if memory_halt.is_set():
                        raise Stop("Memory budget exceeded: stop owned active commands; no retry")
                    if halt.is_set() and plan.get("phase") == "benchmark":
                        raise Stop("Benchmark failure: stop all owned active commands")
                    # Peer/source/admission failure only closes admission. Own stage timeout
                    # still invokes the existing owner-verified cleanup helper.

                try:
                    if halt.is_set():
                        record["status"] = "CLAIMED_NOT_STARTED"
                        write_record(case["case_id"], record)
                        return
                    try:
                        sources()
                        space_check()
                        observe_limits()
                    except Exception as exc:
                        stop_new(str(exc))
                        raise
                    if halt.is_set():
                        raise Stop("Shared admission closed; no retry")
                    args = [
                        str(root / "capstone-sim"),
                        "run",
                        "--scenario",
                        case["scenario"],
                        "--seed",
                        str(case["seed"]),
                        "--duration",
                        str(duration_for(plan, case)),
                        "--instance",
                        str(lane["instance"]),
                        "--capture-interval-ms",
                        str(plan.get("capture_interval_ms", 5)),
                    ]
                    args += plan.get("producer_args", [])
                    if plan.get("speed_comparison"):
                        args += ["--real-time-factor", str(plan["speed_comparison"]["factor"])]
                    if plan["timing"]:
                        args += ["--timing"]
                    if case.get("fault_at_s") is not None:
                        args += ["--fault-at-s", str(case["fault_at_s"])]
                    if case.get("fault_after_mission_s") is not None:
                        args += ["--fault-after-mission-s", str(case["fault_after_mission_s"])]
                    if case.get("normal_variation") is not None:
                        args += [
                            "--normal-variation-json",
                            json.dumps(case["normal_variation"], sort_keys=True),
                        ]
                    if case.get("operation_variation") is not None:
                        args += [
                            "--operation-variation",
                            json.dumps(case["operation_variation"], sort_keys=True),
                        ]
                    tick = time.monotonic()
                    record["flight_started_monotonic"] = tick
                    record["flight_started_utc"] = datetime.now(timezone.utc).isoformat()
                    write_record(case["case_id"], record)
                    flight_progress(force=True)
                    active_check()
                    process_helper.command(
                        args, root, folder / "producer.log", active_check, budgets["flight_wall_s"]
                    )
                    record["timings_s"]["flight_command"] = time.monotonic() - tick
                    record["flight_finished_monotonic"] = time.monotonic()
                    flight_progress(force=True)
                    log = folder / "producer.log"
                    if log.stat().st_size > BLOCK:
                        raise Stop("Unexpectedly large producer result")
                    result = read(log)
                    run = no_links(result["run"])
                    if run.parent != runs or not re.fullmatch("[0-9a-f]{32}", run.name):
                        raise Stop("Producer returned invalid run UUID/path")
                    with admission:
                        if run.name in run_ids:
                            raise Stop("Duplicate returned run UUID")
                        run_ids.add(run.name)
                        known_runs.append(run)
                    record["run_id"] = run.name
                    meta, scenario = read(run / "run.json"), read(run / "evaluation/scenario.json")
                    if (
                        result["status"],
                        meta["status"],
                        meta["run_id"],
                        scenario["name"],
                        scenario["seed"],
                    ) != ("completed", "completed", run.name, case["scenario"], case["seed"]):
                        raise Stop("Run identity/status mismatch")
                    if (
                        meta.get("parent_attempt_id")
                        or scenario.get("continuity") != "independent_reset"
                    ):
                        raise Stop("Unexpected retry/continuation")
                    if scenario.get("requested_duration_s") != duration_for(plan, case):
                        raise Stop("Producer mission duration mismatch")
                    if scenario.get("operation_variation") != case.get(
                        "operation_variation"
                    ) or scenario.get("fault_after_mission_s") != case.get("fault_after_mission_s"):
                        raise Stop("Producer operation conditions/fault schedule mismatch")
                    if (
                        plan.get("speed_comparison")
                        and scenario.get("requested_real_time_factor")
                        != plan["speed_comparison"]["factor"]
                    ):
                        raise Stop("Requested speed not honored by producer")
                    source = read(run / "reference/source_lock.json")
                    verify_run_source_snapshot(plan, source)
                    if meta.get("process_exit_codes") and any(
                        x != 0 for x in meta["process_exit_codes"]
                    ):
                        raise Stop("Owned producer process did not exit cleanly")
                    if (
                        case["role"] not in {"anomaly_dev", "anomaly_test"}
                        and scenario.get("health_eligibility") != "confirmed_healthy"
                    ):
                        raise Stop("Normal health eligibility failed")
                    record["run_id"] = run.name
                    record["producer_wall_duration_s"] = meta.get("wall_duration_s")
                    record["status"] = "WAITING_VERIFY_EXPORT"
                    write_record(case["case_id"], record)
                    wait_started = time.monotonic()
                    with exporter:
                        record["timings_s"]["export_queue"] = time.monotonic() - wait_started
                        active_check()
                        record["status"] = "VERIFYING"
                        record["verification_started_monotonic"] = time.monotonic()
                        write_record(case["case_id"], record)
                        tick = time.monotonic()
                        process_helper.command(
                            [str(root / "capstone-sim"), "verify", "--run", str(run)],
                            root,
                            folder / "verify.log",
                            active_check,
                            budgets["verify_wall_s"],
                        )
                        record["timings_s"]["producer_verify"] = time.monotonic() - tick
                        if plan.get("protocol") in COLLECTION_PROTOCOLS:
                            record["assessment"] = assess_run(
                                plan,
                                case,
                                lane,
                                run,
                                folder,
                                process_helper,
                                active_check,
                                record["timings_s"]["flight_command"],
                            )
                        record["status"] = "ARCHIVING"
                        record["verification_finished_monotonic"] = time.monotonic()
                        record["archive_started_monotonic"] = time.monotonic()
                        write_record(case["case_id"], record)

                        def export_check():
                            active_check()
                            try:
                                space_check()
                            except Exception as exc:
                                stop_new(str(exc))
                                raise

                        record["export"] = archive_run(
                            run, external / (run.name + ".tar.gz"), folder, budgets, export_check
                        )
                        record["archive_finished_monotonic"] = time.monotonic()
                    try:
                        sources()
                    except Stop as exc:
                        record["source_drift"] = str(exc)
                        stop_new(str(exc))
                    record["status"] = (
                        "PRESERVED_SOURCE_DRIFT"
                        if "source_drift" in record
                        else "ARCHIVED_PENDING_QUALITY"
                    )
                    atomic(external / (run.name + ".manifest.json"), record)
                except BaseException as exc:
                    record.update(mission_progress.terminal_evidence())
                    if record.get("producer_run_id"):
                        record.setdefault("run_id", record["producer_run_id"])
                    record.update(
                        observed_run_ids=sorted(mission_progress.matches),
                        failed_stage=record["status"],
                        status="FAILED_PRESERVED",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    if record.get("producer_status") == "failed":
                        record["error"] += (
                            f"; producer {record.get('producer_stage') or 'unknown stage'}: "
                            + record["producer_termination_reason"]
                        )
                    if shared_failure(exc) or (
                        not plan.get("continue_after_case_failure") and attempt >= limit
                    ):
                        stop_new(record["error"])
                    elif not halt.is_set():
                        # Reuse only a demonstrably free lane. Never unlink an unclaimed socket.
                        try:
                            socket_path = Path(f"/tmp/px4-sock-{lane['instance']}")
                            claim_path = folder / "socket-claim.json"
                            if socket_path.exists() and claim_path.exists():
                                from pretrain.common import service_unit
                                from pretrain.socket_cleanup import release

                                task_path = Path(plan["pretrain_cli_approval"]).parent
                                record["socket_cleanup"] = release(
                                    read(claim_path), service_unit(task_path)
                                )
                            check_instance_socket(lane["instance"])
                            for port in lane["ports"]:
                                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                                    sock.bind(("127.0.0.1", port))
                        except Exception as cleanup_exc:
                            record["lane_retired"] = str(cleanup_exc)
                            atomic(folder / "attempt-result.json", record)
                            write_record(case["case_id"], record)
                            return  # Other lanes still consume the finite queue.
                if record["status"] == "FAILED_PRESERVED":
                    history.append(
                        dict(
                            attempt=attempt,
                            run_id=record.get("run_id"),
                            evidence=str(folder / "attempt-result.json"),
                            error=record["error"],
                        )
                    )
                record["attempt_history"] = list(history)
                atomic(folder / "attempt-result.json", record)
                write_record(case["case_id"], record)
                if record["status"] != "FAILED_PRESERVED" or halt.is_set():
                    break

    sources()
    atomic(local / "ledger.json", {"status": "PREPARING", "cases": {}})

    def guarded_worker(lane):
        try:
            worker(lane)
        except BaseException as exc:
            stop_new(f"Worker failure: {type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=plan["concurrency"]) as pool:
        futures = [pool.submit(guarded_worker, lane) for lane in plan["lanes"]]
        try:
            for future in futures:
                future.result()  # finite CLI waits; LLM never needs to poll
        except BaseException as exc:
            stop_new(f"Operator interruption: {type(exc).__name__}: {exc}")
    failed = [
        key for key, value in records.items() if value["status"] != "ARCHIVED_PENDING_QUALITY"
    ]
    not_started = [c["case_id"] for c in cases if c["case_id"] not in records]
    final = {
        "status": "FAILED_DRAINED_STOPPED"
        if halt.is_set()
        else "COLLECTED_WITH_ERRORS_STOPPED"
        if failed or not_started
        else "COLLECTED_PENDING_QUALITY_STOPPED",
        "elapsed_s": time.monotonic() - started,
        "started_utc": started_utc,
        "started_monotonic": started,
        "planned_mission_seconds_total": planned_mission_seconds,
        "resources": resources,
        "data_pool_id": plan.get("data_pool_id"),
        "phase": plan.get("phase", "collection"),
        "training_eligible": False
        if halt.is_set() or failed or not_started or plan.get("phase") in {"benchmark", "endurance"}
        else None,
        "cases": records,
        "not_started": not_started,
        "failed_cases": failed,
        "continue_after_case_failure": plan.get("continue_after_case_failure", False),
        "preserved_new_local_run_paths": [
            str(p) for p in runs.iterdir() if p.is_dir() and p.name not in baseline_dirs
        ],
        "storage": "one gzip tar per run; no persistent extracted consumer copy; local originals retained",
        "stop_reason": stop_reason,
        "ai_compatibility": "Existing R2 raw-directory package cannot consume archives directly; explicit later extraction/admission required.",
    }
    atomic(local / "ledger.json", final)
    try:
        check_output_storage(plan, external, process_helper)
        atomic(external / "manifest.json", final)
    except Exception as exc:
        final["status"] = "FAILED_DRAINED_STOPPED"
        final["external_manifest_error"] = str(exc)
        atomic(local / "ledger.json", final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "manifest", "approval"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan, cases = bound(args.plan, args.manifest, args.approval, args.dry_run)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_NO_PROCESSES_NO_WRITES",
                    "cases": len(cases),
                    "mission_seconds": plan["mission_seconds"],
                    "concurrency": plan["concurrency"],
                    "runtime_disk_mount_ports_claims_checked": False,
                }
            )
        )
        return 0
    result = execute(plan, cases, helper(Path(plan["root"])))
    print(result["status"])
    return 2 if result["status"] == "FAILED_DRAINED_STOPPED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
