"""Prepare X500 MAVLink splits on independently restartable workers."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from tempfile import mkdtemp

from .ai19 import no_links
from .common import HOME, ROOT, read, write

BATCHES = {
    "normal-training-01": ("training_pool", 1000, "normal"),
    "normal-calibration-01": ("normal_calibration", 150, "normal"),
    "normal-final-test-01": ("normal_test", 250, "normal"),
    "abnormal-final-test-01": ("anomaly_test", 200, "abnormal"),
}
VALIDATION_BATCH = {
    "normal-validation-01": ("normal_validation", 150, "normal")
}
PROTOCOL = "capstone.mavlink-preparation/1"
ATTEMPTS = 3
CODE = (
    "ai/learning/new19/buffered.py",
    "ai/learning/new19/transport.py",
    "ai/learning/new19/data.py",
    "ai/learning/new19/prepare_capture.py",
    "ai/localflight/mavlink_input.py",
    "ai/localflight/prepare.py",
    "ai/localflight/operating.py",
    "ai/localflight/prepared.py",
    "ai/learning/history19/preprocessing.py",
    "ai/src/capstone_ai/mapping.py",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(path):
    path = no_links(path)
    if not path.is_file():
        raise ValueError(f"Required evidence is missing: {path}")
    return path


def capture_stamp(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def inventory(capture_root, batches=BATCHES):
    rows = []
    seeds, IDs = set(), set()
    for batch, (role, count, mode) in batches.items():
        folder = capture_root / batch
        configuration = read(regular(folder / "batch.json"))["settings"]
        collection = read(regular(folder / "collection.json"))
        completed = collection["flights"]
        if (
            (
                configuration["count"],
                collection["requested"],
                collection["completed"],
                len(completed),
            )
            != (count,) * 4
            or configuration[f"{mode}_random"] is not True
            or collection["remaining"] != 0
        ):
            raise ValueError(
                f"Unexpected completed collection or split: {batch}"
            )
        expected = set(
            range(configuration["seed"], configuration["seed"] + count)
        )
        batch_seeds = set()
        for result in completed:
            name = result["folder"]
            if not re.fullmatch(r"flight-[0-9]{4,}", name):
                raise ValueError(f"Invalid flight directory: {name}")
            capture = folder / name
            receipt = read(regular(capture / "flight.json"))
            launch = read(regular(capture / "request.json"))
            seed, run = result["seed"], Path(receipt["run"])
            command = launch["command"]
            system = launch["expected_fc_system"]
            if (
                receipt["status"] != "completed"
                or receipt["flight_exit"] != 0
                or receipt["run"] != result["run"]
                or run.parent != ROOT / "simulation/runs"
                or seed not in expected
                or seed in seeds
                or run.name in IDs
                or int(command[command.index("--seed") + 1]) != seed
                or "--mavlink-only" not in command
                or system != launch["instance"] + 1
            ):
                raise ValueError(f"Conflicting flight evidence: {capture}")
            wire = regular(capture / "datagrams.bin")
            execution = regular(run / "evaluation/execution.json")
            rows.append(
                dict(
                    flight_id=name,
                    batch=batch,
                    role=role,
                    seed=seed,
                    run_id=run.name,
                    wire=str(wire),
                    wire_stat=capture_stamp(wire),
                    source_system=system,
                    execution=str(execution),
                    execution_sha256=digest(execution),
                )
            )
            seeds.add(seed)
            batch_seeds.add(seed)
            IDs.add(run.name)
        if batch_seeds != expected:
            raise ValueError(f"Missing or repeated flight seeds: {batch}")
    return rows


def prepared(folder, item, plan_sha):
    target = folder / item["batch"] / item["flight_id"]
    if not target.exists() and not target.is_symlink():
        return None
    admission = read(regular(target / "admission.json"))
    receipt = regular(target / "receipt.json")
    npz = regular(target / "run.npz")
    if (
        admission["plan_sha256"] != plan_sha
        or any(
            admission[k] != item[k]
            for k in ("batch", "flight_id", "run_id", "seed")
        )
        or admission["npz_sha256"] != digest(npz)
        or admission["receipt_sha256"] != digest(receipt)
        or read(receipt)["raw_sha256"] != admission["raw_sha256"]
    ):
        raise ValueError(
            f"Completed files failed verification and were preserved: {target}"
        )
    return admission


def run_flight(item, output, identity, env, fd, stop):
    target = output / item["batch"] / item["flight_id"]
    work = output / ".attempts"
    for _ in range(ATTEMPTS):
        if stop.is_set():
            return None, False
        attempt = Path(
            mkdtemp(prefix=f"{item['batch']}-{item['flight_id']}-", dir=work)
        )
        candidate = attempt / "candidate"
        job = dict(
            **identity, flight=item, candidate=str(candidate), final=str(target)
        )
        proc = subprocess.run(
            [
                identity["python"],
                "-B",
                "-X",
                "faulthandler",
                "-m",
                "learning.new19.prepare_capture",
            ],
            input=json.dumps(job),
            text=True,
            capture_output=True,
            env=env,
            cwd=ROOT,
            start_new_session=True,
            pass_fds=(fd,),
            check=False,
        )
        if target.exists():
            result = prepared(output, item, identity["plan_sha256"])
            attempt.rmdir()
            return result, False
        write(
            attempt / "failure.json",
            dict(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            ),
        )
        print(
            f"{item['batch']}/{item['flight_id']} · 작업 종료 코드 {proc.returncode}",
            flush=True,
        )
    return None, True


def report(rows, finished):
    hashes = Counter(row["raw_sha256"] for row in finished.values())
    groups = {}
    records = []
    for item in rows:
        role = item["role"]
        group = groups.setdefault(
            role,
            dict(
                total=0,
                prepared=0,
                forecast_usable=0,
                whole_run_usable=0,
                both_usable=0,
                unusable_for_both=0,
                duplicate_source=0,
            ),
        )
        group["total"] += 1
        row = finished.get(item["run_id"])
        if not row:
            continue
        group["prepared"] += 1
        unique = hashes[row["raw_sha256"]] == 1
        key = (
            "forecast_windows"
            if role in {"training_pool", "normal_validation"}
            else "scoreable_frames"
        )
        forecast = unique and row[key] > 0
        whole = unique and row["whole_run_usable"]
        group["forecast_usable"] += forecast
        group["whole_run_usable"] += whole
        group["both_usable"] += forecast and whole
        group["unusable_for_both"] += not (forecast or whole)
        group["duplicate_source"] += not unique
        records.append(
            dict(
                **row,
                role=role,
                forecast_usable=forecast,
                whole_run_eligible=whole,
                path=f"{row['batch']}/{row['flight_id']}/run.npz",
            )
        )
    return dict(
        total=len(rows), prepared=len(finished), splits=groups, flights=records
    )


def execute(args, output, fd):
    source = no_links(args.capture_root)
    chosen_profile = regular(args.profile)
    profile_hash = digest(chosen_profile)
    profile = output / "aircraft.json"
    python = ROOT / "ai/.venv/bin/python"
    if not python.is_file():
        raise ValueError("Local AI Python is missing")
    batches = VALIDATION_BATCH if args.validation_only else BATCHES
    rows = inventory(source, batches)
    plan = dict(
        schema=PROTOCOL,
        capture_root=str(source),
        profile_sha256=profile_hash,
        imu_id=args.imu_id,
        sources={name: digest(ROOT / name) for name in CODE},
        interpreter=dict(path=str(python), sha256=digest(python)),
        flights=rows,
    )
    saved = output / "plan.json"
    if saved.exists() or saved.is_symlink():
        if (
            read(regular(saved)) != plan
            or digest(regular(profile)) != profile_hash
        ):
            raise ValueError(
                "Input or preprocessing version differs from the saved plan"
            )
    else:
        if any(path.name != "aircraft.json" for path in output.iterdir()):
            raise ValueError(f"Output directory has no saved plan: {output}")
        if profile.exists() or profile.is_symlink():
            if digest(regular(profile)) != profile_hash:
                raise ValueError("Existing frozen profile differs")
        else:
            profile.write_bytes(chosen_profile.read_bytes())
        write(saved, plan)
    plan_sha = digest(saved)
    (output / ".attempts").mkdir(exist_ok=True)
    for batch in batches:
        (output / batch).mkdir(exist_ok=True)
        no_links(output / batch)
    no_links(output / ".attempts")
    identity = dict(
        profile=str(profile),
        profile_sha256=profile_hash,
        imu_id=args.imu_id,
        plan_sha256=plan_sha,
        python=str(python),
    )
    env = os.environ | dict(
        PYTHONPATH=os.pathsep.join(
            str(ROOT / path) for path in ("ai", "ai/src", "Data/generation")
        ),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    finished, queue = {}, deque()
    for item in rows:
        completed = prepared(output, item, plan_sha)
        if completed is None:
            queue.append(item)
        else:
            finished[item["run_id"]] = completed
    initial = len(finished)
    stopped = threading.Event()

    def pause(_number, _frame):
        stopped.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, pause)
    started = time.monotonic()
    failed = []
    active = {}

    def progress():
        age = time.monotonic() - started
        n = len(finished) - initial
        remaining = round(age * (len(queue) + len(active)) / n) if n else None
        status = dict(
            target=len(rows),
            prepared=len(finished),
            reused=initial,
            active=len(active),
            queued=len(queue),
            failed=len(failed),
            eta_seconds=remaining,
            stopping=stopped.is_set(),
        )
        write(output / "progress.json", status)
        eta = (
            f"약 {remaining / 60:.1f}분" if remaining is not None else "계산 중"
        )
        print(
            f"전처리 {len(finished)}/{len(rows)} · 재사용 {initial} · 진행 {len(active)} · 실패 {len(failed)} · 남은 시간 {eta}",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        def schedule():
            while queue and len(active) < args.workers and not stopped.is_set():
                item = queue.popleft()
                future = executor.submit(
                    run_flight, item, output, identity, env, fd, stopped
                )
                active[future] = item

        schedule()
        progress()
        last_update = time.monotonic()
        while active:
            done, _ = wait(active, timeout=3, return_when=FIRST_COMPLETED)
            for future in done:
                item = active.pop(future)
                item_result, exhausted = future.result()
                if item_result:
                    finished[item["run_id"]] = item_result
                elif exhausted:
                    failed.append(item["run_id"])
            schedule()
            if time.monotonic() - last_update >= 12 or not active:
                progress()
                last_update = time.monotonic()
    summary = report(rows, finished)
    summary.update(
        plan_sha256=plan_sha, failed=failed, interrupted=stopped.is_set()
    )
    write(output / "summary.json", summary)
    print(
        json.dumps(summary["splits"], indent=2, ensure_ascii=False), flush=True
    )
    return 130 if stopped.is_set() else 0 if len(finished) == len(rows) else 4


def main(argv):
    parser = argparse.ArgumentParser(
        prog="drone-pretrain mavlink-prepare", description=__doc__
    )
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Prepare normal-validation-01 in a separate output directory",
    )
    parser.add_argument("--imu-id", type=int, choices=range(256), default=0)
    parser.add_argument("--workers", type=int, choices=range(1, 33), default=4)
    args = parser.parse_args(argv)
    try:
        output = no_links(args.out)
        source = no_links(args.capture_root)
        if (
            output == source
            or output.is_relative_to(source)
            or source.is_relative_to(output)
        ):
            raise ValueError(
                "Prepared output and original MAVLink capture must have separate paths"
            )
        with (HOME / "mavlink-prepare.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            output.mkdir(parents=True, exist_ok=True)
            return execute(args, output, lock.fileno())
    except (OSError, ValueError, KeyError) as error:
        print(f"MAVLink preparation: {error}", flush=True)
        return 3
