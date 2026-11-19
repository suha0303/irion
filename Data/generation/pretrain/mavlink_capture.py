"""Record every received PX4 SITL UDP datagram in arrival order."""

import argparse
import fcntl
import json
import math
import os
import random
import secrets
import signal
import socket
import struct
import subprocess
import sys
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import ExitStack
from multiprocessing import get_context
from pathlib import Path

MAGIC = b"MLUDP001"
PACKET = struct.Struct("!QQ4sHH")
DEFAULT_FLIGHT_WORKERS = 16
TRAINING_FLIGHT_WORKERS = 32
MAX_FLIGHT_WORKERS = (
    64  # MAVLink-only Collection provides instances 0 through 63.
)
FLIGHTS_PER_RECEIVER_PROCESS = 8
GAZEBO_MESSAGE_DISCOVERY_BASE_PORT = 24100
GAZEBO_SERVICE_DISCOVERY_BASE_PORT = 24200
RECEIVE_TIMEOUT = struct.pack("@ll", 0, 100_000)
BATCH_SCHEMA = "drone.mavlink-capture/1"
_receiver_state = None


class BatchOptionsError(ValueError):
    """Invalid command-line settings for a new capture batch."""


def append_packet(receiver, stream):
    try:
        payload, (address, port) = receiver.recvfrom(65535)
    except (BlockingIOError, TimeoutError):
        return False
    header = PACKET.pack(
        time.monotonic_ns(),
        time.time_ns(),
        socket.inet_aton(address),
        port,
        len(payload),
    )
    stream.write(header + payload)
    return True


def capture_flight(receiver, folder, command, discovery_ports):
    from .native_flight import read_result

    result = {}
    try:
        with (
            (folder / "datagrams.bin").open("xb") as capture,
            (folder / "flight.stdout").open("xb") as stdout,
            (folder / "flight.stderr").open("xb") as stderr,
        ):
            capture.write(MAGIC)
            with subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=os.environ
                | {name: str(port) for name, port in discovery_ports.items()}
                | {
                    "CAPSTONE_MAVLINK_RUN_RECEIPT": str(
                        folder / "simulator.json"
                    )
                },
            ) as flight:
                result["flight_started_ns"] = time.monotonic_ns()
                try:
                    while True:
                        exited = flight.poll() is not None
                        # The simulator shuts down PX4 before it exits. Wait
                        # for a full quiet interval after exit to drain UDP.
                        if not append_packet(receiver, capture) and exited:
                            break
                finally:
                    if flight.poll() is None:
                        flight.terminate()
                result["flight_exit"] = flight.wait()
                result["flight_finished_ns"] = time.monotonic_ns()
            capture.flush()
            os.fsync(capture.fileno())
        result.update(read_result(folder / "flight.stdout"))
    except (OSError, ValueError) as error:
        result.update(status="error", error=str(error))
    return result


def initialize_receiver_process(stop, next_number):
    """Leave the batch in charge of forwarding interrupts to each flight."""
    global _receiver_state
    _receiver_state = stop, next_number

    # A process-group TERM can reach the receivers as well as the batch.
    # Keep the receivers alive while their simulators shut down.
    def terminate(_signal, _frame):
        stop.value = 1

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, terminate)


def capture_group(lanes, args, output, first_number, seeds, state=None):
    """Keep one receiver thread per slot in this process."""
    stop, next_number = _receiver_state if state is None else state
    with ThreadPoolExecutor(max_workers=len(lanes)) as flights:
        active = [
            flights.submit(
                capture_slot,
                lane,
                args,
                output,
                stop,
                next_number,
                first_number,
                seeds,
            )
            for lane in lanes
        ]
        for future in as_completed(active):
            future.result()


def normal_training_case(simulation, seed):
    """Sample the duration and operating conditions for one healthy flight."""
    simulator_sources = str(simulation / "sim/src")
    if simulator_sources not in sys.path:
        sys.path.insert(0, simulator_sources)
    from capstone_sim import bounded_wind, flight_program, normal_variation

    rng = random.Random(seed ^ 0xF8EB234F)
    duration = rng.randint(300_000, 900_000) / 1000
    mode = rng.choice(("basic", "mixed", "random"))
    conditions = normal_variation.sample("hover_normal", seed)
    conditions.update(
        soc=0.9,
        temperature=298.15,
        payload=0.0,
        wind_e=0.0,
        wind_n=0.0,
        gust_sigma=0.0,
        wind_profile=bounded_wind.profile(
            rng.choice(("calm", "low", "strong")),
            rng.random() * 360,
            rng.uniform(0.5, 1.5),
        ),
    )
    operation = flight_program.build(
        conditions,
        mode,
        seed,
        duration,
        rng.randrange(len(flight_program.BASIC)),
    )
    return duration, operation


def abnormal_training_case(simulation, seed):
    """Use the healthy flight recipe with one or two fixed wear profiles."""
    duration, operation = normal_training_case(simulation, seed)
    from capstone_sim import battery_aging, propulsion_health

    rng = random.Random(seed ^ 0xAF87C913)
    if rng.choice((False, True)):
        components = ("battery", "propulsion")
    else:
        components = (rng.choice(("battery", "propulsion")),)
    stages = {
        part: rng.choice(catalog.TRAINING_STAGES)
        for part, catalog in (
            ("battery", battery_aging),
            ("propulsion", propulsion_health),
        )
        if part in components
    }
    return duration, operation, stages


def prepare_flight(args, output, simulation, lane, number, seed):
    """Write the launch request for this flight only."""
    from .common import write

    folder = output / f"flight-{number:04d}"
    duration = args.duration
    scenario = args.scenario
    operation = None
    stages = {}
    if args.abnormal_random:
        duration, operation, stages = abnormal_training_case(simulation, seed)
        scenario = (
            "propulsion_loss_existing"
            if "propulsion" in stages
            else "battery_aging_existing"
        )
    elif args.normal_random:
        duration, operation = normal_training_case(simulation, seed)
    # A lane's simulator children and shutdown client share discovery ports.
    discovery_ports = {
        "GZ_DISCOVERY_MSG_PORT": GAZEBO_MESSAGE_DISCOVERY_BASE_PORT + lane,
        "GZ_DISCOVERY_SRV_PORT": GAZEBO_SERVICE_DISCOVERY_BASE_PORT + lane,
    }
    command = [
        str(simulation / "capstone-sim"),
        "run",
        "--configuration",
        "collection",
        "--mavlink-only",
        "--instance",
        str(lane),
        "--scenario",
        scenario,
        "--seed",
        str(seed),
        "--duration",
        str(duration),
        "--real-time-factor",
        str(args.real_time_factor),
    ]
    if operation is not None:
        command.extend(("--operation-variation", json.dumps(operation)))
    for part, stage in stages.items():
        command.extend((f"--mavlink-{part}-stage", str(stage)))
    if args.fault_at_s is not None:
        command.extend(("--fault-at-s", str(args.fault_at_s)))
    folder.mkdir()
    write(
        folder / "request.json",
        dict(
            number=number,
            instance=lane,
            duration_s=duration,
            prepared_monotonic_s=time.monotonic(),
            udp_port=14700 + lane,
            expected_fc_system=1 + lane,
            gazebo_discovery_ports=discovery_ports,
            command=command,
        ),
    )
    return folder, command, discovery_ports


def slot_is_clear(result):
    if result["status"] == "completed":
        return True
    if "run" not in result:
        return False
    from .common import read

    try:
        state = read(Path(result["run"]) / "run.json")
    except (OSError, ValueError):
        return False
    # A failed mission can leave a clean slot. A failed shutdown cannot.
    shutdown = state.get("process_shutdown")
    failures = [
        state.get("primary_failure") or {},
        *state.get("secondary_failures", []),
    ]
    return (
        isinstance(shutdown, list)
        and all(item.get("exit_code") is not None for item in shutdown)
        and not any(item.get("stage") == "cleanup_failure" for item in failures)
    )


def capture_slot(lane, args, output, stop, next_number, first_number, seeds):
    """Rebind this lane after its previous subprocess and recording are closed."""
    from .common import SIMULATION, write

    number = first_number + lane
    last_number = first_number + len(seeds) - 1
    try:
        while not stop.value and number <= last_number:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
                receiver.setsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024
                )
                receiver.bind(("127.0.0.1", 14700 + lane))
                # A kernel timeout keeps the shutdown drain bounded.
                receiver.setsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVTIMEO, RECEIVE_TIMEOUT
                )
                if stop.value:
                    break
                folder, command, ports = prepare_flight(
                    args,
                    output,
                    SIMULATION,
                    lane,
                    number,
                    seeds[number - first_number],
                )
                if stop.value:
                    break
                result = capture_flight(receiver, folder, command, ports)
            if (
                not stop.value
                and next_number.value <= last_number
                and not slot_is_clear(result)
            ):
                result["slot_reuse_blocked"] = True
            write(folder / "flight.json", result)
            if result.get("slot_reuse_blocked"):
                break
            with next_number.get_lock():
                if stop.value or next_number.value > last_number:
                    break
                number = next_number.value
                next_number.value += 1
    except BaseException:
        # Stop sibling slots too when their batch cannot safely continue.
        stop.value = 1
        raise


def random_batch_seed(output, count):
    """Draw a starting seed whose whole range fits between sibling batches."""
    from .common import read

    limit = 2**32
    reserved = [(limit, limit)]
    for path in output.parent.glob("*/batch.json"):
        if path.is_symlink() or path.parent.is_symlink():
            continue
        try:
            batch = read(path)
            if batch.get("schema") != BATCH_SCHEMA:
                continue
            settings = batch["settings"]
            start, size = settings["seed"], settings["count"]
            if (
                type(start) is int
                and type(size) is int
                and 0 <= start < start + size <= limit
            ):
                reserved.append((start, start + size))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue

    free = []
    end = 0
    for start, stop in sorted(reserved):
        if start - end >= count:
            free.append((end, start - end - count + 1))
        end = max(end, stop)
    choices = sum(size for _, size in free)
    if not choices:
        raise BatchOptionsError("No unused uint32 seed range fits this batch")
    offset = secrets.randbelow(choices)
    for start, size in free:
        if offset < size:
            return start + offset
        offset -= size
    raise AssertionError("Random batch seed is outside the available ranges")


def batch_settings(output, supplied):
    """Save the target and seed before launching, or load the same batch."""
    from .common import read, write

    path = output / "batch.json"
    defaults = dict(
        count=16,
        flight_workers=None,
        scenario="hover_normal",
        seed=None,
        duration=None,
        normal_random=False,
        abnormal_random=False,
        real_time_factor=1,
        fault_at_s=None,
    )
    if output.exists():
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "Existing output has no resumable MAVLink batch.json"
            )
        saved = read(path)
        options = (
            saved.get("settings")
            if (isinstance(saved, dict) and saved.get("schema") == BATCH_SCHEMA)
            else None
        )
        if not isinstance(options, dict) or set(options) != defaults.keys():
            raise ValueError("Invalid MAVLink batch settings")
        changed = [
            name for name, value in supplied.items() if options[name] != value
        ]
        if changed:
            raise ValueError(
                "Resume settings differ from batch.json: " + ", ".join(changed)
            )
        return argparse.Namespace(**options)

    options = defaults | supplied
    random_training = options["normal_random"] or options["abnormal_random"]
    if random_training and (
        options["scenario"] != "hover_normal"
        or options["duration"] is not None
        or options["fault_at_s"] is not None
    ):
        raise BatchOptionsError(
            "The random training mode chooses its own scenario and duration per flight"
        )
    count = options["count"]
    if not 1 <= count <= 2**32:
        raise BatchOptionsError(
            "Use a positive count within the uint32 seed range"
        )
    if options["seed"] is None:
        options["seed"] = (
            random_batch_seed(output, count) if random_training else 17
        )
    if not 0 <= options["seed"] <= 2**32 - count:
        raise BatchOptionsError("Batch seed range must fit uint32")
    if options["duration"] is not None and (
        not math.isfinite(options["duration"])
        or not 0 < options["duration"] <= 900
    ):
        raise BatchOptionsError("Duration must be within (0, 900] seconds")
    if options["fault_at_s"] is not None and not math.isfinite(
        options["fault_at_s"]
    ):
        raise BatchOptionsError("Fault time must be finite")
    if options["duration"] is None and not random_training:
        options["duration"] = 90
    if options["flight_workers"] is None:
        options["flight_workers"] = (
            TRAINING_FLIGHT_WORKERS
            if random_training
            else DEFAULT_FLIGHT_WORKERS
        )
    output.mkdir(parents=True, exist_ok=False)
    write(path, dict(schema=BATCH_SCHEMA, settings=options))
    return argparse.Namespace(**options)


def completed_flight(result):
    return (
        result.get("status") == "completed" and result.get("flight_exit") == 0
    )


def read_flights(output, planned_seeds):
    """Per-flight receipts survive interruption even if the summary is stale."""
    from .common import read

    rows = []
    for folder in output.glob("flight-*"):
        number_text = folder.name.removeprefix("flight-")
        if not number_text.isascii() or not number_text.isdecimal():
            continue
        number = int(number_text)
        request = folder / "request.json"
        receipt = folder / "flight.json"
        raw = (folder / "datagrams.bin", folder / "datagrams.bin.zst")
        if (
            folder.name != f"flight-{number:04d}"
            or number < 1
            or not folder.is_dir()
            or folder.is_symlink()
            or any(path.is_symlink() for path in (request, receipt, *raw))
        ):
            raise ValueError(f"Invalid MAVLink capture paths in {folder}")
        result = (
            read(receipt) if receipt.is_file() else {"status": "incomplete"}
        )
        if not isinstance(result, dict) or "status" not in result:
            raise ValueError(f"Invalid MAVLink flight receipt in {folder}")
        if completed_flight(result):
            if not request.is_file() or not any(path.is_file() for path in raw):
                result = dict(
                    result,
                    status="incomplete",
                    reason="Recording or launch request missing",
                )
            else:
                try:
                    command = read(request)["command"]
                    seed = int(command[command.index("--seed") + 1])
                except (
                    ValueError,
                    IndexError,
                    TypeError,
                    KeyError,
                    AttributeError,
                ) as error:
                    raise ValueError(
                        f"Invalid launch seed in {request}"
                    ) from error
                if seed not in planned_seeds:
                    raise ValueError(
                        f"Launch seed outside this batch's range in {request}"
                    )
                result = dict(result, seed=seed)
        rows.append((number, dict(result, folder=folder.name)))
    return sorted(rows)


def completed_seeds(rows):
    return {result["seed"] for _, result in rows if completed_flight(result)}


def collect(args):
    from .ai19 import no_links
    from .common import HOME, ROOT, SIMULATION, write
    from .mavlink_progress import BatchProgress

    output = no_links(args.out)
    context = get_context("spawn")
    stop = context.RawValue("b", 0)

    def interrupt(_signal, _frame):
        stop.value = 1

    with ExitStack() as locks:
        for path in (
            HOME / "pretrain.lock",
            ROOT / "Data/operations/collection.lock",
            SIMULATION / ".cache/setup.lock",
        ):
            handle = locks.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for number in (signal.SIGINT, signal.SIGTERM):
            locks.callback(
                signal.signal, number, signal.signal(number, interrupt)
            )
        options = dict(vars(args))
        options.pop("out")
        args = batch_settings(output, options)
        planned = range(args.seed, args.seed + args.count)
        prior = read_flights(output, planned)
        done = completed_seeds(prior)
        completed = len(done)
        pending = (
            tuple(seed for seed in planned if seed not in done)
            if done
            else planned
        )
        print(
            f"Completed {completed}/{args.count}; remaining {len(pending)}.",
            flush=True,
        )
        if pending and not stop.value:
            first_number = prior[-1][0] + 1 if prior else 1
            slots = min(args.flight_workers, len(pending))
            lanes = range(slots)
            next_number = context.Value("Q", first_number + slots)
            receiver_count = math.ceil(slots / FLIGHTS_PER_RECEIVER_PROCESS)
            failed = sum(
                row["status"] in {"failed", "error"} for _, row in prior
            )
            progress = BatchProgress(
                output,
                SIMULATION / "runs",
                args,
                (completed, failed),
                range(first_number, first_number + len(pending)),
                slots,
                stop,
            )
            with progress.show():
                if receiver_count == 1:
                    capture_group(
                        lanes,
                        args,
                        output,
                        first_number,
                        pending,
                        (stop, next_number),
                    )
                else:
                    with ProcessPoolExecutor(
                        max_workers=receiver_count,
                        mp_context=context,
                        initializer=initialize_receiver_process,
                        initargs=(stop, next_number),
                    ) as receivers:
                        groups = [
                            lanes[index::receiver_count]
                            for index in range(receiver_count)
                        ]
                        active = [
                            receivers.submit(
                                capture_group,
                                group,
                                args,
                                output,
                                first_number,
                                pending,
                            )
                            for group in groups
                        ]
                        try:
                            for future in as_completed(active):
                                future.result()
                        except BaseException:
                            stop.value = 1
                            raise
        rows = read_flights(output, planned)
        receipts = [row for _, row in rows]
        completed = len(completed_seeds(rows))
        write(
            output / "collection.json",
            dict(
                requested=args.count,
                flight_workers=args.flight_workers,
                completed=completed,
                remaining=args.count - completed,
                interrupted=bool(stop.value),
                flights=receipts,
            ),
        )
    print(
        f"Completed {completed}/{args.count}. {output / 'collection.json'}",
        flush=True,
    )
    if stop.value:
        return 130
    return 0 if completed == args.count else 4


def main(argv):
    parser = argparse.ArgumentParser(
        prog="drone-pretrain mavlink",
        description=__doc__,
        epilog=(
            "Press p to finish active flights and pause the batch. "
            "Use n and b to view the next or previous flight slots."
        ),
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New or resumable batch directory",
    )
    parser.add_argument(
        "--count", type=int, help="Target completed flights (default 16)"
    )
    parser.add_argument(
        "--flight-workers",
        type=int,
        choices=range(1, MAX_FLIGHT_WORKERS + 1),
        help="Default 16, or 32 with a random training mode",
    )
    parser.add_argument(
        "--scenario", help="Named scenario (default hover_normal)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="First seed (random, avoiding sibling batches in a training mode if omitted, otherwise 17)",
    )
    parser.add_argument(
        "--duration", type=float, help="Mission seconds (default 90)"
    )
    training = parser.add_mutually_exclusive_group()
    training.add_argument(
        "--normal-random",
        action="store_true",
        help="Healthy X500 with a random 5–15 minute duration, path and wind per flight",
    )
    training.add_argument(
        "--abnormal-random",
        action="store_true",
        help="Random 5–15 minute X500 flight with fixed wear chosen before takeoff",
    )
    parser.add_argument(
        "--real-time-factor",
        type=int,
        choices=(1, 2, 3, 4, 8),
        help="Default 1",
    )
    parser.add_argument("--fault-at-s", type=float)
    args = parser.parse_args(argv)
    try:
        return collect(args)
    except BatchOptionsError as error:
        parser.error(str(error))
    except (OSError, ValueError) as error:
        print(f"MAVLink collection: {error}", file=sys.stderr)
        return 3
