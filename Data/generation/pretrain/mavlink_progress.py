"""Display MAVLink batch progress and accept a pause key in the collector."""

import json
import math
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from threading import Event, Thread
from unicodedata import east_asian_width

from .common import read

TAKEOFF_SECONDS = 22
LANDING_SECONDS = 22
LANDING_SPEED_MPS = 0.3
PHASES = {
    "created": "실행 준비",
    "preparing": "Gazebo 시작",
    "simulator_ready": "PX4 연결 대기",
    "px4_ready": "비행 설정 확인",
    "preflight_ready": "시동 준비",
    "armed": "이륙 중",
    "airborne": "이륙 완료",
    "mission": "비행 중",
    "landing": "착륙 중",
    "disarmed": "착륙 완료",
    "finalizing": "기록 저장 중",
    "completed": "기록 닫는 중",
    "failed": "실패 기록 저장 중",
}
MANEUVERS = {
    "hover": "호버",
    "line": "직선",
    "ellipse": "타원",
    "figure8": "8자",
    "lissajous": "리사주",
    "altitude": "고도 변화",
    "yaw": "요 회전",
    "random": "연속 무작위",
}
WINDS = {"calm": "무풍", "low": "약풍", "strong": "강풍"}


def read_snapshot(path):
    """Progress may be absent while a flight is starting or shutting down."""
    if path.is_symlink():
        return {}
    try:
        value = read(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


@contextmanager
def pause_keyboard():
    """Read individual keys from a foreground terminal and restore its mode."""
    fd = None
    try:
        if sys.stdin.isatty():
            candidate = sys.stdin.fileno()
            if os.tcgetpgrp(candidate) == os.getpgrp():
                original = termios.tcgetattr(candidate)
                tty.setcbreak(candidate, termios.TCSANOW)
                fd = candidate
    except (OSError, ValueError, termios.error):
        pass
    try:
        yield fd
    finally:
        if fd is not None:
            with suppress(termios.error):
                termios.tcsetattr(fd, termios.TCSANOW, original)


def estimate_batch(active, queued, slots, nominal):
    """Estimate when current flights and the queued flights will all finish."""
    latest = max(active, default=0)
    if not queued:
        return latest
    if not slots:
        return None
    available = [*active, *([0] * (slots - len(active)))]
    lo, hi = 0, math.ceil(latest + math.ceil(queued / slots) * nominal)
    # At time t, a slot can finish floor((t - available_at) / nominal) jobs.
    while lo < hi:
        mid = (lo + hi) // 2
        throughput = sum(
            max(0, math.floor((mid - ready) / nominal)) for ready in available
        )
        if throughput < queued:
            lo = mid + 1
        else:
            hi = mid
    return max(latest, lo)


def format_eta(seconds):
    if seconds is None:
        return "계산 중"
    hours, remainder = divmod(math.ceil(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_mission(seconds):
    minutes, seconds = divmod(int(max(0, seconds)), 60)
    return f"{minutes:02d}:{seconds:02d}"


def command_option(command, name, default=""):
    return command[command.index(name) + 1] if name in command else default


def flight_conditions(request, elapsed):
    command = request.get("command", [])
    operation = json.loads(
        command_option(command, "--operation-variation", "{}")
    )
    path = operation.get("path", {})
    kind = path.get("kind")
    if kind == "mixed":
        program = path["program"]
        index = sum(elapsed >= item["start_s"] for item in program) - 1
        segment = program[max(0, index)]["path"]["kind"]
        motion = f"복합 {max(0, index) + 1}/{len(program)} {MANEUVERS.get(segment, segment)}"
    else:
        motion = MANEUVERS.get(kind, command_option(command, "--scenario", "–"))
    wind = operation.get("wind_profile", {})
    environment = wind.get("environment")
    weather = WINDS.get(environment, "–")
    if environment in {"low", "strong"}:
        weather += f" {wind['direction_deg']:.0f}° ×{wind['variation']:.2f}"
    battery = command_option(command, "--mavlink-battery-stage", "0")
    propulsion = command_option(command, "--mavlink-propulsion-stage", "0")
    wear = f"배터리 {battery}/추진계 {propulsion}"
    return f"{motion} · {weather} · {wear}"


def fit_terminal(line, columns):
    """Prevent wrapped rows when Korean labels occupy two terminal cells."""
    used = 0
    for index, char in enumerate(line):
        used += 2 if east_asian_width(char) in {"W", "F"} else 1
        if used >= columns:
            return line[:index]
    return line


@dataclass(frozen=True)
class ActiveFlight:
    folder: Path
    slot: int
    description: str
    conditions: str
    eta: float
    rate: float


class BatchProgress:
    """Summarize published receipts independently of UDP capture workers."""

    def __init__(
        self, output, simulator_runs, args, counts, numbers, slots, stop
    ):
        self.output = output
        self.simulator_runs = simulator_runs
        self.target = args.count
        self.nominal = args.duration or 600
        self.requested_rate = args.real_time_factor
        self.completed, self.failed = counts
        self.numbers = numbers
        self.slots = slots
        self.stop = stop
        self.finished = set()
        self.clocks = {}
        self.retired = 0
        self.last_failure = ""

    def flight_progress(self, folder, request, now):
        run = Path(read_snapshot(folder / "simulator.json").get("run", ""))
        state = {}
        if run.parent == self.simulator_runs and not run.is_symlink():
            state = read_snapshot(run / "run.json")
        duration = request["duration_s"]
        progress = state.get("progress", {})
        stamp = progress.get("host_monotonic_s")
        boot = progress.get("px4_boot_s")
        past_stamp, past_boot, measured_rate = self.clocks.get(
            folder, (None, None, None)
        )
        if stamp is not None and boot is not None:
            if (
                past_stamp is not None
                and stamp > past_stamp
                and boot > past_boot
            ):
                measured_rate = max(
                    0.05, (boot - past_boot) / (stamp - past_stamp)
                )
            self.clocks[folder] = stamp, boot, measured_rate
        rate = measured_rate or self.requested_rate
        phase = state.get("status")
        elapsed = progress.get("sim_elapsed_s", 0)
        age = now - request.get("prepared_monotonic_s", now)
        if phase in {"disarmed", "finalizing", "completed", "failed"}:
            eta = 3
        elif phase == "landing":
            height = progress.get("height_m")
            descent = (
                LANDING_SECONDS
                if height is None
                else max(0, height) / LANDING_SPEED_MPS + 4
            )
            eta = max(2, descent / rate)
        elif phase in {"airborne", "mission"}:
            eta = (max(0, duration - elapsed) + LANDING_SECONDS) / rate
        else:
            eta = (
                max(0, TAKEOFF_SECONDS - age)
                + (duration + LANDING_SECONDS) / rate
            )

        status = PHASES.get(phase, "실행 대기")
        if (
            phase in {"armed", "airborne", "mission", "landing"}
            and stamp is not None
            and now - stamp >= 5
        ):
            status += f" (갱신 지연 {int(now - stamp)}초)"
        rate_text = (
            f"{measured_rate:.2f}×" if measured_rate is not None else "—"
        )
        slot = request.get("instance", 0)
        seed = command_option(request.get("command", []), "--seed", "–")
        description = (
            f"{slot:02d}  #{folder.name.removeprefix('flight-')}  {seed:>10}  "
            f"{format_mission(min(elapsed, duration))}/{format_mission(duration)}  "
            f"{rate_text:>5}  {status}"
        )
        return ActiveFlight(
            folder,
            slot,
            description,
            flight_conditions(request, elapsed),
            eta,
            rate,
        )

    def snapshot(self):
        by_slot = {}
        now = time.monotonic()
        for folder in self.output.glob("flight-*"):
            number_text = folder.name.removeprefix("flight-")
            if not number_text.isascii() or not number_text.isdecimal():
                continue
            number = int(number_text)
            if number not in self.numbers or folder in self.finished:
                continue
            if folder.is_symlink():
                continue
            result = read_snapshot(folder / "flight.json")
            if result:
                self.finished.add(folder)
                self.clocks.pop(folder, None)
                if (
                    result.get("status") == "completed"
                    and result.get("flight_exit") == 0
                ):
                    self.completed += 1
                else:
                    self.failed += 1
                    reason = (
                        result.get("reason")
                        or result.get("error")
                        or "종료 상태 확인 필요"
                    )
                    self.last_failure = f"최근 실패 #{number:04d} · {' '.join(str(reason).split())}"
                self.retired += bool(result.get("slot_reuse_blocked"))
            elif (folder / "datagrams.bin").is_file():
                request = read_snapshot(folder / "request.json")
                if request:
                    flight = self.flight_progress(folder, request, now)
                    previous = by_slot.get(flight.slot)
                    if previous is None or previous[0] < number:
                        by_slot[flight.slot] = (number, flight)
        active = [by_slot[slot][1] for slot in sorted(by_slot)]
        queued = max(0, len(self.numbers) - len(self.finished) - len(active))
        nominal = (self.nominal + TAKEOFF_SECONDS + LANDING_SECONDS) / (
            median(flight.rate for flight in active)
            if active
            else self.requested_rate
        )
        return active, queued, nominal

    def display(self, keyboard, done):
        stream = sys.stderr
        screen = (
            keyboard is not None
            and stream.isatty()
            and os.getenv("TERM") != "dumb"
        )
        if screen:
            with suppress(OSError):
                columns, height = os.get_terminal_size(stream.fileno())
                screen = (columns or 80) >= 68 and (height or 24) >= 10
        if screen:
            print(
                "\x1b[?1049h\x1b[?25l\x1b[2J", file=stream, end="", flush=True
            )
        try:
            self.refresh(stream, screen, keyboard, done)
        finally:
            if screen:
                print("\x1b[?25h\x1b[?1049l", file=stream, end="", flush=True)
        count = f"완주 {self.completed}/{self.target} · 실패 {self.failed} · 진행 중 0"
        print(f"[수집 종료] {count}", file=stream, flush=True)
        if self.last_failure:
            print(self.last_failure, file=stream, flush=True)

    def refresh(self, stream, screen, keyboard, done):
        interval = 1 if screen else 5
        deadline = 0
        was_paused = False
        page = 0
        logged = set()
        last_failure = ""
        while True:
            paused = bool(self.stop.value)
            finished = done.is_set()
            if finished or time.monotonic() >= deadline or paused != was_paused:
                active, queued, nominal = self.snapshot()
                if finished:
                    return
                counts = f"완주 {self.completed}/{self.target} · 실패 {self.failed} · 진행 중 {len(active)}"
                etas = [flight.eta for flight in active]
                if paused:
                    status = f"[중단 대기] {counts}"
                    detail = f"마무리 예상 {format_eta(max(etas, default=0))}"
                else:
                    eta = estimate_batch(
                        etas, queued, self.slots - self.retired, nominal
                    )
                    status = f"[수집 중] {counts} · 대기 {queued}"
                    detail = f"예상 남은 {format_eta(eta)}"
                    if keyboard is not None:
                        detail += " · p 일시정지"
                if screen:
                    page = self.draw_page(stream, status, detail, active, page)
                else:
                    print(f"{status} · {detail}", file=stream, flush=True)
                    for flight in active:
                        if flight.folder not in logged:
                            print(
                                f"{flight.description} · {flight.conditions}",
                                file=stream,
                            )
                            logged.add(flight.folder)
                    if self.last_failure != last_failure:
                        print(self.last_failure, file=stream)
                        last_failure = self.last_failure
                    stream.flush()
                deadline = time.monotonic() + interval
                was_paused = paused
            if keyboard is None:
                done.wait(0.2)
            else:
                try:
                    if not select.select([keyboard], [], [], 0.2)[0]:
                        continue
                    keys = os.read(keyboard, 1024)
                    if keys.lower() == b"p":
                        self.stop.value = 1
                    elif screen and keys.lower() in {b"n", b"b"}:
                        page += 1 if keys.lower() == b"n" else -1
                        deadline = 0
                    if not keys:
                        keyboard = None
                except OSError:
                    keyboard = None

    def draw_page(self, stream, status, detail, active, page):
        try:
            columns, height = os.get_terminal_size(stream.fileno())
        except OSError:
            columns, height = 80, 24
        columns, height = columns or 80, height or 24
        wide = columns >= 117
        rows = max(1, (height - 6) // (1 if wide else 2))
        pages = math.ceil(self.slots / rows)
        page = min(max(page, 0), pages - 1)
        start = page * rows
        lines = [status, detail]
        if pages > 1:
            lines.append(f"슬롯 화면 {page + 1}/{pages} · n 다음 · b 이전")
        lines.append(
            "슬롯  비행       시드       임무 경과/목표   배속    상태"
        )
        for flight in active:
            if start <= flight.slot < start + rows:
                if wide:
                    lines.append(f"{flight.description}  {flight.conditions}")
                else:
                    lines.extend(
                        (flight.description, f"    {flight.conditions}")
                    )
        if self.last_failure:
            lines.append(self.last_failure)
        # Shorter replacement text must erase the old cells on each row.
        body = "\r\n".join(
            f"\x1b[2K{fit_terminal(line, columns)}" for line in lines[:height]
        )
        print(f"\x1b[H{body}\x1b[J", file=stream, end="", flush=True)
        return page

    @contextmanager
    def show(self):
        done = Event()
        with pause_keyboard() as keyboard:
            monitor = Thread(target=self.display, args=(keyboard, done))
            monitor.start()
            try:
                yield
            finally:
                done.set()
                monitor.join()
