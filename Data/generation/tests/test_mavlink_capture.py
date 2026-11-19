"""Check UDP capture boundaries without launching a simulator."""

import json
import signal
from io import StringIO
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import MagicMock

from pretrain import common, mavlink_capture, mavlink_progress, native_flight
from pretrain.common import SIMULATION


def test_batch_eta_accounts_for_free_slots_and_overlapping_flights():
    estimate = mavlink_progress.estimate_batch
    assert estimate([20, 55], queued=4, slots=3, nominal=60) == 120
    assert estimate([20, 55], queued=0, slots=2, nominal=60) == 55
    assert estimate([], queued=1, slots=0, nominal=60) is None


def test_dashboard_erases_shortened_rows_and_removed_slots(tmp_path):
    progress = mavlink_progress.BatchProgress(
        tmp_path,
        tmp_path,
        SimpleNamespace(count=2, duration=90, real_time_factor=1),
        (0, 0),
        range(1, 3),
        2,
        SimpleNamespace(value=0),
    )
    output = StringIO()
    flights = [
        mavlink_progress.ActiveFlight(
            tmp_path / "flight-0001",
            0,
            "#0001 비행 중 (갱신 지연 8초)",
            "강풍",
            10,
            1,
        ),
        mavlink_progress.ActiveFlight(
            tmp_path / "flight-0002", 1, "#0002 비행 중", "약풍", 20, 1
        ),
    ]
    progress.draw_page(output, "[수집 중]", "예상 남은 00:00:20", flights, 0)
    progress.draw_page(
        output, "[중단 대기]", "마무리 예상 00:00:02", flights[1:], 0
    )
    first, second = output.getvalue().split("\x1b[H")[1:]
    assert "\x1b[2K#0001 비행 중 (갱신 지연 8초)" in first
    assert "\x1b[2K#0002 비행 중" in second
    assert "#0001" not in second
    assert all(row.startswith("\x1b[2K") for row in second.split("\r\n"))
    assert second.endswith("\x1b[J")


def test_progress_counts_only_current_attempts_and_uses_measured_speed(
    tmp_path,
):
    output = tmp_path / "capture"
    runs = tmp_path / "runs"
    running = output / "flight-0003"
    run = runs / "sim-id"
    done = output / "flight-0004"
    common.write(running / "simulator.json", {"run": str(run)})
    common.write(running / "request.json", {"duration_s": 90})
    (running / "datagrams.bin").write_bytes(mavlink_capture.MAGIC)
    common.write(
        output / "flight-0001/flight.json",
        {"status": "failed", "flight_exit": 1},
    )
    common.write(
        done / "flight.json", {"status": "completed", "flight_exit": 0}
    )
    progress = mavlink_progress.BatchProgress(
        output,
        runs,
        SimpleNamespace(count=4, duration=90, real_time_factor=1),
        (1, 1),
        range(3, 6),
        2,
        SimpleNamespace(value=0),
    )
    for stamp, elapsed in [(10, 30), (12, 31)]:
        common.write(
            run / "run.json",
            {
                "status": "mission",
                "progress": {
                    "host_monotonic_s": stamp,
                    "px4_boot_s": elapsed + 10,
                    "sim_elapsed_s": elapsed,
                },
            },
        )
        active, queued, nominal = progress.snapshot()
    assert (progress.completed, progress.failed, queued) == (2, 1, 1)
    assert [flight.eta for flight in active] == [162]
    assert "0.50×" in active[0].description
    assert "00:31/01:30" in active[0].description
    assert nominal == 268


def test_progress_displays_the_active_mixed_segment_wind_and_wear():
    operation = {
        "path": {
            "kind": "mixed",
            "program": [
                {"start_s": 0, "path": {"kind": "ellipse"}},
                {"start_s": 90, "path": {"kind": "yaw"}},
            ],
        },
        "wind_profile": {
            "environment": "strong",
            "direction_deg": 289,
            "variation": 1.25,
        },
    }
    launch = {
        "command": [
            "--operation-variation",
            json.dumps(operation),
            "--mavlink-battery-stage",
            "4",
            "--mavlink-propulsion-stage",
            "12",
        ],
    }
    assert mavlink_progress.flight_conditions(launch, 95) == (
        "복합 2/2 요 회전 · 강풍 289° ×1.25 · 배터리 4/추진계 12"
    )


def test_random_seed_reserves_the_whole_range_and_resumes(
    tmp_path, monkeypatch
):
    for name, seed, count in [("first", 0, 10), ("second", 15, 20)]:
        common.write(
            tmp_path / name / "batch.json",
            {
                "schema": mavlink_capture.BATCH_SCHEMA,
                "settings": {"seed": seed, "count": count},
            },
        )
    # Among all valid starts there is exactly one way to fill the five-seed gap.
    draw = MagicMock(side_effect=[0, 1])
    monkeypatch.setattr(mavlink_capture.secrets, "randbelow", draw)
    out = tmp_path / "new"
    initial = mavlink_capture.batch_settings(
        out, {"normal_random": True, "count": 5}
    )
    assert initial.seed == 10
    assert mavlink_capture.batch_settings(out, {}).seed == 10
    next_batch = mavlink_capture.batch_settings(
        tmp_path / "next", {"abnormal_random": True, "count": 5}
    )
    assert next_batch.seed == 36
    assert draw.call_count == 2


def test_normal_case_randomizes_missions_and_wind_but_keeps_initial_conditions():
    seeds = range(202611196, 202611228)
    cases = [
        mavlink_capture.normal_training_case(SIMULATION, seed) for seed in seeds
    ]
    assert cases == [
        mavlink_capture.normal_training_case(SIMULATION, seed) for seed in seeds
    ]
    assert len({duration for duration, _ in cases}) == len(seeds)
    kinds = set()
    winds = set()
    for duration, spec in cases:
        assert 300 <= duration <= 900
        assert (spec["soc"], spec["temperature"], spec["payload"]) == (
            0.9,
            298.15,
            0.0,
        )
        wind = spec["wind_profile"]
        winds.add(wind["environment"])
        assert (
            0 <= wind["direction_deg"] < 360 and 0.5 <= wind["variation"] <= 1.5
        )
        path = spec["path"]
        kind = path["kind"]
        kinds.add(kind if kind in {"mixed", "random"} else "basic")
        if kind in {"mixed", "random"}:
            assert path["duration_s"] == duration
        if kind == "mixed":
            assert len(
                {part["path"]["kind"] for part in path["program"]}
            ) == len(path["program"])
    assert kinds == {"basic", "mixed", "random"}
    assert winds == {"calm", "low", "strong"}


def setup_mock_collection(tmp_path, monkeypatch):
    for path in ("Data/operations", "simulation/.cache", "pretrain"):
        (tmp_path / path).mkdir(parents=True)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "HOME", tmp_path / "pretrain")
    monkeypatch.setattr(common, "SIMULATION", tmp_path / "simulation")
    monkeypatch.setattr(
        mavlink_capture.socket, "socket", MagicMock(side_effect=mock_socket)
    )


def mock_socket(*args):
    receiver = MagicMock()
    receiver.__enter__.return_value = receiver
    return receiver


def complete_recording(folder):
    (folder / "datagrams.bin").write_bytes(
        mavlink_capture.MAGIC + folder.name.encode()
    )
    return {"flight_exit": 0, "status": "completed"}


def test_normal_collection_samples_only_flights_being_started(
    tmp_path, monkeypatch
):
    setup_mock_collection(tmp_path, monkeypatch)
    seed_draw = MagicMock(return_value=59344)
    monkeypatch.setattr(mavlink_capture.secrets, "randbelow", seed_draw)
    sample = MagicMock(
        wraps=lambda _, seed: mavlink_capture_case(SIMULATION, seed)
    )
    mavlink_capture_case = mavlink_capture.normal_training_case
    monkeypatch.setattr(mavlink_capture, "normal_training_case", sample)

    started = Barrier(2, timeout=5)

    def stop_after_start(receiver, folder, command, ports):
        started.wait()
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        return complete_recording(folder)

    monkeypatch.setattr(mavlink_capture, "capture_flight", stop_after_start)
    output = tmp_path / "captures"
    assert (
        mavlink_capture.main(
            [
                "--normal-random",
                "--out",
                str(output),
                "--count",
                "10",
                "--flight-workers",
                "2",
            ]
        )
        == 130
    )
    assert sorted(call.args[1] for call in sample.call_args_list) == [
        59344,
        59345,
    ]
    assert not (output / "normal-plan.json").exists()
    for seed, folder in zip(
        (59344, 59345), sorted(output.glob("flight-*")), strict=True
    ):
        request = json.loads((folder / "request.json").read_text())["command"]
        duration, operation = mavlink_capture_case(SIMULATION, seed)
        assert float(request[request.index("--duration") + 1]) == duration
        assert (
            json.loads(request[request.index("--operation-variation") + 1])
            == operation
        )
    state = common.read(output / "collection.json")
    assert (state["interrupted"], state["completed"], state["remaining"]) == (
        True,
        2,
        8,
    )
    saved = {
        folder: (folder / "datagrams.bin").read_bytes()
        for folder in output.glob("flight-*")
    }

    def resume_recording(receiver, folder, command, ports):
        return complete_recording(folder)

    monkeypatch.setattr(mavlink_capture, "capture_flight", resume_recording)
    assert mavlink_capture.main(["--out", str(output)]) == 0
    assert sorted(call.args[1] for call in sample.call_args_list) == list(
        range(59344, 59354)
    )
    state = common.read(output / "collection.json")
    assert (state["interrupted"], state["completed"], state["remaining"]) == (
        False,
        10,
        0,
    )
    assert all(
        (folder / "datagrams.bin").read_bytes() == raw
        for folder, raw in saved.items()
    )
    seed_draw.assert_called_once()

    capture = MagicMock()
    monkeypatch.setattr(mavlink_capture, "capture_flight", capture)
    assert mavlink_capture.main(["--out", str(output)]) == 0
    assert mavlink_capture.main(["--out", str(output), "--seed", "1"]) == 3
    capture.assert_not_called()


def test_freed_slot_restarts_before_the_other_flight_finishes(
    tmp_path, monkeypatch
):
    setup_mock_collection(tmp_path, monkeypatch)
    second_started = Event()
    replacement_started = Event()
    receivers = {}

    def capture(receiver, folder, command, ports):
        receivers[folder.name] = receiver
        if folder.name == "flight-0001":
            assert second_started.wait(5)
        elif folder.name == "flight-0002":
            second_started.set()
            assert replacement_started.wait(5)
        else:
            first = receivers["flight-0001"]
            assert first.__exit__.called
            assert not receivers["flight-0002"].__exit__.called
            assert receiver.bind.call_args == first.bind.call_args
            replacement_started.set()
        return complete_recording(folder)

    monkeypatch.setattr(mavlink_capture, "capture_flight", capture)
    output = tmp_path / "captures"
    assert (
        mavlink_capture.main(
            [
                "--out",
                str(output),
                "--count",
                "3",
                "--flight-workers",
                "2",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "collection.json").read_text())
    assert [row["folder"] for row in manifest["flights"]] == [
        "flight-0001",
        "flight-0002",
        "flight-0003",
    ]
    assert all(receiver.__exit__.called for receiver in receivers.values())


def test_slot_with_unconfirmed_shutdown_is_not_reused(tmp_path, monkeypatch):
    setup_mock_collection(tmp_path, monkeypatch)
    run = tmp_path / "previous-run"
    run.mkdir()
    common.write(
        run / "run.json",
        {
            "process_shutdown": [{"exit_code": None}],
            "primary_failure": {"stage": "cleanup_failure"},
        },
    )
    capture = MagicMock(
        return_value={
            "status": "failed",
            "flight_exit": 4,
            "run": str(run),
        }
    )
    monkeypatch.setattr(mavlink_capture, "capture_flight", capture)
    output = tmp_path / "captures"
    assert (
        mavlink_capture.main(
            [
                "--out",
                str(output),
                "--count",
                "2",
                "--flight-workers",
                "1",
            ]
        )
        == 4
    )
    assert capture.call_count == 1
    manifest = json.loads((output / "collection.json").read_text())
    assert manifest["flights"][0]["slot_reuse_blocked"] is True
    assert not (output / "flight-0002").exists()


def test_resume_uses_receipts_even_with_stale_summary_and_partial_flight(
    tmp_path, monkeypatch
):
    setup_mock_collection(tmp_path, monkeypatch)
    output = tmp_path / "captures"
    mavlink_capture.batch_settings(
        output, dict(count=3, flight_workers=1, seed=210)
    )
    saved = output / "flight-0001"
    saved.mkdir()
    common.write(saved / "request.json", {"command": ["--seed", "210"]})
    common.write(saved / "flight.json", complete_recording(saved))
    unfinished = output / "flight-0002"
    unfinished.mkdir()
    original_partial = mavlink_capture.MAGIC + b"partial datagram"
    (unfinished / "datagrams.bin").write_bytes(original_partial)
    common.write(output / "collection.json", {"completed": 0, "flights": []})
    launched = []

    def capture(receiver, folder, command, ports):
        launched.append(
            (folder.name, int(command[command.index("--seed") + 1]))
        )
        return complete_recording(folder)

    monkeypatch.setattr(mavlink_capture, "capture_flight", capture)
    assert mavlink_capture.main(["--out", str(output)]) == 0
    assert launched == [("flight-0003", 211), ("flight-0004", 212)]
    rows = common.read(output / "collection.json")
    assert rows["completed"] == 3 and rows["remaining"] == 0
    assert rows["flights"][1]["status"] == "incomplete"
    assert (unfinished / "datagrams.bin").read_bytes() == original_partial


def test_abnormal_cases_reuse_normal_flights_and_only_choose_nonzero_stages():
    stages_by_part = {"battery": set(), "propulsion": set()}
    shapes = set()
    for seed in range(16879, 16943):
        duration, operation, stages = mavlink_capture.abnormal_training_case(
            SIMULATION, seed
        )
        assert (duration, operation) == mavlink_capture.normal_training_case(
            SIMULATION, seed
        )
        shapes.add(tuple(stages))
        for part, stage in stages.items():
            stages_by_part[part].add(stage)
    assert shapes == {("battery",), ("propulsion",), ("battery", "propulsion")}
    assert all(stages == {4, 8, 12, 16} for stages in stages_by_part.values())


def test_fixed_abnormal_profiles_use_measured_battery_and_bounded_propulsion():
    mavlink_capture.normal_training_case(SIMULATION, 16879)
    from capstone_sim import battery_aging, collection_config, propulsion_health

    pack = collection_config.battery()
    battery_catalog = battery_aging.load_catalog()
    propulsion_catalog = propulsion_health.load_catalog()
    nominal = propulsion_catalog["states"]["normal"]
    maximum = propulsion_catalog["states"]["abnormal"]
    for stage in (4, 8, 12, 16):
        battery = battery_aging.resolve(
            "battery_aging_existing",
            None,
            None,
            pack,
            mavlink_fixed_stage=stage,
        )
        measured = battery_aging.pack_snapshot(
            battery_catalog, f"rpt_{stage:02d}", pack
        )
        assert battery["initial"] == battery["target"] == measured
        assert battery["fixed_for_flight"] and battery["transition_s"] == 0
        prop = propulsion_health.resolve(
            "propulsion_loss_existing", mavlink_fixed_stage=stage
        )
        expected = (
            maximum
            if stage == 16
            else {
                key: nominal[key] + stage / 16 * (maximum[key] - nominal[key])
                for key in nominal
            }
        )
        assert prop["initial"] == prop["target"] == expected
        assert prop["fixed_for_flight"] and prop["transition_s"] == 0


def test_capture_preserves_full_datagrams_before_and_after_simulator_exit(
    tmp_path, monkeypatch
):
    first = b"\x00before startup\xff"
    repeated = b"\x00\xfeunexpected payload"
    last = b"\x80" * 65507
    expected = [first, repeated, repeated, b"", last]
    source = ("127.0.0.1", 18570)

    receiver = MagicMock()
    receiver.recvfrom.side_effect = [
        (first, source),
        BlockingIOError(),
        (repeated, source),
        (repeated, source),
        (b"", source),
        (last, source),
        BlockingIOError(),
    ]
    process = MagicMock()
    process.__enter__.return_value = process
    process.poll.side_effect = [None, None, None, None, None, 0, 0, 0]
    process.wait.return_value = 0
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(mavlink_capture.subprocess, "Popen", popen)
    monkeypatch.setattr(
        native_flight, "read_result", lambda _: {"status": "completed"}
    )

    ports = {"GZ_DISCOVERY_MSG_PORT": 24100, "GZ_DISCOVERY_SRV_PORT": 24200}
    monkeypatch.setenv("GZ_DISCOVERY_MSG_PORT", "10317")
    result = mavlink_capture.capture_flight(
        receiver, tmp_path, ["simulator"], ports
    )

    assert result["flight_exit"] == 0
    process.terminate.assert_not_called()
    assert all(
        popen.call_args.kwargs["env"][name] == str(port)
        for name, port in ports.items()
    )
    receiver.setblocking.assert_not_called()
    assert receiver.recvfrom.call_count == 7
    with (tmp_path / "datagrams.bin").open("rb") as capture:
        assert capture.read(8) == mavlink_capture.MAGIC
        for datagram in expected:
            header = capture.read(mavlink_capture.PACKET.size)
            _, _, address, port, size = mavlink_capture.PACKET.unpack(header)
            assert (address, port) == (b"\x7f\x00\x00\x01", 18570)
            assert size == len(datagram)
            assert capture.read(size) == datagram
        assert capture.read() == b""
    assert not list(tmp_path.glob("*.jsonl*"))
