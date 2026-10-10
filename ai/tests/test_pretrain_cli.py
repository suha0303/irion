"""Short local fixtures only. Never start SITL, services or production preparation."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pretrain import cli, collection, common, runner


@pytest.mark.parametrize("count", common.CHOICES)
def test_all_sixteen_lane_ports_are_unique(count):
    lanes = collection.lanes(count)
    ports = [p for lane in lanes for p in lane["ports"]]
    assert len(ports) == len(set(ports)) == 5 * count
    assert [x["instance"] for x in lanes] == list(range(count))


def test_capacity_never_silently_downgrades():
    assert common.admission(4, 300_000_000, 12_000_000_000)["passed"]
    assert not common.admission(16, 800_000_000, 12_000_000_000)["passed"]
    assert not common.admission(1, 300_000_000, 4_100_000_000)["passed"]
    assert not common.admission(16, common.CPU_WORKER_BYTES["preprocess"], 11_040_000_000)["passed"]
    assert common.admission(2, common.CPU_WORKER_BYTES["preprocess"], 11_040_000_000)["passed"]


def test_runner_creates_storage_before_space_check(tmp_path, monkeypatch):
    task = tmp_path / "task"
    common.write(task / "config.json", {"mode": "test", "source_sha256": {}})
    (task / "config.sha256").write_text(common.sha(task / "config.json"))
    (task / "operator").mkdir()
    store = tmp_path / "mounted/Data/X500"
    monkeypatch.setattr(runner, "STORE", store)
    monkeypatch.setattr(runner, "external", lambda path: path)
    r = runner.Runner(task)

    class CheckedStorage(Exception):
        pass

    def resources():
        assert store.is_dir()
        raise CheckedStorage

    monkeypatch.setattr(r, "resources", resources)
    with pytest.raises(CheckedStorage):
        r.run()
    assert not (task / "operator/claimed").exists()

    # Validation must reject an absent mount before mkdir or any work admission.
    other = tmp_path / "unmounted/Data/X500"
    monkeypatch.setattr(runner, "STORE", other)

    def absent(path):
        raise ValueError("mount absent")

    monkeypatch.setattr(runner, "external", absent)
    with pytest.raises(ValueError, match="mount absent"):
        r.run()
    assert not other.parents[1].exists()


@pytest.mark.parametrize("override", [False, True])
def test_memory_violation_snapshot_survives_recovery(tmp_path, monkeypatch, override):
    task = tmp_path / "task"
    common.write(
        task / "config.json",
        {
            "mode": "test",
            "source_sha256": {},
            "ignore_memory_estimate": override,
        },
    )
    r = runner.Runner(task)
    monkeypatch.setattr(runner, "external", lambda path: path)
    from types import SimpleNamespace

    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=100_000_000_000)
    )
    samples = iter(
        [
            dict(available_bytes=3_000_000_000, owned_tree_rss_bytes=8_100_000_000),
            dict(available_bytes=11_000_000_000, owned_tree_rss_bytes=15_000_000),
        ]
    )
    monkeypatch.setattr(r.c, "memory_snapshot", lambda: next(samples))
    with pytest.raises(runner.ResourceBlocked, match="available=3.000GB"):
        r.resources()
    r.resources()
    assert r.state["resources"]["available_bytes"] == 11_000_000_000
    assert r.state["first_resource_violation"]["snapshot"]["available_bytes"] == 3_000_000_000
    assert r.state["resource_peaks"]["max_owned_rss_bytes"] == 8_100_000_000


def test_impossible_sixteen_stops_before_extract(tmp_path, monkeypatch):
    r = fake_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(r, "batch", lambda *a, **k: pytest.fail("must not extract or launch"))
    with pytest.raises(runner.ResourceBlocked, match="before extraction"):
        r.prepare({"runs": []}, "test-16", 16, full=False)
    assert not (tmp_path / "external").exists()


def test_one_gb_floor_keeps_rss_limit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    task = tmp_path / "task"
    common.write(
        task / "config.json",
        {
            "mode": "test",
            "source_sha256": {},
            "ignore_memory_estimate": True,
            "min_available_ram_gb": 1,
        },
    )
    r = runner.Runner(task)
    monkeypatch.setattr(runner, "external", lambda path: path)
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=100_000_000_000)
    )
    sample = dict(available_bytes=1_000_000_000, owned_tree_rss_bytes=7_000_000_000)
    monkeypatch.setattr(r.c, "memory_snapshot", lambda: sample.copy())
    r.resources()
    sample["available_bytes"] = 999_999_999
    with pytest.raises(runner.ResourceBlocked, match="min1"):
        r.resources()
    sample.update(available_bytes=2_000_000_000, owned_tree_rss_bytes=8_000_000_001)
    with pytest.raises(runner.ResourceBlocked, match="max8"):
        r.resources()


def test_expanded_limits_are_default_without_question():
    args = cli.parser().parse_args(["init", "--name", "fixture"])
    assert args.use_available_ram and args.ignore_memory_estimate
    assert args.min_available_ram_gb == 1


def test_estimate_override_reaches_batch_without_claiming_capacity(tmp_path, monkeypatch):
    r = fake_runner(tmp_path, monkeypatch)
    r.config["ignore_memory_estimate"] = True
    estimate = r.cpu_admission(16, "preprocess", 11_000_000_000)
    assert estimate["allowed"] and not estimate["passed"]
    assert estimate["estimated_bytes"] > 30_000_000_000
    assert r.state["admission"]["estimate_only_override"]

    class ReachedExtraction(Exception):
        pass

    monkeypatch.setattr(runner, "external", lambda path: tmp_path / "output")
    monkeypatch.setattr(
        r,
        "rows",
        lambda frozen, label: {
            "runs": [{"path": str(tmp_path / "consumer" / "run")}],
        },
    )

    def extract(*args, **kwargs):
        assert args[1] == "extract"
        raise ReachedExtraction

    monkeypatch.setattr(r, "batch", extract)
    with pytest.raises(ReachedExtraction):
        r.prepare({"runs": []}, "test-16", 16, full=False)


def test_removed_prepare_mode_rejected():
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["init", "--name", "fixture", "--mode", "prepare"])


def test_broken_utf8_input_reprompts_all(monkeypatch, capsys):
    inputs = iter([UnicodeDecodeError("utf-8", b"\xeb", 0, 1, "invalid continuation byte"), "all"])

    def enter(prompt):
        value = next(inputs)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("builtins.input", enter)
    assert cli.choose("범위", ("all", "cpu", "flight"), "all") == "all"
    assert "인코딩 오류" in capsys.readouterr().out


def test_levels_all_and_bad_value():
    assert cli.parser().parse_args(["init", "--name", "fixture", "--levels", "all"]).levels == list(
        common.CHOICES
    )
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["init", "--name", "fixture", "--levels", "all", "16"])


def test_expanded_memory_service_uses_available_minus_reserve(monkeypatch):
    from types import SimpleNamespace

    sample = dict(available_bytes=14_000_000_000)
    monkeypatch.setattr(cli, "collector", lambda: SimpleNamespace(memory_snapshot=lambda: sample))
    cfg = dict(mode="test", use_available_ram=True, min_available_ram_gb=1)
    plan = cli.service_memory(cfg)
    assert plan["max_bytes"] == plan["high_bytes"] == 13_000_000_000
    assert plan["swap_max_bytes"] == 0
    sample["available_bytes"] = 1_000_000_000
    with pytest.raises(ValueError, match="시작하지"):
        cli.service_memory(cfg)


def test_expanded_all_menu_selects_same_policy(monkeypatch):
    inputs = iter(["1", "7", "", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    captured = []
    monkeypatch.setattr(cli, "create", lambda args: captured.append(args) or Path("fixture"))
    monkeypatch.setattr(cli, "start", lambda task: None)
    monkeypatch.setattr(cli, "watch", lambda task: None)
    cli.menu()
    args = captured[0]
    assert args.test_flights and args.use_available_ram and args.ignore_memory_estimate
    assert args.min_available_ram_gb == 1 and args.levels == [7]


def test_expanded_rss_over_eight_and_floor_preserved(tmp_path, monkeypatch):
    from types import SimpleNamespace

    r = fake_runner(tmp_path, monkeypatch)
    r.config.update(
        use_available_ram=True,
        min_available_ram_gb=1,
        ignore_memory_estimate=True,
        test_flights=True,
    )
    monkeypatch.setattr(runner, "external", lambda path: path)
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=100_000_000_000)
    )
    sample = dict(available_bytes=1_100_000_000, owned_tree_rss_bytes=12_000_000_000)
    monkeypatch.setattr(r.c, "memory_snapshot", lambda: sample.copy())
    runner.Runner.resources(r)
    assert r.cpu_admission(16, "preprocess", sample["available_bytes"])["allowed"]
    sample["available_bytes"] = 900_000_000
    with pytest.raises(runner.ResourceBlocked, match="min1"):
        runner.Runner.resources(r)


def test_failed_screen_labels_sixteen_and_not_success(tmp_path):
    common.write(
        tmp_path / "config.json",
        dict(
            mode="test",
            levels=[16],
            test_flights=True,
            workers=4,
            io_workers=4,
            audit_workers=4,
            flight_workers=4,
            export_workers=1,
        ),
    )
    common.write(
        tmp_path / "operator/state.json",
        dict(
            status="FAILED_STOPPED",
            workers=16,
            done=16,
            total=16,
            completed=["extract"],
            active=[],
            reason="RAM limit",
        ),
    )
    output = cli.render(tmp_path)
    assert "시험 병렬 16" in output and "현재 단계 16개" in output
    assert "미완료 · 중지" in output and "단계 판정: 통과 아님" in output
    assert "단계 잔여 예상" not in output and "이전 버전에서 저장하지 않아 미확인" in output


def test_old_successful_flight_screen_uses_proven_completed_count(tmp_path):
    common.write(tmp_path / "config.json", dict(mode="test", levels=[16], test_flights=True))
    common.write(
        tmp_path / "operator/state.json",
        dict(
            status="TEST_COMPLETE_STOPPED",
            stage="flight-test-16",
            done=0,
            total=16,
            completed=["flight-test-16"],
            stages=[dict(name="flight-test-16", completed=16, workers=16)],
        ),
    )
    before = (tmp_path / "operator/state.json").read_bytes()
    assert "100.0%  16/16건" in cli.render(tmp_path)
    assert (tmp_path / "operator/state.json").read_bytes() == before


def test_new_flight_finish_persists_count_and_duration(tmp_path, monkeypatch):
    r = fake_runner(tmp_path, monkeypatch)
    r.begin("flight", total=16)
    r.finish("flight", dict(completed=16, workers=16))
    persisted = common.read(r.state_path)
    assert persisted["done"] == persisted["total"] == 16
    assert persisted["stages"][-1]["elapsed_s"] >= 0


def test_collection_menu_allows_sixteen_without_gates(monkeypatch):
    inputs = iter(["2", "16", "16", "16", "2", "16", "16", "n", "", "", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    captured = []
    monkeypatch.setattr(cli, "create", lambda args: captured.append(args) or Path("fixture"))
    monkeypatch.setattr(cli, "start", lambda task: None)
    monkeypatch.setattr(cli, "watch", lambda task: None)
    cli.menu()
    args = captured[0]
    assert args.mode == "collect" and args.use_available_ram and not args.gates
    assert args.collection_sets == 2 and args.collection_profile == "balanced300"
    assert (
        args.workers
        == args.io_workers
        == args.audit_workers
        == args.flight_workers
        == args.export_workers
        == 16
    )


@pytest.mark.parametrize("mode", ["collect", "test"])
def test_expanded_memory_includes_postcollection_cpu(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace

    r = fake_runner(tmp_path, monkeypatch)
    r.config.update(mode=mode, use_available_ram=True, min_available_ram_gb=1)
    assert r.cpu_admission(16, "preprocess", 11_000_000_000)["allowed"]
    monkeypatch.setattr(
        cli,
        "collector",
        lambda: SimpleNamespace(memory_snapshot=lambda: dict(available_bytes=11_000_000_000)),
    )
    assert cli.service_memory(r.config)["max_bytes"] == 10_000_000_000


def test_create_collection_expansion_frozen_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    # Config serialization only; no service or hash verification is run here.
    monkeypatch.setattr(cli, "sha", lambda path: "fixture-digest")
    args = cli.parser().parse_args(
        [
            "init",
            "--name",
            "collection",
            "--mode",
            "collect",
            "--use-available-ram",
            "--flight-workers",
            "16",
            "--export-workers",
            "16",
            "--workers",
            "16",
        ]
    )
    task = cli.create(args)
    config = common.read(task / "config.json")
    assert config["use_available_ram"] and config["ignore_memory_estimate"]
    assert config["min_available_ram_gb"] == 1
    assert not config["training_authorized"] and not config["gates"]
    assert not (task / "operator/service.json").exists()


def test_claimed_task_preserves_state(tmp_path):
    task = tmp_path / "task"
    common.write(task / "operator/state.json", {"status": "FAILED_STOPPED", "reason": "original"})
    (task / "operator/claimed").touch()
    before = (task / "operator/state.json").read_bytes()
    with pytest.raises(RuntimeError, match="already claimed"):
        runner.worker(task)
    assert (task / "operator/state.json").read_bytes() == before


def test_overlap_counts_intersections_not_requested_workers():
    assert common.overlap([dict(started=1, finished=2), dict(started=2, finished=3)]) == 1
    assert common.overlap([dict(started=1, finished=3), dict(started=2, finished=4)]) == 2


def test_parser_has_no_training_or_gpu_operation():
    args = cli.parser().parse_args(["init", "--name", "fixture", "--workers", "16"])
    assert args.workers == 16 and not args.gates
    for action in ("train", "calibrate", "resume"):
        with pytest.raises(SystemExit):
            cli.parser().parse_args([action])


def test_seed_remap_preserves_required_splits_and_pairing():
    common.package_imports()
    from preprocessing_tools.control import validate_cases

    rows = collection.cases(500000)["cases"]
    validate_cases(rows)
    assert len({r["seed"] for r in rows}) == 38
    assert all(r["mission_seconds"] == 180 for r in rows if r["scenario"] == "low_soc_normal")


def test_start_uses_independent_service_without_wait_or_restart(tmp_path, monkeypatch):
    home = tmp_path / "home"
    task = home / "tasks/unit-test"
    common.write(
        task / "config.json", {"source_sha256": {}, "external_storage": str(tmp_path / "Data")}
    )
    (task / "config.sha256").write_text(common.sha(task / "config.json"))
    monkeypatch.setattr(cli, "HOME", home)
    commands = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kw: commands.append(command))
    cli.start(task)
    command = commands[0]
    assert command[0] == "systemd-run" and "--user" in command
    assert "--property=Restart=no" in command and "--wait" not in command
    assert "--property=KillMode=control-group" in command
    assert "--property=OOMPolicy=stop" in command
    assert "--property=RuntimeMaxSec=604800" in command
    assert "--property=TimeoutStopSec=900" in command
    assert any(
        arg.startswith("--property=ExecStopPost=") and "pretrain.socket_cleanup" in arg
        for arg in command
    )
    assert not {"ssh", "sbatch", "srun"} & set(command)
    assert common.read(task / "operator/service.json")["restart"] == "no"


def test_start_expanded_service_removes_old_fixed_caps(tmp_path, monkeypatch):
    from types import SimpleNamespace

    home = tmp_path / "home"
    task = home / "tasks/unit-test"
    common.write(
        task / "config.json",
        dict(
            source_sha256={},
            mode="test",
            use_available_ram=True,
            min_available_ram_gb=1,
            external_storage=str(tmp_path / "Data"),
        ),
    )
    (task / "config.sha256").write_text(common.sha(task / "config.json"))
    monkeypatch.setattr(cli, "HOME", home)
    monkeypatch.setattr(
        cli,
        "collector",
        lambda: SimpleNamespace(memory_snapshot=lambda: dict(available_bytes=14_000_000_000)),
    )
    commands = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kw: commands.append(command))
    cli.start(task)
    command = commands[0]
    assert "--property=MemoryMax=13000000000" in command
    assert "--property=MemoryHigh=13000000000" in command
    assert "--property=MemorySwapMax=0" in command
    assert "--property=MemoryHigh=8000000000" not in command
    assert common.read(task / "operator/memory-policy.json")["reserve_bytes"] == 1_000_000_000


def test_expanded_floor_stops_only_owned_cpu_workers(tmp_path, monkeypatch):
    r = fake_runner(tmp_path, monkeypatch)
    r.config.update(use_available_ram=True, ignore_memory_estimate=True, min_available_ram_gb=1)
    children = []
    terminated = []
    calls = 0

    def resources():
        nonlocal calls
        calls += 1
        r.state["resources"] = dict(available_bytes=14_000_000_000 if calls <= 2 else 900_000_000)
        if calls > 2:
            raise runner.ResourceBlocked("RAM limit: min1")
        return r.state["resources"]

    class Child:
        def __init__(self, command, **kw):
            self.pid = 1000 + len(children)
            self.code = None
            children.append(self)

        def poll(self):
            return self.code

        def terminate(self):
            terminated.append(self.pid)
            self.code = -15

    monkeypatch.setattr(r, "resources", resources)
    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    with pytest.raises(runner.ResourceBlocked, match="min1"):
        r.batch(
            "floor-test",
            "preprocess",
            [dict(run_id=f"r{i}") for i in range(4)],
            2,
            tmp_path / "out",
        )
    assert terminated == [1000, 1001] and len(children) == 2
    assert common.read(r.task / "units/floor-test/result.json")["failure_kind"] == "resource"


def test_unknown_dead_pid_not_success(tmp_path, monkeypatch):
    common.write(
        tmp_path / "operator/state.json",
        {"status": "RUNNING", "pid": 123, "pid_start_ticks": "old"},
    )
    monkeypatch.setattr(cli, "identity", lambda pid: None)
    assert cli.snapshot(tmp_path)["status"] == "UNKNOWN_NOT_RUNNING"


def fake_runner(tmp_path, monkeypatch):
    task = tmp_path / "task"
    common.write(
        task / "config.json",
        {"mode": "test", "source_sha256": {}, "io_workers": 2, "audit_workers": 2},
    )
    result = runner.Runner(task)
    monkeypatch.setattr(result, "resources", lambda: {"available_bytes": 16_000_000_000})
    return result


@pytest.mark.parametrize("count", common.CHOICES)
def test_all_benchmark_uses_fresh_flights_before_cpu(tmp_path, monkeypatch, count):
    r = fake_runner(tmp_path, monkeypatch)
    r.config.update(levels=[count], seed_base=10000)
    calls = []
    folder = tmp_path / "fresh"
    frozen = {"status": "BENCHMARK_ONLY", "runs": []}

    def flight(phase, workers, rows, label):
        assert phase == "benchmark" and workers == count
        assert len(rows) == max(3, count)
        assert all(row["role"] == "benchmark" and row["mission_seconds"] == 120 for row in rows)
        calls.append("flight")
        return folder

    def freeze(path):
        assert path == folder and calls == ["flight"]
        calls.append("freeze")
        return frozen

    def prepare(value, label, workers, **kwargs):
        assert value is frozen and kwargs == {"full": False}
        assert calls == ["flight", "freeze"] and workers == count
        calls.append("prepare")
        r.state["stages"] = [
            dict(name=label + "-" + action, max_overlap=count)
            for action in ("extract", "preprocess", "audit")
        ]
        return tmp_path / "prepared"

    monkeypatch.setattr(r, "flight", flight)
    monkeypatch.setattr(collection, "freeze_benchmark", freeze)
    monkeypatch.setattr(r, "prepare", prepare)
    r.benchmark()
    result = common.read(r.task / "parallel-results.json")[0]
    assert calls == ["flight", "freeze", "prepare"]
    assert result["status"] == "PASSED" and result["training_eligible"] is False


def test_all_benchmark_stops_before_cpu_on_flight_failure(tmp_path, monkeypatch):
    r = fake_runner(tmp_path, monkeypatch)
    r.config.update(levels=[1, 2], seed_base=10000)
    calls = []

    def fail(*args):
        calls.append(args[1])
        raise RuntimeError("preserved flight failure")

    monkeypatch.setattr(r, "flight", fail)
    monkeypatch.setattr(r, "prepare", lambda *a, **kw: pytest.fail("CPU must not start"))
    with pytest.raises(RuntimeError, match="preserved flight failure"):
        r.benchmark()
    assert calls == [1]
    assert common.read(r.task / "parallel-results.json")[0]["status"] == "FAILED_STOPPED"


@pytest.mark.parametrize("count", common.CHOICES)
def test_scheduler_launch_bound_and_completion(tmp_path, monkeypatch, count):
    r = fake_runner(tmp_path, monkeypatch)
    launched = []

    class Child:
        def __init__(self, argv, **kwargs):
            self.pid = 1000 + len(launched)
            req = common.read(argv[-1])
            launched.append(req)
            common.write(
                req["status"],
                {"status": "COMPLETE", "started": 1, "finished": 2, "peak_rss_bytes": 10_000_000},
            )

        def poll(self):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    result = r.batch(
        "synthetic", "verify", [{"run_id": f"r{i}"} for i in range(count)], count, tmp_path / "out"
    )
    assert len(launched) == count and result["completed"] == count
    assert result["max_overlap"] == count
    assert r.state["active"] == []


def test_failure_drains_active_without_admitting_queue(tmp_path, monkeypatch):
    r = fake_runner(tmp_path, monkeypatch)
    launched = []

    class Child:
        def __init__(self, argv, **kwargs):
            self.pid = 1000 + len(launched)
            req = common.read(argv[-1])
            launched.append(req)
            common.write(req["status"], {"status": "FAILED_STOPPED", "reason": "fixture failure"})

        def poll(self):
            return 2

    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="fixture failure"):
        r.batch("failure", "verify", [{"run_id": f"r{i}"} for i in range(9)], 2, tmp_path / "out")
    assert len(launched) == 2
    assert (r.task / "units/failure/result.json").is_file()
    assert r.state["active"] == []


@pytest.mark.parametrize("cached_audit", [False, True])
def test_parallel_handoff_uses_existing_numeric_algorithms(tmp_path, monkeypatch, cached_audit):
    common.package_imports()
    fixtures = common.load("numeric_fixture", common.ROOT / "ai/tests/fixtures/pretrain_numeric.py")
    dataset = tmp_path / "reference"
    fixtures.make_dataset(dataset)
    frozen = common.read(dataset / "manifest.json")
    frozen = {"runs": frozen["runs"]}
    config = common.read(common.PACKAGE / "preprocessing.json")
    for row in frozen["runs"]:
        row["files"] = {}
        row["path"] = str(tmp_path / "raw" / row["run_id"])
        folder = dataset / "canonical" / row["run_id"]
        summary = common.read(folder / "summary.json")
        summary.update(
            config=config,
            input_hashes={},
            output_hashes={
                n: common.sha(folder / (n + ".jsonl")) for n in ("frames", "windows", "exclusions")
            },
        )
        common.write(folder / "summary.json", summary)
    r = fake_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "STORE", tmp_path / "external")
    monkeypatch.setattr(runner, "external", lambda path: Path(path))

    def batch(name, action, rows, workers, out, **kwargs):
        out.mkdir(parents=True)
        if action == "preprocess":
            for row in rows:
                shutil.copytree(dataset / "canonical" / row["run_id"], out / row["run_id"])
        elif action == "audit":
            for row in rows:
                report, samples, selected = fixtures.audit.inspect_run(
                    kwargs["canonical"] / row["run_id"], row
                )
                folder = out / row["run_id"]
                folder.mkdir()
                arrays = dict(zip(("X", "C", "Y", "run_id", "origin_us", "segment"), zip(*samples)))
                np.savez(
                    folder / "numeric.npz",
                    **{k: np.asarray(v) for k, v in arrays.items()},
                    train_frames=np.asarray(selected),
                )
                common.write(folder / "report.json", report)
                common.write(
                    r.task / "units" / name / (row["run_id"] + ".status.json"),
                    {
                        "status": "COMPLETE",
                        "run_id": row["run_id"],
                        "result": {
                            "numeric_sha256": common.sha(folder / "numeric.npz"),
                            "report_sha256": common.sha(folder / "report.json"),
                        },
                    },
                )

    monkeypatch.setattr(r, "batch", batch)
    if cached_audit:
        output = tmp_path / "external/prepared"
        output.mkdir(parents=True)
        common.write(output / "frozen-manifest.json", frozen)
        batch("cached", "preprocess", frozen["runs"], 2, output / "canonical-work")
        batch(
            "cached",
            "audit",
            frozen["runs"],
            2,
            output / "audit-work",
            canonical=output / "canonical-work",
        )
        saved = {
            row["run_id"]: (
                output / "audit-work" / row["run_id"],
                r.task / "units/cached" / (row["run_id"] + ".status.json"),
            )
            for row in frozen["runs"]
        }
        monkeypatch.setattr(
            r, "batch", lambda *a, **k: pytest.fail("Must reuse completed run audits")
        )
        r.finalize_prepared(frozen, output, "unit-fixture", 2, full=False, preaudited=saved)
    else:
        output = r.prepare(frozen, "unit-fixture", 2, full=False)
    quality = common.read(output / "quality.json")
    assert quality["passed"] and quality["npz_max_abs_error"] == 0
    assert max(quality["scaler_max_abs_errors"].values()) < 1e-10
    assert not common.read(output / "handoff.json")["training_authorized"]


def test_real_worker_on_small_synthetic_raw_stream(tmp_path):
    """Exercise the worker entry point and original engine, with72s synthetic data only."""
    common.package_imports()
    from capstone_ai.cli import process

    from pretrain.worker import execute

    fixtures = common.load("raw_fixture", common.ROOT / "ai/tests/test_preprocessing.py")
    run_id = "1" * 32
    raw = tmp_path / run_id
    meta = copy.deepcopy(fixtures.META)
    meta["run_id"] = run_id
    meta["vehicle"].update(
        nominal_capacity_ah=12.0,
        nominal_parallel_cells=4,
        battery_profile_id="vtc6_4s4p",
        profile_revision="vtc6-4s4p",
    )
    common.write(raw / "observations/metadata.json", meta)
    common.write(
        raw / "run.json", dict(run_id=run_id, origin="synthetic_fixture", status="completed")
    )
    common.write(raw / "reference/source_lock.json", dict(synthetic_fixture=True))
    events = list(fixtures.events(72))
    for event in events:
        event["run_id"] = run_id
    (raw / "observations/streams.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (raw / "observations/watermarks.jsonl").write_text(
        json.dumps(
            dict(
                schema="drone.raw.watermark/2.0",
                run_id=run_id,
                clock_id="px4-boot",
                through_available_time_us=72_000_000,
                recorded_events=len(events),
                closed=True,
            )
        )
        + "\n"
    )
    (raw / "observations/diagnostics.jsonl").touch()
    inventory = {str(p.relative_to(raw)): common.sha(p) for p in raw.rglob("*") if p.is_file()}
    row = dict(run_id=run_id, path=str(raw), files=inventory, role="normal_train")
    config = common.read(common.PACKAGE / "preprocessing.json")
    reference = tmp_path / "reference"
    # The sealed engine is the numerical reference. Alias only its validation
    # input, then restore the real new metadata before inventory-checked work.
    legacy_meta = copy.deepcopy(meta)
    legacy_meta["vehicle"]["battery_profile_id"] = "vtc6_4s4p_r3_long15"
    common.write(raw / "observations/metadata.json", legacy_meta)
    try:
        expected = process(raw, reference / run_id, config)
    finally:
        common.write(raw / "observations/metadata.json", meta)
    assert {str(p.relative_to(raw)): common.sha(p) for p in raw.rglob("*") if p.is_file()} == inventory
    out = tmp_path / "canonical" / run_id
    request = dict(
        action="preprocess",
        row=row,
        out=str(out),
        status=str(tmp_path / "preprocess.json"),
        reference=str(reference),
    )
    execute(request)
    status = common.read(request["status"])
    assert status["status"] == "COMPLETE" and status["result"]["equivalence_passed"]
    assert status["result"]["output_hashes"] == expected["output_hashes"]
    assert common.read(out / "summary.json")["input_hashes"] == inventory
    assert common.read(raw / "observations/metadata.json") == meta
    request.update(
        action="audit",
        out=str(tmp_path / "audited"),
        canonical=str(out.parent),
        status=str(tmp_path / "audit.json"),
    )
    common.write(tmp_path / "request.json", request)
    result = subprocess.run(
        [str(common.PYTHON), "-B", "-m", "pretrain.worker", str(tmp_path / "request.json")],
        env={**__import__("os").environ, "PYTHONPATH": common.pythonpath()},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert common.read(request["status"])["result"]["report"]["accepted_targets"] > 0


def test_arbitrary_positive_worker_counts_and_current_flight_contract():
    args = cli.parser().parse_args(
        [
            "init",
            "--name",
            "fixture",
            "--workers",
            "23",
            "--io-workers",
            "3",
            "--audit-workers",
            "7",
            "--flight-workers",
            "5",
            "--levels",
            "3",
            "7",
            "15",
        ]
    )
    assert (args.workers, args.io_workers, args.audit_workers, args.flight_workers) == (23, 3, 7, 5)
    assert args.levels == [3, 7, 15]
    for n in ("0", "-1", "2.5"):
        with pytest.raises(SystemExit):
            cli.parser().parse_args(["init", "--name", "fixture", "--workers", n])


def test_cpu_failure_records_and_runs_remaining_units(tmp_path, monkeypatch):
    r = fake_runner(tmp_path, monkeypatch)
    r.config["continue_after_case_failure"] = True
    launched = []

    class Child:
        def __init__(self, argv, **kwargs):
            request = common.read(argv[-1])
            launched.append(request["row"]["run_id"])
            self.pid = 1000 + len(launched)
            self.code = 2 if len(launched) == 1 else 0
            common.write(
                request["status"],
                dict(
                    status="FAILED_STOPPED" if self.code else "COMPLETE",
                    reason="fixture failure",
                    started=1,
                    finished=2,
                    peak_rss_bytes=1000,
                ),
            )

        def poll(self):
            return self.code

    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="fixture failure"):
        r.batch(
            "fixture",
            "preprocess",
            [dict(run_id=f"r{i}") for i in range(5)],
            2,
            tmp_path / "output",
        )
    assert launched == [f"r{i}" for i in range(5)]
    result = common.read(r.task / "units/fixture/result.json")
    assert result["completed"] == 4 and result["failed"] == 1
    assert len(result["failure_records"]) == 1


def test_all_collection_sets_attempted_before_partial_dataset_rejected(tmp_path, monkeypatch):
    r = fake_runner(tmp_path, monkeypatch)
    r.config.update(
        collection_sets=2, seed_base=1000, flight_workers=3, collection_profile="train104"
    )
    calls = []

    def flight(phase, count, rows, label):
        calls.append(label)
        if len(calls) == 1:
            raise collection.IncompleteCollection("fixture partial")
        return tmp_path / label

    monkeypatch.setattr(r, "flight", flight)
    monkeypatch.setattr(
        r, "prepare", lambda *a, **k: pytest.fail("incomplete dataset cannot preprocess")
    )
    with pytest.raises(RuntimeError, match="no incomplete dataset"):
        r.collection_sets()
    assert calls == ["collection-set-001", "collection-set-002"]
    assert common.read(r.task / "collection-errors.json")["preprocessing_started"] is False


@pytest.mark.parametrize("action", ["extract", "preprocess", "audit"])
@pytest.mark.parametrize("failures", [2, 3])
def test_unit_retry_bound_preserves_partials_and_moves_on(tmp_path, monkeypatch, action, failures):
    r = fake_runner(tmp_path, monkeypatch)
    r.config.update(unit_attempts_max=3, continue_after_case_failure=True)
    launched = []
    raw = tmp_path / "consumer"
    raw.mkdir()

    class Child:
        def __init__(self, argv, **kwargs):
            req = common.read(argv[-1])
            rid = req["row"]["run_id"]
            attempt = req["attempt"]
            launched.append((rid, attempt))
            self.pid = 1000 + len(launched)
            self.code = 2 if rid == "r0" and attempt <= failures else 0
            dest = Path(req["out"])
            dest.mkdir()
            (dest / "marker").write_text(str(attempt))
            if action == "extract":
                consumer = Path(req["row"]["path"])
                consumer.mkdir()
                (consumer / "raw").write_text(str(attempt))
            common.write(
                req["status"],
                dict(
                    status="FAILED_STOPPED" if self.code else "COMPLETE",
                    reason="fixture failure",
                    started=1,
                    finished=2,
                    peak_rss_bytes=1000,
                ),
            )

        def poll(self):
            return self.code

    monkeypatch.setattr(runner.subprocess, "Popen", Child)

    def run():
        return r.batch(
            "retry",
            action,
            [dict(run_id=f"r{i}", path=str(raw / f"r{i}")) for i in range(2)],
            1,
            tmp_path / "output",
        )

    if failures == 3:
        with pytest.raises(RuntimeError, match="fixture failure"):
            run()
    else:
        run()
    assert launched == [("r0", 1), ("r0", 2), ("r0", 3), ("r1", 1)]
    for attempt in range(1, failures + 1):
        proof = r.task / f"units/retry/attempts/r0/attempt-{attempt:02d}"
        assert (proof / "failure.json").exists()
        assert (proof / "r0.status.json").exists()
    for attempt in (1, 2):
        assert (tmp_path / f"output/.failed/r0/attempt-{attempt:02d}/marker").read_text() == str(
            attempt
        )
        if action == "extract":
            assert (raw / f".failed/r0/attempt-{attempt:02d}/raw").exists()
    result = common.read(r.task / "units/retry/result.json")
    assert result["failed"] == (failures == 3)
    assert result["attempt_counts"] == {"r0": 3, "r1": 1}
    assert r.state["done"] == 2


@pytest.mark.parametrize("code,scope", [(-9, "unit"), (-15, "unit"), (137, "unit"), (1, "shared")])
def test_shared_failure_does_not_retry_or_admit_next(tmp_path, monkeypatch, code, scope):
    r = fake_runner(tmp_path, monkeypatch)
    r.config.update(unit_attempts_max=3, continue_after_case_failure=True)
    launched = []

    class Child:
        pid = 1234

        def __init__(self, argv, **kwargs):
            req = common.read(argv[-1])
            launched.append(req)
            common.write(
                req["status"],
                dict(status="FAILED_STOPPED", reason="fatal fixture", failure_scope=scope),
            )

        def poll(self):
            return code

    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="fatal fixture"):
        r.batch("fatal", "preprocess", [dict(run_id="r0"), dict(run_id="r1")], 1, tmp_path / "out")
    assert len(launched) == 1


def test_extract_retry_never_moves_preexisting_unowned_directory(tmp_path, monkeypatch):
    r = fake_runner(tmp_path, monkeypatch)
    r.config.update(unit_attempts_max=3, continue_after_case_failure=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "keep").write_text("keep")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("no launch"))
    with pytest.raises(RuntimeError, match="existing extraction destination"):
        r.batch("extract", "extract", [dict(run_id="r0", path=str(raw))], 1, tmp_path / "out")
    assert (raw / "keep").read_text() == "keep"
