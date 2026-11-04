"""Synthetic processes/files only; no SITL, external device or monitoring."""

import importlib.util
import json
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "long_collect", Path(__file__).parents[2] / "Data/generation/tools/collect_long_runs.py"
)
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "runs").mkdir()
    (root / "ai").mkdir()
    external, local = tmp_path / "archive", root / "ai/execution"
    producer = root / "capstone-sim"
    producer.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """
import json,sys,time,uuid,hashlib,os
from pathlib import Path
root=Path(__file__).parent
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def event(v):
 with (root/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\\n')
if sys.argv[1]=='verify':
 print(json.dumps({'status':'passed'}));sys.exit(0)
def arg(k):return sys.argv[sys.argv.index(k)+1]
seed=int(arg('--seed'));instance=int(arg('--instance'));scenario=arg('--scenario');duration=int(arg('--duration'))
event({'event':'start','seed':seed,'instance':instance,'time':time.monotonic(),'args':sys.argv[1:]})
run=root/'runs'/uuid.uuid4().hex;run.mkdir()
if scenario=='fail':
 (run/'partial').write_text('preserve');time.sleep(.2);sys.exit(1)
time.sleep(.4)
put(run/'run.json',dict(run_id=run.name,status='completed',parent_attempt_id=None,wall_duration_s=.4,process_exit_codes=[0,0]))
put(run/'evaluation/scenario.json',dict(name=scenario,seed=seed,requested_duration_s=duration,continuity='independent_reset',health_eligibility='confirmed_healthy',operation_variation=json.loads(arg('--operation-variation')) if '--operation-variation' in sys.argv else None,fault_after_mission_s=float(arg('--fault-after-mission-s')) if '--fault-after-mission-s' in sys.argv else None))
runtime=json.loads((root/'synthetic-runtime.json').read_text()) if (root/'synthetic-runtime.json').exists() else {}
put(run/'reference/source_lock.json',dict(local_sources={},runtime_binary_sha256=runtime))
put(run/'observations/streams.jsonl',{'signal':[1,2,3],'seed':seed})
(run/'native').mkdir();(run/'native/data').write_bytes(b'abc'*50000)
integrity={str(p.relative_to(run)):dict(size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in run.rglob('*') if p.is_file()}
put(run/'integrity.json',integrity)
event({'event':'end','seed':seed,'instance':instance,'time':time.monotonic()})
print(json.dumps({'status':'completed','run':str(run)}))
"""
    )
    producer.chmod(0o755)
    plan = dict(
        campaign_id="synthetic",
        root=str(root),
        execution_authorized=True,
        execution_dir=str(local),
        external_output=str(external),
        mission_seconds=1200,
        concurrency=2,
        attempts_max=3,
        timing=False,
        source_hashes={"capstone-sim": c.sha(producer)},
        lanes=[dict(instance=3, ports=[14543]), dict(instance=4, ports=[14544])],
        budgets=dict(
            flight_wall_s=1500,
            verify_wall_s=60,
            archive_wall_s=60,
            archive_verify_wall_s=60,
            byte_stall_s=10,
            campaign_wall_s=2000,
            local_bytes_cap=10**8,
            external_bytes_cap=10**8,
            min_free_bytes=1,
            min_available_ram_bytes=1,
            memory_soft_cap_bytes=10**8,
        ),
    )
    cases = [
        dict(
            case_id=f"c{i}", scenario="hover_normal", seed=i, role="normal_train", group_id=f"g{i}"
        )
        for i in range(1, 4)
    ]
    evidence = tmp_path / "isolation.json"
    dump(
        evidence,
        dict(
            parallel_safe=True,
            uuid_partitioned=True,
            immutable_shared_inputs=True,
            lanes=plan["lanes"],
            source_hashes=plan["source_hashes"],
        ),
    )
    plan["isolation_evidence"] = dict(path=str(evidence), sha256=c.sha(evidence))

    def command(args, cwd, log, check, seconds):
        check()
        with Path(log).open("wb") as f:
            result = subprocess.run(
                args, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=seconds
            )
        check()
        if result.returncode:
            raise c.Stop("fake producer failure")

    helper = SimpleNamespace(command=command, check_mount=lambda path: 0)
    monkeypatch.setattr(c, "preflight", lambda plan, helper: (root, external, local))
    monkeypatch.setattr(
        c, "memory_snapshot", lambda: dict(available_bytes=10**10, owned_tree_rss_bytes=0)
    )
    return plan, cases, helper, tmp_path


def test_processing_failure_before_admission_starts_no_flight(setup):
    plan, cases, helper, tmp = setup
    plan.update(
        phase="collection",
        pretrain_cli_approval=str(tmp / "task/config.json"),
        processing_stop_signal=str(tmp / "task/operator/processing-stop.json"),
    )
    dump(Path(plan["processing_stop_signal"]), dict(reason="failed CPU processing"))
    result = c.execute(plan, cases, helper)
    assert result["status"] == "FAILED_DRAINED_STOPPED"
    assert len(result["not_started"]) == len(cases)
    assert not (Path(plan["root"]) / "events.jsonl").exists()


def test_processing_failure_drains_current_flight_even_inside_sample_interval(setup):
    plan, cases, helper, tmp = setup
    plan.update(
        concurrency=1,
        phase="collection",
        pretrain_cli_approval=str(tmp / "task/config.json"),
        processing_stop_signal=str(tmp / "task/operator/processing-stop.json"),
    )
    plan["lanes"] = plan["lanes"][:1]
    original = helper.command

    def command(args, cwd, log, check, seconds):
        result = original(args, cwd, log, check, seconds)
        # Producer has finished, but verification/export of its owned run must drain.
        if args[1] != "verify":
            dump(Path(plan["processing_stop_signal"]), dict(reason="CPU failed"))
        return result

    helper.command = command
    result = c.execute(plan, cases, helper)
    assert result["status"] == "FAILED_DRAINED_STOPPED"
    assert len(result["cases"]) == 1 and len(result["not_started"]) == 2
    assert next(iter(result["cases"].values()))["status"] == "ARCHIVED_PENDING_QUALITY"
    assert "Processing failure" in str(result["stop_reason"])


def test_byte_accounting_tolerates_retired_previous_group(tmp_path):
    assert c.bytes_under(tmp_path / "already-retired") == 0
    assert list(c.files(tmp_path / "already-retired", missing_ok=True)) == []
    with pytest.raises(FileNotFoundError):
        list(c.files(tmp_path / "already-retired"))


def test_two_lanes_isolated_and_lossless(setup):
    plan, cases, helper, tmp = setup
    result = c.execute(plan, cases, helper)
    assert result["status"] == "COLLECTED_PENDING_QUALITY_STOPPED"
    events = [json.loads(x) for x in (Path(plan["root"]) / "events.jsonl").read_text().splitlines()]
    starts = [x for x in events if x["event"] == "start"]
    ends = [x for x in events if x["event"] == "end"]
    assert max(x["time"] for x in starts[:2]) < min(x["time"] for x in ends[:2])
    assert {x["instance"] for x in starts[:2]} == {3, 4}
    assert len({r["run_id"] for r in result["cases"].values()}) == 3
    for record in result["cases"].values():
        export = record["export"]
        assert c.sha(export["archive"]) == export["sha256"]
        with tarfile.open(export["archive"], "r:gz") as tar:
            assert tar.extractfile("native/data").read() == b"abc" * 50000
        assert (
            Path(plan["execution_dir"]) / record["case"]["case_id"] / "archive-progress.json"
        ).exists()
    assert len(list(Path(plan["external_output"]).glob("*.tar.gz"))) == 3
    assert not list(Path(plan["external_output"]).glob("*/observations"))


def test_new_collection_recreates_deleted_output_parents(setup, monkeypatch):
    plan, cases, helper, tmp = setup
    root = Path(plan["root"])
    (root / "runs").rmdir()
    plan["execution_dir"] = str(root / "ai/missing/task/execution")
    plan["external_output"] = str(tmp / "missing/archive/task")
    monkeypatch.setattr(
        c,
        "preflight",
        lambda plan, helper: (root, Path(plan["external_output"]), Path(plan["execution_dir"])),
    )
    result = c.execute(plan, cases[:1], helper)
    assert result["status"] == "COLLECTED_PENDING_QUALITY_STOPPED"
    assert (root / "runs").is_dir()
    assert Path(plan["external_output"]).is_dir()
    with pytest.raises(FileExistsError):
        c.execute(plan, cases[:1], helper)


def test_external_parent_creation_requires_real_mount(tmp_path, monkeypatch):
    mount = tmp_path / "X31"
    mount.mkdir()
    output = mount / "Data/X500/archives/new-task"
    checked = []
    helper = SimpleNamespace(check_mount=lambda path: checked.append(path))
    monkeypatch.setattr(Path, "is_mount", lambda self: False)
    with pytest.raises(c.Stop, match="External mount absent"):
        c.prepare_external_parent(output, helper, mount)
    assert not (mount / "Data").exists()
    monkeypatch.setattr(Path, "is_mount", lambda self: self == mount)
    c.prepare_external_parent(output, helper, mount)
    assert output.parent.is_dir() and not output.exists()
    assert checked == [output.parent]
    (mount / "Data/link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(c.Stop, match="Symlink"):
        c.prepare_external_parent(mount / "Data/link/new-task", helper, mount)


def test_first_failure_drains_peer_and_no_third_start(setup):
    plan, cases, helper, tmp = setup
    cases[0]["scenario"] = "fail"
    result = c.execute(plan, cases, helper)
    assert result["status"] == "FAILED_DRAINED_STOPPED"
    assert result["cases"]["c1"]["status"] == "FAILED_PRESERVED"
    assert result["cases"]["c2"]["status"] == "ARCHIVED_PENDING_QUALITY"
    assert result["not_started"] == ["c3"]
    events = [json.loads(x) for x in (Path(plan["root"]) / "events.jsonl").read_text().splitlines()]
    assert {x["seed"] for x in events if x["event"] == "start"} == {1, 2}
    assert any(p.name == "partial" for p in (Path(plan["root"]) / "runs").rglob("*"))


def test_byte_progress_timeout_does_not_use_heartbeat(tmp_path):
    clock = [0.0]
    p = c.Progress(tmp_path / "progress.json", "copy", 100, 5, clock=lambda: clock[0])
    clock[0] = 4
    p.advance(10)
    clock[0] = 8
    p.persist("RUNNING")
    p.advance(0)
    clock[0] = 9
    with pytest.raises(c.Stop, match="byte-progress"):
        p.check()


def test_source_change_stops_new_cases_but_preserves_peers(setup):
    plan, cases, helper, tmp = setup
    original = helper.command
    changed = [False]

    def mutate_after_verify(args, *rest):
        original(args, *rest)
        if args[1] == "verify" and not changed[0]:
            changed[0] = True
            with Path(plan["root"], "capstone-sim").open("a") as f:
                f.write("\n# changed\n")

    helper.command = mutate_after_verify
    result = c.execute(plan, cases, helper)
    assert result["status"] == "FAILED_DRAINED_STOPPED"
    assert len(result["cases"]) == 2
    assert all(x["status"] == "PRESERVED_SOURCE_DRIFT" for x in result["cases"].values())


def test_claim_prevents_duplicate_after_failure(setup):
    plan, cases, helper, tmp = setup
    cases[0]["scenario"] = "fail"
    c.execute(plan, cases, helper)
    with pytest.raises(FileExistsError):
        c.execute(plan, cases, helper)


def test_bound_dry_run_no_process_or_writes(setup, monkeypatch):
    plan, cases, helper, tmp = setup
    plan["execution_authorized"] = False
    pp, mp, ap = [tmp / x for x in ("plan.json", "manifest.json", "approval.json")]
    dump(pp, plan)
    dump(mp, {"cases": cases})
    dump(
        ap,
        dict(
            authorized=False,
            action="DRAFT",
            plan_sha256=c.sha(pp),
            manifest_sha256=c.sha(mp),
            collector_sha256=c.sha(c.__file__),
        ),
    )
    monkeypatch.setattr(c, "execute", lambda *a: pytest.fail("dry run must not execute"))
    assert (
        c.main(["--plan", str(pp), "--manifest", str(mp), "--approval", str(ap), "--dry-run"]) == 0
    )
    assert not Path(plan["execution_dir"]).exists()
    with pytest.raises(c.Stop, match="approval"):
        c.bound(pp, mp, ap)


def test_reject_split_overlap_and_short_duration(setup):
    plan, cases, helper, tmp = setup
    pp, mp, ap = [tmp / x for x in ("plan.json", "manifest.json", "approval.json")]

    def check():
        dump(pp, plan)
        dump(mp, {"cases": cases})
        dump(
            ap,
            dict(
                authorized=True,
                action="START_LONG_COLLECTION",
                plan_sha256=c.sha(pp),
                manifest_sha256=c.sha(mp),
                collector_sha256=c.sha(c.__file__),
            ),
        )
        return c.bound(pp, mp, ap)

    cases[1].update(seed=1, role="normal_validation")
    with pytest.raises(c.Stop, match="split leakage"):
        check()
    cases[1].update(seed=2, role="normal_train")
    plan["mission_seconds"] = 180
    with pytest.raises(c.Stop, match="1200"):
        check()


def test_verify_and_archive_share_one_serial_slot(setup, monkeypatch):
    plan, cases, helper, tmp = setup
    lock = threading.Lock()
    active = [0]
    peak = [0]
    events = []
    original_command, original_archive = helper.command, c.archive_run

    def tracked(kind, call, *args):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            events.append(kind)
        try:
            time.sleep(0.05)
            return call(*args)
        finally:
            with lock:
                active[0] -= 1

    def command(args, *rest):
        if args[1] == "verify":
            return tracked("verify", original_command, args, *rest)
        return original_command(args, *rest)

    helper.command = command
    monkeypatch.setattr(c, "archive_run", lambda *args: tracked("archive", original_archive, *args))
    result = c.execute(plan, cases, helper)
    assert result["status"] == "COLLECTED_PENDING_QUALITY_STOPPED"
    assert peak[0] == 1
    assert events == ["verify", "archive"] * 3


@pytest.mark.parametrize("damage", ["integrity", "archive"])
def test_archive_failure_preserves_partial_without_publishing(tmp_path, monkeypatch, damage):
    run = tmp_path / "run"
    run.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    data = run / "data"
    data.write_bytes(b"original" * 10000)
    dump(run / "integrity.json", {"data": dict(size=data.stat().st_size, sha256=c.sha(data))})
    destination = tmp_path / "run.tar.gz"
    partial = tmp_path / "run.tar.gz.partial"
    if damage == "integrity":
        data.write_bytes(b"altered!" + data.read_bytes()[8:])
    else:
        original = c.Progress.finish

        def corrupt_after_write(progress):
            elapsed = original(progress)
            if progress.stage == "archive_transfer":
                with partial.open("r+b") as stream:
                    stream.write(b"BAD!")
            return elapsed

        monkeypatch.setattr(c.Progress, "finish", corrupt_after_write)
    budgets = dict(archive_wall_s=10, archive_verify_wall_s=10, byte_stall_s=5)
    with pytest.raises((c.Stop, c.gzip.BadGzipFile)):
        c.archive_run(run, destination, stage, budgets, lambda: None)
    assert not destination.exists()
    assert partial.exists() and data.exists()
    assert c.read(stage / "archive-progress.json")["status"] == "FAILED"


def test_existing_instance_socket_blocks_without_removal(monkeypatch):
    seen = []
    monkeypatch.setattr(c.os.path, "lexists", lambda path: seen.append(str(path)) or True)
    with pytest.raises(c.Stop, match="never removed"):
        c.check_instance_socket(3)
    assert seen == ["/tmp/px4-sock-3"]


@pytest.fixture
def collection_setup(setup):
    plan, cases, helper, tmp = setup
    runtime = {}
    for name in (
        "vendor/PX4-Autopilot/build/px4_sitl_default/bin/px4",
        "build/current/libDronePowertrain.so",
        "build/libDronePowertrain.so",
    ):
        path = Path(plan["root"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture: " + name)
        plan["source_hashes"][name] = c.sha(path)
        if name != "build/libDronePowertrain.so":
            runtime[name] = c.sha(path)
    dump(Path(plan["root"]) / "synthetic-runtime.json", runtime)
    plan.update(
        protocol=c.COLLECTION_PROTOCOL,
        phase="collection",
        profile_id="synthetic-4s4p",
        data_pool_id="collection/x500",
        mission_seconds=900,
        producer_args=["--configuration", "collection"],
        assessment_command=["fake-assess", "{run}", "{instance}", "{duration}", "{profile_id}"],
        assessment_code_paths=["capstone-sim"],
        preparation_assertions={"ready": True},
        quality_thresholds={"min_effective_fraction": 0.8, "min_effective_rtf": 0.1},
    )
    plan["budgets"].update(
        min_available_ram_bytes=4_000_000_000, memory_soft_cap_bytes=8_000_000_000
    )
    for lane in plan["lanes"]:
        lane["ports"] = [p + lane["instance"] for p in (14540, 14580, 18570, 14280, 13030)]
    for case, duration in zip(cases, (180, 480, 900)):
        case["mission_seconds"] = duration
    cases[0]["scenario"] = "low_soc_normal"
    preparation = tmp / "preparation.json"
    dump(preparation, {"ready": True})
    plan["producer_preparation"] = dict(path=str(preparation), sha256=c.sha(preparation))
    isolation = Path(plan["isolation_evidence"]["path"])
    value = c.read(isolation)
    value["lanes"] = plan["lanes"]
    value["source_hashes"] = plan["source_hashes"]
    dump(isolation, value)
    plan["isolation_evidence"]["sha256"] = c.sha(isolation)
    receipt = tmp / "receipt.json"
    benchmark = tmp / "benchmark.json"
    proof = dict(
        profile_id=plan["profile_id"],
        source_hashes=plan["source_hashes"],
        collector_sha256=c.sha(c.__file__),
        driver_sha256="synthetic",
        producer_preparation=plan["producer_preparation"],
    )
    dump(benchmark, dict(proof, status="BENCHMARK_PASSED", eligible_concurrency=[2, 4, 8]))
    dump(
        receipt,
        dict(
            proof,
            status="ENDURANCE_PASSED",
            selected_concurrency=2,
            benchmark_receipt=dict(path=str(benchmark), sha256=c.sha(benchmark)),
        ),
    )
    plan["gate_receipt"] = dict(path=str(receipt), sha256=c.sha(receipt))
    original = helper.command

    def command(args, cwd, log, check, seconds):
        if args[0] != "fake-assess":
            return original(args, cwd, log, check, seconds)
        check()
        run, instance, duration, profile = args[1:]
        dump(
            log,
            dict(
                schema="drone.run-assessment/1",
                run_id=Path(run).name,
                instance=int(instance),
                gz_partition="x500_" + Path(run).name,
                profile_id=profile,
                contract_passed=True,
                instance_isolation_passed=True,
                missing_data_passed=True,
                physical_gate_passed=True,
                observed_valid_seconds=int(duration) * 0.9,
            ),
        )

    helper.command = command
    return plan, cases, helper, tmp


@pytest.mark.parametrize("mutation", [None, "missing", "wrong", "legacy", "extra"])
def test_collection_runtime_checks_active_plugin_not_preservation_inventory(
    collection_setup, mutation
):
    plan, _, _, _ = collection_setup
    runtime = c.read(Path(plan["root"]) / "synthetic-runtime.json")
    plugin = "build/current/libDronePowertrain.so"
    if mutation == "missing":
        runtime.pop(plugin)
    elif mutation == "wrong":
        runtime[plugin] = "0" * 64
    elif mutation == "legacy":
        runtime.pop(plugin)
        runtime["build/libDronePowertrain.so"] = plan["source_hashes"][
            "build/libDronePowertrain.so"
        ]
    elif mutation == "extra":
        runtime["unexpected.so"] = "0" * 64
    source = dict(local_sources={}, runtime_binary_sha256=runtime)
    if mutation is None:
        c.verify_run_source_snapshot(plan, source)
    else:
        with pytest.raises(c.Stop, match="runtime binary snapshot mismatch"):
            c.verify_run_source_snapshot(plan, source)


def test_runtime_requires_exact_pinned_transport_and_seal(collection_setup):
    plan, _, _, _ = collection_setup
    runtime = c.read(Path(plan["root"]) / "synthetic-runtime.json")
    transport = "runtime/gazebo-transport/lib/libgz-transport13.so.13.6.0"
    plan["source_hashes"][transport] = "a" * 64
    source = dict(local_sources={}, runtime_binary_sha256=dict(runtime, **{transport: "a" * 64}))
    with pytest.raises(c.Stop, match="requires the installed producer"):
        c.verify_run_source_snapshot(plan, source)
    plan["source_hashes"]["artifacts/runtime-manifest.json"] = "b" * 64
    c.verify_run_source_snapshot(plan, source)
    source["runtime_binary_sha256"].pop(transport)
    with pytest.raises(c.Stop, match="runtime binary snapshot mismatch"):
        c.verify_run_source_snapshot(plan, source)


def binding_files(plan, cases, tmp):
    pp, mp, ap = [tmp / x for x in ("plan.json", "manifest.json", "approval.json")]
    dump(pp, plan)
    dump(mp, {"cases": cases, "data_pool_id": plan.get("data_pool_id")})
    dump(
        ap,
        dict(
            authorized=True,
            action="START_LONG_COLLECTION",
            plan_sha256=c.sha(pp),
            manifest_sha256=c.sha(mp),
            collector_sha256=c.sha(c.__file__),
        ),
    )
    return pp, mp, ap


def test_collection_mixed_duration_and_measured_rtf(collection_setup):
    plan, cases, helper, tmp = collection_setup
    plan, cases = c.bound(*binding_files(plan, cases, tmp))
    result = c.execute(plan, cases, helper)
    assert result["status"] == "COLLECTED_PENDING_QUALITY_STOPPED"
    for row, duration in zip(result["cases"].values(), (180, 480, 900)):
        # Compare by declared case, since threads can finish in either order.
        actual = row["case"]["mission_seconds"]
        assert row["assessment"]["observed_valid_seconds"] == actual * 0.9
        assert (
            row["assessment"]["effective_rtf"] == actual * 0.9 / row["timings_s"]["flight_command"]
        )
    assert result["resources"]["samples"] >= 1


def final_operating_cases(plan, cases):
    plan.update(
        collection_profile="final324", dataset_purpose="locked_final_test", mission_seconds=480
    )
    for index, case in enumerate(cases):
        case.update(
            scenario="battery_resistance_ramp" if index == 0 else "hover_normal",
            role="anomaly_test" if index == 0 else "normal_test",
            mission_seconds=180 if index == 0 else 480,
            operation_variation=dict(
                soc=0.45 if index == 0 else 0.9,
                temperature=298.15,
                payload=0,
                wind_e=0,
                wind_n=0,
                gust_sigma=0,
                path=dict(
                    kind="altitude",
                    height=3.5,
                    radius=1,
                    omega=0.06,
                    vertical_amplitude=0.4,
                    yaw_amplitude=0.3,
                    phase=1,
                    heading=0,
                ),
            ),
        )
        if index == 0:
            case["fault_after_mission_s"] = 100


def test_final_low_soc_anomaly_binds_and_producer_receives_operating_request(collection_setup):
    plan, cases, helper, tmp = collection_setup
    final_operating_cases(plan, cases)
    plan, cases = c.bound(*binding_files(plan, cases, tmp))
    result = c.execute(plan, cases, helper)
    assert result["status"] == "COLLECTED_PENDING_QUALITY_STOPPED"
    starts = [
        json.loads(line) for line in (Path(plan["root"]) / "events.jsonl").read_text().splitlines()
    ]
    starts = {row["seed"]: row["args"] for row in starts if row["event"] == "start"}
    for case in cases:
        args = starts[case["seed"]]
        assert (
            json.loads(args[args.index("--operation-variation") + 1]) == case["operation_variation"]
        )
        assert "--normal-variation-json" not in args and "--fault-at-s" not in args
        if case["role"] == "anomaly_test":
            assert args[args.index("--fault-after-mission-s") + 1] == "100"
        else:
            assert "--fault-after-mission-s" not in args


@pytest.mark.parametrize(
    "issue",
    [
        "wrong_profile",
        "wrong_purpose",
        "wrong_role",
        "missing_variation",
        "health_override",
        "conflicting_variation",
        "conflicting_clocks",
        "low_soc_long",
        "high_soc_short",
        "nonfinite_soc",
        "missing_onset",
        "late_onset",
        "boolean_onset",
        "normal_onset",
    ],
)
def test_final_operating_request_rejects_unsafe_or_ambiguous_conditions(collection_setup, issue):
    plan, cases, _, tmp = collection_setup
    final_operating_cases(plan, cases)
    case = cases[0]
    if issue == "wrong_profile":
        plan["collection_profile"] = "balanced300"
    elif issue == "wrong_purpose":
        plan["dataset_purpose"] = "development"
    elif issue == "wrong_role":
        case["role"] = "anomaly_dev"
    elif issue == "missing_variation":
        case.pop("operation_variation")
    elif issue == "health_override":
        case["operation_variation"]["r0"] = 0.01
    elif issue == "conflicting_variation":
        case["normal_variation"] = case["operation_variation"]
    elif issue == "conflicting_clocks":
        case["fault_at_s"] = 180
    elif issue == "low_soc_long":
        case["mission_seconds"] = 480
    elif issue == "high_soc_short":
        case["operation_variation"]["soc"] = 0.9
    elif issue == "nonfinite_soc":
        case["operation_variation"]["soc"] = float("nan")
    elif issue == "missing_onset":
        case.pop("fault_after_mission_s")
    elif issue == "late_onset":
        case["fault_after_mission_s"] = 180
    elif issue == "boolean_onset":
        case["fault_after_mission_s"] = True
    else:
        case["scenario"] = "hover_normal"
    with pytest.raises(c.Stop):
        c.bound(*binding_files(plan, cases, tmp))


@pytest.mark.parametrize("field", ["operation_variation", "fault_after_mission_s"])
def test_final_producer_cannot_silently_ignore_requested_conditions(collection_setup, field):
    plan, cases, helper, tmp = collection_setup
    final_operating_cases(plan, cases)
    cases = cases[:1]
    plan["attempts_max"] = 1
    plan, cases = c.bound(*binding_files(plan, cases, tmp))
    original = helper.command

    def changed_metadata(args, cwd, log, check, seconds):
        original(args, cwd, log, check, seconds)
        if args[0].endswith("capstone-sim") and args[1] == "run":
            run = Path(c.read(log)["run"])
            path = run / "evaluation/scenario.json"
            value = c.read(path)
            value.pop(field)
            dump(path, value)

    helper.command = changed_metadata
    result = c.execute(plan, cases, helper)
    assert result["status"] == "FAILED_DRAINED_STOPPED"
    assert "conditions/fault schedule mismatch" in str(result["stop_reason"])


@pytest.mark.parametrize("lanes", [4, 8])
def test_collection_four_and_eight_lanes_bind(collection_setup, lanes):
    plan, cases, helper, tmp = collection_setup
    plan["concurrency"] = lanes
    plan["lanes"] = [
        dict(instance=i, ports=[p + i for p in (14540, 14580, 18570, 14280, 13030)])
        for i in range(lanes)
    ]
    iso = Path(plan["isolation_evidence"]["path"])
    value = c.read(iso)
    value["lanes"] = plan["lanes"]
    dump(iso, value)
    plan["isolation_evidence"]["sha256"] = c.sha(iso)
    receipt = Path(plan["gate_receipt"]["path"])
    value = c.read(receipt)
    value["selected_concurrency"] = lanes
    dump(receipt, value)
    plan["gate_receipt"]["sha256"] = c.sha(receipt)
    assert c.bound(*binding_files(plan, cases, tmp))[0]["concurrency"] == lanes


@pytest.mark.parametrize(
    "issue", ["old_receipt", "missing_preparation", "wrong_port", "test_leak", "bad_assessment"]
)
def test_collection_rejects_missing_or_wrong_evidence(collection_setup, issue):
    plan, cases, helper, tmp = collection_setup
    if issue == "old_receipt":
        plan["gate_receipt"]["sha256"] = "0" * 64
    elif issue == "missing_preparation":
        Path(plan["producer_preparation"]["path"]).unlink()
    elif issue == "wrong_port":
        plan["lanes"][0]["ports"] = [14543]
    elif issue == "test_leak":
        cases[1].update(role="normal_test", group_id=cases[0]["group_id"])
    else:
        original = helper.command

        def bad(args, *rest):
            original(args, *rest)
            if args[0] == "fake-assess":
                path = rest[1]
                value = c.read(path)
                value["physical_gate_passed"] = False
                dump(path, value)

        helper.command = bad
        result = c.execute(plan, cases, helper)
        assert result["status"] == "FAILED_DRAINED_STOPPED"
        assert "physical_gate_passed" in result["stop_reason"][0]
        return
    with pytest.raises((c.Stop, FileNotFoundError)):
        c.bound(*binding_files(plan, cases, tmp))


@pytest.mark.parametrize("count", [4, 8])
def test_collection_parallel_synthetic_commands_are_isolated(collection_setup, count):
    plan, cases, helper, tmp = collection_setup
    plan["concurrency"] = count
    plan["attempts_max"] = count
    plan["lanes"] = [
        dict(instance=i, ports=[p + i for p in (14540, 14580, 18570, 14280, 13030)])
        for i in range(count)
    ]
    cases = [
        dict(
            case_id=f"collection-{i}",
            scenario="hover_normal",
            seed=100 + i,
            role="normal_train",
            group_id=f"collection-{i}",
            mission_seconds=900,
        )
        for i in range(count)
    ]
    result = c.execute(plan, cases, helper)
    assert result["status"] == "COLLECTED_PENDING_QUALITY_STOPPED"
    rows = list(result["cases"].values())
    assert len({row["lane"]["instance"] for row in rows}) == count
    assert len({row["run_id"] for row in rows}) == count
    assert max(row["flight_started_monotonic"] for row in rows) < min(
        row["flight_finished_monotonic"] for row in rows
    )
    assert all(row["assessment"]["instance"] == row["lane"]["instance"] for row in rows)


def test_collection_benchmark_cannot_use_collection_entrypoint(collection_setup):
    plan, cases, helper, tmp = collection_setup
    plan.update(phase="benchmark", mission_seconds=120)
    for row in cases:
        row.update(role="benchmark", mission_seconds=120)
    paths = binding_files(plan, cases, tmp)
    assert c.bound(*paths, dry_run=True)[0]["phase"] == "benchmark"
    with pytest.raises(c.Stop, match="finite gate driver"):
        c.bound(*paths)


def test_collection_benchmark_failure_stops_active_peer(collection_setup):
    plan, cases, helper, tmp = collection_setup
    plan.update(phase="benchmark", mission_seconds=120)
    for row in cases:
        row.update(role="benchmark", mission_seconds=120)
    cases[0]["scenario"] = "fail"
    result = c.execute(plan, cases, helper)
    assert result["status"] == "FAILED_DRAINED_STOPPED"
    assert result["not_started"] == ["c3"]
    assert result["cases"]["c2"]["status"] == "FAILED_PRESERVED"
    assert "Benchmark failure" in result["cases"]["c2"]["error"]
    assert not list(Path(plan["external_output"]).glob("*.tar.gz"))
    assert result["training_eligible"] is False


def test_project_lock_blocks_direct_execute_before_processes(setup):
    plan, cases, helper, tmp = setup
    lock = Path(plan["root"]) / "Data/operations/collection.lock"
    lock.parent.mkdir(parents=True)
    with lock.open("a") as stream:
        c.fcntl.flock(stream, c.fcntl.LOCK_EX | c.fcntl.LOCK_NB)
        with pytest.raises(c.Stop, match="Project collection lock"):
            c.execute(plan, cases, helper)
    assert not (Path(plan["root"]) / "events.jsonl").exists()
    assert not Path(plan["execution_dir"]).exists()
    assert lock.exists()  # Never remove a shared lock inode.


def test_size_accounting_tolerates_atomic_status_replacement(tmp_path, monkeypatch):
    vanished = tmp_path / "atomic-temp"
    monkeypatch.setattr(c, "files", lambda *a, **kw: iter([(vanished, c.stat.S_IFREG)]))
    assert c.bytes_under(tmp_path) == 0


@pytest.fixture
def mission_progress_fixture(tmp_path):
    from datetime import datetime, timezone

    root = tmp_path / "progress-repo"
    run = root / "runs" / ("a" * 32)
    now = datetime(2026, 10, 15, 0, 0, tzinfo=timezone.utc)
    case = dict(scenario="hover_normal", seed=42)
    state = dict(
        run_id=run.name,
        instance=3,
        configuration="collection",
        gz_partition="x500_" + run.name,
        clock=dict(id="px4-boot", epoch="simulation_boot", unit="us"),
        progress=dict(
            sim_elapsed_s=10, target_s=900, px4_boot_s=50, last_update_utc=now.isoformat()
        ),
    )
    dump(run / "run.json", state)
    dump(run / "evaluation/scenario.json", dict(name=case["scenario"], seed=case["seed"]))
    tracker = c.MissionProgress(root, set(), case, {"instance": 3}, 900)
    return tracker, run, state, now


def test_live_progress_uuid_cache_and_measured_interval_rtf(mission_progress_fixture, monkeypatch):
    from datetime import timedelta

    tracker, run, state, now = mission_progress_fixture
    seen = []
    original = tracker.small_json
    monkeypatch.setattr(
        tracker, "small_json", lambda path: seen.append(path.name) or original(path)
    )
    first = tracker.sample(now, 100)
    assert first["observed_simulation_elapsed_s"] == 10
    assert first["live_rtf"] is None
    assert first["producer_run_id"] == run.name
    assert seen.count("scenario.json") == 1
    state["progress"].update(
        sim_elapsed_s=12, px4_boot_s=52, last_update_utc=(now + timedelta(seconds=1)).isoformat()
    )
    dump(run / "run.json", state)
    second = tracker.sample(now + timedelta(seconds=1), 101)
    assert second["live_rtf"] == 2
    assert second["producer_target_s"] == 900
    assert seen.count("scenario.json") == 1  # Cached identity: only tiny run.json reread.
    assert tracker.sample(now + timedelta(seconds=2), 102)["live_rtf"] is None


def test_failed_producer_retains_uuid_and_landing_reason(setup):
    plan, cases, helper, tmp = setup
    plan.update(concurrency=1)
    plan["lanes"] = plan["lanes"][:1]
    run_id = "a" * 32
    run = Path(plan["root"]) / "runs" / run_id

    def command(args, cwd, log, check, seconds):
        dump(
            run / "run.json",
            dict(
                run_id=run_id,
                instance=3,
                configuration="collection",
                gz_partition="x500_" + run_id,
                status="failed",
                termination_reason="landing timeout",
                primary_failure=dict(stage="landing", reason="landing timeout"),
            ),
        )
        dump(
            run / "evaluation/scenario.json",
            dict(name="hover_normal", seed=1, requested_duration_s=1200),
        )
        raise c.Stop("Command exit 4: synthetic producer")

    helper.command = command
    result = c.execute(plan, cases[:1], helper)
    record = result["cases"]["c1"]
    assert result["training_eligible"] is False
    assert record["status"] == "FAILED_PRESERVED"
    assert record["run_id"] == record["producer_run_id"] == run_id
    assert record["producer_stage"] == "landing"
    assert record["producer_termination_reason"] == "landing timeout"
    assert "Command exit 4" in record["error"] and "landing timeout" in record["error"]
    assert record["attempt_history"][0]["run_id"] == run_id
    assert not list(Path(plan["external_output"]).glob("*.tar.gz"))


@pytest.mark.parametrize(
    "issue", [None, "seed", "duration", "instance", "partition", "duplicate", "symlink"]
)
def test_terminal_evidence_rechecks_identity_without_live_clock_requirement(
    mission_progress_fixture, issue
):
    tracker, run, state, now = mission_progress_fixture
    scenario = dict(name="hover_normal", seed=42, requested_duration_s=900)
    state.update(
        status="failed",
        termination_reason="landing timeout",
        primary_failure=dict(stage="landing", reason="landing timeout"),
    )
    dump(run / "run.json", state)
    dump(run / "evaluation/scenario.json", scenario)
    tracker.sample(now, 100)  # Cache the original valid identity before metadata changes.
    if issue == "seed":
        scenario["seed"] = 43
    elif issue == "duration":
        scenario["requested_duration_s"] = 1200
    elif issue == "instance":
        state["instance"] = 4
    elif issue == "partition":
        state["gz_partition"] = "another-run"
    elif issue == "duplicate":
        other = run.parent / ("b" * 32)
        dump(other / "run.json", dict(state, run_id=other.name, gz_partition="x500_" + other.name))
        dump(other / "evaluation/scenario.json", scenario)
    dump(run / "run.json", state)
    dump(run / "evaluation/scenario.json", scenario)
    if issue == "symlink":
        (run / "run.json").rename(run / "saved.json")
        (run / "run.json").symlink_to(run / "saved.json")
    evidence = tracker.terminal_evidence()
    if issue is None:
        assert evidence["producer_run_id"] == run.name
        assert evidence["producer_stage"] == "landing"
    else:
        assert evidence == {}


@pytest.mark.parametrize(
    "issue",
    [
        "instance",
        "configuration",
        "clock",
        "seed",
        "target",
        "stale",
        "future",
        "nonfinite",
        "duplicate",
        "baseline",
    ],
)
def test_live_progress_rejects_ambiguous_or_wrong_evidence(mission_progress_fixture, issue):
    from datetime import timedelta

    tracker, run, state, now = mission_progress_fixture
    if issue == "instance":
        state["instance"] = 4
    elif issue == "configuration":
        state["configuration"] = "old"
    elif issue == "clock":
        state["clock"]["id"] = "other-clock"
    elif issue == "seed":
        dump(run / "evaluation/scenario.json", dict(name="hover_normal", seed=99))
    elif issue == "target":
        state["progress"]["target_s"] = 300
    elif issue == "stale":
        state["progress"]["last_update_utc"] = (now - timedelta(seconds=11)).isoformat()
    elif issue == "future":
        state["progress"]["last_update_utc"] = (now + timedelta(seconds=1)).isoformat()
    elif issue == "nonfinite":
        state["progress"]["sim_elapsed_s"] = float("nan")
    elif issue == "baseline":
        tracker.baseline = {run.name}
    else:
        assert tracker.sample(now, 100)["producer_run_id"] == run.name
        other = run.parent / ("b" * 32)
        duplicate = dict(state, run_id=other.name, gz_partition="x500_" + other.name)
        dump(other / "run.json", duplicate)
        dump(other / "evaluation/scenario.json", dict(name="hover_normal", seed=42))
    dump(run / "run.json", state)
    result = tracker.sample(now, 101)
    assert result["observed_simulation_elapsed_s"] is None
    assert result["live_rtf"] is None


def test_live_progress_clock_regression_and_initial_metadata(mission_progress_fixture):
    from datetime import timedelta

    tracker, run, state, now = mission_progress_fixture
    initial = {k: v for k, v in state.items() if k not in {"instance", "configuration"}}
    dump(run / "run.json", initial)
    assert tracker.sample(now, 100)["observed_simulation_elapsed_s"] is None
    dump(run / "run.json", state)
    assert tracker.sample(now, 101)["observed_simulation_elapsed_s"] == 10
    state["progress"].update(
        sim_elapsed_s=9, last_update_utc=(now + timedelta(seconds=1)).isoformat()
    )
    dump(run / "run.json", state)
    assert tracker.sample(now + timedelta(seconds=1), 102)["observed_simulation_elapsed_s"] is None


def test_continue_policy_attempts_next_case_and_never_reports_success(setup):
    plan, cases, helper, tmp = setup
    plan["continue_after_case_failure"] = True
    cases[0]["scenario"] = "fail"
    result = c.execute(plan, cases, helper)
    assert result["status"] == "COLLECTED_WITH_ERRORS_STOPPED"
    assert result["not_started"] == []
    assert result["failed_cases"] == ["c1"]
    assert result["cases"]["c1"]["status"] == "FAILED_PRESERVED"
    assert (
        result["cases"]["c2"]["status"]
        == result["cases"]["c3"]["status"]
        == "ARCHIVED_PENDING_QUALITY"
    )
    events = [json.loads(x) for x in (Path(plan["root"]) / "events.jsonl").read_text().splitlines()]
    assert sorted(x["seed"] for x in events if x["event"] == "start") == [1, 2, 3]
    assert any(p.name == "partial" for p in (Path(plan["root"]) / "runs").rglob("*"))


def test_continue_policy_still_stops_on_shared_resource_failure(setup, monkeypatch):
    plan, cases, helper, tmp = setup
    plan["continue_after_case_failure"] = True
    monkeypatch.setattr(
        c, "memory_snapshot", lambda: dict(available_bytes=0, owned_tree_rss_bytes=0)
    )
    result = c.execute(plan, cases, helper)
    assert result["status"] == "FAILED_DRAINED_STOPPED"
    assert result["not_started"] == ["c1", "c2", "c3"]


@pytest.mark.parametrize("failures", [2, 3])
def test_bounded_case_retries_preserve_raw_and_continue(setup, failures):
    plan, cases, helper, tmp = setup
    plan.update(unit_attempts_max=3, continue_after_case_failure=True, concurrency=1)
    plan["lanes"] = plan["lanes"][:1]
    command = helper.command
    calls = []

    def failing(args, cwd, log, check, seconds):
        command(args, cwd, log, check, seconds)
        if args[1] == "run":
            seed = int(args[args.index("--seed") + 1])
            calls.append(seed)
            if seed == 1 and calls.count(1) <= failures:
                raise c.Stop("synthetic transient after raw written")

    helper.command = failing
    result = c.execute(plan, cases, helper)
    assert calls == [1, 1, 1, 2, 3]
    assert result["cases"]["c1"]["attempt"] == 3
    assert len(list((Path(plan["root"]) / "runs").iterdir())) == 5
    assert len(list(Path(plan["external_output"]).glob("*.tar.gz"))) == (3 if failures == 2 else 2)
    assert result["status"] == (
        "COLLECTED_PENDING_QUALITY_STOPPED" if failures == 2 else "COLLECTED_WITH_ERRORS_STOPPED"
    )
    for folder in ("c1", "c1/attempt-02", "c1/attempt-03"):
        proof = json.loads(
            (Path(plan["execution_dir"]) / folder / "attempt-result.json").read_text()
        )
        assert proof["status"] in {"FAILED_PRESERVED", "ARCHIVED_PENDING_QUALITY"}
    assert result["cases"]["c1"]["previous_attempts"][0]["attempt"] == 1


def test_shared_storage_failure_never_retries(setup):
    import errno

    plan, cases, helper, tmp = setup
    plan.update(unit_attempts_max=3, continue_after_case_failure=True, concurrency=1)
    plan["lanes"] = plan["lanes"][:1]
    calls = []

    def broken(*args):
        calls.append(1)
        raise OSError(errno.ENOSPC, "storage full")

    helper.command = broken
    result = c.execute(plan, cases, helper)
    assert len(calls) == 1
    assert result["status"] == "FAILED_DRAINED_STOPPED"
    assert result["not_started"] == ["c2", "c3"]


def test_unsafe_lane_is_not_retried(setup, monkeypatch):
    plan, cases, helper, tmp = setup
    plan.update(unit_attempts_max=3, continue_after_case_failure=True, concurrency=1)
    plan["lanes"] = plan["lanes"][:1]
    calls = []

    def broken(*args):
        calls.append(1)
        raise c.Stop("flight error")

    helper.command = broken

    def occupied(instance):
        raise c.Stop("unowned socket preserved")

    monkeypatch.setattr(c, "check_instance_socket", occupied)
    result = c.execute(plan, cases, helper)
    assert calls == [1]
    assert result["cases"]["c1"]["lane_retired"] == "unowned socket preserved"


@pytest.mark.parametrize("limit", [0, 4, True, "3"])
def test_invalid_retry_limit_is_rejected(setup, limit):
    plan, cases, helper, tmp = setup
    plan["unit_attempts_max"] = limit
    with pytest.raises(c.Stop, match="attempt limit"):
        c.execute(plan, cases, helper)


def test_retry_limit_requires_matching_task_config(setup):
    plan, cases, helper, tmp = setup
    plan["unit_attempts_max"] = 3
    approval = tmp / "task-config.json"
    dump(approval, dict(unit_attempts_max=1))
    plan["pretrain_cli_approval"] = str(approval)
    with pytest.raises(c.Stop, match="Retry policy"):
        c.bound(*binding_files(plan, cases, tmp))
    dump(approval, dict(unit_attempts_max=3))
    assert c.bound(*binding_files(plan, cases, tmp))[0]["unit_attempts_max"] == 3
