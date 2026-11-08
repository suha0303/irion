"""Finite fixture tests; no native flight, model, fit or service submission."""

import copy
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_ai19 import mission, raw_fixture, row

from pretrain import ai19, common, pilot19, pilot45


@pytest.mark.parametrize("seed", [0, 1, 2**32 - 1])
def test_frozen45_composition_and_independent_pair_members(seed):
    cases = pilot45.build_cases(seed)
    assert len(cases) == 45
    assert Counter(c["role"] for c in cases) == {
        "normal_test": 36,
        "anomaly_test": 9,
    }
    assert Counter(
        c["environment"] for c in cases if c["category"] == "anomaly"
    ) == {
        "calm": 3,
        "low": 3,
        "strong": 3,
    }
    assert len({c["case_id"] for c in cases}) == 45
    for anomaly in (c for c in cases if c["category"] == "anomaly"):
        control = next(
            c
            for c in cases
            if c["group_id"] == anomaly["group_id"]
            and c["category"] == "paired_normal"
        )
        assert anomaly["case_id"] != control["case_id"]
        assert anomaly["retry_family_id"] != control["retry_family_id"]
        assert anomaly["operation_variation"] == control["operation_variation"]
        assert anomaly["seed"] == control["seed"]
    broken = copy.deepcopy(cases)
    broken[-1]["operation_variation"]["soc"] = 0.5
    with pytest.raises(ValueError):
        pilot45.validate_cases(broken)


@pytest.mark.parametrize(
    "mutation", ["duration", "role", "cohort", "attempts_max"]
)
def test_manifest_rejects_unapproved_changes(mutation):
    cases = pilot45.build_cases(1234)
    cases[0][mutation] = "changed"
    with pytest.raises(ValueError):
        pilot45.validate_cases(cases)


def test_past_seed_or_pair_cannot_reenter():
    case = pilot45.build_cases(1)[0]
    for previous in [
        dict(scenario=case["scenario"], seed=case["seed"]),
        dict(paired_case_id=case["paired_case_id"]),
    ]:
        with pytest.raises(pilot45.Blocked):
            pilot45.fresh_cases([case], dict(records=[previous]))


@pytest.mark.parametrize(
    "number,stopped,reason,expected",
    [
        (
            1,
            False,
            "recording_stalled: PX4 clock advanced but raw capture did not",
            True,
        ),
        (
            2,
            False,
            "recording_stalled: PX4 clock advanced but raw capture did not",
            True,
        ),
        (
            3,
            False,
            "recording_stalled: PX4 clock advanced but raw capture did not",
            False,
        ),
        (
            1,
            True,
            "recording_stalled: PX4 clock advanced but raw capture did not",
            False,
        ),
        (1, False, "SIGSEGV", False),
        (1, False, "semantic gate", False),
        (1, False, "resource exhausted", False),
    ],
)
def test_retry_is_bounded_and_not_for_semantic_resource_or_user_stop(
    number, stopped, reason, expected
):
    assert (
        pilot45.retryable(dict(status="failed", reason=reason), number, stopped)
        is expected
    )


def init_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot45, "HOME", tmp_path)
    monkeypatch.setattr(pilot45, "authorization", lambda: {})
    monkeypatch.setattr(pilot45, "protection", lambda: dict(records=[]))
    monkeypatch.setattr(
        pilot45,
        "global_protection",
        lambda *args: dict(schema="drone-ai.global-protection/1", entries=[]),
    )
    monkeypatch.setattr(
        pilot45,
        "estimate",
        lambda: dict(per_case_conservative_increment_bytes=1),
    )
    monkeypatch.setattr(pilot45, "runtime_identity", lambda: {})
    monkeypatch.setattr(pilot45, "capture", lambda p: {})
    monkeypatch.setattr(pilot45, "frozen_sources", lambda *args: {})
    monkeypatch.setattr(
        ai19,
        "producer_binding",
        lambda: {
            k: "fixture"
            for k in (
                "battery_policy",
                "battery_revision",
                "catalog_sha256",
                "propulsion_revision",
                "propulsion_catalog_sha256",
                "runtime_sha256",
                "source_manifest_sha256",
            )
        },
    )


@pytest.mark.local_evidence
def test_init_single_scope_does_not_launch_and_os_seed_is_frozen(
    monkeypatch, tmp_path
):
    init_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(pilot45.secrets, "randbits", lambda n: 12345)
    task = pilot45.init("one", single=True)
    manifest, config = (
        common.read(task / "manifest.json"),
        common.read(task / "config.json"),
    )
    assert (
        manifest["seed_origin"] == "OS_random"
        and manifest["seed_base"] == 12345
    )
    assert len(pilot45.validate_manifest(manifest)) == 1
    assert config["planned_cases"] == config["unit_attempts_max"] == 1
    assert config["paired_run_options"] == {} and not config["fit_authorized"]
    assert not (task / "operator/service.json").exists()
    assert common.read(task / "ledger.json")["attempts"] == []
    with pytest.raises(ValueError, match="Existing task"):
        pilot45.init("one", single=True)
    manifest["cases"][0]["attempts_max"] = 3
    with pytest.raises(ValueError, match="one healthy"):
        pilot45.validate_manifest(manifest)


@pytest.mark.local_evidence
def test_45_cannot_launch_with_missing_or_fake_passed_receipts(
    monkeypatch, tmp_path
):
    init_fixture(monkeypatch, tmp_path)
    task = pilot45.init("blocked45", seed=7)
    assert common.read(task / "manifest.json")["seed_origin"] == "user"
    fake = tmp_path / "fake.json"
    common.write(fake, dict(status="PASSED", passed=True))
    for receipt in (tmp_path / "missing", fake):
        with pytest.raises(pilot45.Blocked, match="paired6"):
            pilot45.seal(task, receipt, receipt)
        with pytest.raises(pilot45.Blocked, match="remains blocked"):
            pilot45.start(task)
    assert not (task / "operator/service.json").exists()


def test_resource_floor_uses_actual_available_ram_and_disk(monkeypatch):
    module = SimpleNamespace(
        memory_snapshot=lambda: dict(available_bytes=1_100_000_000)
    )
    monkeypatch.setattr(pilot45, "collector", lambda: module)
    monkeypatch.setattr(
        pilot45.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=51_000_000_000),
    )
    assert pilot45.resources()["available_bytes"] < 8_000_000_000
    module.memory_snapshot = lambda: dict(available_bytes=999_999_999)
    with pytest.raises(pilot45.Blocked, match="RAM"):
        pilot45.resources()
    module.memory_snapshot = lambda: dict(available_bytes=2_000_000_000)
    with pytest.raises(pilot45.Blocked, match="Disk"):
        pilot45.resources(1, 2_000_000_000)


def test_global_registry_does_not_allow_cross_task_lineage_reuse(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(pilot45, "REGISTRY", tmp_path / "registry.jsonl")
    event = dict(
        run_id="fixture", paired_case_id="pair", scenario="hover_normal", seed=7
    )
    pilot45.register(Path("one"), event)
    pilot45.register(
        Path("one"), event
    )  # Same task retains attempt/completion journal.
    common.write(tmp_path / "two/config.json", {})
    with pytest.raises(pilot45.Blocked, match="reuse"):
        pilot45.register(tmp_path / "two", event)
    assert (
        len(
            (tmp_path / "registry.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 2
    )


def test_completed_raw_is_retained_after_wrapper_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot45, "ROOT", tmp_path)
    monkeypatch.setattr(pilot45, "register", lambda *args: None)
    task, raw = tmp_path / "task", tmp_path / "simulation/runs/synthetic"
    case = pilot45.build_cases(5)[0]
    folder = task / "pilot/execution" / case["case_id"]
    folder.mkdir(parents=True)
    common.write(task / "config.json", {})
    common.write(
        raw / "run.json", dict(status="completed", parent_attempt_id=None)
    )
    common.write(
        raw / "evaluation/scenario.json",
        dict(
            name=case["scenario"],
            seed=case["seed"],
            operation_variation=case["operation_variation"],
        ),
    )

    def fail_child(task, request, log):
        common.write(
            common.read(request)["receipt"],
            dict(run=str(raw), status="completed"),
        )
        raise pilot45.Blocked("wrapper failed after raw was closed")

    monkeypatch.setattr(pilot45, "child", fail_child)
    entry, ledger = dict(attempts=[]), dict(attempts=[], cases={})
    ledger["cases"][case["case_id"]] = entry
    with pytest.raises(pilot45.Blocked, match="Completed raw retained"):
        pilot45.physical(task, case, entry, ledger, folder, set())
    assert entry["run"] == str(raw)
    assert len(ledger["attempts"]) == 1
    assert entry["attempts"][0]["state"] == "COMPLETED_RAW_CHILD_ERROR"
    pilot45.physical(task, case, entry, ledger, folder, set())
    assert (
        len(ledger["attempts"]) == 1
    )  # Resume processing never regenerates successful raw.


def test_one_run_has_no_automatic_retry_even_for_recorder_stall(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(pilot45, "ROOT", tmp_path)
    monkeypatch.setattr(pilot45, "register", lambda *args: None)
    task, raw = tmp_path / "task", tmp_path / "simulation/runs/failed-fixture"
    case = pilot45.build_cases(6)[0]
    case["attempts_max"] = 1
    folder = task / "pilot/execution" / case["case_id"]
    folder.mkdir(parents=True)
    common.write(task / "config.json", {})
    common.write(
        raw / "run.json", dict(status="failed", parent_attempt_id=None)
    )
    common.write(
        raw / "evaluation/scenario.json",
        dict(
            name=case["scenario"],
            seed=case["seed"],
            operation_variation=case["operation_variation"],
        ),
    )

    def fail_child(task, request, log):
        common.write(
            common.read(request)["receipt"],
            dict(
                run=str(raw),
                status="failed",
                reason=(
                    "recording_stalled: PX4 clock advanced but raw capture did "
                    "not"
                ),
            ),
        )
        raise pilot45.Blocked("fixture failure")

    monkeypatch.setattr(pilot45, "child", fail_child)
    entry, ledger = dict(attempts=[]), dict(attempts=[], cases={})
    ledger["cases"][case["case_id"]] = entry
    with pytest.raises(pilot45.Blocked, match="not blindly retried"):
        pilot45.physical(task, case, entry, ledger, folder, set())
    assert (
        len(ledger["attempts"]) == 1
        and not entry["attempts"][0]["retry_allowed"]
    )


def test_foreign_service_is_never_stopped(monkeypatch, tmp_path):
    task = tmp_path / "one"
    common.write(
        task / "config.json",
        dict(
            mode="pilot45",
            owner_uid=os.getuid(),
            service_unit="drone-pretrain-one.service",
        ),
    )
    common.write(
        task / "operator/service.json", dict(unit="drone-pretrain-one.service")
    )
    commands = []

    def show(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            stdout=(
                "MainPID=0\nActiveState="
                "inactive\nLoadState=loaded\nExecStart="
                "foreign task\n"
            )
        )

    monkeypatch.setattr(pilot45.subprocess, "run", show)
    with pytest.raises(pilot45.Blocked, match="does not belong"):
        pilot45.service_identity(task)
    assert (
        len(commands) == 1
        and "show" in commands[0]
        and "stop" not in commands[0]
    )


@pytest.mark.local_evidence
def test_standalone_worker_runs_one_case_then_stops_without_fit(
    monkeypatch, tmp_path
):
    init_fixture(monkeypatch, tmp_path)
    task = pilot45.init("standalone", 8, single=True)
    monkeypatch.setattr(pilot45, "ROOT", tmp_path)
    monkeypatch.setattr(pilot45, "STORE", tmp_path / "Data")
    monkeypatch.setattr(pilot45, "REPORT", tmp_path / "reports")
    monkeypatch.setattr(pilot45, "preflight", lambda *args: {})
    monkeypatch.setattr(pilot45, "resources", lambda *args: {})
    monkeypatch.setattr(pilot45, "register", lambda *args: None)
    (tmp_path / "Data/operations").mkdir(parents=True)
    actions = []
    raw = tmp_path / "raw"
    raw.mkdir()

    def physical(task, case, entry, ledger, folder, denied):
        actions.append("flight")
        attempt = dict(number=1, state="COMPLETED", parent_attempt_id=None)
        entry.update(run=str(raw), attempts=[attempt])
        ledger["attempts"].append(dict(case_id=case["case_id"], **attempt))

    def child(task, request, log):
        value = common.read(request)
        actions.append(value["action"])
        output = Path(value["output"])
        if value["action"] == "prepare":
            item = row(
                "raw",
                role="normal_test",
                cohort=pilot19.COHORT,
                purpose=pilot19.COHORT,
                raw_sha256="0" * 64,
            )
            item["lineage"] = ai19.lineage(item)
            result = dict(
                run=item,
                files={},
                consumer_bytes=0,
                online=dict(
                    available=True, valid_windows=10, target_valid_seconds=10
                ),
                run_health=dict(status="available", post_initial100_seconds=10),
            )
            common.write(output / "receipt.json", result)
            common.write(
                output / "seal.json",
                dict(receipt_sha256=common.sha(output / "receipt.json")),
            )
        else:
            result = dict(
                export_receipt_sha256=common.sha(output / "receipt.json")
            )
        common.write(value["receipt"], result)

    monkeypatch.setattr(pilot45, "physical", physical)
    monkeypatch.setattr(pilot45, "child", child)
    pilot45.worker(task)
    assert actions == ["flight", "prepare", "audit"]
    quality = common.read(task / "quality.json")
    assert quality["status"] == "INTEGRATION19_QUALITY_PASSED_STOPPED"
    assert (
        quality["planned_cases"]
        == quality["physical_attempts"]
        == quality["prepared_and_audited"]
        == 1
    )
    assert (
        not quality["training_started"] and not quality["next_stage_authorized"]
    )


def test_fixture_archive_shared_frames_two_views_and_independent_audit(
    monkeypatch, tmp_path
):
    # Only admission and weather evidence are synthetic here.
    # Archive/extraction,
    # the causal AI engine, both actual readers and numeric audit execute
    # unchanged.
    raw = raw_fixture(tmp_path / "raw", rid="raw")
    mapping, _, _, _ = ai19.ai_modules()
    import learning.admission19 as admission
    import learning.wind_metadata as wind

    common.package_imports()
    from pretrain import archive

    store = tmp_path / "store"
    monkeypatch.setattr(pilot19, "STORE", store)
    monkeypatch.setattr(archive, "STORAGE", store)
    monkeypatch.setattr(archive, "storage_path", Path)
    monkeypatch.setattr(ai19, "external", Path)
    binding = dict(
        source_hashes={}, vehicle_profile_revision=ai19.PROFILE_REVISION
    )
    binding.update(
        battery_policy=ai19.POLICY,
        battery_revision="cmu-vah01-baseline/1",
        propulsion_revision="x500-propulsion-binary-v1",
    )
    binding.update(
        {
            key: "0" * 64
            for key in (
                "catalog_sha256",
                "propulsion_catalog_sha256",
                "runtime_sha256",
                "source_manifest_sha256",
            )
        }
    )
    monkeypatch.setattr(ai19, "producer_binding", lambda: binding)
    selection = row(
        raw.name, role="normal_test", path=str(raw), cohort=pilot19.COHORT
    )
    interval = mission(2_050_000, 22_370_000)
    interval["native_sources_sha256"] = {
        "evaluation/joint-origin.json": "1" * 64,
        "evaluation/joint-landing.json": "2" * 64,
    }
    monkeypatch.setattr(
        ai19, "admit", lambda s, b: (dict(s), dict(mission_interval=interval))
    )
    monkeypatch.setattr(pilot19, "source_evidence", lambda *args: {})
    monkeypatch.setattr(admission, "source_evidence", lambda *args: None)
    monkeypatch.setattr(
        wind, "describe", lambda *args: dict(wind_mode="bounded_profile")
    )
    expected, protected = (
        tmp_path / "expected.json",
        tmp_path / "protection.json",
    )
    common.write(expected, binding)
    common.write(
        protected, dict(schema="drone-ai.global-protection/1", entries=[])
    )
    output = store / "processed/case/attempt-01"
    result = pilot19.prepare_one(
        selection,
        output,
        expected,
        tmp_path / "case/archive-cache.json",
        protected,
    )
    checked = pilot19.audit_one(output, expected)
    assert (
        common.read(output / "online.json")["run"]["mission_interval"]
        == interval
    )
    assert (
        common.read(output / "run.json")["runs"][0]["mission_interval"]
        == interval
    )
    assert (
        result["run"]["mission_interval"]["native_sources_sha256"]
        == interval["native_sources_sha256"]
    )
    assert checked["online_available"] and checked["run_available"]
    assert (
        checked["mission_frames"] == 203
        and checked["mission_frames"] % 100 == 3
    )
    with np.load(output / "run.npz", allow_pickle=False) as a:
        assert a["time_us"].dtype == np.int64 and a["values"].shape == (203, 19)
        np.testing.assert_allclose(
            a["values"][:, mapping.FEATURES.index("pack_voltage")], 14.8
        )
    with np.load(output / "online.npz", allow_pickle=False) as a:
        assert a["X"].shape[1:] == (100, 19) and a["Y"].shape[1:] == (10, 4)
        np.testing.assert_allclose(a["X"][:, :, 17], 3.7)
        np.testing.assert_allclose(a["X"][:, :, 18], 0.5)
    assert result["no_fit"] and not result["training_started"]
    assert not (Path(result["consumer_path"]) / "evaluation").exists()
    # Even re-hashing a corrupted export cannot hide a physical/online mismatch.
    with np.load(output / "run.npz", allow_pickle=False) as a:
        changed = {k: a[k] for k in a.files}
    changed["values"][-1, 17] += 1
    np.savez(output / "run.npz", **changed)
    manifest = common.read(output / "run.json")
    manifest["runs"][0]["sha256"] = common.sha(output / "run.npz")
    common.write(output / "run.json", manifest)
    receipt = common.read(output / "receipt.json")
    for key in ("run.npz", "run.json"):
        receipt["files"][key] = common.sha(output / key)
    common.write(output / "receipt.json", receipt)
    common.write(
        output / "seal.json",
        dict(receipt_sha256=common.sha(output / "receipt.json")),
    )
    with pytest.raises(AssertionError):
        pilot19.audit_one(output, expected)
