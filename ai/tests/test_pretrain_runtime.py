"""Runtime pinning and real four-worker numeric equivalence; no SITL or services."""

import concurrent.futures
import os
import subprocess

import pytest

from pretrain import cli, common


def test_runtime_drift_blocks_service_before_submission(tmp_path, monkeypatch):
    home = tmp_path / "home"
    task = home / "tasks" / "changed-runtime"
    runtime = common.runtime_identity()
    runtime["sha256"] = "0" * 64
    common.write(task / "config.json", dict(source_sha256={}, python_runtime=runtime))
    (task / "config.sha256").write_text(common.sha(task / "config.json"))
    monkeypatch.setattr(cli, "HOME", home)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("must not submit"))
    with pytest.raises(ValueError, match="Python runtime changed"):
        cli.start(task)
    assert not (task / "operator/memory-policy.json").exists()
    assert not (task / "operator/service.json").exists()


def test_new_task_pins_actual_runtime_without_execution(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cli, "HOME", home)
    args = cli.parser().parse_args(["init", "--name", "runtime-pin", "--mode", "collect"])
    task = cli.create(args)
    config = common.read(task / "config.json")
    runtime = common.runtime_identity()
    assert config["python_runtime"] == runtime
    assert config["source_sha256"][runtime["executable"]] == runtime["sha256"]
    common.verify_runtime(config)
    common.verify(config["source_sha256"])
    assert config["collection_run_count"] == 300
    assert not (task / "operator/claimed").exists()
    assert not (task / "operator/service.json").exists()


def test_four_real_preprocessors_match_verified_fixture(tmp_path, monkeypatch):
    fixture = common.load(
        "runtime_raw_fixture", common.ROOT / "ai/tests/test_pretrain_streaming.py"
    )
    data = fixture.setup_run(tmp_path, monkeypatch)
    reference = common.read(data.canonical / "summary.json")
    requests = []
    for i in range(4):
        path = tmp_path / f"parallel-{i}.json"
        common.write(
            path,
            dict(
                action="preprocess",
                row=data.row,
                out=str(tmp_path / f"canonical-{i}"),
                status=str(tmp_path / f"status-{i}.json"),
            ),
        )
        requests.append(path)

    def execute(path):
        result = subprocess.run(
            [str(common.PYTHON), "-B", "-X", "faulthandler", "-m", "pretrain.worker", str(path)],
            env={
                **os.environ,
                "PYTHONPATH": common.pythonpath(),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return common.read(common.read(path)["status"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(execute, requests))
    assert all(r["status"] == "COMPLETE" for r in results)
    assert (
        common.overlap([dict(started=r["started"], finished=r["finished"]) for r in results]) == 4
    )
    assert all(r["result"]["output_hashes"] == reference["output_hashes"] for r in results)
