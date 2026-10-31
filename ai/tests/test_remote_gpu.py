import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gpu_probe", ROOT / "ai/remote/gpu_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def allocation():
    return {"JobId": "123", "Partition": "p02", "NumNodes": "1", "NumTasks": "1",
            "JobState": "RUNNING", "UserId": "example(1000)",
            "AllocTRES": "cpu=1,node=1,gres/gpu=1,gres/gpu:a6000=1"}


def test_allowed_allocation_metadata():
    probe.validate_allocation(allocation(), {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "GPU-assigned"},
                              "dis01.example", ["DIS01"], 1000)


def test_observed_legacy_slurm_running_tres():
    job = allocation()
    job["TRES"] = job.pop("AllocTRES")
    probe.validate_allocation(job, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"},
                              "DIS01", ["DIS01"], 1000)
    job["JobState"] = "PENDING"
    with pytest.raises(ValueError):
        probe.validate_allocation(job, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"},
                                  "DIS01", ["DIS01"], 1000)
    job["JobState"] = "RUNNING"
    job["ReqTRES"] = job.pop("TRES")
    with pytest.raises(ValueError):
        probe.validate_allocation(job, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"},
                                  "DIS01", ["DIS01"], 1000)


@pytest.mark.parametrize("key,value", [("Partition", "p01"), ("NumNodes", "2"), ("NumTasks", "2"),
                                      ("AllocTRES", "gres/gpu=2"), ("AllocTRES", "cpu=1"),
                                      ("AllocTRES", "gres/gpu=1,gres/gpu:a6000=2"),
                                      ("JobState", "PENDING"), ("UserId", "other(1001)")])
def test_wrong_or_unverified_resource_is_rejected(key, value):
    job = {**allocation(), key: value}
    with pytest.raises(ValueError):
        probe.validate_allocation(job, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"},
                                  "DIS01", ["DIS01"], 1000)


@pytest.mark.parametrize("hostname,nodes,env", [
    ("login", ["DIS01"], {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"}),
    ("DIS01", ["DIS06"], {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"}),
    ("DIS01", ["DIS01"], {"SLURM_JOB_ID": "123"}),
    ("DIS01", ["DIS01"], {"CUDA_VISIBLE_DEVICES": "0"}),
])
def test_login_wrong_node_and_missing_slurm_environment_rejected(hostname, nodes, env):
    with pytest.raises(ValueError):
        probe.validate_allocation(allocation(), env, hostname, nodes, 1000)


def test_visible_gpu_and_version_gate():
    good = [(3, 10, 22), "2.1.2+cu121", True, 1, "NVIDIA RTX A6000", 48 * 1024**3]
    probe.validate_torch_environment(*good)
    for index, replacement in [(0, (3, 11, 1)), (1, "2.2.0"), (2, False), (3, 2), (3, 0),
                               (4, "NVIDIA RTX 3090"), (5, 24 * 1024**3)]:
        bad = list(good)
        bad[index] = replacement
        with pytest.raises(ValueError):
            probe.validate_torch_environment(*bad)


def test_real_cli_refuses_local_non_slurm_execution(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "ai/remote/gpu_probe.py"),
                             "--package", str(tmp_path / "missing-package"), "--out", str(tmp_path / "out"),
                             "--runtime-profile", "mamba25"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    report = json.loads((tmp_path / "out/result.json").read_text())
    assert "no Slurm allocation" in report["error"]
    assert report["gpu_operation"] == "not_executed"
    assert report["model_batch"] == "not_executed"


def test_real_frozen_model_batch_on_cpu():
    package = ROOT / "ai/artifacts/gpu-handoff-v6/release/x500-gpu-smoke-v1"
    before = probe.check_package(package)
    # The existing suite imports the local package. Isolate the portable import
    # in a fresh interpreter, just as the remote CLI does after allocation.
    program = (
        "import importlib.util,json; from pathlib import Path; "
        "spec=importlib.util.spec_from_file_location('probe'," + repr(str(ROOT / "ai/remote/gpu_probe.py")) + "); "
        "probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe); "
        "print(json.dumps(probe.batch_backward(Path(" + repr(str(package)) + "),'cpu')))"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["shape"] == [2, 10, 4]
    assert report["gradient_l2"] > 0
    assert report["optimizer_steps"] == 0
    assert probe.check_package(package) == before


@pytest.mark.parametrize("own_jobs", ["", "456"])
def test_launcher_fixed_resources_and_existing_jobs(tmp_path, own_jobs):
    """Stub Slurm only: this does not allocate or claim a real GPU test."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    queue = bin_dir / "squeue"
    queue.write_text("#!/bin/sh\nprintf '%s' '" + own_jobs + "'\n")
    queue.chmod(0o755)
    submit = bin_dir / "srun"
    submit.write_text(
        "#!" + sys.executable + "\nimport json,sys,os\nfrom pathlib import Path\n"
        "Path(os.environ['STUB_CALL']).write_text(json.dumps(sys.argv[1:]))\n"
        "out=Path(sys.argv[sys.argv.index('--out')+1]); out.mkdir()\n"
        "(out/'verified-job-id.txt').write_text('123\\n')\n"
    )
    submit.chmod(0o755)
    package = tmp_path / "package"
    package.mkdir()
    (package / "MANIFEST.json").write_text("{}")
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               HOME=str(tmp_path), STUB_CALL=str(tmp_path / "stub-call.json"))
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(["bash", str(ROOT / "ai/remote/allocate_probe.sh"),
                             sys.executable, str(package), str(tmp_path / "out")],
                            env=env, text=True, capture_output=True)
    if own_jobs:
        assert result.returncode == 3
        assert not (tmp_path / "stub-call.json").exists()
    else:
        assert result.returncode == 0
        args = json.loads((tmp_path / "stub-call.json").read_text())
        assert args[:5] == ["--partition=p02", "--nodelist=DIS01", "--nodes=1",
                            "--ntasks=1", "--gres=gpu:1"]
        assert "allocation released" in (tmp_path / "out/release.txt").read_text()


def test_explicit_mamba25_profile_requires_exact_runtime():
    args = [(3, 10, 18), "2.5.1+cu121", True, 1, "NVIDIA RTX A6000", 48 * 1024**3]
    probe.validate_torch_environment(*args, profile="mamba25", cuda_build="12.1")
    for index, replacement in [(0, (3, 13, 5)), (1, "2.5.1+cu118"), (1, "2.8.0"), (3, 2)]:
        bad = list(args)
        bad[index] = replacement
        with pytest.raises(ValueError):
            probe.validate_torch_environment(*bad, profile="mamba25", cuda_build="12.1")
    with pytest.raises(ValueError):
        probe.validate_torch_environment(*args, profile="mamba25", cuda_build="11.8")
    with pytest.raises(ValueError):
        probe.validate_torch_environment(*args, profile="torch21", cuda_build="12.1")
