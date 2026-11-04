"""Synthetic CPU data and mocked Slurm/SSH only; never submits or trains production data."""

import json
import os
import sys

import numpy as np
import pytest

from execution import cli, common, engine, prepare, remote


def server_profile(root):
    return dict(
        schema="drone-ai.profile/1",
        name="fixture",
        project_root=str(root),
        python=sys.executable,
        runtime_profile="torch21",
        dataset_root=str(root / "dataset"),
        output_root=str(root / "results"),
        ssh=dict(host="gpu.example.test", user="tester", port=2200),
        slurm=dict(partition="p02", node="DIS01", nodes=1, ntasks=1, gpus=1, cpus=4, memory_gib=16),
    )


def test_bound_attempt_profile_precedes_inherited_machine_profile(tmp_path, monkeypatch):
    from console.profile import write

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    bound = server_profile(tmp_path)
    inherited = dict(bound, name="other", python="/other/python")
    write(attempt / "server-profile.json", bound)
    other = write(tmp_path / "other.json", inherited)
    monkeypatch.setenv("DRONE_AI_PROFILE", str(other))
    assert common.runtime_profile(attempt, required=True) == bound


def config():
    return dict(
        models=list(common.MODELS),
        seed=17,
        hidden=32,
        batch_size=32,
        learning_rate=0.001,
        epochs=20,
        early_stopping_patience=5,
        precision="float32",
        amp=False,
        retries=0,
        final_test_used=False,
        memory_gib=16,
        cpus=4,
        wall_hours=12,
        queue_hours=24,
    )


class Progress:
    def __init__(self):
        self.updates = []

    def check(self):
        pass

    def update(self, **fields):
        self.updates.append(fields)


def arrays(n, steps=100, features=19, context=6):
    rng = np.random.default_rng(n)
    return dict(
        X=rng.normal(size=(n, steps, features)),
        C=rng.normal(size=(n, context)),
        Y=rng.normal(size=(n, 10, 4)),
    )


def test_primal_ridge_matches_existing_dual_solver(tmp_path):
    train, val = arrays(12, 3, 2, 2), arrays(4, 3, 2, 2)
    result = engine.ridge_fit(train, val, tmp_path, Progress())
    from capstone_ai.training import flat

    x, v = flat(train), flat(val)
    expected = []
    for alpha in (1.0, 10.0, 100.0):
        beta = x.T @ np.linalg.solve(
            x @ x.T + alpha * np.eye(len(x)), train["Y"].reshape(len(x), 40)
        )
        error = ((v @ beta).reshape(-1, 10, 4) - val["Y"]) ** 2
        score = 0.5 * error[..., :3].mean() + 0.5 * error[..., 3].mean()
        expected.append((score, alpha, beta))
    best = min(expected, key=lambda v: v[0])
    archive = np.load(tmp_path / "ridge.npz")
    np.testing.assert_allclose(archive["coefficients"], best[2], atol=1e-10, rtol=1e-10)
    assert float(archive["alpha"]) == best[1]
    np.testing.assert_allclose(
        [r["validation_loss"] for r in result["history"]], [v[0] for v in expected], atol=1e-12
    )


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_neural_export_restore_and_same_seed_cpu_fixture(tmp_path, name):
    cfg = {**config(), "hidden": 8, "batch_size": 2, "epochs": 2}
    train, val = arrays(4), arrays(2)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    progress = Progress()
    result = engine.neural_fit(name, cfg, train, val, first, progress, device="cpu")
    repeated = engine.neural_fit(name, cfg, train, val, second, Progress(), device="cpu")
    assert result["save_load_max_abs"] <= 1e-6
    assert [r["validation_loss"] for r in result["history"]] == [
        r["validation_loss"] for r in repeated["history"]
    ]
    assert (first / "last.pt").exists() and (first / "best.pt").exists()
    assert any(
        x.get("stage") == "training" and x["batch"] <= x["batches"] for x in progress.updates
    )
    assert common.sha(first / "model.safetensors") == common.sha(second / "model.safetensors")


def test_two_model_batch_admission_cpu_fixture(tmp_path):
    result = engine.probe_models(
        {**config(), "hidden": 8, "batch_size": 2}, arrays(2), tmp_path, Progress(), device="cpu"
    )
    assert set(result) == {"lstm", "transformer"}


def test_headers_and_memory_estimate(tmp_path):
    path = tmp_path / "input.npz"
    np.savez_compressed(path, **arrays(5))
    headers = prepare.npz_headers(path)
    assert headers["X"]["shape"] == (5, 100, 19)
    plan = prepare.memory_plan(
        {"normal_train": headers, "normal_validation": headers, "normal_calibration": headers}
    )
    assert plan["primal_matrix_bytes"] == 1907**2 * 8
    assert plan["estimated_host_peak_bytes"] < plan["requested_host_bytes"]
    np.savez(path, X=np.array([{"unsafe": True}], dtype=object))
    with pytest.raises(ValueError, match="Object"):
        prepare.npz_headers(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("models", ["lstm"]),
        ("seed", 42),
        ("retries", 2),
        ("final_test_used", True),
        ("memory_gib", 1),
        ("amp", True),
    ],
)
def test_fixed_experiment_contract(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError):
        common.config_check(cfg)


def remote_fixture(tmp_path, monkeypatch):
    root = tmp_path / "job/input"
    root.mkdir(parents=True)
    common.write(root / "config.json", config())
    common.write(root / "inventory.json", {"config.json": common.sha(root / "config.json")})
    profile = tmp_path / "server-profile.json"
    common.write(profile, server_profile(tmp_path))
    monkeypatch.setenv("DRONE_AI_PROFILE", str(profile))
    monkeypatch.setattr(remote, "REMOTE_LOCK", tmp_path / "project.lock")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return root


def test_existing_own_job_blocks_submission(tmp_path, monkeypatch):
    root = remote_fixture(tmp_path, monkeypatch)
    calls = []

    def command(argv):
        calls.append(argv)
        assert argv[0] == "squeue"
        return "19156 bash RUNNING"

    monkeypatch.setattr(remote, "command", command)
    with pytest.raises(ValueError, match="Existing own"):
        remote.submit(root)
    assert len(calls) == 1 and not (root.parent / "submitted.claim").exists()


def test_submission_pinned_once_and_no_cuda_environment_override(tmp_path, monkeypatch):
    root = remote_fixture(tmp_path, monkeypatch)
    calls = []

    def command(argv):
        calls.append(argv)
        return "" if argv[0] == "squeue" else "12345"

    monkeypatch.setattr(remote, "command", command)
    remote.submit(root)
    argv = calls[-1]
    for arg in (
        "--partition=p02",
        "--nodelist=DIS01",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:1",
        "--no-requeue",
        "--mem=16G",
    ):
        assert arg in argv
    script = (root.parent / "job.sh").read_text()
    assert "CUDA_VISIBLE_DEVICES" not in script
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in script
    with pytest.raises(FileExistsError):
        remote.submit(root)
    assert sum(c[0] == "sbatch" for c in calls) == 1


def test_release_requires_accounting_and_owner(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    common.write(tmp_path / "submission.json", dict(job_id="123", uid=os.getuid()))
    monkeypatch.setattr(remote, "command", lambda argv: "")
    assert remote.snapshot(root)["released"] is False
    import pwd

    name = pwd.getpwuid(os.getuid()).pw_name
    monkeypatch.setattr(
        remote, "command", lambda argv: "" if argv[0] == "squeue" else f"123|{name}|COMPLETED|0:0|"
    )
    result = remote.snapshot(root)
    assert result["released"] and result["exit_code"] == "0:0"
    monkeypatch.setattr(
        remote, "command", lambda argv: "" if argv[0] == "squeue" else "123|other|COMPLETED|0:0|"
    )
    with pytest.raises(ValueError, match="owner"):
        remote.snapshot(root)


def test_scope_progress_reset_and_display(tmp_path):
    p = common.Progress(tmp_path / "progress.json")
    p.update(model="lstm", stage="training", epoch=20, train_loss=0.1, validation_loss=0.2)
    p.update(model="transformer", stage="model_start")
    assert "epoch" not in p.state and "validation_loss" not in p.state
    common.write(
        tmp_path / "controller.json",
        dict(
            status="RUNNING",
            updated=__import__("time").time(),
            remote_progress=dict(
                stage="training",
                model="transformer",
                epoch=1,
                epochs=20,
                batch=3,
                batches=10,
                done=3,
                total=200,
                train_loss=0.2,
                phase_eta_s=7,
            ),
        ),
    )
    text = cli.render(tmp_path)
    assert "transformer" in text and "1.5%" in text and "전체 시간 진행률 아님" in text


def test_package_paths_reject_traversal(tmp_path):
    with pytest.raises(ValueError):
        common.child(tmp_path, "../outside")
    (tmp_path / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        common.child(tmp_path, "link/file")


@pytest.mark.parametrize("next_version", [False, True, "comparison"])
def test_four_model_end_to_end_synthetic_cpu_and_fixed_development_evaluator(
    tmp_path, next_version
):
    import io
    import tarfile

    from capstone_ai.mapping import FEATURES
    from execution import evaluate

    root = tmp_path / "input"
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    out = tmp_path / "download"
    out.mkdir()
    cfg = {**config(), "hidden": 8, "epochs": 1, "batch_size": 2, "dataset": str(dataset)}
    rows = []
    for role, n in [("normal_train", 6), ("normal_validation", 4), ("normal_calibration", 4)]:
        rid = "fixture-" + role
        rows.append(dict(run_id=rid, role=role, paired_case_id=rid))
        values = arrays(n)
        np.savez_compressed(
            dataset / (role + ".npz"),
            **values,
            run_id=np.array([rid] * n),
            origin_us=np.arange(n) * 1000000,
            segment=np.zeros(n, dtype=int),
        )
    np.savez(
        dataset / "scaler.npz",
        x_mean=np.zeros(19),
        x_scale=np.ones(19),
        c_mean=np.zeros(6),
        c_scale=np.ones(6),
        y_mean=np.zeros(4),
        y_scale=np.ones(4),
    )
    preprocess = common.read(
        common.ROOT / "Data/generation/pretrain/resources/preprocessing/preprocessing.json"
    )
    common.write(root / "preprocessing.json", preprocess)
    for i in range(23):
        rid = f"dev-{i}"
        role = "normal_dev_eval" if i < 11 else "anomaly_dev"
        body = (
            b""
            if i < 11
            else (
                json.dumps(
                    dict(event_id=f"e{i}", onset_kind="pre_existing", observation_start_time_us=0)
                )
                + "\n"
            ).encode()
        )
        archive = tmp_path / (rid + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo("evaluation/events.jsonl")
            member.size = len(body)
            tar.addfile(member, io.BytesIO(body))
        import hashlib

        rows.append(
            dict(
                run_id=rid,
                role=role,
                group_id=rid,
                paired_case_id=rid,
                scenario="fixture",
                archive=dict(
                    path=str(archive),
                    sha256=common.sha(archive),
                    bytes=archive.stat().st_size,
                    members={
                        "evaluation/events.jsonl": dict(
                            kind="file", bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
                        )
                    },
                ),
            )
        )
        folder = dataset / "canonical" / rid
        folder.mkdir(parents=True)
        frames = [
            dict(
                run_id=rid,
                clock_id="fixture",
                segment_id=0,
                time_us=j * 100000,
                values={n: (16.0 if n == "pack_voltage" else 0.0) for n in FEATURES},
                quality=dict.fromkeys(FEATURES, "valid"),
                operating_state="airborne",
                warmup=False,
            )
            for j in range(12)
        ]
        windows = [
            dict(
                clock_id="fixture",
                segment_id=0,
                origin_us=0,
                X=np.zeros((100, 19)).tolist(),
                C=[0.0] * 6,
            )
        ]
        for name, values in [("frames", frames), ("windows", windows)]:
            (folder / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in values))
        common.write(
            folder / "summary.json",
            dict(
                config=preprocess,
                exclusions={},
                output_hashes={
                    name: common.sha(folder / (name + ".jsonl")) for name in ("frames", "windows")
                },
            ),
        )
    common.write(
        dataset / "manifest.json",
        dict(runs=rows, hashes={p.name: common.sha(p) for p in dataset.glob("*.npz")}),
    )
    progress = Progress()
    expected_models = list(common.MODELS)
    if next_version == "comparison":
        from learning import comparison

        expected_models = comparison.MODELS
        cfg.update(
            revision=comparison.REVISION,
            model_revision="explicit_dropout_v1",
            variants=comparison.VARIANTS,
            models=expected_models,
            seed=0,
        )
    if next_version:
        from learning.lifecycle import make_bundles
        from pipeline.pipeline import normalized_role

        cfg.update(
            layers=2,
            heads=4,
            binding=dict(data="synthetic", scaler="synthetic", sources="synthetic"),
        )
        scaler = dict(np.load(dataset / "scaler.npz"))
        train, val = [
            normalized_role(dataset, role, scaler) for role in ("normal_train", "normal_validation")
        ]
        make_bundles(cfg, root, out, train, val, progress, device="cpu")
    else:
        engine.bundles(root, out, cfg, progress, device="cpu")
    engine.calibrate(root, out, progress, model_names=expected_models)
    assert sum(1 for p in (out / "calibration").glob("*/*/seal.json")) == 23 * len(expected_models)
    cfg["dataset_files"] = {
        str(p.relative_to(dataset)): common.sha(p) for p in dataset.rglob("*") if p.is_file()
    }
    common.write(
        out / "result-seal.json",
        dict(
            files={str(p.relative_to(out)): common.sha(p) for p in out.rglob("*") if p.is_file()},
            models=expected_models,
            final_test_used=False,
            truth_join=False,
            status="TRAIN_CALIBRATION_INFERENCE_COMPLETE",
        ),
    )
    common.write(
        tmp_path / "release.json", dict(released=True, slurm_state="COMPLETED", exit_code="0:0")
    )
    metrics = evaluate.evaluate(
        cfg, out, tmp_path / "evaluation", progress, model_names=expected_models
    )
    assert set(metrics["models"]) == set(expected_models)
    assert metrics["seed"] == cfg["seed"]
    assert metrics["common_valid_origins"] == 23
    for record in metrics["models"].values():
        assert len(record["per_run"]) == 23
        assert record["events"]["events"] == 12
        assert record["forecast_errors_common_origins"]["scored_forecasts"] == 23
    report = common.read(tmp_path / "evaluation/report.json")
    assert report["status"] == "REPORT_READY_STOPPED" and report["next_round_authorized"] is False
    # No evaluation rerun on an existing result, and failed release never admits truth.
    common.write(tmp_path / "release.json", dict(released=False, slurm_state="RUNNING"))
    with pytest.raises(ValueError, match="release"):
        evaluate.evaluate(cfg, out, tmp_path / "not-created", progress)
    assert not (tmp_path / "not-created").exists()


def test_purged_squeue_job_requires_successful_accounting(tmp_path, monkeypatch):
    import pwd
    import subprocess

    root = tmp_path / "input"
    root.mkdir()
    common.write(tmp_path / "submission.json", dict(job_id="123", uid=os.getuid()))
    name = pwd.getpwuid(os.getuid()).pw_name

    def command(argv):
        if argv[0] == "squeue":
            if "-j" in argv:
                raise subprocess.CalledProcessError(
                    1, argv, stderr="slurm_load_jobs error: Invalid job id specified"
                )
            return ""
        return f"123|{name}|COMPLETED|0:0|"

    monkeypatch.setattr(remote, "command", command)
    assert remote.snapshot(root)["released"] is True

    def bad(argv):
        raise subprocess.CalledProcessError(1, argv, stderr="controller unavailable")

    monkeypatch.setattr(remote, "command", bad)
    with pytest.raises(subprocess.CalledProcessError):
        remote.snapshot(root)


def test_worker_stops_before_fit_on_allocation_failure(tmp_path, monkeypatch):
    import pipeline.runtime

    root = remote_fixture(tmp_path, monkeypatch)
    (root.parent / "output").mkdir()

    def reject(*args, **kwargs):
        raise ValueError("DIS01 allocation mismatch")

    monkeypatch.setattr(pipeline.runtime, "check", reject)
    monkeypatch.setattr(engine, "bundles", lambda *a, **k: pytest.fail("must not fit"))
    with pytest.raises(ValueError, match="allocation mismatch"):
        remote.worker(root, common.sha(root / "inventory.json"))
    assert common.read(root.parent / "output/progress.json")["status"] == "FAILED_STOPPED"


def test_duplicate_controller_preserves_active_state(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "plan", config)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    task = tmp_path / "tasks/active"
    common.write(task / "controller.json", {"status": "RUNNING", "job_id": "123"})
    previous = (task / "controller.json").read_bytes()
    with (tmp_path / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            cli.launch("recover", task)
        with pytest.raises(BlockingIOError):
            cli.worker(task, "recover")
    assert (task / "controller.json").read_bytes() == previous


def test_prepared_input_binding_and_one_use(tmp_path, monkeypatch):
    home, store = tmp_path / "home", tmp_path / "store"
    output = store / "gpu104/prepared"
    monkeypatch.setattr(cli, "HOME", home)
    monkeypatch.setattr(cli, "STORE", store)
    cfg = config()
    common.write(home / "plan.json", cfg)
    common.write(output / "input/config.json", cfg)
    common.write(output / "input/inventory.json", {})
    receipt = dict(
        plan_sha256=common.sha(home / "plan.json"),
        root=common.REMOTE_BASE + "/prepared/input",
        output=str(output),
        inventory_sha256=common.sha(output / "input/inventory.json"),
    )
    common.write(home / "prepared-input.json", receipt)
    assert cli.prepared_input(cfg)["prepared"] is True
    common.write(output / "execution.claim", {})
    with pytest.raises(ValueError, match="already claimed"):
        cli.prepared_input(cfg)
    (output / "execution.claim").unlink()
    common.write(output / "input/config.json", {**cfg, "seed": 42})
    with pytest.raises(ValueError, match="configuration changed"):
        cli.prepared_input(cfg)


def test_remote_verify_never_submits_and_detects_tamper(tmp_path, monkeypatch):
    root = remote_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(remote, "command", lambda *a: pytest.fail("no Slurm calls allowed"))
    assert remote.verify_input(root)["status"] == "INPUT_VERIFIED_NO_TRAINING"
    common.write(root / "config.json", {**config(), "seed": 42})
    with pytest.raises(ValueError):
        remote.verify_input(root)


def test_selected_ssh_profile_without_old_control_socket(tmp_path, monkeypatch):
    profile = tmp_path / "server-profile.json"
    common.write(profile, server_profile(tmp_path))
    monkeypatch.setenv("DRONE_AI_PROFILE", str(profile))
    argv = cli.ssh_argv()
    assert argv[-1] == "tester@gpu.example.test"
    assert argv[argv.index("-p") + 1] == "2200"
    assert "-S" not in argv
    assert "StrictHostKeyChecking=yes" in argv
    monkeypatch.delenv("DRONE_AI_PROFILE")
    with pytest.raises(ValueError, match="explicit server profile"):
        cli.ssh_argv()
