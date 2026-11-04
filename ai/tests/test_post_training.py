"""Post-training authorization, sequencing and evidence tests; no server/GPU work."""

import os
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from execution.common import read, sha, write
from learning import comparison, operator
from learning import post_training as post


def config():
    return dict(
        schema=post.SCHEMA,
        revision=comparison.REVISION,
        execution_scope=post.SCOPE,
        models=list(comparison.MODELS),
        seed=0,
        production_training_authorized=False,
        calibration_authorized=True,
        evaluation_authorized=True,
        final_test_used=False,
        next_round_authorized=False,
        retries=0,
        memory_gib=64,
        cpus=4,
        gpu_jobs_max=1,
        models_sequential=True,
        development_counts=dict(normal_dev_eval=128, anomaly_dev=144),
        normal_counts=dict(normal_validation=104, normal_calibration=104),
        wall_hours=24,
        evaluation_hours=12,
        queue_hours=24,
        memory_required_bytes=16 * 1024**3,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("production_training_authorized", True),
        ("final_test_used", True),
        ("calibration_authorized", False),
        ("retries", 1),
        ("seed", 17),
        ("models", ["lstm"]),
        ("wall_hours", 0),
        ("evaluation_hours", 169),
        ("memory_gib", 128),
        ("memory_required_bytes", 65 * 1024**3),
    ],
)
def test_scope_rejects_unapproved_work(key, value):
    cfg = config()
    post.config_check(cfg)
    cfg[key] = value
    with pytest.raises(ValueError):
        post.config_check(cfg)


def test_submit_cannot_fit_or_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "checked", lambda root: (root, config()))
    monkeypatch.setattr(post, "SUBMIT_LOCK", tmp_path / "submit.lock")
    monkeypatch.setattr(operator, "own_queue", lambda: "123|other|RUNNING|DIS01")
    with pytest.raises(ValueError, match="Only calibration"):
        post.submit(tmp_path, "fit", config())
    with pytest.raises(ValueError, match="concurrent"):
        post.submit(tmp_path, "calibrate", config())
    assert not list(tmp_path.glob("*-submitted.claim"))


def test_evaluator_submission_requires_successful_release(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "checked", lambda root: (root, config()))
    monkeypatch.setattr(post, "SUBMIT_LOCK", tmp_path / "submit.lock")
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    write(tmp_path / "release.json", dict(released=True, slurm_state="FAILED", exit_code="1:0"))
    with pytest.raises(RuntimeError, match="successfully"):
        post.submit(tmp_path, "evaluate", config())
    assert not (tmp_path / "evaluate-submitted.claim").exists()


def test_submit_is_pinned_and_never_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "checked", lambda root: (root, config()))
    monkeypatch.setattr(post, "SUBMIT_LOCK", tmp_path / "submit.lock")
    monkeypatch.setattr(post, "python_for", lambda *a, **kw: "/approved/python")
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    submitted = []
    monkeypatch.setattr(operator, "command", lambda argv: submitted.append(argv) or "123;cluster")
    result = post.submit(tmp_path, "calibrate", config())
    assert result["job_id"] == "123"
    for flag in (
        "--partition=p02",
        "--nodelist=DIS01",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:1",
        "--mem=64G",
        "--cpus-per-task=4",
        "--no-requeue",
    ):
        assert flag in submitted[0]
    script = (tmp_path / "calibrate-job.sh").read_text()
    assert "learning.post_training calibrate" in script
    assert "CUDA_VISIBLE_DEVICES" not in script
    with pytest.raises(FileExistsError):
        post.submit(tmp_path, "calibrate", config())
    assert len(submitted) == 1


def test_control_failure_stops_before_evaluator(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setattr(post, "checked", lambda root: (root, cfg))
    monkeypatch.setattr(post, "python_for", lambda *a, **kw: sys.executable)
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    phases = []

    def submit(root, phase, cfg):
        phases.append(phase)
        return dict(phase=phase, job_id="123", uid=os.getuid())

    monkeypatch.setattr(post, "submit", submit)
    failed = dict(released=True, slurm_state="FAILED", exit_code="1:0", queue=[])
    monkeypatch.setattr(operator, "wait_release", lambda *a: failed)
    monkeypatch.setattr(operator, "snapshot", lambda *a: failed)
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    with pytest.raises(RuntimeError):
        post.control(tmp_path)
    assert phases == ["calibrate"]
    assert read(tmp_path / "controller.json")["status"] == "FAILED_STOPPED"
    assert read(tmp_path / "controller.json")["allocation_released"] is True
    assert not (tmp_path / "completion.json").exists()


def test_no_final_completion_until_both_jobs_release(tmp_path, monkeypatch):
    cfg = {**config(), "source_training": "/preserved/source"}
    monkeypatch.setattr(post, "checked", lambda root: (root, cfg))
    monkeypatch.setattr(post, "python_for", lambda *a, **kw: sys.executable)
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    phases = []

    def submit(root, phase, cfg):
        assert not (root / "completion.json").exists()
        if phase == "evaluate":
            operator.require_success(read(root / "release.json"))
        phases.append(phase)
        return dict(phase=phase, job_id=str(len(phases)), uid=os.getuid())

    monkeypatch.setattr(post, "submit", submit)

    def wait(root, record, state, cfg):
        assert not (root / "completion.json").exists()
        return dict(
            job_id=record["job_id"],
            released=True,
            slurm_state="COMPLETED",
            exit_code="0:0",
            queue=[],
        )

    monkeypatch.setattr(operator, "wait_release", wait)
    monkeypatch.setattr(post, "wait_files", lambda *a: None)
    monkeypatch.setattr(post, "seal_check", lambda *a: {"files": {}})
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    write(tmp_path / "compute/result-seal.json", {"files": {}})
    write(tmp_path / "evaluation/metrics.json", {"example": "metrics"})
    write(tmp_path / "evaluation/evidence.json", {"example": "evidence"})
    write(
        tmp_path / "evaluation/report.json",
        {
            "metrics_sha256": sha(tmp_path / "evaluation/metrics.json"),
            "evidence_sha256": sha(tmp_path / "evaluation/evidence.json"),
        },
    )
    monkeypatch.setattr(
        operator, "report", lambda root, cfg: (root / "report.txt").write_text("report")
    )
    post.control(tmp_path)
    completed = read(tmp_path / "completion.json")
    assert phases == ["calibrate", "evaluate"]
    assert completed["status"] == "REPORT_READY_STOPPED"
    assert completed["calibration_release"]["job_id"] == "1"
    assert completed["evaluation_release"]["job_id"] == "2"
    assert completed["training_repeated"] is False


def test_truth_archive_never_opened_before_successful_release(tmp_path):
    write(tmp_path / "release.json", dict(released=False))
    with pytest.raises(RuntimeError):
        post.extract_truth(tmp_path, {}, None)
    assert not (tmp_path / "truth").exists()


@pytest.mark.parametrize("unknown_phase", ["calibrate", "evaluate"])
def test_unacknowledged_submission_never_counts_as_stopped(tmp_path, monkeypatch, unknown_phase):
    cfg = config()
    monkeypatch.setattr(post, "checked", lambda root: (root, cfg))
    monkeypatch.setattr(post, "python_for", lambda *a, **kw: sys.executable)
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    release = dict(released=True, slurm_state="COMPLETED", exit_code="0:0", queue=[])

    def submit(root, phase, cfg):
        if phase == unknown_phase:
            write(root / (phase + "-submission-intent.json"), {"argv": ["sbatch"]})
            raise TimeoutError("sbatch acknowledgement missing")
        return dict(phase=phase, job_id="123", uid=os.getuid())

    monkeypatch.setattr(post, "submit", submit)
    monkeypatch.setattr(operator, "wait_release", lambda *a: release)
    monkeypatch.setattr(operator, "snapshot", lambda *a: release)
    monkeypatch.setattr(operator, "own_queue", lambda: "")
    monkeypatch.setattr(post, "wait_files", lambda *a: None)
    with pytest.raises(TimeoutError):
        post.control(tmp_path)
    state = read(tmp_path / "controller.json")
    assert state["status"] == "CONTROLLER_STOPPED_JOB_STATE_UNCONFIRMED"
    assert state["submission_outcome_unconfirmed"] is True
    assert not (tmp_path / "completion.json").exists()


def test_extract_truth_is_bounded_and_hash_verified(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    payload = b'{"event":"example"}\n'
    original = bundle / "run.jsonl"
    original.write_bytes(payload)
    archive = bundle / "truth.tar.xz"
    with tarfile.open(archive, "w:xz") as stream:
        stream.add(original, arcname="run.jsonl")
    ref = dict(
        path=archive.name,
        bytes=archive.stat().st_size,
        sha256=sha(archive),
        members={"run.jsonl": dict(bytes=len(payload), sha256=sha(original))},
    )
    write(bundle / "receipt.json", {"evaluator_truth": ref})
    root = tmp_path / "evaluation"
    write(root / "release.json", dict(released=True, slurm_state="COMPLETED", exit_code="0:0"))
    monkeypatch.setattr(post, "seal_check", lambda *a: {})
    cfg = dict(
        source_bundle=str(bundle), truth_archive=ref, receipt_sha256=sha(bundle / "receipt.json")
    )
    post.extract_truth(root, cfg, SimpleNamespace(check=lambda: None))
    assert (root / "truth/run.jsonl").read_bytes() == payload
    assert read(root / "truth/index.json") == ref["members"]


def test_changed_model_files_block_start_without_mutation(tmp_path, monkeypatch):
    cfg = config()
    source = tmp_path / "source"
    source.mkdir()
    root = tmp_path / "attempt"
    write(
        root / "input/config.json",
        {
            **cfg,
            "source_training": str(source),
            "source_training_files": {},
            "model_files": {"models/persistence/model.json": "bad-hash"},
        },
    )
    write(root / "input/inventory.json", {"config.json": sha(root / "input/config.json")})
    write(root / "server-profile.json", {})
    write(
        root / "authorization.json",
        dict(config_sha256=sha(root / "input/config.json"), execution_scope=post.SCOPE),
    )
    write(
        root / "prepared.json",
        dict(
            input_inventory_sha256=sha(root / "input/inventory.json"),
            server_profile_sha256=sha(root / "server-profile.json"),
            authorization_sha256=sha(root / "authorization.json"),
        ),
    )
    write(root / "compute/models/persistence/model.json", {"changed": True})
    monkeypatch.setattr(post, "runtime_profile", lambda *a, **kw: {})
    monkeypatch.setattr(post, "profile_check", lambda *a: None)
    before = (root / "compute/models/persistence/model.json").read_bytes()
    with pytest.raises(ValueError, match="Input hash changed"):
        post.start(root)
    assert (root / "compute/models/persistence/model.json").read_bytes() == before
    assert not (root / "start.claim").exists()


@pytest.mark.parametrize(
    "fresh_calibration,policy_kind,normalization_study,include_baselines,parallel_workers",
    [(False, None, False, False, None), (True, None, False, False, None),
     (True, "numerical", False, False, None), (True, "sensitivity", False, False, None),
     (True, "sensitivity", True, False, None), (True, "sensitivity", True, True, None),
     (True, "sensitivity", True, True, 2)],
)
def test_prepare_preserves_source_and_frozen_runtime_is_self_contained(
    tmp_path, monkeypatch, fresh_calibration, policy_kind, normalization_study, include_baselines,
    parallel_workers,
):
    """Exercise actual preparation/copy/binding, with only bulky data admission mocked."""
    source, upstream, bundle, root = [
        tmp_path / n for n in ("trained", "upstream", "bundle", "evaluation")
    ]
    upstream.mkdir()
    bundle.mkdir()
    source_cfg = dict(
        config(),
        execution_scope="training_only",
        production_training_authorized=True,
        calibration_authorized=False,
        evaluation_authorized=False,
        evaluation_hours=0,
        variants=comparison.VARIANTS,
        batch_size=256,
        epochs=60,
        early_stopping_patience=10,
        learning_rate=0.001,
        dropout=0.1,
        weight_decay=0.00001,
        model_revision="explicit_dropout_v1",
        history=100,
        future=10,
        step_us=100000,
        precision="float32",
        amp=False,
        gate_epochs=4,
        gate_windows=512,
        training_stride_us=1000000,
        memory={"estimated_host_peak_bytes": 12 * 1024**3},
        upstream=str(upstream),
        dataset_files={},
        handoff_sha256="handoff",
        source_dataset_sha256="dataset",
    )
    if normalization_study:
        from learning.study import configure

        source_cfg = configure(source_cfg)
    comparison.config_check(source_cfg)
    write(source / "input/config.json", source_cfg)
    write(source / "input/inventory.json", {"config.json": sha(source / "input/config.json")})
    write(source / "server-profile.json", {"preserved": True})
    write(
        source / "authorization.json",
        dict(
            config_sha256=sha(source / "input/config.json"),
            execution_scope="training_only",
            calibration_authorized=False,
            evaluation_authorized=False,
        ),
    )
    write(
        source / "prepared.json",
        dict(
            input_inventory_sha256=sha(source / "input/inventory.json"),
            server_profile_sha256=sha(source / "server-profile.json"),
        ),
    )
    files = {}
    for name in source_cfg["models"]:
        folder = source / "compute/models" / name
        write(folder / "model.json", {"model_id": name})
        write(folder / "integrity.json", {"model.json": sha(folder / "model.json")})
        files.update(
            {"models/" + name + "/" + n: sha(folder / n) for n in ("model.json", "integrity.json")}
        )
    write(
        source / "compute/result-seal.json",
        dict(
            files=files,
            config_sha256=sha(source / "input/config.json"),
            status="TRAINING_COMPLETE",
            models=source_cfg["models"],
            calibration_started=False,
            inference_started=False,
            evaluation_started=False,
            truth_join=False,
            final_test_used=False,
        ),
    )
    (source / "training-report.txt").write_text("sealed training report")
    release = dict(job_id="123", released=True, slurm_state="COMPLETED", exit_code="0:0", queue=[])
    write(source / "release.json", release)
    write(source / "fit-submission.json", dict(job_id="123", phase="fit", uid=os.getuid()))
    write(source / "controller.json", dict(status=post.COMPLETE))
    write(
        source / "completion.json",
        dict(
            status=post.COMPLETE,
            execution_scope="training_only",
            fit_release=release,
            calibration_started=False,
            evaluation_started=False,
            next_round_authorized=False,
            compute_seal_sha256=sha(source / "compute/result-seal.json"),
            report_sha256=sha(source / "training-report.txt"),
        ),
    )
    archived = bundle / "truth.tar.xz"
    with tarfile.open(archived, "w:xz"):
        pass
    write(
        bundle / "receipt.json",
        dict(
            evaluator_truth=dict(
                path=archived.name, bytes=archived.stat().st_size, sha256=sha(archived), members={}
            )
        ),
    )
    profile = tmp_path / "profile.json"
    write(
        profile,
        dict(
            schema="drone-ai.profile/1",
            name="test",
            project_root=str(tmp_path),
            python=sys.executable,
            runtime_profile="mamba25",
            dataset_root=str(upstream),
            output_root=str(root),
            ssh=None,
            slurm=dict(
                partition="p02", node="DIS01", nodes=1, ntasks=1, gpus=1, cpus=4, memory_gib=64
            ),
        ),
    )
    monkeypatch.setattr(post, "dataset_binding", lambda *a: {})
    import execution.prepare

    monkeypatch.setattr(execution.prepare, "npz_headers", lambda *a: {"X": {"bytes": 1024}})
    policy_options = {}
    if policy_kind is not None:
        if policy_kind == "sensitivity":
            from test_post_training_sensitivity import policy

            import learning.sensitivity_data

            monkeypatch.setattr(
                learning.sensitivity_data, "build_plan", lambda *a: {"fixture": "plan"}
            )
            monkeypatch.setattr(
                learning.sensitivity_data,
                "verify_selection_layout",
                lambda *a: {"status": "FIXTURE_LAYOUT_ONLY"},
            )
        else:
            from test_post_training_diagnostic import policy

        policy_path = tmp_path / "policies" / (policy_kind + ".json")
        write(policy_path, policy())
        policy_digest = sha(policy_path)
        policy_options[policy_kind + "_policy"] = policy_path
    before = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    baseline = None
    baseline_before = None
    if include_baselines:
        baseline = tmp_path / "preserved-baselines"
        baseline_files = {}
        for name in post.BASELINES:
            folder = baseline / "compute/models" / name
            write(folder / "model.json", dict(model_id=name))
            write(folder / "integrity.json", {"model.json": sha(folder / "model.json")})
            for filename in ("model.json", "integrity.json"):
                baseline_files[f"models/{name}/{filename}"] = sha(folder / filename)
        for filename in ("input/config.json", "compute/result-seal.json"):
            write(baseline / filename, dict(frozen=True))
        binding = {filename: sha(baseline / filename) for filename in
                   ("input/config.json", "compute/result-seal.json")}
        monkeypatch.setattr(post, "baseline_source_proof",
                            lambda *a: ({}, dict(files=baseline_files), binding))
        baseline_before = {str(p.relative_to(baseline)): sha(p) for p in baseline.rglob("*") if p.is_file()}
    post.prepare(
        SimpleNamespace(
            root=root,
            source=source,
            bundle=bundle,
            profile=profile,
            approve_post_training=True,
            calibration_hours=24,
            evaluation_hours=12,
            queue_hours=24,
            fresh_calibration=fresh_calibration,
            baseline_source=baseline,
            parallel_replay_workers=parallel_workers,
            **policy_options,
        )
    )
    assert before == {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    assert read(root / "prepared.json")["status"] == "AUTHORIZED_NOT_STARTED"
    if parallel_workers is not None:
        assert read(root / "input/config.json")["parallel_replay_workers"] == 2
        assert (root / "input/ai/learning/parallel_replay.py").is_file()
    if include_baselines:
        assert read(root / "input/config.json")["models"] == source_cfg["models"] + list(post.BASELINES)
        assert read(root / "authorization.json")["models"] == source_cfg["models"] + list(post.BASELINES)
        assert all((root / "compute/models" / name / "model.json").is_file() for name in post.BASELINES)
        assert baseline_before == {str(p.relative_to(baseline)): sha(p) for p in baseline.rglob("*") if p.is_file()}
    assert not (root / "truth").exists()
    assert not list(root.glob("*-submitted.claim"))
    if policy_kind is not None:
        key = policy_kind + "_policy"
        prepared_cfg = read(root / "input/config.json")
        frozen_path = root / "input" / (policy_kind + "-policy.json")
        assert prepared_cfg[key] == read(frozen_path) == policy()
        assert prepared_cfg[key + "_sha256"] == sha(frozen_path)
        assert sha(policy_path) == policy_digest
    if fresh_calibration:
        prepared_cfg = read(root / "input/config.json")
        assert prepared_cfg["all_model_calibration_fresh"] is True
        assert prepared_cfg["all_model_replay_fresh"] is True
        assert prepared_cfg["prediction_batch_size"] == 256
        assert prepared_cfg["verify_prediction_batch"] is True
        assert prepared_cfg["reuse_calibration"] == {}
        assert "continuation" not in prepared_cfg
        assert not (root / "input/continuation.json").exists()
        assert not (root / "compute/calibration").exists()
        assert {p.name for p in (root / "compute").iterdir()} == {"models"}
        assert {p.name for p in (root / "compute/models").iterdir()} == set(
            source_cfg["models"] + (list(post.BASELINES) if include_baselines else [])
        )
    env = {**os.environ, "PYTHONPATH": str(root / "input/ai") + ":" + str(root / "input/ai/src")}
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from learning.post_training import checked; from pathlib import Path; checked(Path(__import__('sys').argv[1]))",
            str(root),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("policy_kind", ["numerical", "sensitivity"])
@pytest.mark.parametrize(
    "invalid_path",
    ["missing", "directory", "symlink", "parent_symlink", "dangling", "traversal",
     "fifo", "invalid_json", "invalid_schema"],
)
def test_prepare_rejects_invalid_policy_before_creating_output(
    tmp_path, policy_kind, invalid_path
):
    folder = tmp_path / "policies"
    folder.mkdir()
    path = folder / "policy.json"
    if policy_kind == "sensitivity":
        from test_post_training_sensitivity import policy
    else:
        from test_post_training_diagnostic import policy

    if invalid_path == "directory":
        path.mkdir()
    elif invalid_path in {"symlink", "parent_symlink", "traversal"}:
        write(path, policy())
        if invalid_path == "symlink":
            link = folder / "link.json"
            link.symlink_to(path)
            path = link
        elif invalid_path == "parent_symlink":
            link = tmp_path / "linked"
            link.symlink_to(folder, target_is_directory=True)
            path = link / path.name
        else:
            path = folder / ".." / folder.name / path.name
    elif invalid_path == "dangling":
        path.symlink_to(folder / "absent.json")
    elif invalid_path == "fifo":
        os.mkfifo(path)
    elif invalid_path == "invalid_json":
        path.write_text("{broken-json")
    elif invalid_path == "invalid_schema":
        write(path, {"schema": "unrecognized"})
    root = tmp_path / "evaluation"
    with pytest.raises(ValueError):
        post.prepare(SimpleNamespace(
            approve_post_training=True,
            fresh_calibration=True,
            root=root,
            **{policy_kind + "_policy": path},
        ))
    assert not root.exists()


def fresh_config():
    return dict(
        config(),
        prediction_batch_size=256,
        verify_prediction_batch=True,
        all_model_replay_fresh=True,
        all_model_calibration_fresh=True,
        reuse_calibration={},
    )


@pytest.mark.parametrize("change", [
    {"reuse_calibration": {"persistence": {"files": {}}}},
    {"continuation": {}},
    {"prediction_batch_size": 1},
    {"all_model_calibration_fresh": False},
    {"verify_prediction_batch": False},
])
def test_fresh_calibration_rejects_reuse_or_changed_execution(change):
    cfg = fresh_config()
    post.config_check(cfg)
    cfg.update(change)
    with pytest.raises(ValueError):
        post.config_check(cfg)


def test_fresh_calibration_rejects_continuation_before_io(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        post.prepare(SimpleNamespace(
            approve_post_training=True,
            fresh_calibration=True,
            continue_from=tmp_path / "old",
        ))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("artifact", ["compute/calibration", "input/continuation.json"])
@pytest.mark.parametrize("dangling_symlink", [False, True])
def test_fresh_worker_guard_rejects_existing_artifacts(tmp_path, artifact, dangling_symlink):
    cfg = fresh_config()
    post.require_fresh_calibration_inputs(tmp_path, cfg)
    path = tmp_path / artifact
    if dangling_symlink:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(tmp_path / "missing-target")
    elif path.suffix:
        write(path, {"existing": True})
    else:
        path.mkdir(parents=True)
    with pytest.raises(ValueError, match="without calibration or reuse artifacts"):
        post.require_fresh_calibration_inputs(tmp_path, cfg)
    assert path.exists() or path.is_symlink()
