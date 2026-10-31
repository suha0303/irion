"""Small synthetic CPU regressions only; no production data or GPU execution."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from capstone_ai.models import make_model as reference
from learning.benchmark import measure
from learning.models import ModelSpec, make_model
from learning.profiles import PROFILES, select_batch, specs
from learning.training import Batches, backend, run_fit, safe_step


def arrays(n=5):
    rng = np.random.default_rng(17)
    return dict(
        X=rng.normal(size=(n, 100, 19)), C=rng.normal(size=(n, 6)), Y=rng.normal(size=(n, 10, 4))
    )


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_reference_eval_equivalence(name):
    torch.manual_seed(17)
    old = reference(name, 32).eval()
    new = make_model(ModelSpec(name, 32)).eval()
    if name == "lstm":
        for stack in ["encoder", "decoder"]:
            source = getattr(old, stack).state_dict()
            for i, layer in enumerate(getattr(new, stack).layers):
                layer.load_state_dict(
                    {k: source[k.replace("_l0", f"_l{i}")] for k in layer.state_dict()}
                )
        for module in ["fuse", "lead", "head"]:
            getattr(new, module).load_state_dict(getattr(old, module).state_dict())
    else:
        new.load_state_dict(old.state_dict())
    x, c, _ = Batches(arrays(2), 2, "cpu").get([0, 1])
    with backend("cpu"), torch.no_grad():
        torch.testing.assert_close(new(x, c), old(x, c), rtol=1e-6, atol=1e-6)
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    if name == "lstm":
        assert all(
            layer.dropout == 0 for stack in [new.encoder, new.decoder] for layer in stack.layers
        )
        assert new.encoder.dropout.p == 0.1


@pytest.mark.parametrize("profile", list(PROFILES))
def test_size_profiles_shape(profile):
    torch.set_num_threads(2)
    for spec in specs(profile):
        model = make_model(spec).eval()
        with backend("cpu"), torch.no_grad():
            assert model(torch.zeros(1, 100, 19), torch.zeros(1, 6)).shape == (1, 10, 4)


def test_input_conversion_once_and_tail():
    data = arrays()
    source = data["X"].copy()
    loader = Batches(data, 4, "cpu")
    x, c, y = loader.get([4])
    assert x.dtype == torch.float32 and x.shape == (1, 100, 19)
    np.testing.assert_array_equal(data["X"], source)
    with pytest.raises(ValueError, match="capacity"):
        loader.get(range(5))


def test_invalid_gradients_stop_before_optimizer():
    torch.set_num_threads(2)
    m = make_model(ModelSpec("lstm", 8))
    opt = torch.optim.AdamW(m.parameters())
    before = {k: v.clone() for k, v in m.state_dict().items()}
    h = next(m.parameters()).register_hook(lambda g: torch.full_like(g, float("inf")))
    with backend("cpu"), pytest.raises(ValueError, match="Nonfinite gradients"):
        safe_step(m, opt, *Batches(arrays(2), 2, "cpu").get([0, 1]))
    h.remove()
    assert not opt.state
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k])


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_three_fresh_process_resume(tmp_path, name):
    data = tmp_path / "data.npz"
    np.savez(data, **arrays(4))
    runner = tmp_path / "run.py"
    runner.write_text("""import sys,numpy as np
from learning.models import ModelSpec
from learning.training import run_fit
phase,name,path,out,resume=sys.argv[1:]
a=dict(np.load(path))
run_fit(ModelSpec(name,8),a,a,out,binding={'data':'fixture','scaler':'fixture','sources':'fixture'},batch_size=2,epochs=4,patience=5,resume=None if resume=='-' else resume,stop_after_epoch=2 if phase=='partial' else None)
""")
    for phase in ["full", "partial", "resume"]:
        subprocess.run(
            [
                sys.executable,
                str(runner),
                phase,
                name,
                str(data),
                str(tmp_path / phase),
                str(tmp_path / "partial/last.pt") if phase == "resume" else "-",
            ],
            check=True,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            timeout=30,
        )
    full = torch.load(tmp_path / "full/last.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resume/last.pt", weights_only=False)
    for k, v in full["model"].items():
        assert torch.equal(v, resumed["model"][k])
    for key in ["train_loss", "validation_loss"]:
        assert [x[key] for x in full["history"]] == [x[key] for x in resumed["history"]]
    for i, state in full["optimizer"]["state"].items():
        for k, v in state.items():
            assert torch.equal(v, resumed["optimizer"]["state"][i][k])
    assert torch.equal(full["rng"]["cpu"], resumed["rng"]["cpu"])
    assert full["next_epoch"] == resumed["next_epoch"] == 4
    assert (tmp_path / "full/model.safetensors").read_bytes() == (
        tmp_path / "resume/model.safetensors"
    ).read_bytes()
    with pytest.raises(ValueError, match="mismatch"):
        run_fit(
            ModelSpec(name, 8),
            arrays(4),
            arrays(4),
            tmp_path / "bad",
            binding={"data": "changed", "scaler": "fixture", "sources": "fixture"},
            batch_size=2,
            epochs=4,
            patience=5,
            resume=tmp_path / "partial/last.pt",
        )


def test_batch_selection_never_fills_memory_for_its_own_sake():
    gib = 1024**3
    rows = [
        dict(
            batch=128,
            windows_per_s=1000,
            peak_reserved_bytes=3 * gib,
            free_before_bytes=48 * gib,
            free_after_bytes=40 * gib,
            passed=True,
        ),
        dict(
            batch=256,
            windows_per_s=1020,
            peak_reserved_bytes=6 * gib,
            free_before_bytes=48 * gib,
            free_after_bytes=37 * gib,
            passed=True,
        ),
        dict(
            batch=1024,
            windows_per_s=500,
            peak_reserved_bytes=40 * gib,
            free_before_bytes=48 * gib,
            free_after_bytes=5 * gib,
            passed=True,
        ),
    ]
    assert select_batch(rows)["batch"] == 128


def test_finite_benchmark_cpu_and_unauthorized_cuda(tmp_path):
    r = measure(ModelSpec("lstm", 8), arrays(5), [2, 4], tmp_path / "cpu", warmup=1, steps=3)
    assert len(r) == 2 and all(x["windows_per_s"] > 0 and x["discarded_weights"] for x in r)
    assert (
        json.loads((tmp_path / "cpu/result.json").read_text())["next_training_authorized"] is False
    )
    with pytest.raises(ValueError, match="approval"):
        measure(ModelSpec("lstm", 8), arrays(5), [2], tmp_path / "cuda", device="cuda")


def test_cuda_fit_requires_explicit_approval(tmp_path):
    with pytest.raises(ValueError, match="approval"):
        run_fit(
            ModelSpec("lstm", 8),
            arrays(),
            arrays(),
            tmp_path / "fit",
            binding={"data": "fixture", "scaler": "fixture", "sources": "fixture"},
            device="cuda",
        )
    assert json.loads((tmp_path / "fit/failure.json").read_text())["automatic_retry"] is False


def test_backend_restores_flags_on_failure():
    old_cudnn = torch.backends.cudnn.enabled
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    with pytest.raises(RuntimeError), backend("cpu"):
        assert not torch.backends.cudnn.enabled
        assert not torch.backends.cuda.matmul.allow_tf32
        raise RuntimeError("intentional")
    assert torch.backends.cudnn.enabled == old_cudnn
    assert torch.backends.cuda.matmul.allow_tf32 == old_tf32


def test_checkpoint_schema_is_required(tmp_path):
    binding = {"data": "fixture", "scaler": "fixture", "sources": "fixture"}
    run_fit(
        ModelSpec("lstm", 8),
        arrays(2),
        arrays(2),
        tmp_path / "first",
        binding=binding,
        batch_size=2,
        epochs=2,
        stop_after_epoch=1,
    )
    ck = torch.load(tmp_path / "first/last.pt", weights_only=False)
    ck["schema"] = "different-checkpoint"
    torch.save(ck, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="mismatch"):
        run_fit(
            ModelSpec("lstm", 8),
            arrays(2),
            arrays(2),
            tmp_path / "resume",
            binding=binding,
            batch_size=2,
            epochs=2,
            resume=tmp_path / "wrong.pt",
        )


def test_diagnostic_admission_and_exact_comparison():
    from learning.diagnostic import candidate_batches, same

    gib = 1024**3
    selected, rejected = candidate_batches(
        "large", {"free_before_bytes": 45 * gib, "peak_reserved_bytes": gib}
    )
    assert selected == [128, 512]
    assert rejected == [1024, 2048, 4096]
    with pytest.raises(MemoryError):
        candidate_batches("large", {"free_before_bytes": 3 * gib, "peak_reserved_bytes": gib})
    a = {"weights": torch.tensor([1.0]), "rng": np.array([2]), "history": [3.0]}
    assert same(a, a)
    assert not same(a, {**a, "weights": torch.tensor([1.01])})
