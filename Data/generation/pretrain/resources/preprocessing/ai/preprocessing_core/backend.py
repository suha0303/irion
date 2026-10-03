"""Explicit R1P1 LSTM backend and checkpoint compatibility policy."""

import importlib.metadata
import os
import platform
from contextlib import contextmanager

import numpy as np
import torch

SCHEMA = "round001-r1p1-checkpoint-v1"


def cudnn_flags():
    return {
        k: getattr(torch.backends.cudnn, k)
        for k in ("enabled", "benchmark", "benchmark_limit", "deterministic", "allow_tf32")
    }


def validate_backend(config):
    expected = {"cuda": "aten_cuda", "cpu": "cpu_regression"}
    if (
        config.get("device") not in expected
        or config.get("lstm_backend") != expected[config["device"]]
    ):
        raise ValueError("explicit LSTM backend/device policy required; no fallback")


@contextmanager
def backend_scope(config):
    validate_backend(config)
    before = cudnn_flags()
    # Pass every flag: cudnn.flags defaults would otherwise alter unrelated settings.
    with torch.backends.cudnn.flags(**{**before, "enabled": False}):
        yield
    if cudnn_flags() != before:
        raise RuntimeError("cuDNN scope did not restore flags")


def execution_environment(device):
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "safetensors": importlib.metadata.version("safetensors"),
        "device": device.type,
        "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_flags": cudnn_flags(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "threads": torch.get_num_threads(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "gpu_capability": list(torch.cuda.get_device_capability())
        if device.type == "cuda"
        else None,
    }


def resume_guard(checkpoint, config, signature, sources, environment):
    for key, expected in {
        "schema": SCHEMA,
        "config": config,
        "data_signature": signature,
        "source_hashes": sources,
        "environment": environment,
        "lstm_backend": config["lstm_backend"],
        "dropout": 0.1,
    }.items():
        if checkpoint.get(key) != expected:
            raise ValueError("resume mismatch at checkpoint." + key)


def check_model(model, device):
    if torch.backends.cudnn.enabled:
        raise ValueError("cuDNN enabled inside explicit ATen scope")
    for module in (model.encoder, model.decoder):
        if module.dropout != 0.1 or module.num_layers != 2:
            raise ValueError("fixed two-layer LSTM dropout0.1 required")
    if any(p.device.type != device.type for p in model.parameters()):
        raise ValueError("model device differs from backend policy")


def validate_operators(operators, device, training):
    # This predicate runs even when cuDNN is disabled; it is not a cuDNN kernel.
    if any("cudnn" in name.lower() and name != "aten::cudnn_is_acceptable" for name in operators):
        raise ValueError("cuDNN operator observed in ATen policy")
    if "aten::lstm" not in operators:
        raise ValueError("missing actual LSTM operator evidence")
    if device.type == "cuda" and "aten::_thnn_fused_lstm_cell" not in operators:
        raise ValueError("missing native ATen CUDA LSTM cell evidence")
    if training and "aten::dropout" not in operators:
        raise ValueError("missing training dropout operator evidence")


@contextmanager
def trace_batch(folder, phase, device, training, enabled):
    """Observe an existing batch; no additional forward/backward or fit."""
    if not enabled:
        yield
        return
    from .control import write

    # CPU dispatcher events include CUDA ATen calls; no CUPTI installation is needed.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        yield
    operators = {event.key: event.count for event in profile.key_averages()}
    write(
        folder / ("backend-" + phase + ".json"),
        {
            "scope": "existing first batch only; dispatcher events, not kernel timing",
            "device": device.type,
            "training": training,
            "cudnn_flags": cudnn_flags(),
            "operators": operators,
        },
    )
    validate_operators(operators, device, training)
