"""Synthetic functional/latency check, explicitly not real-aircraft
validation.
"""

import gc
import platform
import resource
import time

import numpy as np
import torch

from .catalog import Catalog
from .runtime import Runtime


def benchmark(bundle, *, repeats=5, threads=2):
    if not 1 <= repeats <= 100:
        raise ValueError("Repeats must be 1..100")
    catalog = Catalog(bundle)
    rows = []
    for entry in catalog.list():
        started = time.perf_counter()
        runtime = Runtime(entry["id"], catalog=catalog, threads=threads)
        load_seconds = time.perf_counter() - started
        frames = 100 if entry["task"] == "forecast" else 4800
        a = (
            np.random.default_rng(0)
            .normal(0, 0.1, (1, frames, 19))
            .astype("f4")
        )
        lengths = None if frames == 100 else np.array([frames], dtype="i8")
        first = runtime.predict_scaled(a, lengths)
        durations = []
        max_difference = 0.0
        for _ in range(repeats):
            started = time.perf_counter()
            actual = runtime.predict_scaled(a, lengths)
            durations.append(time.perf_counter() - started)
            max_difference = max(
                max_difference, float(np.max(np.abs(actual - first)))
            )
        rows.append(
            dict(
                id=entry["id"],
                load_seconds=load_seconds,
                p50_ms=float(np.median(durations) * 1000),
                p95_ms=float(np.quantile(durations, 0.95) * 1000),
                repeat_max_abs=max_difference,
                output_shape=list(first.shape),
                frames=frames,
                finite=True,
            )
        )
        del runtime
        gc.collect()
    return dict(
        scope=(
            "synthetic CPU functional benchmark; not real-aircraft accuracy or "
            "live certification"
        ),
        torch=str(torch.__version__),
        platform=platform.platform(),
        processor=platform.processor(),
        threads=threads,
        repeats=repeats,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_initialized=torch.cuda.is_initialized(),
        models=rows,
    )
