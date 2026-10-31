"""Reviewable candidates, never an automatically executed experiment queue."""

from .models import ModelSpec

PROFILES = {
    "reference": dict(hidden=32, layers=2, heads=4, batches=[32, 64, 128]),
    "small": dict(hidden=64, layers=2, heads=4, batches=[128, 256, 512]),
    "balanced": dict(hidden=128, layers=2, heads=4, batches=[128, 256, 512, 1024]),
    "deeper": dict(hidden=128, layers=4, heads=4, batches=[128, 256, 512]),
    "wide": dict(hidden=256, layers=4, heads=8, batches=[64, 128, 256, 512]),
    "large": dict(hidden=512, layers=4, heads=8, batches=[32, 64, 128, 256]),
}


def specs(profile):
    p = PROFILES[profile]
    return [
        ModelSpec(name, p["hidden"], p["layers"], p["heads"]) for name in ("lstm", "transformer")
    ]


def select_batch(measurements):
    """Prefer throughput, never allocate useless tensors to reach a memory target."""
    admitted = [
        r
        for r in measurements
        if r.get("passed")
        and r["windows_per_s"] > 0
        and r["peak_reserved_bytes"] <= 0.85 * r["free_before_bytes"]
        and r["free_after_bytes"] >= 4 * 1024**3
    ]
    if not admitted:
        raise ValueError("No admitted measured batch; no automatic retry")
    best = max(r["windows_per_s"] for r in admitted)
    return min((r for r in admitted if r["windows_per_s"] >= 0.95 * best), key=lambda r: r["batch"])
