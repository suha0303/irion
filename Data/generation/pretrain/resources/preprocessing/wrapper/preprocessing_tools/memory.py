"""Recheck actual allocated CPU RAM before the unchanged dual-ridge solver."""

import re
from pathlib import Path


def allocated_memory_gate(audit, environment, meminfo=None):
    job = environment["job"]
    tres = job.get("AllocTRES", job.get("TRES", ""))
    match = re.search(r"(?:^|,)mem=(\d+(?:\.\d+)?)([KMGT]?)B?(?:,|$)", tres)
    if not match:
        raise ValueError("actual Slurm CPU RAM allocation unverified")
    # Slurm bare memory amounts are MiB; this never uses the GPU's VRAM.
    unit = match[2] or "M"
    allocated = int(float(match[1]) * 1024 ** ("KMGT".index(unit) + 1))
    text = Path("/proc/meminfo").read_text() if meminfo is None else meminfo
    match = re.search(r"^MemAvailable:\s+(\d+) kB$", text, re.M)
    if not match:
        raise ValueError("compute-node available CPU RAM unverified")
    available = int(match[1]) * 1024
    memory = audit["memory"]
    # Exactly the independent audit's declared conservative planning expression.
    needed = memory["planning_working_bytes"] + memory["runtime_reserve_bytes"]
    if needed > 0.75 * min(available, allocated):
        raise ValueError(
            "ridge planned CPU memory exceeds actual allocation/available RAM; no subsampling/baseline omission"
        )
    return dict(
        allocated_bytes=allocated,
        available_bytes=available,
        required_bytes=needed,
        passed=True,
        peak_rss_measured=False,
        solver="unchanged dual ridge",
    )
