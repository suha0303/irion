"""Finite wall-time detection for a frozen simulator clock, independent of phase timers."""

import math


class SimulationClockStalled(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(
            f"simulation_clock_stalled: no PX4 clock advance for {evidence['stalled_wall_s']:.1f}s "
            f"in phase {evidence['phase']} (limit {evidence['limit_wall_s']:g}s)"
        )


class SimulationClockGuard:
    def __init__(self, wall_now, limit_wall_s=300.0):
        if not math.isfinite(limit_wall_s) or limit_wall_s <= 0:
            raise ValueError("Positive finite clock-stall limit required")
        self.limit = limit_wall_s
        self.last_advance_wall = wall_now
        self.last_clock = 0.0

    def observe(self, wall_now, px4_boot_s, phase):
        if not math.isfinite(px4_boot_s) or px4_boot_s < self.last_clock:
            raise RuntimeError("PX4 clock invalid or reversed")
        if px4_boot_s > self.last_clock:
            self.last_clock = px4_boot_s
            self.last_advance_wall = wall_now
        stalled = wall_now - self.last_advance_wall
        if stalled >= self.limit:
            raise SimulationClockStalled(dict(
                phase=phase, px4_boot_s=px4_boot_s, stalled_wall_s=stalled,
                limit_wall_s=self.limit, last_clock_advance_wall=self.last_advance_wall,
            ))


def record_secondary_failure(state, stage, error):
    """Keep the execution failure as the root cause if later finalization also fails."""
    reason = f"{stage}: {error}"
    failure = dict(stage=stage, reason=str(error))
    if state.get("primary_failure"):
        state.setdefault("secondary_failures", []).append(failure)
    else:
        state["primary_failure"] = failure
        state["termination_reason"] = reason
    state["status"] = "failed"
