"""Acknowledge a native fault schedule without blocking the raw-capture loop."""

import math
import re
import subprocess
import time


def schedule_evidence(origin_s, offset_s):
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (origin_s, offset_s)):
        raise ValueError("Finite mission fault times required")
    origin_us, offset_us = round(origin_s * 1e6), round(offset_s * 1e6)
    if not 0 <= origin_us <= 10**15 or not 0 < offset_us <= 1800_000_000:
        raise ValueError("Mission fault times outside supported bounds")
    if any(abs(v * 1e6 - us) > 1e-6 for v, us in ((origin_s, origin_us), (offset_s, offset_us))):
        raise ValueError("Mission fault times must resolve to integer microseconds")
    return dict(
        clock="px4-boot",
        origin="airborne_path_zero",
        mission_origin_time_us=origin_us,
        requested_offset_us=offset_us,
        scheduled_time_us=origin_us + offset_us,
        acknowledged=False,
        applied_time_us=None,
    )


def complete_evidence(schedule, applied_us, duration_s):
    """Truth, not transport acknowledgement, proves the physical change happened."""
    if (
        not isinstance(schedule, dict) or schedule.get("acknowledged") is not True
        or type(applied_us) is not int
        or schedule["scheduled_time_us"] != schedule["mission_origin_time_us"] + schedule["requested_offset_us"]
        or not schedule["scheduled_time_us"] <= applied_us < schedule["mission_origin_time_us"] + round(duration_s * 1e6)
    ):
        raise ValueError("Missing, inconsistent or out-of-mission fault application truth")
    return dict(
        schedule,
        applied_time_us=applied_us,
        application_delay_us=applied_us - schedule["scheduled_time_us"],
    )


class MissionFaultRequest:
    def __init__(self, origin_s, offset_s, env):
        self.evidence = schedule_evidence(origin_s, offset_s)
        if not re.fullmatch(r"x500_[0-9a-f]{32}", env.get("GZ_PARTITION", "")) or env.get("GZ_IP") != "127.0.0.1":
            raise ValueError("Run-specific Gazebo partition/IP required for fault schedule")
        self.started = time.monotonic()
        self.finished = False
        self.process = subprocess.Popen(
            [
                "gz", "service", "-s", "/x500_0/schedule_fault",
                "--reqtype", "gz.msgs.Double_V", "--reptype", "gz.msgs.Double_V",
                "--timeout", "3000", "--req",
                f"data: 1 data: {self.evidence['mission_origin_time_us']}",
            ],
            env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )

    def poll(self):
        if self.finished:
            return None
        code = self.process.poll()
        if code is None:
            if time.monotonic() - self.started > 15:
                raise TimeoutError("Native mission fault scheduling acknowledgement timed out")
            return None
        stdout, stderr = self.process.communicate()
        values = [float(v) for v in re.findall(r"^data: ([-+0-9.eE]+)$", stdout, re.MULTILINE)]
        if code or values != [0, self.evidence["scheduled_time_us"]]:
            raise RuntimeError(f"Native mission fault schedule rejected or mismatched: {code}, {stdout!r}, {stderr!r}")
        self.finished = True
        self.evidence["acknowledged"] = True
        return dict(self.evidence)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        for pipe in (self.process.stdout, self.process.stderr):
            if pipe:
                pipe.close()
