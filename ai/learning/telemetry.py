"""Read-only telemetry for the already verified allocation. Never selects a CUDA device."""

import csv
import io
import json
import os
import subprocess
import threading
import time

from execution.common import write


def query(fields, kind="gpu"):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}=" + fields, "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    return [[v.strip() for v in row] for row in csv.reader(io.StringIO(result.stdout)) if row]


def own_gpu_uuid(rows, pid):
    matches = {row[1] for row in rows if len(row) == 2 and row[0] == str(pid)}
    if len(matches) != 1:
        raise ValueError("Cannot identify the worker's single allocated GPU; telemetry unavailable")
    return matches.pop()


def parse_gpu(rows, uuid):
    matches = [row for row in rows if row[0] == uuid]
    if len(matches) != 1 or len(matches[0]) != 7:
        raise ValueError("Allocated GPU telemetry missing/ambiguous")
    fields = (
        "utilization_pct",
        "memory_activity_pct",
        "used_mib",
        "total_mib",
        "power_w",
        "temperature_c",
    )
    values = [
        float(v) if v not in {"N/A", "[N/A]", "[Not Supported]"} else None for v in matches[0][1:]
    ]
    return dict(
        zip(fields, values), uuid=uuid, scope="allocated device; not process-only utilization"
    )


class Telemetry:
    def __init__(self, output):
        self.output = output
        self.stop_event = threading.Event()
        self.thread = None
        self.count = 0
        self.usage_sum = 0.0
        self.usage_count = 0
        self.usage_max = 0.0
        self.memory_max = 0.0

    def start(self):
        # Called only after DIS01/owner/GPU-count validation. Initialize this PID's context.
        import torch

        self.marker = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        try:
            self.uuid = own_gpu_uuid(query("pid,gpu_uuid", "compute-apps"), os.getpid())
        except Exception as exc:
            write(
                self.output / "gpu-live.json",
                dict(status="UNAVAILABLE", reason=str(exc), timestamp=time.time()),
            )
            return
        self.thread = threading.Thread(
            target=self.loop, name="allocated-gpu-telemetry", daemon=True
        )
        self.thread.start()

    def loop(self):
        with (self.output / "gpu-samples.jsonl").open("x") as log:
            while not self.stop_event.is_set():
                try:
                    sample = parse_gpu(
                        query(
                            "uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"
                        ),
                        self.uuid,
                    )
                    sample.update(status="SAMPLED", timestamp=time.time())
                    self.count += 1
                    if sample["utilization_pct"] is not None:
                        self.usage_count += 1
                        self.usage_sum += sample["utilization_pct"]
                        self.usage_max = max(self.usage_max, sample["utilization_pct"])
                    self.memory_max = max(self.memory_max, sample["used_mib"] or 0)
                except Exception as exc:
                    sample = dict(status="UNAVAILABLE", reason=str(exc), timestamp=time.time())
                write(self.output / "gpu-live.json", sample)
                log.write(json.dumps(sample, allow_nan=False) + "\n")
                log.flush()
                self.stop_event.wait(2)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=8)
            if self.thread.is_alive():
                raise RuntimeError("Telemetry did not finish; output cannot be sealed")
        write(
            self.output / "gpu-summary.json",
            dict(
                samples=self.count,
                mean_utilization_pct=self.usage_sum / self.usage_count
                if self.usage_count
                else None,
                max_utilization_pct=self.usage_max if self.usage_count else None,
                max_device_used_mib=self.memory_max if self.count else None,
                period_seconds=2,
                includes="all phases actually executed by this worker; device-level samples",
            ),
        )
