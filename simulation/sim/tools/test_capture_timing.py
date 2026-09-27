"""Cross-process Linux host-clock and native-sidecar tests, without SITL."""

import json
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
binary = ROOT / "build/capture_timing_fixture"
with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as tmp:
    root = Path(tmp)
    raw, trace = root / "capture.jsonl", root / "timing.jsonl"
    before = time.monotonic_ns()
    subprocess.run([str(binary), str(raw), str(trace)], check=True)
    after = time.monotonic_ns()
    rows = [json.loads(s) for s in trace.read_text().splitlines()]
    assert len(rows) == 3
    assert [r["capture_start_px4_hrt_us"] for r in rows] == [1000, 1000, 900000000000]
    for i, row in enumerate(rows):
        stamps = [v for k, v in row.items() if k.endswith("_host_monotonic_ns")]
        assert all(before <= t <= after for t in stamps), "native syscall != Python host clock"
        assert row["first_event_index"] == row["last_event_index"] == i + 1
        assert row["event_count"] == 1
        assert row["native_byte_end"] > row["native_byte_start"]
        assert (
            row["native_flush_end_host_monotonic_ns"] >= row["native_flush_start_host_monotonic_ns"]
        )
        event = json.loads(raw.read_bytes()[row["native_byte_start"] : row["native_byte_end"]])
        assert event["available_time_us"] == row["last_available_time_us"]
    disabled = root / "disabled.jsonl"
    subprocess.run([str(binary), str(disabled), "disabled"], check=True)
    assert raw.read_bytes() == disabled.read_bytes(), "instrumentation changed raw bytes"
    failed = subprocess.run(
        [str(binary), str(root / "failure.jsonl"), "/dev/full"], capture_output=True
    )
    assert failed.returncode == 74
    print(
        json.dumps(
            {
                "origin": "synthetic_clock_writer_fixture",
                "passed": True,
                "native_response_rows": len(rows),
                "python_clock": time.get_clock_info("monotonic").implementation,
                "host_bracket_ns": [before, after],
                "sim_pause_and_jump_tested": True,
                "sidecar_failure_exit_code": failed.returncode,
            }
        )
    )
