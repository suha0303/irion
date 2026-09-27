import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from capstone_sim.recording import Recorder


class TimingTest(unittest.TestCase):
    def run_dir(self, root):
        run = Path(root) / "fixture"
        for part in ["native_logs", "observations"]:
            (run / part).mkdir(parents=True)
        return run

    def append(self, run, timestamp):
        with (run / "native_logs/capture.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "topic": "battery_status",
                        "sample_time_us": timestamp - 1,
                        "available_time_us": timestamp,
                        "fields": {"voltage_v": 15.1, "current_a": 2.2},
                    }
                )
                + "\n"
            )

    def test_opt_in_preserves_observation_bytes_and_tied_watermarks(self):
        results = []
        for timing in [False, True]:
            with tempfile.TemporaryDirectory() as tmp:
                run = self.run_dir(tmp)
                rec = Recorder(run, origin="synthetic_contract_fixture", timing=timing)
                start = time.monotonic_ns()
                for timestamp in [12, 12, 20]:
                    self.append(run, timestamp)
                    rec.consume(scheduled_host_monotonic_ns=start)
                rec.close()
                end = time.monotonic_ns()
                results.append({p.name: p.read_bytes() for p in (run / "observations").iterdir()})
                sidecar = run / "native_logs/producer-timing.jsonl"
                if not timing:
                    self.assertFalse(sidecar.exists())
                    continue
                rows = [json.loads(s) for s in sidecar.read_text().splitlines()]
                commits = [r for r in rows if r["kind"] == "watermark_commit"]
                watermarks = [
                    json.loads(s)
                    for s in (run / "observations/watermarks.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(commits), len(watermarks))
                for row, w in zip(commits, watermarks):
                    for key in [
                        "run_id",
                        "clock_id",
                        "recorded_events",
                        "through_available_time_us",
                        "closed",
                    ]:
                        self.assertEqual(row[key], w[key])
                    keys = [
                        "stream_flush_start",
                        "stream_flush_end",
                        "stream_fsync_start",
                        "stream_fsync_end",
                        "watermark_write_start",
                        "watermark_write_end",
                        "watermark_flush_start",
                        "watermark_flush_end",
                        "watermark_fsync_start",
                        "watermark_fsync_end",
                    ]
                    stamps = [row[k + "_host_monotonic_ns"] for k in keys]
                    self.assertEqual(stamps, sorted(stamps))
                    self.assertLessEqual(start, min(stamps))
                    self.assertLessEqual(max(stamps), end)
                self.assertEqual(
                    [r["through_available_time_us"] for r in commits], [11, 11, 19, 20]
                )
        self.assertEqual(results[0], results[1])

    def test_partial_line_and_fsync_delay_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_dir(tmp)
            rec = Recorder(run, timing=True)
            self.append(run, 10)
            raw = run / "native_logs/capture.jsonl"
            full = raw.read_bytes()
            raw.write_bytes(full[:20])
            rec.consume()
            self.assertEqual(rec.count, 0)
            with raw.open("ab") as f:
                f.write(full[20:])
            import os

            fsync = os.fsync

            def delayed(fd):
                time.sleep(0.003)
                fsync(fd)

            with patch("capstone_sim.recording.os.fsync", side_effect=delayed):
                rec.consume()
            rec.close()
            rows = [
                json.loads(s)
                for s in (run / "native_logs/producer-timing.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[0]["kind"], "capture_empty")
            commit = next(r for r in rows if r["kind"] == "watermark_commit")
            self.assertEqual(commit["pending_bytes_before"], 20)
            self.assertEqual(commit["native_read_byte_start"], 20)
            self.assertEqual(commit["native_read_byte_end"], len(full))
            for stage in ["stream_fsync", "watermark_fsync"]:
                self.assertGreaterEqual(
                    commit[stage + "_end_host_monotonic_ns"]
                    - commit[stage + "_start_host_monotonic_ns"],
                    3000000,
                )

    def test_failed_stream_never_emits_successful_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_dir(tmp)
            rec = Recorder(run, timing=True)
            rec.stream.close()
            rec.stream = open("/dev/full", "w")
            rec.stream.write("fail")
            with self.assertRaises(OSError):
                rec.flush(True)
            rec.release()
            self.assertEqual((run / "observations/watermarks.jsonl").read_text(), "")
            self.assertEqual((run / "native_logs/producer-timing.jsonl").read_text(), "")

    def test_sidecar_failure_is_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_dir(tmp)
            rec = Recorder(run, timing=True)
            rec.timing.close()
            rec.timing = open("/dev/full", "w")
            with self.assertRaises(OSError):
                rec.consume()
            rec.release()


if __name__ == "__main__":
    unittest.main()
