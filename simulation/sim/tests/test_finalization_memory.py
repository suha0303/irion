"""Long-run timestamp extraction and hashing must fit parallel flight memory."""

import hashlib
import json
import tempfile
import tracemalloc
import unittest
from pathlib import Path

from capstone_sim.recording import sha
from capstone_sim.runner import fault_evidence


class FinalizationMemoryTest(unittest.TestCase):
    def test_large_truth_preserves_first_timestamps_with_bounded_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "truth.jsonl"
            expected_digest = hashlib.sha256()
            with path.open("wb") as stream:
                for index in range(32768):
                    row = dict(time_us=index * 10000, fault_applied=index >= 100,
                               fault_applied_time_us=1_000_000,
                               fault_boundary_time_us=2_000_000 if index >= 200 else -1,
                               payload="x" * 1024)
                    line = (json.dumps(row) + "\n").encode()
                    stream.write(line)
                    expected_digest.update(line)
            tracemalloc.start()
            try:
                evidence = fault_evidence(path)
                digest = sha(path)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            self.assertEqual(evidence, dict(first_time_us=0, applied_time_us=1_000_000,
                                            boundary_time_us=2_000_000))
            self.assertEqual(digest, expected_digest.hexdigest())
            self.assertLess(peak, 4 * 1024 * 1024)

    def test_empty_missing_and_legacy_fault_time_keep_original_meaning(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "truth.jsonl"
            expected = dict(first_time_us=None, applied_time_us=None, boundary_time_us=None)
            self.assertEqual(fault_evidence(path), expected)
            path.touch()
            self.assertEqual(fault_evidence(path), expected)
            path.write_text('{"time_us":5,"fault_applied":true}\n')
            self.assertEqual(fault_evidence(path), dict(first_time_us=5, applied_time_us=5,
                                                       boundary_time_us=None))
