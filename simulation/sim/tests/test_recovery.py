import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from capstone_sim.recovery import recover

ROOT = Path(__file__).resolve().parents[2]


class RecoveryTest(unittest.TestCase):
    def test_preserves_original_and_only_recovers_complete_rows(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as tmp:
            root = Path(tmp)
            src = root / "run"
            (src / "observations").mkdir(parents=True)
            data = b'{"schema":"drone.raw.event/2.0","available_time_us":100}\n{"schema":'
            (src / "observations/streams.jsonl").write_bytes(data)
            result = recover(src, root / "recovered")
            self.assertEqual(result["recovered_events"], 1)
            self.assertEqual(result["discarded_incomplete_bytes"], 10)
            self.assertEqual((src / "observations/streams.jsonl").read_bytes(), data)

    def test_os_file_size_limit_produces_real_write_error(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as tmp:
            # EFBIG from an OS limit, not an assertion that ENOSPC handling was tested.
            path = str(Path(tmp) / "limited.jsonl")
            code = """import resource,signal,sys,json
signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE,(64,64))
try:
 with open(sys.argv[1],'wb',buffering=0) as f:
  f.write(b'x'*64)
  f.write(b'y')
except OSError as e:
 print(json.dumps({'errno':e.errno,'message':str(e)}))
 sys.exit(4)
"""
            p = subprocess.run(
                [sys.executable, "-c", code, path], capture_output=True, text=True, check=False
            )
            self.assertEqual(p.returncode, 4)
            self.assertEqual(json.loads(p.stdout)["errno"], 27)
