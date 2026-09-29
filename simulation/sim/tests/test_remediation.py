import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from capstone_sim import cli
from capstone_sim.control import parse_response
from capstone_sim.profile import header_text, load_profile
from capstone_sim.recording import Recorder


class BoundaryTest(unittest.TestCase):
    def test_doctor_missing_spec_fails_before_replacing_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts").mkdir()
            lock = root / "artifacts/source_lock.json"
            lock.write_text('{"previous": "preserve"}\n')
            with patch.object(cli, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "spec"):
                    cli.doctor(root / "artifacts/environment.json")
            self.assertEqual(lock.read_text(), '{"previous": "preserve"}\n')
            self.assertFalse((root / "artifacts/environment.json").exists())

    def test_rpc_rejection_is_not_transport_success(self):
        self.assertEqual(parse_response("apply_parameter_change", 0, "data: -1\n")["exit_code"], 4)
        self.assertEqual(
            parse_response("apply_parameter_change", 0, "data: 1\n")["status"], "duplicate"
        )
        self.assertEqual(parse_response("status", 0, "data: 1000\ndata: 2\n")["revision"], 2)
        self.assertEqual(parse_response("status", 0, "data: -1\ndata: 0\n")["status"], "queued")
        self.assertEqual(parse_response("apply_parameter_change", 0, "")["exit_code"], 4)

    def test_profile_rejects_nonphysical_inertia_and_wrong_order(self):
        base = load_profile()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.yaml"
            for field in ["inertia", "order"]:
                p = copy.deepcopy(base)
                if field == "inertia":
                    p["geometry"]["body_inertia_diagonal_kg_m2"] = [1, 1, 10]
                else:
                    p["motor_order"].reverse()
                path.write_text(yaml.safe_dump(p))
                with self.assertRaises(ValueError):
                    load_profile(path)
        text = header_text(base, "hash")
        self.assertIn("0.174,-0.174,0.06", text)
        self.assertIn("hash", text)

    def test_release_closes_all_descriptors_after_real_write_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "observations").mkdir(parents=True)
            (run / "native_logs").mkdir()
            recorder = Recorder(run)
            recorder.stream.close()
            recorder.stream = open("/dev/full", "w")
            recorder.stream.write("real ENOSPC")
            with self.assertRaises(OSError):
                recorder.flush(True)
            recorder.release()
            self.assertTrue(
                all(s.closed for s in (recorder.stream, recorder.watermarks, recorder.diag))
            )
            self.assertEqual((run / "observations/watermarks.jsonl").read_text(), "")
