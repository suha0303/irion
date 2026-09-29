"""Preview preparation is offline; GUI uses a private QGC link and visual guides."""

import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from capstone_sim import collection_config, preview, runner
from capstone_sim.recording import dump, integrity


class RandomPreviewTest(unittest.TestCase):
    def test_plan_does_not_launch_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as td, patch.object(preview, "ROOT", Path(td)):
            path = Path(td) / "artifacts/random-continuous-v3/preview.json"
            with patch.object(preview.subprocess, "run", side_effect=AssertionError("no process")):
                result = preview.prepare(path, 17, 900)
                original = path.read_bytes()
                self.assertFalse(result["flight_started"])
                self.assertFalse(result["data_collection_authorized"])
                preview.prepare(path, 17, 900)
                with self.assertRaises(ValueError):
                    preview.prepare(path, 18, 900)
                with self.assertRaises(ValueError):
                    preview.prepare(path.parent / "../../protected.json", 17, 900)
            self.assertEqual(path.read_bytes(), original)

    def test_explicit_fly_dispatches_one_preview_and_rejects_changed_plan(self):
        with tempfile.TemporaryDirectory() as td, patch.object(preview, "ROOT", Path(td)):
            path = Path(td) / "artifacts/random-continuous-v3/preview.json"
            preview.prepare(path, 17, 900)
            with (
                patch.object(preview.socket, "socket"),
                patch.object(runner, "run_flight", return_value={"status": "completed"}) as flight,
            ):
                preview.fly(path)
                flight.assert_called_once()
                self.assertTrue(flight.call_args.kwargs["gui_preview"])
                self.assertEqual(flight.call_args.kwargs["real_time_factor"], 1)
                record = json.loads(path.read_text())
                record["normal_variation"]["soc"] = 0.5
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    preview.fly(path)
                flight.assert_called_once()

    def test_private_fc_disables_qgc_forwarding_only_in_preview(self):
        source = runner.PX4 / "build/px4_sitl_default/etc"
        for gui in (False, True):
            with tempfile.TemporaryDirectory() as td:
                etc = collection_config.prepare_fc(Path(td), source, gui_preview=gui)
                text = (etc / "init.d-posix/px4-rc.mavlink").read_text()
                self.assertEqual("udp_gcs_port_remote=14550" in text, gui)
                self.assertEqual(" -f" in text, not gui)

    def test_visual_markers_have_no_collisions_and_camera_uses_scoped_environment(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "reference").mkdir()
            world = run / "world.sdf"
            world.write_text(
                '<sdf version="1.9"><world name="x500_world"><scene><grid>false</grid></scene></world></sdf>'
            )
            record = preview.plan(17, 900)
            config = preview.prepare_gui(run, world, record["normal_variation"])
            tree = ET.parse(world)
            self.assertEqual(
                len(tree.findall("world/model")), len(record["normal_variation"]["path"]["program"])
            )
            self.assertFalse(tree.findall(".//collision"))
            self.assertEqual(tree.findtext("world/scene/grid"), "true")
            self.assertNotIn('filename="ComponentInspector"', config.read_text())
            gui = ET.fromstring("<root>" + config.read_text() + "</root>")
            self.assertEqual(
                gui.findtext("plugin[@filename='CameraTracking']/follow_target"), "x500_0"
            )
        env = {"GZ_PARTITION": "x500_synthetic", "GZ_IP": "127.0.0.1"}
        with patch.object(
            preview.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="data: true\n"),
        ) as call:
            result = preview.follow_camera(env)
            self.assertEqual(result["target"], "x500_0")
            self.assertEqual(call.call_count, 2)
            self.assertTrue(all(c.kwargs["env"] == env for c in call.call_args_list))

    def test_successful_preview_cannot_be_admitted_as_collection_data(self):
        spec = importlib.util.spec_from_file_location(
            "assessment_fixture", Path(__file__).with_name("test_run_assessment.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helper = module.RunAssessmentTest()
        with tempfile.TemporaryDirectory() as td:
            run = helper.fixture(Path(td))
            state = json.loads((run / "run.json").read_text())
            state.update(demo_only=True, dataset_admission="excluded_gui_preview")
            dump(run / "run.json", state)
            integrity(run)
            result = helper.assess(run)
            self.assertFalse(result["contract_passed"])
            self.assertIn("GUI preview is excluded from collection datasets", result["errors"])


if __name__ == "__main__":
    unittest.main()
