"""Measured snapshots, launch-boundary isolation, and native-truth admission."""

import contextlib
import copy
import fcntl
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_vehicle_runner as runner_fixture
from capstone_sim import battery_aging as aging
from capstone_sim import cli, collection_config
from capstone_sim import control as controls


class BatteryAgingTest(unittest.TestCase):
    def setUp(self):
        self.catalog = aging.load_catalog()
        self.pack = collection_config.battery()

    def test_measured_stages_have_disjoint_fit_development_holdout_rows(self):
        self.assertEqual(len(self.catalog["stages"]), 17)
        self.assertEqual(self.catalog["source"]["file"], "VAH01.csv")
        capacities = []
        for stage in self.catalog["stages"]:
            capacities.append(stage["cell"]["capacity_ah"])
            windows = stage["mission_rows"]
            self.assertEqual([len(windows[k]) for k in ("fit", "development", "holdout")], [4, 4, 4])
            ordered = windows["fit"] + windows["development"] + windows["holdout"]
            for a, b in zip(ordered, ordered[1:]):
                self.assertLessEqual(a[1], b[0])
            self.assertLessEqual(max(w["rmse_cell_v"] for w in stage["holdout"]), .1)
            self.assertEqual(stage["active_bounds"], [0, 0, 0, 0])
            self.assertGreater(stage["jacobian_condition"], 1)
        self.assertGreater(capacities[0], capacities[-1])
        self.assertGreater(capacities[-1] / capacities[0], .8)
        self.assertLess(capacities[-1] / capacities[0], .9)

    def test_normal_and_aged_packs_use_same_cell_to_pack_rules_and_keep_nameplate(self):
        normal = aging.resolve("hover_normal", None, None, self.pack)
        aged = aging.resolve("battery_aging_existing", "rpt_16", None, self.pack)
        for selection, index in ((normal, 0), (aged, 16)):
            cell = self.catalog["stages"][index]["cell"]
            p = selection["initial"]
            self.assertAlmostEqual(p["capacity_ah"], 4 * cell["capacity_ah"])
            self.assertAlmostEqual(p["r0_ohm"], cell["r0_ohm"] + .005)
            self.assertEqual(p["ocv_cell_v"], cell["ocv_cell_v"])
            self.assertEqual(selection["nominal_capacity_ah"], 12)
            self.assertTrue(selection["fixed_for_flight"])
        self.assertNotEqual(normal["initial"]["capacity_ah"], aged["initial"]["capacity_ah"])

    def test_linear_step_and_ramp_have_identical_endpoints_and_explicit_times(self):
        step = aging.resolve("battery_aging_step", "rpt_16", None, self.pack)
        ramp = aging.resolve("battery_aging_ramp", "rpt_16", 30, self.pack)
        self.assertEqual(step["initial"], ramp["initial"])
        self.assertEqual(step["target"], ramp["target"])
        self.assertEqual(step["transition_s"], 1)
        self.assertEqual(ramp["transition_s"], 30)

    def test_aged_flights_default_to_rpt16_and_other_stages_cannot_be_launched(self):
        for scenario in aging.SCENARIOS:
            with self.subTest(scenario=scenario):
                default = aging.resolve(scenario, None, None, self.pack)
                explicit = aging.resolve(scenario, "rpt_16", None, self.pack)
                self.assertEqual(default, explicit)
                self.assertEqual(default["target_stage"], "rpt_16")
                self.assertEqual(default["initial_stage"],
                                 "rpt_00" if scenario in aging.DYNAMIC else "rpt_16")
        for scenario in (*aging.SCENARIOS, "motor_resistance_existing"):
            for index in range(16):
                with self.subTest(scenario=scenario, stage=index), self.assertRaises(ValueError):
                    aging.resolve(scenario, f"rpt_{index:02d}", None, self.pack)

    def test_policy_is_bound_to_catalog_and_historical_runs_keep_their_original_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "catalog.json"
            for historical in (False, True):
                catalog = copy.deepcopy(self.catalog)
                selection = aging.resolve("battery_aging_existing", None, None, self.pack)
                selection.update(initial_stage="rpt_12", target_stage="rpt_12")
                selection["initial"] = aging.pack_snapshot(catalog, "rpt_12", self.pack)
                selection["target"] = copy.deepcopy(selection["initial"])
                if historical:
                    catalog.pop("flight_policy")
                    selection.pop("flight_policy")
                path.write_text(json.dumps(catalog))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                selection["catalog_sha256"] = digest
                if historical:
                    aging.verify_selection(selection, path, self.pack, "battery_aging_existing", digest)
                else:
                    with self.assertRaisesRegex(ValueError, "rpt_16-only"):
                        aging.verify_selection(selection, path, self.pack, "battery_aging_existing", digest)

    def test_bad_selection_and_failed_or_nonfinite_catalog_are_rejected(self):
        for args in (("hover_normal", "rpt_16", None, "collection"),
                     ("battery_aging_existing", "rpt_00", None, "collection"),
                     ("battery_aging_existing", "rpt_16", 1, "collection"),
                     ("battery_aging_step", "rpt_16", math.nan, "collection"),
                     ("battery_aging_step", "rpt_16", 1, "unsupported")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                aging.validate_selection(*args)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "catalog.json"
            for damage in ("failed", "nan", "duplicate"):
                catalog = copy.deepcopy(self.catalog)
                if damage == "failed":
                    catalog["holdout_status"] = "failed"
                elif damage == "nan":
                    catalog["stages"][1]["cell"]["r0_ohm"] = math.nan
                else:
                    catalog["stages"][1]["id"] = "rpt_00"
                path.write_text(json.dumps(catalog))
                with self.assertRaises(ValueError):
                    aging.load_catalog(path)

    def truth(self, selection, fraction, time_us):
        row = {k: a + fraction * (selection["target"][k] - a)
               for k, a in selection["initial"].items() if k != "ocv_cell_v"}
        row.update(time_us=time_us, battery_stage_fraction=fraction,
                   battery_stage_onset_us=1_000_000, fault_applied=fraction > 0,
                   fault_applied_time_us=1_001_000,
                   battery_ocv_cell_v=[a + fraction * (b - a) for a, b in
                                       zip(selection["initial"]["ocv_cell_v"],
                                           selection["target"]["ocv_cell_v"])])
        return row

    def test_truth_rejects_wrong_timing_parameters_and_incomplete_transition(self):
        selection = aging.resolve("battery_aging_step", "rpt_16", 2, self.pack)
        rows = [self.truth(selection, f, t) for f, t in
                ((0, 0), (.5, 2_000_000), (1, 3_000_000))]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "truth.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(aging.verify_truth(path, selection)["status"], "passed")
            quantized = copy.deepcopy(rows)
            quantized[1]["time_us"] -= 1
            path.write_text("".join(json.dumps(r) + "\n" for r in quantized))
            self.assertEqual(aging.verify_truth(path, selection)["status"], "passed")
            for damage in ("time", "parameter", "unfinished"):
                bad = copy.deepcopy(rows)
                if damage == "time":
                    bad[1]["time_us"] += 100_000
                elif damage == "parameter":
                    bad[1]["capacity_ah"] += .1
                else:
                    bad.pop()
                path.write_text("".join(json.dumps(r) + "\n" for r in bad))
                with self.assertRaises(ValueError):
                    aging.verify_truth(path, selection)

    def test_cli_carries_selected_stage_and_transition(self):
        args = ["capstone-sim", "run", "--configuration", "collection", "--scenario",
                "battery_aging_ramp", "--battery-stage", "rpt_16", "--battery-transition-s",
                "30", "--fault-at-s", "180", "--duration", "480"]
        with patch("sys.argv", args), patch.object(cli, "run_flight") as run:
            run.return_value = {"status": "completed"}
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(run.call_args.kwargs["battery_stage"], "rpt_16")
        self.assertEqual(run.call_args.kwargs["battery_transition_s"], 30)

    def test_live_control_cannot_override_a_measured_battery(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / ("a" * 32)
            (run / "evaluation").mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"run_id": run.name, "status": "running"}))
            (run / "evaluation/battery-selection.json").write_text("{}")
            with patch.object(controls.subprocess, "run") as service:
                for target in ("battery_r0", "battery_r1"):
                    with self.assertRaisesRegex(ValueError, "locked"):
                        controls.control(run, "apply_parameter_change", target=target, value=.03)
                service.assert_not_called()

    def test_bound_catalog_does_not_allow_relabeling_an_aged_battery_as_normal(self):
        selection = aging.resolve("battery_aging_existing", "rpt_16", None, self.pack)
        with self.assertRaisesRegex(ValueError, "scenario"):
            aging.verify_selection(selection, aging.CATALOG, self.pack, "hover_normal",
                                   selection["catalog_sha256"])
        aging.verify_selection(selection, aging.CATALOG, self.pack, "battery_aging_existing",
                               selection["catalog_sha256"])
        selection["initial"]["r0_ohm"] *= 1.1
        with self.assertRaisesRegex(ValueError, "changed"):
            aging.verify_selection(selection, aging.CATALOG, self.pack, "battery_aging_existing",
                                   selection["catalog_sha256"])

    def test_fit_respects_active_collection_lock_before_reading_or_writing_data(self):
        from capstone_sim import battery_fit

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "simulation"
            lock = root.parent / "Data/operations/collection.lock"
            lock.parent.mkdir(parents=True)
            with lock.open("a") as stream, patch.object(battery_fit, "ROOT", root):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(battery_fit, "source_data") as fit:
                    with self.assertRaisesRegex(RuntimeError, "active"):
                        battery_fit.fit_to_file()
                    fit.assert_not_called()

    def test_reset_preserves_stage_duration_and_refuses_changed_catalog(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / ("a" * 32)
            (run / "evaluation").mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"run_id": run.name, "status": "completed",
                                                       "configuration": "collection", "instance": 2}))
            (run / "evaluation/scenario.json").write_text(json.dumps({
                "name": "battery_aging_ramp", "seed": 7, "requested_duration_s": 480,
                "parameters": {"fault_at": 180},
            }))
            selection = aging.resolve("battery_aging_ramp", "rpt_16", 30, self.pack)
            path = run / "evaluation/battery-selection.json"
            path.write_text(json.dumps(selection))
            with patch.object(controls, "run_flight", return_value={}) as flight:
                controls.control(run, "reset")
                self.assertEqual(flight.call_args.kwargs["battery_stage"], "rpt_16")
                self.assertEqual(flight.call_args.kwargs["battery_transition_s"], 30)
                selection["catalog_sha256"] = "0" * 64
                path.write_text(json.dumps(selection))
                with self.assertRaisesRegex(ValueError, "catalog changed"):
                    controls.control(run, "reset")


class BatteryLaunchTest(unittest.TestCase):
    def test_motor_losses_share_the_normal_battery_and_accept_a_fixed_aged_stage(self):
        fixture = runner_fixture.VehicleRunnerTest()
        baseline = aging.resolve("hover_normal", None, None, collection_config.battery())["initial"]
        for scenario in ("propulsion_loss_existing",):
            for stage in (None, "rpt_16"):
                with self.subTest(scenario=scenario, stage=stage), tempfile.TemporaryDirectory() as folder:
                    result = fixture.run_mocked(
                        Path(folder), custom=False, configuration="collection", scenario=scenario,
                        flight_options={"battery_stage": stage},
                    )
                    selection = json.loads((result["run"] / "evaluation/battery-selection.json").read_text())
                    self.assertTrue(selection["fixed_for_flight"])
                    self.assertEqual(selection["initial"], selection["target"])
                    self.assertEqual(selection["initial_stage"], stage or "rpt_00")
                    if stage is None:
                        self.assertEqual(selection["initial"], baseline)
                    self.assertNotIn("X500_R0", result["environment"])
                    self.assertNotIn("X500_CAPACITY", result["environment"])

    def test_selected_snapshot_reaches_native_launch_without_leaking_into_nameplate(self):
        fixture = runner_fixture.VehicleRunnerTest()
        with tempfile.TemporaryDirectory() as folder:
            result = fixture.run_mocked(
                Path(folder), custom=False, configuration="collection",
                scenario="battery_aging_existing", flight_options={"battery_stage": "rpt_16"},
            )
            env = result["environment"]
            selection = json.loads((result["run"] / "evaluation/battery-selection.json").read_text())
            self.assertEqual(env["X500_BATTERY_PROFILE_LOCKED"], "1")
            self.assertNotIn("X500_CAPACITY", env)
            loaded = list(map(float, Path(env["X500_BATTERY_FILE"]).read_text().split()))
            self.assertEqual(loaded[3], selection["initial"]["capacity_ah"])
            self.assertEqual(env["X500_BATTERY_FILE"], str(result["run"] / "evaluation/battery-initial.txt"))
            self.assertEqual(selection["initial"], selection["target"])
            self.assertEqual(selection["nominal_capacity_ah"], 12)
            self.assertEqual(result["result"]["status"], "failed")  # mocked launch; never a real flight

    def test_normal_and_transition_use_same_initial_battery_and_real_target_file(self):
        fixture = runner_fixture.VehicleRunnerTest()
        selections = []
        for scenario in ("hover_normal", "battery_aging_step", "battery_aging_ramp"):
            with tempfile.TemporaryDirectory() as folder:
                options = {} if scenario == "hover_normal" else {
                    "battery_stage": "rpt_16", "battery_transition_s": 3.0, "fault_at_s": 180,
                }
                result = fixture.run_mocked(
                    Path(folder), custom=False, configuration="collection", scenario=scenario,
                    duration=240, flight_options=options,
                )
                env = result["environment"]
                selections.append(json.loads((result["run"] / "evaluation/battery-selection.json").read_text()))
                self.assertEqual(env["X500_BATTERY_PROFILE_LOCKED"], "1")
                if scenario != "hover_normal":
                    self.assertEqual(env["X500_FAULT"], "battery_stage")
                    self.assertEqual(float(env["X500_FAULT_RAMP_S"]), 3)
                    self.assertTrue(Path(env["X500_BATTERY_TARGET_FILE"]).is_file())
                else:
                    self.assertNotIn("X500_BATTERY_TARGET_FILE", env)
        self.assertEqual(selections[0]["initial"], selections[1]["initial"])
        self.assertEqual(selections[0]["initial"], selections[2]["initial"])


if __name__ == "__main__":
    unittest.main()
