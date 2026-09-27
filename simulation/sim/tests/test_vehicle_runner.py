"""Run orchestration wiring with fake processes/MAVLink; no flight or sockets."""

import fcntl
import json
import shutil
import struct
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from capstone_sim import collection_config, runner, vehicle_assets, vehicle_runtime
from capstone_sim.profile import load_battery, load_profile


class RecorderFixture:
    def __init__(self, run, *, timing=False, vehicle_profile=None):
        self.run = run
        self.selected_vehicle = vehicle_profile
        self.through, self.count, self.latest = 0, 0, {}
        (run / "observations/metadata.json").write_text('{"vehicle": {}}')
        (run / "evaluation/truth.jsonl").write_text('{"time_us":0,"model_valid":true}\n')
        self.closed = self.released = False

    def close(self):
        self.closed = True

    def release(self):
        self.released = True


def message(kind, **fields):
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: 1, **fields)


class VehicleRunnerTest(unittest.TestCase):
    def fixture(self, directory):
        root = directory / "simulation"
        root.mkdir()
        prepared = root / "build/vehicles/test"
        prepared.mkdir(parents=True)
        example = runner.ROOT / "profiles/examples/hex_example/vehicle.yaml"
        vehicle, pack = load_profile(example), load_battery(example)
        for name in ("vehicle.yaml", "battery.json", "propulsion.json"):
            shutil.copy2(example.parent / name, prepared / name)
        (prepared / "profile.hpp").write_text("// mocked compiled header")
        (prepared / "libDronePowertrain.so").write_text("mocked plugin")
        selected = dict(
            vehicle=vehicle,
            battery=pack,
            propulsion=vehicle["propulsion"],
            observation_vehicle=vehicle_runtime.observation_profile(vehicle, pack),
            hover_estimate=vehicle_runtime.hover_estimate(vehicle, pack),
        )
        for name in (
            "artifacts/source_lock.json",
            "artifacts/runtime-manifest.json",
            "build/libDronePowertrain.so",
            "build/current/libDronePowertrain.so",
            "runtime/gazebo-transport/lib/libgz-transport13.so.13.6.0",
            "px4",
        ):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        for name in (
            "profiles/cmu_vah07_4s2p.json",
            "profiles/x500_reference.yaml",
            "artifacts/propulsion-comparison.json",
        ):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runner.ROOT / name, path)
        return root, prepared, selected

    def run_mocked(
        self,
        directory,
        *,
        custom=True,
        configuration=None,
        scenario="hover_normal",
        gate=None,
        recorder_error=False,
        instance=0,
    ):
        root, prepared, selected = self.fixture(directory)
        world = root / "mock-world.sdf"
        world.write_text("<sdf/>")
        connection = MagicMock()
        launch_calls, recorders = [], []
        processes = [
            SimpleNamespace(pid=60000 + i, poll=lambda: None, returncode=0) for i in range(2)
        ]

        def construct_recorder(*args, **kwargs):
            if recorder_error:
                raise ValueError("mocked startup recorder failure")
            recorder = RecorderFixture(*args, **kwargs)
            recorders.append(recorder)
            return recorder

        def launch(*args, **kwargs):
            launch_calls.append((args, kwargs))
            if gate is None:
                raise RuntimeError("mocked stop before process launch")
            return processes[len(launch_calls) - 1]

        batches = 0

        def receive(*_):
            nonlocal batches
            batches += 1
            if batches > 2:
                raise RuntimeError("mocked stop after FC gate check")
            for recorder in recorders:
                recorder.through = 13_000_000 if batches == 1 else 16_000_000
                recorder.count += 1
            data = [
                message(
                    "HEARTBEAT",
                    autopilot=runner.mavutil.mavlink.MAV_AUTOPILOT_PX4,
                    base_mode=0,
                    custom_mode=6 << 16,
                ),
                message(
                    "LOCAL_POSITION_NED",
                    time_boot_ms=13000 if batches == 1 else 16000,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                ),
            ]
            if gate == "complete" and batches == 1:
                parameters = vehicle_assets.fc_parameters(
                    selected["vehicle"],
                    selected["battery"],
                    selected["hover_estimate"]["hover_throttle"],
                )
                types = vehicle_assets.fc_parameter_types(parameters)
                for key, value in parameters.items():
                    wire = (
                        struct.unpack("<f", struct.pack("<i", int(value)))[0]
                        if types[key] == 6
                        else value
                    )
                    data.append(
                        message(
                            "PARAM_VALUE", param_id=key, param_value=wire, param_type=types[key]
                        )
                    )
            return data

        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", root))
            stack.enter_context(patch.object(runner, "BINARY", root / "px4"))
            stack.enter_context(patch.object(runner, "Recorder", side_effect=construct_recorder))
            stack.enter_context(patch.object(runner, "integrity"))
            stack.enter_context(patch.object(runner, "service_capture", return_value=0))
            stack.enter_context(patch.object(runner, "receive_before_capture", side_effect=receive))
            stack.enter_context(patch.object(runner, "stop_owned_process", return_value={}))
            legacy_world = stack.enter_context(
                patch.object(runner, "prepare_world", return_value=world)
            )
            custom_world = stack.enter_context(
                patch.object(vehicle_assets, "prepare_world", return_value=world)
            )
            custom_fc = stack.enter_context(
                patch.object(vehicle_assets, "prepare_fc", return_value=root / "private-etc")
            )
            stack.enter_context(
                patch.object(vehicle_runtime, "read_prepared", return_value=(prepared, selected))
            )
            stack.enter_context(patch.object(collection_config, "assert_frozen"))
            stack.enter_context(
                patch.object(collection_config, "PLUGIN", root / "build/current/libDronePowertrain.so")
            )
            stack.enter_context(
                patch.object(
                    collection_config, "producer_lock_path", return_value=root / "artifacts/runtime-manifest.json"
                )
            )
            stack.enter_context(
                patch.object(collection_config, "transport_environment", side_effect=lambda env: dict(env))
            )
            stack.enter_context(
                patch.object(
                    collection_config,
                    "transport_library",
                    return_value=root / "runtime/gazebo-transport/lib/libgz-transport13.so.13.6.0",
                )
            )
            stack.enter_context(patch.object(collection_config, "prepare_fc", return_value=root / "collection_config-etc"))
            stack.enter_context(patch.object(collection_config, "owned_socket_identity", return_value=None))
            stack.enter_context(patch.object(collection_config, "release_owned_socket", return_value="mocked"))
            stack.enter_context(patch.object(runner.os.path, "lexists", return_value=False))
            popen = stack.enter_context(
                patch.object(runner.subprocess, "Popen", side_effect=launch)
            )
            stack.enter_context(
                patch.object(runner.mavutil, "mavlink_connection", return_value=connection)
            )
            stack.enter_context(
                patch.dict(
                    runner.os.environ,
                    {
                        "CAPSTONE_ESC_ORDER": "stale",
                        "CAPSTONE_ROTOR_COUNT": "8",
                        "X500_CAPACITY": "999",
                    },
                )
            )
            result = runner.run_flight(
                scenario,
                duration=10,
                instance=instance,
                configuration=configuration,
                vehicle_profile=Path("selected.yaml") if custom else None,
            )
        run = Path(result["run"])
        return dict(
            result=result,
            run=run,
            selected=selected,
            prepared=prepared,
            environment=launch_calls[0][1]["env"],
            recorder=recorders[0],
            custom_world=custom_world,
            legacy_world=legacy_world,
            custom_fc=custom_fc,
            connection=connection,
            popen=popen,
        )

    def test_custom_profile_uses_selected_assets_and_sanitizes_old_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_mocked(Path(temporary))
            self.assertEqual(result["result"]["status"], "failed")
            self.assertEqual(result["result"]["reason"], "mocked stop before process launch")
            self.assertEqual(result["environment"]["CAPSTONE_ROTOR_COUNT"], "6")
            self.assertEqual(
                result["environment"]["CAPSTONE_ESC_ORDER"],
                ",".join(result["selected"]["vehicle"]["motor_order"]),
            )
            self.assertEqual(
                float(result["environment"]["X500_CAPACITY"]),
                result["selected"]["battery"]["capacity_ah"],
            )
            self.assertEqual(
                result["environment"]["X500_BATTERY_FILE"],
                str(result["prepared"] / "battery_runtime.txt"),
            )
            self.assertEqual(
                result["environment"]["X500_PROPULSION_FILE"],
                str(result["prepared"] / "propulsion_runtime.txt"),
            )
            self.assertEqual(
                result["recorder"].selected_vehicle, result["selected"]["observation_vehicle"]
            )
            result["legacy_world"].assert_not_called()
            result["custom_world"].assert_called_once()
            self.assertTrue(result["recorder"].closed and result["recorder"].released)
            expected = json.loads((result["run"] / "reference/fc-configuration.json").read_text())
            self.assertFalse(expected["verified"])
            self.assertEqual(expected["expected"]["CA_ROTOR_COUNT"], 6)

    def test_existing_default_and_collection_paths_do_not_opt_into_custom_recording(self):
        for configuration in (None, "collection"):
            with (
                self.subTest(configuration=configuration),
                tempfile.TemporaryDirectory() as temporary,
            ):
                result = self.run_mocked(Path(temporary), custom=False, configuration=configuration)
                self.assertIsNone(result["recorder"].selected_vehicle)
                result["custom_world"].assert_not_called()
                result["custom_fc"].assert_not_called()
                result["legacy_world"].assert_called_once()
                self.assertNotIn("CAPSTONE_ROTOR_COUNT", result["environment"])
                self.assertNotIn("CAPSTONE_ESC_ORDER", result["environment"])

    def test_reference_run_records_nonzero_instance_and_reset_reuses_it(self):
        from capstone_sim import control

        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_mocked(Path(temporary), custom=False, instance=3)
            state = json.loads((result["run"] / "run.json").read_text())
            self.assertEqual(state["instance"], 3)
            self.assertEqual(state["gz_partition"], "x500_" + result["run"].name)
            with patch.object(control, "run_flight", return_value={}) as flight:
                control.control(result["run"], "reset")
            self.assertEqual(flight.call_args.args[3], 3)

    def test_fc_gate_prevents_arming_until_all_selected_parameters_arrive(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_mocked(Path(temporary), gate="missing")
            result["connection"].mav.command_long_send.assert_not_called()
            self.assertGreater(result["connection"].mav.param_request_read_send.call_count, 3)
            state = json.loads((result["run"] / "reference/fc-configuration.json").read_text())
            self.assertFalse(state["verified"])

    def test_fc_gate_decodes_complete_custom_mapping_before_arming(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_mocked(Path(temporary), gate="complete")
            state = json.loads((result["run"] / "reference/fc-configuration.json").read_text())
            self.assertTrue(state["verified"])
            self.assertEqual(state["observed"]["CA_ROTOR_COUNT"], 6)
            commands = [
                call.args[2] for call in result["connection"].mav.command_long_send.call_args_list
            ]
            self.assertEqual(commands, [176, 400])

    def test_payload_hover_parameters_are_recomputed_for_the_actual_body_mass(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_mocked(Path(temporary), scenario="payload_normal")
            expected = vehicle_runtime.hover_estimate(
                result["selected"]["vehicle"], result["selected"]["battery"], payload=0.2
            )["hover_throttle"]
            self.assertAlmostEqual(result["custom_fc"].call_args.kwargs["hover_throttle"], expected)
            self.assertEqual(result["custom_world"].call_args.kwargs["payload"], 0.2)

    def test_custom_profile_cannot_be_mixed_with_x500_configuration_or_variation(self):
        with patch.object(vehicle_runtime, "read_prepared") as prepared:
            for arguments in ({"configuration": "collection"}, {"normal_variation": {}}):
                with self.assertRaisesRegex(ValueError, "cannot be mixed"):
                    runner.run_flight(
                        "hover_normal", vehicle_profile=Path("example.yaml"), **arguments
                    )
            prepared.assert_not_called()

    def test_custom_loss_severity_scales_from_its_own_propulsion_baseline(self):
        vehicle = {"propulsion": {"friction_nms": 8e-5}}
        pack = {"r0_ohm": 0.08, "interconnect_resistance_ohm": 0.01, "capacity_ah": 12}
        for scenario in (
            "propulsion_loss_existing",
            "propulsion_loss_step",
            "propulsion_loss_ramp",
        ):
            with self.subTest(scenario=scenario):
                selected = runner.custom_scenario_parameters(scenario, None, vehicle, pack)
                key = "friction" if "friction" in selected else "fault_value"
                self.assertAlmostEqual(selected[key], 8e-5 * 20)
                self.assertGreater(selected[key], vehicle["propulsion"]["friction_nms"])
                legacy = runner.scenario_parameters(scenario)
                self.assertEqual(legacy[key], 4e-5)
        combined = runner.custom_scenario_parameters("combined_existing", None, vehicle, pack)
        legacy = runner.scenario_parameters("combined_existing")
        self.assertAlmostEqual(combined["r0"], 0.01 + 0.07 * legacy["r0"] / 0.04)
        capacity = runner.custom_scenario_parameters(
            "battery_capacity_existing", None, vehicle, pack
        )
        self.assertAlmostEqual(capacity["capacity"], 12 * 0.7)
        dynamic = runner.custom_scenario_parameters("battery_resistance_step", None, vehicle, pack)
        self.assertEqual(dynamic["fault_value"], 1.4)
        with self.assertRaisesRegex(ValueError, "positive profile baseline friction"):
            runner.custom_scenario_parameters(
                "propulsion_loss_step", None, {"propulsion": {"friction_nms": 0}}, pack
            )

    def test_startup_recorder_failure_preserves_evidence_and_releases_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            caught = None
            try:
                self.run_mocked(directory, recorder_error=True)
            except ValueError as error:
                caught = error
            self.assertIsNotNone(caught, "startup failure must be reraised after cleanup")
            self.assertIn("mocked startup recorder failure", str(caught))
            # Keep the traceback alive: cleanup must not depend on Python's GC.
            try:
                lease = directory / "simulation/.cache/instances/0.lock"
                with lease.open("a") as guard:
                    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                runs = list((directory / "simulation/runs").iterdir())
                self.assertEqual(len(runs), 1)
                state = json.loads((runs[0] / "run.json").read_text())
                self.assertEqual(state["status"], "failed")
                self.assertIn("mocked startup recorder failure", state["termination_reason"])
            finally:
                # A failing regression test must not retain its own fixture FD.
                frame = caught.__traceback__
                while frame is not None:
                    if frame.tb_frame.f_code is runner.run_flight.__code__:
                        owned = frame.tb_frame.f_locals.get("lease")
                        if owned is not None:
                            owned.close()
                    frame = frame.tb_next


if __name__ == "__main__":
    unittest.main()
