"""Profile preparation/export dispatch never implies a flight or AI compatibility."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from capstone_sim import cli, runner, vehicle_runtime

ROOT = Path(__file__).resolve().parents[2]


class VehicleCliTest(unittest.TestCase):
    def invoke(self, arguments):
        output = io.StringIO()
        with (
            patch("sys.argv", ["capstone-sim", *arguments]),
            patch.object(
                cli, "run_flight", side_effect=AssertionError("CLI must not start a flight")
            ),
            patch.object(
                runner, "run_flight", side_effect=AssertionError("Runner must not start a flight")
            ),
            redirect_stdout(output),
        ):
            code = cli.main()
        return code, json.loads(output.getvalue())

    def test_inspect_reads_real_example_without_process_or_flight(self):
        path = ROOT / "profiles/examples/hex_example/vehicle.yaml"
        with patch.object(
            vehicle_runtime.subprocess,
            "run",
            side_effect=AssertionError("Inspect must not launch a process"),
        ):
            code, result = self.invoke(["profile", "inspect", str(path)])
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "PROFILE_VALIDATED_NO_FLIGHT")
        self.assertEqual(result["rotor_count"], 6)
        self.assertEqual(len(result["inputs"]), 3)
        self.assertFalse(result["ai_compatible"])
        self.assertFalse(result["actual_flight_started"])
        self.assertFalse(result["training_started"])

    def test_prepare_dispatches_explicit_build_options_without_starting_flight(self):
        path = ROOT / "profiles/examples/hex_example/vehicle.yaml"
        prepared = dict(status="PREPARED_NO_FLIGHT", manifest="synthetic/manifest.json")
        with patch.object(vehicle_runtime, "prepare", return_value=prepared) as prepare:
            code, result = self.invoke(["profile", "prepare", str(path), "--jobs", "3", "--retry"])
        self.assertEqual(code, 0)
        self.assertEqual(result, prepared)
        prepare.assert_called_once_with(path, jobs=3, retry=True)

    def test_omitted_profile_is_x500_and_never_an_example(self):
        with patch.object(
            vehicle_runtime.subprocess, "run", side_effect=AssertionError("no process")
        ):
            code, result = self.invoke(["profile", "inspect"])
        self.assertEqual(code, 0)
        self.assertEqual(result["vehicle"], "x500")
        self.assertEqual(result["profile_id"], "x500_reference")
        self.assertEqual(result["configuration"], "reference")
        self.assertEqual(result["rotor_count"], 4)
        self.assertFalse(result["actual_flight_started"])

    def test_bad_yaml_is_a_structured_input_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text("schema: [")
            code, result = self.invoke(["profile", "validate", str(path)])
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Invalid vehicle profile", result["error"])

    def test_default_run_dispatch_and_explicit_x500_are_the_same_preset(self):
        for options in ([], ["--vehicle", "x500"], ["--vehicle", "x500_reference"]):
            output = io.StringIO()
            with (
                patch("sys.argv", ["capstone-sim", "run", *options]),
                patch.object(cli, "run_flight", return_value=dict(status="completed")) as run,
                redirect_stdout(output),
            ):
                self.assertEqual(cli.main(), 0)
            self.assertIsNone(run.call_args.kwargs["vehicle_profile"])
            self.assertIsNone(run.call_args.kwargs["configuration"])
            self.assertEqual(run.call_args.args[0], "hover_normal")

    def test_export_validates_corresponding_old_and_generic_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "observations").mkdir()
            metadata = run / "observations/metadata.json"
            for format_name, schema in (
                ("drone-raw-2", "drone.observation.metadata/2.0"),
                ("drone-raw-3", "drone.observation.metadata/3.0"),
            ):
                with self.subTest(format=format_name):
                    metadata.write_text(json.dumps(dict(schema=schema)))
                    with patch.object(
                        cli,
                        "verify",
                        return_value=dict(status="passed", C9="synthetic_verifier_fixture"),
                    ) as verify:
                        code, result = self.invoke(
                            ["export", "--run", str(run), "--format", format_name]
                        )
                    self.assertEqual(code, 0)
                    verify.assert_called_once_with(str(run))
                    self.assertEqual(result["path"], str((run / "observations").resolve()))
                    self.assertEqual(result["C9"], "synthetic_verifier_fixture")

    def test_export_rejects_wrong_or_unknown_contract_before_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "observations").mkdir()
            for format_name, schema in (
                ("drone-raw-2", "drone.observation.metadata/3.0"),
                ("drone-raw-3", "drone.observation.metadata/2.0"),
                ("drone-raw-3", "unknown"),
            ):
                with self.subTest(format=format_name, schema=schema):
                    (run / "observations/metadata.json").write_text(json.dumps(dict(schema=schema)))
                    with patch.object(
                        cli,
                        "verify",
                        side_effect=AssertionError("Mismatched format must not reach verifier"),
                    ) as verify:
                        code, result = self.invoke(
                            ["export", "--run", str(run), "--format", format_name]
                        )
                    self.assertEqual(code, 2)
                    self.assertEqual(result["status"], "failed")
                    self.assertIn("format must match", result["error"])
                    verify.assert_not_called()

    def test_export_propagates_verifier_failure_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "observations").mkdir()
            (run / "observations/metadata.json").write_text(
                json.dumps(dict(schema="drone.observation.metadata/3.0"))
            )
            with patch.object(
                cli, "verify", return_value=dict(status="failed", errors=["profile mismatch"])
            ):
                code, result = self.invoke(["export", "--run", str(run), "--format", "drone-raw-3"])
            self.assertEqual(code, 4)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["errors"], ["profile mismatch"])


if __name__ == "__main__":
    unittest.main()
