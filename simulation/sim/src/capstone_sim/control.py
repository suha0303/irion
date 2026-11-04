import json
import math
import os
import re
import subprocess
from pathlib import Path

from . import collection_config
from .runner import run_flight


def control(
    run, operation, request_id=1, target="battery_r0", value=0.0, requested_us=0, scenario=None
):
    run = Path(run).resolve()
    state = json.loads((run / "run.json").read_text())
    if run.name != state["run_id"]:
        raise ValueError("run identity mismatch")
    if operation in ["reset", "set_scenario"]:
        if state["status"] not in ["completed", "failed"]:
            raise ValueError(
                "finalize the current run before reset; never mix clocks in one stream"
            )
        instance = state.get("instance")
        limit = 15 if state.get("configuration") else 9
        if type(instance) is not int or not 0 <= instance <= limit:
            raise ValueError(
                "Original PX4 instance is missing or invalid; reset cannot assume instance 0. "
                "Keep this evidence and start a separate run with an explicit --instance."
            )
        old = json.loads((run / "evaluation/scenario.json").read_text())
        selected_scenario = scenario or old["name"]
        same_scenario = selected_scenario == old["name"]
        options = dict(real_time_factor=old.get("requested_real_time_factor", 2))
        if same_scenario:
            if old.get("fault_after_mission_s") is not None:
                options["fault_after_mission_s"] = old["fault_after_mission_s"]
            else:
                options["fault_at_s"] = old.get("parameters", {}).get("fault_at")
        if state.get("configuration", "").startswith("vehicle_profile:"):
            from .recording import verify_generic_metadata
            from .vehicle_runtime import read_prepared

            metadata = json.loads((run / "observations/metadata.json").read_text())
            verify_generic_metadata(run, metadata)
            original = json.loads((run / "reference/vehicle-profile.json").read_text())
            if state["configuration"] != "vehicle_profile:" + original["profile_id"] or (
                metadata["run_id"] != state["run_id"]
            ):
                raise ValueError("Original custom run/profile identity mismatch")
            _, current = read_prepared(original["source_profile"])
            if current["recipe_sha256"] != original["recipe_sha256"]:
                raise ValueError("Original custom profile changed; reset cannot switch vehicles")
            return run_flight(
                selected_scenario,
                old["seed"],
                old["requested_duration_s"],
                instance,
                state["run_id"],
                vehicle_profile=original["source_profile"],
                **options,
            )
        if state.get("configuration") in (collection_config.NAME, collection_config.LEGACY_NAME):
            if not same_scenario and selected_scenario.endswith(("step", "ramp")):
                options["fault_at_s"] = 180
            return run_flight(
                selected_scenario,
                old["seed"],
                old["requested_duration_s"],
                instance,
                state["run_id"],
                configuration="collection",
                normal_variation=old.get("normal_variation") if same_scenario else None,
                operation_variation=old.get("operation_variation") if same_scenario else None,
                **options,
            )
        if state.get("configuration"):
            raise ValueError("Original simulation configuration is not supported for reset")
        return run_flight(
            selected_scenario,
            old["seed"],
            old["requested_duration_s"],
            instance,
            state["run_id"],
            **options,
        )
    if operation == "status" and state["status"] in ["completed", "failed"]:
        return state
    env = os.environ.copy()
    if state.get("configuration"):
        env = collection_config.transport_environment(env)
    env.update(GZ_PARTITION="x500_" + run.name, GZ_IP="127.0.0.1", GZ_HOMEDIR=str(run / "gz"))
    if operation == "apply_parameter_change":
        fields = {"battery_r0": 0, "motor_friction": 1, "battery_r1": 2}
        if target not in fields:
            raise ValueError("unsupported physical parameter")
        if not math.isfinite(value) or value < 0 or (target == "battery_r1" and value == 0):
            raise ValueError("invalid parameter; dynamic R1=0 is unsupported")
        values = [1, request_id, fields[target], value, requested_us]
        service = "apply_parameter_change"
    else:
        values = [request_id]
        service = "change_status"
    request = " ".join(f"data: {v}" for v in values)
    p = subprocess.run(
        [
            "gz",
            "service",
            "-s",
            f"/x500_0/{service}",
            "--reqtype",
            "gz.msgs.Double_V",
            "--reptype",
            "gz.msgs.Double_V",
            "--timeout",
            "3000",
            "--req",
            request,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    response = parse_response(operation, p.returncode, p.stdout)
    return dict(
        **response,
        request_id=request_id,
        operation=operation,
        target=target,
        requested_time_us=requested_us,
        requested_value=value,
        stdout=p.stdout,
        stderr=p.stderr,
    )


def parse_response(operation, transport_exit, text):
    values = [float(v) for v in re.findall(r"^data: ([-+0-9.eE]+)$", text, re.MULTILINE)]
    result = dict(
        exit_code=4, transport_exit_code=transport_exit, status="transport_or_protocol_error"
    )
    if transport_exit or not all(math.isfinite(v) for v in values):
        return result
    if values == [-1]:
        result["status"] = "rejected"
    elif operation == "apply_parameter_change" and values in ([0], [1]):
        result.update(exit_code=0, status="queued" if values == [0] else "duplicate")
    elif operation == "status" and len(values) == 2 and values[1] >= 0:
        result.update(
            exit_code=0,
            status="applied" if values[0] >= 0 else "queued",
            applied_time_us=int(values[0]) if values[0] >= 0 else None,
            revision=int(values[1]),
        )
    return result
