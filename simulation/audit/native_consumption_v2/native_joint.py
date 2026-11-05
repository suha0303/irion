"""CPU audit v2: identical same-clock reconsumption is counted, never removed."""
import csv
import hashlib
import io
import json
import math
import struct
import subprocess
from pathlib import Path

REVISION = "joint-native-consumption-c2-v1"
ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "profiles/collection/joint-phase-policy.json"


def policy():
    value = json.loads(POLICY.read_text())
    if value["revision"] != REVISION:
        raise ValueError("Joint phase policy revision")
    return dict(value, policy_sha256=hashlib.sha256(POLICY.read_bytes()).hexdigest())




def reference_rows(spec):
    from capstone_sim.normal_variation import target
    from capstone_sim.random_flight import kinematics

    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]
    for index in range(4801):
        t = index / 10
        p = target(t, spec)
        velocity = acceleration = (0., 0., 0.)
        if spec["path"]["kind"] == "random":
            p, v, a = kinematics(t, spec["path"])
            velocity, acceleration = v[:3], a[:3]
        yaw = math.atan2(math.sin(p[3]), math.cos(p[3]))
        yield [index * 100_000, *map(f32, (*p[:3], *velocity, *acceleration, yaw))]






def receipt(run, kind):
    path = Path(run) / "evaluation" / f"joint-{kind}.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (value.get("schema") != "drone.joint-phase-native-receipt/1"
            or value.get("revision") != REVISION
            or value.get("policy_sha256") != policy()["policy_sha256"]
            or value.get("consumer") != "PositionControl::setInputSetpoint"
            or type(value.get("actual_time_us")) is not int):
        raise ValueError("Invalid actual native consumption receipt")
    return value


def verify(run):
    run = Path(run)
    start, end = receipt(run, "origin"), receipt(run, "landing")
    if start is None or end is None:
        raise ValueError("Missing actual joint phase boundaries")
    origin, finish = start["actual_start_us"], end["actual_time_us"]
    if not origin + 480_000_000 <= finish <= origin + 480_100_000:
        raise ValueError("Actual joint phase duration")
    with (run / "evaluation/joint-reference.csv").open() as stream:
        expected = list(csv.reader(stream))
    scenario = json.loads((run / "evaluation/scenario.json").read_text())
    canonical = io.StringIO()
    csv.writer(canonical, lineterminator="\n").writerows(reference_rows(scenario.get("operation_variation") or scenario["normal_variation"]))
    if (hashlib.sha256(canonical.getvalue().encode()).hexdigest()
            != scenario["joint_phase_policy"]["reference_sha256"]
            or hashlib.sha256((run / "evaluation/joint-reference.csv").read_bytes()).hexdigest()
            != scenario["joint_phase_policy"]["reference_sha256"]):
        raise ValueError("Preloaded reference differs from declared analytic recipe")
    seen = set()
    first = last = None
    maximum_gap = 0
    count = 0
    same_source_time_consumptions = 0
    previous_payload = None
    with (run / "evaluation/joint-consumption.csv").open() as stream:
        for row in csv.DictReader(stream):
            sequence, stamp, phase, index = (int(row[key]) for key in ("sequence", "source_us", "phase_us", "index"))
            if sequence != count + 1 or stamp - origin != phase or index != min(4800, phase // 100_000):
                raise ValueError("Native consumption sequence/phase/index")
            actual = [float(row[key]) for key in ("px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "az", "yaw")]
            if any(not math.isfinite(v) or abs(v-float(w)) > 1e-8 for v, w in zip(actual, expected[index][1:])):
                raise ValueError("Actual consumed reference differs from preloaded phase function")
            payload = {key: value for key, value in row.items() if key != "sequence"}
            if last is not None:
                if stamp < last:
                    raise ValueError("Consumption source time reversal")
                if stamp == last:
                    if payload != previous_payload:
                        raise ValueError("Conflicting consumption at identical source time")
                    same_source_time_consumptions += 1
                maximum_gap = max(maximum_gap, stamp-last)
            previous_payload = payload
            first = stamp if first is None else first
            last = stamp
            seen.add(index)
            count += 1
    if first != origin or last is None or last < origin + 480_000_000 or seen != set(range(4801)) or maximum_gap > 40_000:
        raise ValueError("Incomplete actual reference consumption coverage")
    return dict(status="passed", actual_start_us=origin, actual_landing_us=finish,
                actual_duration_s=(finish-origin)/1e6, terminal_hold_tail_us=finish-origin-480_000_000,
                consumed_samples=count, same_source_time_consumptions=same_source_time_consumptions,
                unique_source_time_samples=count-same_source_time_consumptions,
                audit_revision="native-consumption-identical-clock-v2",
                backward_source_time_count=0, conflicting_same_source_time_count=0,
                first_consumption_source_us=first, last_consumption_source_us=last,
                reference_rows_seen=len(seen), max_consumption_gap_us=maximum_gap,
                requested_and_actual_distinct=True)




def verify_pair(normal_run, anomaly_run, output, *, cpu_audit=None):
    """No interpolation: compare every applied physics tick and actual consumer receipts."""
    from capstone_sim.recording import sha
    from capstone_sim.runtime_manifest import verify_runtime

    normal, anomaly = Path(normal_run).resolve(), Path(anomaly_run).resolve()
    output = Path(output).absolute()
    if (output.exists() or output.is_symlink()
            or any(path.is_symlink() for path in output.parents)
            or output.resolve().is_relative_to(normal) or output.resolve().is_relative_to(anomaly)):
        raise ValueError("Pair output must be a new external evidence file")
    report: dict = dict(schema="drone.joint-phase-pair-check/1", status="failed", errors=[],
                       policy=dict(revision=REVISION, policy_sha256=policy()["policy_sha256"]),
                       normal_run=str(normal), anomaly_run=str(anomaly),
                       original_interval_us=[0, 480_000_000], interpolation_used=False)
    if cpu_audit is not None:
        report["cpu_audit"] = cpu_audit
    try:
        verify_runtime(ROOT)
        scenarios = [json.loads((run / "evaluation/scenario.json").read_text()) for run in (normal, anomaly)]
        left, right = scenarios
        if (normal == anomaly or left["seed"] != right["seed"]
                or left["operation_variation"] != right["operation_variation"]
                or left["wind_profile"] != right["wind_profile"]
                or left["joint_phase_policy"]["policy_sha256"] != right["joint_phase_policy"]["policy_sha256"]
                or left["paired_case_id"] != right["paired_case_id"]
                or left["producer_lock_sha256"] != right["producer_lock_sha256"]
                or left["producer_lock_sha256"] != sha(ROOT / "artifacts/runtime-manifest.json")
                or left["health_eligibility"] != "confirmed_healthy" or right["health_eligibility"] != "abnormal"):
            raise ValueError("Pair forcing/source/role/identity mismatch")
        report["source_sha256"] = sha(ROOT / "config/producer-sources.json")
        report["runtime_sha256"] = sha(ROOT / "artifacts/runtime-manifest.json")
        native = [verify(run) for run in (normal, anomaly)]
        report["native_consumption"] = dict(normal=native[0], anomaly=native[1])
        oracle = []
        for run, evidence, scenario in zip((normal, anomaly), native, scenarios):
            profile = scenario["wind_profile"]
            result = subprocess.run([str(ROOT / "build/current/joint_wind_verify"),
                str(run / "evaluation/truth.jsonl.joint-wind.csv"), str(evidence["actual_start_us"]),
                str(scenario["seed"] % 2**32), profile["environment"], str(profile["direction_deg"]), str(profile["variation"])],
                capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise ValueError("Native applied-wind function check: " + result.stderr.strip())
            oracle.append(json.loads(result.stdout))
        report["wind_function_checks"] = oracle
        maximum = squared = 0.
        count = 0

        def physical_samples(stream, origin):
            for row in csv.DictReader(stream):
                phase = int(row["source_us"]) - origin
                if phase < 0:
                    continue
                if phase > 480_000_000:
                    break
                yield phase, [float(row[key]) for key in ("east", "north", "up")]

        with (normal / "evaluation/truth.jsonl.joint-wind.csv").open() as a, (anomaly / "evaluation/truth.jsonl.joint-wind.csv").open() as b:
            first = physical_samples(a, native[0]["actual_start_us"])
            second = physical_samples(b, native[1]["actual_start_us"])
            for index in range(480001):
                p, q = next(first, None), next(second, None)
                if p is None or q is None or p[0] != index*1000 or q[0] != index*1000:
                    raise ValueError("Missing applied paired wind tick; no intersection/crop fallback")
                difference = math.dist(p[1], q[1])
                maximum = max(maximum, difference)
                squared += difference*difference
                count += 1
            if next(first, None) is not None or next(second, None) is not None:
                raise ValueError("Extra paired weather ticks")
        rms = math.sqrt(squared/count)
        report["wind"] = dict(samples=count, expected_samples=480001, missing_samples=0,
                              max_vector_difference_mps=maximum, rms_vector_difference_mps=rms)
        if maximum > 2e-9 or rms > 2e-9:
            raise ValueError("Paired actual wind differs")
        if (normal / "evaluation/joint-reference.csv").read_bytes() != (anomaly / "evaluation/joint-reference.csv").read_bytes():
            raise ValueError("Full preloaded reference functions differ")
        report["reference"] = dict(rows=4801, missing_rows=0, max_position_difference_m=0.,
            max_velocity_difference_mps=0., max_circular_yaw_difference_rad=0.,
            basis="Identical full native sampled function; every actual consumer reference matched the function at its recorded native phase. Controller query clocks/gaps reported separately; no resampling of responses.")
        report["origins"] = dict(normal=native[0]["actual_start_us"], anomaly=native[1]["actual_start_us"],
            absolute_start_difference_us=native[1]["actual_start_us"]-native[0]["actual_start_us"],
            comparison_clock="declared actual-consumption-relative phase; explicit new policy, not old absolute origin gate")
        report["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        report["errors"].append(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report
