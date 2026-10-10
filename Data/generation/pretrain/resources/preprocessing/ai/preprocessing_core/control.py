"""Fail-closed R1 admission. No remote calls or experiment execution on import."""

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

PLAN_SHA = "1e6c7fcbdd902c5325ce442db6a0ce0bbcac804e4108b33d0921318669ce6fce"
COUNTS = {
    "normal_train": 9,
    "normal_validation": 9,
    "normal_calibration": 9,
    "normal_dev_eval": 10,
    "anomaly_dev": 12,
}
FIT_ROLES = {"normal_train", "normal_validation", "normal_calibration"}
MODELS = ("persistence", "ridge_forecast", "lstm")
CONFIG = {
    "device": "cuda",
    "batch_size": 32,
    "epochs": 20,
    "hidden": 32,
    "learning_rate": 0.001,
    "seed": 17,
    "early_stopping_patience": 5,
}
REMOTE_LEDGER = Path("/abr/coss43/.x500-round001-r1")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def approval(plan, authorization):
    record = read(authorization)
    if (
        sha(plan) != PLAN_SHA
        or any(
            record.get(k) != v
            for k, v in {
                "round_id": "ROUND_001",
                "revision": "R1",
                "plan_sha256": PLAN_SHA,
            }.items()
        )
        or not record.get("exact_message")
    ):
        raise ValueError("round/revision/approval plan hash mismatch")
    return record


def verify_files(root, inventory):
    root = Path(root).resolve()
    if not inventory:
        raise ValueError("empty integrity inventory")
    for name, digest in inventory.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError("input/source hash mismatch: " + name)


def validate_config(config, models=MODELS):
    if tuple(models) != MODELS or config != CONFIG:
        raise ValueError("R1 fixed LSTM seed17/baseline/config allowlist mismatch")


def lineage(row):
    keys = {"run:" + row["run_id"]}
    keys.update("run:" + r for r in row.get("previous_attempts", []))
    for field in (
        "parent_attempt_id",
        "paired_case_id",
        "raw_sha256",
        "raw_flight_id",
        "continuation_id",
    ):
        if row.get(field):
            keys.add(("run:" if field == "parent_attempt_id" else field + ":") + row[field])
    return keys


def validate_roles(rows, cases, protected):
    """Only manifest identities are read; protected observations/truth are never opened."""
    if Counter(r["role"] for r in rows) != Counter(COUNTS):
        raise ValueError("R1 role counts mismatch; final/test roles forbidden")
    expected = {c["case_id"]: c for c in cases}
    if len(expected) != 49 or len({r["run_id"] for r in rows}) != 49:
        raise ValueError("duplicate or incomplete cases/runs")
    if len({r.get("raw_sha256") for r in rows}) != 49:
        raise ValueError("duplicate raw recordings")
    if {r["case_id"] for r in rows} != set(expected):
        raise ValueError("case membership mismatch")
    forbidden = set().union(*(lineage(r) for r in protected))
    seen = {}
    for row in rows:
        case = expected[row["case_id"]]
        for key in ("role", "seed", "scenario", "group_id"):
            if row.get(key) != case.get(key):
                raise ValueError("planned case mismatch: " + key)
        if row.get("status") != "completed" or row.get("origin") != "px4_sitl":
            raise ValueError("unfinished/non-SITL run")
        if row.get("continuity") != "independent_reset" or row.get("previous_attempts"):
            raise ValueError("continuation/retry forbidden in R1")
        if row["role"] != "anomaly_dev" and row.get("health_eligibility") != "confirmed_healthy":
            raise ValueError("normal role lacks health eligibility")
        if row["role"] == "anomaly_dev" and row.get("health_eligibility") != "abnormal":
            raise ValueError("anomaly role lacks scenario eligibility")
        if not row.get("raw_sha256") or not row.get("paired_case_id"):
            raise ValueError("missing raw/group identity")
        keys = lineage(row) | {"group:" + row["group_id"]}
        if keys & forbidden:
            raise ValueError("protected old/final lineage")
        # Normal controls and paired anomalies share only the development partition.
        partition = row["role"] if row["role"] in FIT_ROLES else "development"
        for key in keys:
            if key in seen and seen[key] != partition:
                raise ValueError("raw/paired/retry split leakage")
            seen[key] = partition


def claim(output, ledger, stage):
    """Durable attempt marker survives failure. No implicit retry or overwrite."""
    if stage not in {"gate", "main"}:
        raise ValueError("unknown execution stage")
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    if output.exists():
        raise ValueError("output already exists")
    ledger.mkdir(parents=True, exist_ok=True)
    marker = ledger / (stage + ".claimed")
    with marker.open("x") as stream:
        json.dump({"stage": stage, "output": str(output), "pid": os.getpid()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    output.mkdir(parents=True, exist_ok=False)


def check_claim(output, stage):
    marker = read(REMOTE_LEDGER / (stage + ".claimed"))
    if marker["stage"] != stage or marker["output"] != str(Path(output).resolve()):
        raise ValueError("stage output differs from durable account claim")


class Budget:
    def __init__(self, output, seconds, bytes_limit):
        if seconds <= 0 or bytes_limit <= 0:
            raise ValueError("positive finite budget required")
        self.output = Path(output)
        self.deadline = time.monotonic() + seconds
        self.bytes_limit = bytes_limit

    def check(self):
        if time.monotonic() >= self.deadline:
            raise RuntimeError("wall budget exhausted; STOPPED")
        size = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
        if size > self.bytes_limit:
            raise RuntimeError("storage budget exhausted; STOPPED")


def best_update(value, best):
    # Strict improvement ensures the earliest checkpoint survives equal validation loss.
    return value < best
