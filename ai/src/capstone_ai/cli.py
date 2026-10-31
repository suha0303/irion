import argparse
import json
import platform
import sys
import time
from collections import Counter
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from .mapping import CONTEXT
from .preprocessing import Preprocessor
from .source import DirectoryTail, dump, read_json, resolve_profile, sha

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "ai/configs/preprocessing.json"


def output(engine, tail, out, mode, evidence):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("frames", engine.frames),
        ("windows", engine.windows),
        ("exclusions", engine.exclusions),
    ]:
        p = out / (name + ".jsonl")
        tmp = p.with_suffix(".tmp")
        with tmp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        tmp.replace(p)
    files = [
        "run.json",
        "observations/metadata.json",
        "observations/streams.jsonl",
        "observations/watermarks.jsonl",
        "observations/diagnostics.jsonl",
        "reference/source_lock.json",
    ]
    report = {
        "run": str(tail.run),
        "run_id": engine.metadata["run_id"],
        "origin": engine.metadata["origin"],
        "transport": mode,
        "evidence": evidence,
        "closed": tail.closed,
        "frames": len(engine.frames),
        "valid_frames": sum(
            all(q == "valid" for q in f["quality"].values()) for f in engine.frames
        ),
        "windows": len(engine.windows),
        "feature_order": engine.names,
        "context_order": CONTEXT,
        "exclusions": dict(Counter(r for x in engine.exclusions for r in x["reasons"])),
        "diagnostics": dict(engine.diagnostics),
        "config": engine.config,
        "final_watermark": tail.last_mark,
        "input_hashes": {n: sha(tail.run / n) for n in files if (tail.run / n).exists()},
        "output_hashes": {
            n: sha(out / (n + ".jsonl")) for n in ["frames", "windows", "exclusions"]
        },
        "consumer_revision": {
            str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "ai/src/capstone_ai").glob("*.py")
        },
        "contract_sha256": sha(ROOT / "docs/specifications/CODEX_REBUILD_PACKAGE_V4_2/00_SHARED_DATA_CONTRACT.md"),
        "python": sys.version,
        "partial_stream_bytes": len(tail.stream.buffer),
    }
    dump(out / "summary.json", report)
    dump(out / "checkpoint.json", tail.checkpoint())
    return report


def live_evidence_verified(mode, initial_status, grew, closed):
    # The producer enters mission after confirming airborne/OFFBOARD and command ACKs.
    active = {"airborne", "mission", "taking_off", "landing", "armed", "running"}
    return mode == "live" and initial_status in active and grew and closed


def process(run, out, config, mode="replay", chunk_bytes=1048576, timeout=300, resume=None):
    run = Path(run).resolve()
    engine = Preprocessor(read_json(run / "observations/metadata.json"), config)
    tail = DirectoryTail(run, engine, chunk_bytes)
    if resume:
        tail.verify_checkpoint(read_json(resume))
    initial = read_json(run / "run.json")
    evidence = {
        "initial_status": initial["status"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_grew_while_attached": False,
        "live_verified": False,
    }
    start_size = (run / "observations/streams.jsonl").stat().st_size
    start = time.monotonic()
    while not tail.closed:
        progress = tail.poll()
        evidence["source_grew_while_attached"] |= (
            run / "observations/streams.jsonl"
        ).stat().st_size > start_size
        if not progress:
            if mode == "replay":
                break
            if time.monotonic() - start > timeout:
                break
            time.sleep(0.02)
    evidence["live_verified"] = live_evidence_verified(
        mode, initial["status"], evidence["source_grew_while_attached"], tail.closed
    )
    report = output(engine, tail, out, mode, evidence)
    return report


def compare(left, right, config):
    a, b = read_json(Path(left) / "summary.json"), read_json(Path(right) / "summary.json")
    errors, max_error = [], 0.0

    def equal(x, y, path):
        nonlocal max_error
        if isinstance(x, bool) or x is None or isinstance(x, str):
            if x != y:
                errors.append(path)
        elif isinstance(x, (float, int)) and isinstance(y, (float, int)):
            error = abs(x - y)
            max_error = max(max_error, error)
            if error > config["comparison_atol"] + config["comparison_rtol"] * abs(x):
                errors.append(path)
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                errors.append(path + ":length")
            else:
                for i, (u, v) in enumerate(zip(x, y)):
                    equal(u, v, path + f"/{i}")
        elif isinstance(x, dict) and isinstance(y, dict) and x.keys() == y.keys():
            for k in x:
                equal(x[k], y[k], path + "/" + k)
        else:
            errors.append(path + ":type_or_keys")

    for key in ["run_id", "origin", "config", "final_watermark", "input_hashes"]:
        equal(a[key], b[key], key)
    for name in ["frames", "windows", "exclusions"]:
        x = [json.loads(s) for s in (Path(left) / (name + ".jsonl")).read_text().splitlines()]
        y = [json.loads(s) for s in (Path(right) / (name + ".jsonl")).read_text().splitlines()]
        equal(x, y, name)
    if min(a["valid_frames"], b["valid_frames"], a["windows"], b["windows"]) == 0:
        errors.append("empty_valid_frames_or_post_warmup_windows")
    if not a["closed"] or not b["closed"]:
        errors.append("not_final_watermark")
    return {
        "passed": not errors,
        "max_absolute_error": max_error,
        "atol": config["comparison_atol"],
        "rtol": config["comparison_rtol"],
        "errors": errors[:30],
        "frames": a["frames"],
        "windows": a["windows"],
        "C9_live": "verified"
        if not errors and (a["evidence"]["live_verified"] or b["evidence"]["live_verified"])
        else "not_executed",
    }


def main():
    parser = argparse.ArgumentParser(description="X500 AI: model-free DRONE-RAW preprocessing")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ["preprocess", "replay", "live"]:
        p = sub.add_parser(name)
        inputs = p.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--input")
        inputs.add_argument("--source")
        p.add_argument("--out", required=True)
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        p.add_argument("--chunk-bytes", type=int, default=1048576)
        p.add_argument("--timeout", type=float, default=300)
        p.add_argument("--resume")
    p = sub.add_parser("compare")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p = sub.add_parser("audit")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--out", required=True)
    p = sub.add_parser("dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    for name in ["calibrate", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--bundle", required=True)
        p.add_argument("--dataset", required=True)
        if name == "evaluate":
            p.add_argument("--out", required=True)
    p = sub.add_parser("forecast")
    p.add_argument("--canonical", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("ulog-parity")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    result: dict[str, Any]
    try:
        if args.action == "dataset":
            from .dataset import build

            build(args.config, args.out)
            result = {"status": "built", "manifest": str(Path(args.out) / "manifest.json")}
        elif args.action == "train":
            from .training import train_models

            result = train_models(args.config)
        elif args.action in {"calibrate", "evaluate"}:
            import torch

            from .evaluation import calibrate, evaluate

            torch.set_num_threads(2)
            result = (
                calibrate(args.bundle, args.dataset)
                if args.action == "calibrate"
                else evaluate(args.bundle, args.dataset, args.out)
            )
        elif args.action == "forecast":
            from .runtime import replay_forecasts

            result = replay_forecasts(args.canonical, args.bundle, args.out)
        elif args.action == "ulog-parity":
            from .ulog import parity

            result = parity(args.input, args.out)
        elif args.action == "doctor":
            result = {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {d.metadata["Name"]: d.version for d in distributions()},
            }
            dump(args.out, result)
        elif args.action == "audit":
            from .audit import audit

            result = audit(args.input, args.out)
        elif args.action == "compare":
            result = compare(args.left, args.right, read_json(args.config))
            dump(args.out, result)
        else:
            result = process(
                resolve_profile(args.source) if args.source else args.input,
                args.out,
                read_json(args.config),
                "live" if args.action == "live" else "replay",
                args.chunk_bytes,
                args.timeout,
                args.resume,
            )
        print(json.dumps(result, allow_nan=False))
        if result.get("passed") is False or result.get("closed") is False:
            return 4
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
