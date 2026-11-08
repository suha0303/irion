"""One-shot user-authorized shutdown after an already verified final download.

No GPU actions, LLM calls, forced shutdown, or automatic retries on failure.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ready(config, receipt):
    if receipt.get("status") != "LOCAL_FINAL_READY":
        return False
    dest = Path(config["destination"])
    if (
        receipt.get("destination") != str(dest)
        or not dest.is_dir()
        or dest.is_symlink()
    ):
        raise ValueError("Verified destination mismatch")
    manifest_path = dest / "MANIFEST.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != receipt.get("manifest_sha256"):
        raise ValueError("Missing or changed verified manifest receipt")
    manifest = read(manifest_path)
    expected = {
        f"{task}_{arch}_h{h}_l{layers}"
        for task in ("forecast", "run_health")
        for arch in ("lstm", "transformer")
        for h, layers in ((64, 2), (64, 4), (128, 2), (128, 4))
    }
    if (
        set(manifest.get("models", [])) != expected
        or receipt.get("models") != 16
        or receipt.get("files") != len(manifest.get("files", {}))
        or manifest.get("source") != config["remote_base"]
    ):
        raise ValueError("Final inventory mismatch")
    # The fetcher already verified every file hash before its atomic publish.
    for name in expected:
        if not (dest / "models" / name / "model.safetensors").is_file():
            raise ValueError("Missing published model")
    return True


def run(config_path, grace=120):
    config = read(config_path)
    work = Path(config["work"])
    status_path = work / "poweroff-state.json"

    def record(status, **details):
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(dict(status=status, at=time.time(), **details), indent=2)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(status_path)

    def cancelled():
        return (work / "CANCEL_POWEROFF").exists()

    deadline = time.monotonic() + 86400
    record(
        "WAITING_FOR_VERIFIED_DOWNLOAD",
        grace_seconds=grace,
        destination=config["destination"],
    )
    try:
        while time.monotonic() < deadline:
            if cancelled():
                record("CANCELLED")
                return
            receipt = read(work / "state.json")
            if receipt.get("status") in {
                "DELIVERY_FAILED_STOPPED",
                "UPSTREAM_STOPPED_NO_FINAL_COPY",
                "FETCH_STOPPED_BY_USER",
            }:
                record("BLOCKED_NO_SHUTDOWN", delivery_status=receipt["status"])
                return
            if ready(config, receipt):
                record(
                    "VERIFIED_DOWNLOAD_SHUTDOWN_GRACE",
                    shutdown_after=time.time() + grace,
                )
                end = time.monotonic() + grace
                while time.monotonic() < end:
                    if cancelled():
                        record("CANCELLED")
                        return
                    time.sleep(min(2, max(0, end - time.monotonic())))
                if not ready(config, read(work / "state.json")):
                    raise ValueError("Completion receipt changed during grace")
                capability = subprocess.run(
                    [
                        "busctl",
                        "call",
                        "org.freedesktop.login1",
                        "/org/freedesktop/login1",
                        "org.freedesktop.login1.Manager",
                        "CanPowerOff",
                    ],
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=15,
                )
                if capability.stdout.strip() != 's "yes"':
                    raise RuntimeError(
                        "Unattended normal poweroff is not authorized"
                    )
                record("REQUESTING_NORMAL_POWEROFF")
                result = subprocess.run(
                    [
                        "busctl",
                        "call",
                        "org.freedesktop.login1",
                        "/org/freedesktop/login1",
                        "org.freedesktop.login1.Manager",
                        "PowerOff",
                        "b",
                        "false",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                if result.returncode:
                    raise RuntimeError(result.stderr[-1500:])
                record("NORMAL_POWEROFF_REQUEST_ACCEPTED")
                return
            service = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "is-active",
                    "drone-ai-final-fetch.service",
                ],
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            if service.returncode:
                raise RuntimeError(
                    "Fetch service inactive before verified completion"
                )
            time.sleep(15)
        record("TIMEOUT_NO_SHUTDOWN")
    except Exception as exc:
        record("BLOCKED_NO_SHUTDOWN", reason=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--grace", type=int, default=120)
    args = parser.parse_args()
    if args.grace < 120:
        parser.error("At least120 seconds of cancellation grace required")
    run(args.config, args.grace)
