"""Persistent local artifact fetch; no LLM, GPU commands, or experiment retries."""

import argparse
import concurrent.futures
import fcntl
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from .export_final import read, safe, sha, write

PROBE = """import json,pathlib
b=pathlib.Path(BASE);p=b/'output/state.json'
s=json.loads(p.read_text()) if p.exists() else {}
a=b/'allocation-control.json';a=json.loads(a.read_text()) if a.exists() else {}
print(json.dumps({'status':s.get('status','NOT_STARTED'),'reason':s.get('reason'),'allocation_control':a.get('status')}))
"""


def verify(folder, manifest):
    expected = set(manifest["files"])
    actual = {
        str(p.relative_to(folder))
        for p in Path(folder).rglob("*")
        if p.is_file()
    } - {"MANIFEST.json"}
    if actual != expected:
        raise ValueError("Unexpected/missing delivery files")
    for name, digest in manifest["files"].items():
        if sha(safe(folder, name)) != digest:
            raise ValueError("Local artifact checksum mismatch: " + name)


def run(config):
    c = read(config)
    work = Path(c["work"])
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / "fetch.lock").open("a", encoding="utf-8")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    final = Path(c["destination"])
    stage = final.with_name("." + final.name + ".incoming")
    ssh = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        c["host"],
    ]
    status = work / "state.json"

    def record(state, **kw):
        write(
            status,
            {
                "status": state,
                "at": time.time(),
                "destination": str(final),
                **kw,
            },
        )

    def remote(argv, code=None):
        return subprocess.run(
            ssh + [shlex.join(argv)],
            input=code,
            text=True,
            capture_output=True,
            timeout=600,
            check=False,
        )

    record("WAITING_FOR_FINAL_COMPLETION")
    while True:
        if (work / "STOP").exists():
            record("FETCH_STOPPED_BY_USER")
            return
        try:
            result = remote(
                [c["python"], "-B", "-"],
                PROBE.replace("BASE", repr(c["remote_base"])),
            )
            if result.returncode:
                record("WAITING_FOR_SSH", reason=result.stderr[-1500:])
                time.sleep(60)
                continue
            state = json.loads(result.stdout)
            if state["status"] in {"FAILED_STOPPED", "USER_STOPPED"} or state[
                "allocation_control"
            ] in {
                "ALLOCATION_FAILED_NO_RETRY",
                "BLOCKED_OTHER_OWNER_JOBS",
                "STOPPED_NO_RETRY",
            }:
                record("UPSTREAM_STOPPED_NO_FINAL_COPY", upstream=state)
                return
            if state["status"] != "REPORT_READY_STOPPED":
                record("WAITING_FOR_FINAL_COMPLETION", upstream=state)
                time.sleep(60)
                continue
            record("PREPARING_FINAL_EXPORT")
            result = remote(
                [c["python"], "-B", c["export_script"], c["remote_base"]]
            )
            if result.returncode:
                if result.returncode == 255:
                    record("WAITING_FOR_SSH", reason=result.stderr[-1500:])
                    time.sleep(60)
                    continue
                raise ValueError(
                    "Remote export failed: " + result.stderr[-2000:]
                )
            export = json.loads(result.stdout)
            if export["status"] == "EXPORT_IN_PROGRESS":
                time.sleep(60)
                continue
            if (
                export["status"] != "FINAL_EXPORT_READY"
                or export["path"] != c["remote_base"] + "/delivery/final"
            ):
                raise ValueError("Export not ready/path mismatch")
            result = remote(["cat", export["path"] + "/MANIFEST.json"])
            if result.returncode:
                record("WAITING_FOR_SSH", reason=result.stderr[-1500:])
                time.sleep(60)
                continue
            manifest = json.loads(result.stdout)
            if len(manifest["models"]) != 16 or any(
                not re.fullmatch("[a-zA-Z0-9_-]+", n)
                for n in manifest["models"]
            ):
                raise ValueError("Invalid model inventory")
            for name in manifest["files"]:
                safe(stage, name)
            if final.exists():
                if sha(final / "MANIFEST.json") != export["manifest_sha256"]:
                    raise ValueError(
                        "Destination already contains a different final bundle"
                    )
                verify(final, manifest)
                record(
                    "LOCAL_FINAL_READY",
                    files=len(manifest["files"]),
                    models=len(manifest["models"]),
                )
                return
            stage.mkdir(parents=True, exist_ok=True)
            # Compression and four independent model streams reduce transfer wall time.
            rsync = [
                "rsync",
                "-rltz",
                "--partial",
                "--timeout=120",
                "-e",
                "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3",
            ]

            def copy(relative):
                target = stage / relative
                target.mkdir(parents=True, exist_ok=True)
                cmd = (
                    rsync
                    + (["--exclude=models/"] if not relative else [])
                    + [
                        c["host"] + ":" + export["path"] + "/" + relative + "/",
                        str(target) + "/",
                    ]
                )
                return subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )

            record("DOWNLOADING", models=len(manifest["models"]))
            result = copy("")
            if result.returncode:
                record("TRANSFER_INTERRUPTED", reason=result.stderr[-1500:])
                time.sleep(60)
                continue
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(copy, ["models/" + n for n in manifest["models"]])
                )
            failures = [r.stderr[-1000:] for r in results if r.returncode]
            if failures:
                record("TRANSFER_INTERRUPTED", reason=failures)
                time.sleep(60)
                continue
            record("VERIFYING_LOCAL_FILES")
            if sha(stage / "MANIFEST.json") != export["manifest_sha256"]:
                raise ValueError("Manifest transfer changed")
            verify(stage, manifest)
            stage.rename(final)
            record(
                "LOCAL_FINAL_READY",
                files=len(manifest["files"]),
                models=len(manifest["models"]),
                manifest_sha256=export["manifest_sha256"],
            )
            return
        except (subprocess.TimeoutExpired, ConnectionError) as e:
            record("WAITING_FOR_SSH", reason=str(e))
            time.sleep(60)
        except Exception as e:
            record("DELIVERY_FAILED_STOPPED", reason=f"{type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    a = p.parse_args()
    run(a.config)
