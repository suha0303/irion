"""Isolate Windows loopback connectivity before assembling the installer."""

import json
import os
import queue
import socket
import subprocess
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import ProxyHandler, build_opener

from localflight.desktop_native import (
    NATIVE_INPUTS,
    RUNTIME_RELEASE,
    RUNTIME_VERSION,
    download,
)

BASIC_SERVER = """
import json, socket, sys
server = socket.create_server(('127.0.0.1', 0))
print(json.dumps({'port': server.getsockname()[1]}), flush=True)
client, address = server.accept()
with client:
    client.sendall(b'ok')
"""


def probe(label, python, command, env, http):
    print(f"Checking {label}", flush=True)
    process = subprocess.Popen(
        [python, "-s", "-B", *command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )
    first = queue.Queue()
    threading.Thread(
        target=lambda: first.put(process.stdout.readline()), daemon=True
    ).start()
    try:
        ready = first.get(timeout=20)
        port = json.loads(ready)["port"]
        print(f"{label}: announced localhost port {port}", flush=True)
        if http:
            with build_opener(ProxyHandler({})).open(
                f"http://127.0.0.1:{port}/api/health", timeout=5
            ) as response:
                assert json.load(response)["status"] == "ok"
        else:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                assert client.recv(2) == b"ok"
        print(f"{label}: OK", flush=True)
    except Exception as error:
        print(f"{label}: {type(error).__name__} {error}", flush=True)
        if process.poll() is None:
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Get-NetTCPConnection -OwningProcess {process.pid} -State Listen -ErrorAction SilentlyContinue | Select-Object LocalAddress,LocalPort,State",
                ],
                check=False,
                timeout=15,
            )
        return False
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return True


def main():
    with TemporaryDirectory(prefix="arion-probe-") as temporary:
        root = Path(temporary)
        triple, sha, _ = NATIVE_INPUTS["win32-x64"]
        archive = download(
            root,
            f"https://github.com/astral-sh/python-build-standalone/releases/download/{RUNTIME_RELEASE}",
            f"cpython-{RUNTIME_VERSION}+{RUNTIME_RELEASE}-{triple}-install_only.tar.gz",
            sha,
        )
        subprocess.run(["tar", "-xzf", archive, "-C", root], check=True)
        runtime = root / "python"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--only-binary=:all:",
                "-r",
                "ai/localflight/requirements-desktop-ubuntu.lock",
                "numpy==1.26.4",
                "pymavlink==2.4.50",
            ],
            check=True,
        )
        runtime_path = subprocess.check_output(
            [
                sys.executable,
                "-c",
                'import sysconfig;print(sysconfig.get_paths()["purelib"])',
            ],
            text=True,
        ).strip()
        env = dict(os.environ, PYTHONUTF8="1", ARION_DATA_DIR=str(root / "stock-data"))
        bundled = dict(
            env, PYTHONHOME=str(runtime), ARION_DATA_DIR=str(root / "bundled-data")
        )
        bundled["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], runtime_path))
        variants = (
            ("stock socket", sys.executable, ["-c", BASIC_SERVER], env, False),
            (
                "bundled socket",
                runtime / "python.exe",
                ["-c", BASIC_SERVER],
                bundled,
                False,
            ),
            ("stock ARION", sys.executable, ["-m", "arion.desktop"], env, True),
            (
                "bundled ARION",
                runtime / "python.exe",
                ["-m", "arion.desktop"],
                bundled,
                True,
            ),
        )
        return 0 if all([probe(*variant) for variant in variants]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
