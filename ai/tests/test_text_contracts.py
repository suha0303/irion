"""JSON contracts remain UTF-8 when the process locale defaults to ASCII."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_json_metadata_is_independent_of_process_locale(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "profile.json"
    expected = {"name": "시험 기체", "notes": "배터리 전압"}
    source.write_text(
        json.dumps(expected, ensure_ascii=False), encoding="utf-8"
    )
    code = r"""
import json, locale, pathlib, sys
from console import control as console
from execution import common as execution
from learning.parallel19 import control as parallel
from pipeline import control as pipeline
from pretrain import common as data
from capstone_ai import source

assert locale.getpreferredencoding(False).lower() in {"ascii", "ansi_x3.4-1968"}
path = pathlib.Path(sys.argv[1])
expected = json.loads(sys.argv[2])
for reader in (console.read, execution.read, parallel.read, pipeline.read,
               data.read, source.read_json):
    assert reader(path) == expected, reader.__module__
for index, writer in enumerate((console.write, execution.write, parallel.write,
                                pipeline.write, data.write, source.dump)):
    output = path.with_name(f"result-{index}.json")
    writer(output, expected)
    assert source.read_json(output) == expected, writer.__module__
"""
    environment = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            str(root / part) for part in ("ai", "ai/src", "Data/generation")
        ),
    }
    subprocess.run(
        [sys.executable, "-B", "-c", code, str(source), json.dumps(expected)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
