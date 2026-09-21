"""Fetch the pinned public comparison table; never overwrite a differing local file."""

import hashlib
import json
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[2]
url = "https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_static_kt0835.txt"
expected = "20519deb7e9678b0674005c38fdb8a23e916800bf5fc1b5714612f3f4c13d9a0"
path = root / "data/public/uiuc_apcsf_10x4.7_static.txt"
if not path.exists():
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("upstream reference changed; review required")
    path.write_bytes(data)
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise ValueError("local reference differs; preserved without overwrite")
(root / "data/public/uiuc-provenance.json").write_text(
    json.dumps(
        dict(
            source=url,
            sha256=actual,
            role="different-propeller contextual comparison only",
            license="copyright UIUC; public download; no explicit redistribution license identified",
        ),
        indent=2,
    )
    + "\n"
)
print(path)
