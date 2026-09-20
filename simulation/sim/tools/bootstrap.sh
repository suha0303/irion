#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if ! command -v python3.12 >/dev/null; then
  printf '%s\n' 'Python 3.12 is required. On Ubuntu 24.04 install python3.12-venv and python3.12-dev.' >&2
  exit 3
fi
exec python3.12 -B "$ROOT/sim/tools/install.py" "$@"
