#!/usr/bin/env bash
set -euo pipefail
dashboard_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x "$dashboard_dir/.venv/bin/python" || ! -f "$dashboard_dir/frontend/dist/index.html" ]]; then
  echo "먼저 $dashboard_dir/setup.sh 를 실행하세요." >&2
  exit 1
fi
cd "$dashboard_dir/backend"
exec ../.venv/bin/python -m uvicorn arion.main:app --host 127.0.0.1 --port 8765
