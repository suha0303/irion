#!/usr/bin/env bash
set -euo pipefail
dashboard_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x "$dashboard_dir/.venv/bin/python" ]]; then
  echo "먼저 $dashboard_dir/setup.sh 를 실행하세요." >&2
  exit 1
fi
cd "$dashboard_dir/backend"
exec ../.venv/bin/python -m arion.service "${1:-start}"
