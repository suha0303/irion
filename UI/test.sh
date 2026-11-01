#!/usr/bin/env bash
set -euo pipefail
dashboard_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$dashboard_dir/backend"
../.venv/bin/python -m pytest -q
cd "$dashboard_dir/frontend"
npm test
npm run build
