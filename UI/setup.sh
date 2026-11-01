#!/usr/bin/env bash
set -euo pipefail
dashboard_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$dashboard_dir"
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.lock.txt
npm --prefix frontend ci
npm --prefix frontend run build
echo "설치 완료. ./start.sh 실행 후 http://127.0.0.1:8765 에 접속하세요."
