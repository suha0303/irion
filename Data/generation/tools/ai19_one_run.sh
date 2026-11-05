#!/usr/bin/env bash
# Coordinator starts exactly one standalone480s integration; no fit, retry or GPU.
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$TASK_ROOT"
if [[ $# -ne 1 || ! $1 =~ ^[A-Za-z0-9_-]{1,64}$ ]]; then
  printf '%s\n' 'Usage: bash Data/generation/tools/ai19_one_run.sh NEW_TASK_NAME' >&2
  exit 2
fi
./Data/generation/drone-pretrain integration19 init --name "$1"
exec ./Data/generation/drone-pretrain integration19 start "$TASK_ROOT/Data/operations/pretrain/tasks/$1"
