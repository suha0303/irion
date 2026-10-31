#!/usr/bin/env bash
# Run on the SSH login host, using an already verified interpreter and shared paths.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: bash allocate_probe.sh /actual/python /actual/x500-gpu-smoke-v1 /new/output' >&2
  exit 2
fi
ai_python=$1
ai_package=$2
ai_output=$3
ai_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  echo 'Already inside an allocation. Do not submit another job.' >&2
  exit 2
fi
if [[ $ai_python != /* || $ai_package != /* || $ai_output != /* || ! -x $ai_python || ! -f $ai_package/MANIFEST.json ]]; then
  echo 'Provide verified absolute interpreter/package/output paths.' >&2
  exit 2
fi
command -v flock >/dev/null
command -v squeue >/dev/null
command -v srun >/dev/null
# All project submissions must use this same account-wide lock, across checkouts.
exec 9>"$HOME/.x500-ai-gpu.lock"
if ! flock -n 9; then
  echo 'Another project GPU launcher holds the lock; no job submitted.' >&2
  exit 3
fi
ai_uid=$(id -u)
ai_jobs=$(squeue -h -u "$ai_uid" -o '%A')
if [[ -n $ai_jobs ]]; then
  echo 'Own pending/running jobs exist; inspect them before submitting. No jobs changed.' >&2
  squeue -u "$ai_uid" -o '%.18i %.9P %.32j %.8T %.24R %.30b'
  exit 3
fi
mkdir -- "$ai_output"
printf '%s\n' 'Waiting only for p02 / DIS01 / nodes1 / tasks1 / GPU1; do not submit another job.'
set +e
srun --partition=p02 --nodelist=DIS01 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --job-name=x500-ai-probe --time=00:05:00 \
  "$ai_python" "$ai_script_dir/gpu_probe.py" --package "$ai_package" --out "$ai_output/probe" \
  --runtime-profile mamba25 \
  2>&1 | tee "$ai_output/srun.log"
ai_code=${PIPESTATUS[0]}
set -e
printf '%s\n' "$ai_code" > "$ai_output/srun-exit-code.txt"
if [[ ! -f $ai_output/probe/verified-job-id.txt ]]; then
  echo 'No validated job ID recorded. Allocation/release is unverified; inspect srun.log and your queue.' >&2
  exit 2
fi
ai_job_id=$(cat "$ai_output/probe/verified-job-id.txt")
if [[ ! $ai_job_id =~ ^[0-9]+$ ]]; then exit 2; fi
# Read only. Never cancel an inferred, existing, or another user's job.
for ai_attempt in {1..10}; do
  ai_jobs=$(squeue -h -u "$ai_uid" -o '%A')
  if ! printf '%s\n' "$ai_jobs" | grep -Fxq "$ai_job_id"; then
    printf 'Job %s absent from own active queue after srun exit; allocation released.\n' "$ai_job_id" | tee "$ai_output/release.txt"
    exit "$ai_code"
  fi
  sleep 1
done
printf 'Job %s still listed; release is not confirmed.\n' "$ai_job_id" | tee "$ai_output/release.txt"
exit 4
