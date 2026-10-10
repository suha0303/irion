# R3_LONG15 AI execution connection

New revision only; old R2 package/failures/models remain immutable. No real collection,
SSH, allocation, preparation, training or calibration is performed by implementation tests.
Code reuses R1 fit/backend, model, calibration and inference formulas. Six common source
files add an explicit battery-nameplate parameter: mapping.py, preprocessing.py, dataset.py,
detection_replay.py, live_inference.py and preprocessing_core/pipeline.py. Legacy/default remains6Ah;
R3 explicitly requires `battery_profile_id=vtc6_4s4p_r3_long15`,4S4P,12Ah. Current features,
C-rate context and charge/energy fractions use12Ah consistently in training and inference.
No SOC/health truth is a model input. Cell voltage normalization stays4S; nominal3.7V remains
the original declared engineering reference. Old C9 configs and archived packages are unchanged.
`derived.py` independently divides R3 current by12 with unchanged other numeric audit formulas.
Historical GPU compatibility applies to unchanged model/optimizer/backend; the new nameplate
path has new CPU fixture evidence, not a historical full-source execution claim.
The memory estimate is admission planning, not measured RSS.

## Fixed scope

The coordinator owns `../round-plan.json` and `../case_manifest.json`. They are hash-bound
at packaging. Exactly50 cases: normal_train9, validation9, calibration9, normal_dev_eval11,
anomaly_dev12. The same9 normal scenarios occur independently in each split; low_soc_normal
180s, others900s. Two paired hover controls480s and six original anomaly families ×two
distinct seeds480s are development only. All rows use a new VTC6_4S4P producer revision.
Old lineage (including R2) is forbidden. No test/final/truth fitting access is allowed.

10Hz/history100/context60s/future10 at0.1s, window stride **1 second**. LSTM hidden32,
seed17, batch32, FP32 ATen CUDA,20epochs/patience5; unchanged persistence and dual ridge.
No baseline omission, thinning, solver change or automatic retraining on poor results.
GPU main: p02/DIS01/nodes1/tasks1/GPU1, own queue empty, shared project GPU lock,3hours,
queue15min, retries0. Prepare CPU2hours. Calibration plus coordinator evaluation share7200s.
External cumulative working budget80GB, minimum free50GB; no fallback onto system disk.

## Manifest and source requirements

`../required-revision.json` must contain the original five producer/firmware/physics/
profile/preprocessing lock fields plus `battery_profile: "VTC6_4S4P"`, verified by the
coordinator against the new producer. This file is **not inferred from old R2**.
Its preprocessing_sha256 must be the hash of this directory's `preprocessing.json`,
including the two explicit12Ah profile fields; do not reuse the R2 preprocessing hash.
`../protected-inventory.json` must be complete with runs, deny_paths and includes_revisions
containing `R2_1GB`, including actual R2 failed/completed run identities as well as older ones.

`../frozen-manifest.json` has status DATA_FROZEN, revision R3_LONG15, plan_sha256 matching
the exact packaged round-plan bytes, revision_lock equal to the packaged required revision,
and50runs. Each row includes the case's case_id/role/seed/scenario/group_id/mission_seconds,
run_id, raw_flight_id, raw_sha256, paired_case_id, completed px4_sitl, independent_reset,
previous_attempts=[], null parent_attempt_id/continuation_id, health_eligibility
confirmed_healthy (normal) or abnormal, and that same revision_lock.

`path` is the **future extraction destination**:
`/media/sa/X31/Data/X500/ROUND_001/R3_LONG15/consumer/<run_id>`.
`files` maps exactly these6 relative names to hashes:
run.json, observations/metadata.json, observations/streams.jsonl,
observations/watermarks.jsonl, observations/diagnostics.jsonl, reference/source_lock.json.
raw_sha256 equals streams hash. `archive` is the collector's record with path,sha256,bytes,
members{name:{kind,bytes,sha256}}; archive path must be under external R3 `archives/`.
Each tar member is relative to run root (no enclosing run-ID directory).

Archive hash/size is checked before selective materialization. Only the6 consumer files
are written, byte-limited and individually hashed. No extractall, path traversal, duplicate
members, selected links/special/sparse files or overwrite. Skipped members are never
materialized or parsed as observations; compressed traversal necessarily reads archive bytes.
`events.jsonl` is **not extracted by AI**. The coordinator extracts it into a separate
evaluation-only tree and preserves archive/member hashes. Never point consumer paths
at that truth tree. Failed partial extraction is retained; claims prohibit retries.

## Build and explicit CLI

After collection reaches COLLECTED_PENDING_QUALITY_STOPPED, the freeze CLI accepts the
coordinator's exact plan/cases/revision/protected inventory and DATA's final ledger. It
verifies every archive and bounded run/source-lock/scenario identity metadata, never
events. All50 passed producer assessments are required. No unfinished/failed case is dropped.
Source-lock digest and pinned PX4/plugin/profile hashes must match the new revision.
`raw_flight_id` uses the explicit producer field, or the unique independently reset run ID
when that field is absent; paired_case_id preserves producer scenario identity when present,
otherwise uses the pre-window planned group_id. Both identities remain in metadata; neither
producer retries nor continuations are silently relabeled as independent resets.

```bash
PYTHONPATH=ai/experiments/ROUND_001/long-runs-02/ai:ai:ai/src ai/.venv/bin/python -B -m preprocessing_tools.freeze --plan ai/experiments/ROUND_001/long-runs-02/round-plan.json --cases ai/experiments/ROUND_001/long-runs-02/case_manifest.json --ledger ai/experiments/ROUND_001/long-runs-02/collection/execution/ledger.json --revision ai/experiments/ROUND_001/long-runs-02/required-revision.json --protected ai/experiments/ROUND_001/long-runs-02/protected-inventory.json --out ai/experiments/ROUND_001/long-runs-02/frozen-manifest.json
```

After the actual new revision is verified (placeholders are not usable authorization):

```bash
PYTHONPATH=ai/experiments/ROUND_001/long-runs-02/ai:ai:ai/src ai/.venv/bin/python -B -m preprocessing_tools.build --project /home/sa/CapstoneV3 --source /home/sa/CapstoneV3/ai/experiments/ROUND_001/long-runs-02/ai --plan /home/sa/CapstoneV3/ai/experiments/ROUND_001/long-runs-02/round-plan.json --cases /home/sa/CapstoneV3/ai/experiments/ROUND_001/long-runs-02/case_manifest.json --revision /home/sa/CapstoneV3/ai/experiments/ROUND_001/long-runs-02/required-revision.json --out /home/sa/CapstoneV3/ai/experiments/ROUND_001/long-runs-02/ai/release
```

The package has no executable approval. Coordinator verifies its SHA and transfers it
once to an explicit fresh `/abr/coss43/...` root. This task does not guess that remote path.
For each actual stage, the coordinator supplies a different request:

```json
{
  "action":"START", "authorized":true, "revision":"R3_LONG15", "stage":"prepare",
  "exact_message":"User-approved R3 one-round explicit stage invocation",
  "package":"/absolute/extracted/x500-round001-r3-long15",
  "package_sha256":"ACTUAL_PACKAGE_JSON_SHA256",
  "inputs_sha256":{"frozen-manifest.json":"ACTUAL_SHA256","protected-inventory.json":"ACTUAL_SHA256"},
  "remote_package":"/abr/coss43/ACTUAL_VERIFIED_PACKAGE",
  "remote_stage":"/abr/coss43/ACTUAL_NEW_MUTABLE_STAGE"
}
```

Use the package paths as PYTHONPATH; local Python stays ai/.venv. Substitute stage and
its exact request path; run one command per explicit coordinator authorization:

```bash
PYTHONPATH="$PACKAGE/wrapper:$PACKAGE/ai:$PACKAGE/ai/src" /home/sa/CapstoneV3/ai/.venv/bin/python -B -m preprocessing_tools.operator prepare --request /absolute/prepare-request.json
PYTHONPATH="$PACKAGE/wrapper:$PACKAGE/ai:$PACKAGE/ai/src" /home/sa/CapstoneV3/ai/.venv/bin/python -B -m preprocessing_tools.operator train --request /absolute/train-request.json
PYTHONPATH="$PACKAGE/wrapper:$PACKAGE/ai:$PACKAGE/ai/src" /home/sa/CapstoneV3/ai/.venv/bin/python -B -m preprocessing_tools.operator calibrate --request /absolute/calibrate-request.json
```

Create the external `consumer`, `processed`, `models`, `calibration`, `metadata` categories
before execution; existing archives belong to DATA. New stage output is category/attempt-01.
Operator request/profile/package must match; approvals bind all hashes/paths and are outside
the immutable package. Exclusive durable operator and launcher claims survive failures.
No automatic next stage/resume/retry. The total coordinator program may invoke stages under
its separate user authorization, but must stop on any nonzero exit.

## Train gate and remote behavior

Train first verifies prepare artifact/exit/release. It obtains a fresh DIS01 CPU RAM
snapshot through the existing `data-1gb-01/ssh-session/control`; no fresh authentication,
no password/key files, strict host keys, BatchMode, no direct fallback. Optional verified
`--ram-evidence` uses the same schema as the R2 independent auditor. Slurm JobDefaults
DefMemPerGPU is supported, unknown/zero/default-less CPU budget fails closed.

The independent derived audit recomputes all actual fit arrays, target/drop accounting,
train-only scaler (unique frame rows including warmup, context accepted origins only),
lineage and all50 per-run quality/exposure. Actual N,D=1907,K=40 feed8Gram+2design+
2normalized+3targets+2coefficients+2GiB planning; require<=75% min(CPU allocation,available).
Insufficient/stale RAM retains a blocking report. No large Gram is allocated by this gate.
Inside the allocated job, actual Slurm CPU mem and /proc/meminfo MemAvailable are checked
again against the same plan before fitting. GPU VRAM is never substituted for CPU RAM.

Only sealed train/val/scaler/manifest/canonical-seal metadata/ridge plan and completion
proofs are copied remotely; no calibration NPZ, raw, canonical body or truth transfer.
Remote package/source hashes and input hashes are checked before existing launch.
The shared GPU lock/empty own queue/same-job environment checks are preserved. Slurm's
CUDA_VISIBLE_DEVICES is untouched. CPU ridge, then LSTM, run in the same allocated job.
All returned files plus job exit/release evidence are retrieved once to external models,
hash-verified before calibration. Network failure keeps unknown state, never cancels/retries.
Only this job's existing epoch log lines are forwarded over the same SSH; no LLM calls,
no extra job, no100% inference from an epoch count. Programmatic forwarding is bounded.

## Calibration and evaluator contract

The unchanged normal-only protocol selects one of7 finite quantiles for union FA<=1/h;
if none qualifies it retains the best finite candidate with `target_unmet`. Development
results never retune it. CPU observation inference produces **69 seals**:3models×23devruns.
Zero-window/exposure cases are retained unavailable, never silently removed or healthy-zero.

`calibration/attempt-01/{artifact.json,exit.json,release.json,evaluator-handoff.json}` keeps
the existing shape. The artifact's files include `sealed/<model>/calibration.json`,
per-run forecasts.jsonl/decisions.jsonl/summary.json/seal.json, and all-forecast-seals.json.
Coordinator must verify exit0/released=true and exact artifact hash before reading truth.
It then uses its R3 evaluator with frozen manifest and separate verified truth paths,
retains all23dev cases and all events, computes common-origin physical errors and prior
protocol metrics. Evaluation wall budget is7200 minus calibration's actual
exit.json.wall_including_queue_s. No positive remainder means stop. No final test, automatic
retraining or claim of SOH/RUL/health certification.

## Scope of current evidence

Readiness reports synthetic tests, source hashes and CLI checks only. Actual producer lock,
complete collection/archive freeze, fresh current RAM, real preprocessing/window counts,
GPU release and calibration/evaluation results remain separate required evidence. Synthetic
package builds use explicitly fixture-only revision metadata and are never release packages.
