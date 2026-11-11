# Preserved test contracts

`simulation/conftest.py` supplies explicit test-only contracts for sealed tests
whose production dependencies subsequently changed. No test is skipped, marked
xfail, or given a replacement verifier/assertion. Run pytest normally beneath
`simulation/`; bypassing this conftest reproduces the historical mismatches.

- The recorder fake now records the `sync_interval_s` argument used by the real
  runner. It still creates no actual recorder, processes, or flight.
- `test_joint_duration.py` was written for 40 ms acceptance. Its synthetic runs
  receive a temporary 40 ms policy. New tests separately exercise the current
  3 s policy, including rejection when complete reference coverage is missing.
- `native_consumption_v2` is a frozen v1 audit. Its seven original tests receive
  the exact historical generator and policy listed in its manifest, instead of
  importing the current v2 generator. Both old strict and audit verifiers run.

`frozen_joint_v1` contains byte-identical copies recovered from the preserved
`simulation/artifacts/paired-origin-20261107/recovery-01/offline-source/` snapshot.
SHA-256 is checked against `audit/native_consumption_v2/manifest.json` before
loading. These are historical test dependencies, not another production runner.
Current production code, policy, model, data and all existing seals are unchanged.
