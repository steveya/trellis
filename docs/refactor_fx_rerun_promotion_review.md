# FX Rerun and Promotion Discipline Review

Date: 2026-03-26

This tranche completed the rerun/reporting side of the FX proving ground and
promotion discipline work.

## What changed

- Added `scripts/run_fx_tranche.py` for the FX proving-ground task set:
  - `E25`
  - `T105`
  - `T108`
  - optional `T94`
- Added promotion-discipline summaries to `trellis.agent.evals` and to the
  batch runner output.
- Extended the persisted task-run contract so successful runs now summarize the
  reusable learning artifacts they left behind:
  - captured lessons
  - lesson attribution
  - cookbook candidates
  - knowledge traces

## Why this matters

The promotion loop was previously visible only indirectly through individual
reflection payloads. It is now measurable at the tranche level, which lets us
answer:

- which successful reruns actually left reusable knowledge behind
- which successful reruns still need stronger promotion behavior
- how shared-memory and promotion signals changed between baseline and candidate
  batches

## What remains open

The remaining `M3.4` work is explicitly Knowledge-Agent-owned:

- canonical FX cookbook/method-requirement/contract registration under
  `trellis/agent/knowledge/`
- live FX reruns once provider budget/noise is acceptable

This tranche deliberately stopped short of editing canonical knowledge files in
order to respect the repo ownership boundary in `AGENTS.md`.
