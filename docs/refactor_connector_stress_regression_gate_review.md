# M2.3 Review: Connector Stress Tranche Regression Gate

Date: March 25, 2026

## Goal

Turn the existing stress tranche into a standing connector regression gate so
later proving-ground failures can be treated as substrate issues instead of
mock-market drift.

## What Changed

### 1. Stress-task live grading now exists alongside preflight grading

[/Users/steveyang/Projects/steveya/trellis/trellis/agent/evals.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/evals.py)
now adds:

- `grade_stress_task_result(...)`
- `summarize_stress_tranche(...)`

This means the stress manifest is no longer only a preflight readiness check.
It can now also grade live outcomes against:

- outcome class (`compare_ready` vs `honest_block`)
- forbidden failure patterns
- per-task failure buckets

### 2. There is now a dedicated batch runner for the connector stress tranche

[/Users/steveyang/Projects/steveya/trellis/scripts/run_stress_tranche.py](/Users/steveyang/Projects/steveya/trellis/scripts/run_stress_tranche.py)
now:

- loads `E21`-`E28`
- runs deterministic preflight first
- fails fast if preflight is not clean
- persists:
  - `task_results_stress_connector_<timestamp>.json`
  - `task_results_stress_connector_<timestamp>_summary.json`

The summary includes both the general task summary and the connector-specific
stress summary.

## Why This Matters

This tranche now answers a more useful question than “did some tasks run?”:

`is the connector-aware task path healthy enough that later proving-ground work can trust its market-data substrate?`

That is the right gate before the FX and local-vol proving grounds.

## Validation

Focused slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_evals.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_stress_task_preflight.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py \
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

Result:

- covered in the standing focused slice below

## Result

The stress tranche is now executable as a named regression gate rather than
just a static manifest plus a few unit tests.
