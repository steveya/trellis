# Task-Level Market Specs Review

## Goal

Move the stress-task tranche from "run on a broad default mock market state" to
"select named snapshot components and validate that selection before build."

## Changes

Implemented in:

- [TASKS.yaml](/Users/steveyang/Projects/steveya/trellis/TASKS.yaml)
- [task_runtime.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/task_runtime.py)
- [evals.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/evals.py)
- [test_task_runtime.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py)
- [test_stress_task_preflight.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_stress_task_preflight.py)

This tranche adds:

- `market:` blocks on `E21` through `E28`
- `market_assertions:` for required capabilities and expected named component
  selection
- task-local market resolution through `resolve_market_snapshot(source="mock")`
- recorded market provenance in task results
- deterministic failure on task/assertion mismatch before build confusion

## Why This Matters

The mock connector is only being tested meaningfully if tasks specify which
curve, surface, FX quote, or parameter pack they intended to use.

Without that, tasks only prove that "some default market state existed."

## Validation

Focused:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_task_runtime.py -k 'market_state_for_task or uses_task_specific_market_state' \
  tests/test_agent/test_stress_task_preflight.py \
  tests/test_agent/test_evals.py
```

Broader:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_task_runtime.py -k 'not generic_cached_transform_task' \
  tests/test_agent/test_evals.py \
  tests/test_agent/test_stress_task_preflight.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_data/test_resolver.py \
  tests/test_data/test_mock.py
```
