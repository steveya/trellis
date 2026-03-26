# Stress-Task Tranche Review

## Goal

Add a dedicated stress-task corpus that exercises the current mock connector and
comparison-task machinery before introducing task-level market selection.

## Review

Before this tranche:

- `TASKS.yaml` had no dedicated stress-task block for mock-connector coverage
- the task runner already supported comparison targets and runtime
  cross-validation
- the mock connector exposed more capabilities than the task inventory was
  explicitly probing

## Changes

Implemented in:

- [TASKS.yaml](/Users/steveyang/Projects/steveya/trellis/TASKS.yaml)
- [stress_tasks.yaml](/Users/steveyang/Projects/steveya/trellis/tests/evals/stress_tasks.yaml)
- [evals.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/evals.py)
- [test_stress_task_preflight.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_stress_task_preflight.py)

This tranche adds:

- `E21` through `E28` as market-connector stress tasks
- a manifest describing expected outcome class, required mock capabilities, and
  comparison targets
- deterministic preflight grading for:
  - mock market capability alignment
  - comparison target inventory
  - same-family target separation

## Why This Comes Before Task-Level Market Specs

The first question is whether the current default mock-backed task path is
already good enough to support the compare-ready tasks and to avoid
misdiagnosing honest-block tasks as market-data failures.

Only after this corpus is in place does it make sense to add explicit `market:`
selection and `market_assertions:` to `TASKS.yaml`.

## Validation

Run the focused deterministic slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_evals.py \
  tests/test_agent/test_stress_task_preflight.py
```
