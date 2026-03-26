# Task Manifest Split Review

## Goal

Keep the pricing-task runner semantically clean by moving non-priceable
framework, infrastructure, and experience tasks out of `TASKS.yaml`.

## What Changed

- `TASKS.yaml` now contains only priceable tasks.
- `FRAMEWORK_TASKS.yaml` now holds the non-priceable framework/meta inventory:
  - `T91`-`T93`
  - `E01`-`E20`
- `trellis.agent.task_runtime.load_tasks()` now loads only pricing tasks.
- `trellis.agent.task_runtime.load_framework_tasks()` now loads the framework/meta
  inventory.
- `scripts/remediate.py` now reuses `load_tasks()` instead of opening
  `TASKS.yaml` directly.
- The UI `/tasks` endpoint now explicitly exposes the pricing task inventory.

## Why

The pricing-task runner should only execute tasks that can coherently compile
into pricing builds. Framework/meta tasks are still valuable, but they belong in
a separate inventory and should eventually run through a dedicated
framework-evolution harness. That follow-up is now tracked explicitly in
`docs/autonomous_library_development_workstream.md` under the parallel
framework/meta task harness workstream (`F1`-`F3`).

## Validation

- task-runtime loader tests now assert that `T91` and `E01` are absent from
  `load_tasks()` and present in `load_framework_tasks()`
- task-runtime contract test still verifies that a framework-shaped task fails
  early with a structured `TaskContractError`

## Result

- `TASKS.yaml`: 131 priceable tasks
- `FRAMEWORK_TASKS.yaml`: 23 framework/meta tasks
- overlap: none
