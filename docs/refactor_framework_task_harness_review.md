# Framework Task Harness Review

Date: 2026-03-26

This tranche implemented the first dedicated execution path for the tasks in
`FRAMEWORK_TASKS.yaml`.

## What changed

- Added `trellis.agent.framework_runtime` as the deterministic framework/meta
  runner.
- Added `scripts/run_framework_tasks.py` for batch execution.
- Added shared trigger evaluation for:
  - explicit prerequisite task lists
  - `every_10_tasks`
  - `every_10_entries`
- Added structured framework outcomes:
  - `extraction_candidate`
  - `consolidation_candidate`
  - `infrastructure_review`
  - `does_not_yet_apply`

## Shared audit model

Framework runs now persist through the same canonical task-run store as pricing
runs. The shared record contract carries:

- `task_kind`
- common `workflow`
- common `issue_refs`
- common `learning`
- framework-specific payload under `framework`

This keeps history/latest storage, later UI work, and issue/audit plumbing on
one path rather than creating a separate framework-only store.

## Validation

- deterministic framework runner tests
- shared task-run-store tests for framework records
- nearby pricing-task regression slice to verify the shared store did not break
  the pricing path

## Remaining work

- `F2`: surface framework runs in the UI/API
- `F3`: define the promotion loop from framework outputs into library and
  knowledge follow-up work
